#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-1.3 — kanun_madde EKİM BETİĞİ (17 kanun maddesi → Qdrant)
=============================================================
data/legal_korpus/*.txt (17 kanun, MADDE_HEAD satırlı) dosyalarındaki her
maddeyi QDRANT_URL'deki (varsayılan localhost:6333) 'kanun_madde' koleksiyonuna
yazar (gövde = madde başlığı + izleyen satırlar; prepare_legal_dataset.py
load_corpus ayrıştırıcısıyla AYNI regex/kurallar).

- Embedding yolu: gateway'in AYNISI — KristalTokenizer (anka vocab 33.114)
  + generate_kristal_vector(768) + generate_sparse_vector.
- text (gömme/görüntü): '{kanun_adi} (Sayı: {kanun_no}) {etiket} {metin≤2000}';
  payload.metin TAM madde metnini taşır (gövde-faithful).
- Mülga maddeler ELENİR (prepare_legal_dataset.is_mulga kalıbı — grounding'e
  yürürlükten kalkmış hüküm enjekte etme).
- Nokta KİMLİĞİ (P2.1): tam-kayıt (kanun_no|etiket|metin) sha256 → 2^63-mod →
  idempotent upsert; kesilen koşum kaldığı yerden devam eder.
- YAZMA YETKİSİ YALNIZ 'kanun_madde' (SERT KURAL); anka/muhakeme/simulasyon
  yalnız OKUNUR, değişmediği kanıtlanır.

Kullanım:
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_kanun_madde.py --dry-run
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_kanun_madde.py
"""

import argparse
import hashlib
import json
import re
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_memory import VectorMemory, resolve_qdrant_url

VARSAYILAN_URL = "http://localhost:6333"
KOLEKSIYON = "kanun_madde"  # SERT KURAL: yazma yetkisi yalnız bu koleksiyon
DOKUNULMAZ = ["anka_bellek", "muhakeme_bellek", "simulasyon_bellek"]
VEKTOR_BOYUTU = 768
KORPUS = "data/legal_korpus"
MANIFEST = "data/legal_korpus/korpus_manifest.json"
TEXT_CAP = 2000     # koşullanma metni tavanı (karakter)
BATCH = 128
RETRIEVE_CHUNK = 512

# prepare_legal_dataset.py ile BİREBİR ayrıştırıcı kalıpları (FAZ-1 kanonik)
MADDE_HEAD = re.compile(r"^(Geçici\s+|Ek\s+)?Madde\s*(\d+)\s*[-–—]\s*(.*)$")
MULGA_MARK = re.compile(r"^\s*\(?\s*Mülga\b", re.I)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def kanunlari_ayrıştır(dry_limit=None):
    """Manifest'teki 17 kanunu ayrıştırır → madde kayıtları.

    Dönüş: kayitlar[dict] — kanun_adi, kanun_no, etiket, madde_no, gecici,
    metin (TAM), mulga (bool), kaynak_url
    """
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    kanunlar = {k["no"]: k for k in manifest}
    kayitlar = []
    for dosya in sorted(Path(KORPUS).glob("*.txt")):
        no = int(dosya.name[:4])
        bilgi = kanunlar.get(no)
        if bilgi is None:
            raise ValueError(f"[EKIM_HATA] manifest'te kanun yok: {dosya.name}")
        kanun_adi = bilgi["ad"]
        kaynak = bilgi["url"]
        cur = None
        for ln in dosya.read_text(encoding="utf-8").split("\n"):
            if ln.startswith("#"):
                continue
            m = MADDE_HEAD.match(ln)
            if m:
                prefix = (m.group(1) or "").strip().title()  # 'Geçici' / 'Ek'
                etiket = f"{prefix + ' ' if prefix else ''}Madde {int(m.group(2))} -"
                cur = {
                    "kanun_adi": kanun_adi, "kanun_no": no,
                    "etiket": etiket, "madde_no": int(m.group(2)),
                    "gecici": bool(m.group(1)),
                    "lines": [],
                    "kaynak_url": kaynak,
                }
                head_text = m.group(3).strip()
                if head_text:
                    cur["lines"].append(head_text)
                kayitlar.append(cur)
            elif cur is not None:
                cur["lines"].append(ln)
    # metin kur + mülga ele
    out = []
    mulga_sayisi = 0
    for k in kayitlar:
        metin = " ".join(l.strip() for l in k.pop("lines") if l.strip())
        k["metin"] = metin
        head = k["etiket"]
        ilk = metin.split("\n")[0] if metin else ""
        is_mulga = bool(MULGA_MARK.match(metin)) or (
            len(metin) < 40 and "mülga" in metin.lower())
        if not metin or len(metin) < 10:
            continue  # boş madde gövdesi (ör. yalnız dipnot satırı)
        if is_mulga:
            mulga_sayisi += 1
            continue
        out.append(k)
    if dry_limit:
        out = out[:dry_limit]
    return out, mulga_sayisi


def nokta_kimligi(k: dict) -> int:
    tam = f"{k['kanun_no']}|{k['etiket']}|{k['metin']}"
    return int(hashlib.sha256(tam.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 63)


def doc_text(k: dict) -> str:
    govde = k["metin"][:TEXT_CAP]
    if len(k["metin"]) > TEXT_CAP:
        govde = govde.rstrip() + "…"
    return f"{k['kanun_adi']} (Sayı: {k['kanun_no']}) {k['etiket']} {govde}"


def _isici_init(vocab_yolu, lexicon_yolu):
    global _TOK
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary
    from src.rag.embedding import generate_kristal_vector, generate_sparse_vector
    vocab = Vocabulary()
    vocab.load(vocab_yolu)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_yolu)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    _TOK = KristalTokenizer(compiler, vocab)
    _TOK.gkv = generate_kristal_vector
    _TOK.gsv = generate_sparse_vector


def _isici_parca(kayitlar):
    global _TOK
    ids, texts, denses, sparses, metas = [], [], [], [], []
    for k in kayitlar:
        text = doc_text(k)
        tok_ids = _TOK.encode(text)
        tags = _TOK.decode(tok_ids)
        meta = {
            "kanun_adi": k["kanun_adi"],
            "kanun_no": k["kanun_no"],
            "madde_no": k["madde_no"],
            "madde_etiketi": k["etiket"],
            "gecici": k["gecici"],
            "metin": k["metin"],
            "kaynak": k["kaynak_url"],
            "domain": "kanun_madde",
            "crystal_tags": tags,
            "token_ids": tok_ids,
        }
        ids.append(nokta_kimligi(k))
        texts.append(text)
        denses.append(_TOK.gkv(tok_ids, tags, 768))
        sparses.append(_TOK.gsv(tok_ids, tags))
        metas.append(meta)
    return ids, texts, denses, sparses, metas


def main() -> int:
    ayarla = argparse.ArgumentParser(
        description="17 kanun maddesi → kanun_madde ekim betiği (idempotent upsert)")
    ayarla.add_argument("--url", type=str, default=None)
    ayarla.add_argument("--collection", type=str, default=KOLEKSIYON)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--limit", type=int, default=None, help="İlk N madde (pilot)")
    ayarla.add_argument("--batch", type=int, default=BATCH)
    ayarla.add_argument("--timeout", type=float, default=30.0)
    ayarla.add_argument("--workers", type=int, default=4)
    ayarla.add_argument("--dry-run", action="store_true")
    args = ayarla.parse_args()

    hedef = resolve_qdrant_url(args.url)
    if hedef is None:
        hedef = ("localhost", 6333)
    print(f"[EKIM] hedef Qdrant: {hedef[0]}:{hedef[1]}")
    print(f"[EKIM] koleksiyon: {args.collection} (SERT KURAL: yazma yalnız bu; "
          f"{', '.join(DOKUNULMAZ)} dokunulmaz)")

    t0 = time.time()
    kayitlar, mulga = kanunlari_ayrıştır()
    print(f"[EKIM] ayrıştırma {time.time()-t0:.1f}s — madde: {len(kayitlar)} "
          f"(mülga elendi: {mulga}; boş gövde ayrıca elendi)")
    kanun_dagilim = {}
    for k in kayitlar:
        kanun_dagilim[k["kanun_no"]] = kanun_dagilim.get(k["kanun_no"], 0) + 1
    print(f"[EKIM] kanun dağılımı: {dict(sorted(kanun_dagilim.items()))}")
    if args.limit:
        kayitlar = kayitlar[: args.limit]
        print(f"[EKIM] pilot limiti: {len(kayitlar)}")
    if not kayitlar:
        print("[EKIM][HATA] ayrıştırılamadı — DUR")
        return 1

    if args.dry_run:
        print("[EKIM] --dry-run: bağlantı KURULMADI, yazma YOK")
        for i in (0, len(kayitlar) // 2, len(kayitlar) - 1):
            k = kayitlar[i]
            print(f"  örnek[{i}] id={nokta_kimligi(k)} {k['kanun_adi']} {k['etiket']} "
                  f"metin={k['metin'][:70]!r}")
        return 0

    bellek = None
    for deneme in range(1, 4):
        try:
            bellek = VectorMemory(
                collection_name=args.collection,
                vector_size=VEKTOR_BOYUTU,
                host=hedef[0], port=hedef[1], timeout=args.timeout,
            )
            break
        except Exception as e:  # noqa: BLE001
            print(f"[EKIM][DENEME {deneme}/3] bağlantı hatası: {type(e).__name__}: {e}")
            if deneme < 3:
                time.sleep(5)
    if bellek is None:
        print(f"[EKIM][HATA] 3 denemede bağlanılamadı — DUR")
        return 1
    print(f"[EKIM] bağlandı: {bellek.storage_type}")

    def sayi_oku(k):
        try:
            if not bellek.client.collection_exists(k):
                return None
            return bellek.client.count(collection_name=k).count
        except Exception as e:  # noqa: BLE001
            return f"HATA:{type(e).__name__}"

    onceki = bellek.get_document_count()
    dokunulmaz_once = {ad: sayi_oku(ad) for ad in DOKUNULMAZ}
    print(f"[EKIM] ekim-ÖNCESİ {args.collection}: {onceki}; dokunulmaz: {dokunulmaz_once}")

    tum_idler = [nokta_kimligi(k) for k in kayitlar]
    if len(set(tum_idler)) != len(tum_idler):
        print(f"[EKIM][UYARI] yinelenik nokta kimliği: {len(tum_idler) - len(set(tum_idler))} "
              "— upsert bunları TEK noktada birleştirir")
    mevcut_idler = set()
    t0 = time.time()
    for bas in range(0, len(tum_idler), RETRIEVE_CHUNK):
        parca = tum_idler[bas: bas + RETRIEVE_CHUNK]
        geri = bellek.client.retrieve(
            collection_name=args.collection, ids=parca, with_payload=False, with_vectors=False)
        mevcut_idler |= {p.id for p in geri}
    eksikler = [k for k, kid in zip(kayitlar, tum_idler) if kid not in mevcut_idler]
    print(f"[EKIM] mevcut: {len(mevcut_idler)}, EKSİK: {len(eksikler)} ({time.time()-t0:.1f}s)")

    from collections import deque
    from qdrant_client.http import models

    def _upsert(ids, texts, denses, sparses, metas):
        bellek.add_documents_batch(
            texts=texts, dense_vectors=denses,
            sparse_vectors=[models.SparseVector(indices=s.indices, values=s.values) for s in sparses],
            metadatas=metas, upsert=True, point_ids=ids,
        )

    bas_zaman = time.time()
    yazilan = 0
    isci = None
    if args.workers > 1 and len(eksikler) > args.batch:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        isci = ctx.Pool(processes=args.workers, initializer=_isici_init,
                        initargs=(args.vocab, args.lexicon))
        print(f"[EKIM] gömme işçileri: {args.workers} (uçta ≤4 parça)")

    parcalar = [eksikler[i: i + args.batch] for i in range(0, len(eksikler), args.batch)]
    toplam_parca = len(parcalar)
    pencere = deque()

    def _rapor(pidx):
        gecen = time.time() - bas_zaman
        hiz = yazilan / gecen if gecen > 0 else 0.0
        kalan = len(eksikler) - yazilan
        eta = kalan / hiz if hiz > 0 else float("inf")
        print(f"  [EKIM] {yazilan}/{len(eksikler)} parça {pidx}/{toplam_parca} "
              f"({hiz:.0f} madde/s, ETA {eta/60:.1f} dk, RSS {rss_mb():.0f} MB)", flush=True)

    if isci is None:
        _isici_init(args.vocab, args.lexicon)
        for pidx, parca in enumerate(parcalar, start=1):
            _upsert(*_isici_parca(parca))
            yazilan += len(parca)
            if pidx == 1 or pidx == toplam_parca:
                _rapor(pidx)
    else:
        pidx = 0
        for parca in parcalar:
            while len(pencere) >= 4:
                ids, texts, denses, sparses, metas = pencere.popleft().get()
                _upsert(ids, texts, denses, sparses, metas)
                yazilan += len(texts)
                pidx += 1
                if pidx == 1 or pidx == toplam_parca or pidx % 10 == 0:
                    _rapor(pidx)
            pencere.append(isci.apply_async(_isici_parca, (parca,)))
        while pencere:
            ids, texts, denses, sparses, metas = pencere.popleft().get()
            _upsert(ids, texts, denses, sparses, metas)
            yazilan += len(texts)
            pidx += 1
            if pidx == toplam_parca:
                _rapor(pidx)

    if isci is not None:
        isci.close()
        isci.join()
    sure = time.time() - bas_zaman

    sonraki = bellek.get_document_count()
    print(f"[EKIM] ekim-SONRASI nokta sayısı: {sonraki} (öncesi: {onceki}, yazma: {sure:.1f}s)")

    ornek_yerler = [0, len(kayitlar) // 2, len(kayitlar) - 1]
    ornekler = [kayitlar[i] for i in ornek_yerler]
    ornek_idler = [nokta_kimligi(k) for k in ornekler]
    geri = bellek.client.retrieve(
        collection_name=args.collection, ids=ornek_idler, with_payload=True, with_vectors=False)
    geri_map = {p.id: p.payload for p in geri}
    eslesen = 0
    for idx, k in zip(ornek_idler, ornekler):
        p = geri_map.get(idx)
        if p is None:
            print(f"  [KANIT] id={idx} GERİ OKUNAMADI — DOĞRULAMA BAŞARISIZ")
            continue
        ayni = (p.get("kanun_adi") == k["kanun_adi"] and p.get("madde_no") == k["madde_no"]
                and p.get("metin") == k["metin"] and p.get("text") == doc_text(k))
        eslesen += 1 if ayni else 0
        print(f"  [KANIT] id={idx} birebir={'EVET' if ayni else 'HAYIR'} "
              f"{k['kanun_adi']} {k['etiket']} metin={str(p.get('metin'))[:60]!r}")

    dokunulmaz_sonra = {ad: sayi_oku(ad) for ad in DOKUNULMAZ}
    print(f"[EKIM] dokunulmaz ÖNCE/SONRA: {dokunulmaz_once} → {dokunulmaz_sonra} "
          f"({'DEĞİŞMEDİ' if dokunulmaz_once == dokunulmaz_sonra else 'DEĞİŞTİ!!'})")
    bellek.close()

    beklenen = len(mevcut_idler | set(tum_idler))
    hatalar = []
    if sonraki < beklenen:
        hatalar.append(f"read-back {sonraki} < beklenen {beklenen}")
    if eslesen != len(ornek_idler):
        hatalar.append(f"örnek read-back {eslesen}/{len(ornek_idler)}")
    if dokunulmaz_once != dokunulmaz_sonra:
        hatalar.append("DOKUNULMAZ koleksiyon sayısı DEĞİŞTİ — KURAL İHLALİ")
    if hatalar:
        print(f"[EKIM][HATA] {'; '.join(hatalar)} — DOĞRULAMA BAŞARISIZ")
        return 1
    print(f"[EKIM] TAMAM: {yazilan} yeni madde upsert ({sure:.1f}s), koleksiyonda {sonraki} nokta, "
          f"3/3 örnek birebir, dokunulmazlar değişmedi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
