#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-1.1 — gts_bellek EKİM BETİĞİ (GTS 128.327 satır → Qdrant)
============================================================
data/kulliyat_v2/gts_definitions.txt (MADDE<TAB>anlam) satırlarını
QDRANT_URL'deki (varsayılan localhost:6333) 'gts_bellek' koleksiyonuna yazar.

- Embedding yolu: VectorMemory/gateway'in AYNISI — KristalTokenizer
  (anka vocab 33.114) + generate_kristal_vector(768) + generate_sparse_vector.
- DEDUPE (T3.3 dersi): aynı (madde, anlam) çifti TEK nokta — 128.327 satırda
  11.912 yinelenik anlam satırı biliniyor; birebir madde+anlam tek noktaya iner.
- Nokta KİMLİĞİ (P2.1 dersi): TAM-SATIR (kanonik 'madde<TAB>anlam') sha256
  → ilk 16 hex → 2^63-mod. Birebir aynı satır = aynı kimlik → idempotent upsert.
- İDEMPOTENT + SÜRDÜRÜLEBİLİR: mevcut kimlikler chunk'lı retrieve ile saptanır;
  yalnız EKSİK noktalar üretilip yazılır. Kesilen koşum yeniden koşumda
  kaldığı yerden devam eder (batch planı), tekrar koşum sayıyı DEĞİŞTİRMEZ.
- AKIŞ-BATCH: bellekte tek bir parça tutulur (RSS-güvenli; 128K×768 vektör
  önceden üretilemez — pilot RSS ölçümü raporlanır).
- YAZMA YETKİSİ YALNIZ 'gts_bellek' (SERT KURAL); anka_bellek / muhakeme_bellek /
  simulasyon_bellek yalnız OKUNUR ve değişmediği KANITLANIR.

Kullanım:
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_gts_bellek.py --dry-run
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_gts_bellek.py --limit 1000   # pilot
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_gts_bellek.py               # tam ekim
"""

import argparse
import hashlib
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_memory import VectorMemory, resolve_qdrant_url

VARSAYILAN_URL = "http://localhost:6333"
KOLEKSIYON = "gts_bellek"  # SERT KURAL: yazma yetkisi yalnız bu koleksiyon
DOKUNULMAZ = ["anka_bellek", "muhakeme_bellek", "simulasyon_bellek"]  # salt-okuma kanıtı
VEKTOR_BOYUTU = 768
KAYNAK = "data/kulliyat_v2/gts_definitions.txt"
BATCH = 512          # upsert parça boyu
RETRIEVE_CHUNK = 512  # mevcut-kimlik sorgulama parçası


def rss_mb() -> float:
    """macOS ru_maxrss bayt cinsindendir → MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def satirlari_oku(yol: str, limit=None):
    """GTS satırlarını okur; (madde, anlam) birebir yinelenikleri düşürür.

    Dönüş: (kayitlar[(madde, anlam, ham_satir_no)], yinelenik_sayisi)
    """
    kayitlar = []
    gorulen = set()
    yinelenik = 0
    with open(yol, "r", encoding="utf-8") as f:
        for i, satir in enumerate(f, start=1):
            if limit is not None and len(kayitlar) >= limit:
                break
            satir = satir.rstrip("\n").rstrip("\r")
            if not satir.strip():
                continue
            if "\t" not in satir:
                raise ValueError(f"[EKIM_HATA] satır {i}: sekme (MADDE<TAB>anlam) yok: {satir[:60]!r}")
            madde, _, anlam = satir.partition("\t")
            madde = madde.strip()
            anlam = anlam.strip()
            if not madde or not anlam:
                raise ValueError(f"[EKIM_HATA] satır {i}: boş madde veya anlam")
            anahtar = (madde, anlam)
            if anahtar in gorulen:
                yinelenik += 1
                continue
            gorulen.add(anahtar)
            kayitlar.append((madde, anlam, i))
    return kayitlar, yinelenik


def nokta_kimligi(madde: str, anlam: str) -> int:
    """P2.1: TAM-SATIR (kanonik 'madde<TAB>anlam') sha256 → 2^63-mod."""
    tam = f"{madde}\t{anlam}".encode("utf-8")
    return int(hashlib.sha256(tam).hexdigest()[:16], 16) % (2 ** 63)


def _isici_init(vocab_yolu, lexicon_yolu):
    """Multiprocessing işçisi: tokenizer'ı kez-başına kur (global)."""
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
    """Bir parça (madde, anlam) kaydını gömme+yük paketine çevirir."""
    global _TOK
    texts, denses, sparses, metas, ids = [], [], [], [], []
    for madde, anlam, _satir_no in kayitlar:
        text = f"{madde}: {anlam}"
        tok_ids = _TOK.encode(text)
        tags = _TOK.decode(tok_ids)
        dense = _TOK.gkv(tok_ids, tags, 768)
        sparse = _TOK.gsv(tok_ids, tags)
        meta = {
            "madde": madde,
            "anlam": anlam,
            "kaynak": KAYNAK,
            "domain": "gts_sozluk",
            "crystal_tags": tags,
            "token_ids": tok_ids,
        }
        texts.append(text)
        denses.append(dense)
        sparses.append(sparse)
        metas.append(meta)
        ids.append(nokta_kimligi(madde, anlam))
    return ids, texts, denses, sparses, metas


def main() -> int:
    ayarla = argparse.ArgumentParser(
        description="gts_definitions.txt → gts_bellek ekim betiği (idempotent, akış-batchli, dedupe'li)")
    ayarla.add_argument("--url", type=str, default=None,
                        help="Qdrant URL'i; yoksa QDRANT_URL env, o da yoksa localhost:6333")
    ayarla.add_argument("--gts", type=str, default=KAYNAK)
    ayarla.add_argument("--collection", type=str, default=KOLEKSIYON)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--limit", type=int, default=None,
                        help="İlk N BENZERSİZ kayıt (pilot/ETA ölçümü)")
    ayarla.add_argument("--batch", type=int, default=BATCH)
    ayarla.add_argument("--timeout", type=float, default=30.0)
    ayarla.add_argument("--workers", type=int, default=4,
                        help="Gömme işçi süreç sayısı (1 = tek-süreç)")
    ayarla.add_argument("--dry-run", action="store_true",
                        help="Bağlantı KURMAZ: satır/dedupe/benzersiz-nokta sayısını basar")
    args = ayarla.parse_args()

    hedef = resolve_qdrant_url(args.url)
    if hedef is None:
        hedef = ("localhost", 6333)
    hedef_url = args.url or __import__("os").environ.get("QDRANT_URL", VARSAYILAN_URL)
    print(f"[EKIM] hedef Qdrant: {hedef[0]}:{hedef[1]} (url={hedef_url!r})")
    print(f"[EKIM] koleksiyon: {args.collection} (SERT KURAL: yazma yalnız bu; "
          f"{', '.join(DOKUNULMAZ)} dokunulmaz)")

    t0 = time.time()
    if args.limit:
        kayitlar, yinelenik = satirlari_oku(args.gts, limit=None)  # dedupe bütünüyle
        kayitlar = kayitlar[: args.limit]
        yinelenik_gorunen = yinelenik
    else:
        kayitlar, yinelenik_gorunen = satirlari_oku(args.gts, limit=None)
    okuma_sure = time.time() - t0
    benzersiz = len(kayitlar)
    print(f"[EKIM] kaynak: {args.gts} — benzersiz (madde,anlam): {benzersiz}"
          f"{f' (pilot limiti: {args.limit})' if args.limit else ''}, "
          f"yinelenik satır: {yinelenik_gorunen}, okuma {okuma_sure:.1f}s")
    print(f"[EKIM] örnekler: {kayitlar[0][0]!r} | {kayitlar[benzersiz//2][0]!r} | {kayitlar[-1][0]!r}")
    print(f"[EKIM] RSS (okuma sonrası): {rss_mb():.1f} MB")

    if args.dry_run:
        print("[EKIM] --dry-run: bağlantı KURULMADI, yazma YOK")
        for i in (0, benzersiz // 2, benzersiz - 1):
            m, a, n = kayitlar[i]
            print(f"  örnek[{i}] satır={n} id={nokta_kimligi(m, a)} {m!r}: {a[:60]!r}")
        return 0

    # Bağlantı: fail-closed + 3 deneme (LAN aralıklı erişilebilirlik kalıbı)
    bellek = None
    for deneme in range(1, 4):
        try:
            bellek = VectorMemory(
                collection_name=args.collection,
                vector_size=VEKTOR_BOYUTU,
                host=hedef[0],
                port=hedef[1],
                timeout=args.timeout,
            )
            break
        except Exception as e:  # noqa: BLE001
            print(f"[EKIM][DENEME {deneme}/3] bağlantı hatası: {type(e).__name__}: {e}")
            if deneme < 3:
                time.sleep(5)
    if bellek is None:
        print(f"[EKIM][HATA] 3 denemede bağlanılamadı: {hedef[0]}:{hedef[1]} — DUR")
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
    print(f"[EKIM] ekim-ÖNCESİ {args.collection}: {onceki} nokta; dokunulmaz: {dokunulmaz_once}")

    # Mevcut kimlikler (idempotens + sürdürülebilirlik): chunk'lı retrieve
    tum_idler = [nokta_kimligi(m, a) for m, a, _ in kayitlar]
    mevcut_idler = set()
    t0 = time.time()
    for bas in range(0, len(tum_idler), RETRIEVE_CHUNK):
        parca = tum_idler[bas: bas + RETRIEVE_CHUNK]
        try:
            geri = bellek.client.retrieve(
                collection_name=args.collection, ids=parca, with_payload=False, with_vectors=False)
            mevcut_idler |= {p.id for p in geri}
        except Exception as e:  # noqa: BLE001
            print(f"[EKIM][HATA] mevcut-kimlik sorgusu ({bas}): {type(e).__name__}: {e}")
            bellek.close()
            return 1
    eksikler = [k for k, kid in zip(kayitlar, tum_idler) if kid not in mevcut_idler]
    print(f"[EKIM] mevcut: {len(mevcut_idler)}, EKSİK: {len(eksikler)} "
          f"(kontrol {time.time()-t0:.1f}s) — yalnız eksikler yazılır")

    # Gömme+upsert: akış-batchli (RSS-güvenli) + çok-işçili gömme.
    # Paralel gömme: en fazla 4 parça UÇTA (bellek tavanı bağlı); upsert ana
    # süreçte sıralı. Pool.apply BLOKLAYICIDIR (serileştirir) — async pencere şart.
    from collections import deque
    from qdrant_client.http import models

    def _upsert(ids, texts, denses, sparses, metas):
        bellek.add_documents_batch(
            texts=texts,
            dense_vectors=denses,
            sparse_vectors=[models.SparseVector(indices=s.indices, values=s.values) for s in sparses],
            metadatas=metas,
            upsert=True,
            point_ids=ids,
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
              f"({hiz:.0f} nokta/s, ETA {eta/60:.1f} dk, RSS {rss_mb():.0f} MB)", flush=True)

    if isci is None:
        _isici_init(args.vocab, args.lexicon)
        for pidx, parca in enumerate(parcalar, start=1):
            ids, texts, denses, sparses, metas = _isici_parca(parca)
            _upsert(ids, texts, denses, sparses, metas)
            yazilan += len(parca)
            if pidx == 1 or pidx % 20 == 0 or pidx == toplam_parca:
                _rapor(pidx)
    else:
        pidx = 0
        for parca in parcalar:
            while len(pencere) >= 4:
                ids, texts, denses, sparses, metas = pencere.popleft().get()
                _upsert(ids, texts, denses, sparses, metas)
                yazilan += len(texts)
                pidx += 1
                if pidx == 1 or pidx % 20 == 0 or pidx == toplam_parca:
                    _rapor(pidx)
            pencere.append(isci.apply_async(_isici_parca, (parca,)))
        while pencere:
            ids, texts, denses, sparses, metas = pencere.popleft().get()
            _upsert(ids, texts, denses, sparses, metas)
            yazilan += len(texts)
            pidx += 1
            if pidx == 1 or pidx % 20 == 0 or pidx == toplam_parca:
                _rapor(pidx)

    if isci is not None:
        isci.close()
        isci.join()
    sure = time.time() - bas_zaman

    # Read-back doğrulama
    sonraki = bellek.get_document_count()
    print(f"[EKIM] ekim-SONRASI nokta sayısı: {sonraki} (öncesi: {onceki}, yazma: {sure:.1f}s)")

    ornek_yerler = [0, benzersiz // 2, benzersiz - 1]
    ornek_kayitlar = [kayitlar[i] for i in ornek_yerler]
    ornek_idler = [nokta_kimligi(m, a) for m, a, _ in ornek_kayitlar]
    geri = bellek.client.retrieve(
        collection_name=args.collection, ids=ornek_idler, with_payload=True, with_vectors=False)
    geri_map = {p.id: p.payload for p in geri}
    eslesen = 0
    for idx, (m, a, n) in zip(ornek_idler, ornek_kayitlar):
        p = geri_map.get(idx)
        if p is None:
            print(f"  [KANIT] id={idx} GERİ OKUNAMADI — DOĞRULAMA BAŞARISIZ")
            continue
        ayni = (p.get("madde") == m and p.get("anlam") == a
                and p.get("text") == f"{m}: {a}" and p.get("kaynak") == KAYNAK)
        eslesen += 1 if ayni else 0
        print(f"  [KANIT] id={idx} satır={n} birebir={'EVET' if ayni else 'HAYIR'} "
              f"madde={p.get('madde')!r} anlam={str(p.get('anlam'))[:60]!r}")

    dokunulmaz_sonra = {ad: sayi_oku(ad) for ad in DOKUNULMAZ}
    print(f"[EKIM] dokunulmaz ÖNCE/SONRA: {dokunulmaz_once} → {dokunulmaz_sonra} "
          f"({'DEĞİŞMEDİ' if dokunulmaz_once == dokunulmaz_sonra else 'DEĞİŞTİ!!'})")
    bellek.close()

    beklenen = len(mevcut_idler) + len(eksikler)
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
    print(f"[EKIM] TAMAM: {yazilan} yeni nokta upsert ({sure:.1f}s), koleksiyonda {sonraki} nokta "
          f"(≥{beklenen}), 3/3 örnek birebir, dokunulmazlar değişmedi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
