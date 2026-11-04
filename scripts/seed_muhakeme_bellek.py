#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.1 — muhakeme_bellek EKİM BETİĞİ (CoT Vault → Qdrant)
=======================================================
data/pedagogy/cot_vault.jsonl'deki CoT kayıtlarını QDRANT_URL'deki (varsayılan
localhost:6333) 'muhakeme_bellek' koleksiyonuna yazar.

- Embedding yolu: VectorMemory/anka_bellek'in AYNISI — KristalTokenizer
  (anka vocab 33.114) + generate_kristal_vector(768) + generate_sparse_vector;
  koleksiyon şeması VectorMemory._create_hybrid_collection(768)'den türer
  (dense 768 Cosine + sparse IDF).
- doc_text/payload formatı gateway inject_reasoning_trace (T-0148 5D) ile
  BİREBİR aynı biçimdedir; nokta KİMLİĞİ tam-kayıt (kanonik JSON sha256)
  türevlidir — vault'un 91 ayrı süpervizyon olayı 91 ayrı nokta olur.
- IDEMPOTENT: upsert=True + kayıt-kimliği → betik tekrar koşunca ÇOĞALTMAZ,
  aynı kayıtları GÜNCELLER (birebir aynı satır eklenirse de tek nokta kalır).
- Yazma yetkisi YALNIZ 'muhakeme_bellek' (anka_bellek DOKUNULMAZ).

Kullanım:
    venv/bin/python scripts/seed_muhakeme_bellek.py                       # env/default
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_muhakeme_bellek.py
    venv/bin/python scripts/seed_muhakeme_bellek.py --url http://192.168.1.9:6333
    venv/bin/python scripts/seed_muhakeme_bellek.py --dry-run             # bağlantısız sayım
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_memory import VectorMemory, resolve_qdrant_url
from src.rag.embedding import generate_kristal_vector, generate_sparse_vector
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

VARSAYILAN_URL = "http://localhost:6333"
KOLEKSIYON = "muhakeme_bellek"
VEKTOR_BOYUTU = 768  # anka_bellek/gateway ile aynı (dense)


def doc_text_olustur(rec: dict) -> str:
    """inject_reasoning_trace (agent_gateway.py) doc_text biçimi BİREBİR."""
    return (
        f"Soru: {rec['query']}\n\n[Akıl Yürütme]:\n{rec['thought_trace'].strip()}\n\n"
        f"[Çözüm]: {rec['final_answer'].strip()}"
    )


def nokta_kimligi(rec: dict) -> int:
    """P2.1 kayıt-kimliği: TAM KAYITTAN (tüm alanlar, kanonik JSON, sıralı
    anahtarlar) sha256 → 2^63-mod. Vault'un 91 kaydının HER BİRİ ayrı
    süpervizyon olayıdır (91 ayrı timestamp; içerik bazında yalnız 22 benzersiz
    — tek sorgu 70x yinelenmiştir). Tam-kayıt kimliği → 91 ayrı nokta; birebir
    aynı satır yeniden eklenirse AYNI noktaya güncelleme (idempotent upsert).
    """
    kanonik = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(kanonik.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 63)


def kayitlari_yukle(vault_path: str) -> list:
    kayitlar = []
    with open(vault_path, "r", encoding="utf-8") as f:
        for i, satir in enumerate(f):
            satir = satir.strip()
            if not satir:
                continue
            rec = json.loads(satir)
            for zorunlu in ("query", "thought_trace", "final_answer"):
                if zorunlu not in rec:
                    raise ValueError(f"[EKIM_HATA] satır {i + 1}: '{zorunlu}' alanı yok")
            kayitlar.append(rec)
    if not kayitlar:
        raise ValueError(f"[EKIM_HATA] vault boş: {vault_path}")
    return kayitlar


def tokenizer_kur(vocab_path: str, lexicon_path: str) -> KristalTokenizer:
    """Gateway create_default ile aynı tokenizer hattı (anka vocab 33.114)."""
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    return KristalTokenizer(compiler, vocab)


def main() -> int:
    ayarla = argparse.ArgumentParser(description="cot_vault.jsonl → muhakeme_bellek ekim betiği (idempotent upsert)")
    ayarla.add_argument("--url", type=str, default=None,
                        help="Qdrant URL'i (ör. http://192.168.1.9:6333); yoksa QDRANT_URL env, o da yoksa localhost:6333")
    ayarla.add_argument("--vault", type=str, default="data/pedagogy/cot_vault.jsonl")
    ayarla.add_argument("--collection", type=str, default=KOLEKSIYON)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--limit", type=int, default=None, help="İlk N kayıt (smoke/probe)")
    ayarla.add_argument("--timeout", type=float, default=30.0,
                        help="Uzak Qdrant istemci zaman aşımı saniyesi (varsayılan 30; VectorMemory varsayılanı 2.0'dır)")
    ayarla.add_argument("--dry-run", action="store_true",
                        help="Bağlantı KURMAZ: kayıt/benzersiz-nokta sayısını ve şemayı basar")
    args = ayarla.parse_args()

    # URL çözümü: --url > QDRANT_URL env > localhost:6333
    hedef = resolve_qdrant_url(args.url)
    if hedef is None:
        hedef = ("localhost", 6333)
    hedef_url = args.url or __import__("os").environ.get("QDRANT_URL", VARSAYILAN_URL)
    print(f"[EKIM] hedef Qdrant: {hedef[0]}:{hedef[1]} (url={hedef_url!r})")
    print(f"[EKIM] koleksiyon: {args.collection} | vault: {args.vault}")

    kayitlar = kayitlari_yukle(args.vault)
    if args.limit:
        kayitlar = kayitlar[: args.limit]
    print(f"[EKIM] vault kaydı: {len(kayitlar)}")

    tokenizer = tokenizer_kur(args.vocab, args.lexicon)
    print(f"[EKIM] tokenizer hazır (vocab={len(tokenizer.vocab.stoi)})")

    # Önden işle: doc_text + embedding + meta (bağlantısız — dry-run dahil)
    dokumanlar, kimlikler = [], []
    simdi = datetime.now(timezone.utc).isoformat()
    for idx, rec in enumerate(kayitlar):
        doc_text = doc_text_olustur(rec)
        token_ids = tokenizer.encode(doc_text)
        tags = tokenizer.decode(token_ids)
        meta = {
            "query": rec["query"],
            "domain": "general",
            "thought_text": rec["thought_trace"],
            "final_card": rec["final_answer"],
            "crystal_tags": tags,
            "token_ids": token_ids,
            "injected_by": "seed_muhakeme_bellek",
            "timestamp": simdi,
            "source": rec.get("source", ""),
            "instruction": rec.get("instruction", ""),
            "vault_timestamp": rec.get("timestamp", ""),
            "vault_index": idx,
        }
        dokumanlar.append((doc_text, token_ids, tags, meta))
        kimlikler.append(nokta_kimligi(rec))

    benzersiz = len(set(kimlikler))
    print(f"[EKIM] benzersiz nokta (tam-kayıt sha256): {benzersiz}/{len(kayitlar)}")
    if benzersiz != len(kayitlar):
        print("[EKIM][UYARI] vault'ta birebir yinelenik TAM KAYIT var — upsert bunları TEK noktada birleştirir")

    if args.dry_run:
        print("[EKIM] --dry-run: bağlantı KURULMADI, yazma YOK")
        for i, (doc_text, _, _, meta) in enumerate(dokumanlar[:3]):
            print(f"  örnek[{i}] id={kimlikler[i]} query={meta['query']!r}")
        return 0

    # Bağlantı: T-0148 4A fail-closed (allow_fallback default False).
    # Uzak Qdrant aralıklı erişilebilir olabilir (LAN; ölçülen kalıp) →
    # geçici-ağ hatasında KONTROLLÜ yeniden-dene (toplam 3 deneme, 5s aralık);
    # koleksiyon sunucuda yaratılmışsa ikinci deneme 'var' dalından devam eder.
    import time as _time
    bellek = None
    son_hata = None
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
        except Exception as e:  # noqa: BLE001 — hata mesajı raporlanır
            son_hata = e
            print(f"[EKIM][DENEME {deneme}/3] bağlantı hatası: {type(e).__name__}: {e}")
            if deneme < 3:
                _time.sleep(5)
    if bellek is None:
        print(f"[EKIM][HATA] 3 denemede bağlanılamadı: {hedef[0]}:{hedef[1]} — DUR (uydurma yazma YOK)")
        return 1
    print(f"[EKIM] bağlandı: {bellek.storage_type}")

    onceki = bellek.get_document_count()
    print(f"[EKIM] ekim-öncesi nokta sayısı: {onceki}")

    parcaboyu = 32
    toplam_yazilan = 0
    for bas in range(0, len(dokumanlar), parcaboyu):
        dilim = dokumanlar[bas: bas + parcaboyu]
        dilim_kimlikler = kimlikler[bas: bas + parcaboyu]
        bellek.add_documents_batch(
            texts=[d[0] for d in dilim],
            dense_vectors=[generate_kristal_vector(d[1], d[2], VEKTOR_BOYUTU) for d in dilim],
            sparse_vectors=[generate_sparse_vector(d[1], d[2]) for d in dilim],
            metadatas=[d[3] for d in dilim],
            upsert=True,
            point_ids=dilim_kimlikler,
        )
        toplam_yazilan += len(dilim)
        print(f"  [EKIM] upsert {bas + len(dilim)}/{len(dokumanlar)}")

    # Read-back doğrulama
    sonraki = bellek.get_document_count()
    print(f"[EKIM] ekim-SONRASI nokta sayısı: {sonraki} (öncesi: {onceki})")
    ornekler = bellek.list_documents(limit=3)
    for o in ornekler:
        p = o["payload"]
        print(f"  [KANIT] id={o['id']} query={p.get('query')!r} domain={p.get('domain')!r} "
              f"injected_by={p.get('injected_by')!r} | {str(p.get('text', ''))[:60]!r}")

    if sonraki < benzersiz:
        print(f"[EKIM][HATA] read-back {sonraki} < beklenen {benzersiz} — DOĞRULAMA BAŞARISIZ")
        return 1
    print(f"[EKIM] TAMAM: {toplam_yazilan} kayıt upsert, koleksiyonda {sonraki} nokta (≥{benzersiz} doğrulandı)")
    bellek.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())