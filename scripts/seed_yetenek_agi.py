#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-2 — yetenek_agi EKİM BETİĞİ (Vektörel Gezgin v1 — yetenek kartları)
======================================================================
src/rag/yetenek_agi.py'deki GERÇEK kart envanterini (36 kart) QDRANT_URL'deki
(varsayılan localhost:6333) 'yetenek_agi' koleksiyonuna yazar.

- Embedding yolu: VectorMemory/anka_bellek'in AYNISI — KristalTokenizer
  (anka vocab 33.114) + generate_kristal_vector(768) + generate_sparse_vector;
  koleksiyon şeması VectorMemory._create_hybrid_collection(768)'den türer
  (dense 768 Cosine + sparse IDF).
- IDEMPOTENT: nokta kimliği = sha256('yetenek:<kimlik>') 2^63-mod → betik
  tekrar koşunca ÇOĞALTMAZ, aynı kartları GÜNCELLER (read-back sayısı sabit).
- Yazma yetkisi YALNIZ 'yetenek_agi' (anka_bellek/muhakeme_bellek/
  simulasyon_bellek DOKUNULMAZ — YetenekAgi kurulumundaki ad-kapısı zorlar).
- Read-back: ekim sonrası sayı + örnek kart payload'ları KANIT olarak basılır.

Kullanım:
    venv/bin/python scripts/seed_yetenek_agi.py                       # env/default
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_yetenek_agi.py
    venv/bin/python scripts/seed_yetenek_agi.py --url http://192.168.1.9:6333
    venv/bin/python scripts/seed_yetenek_agi.py --dry-run             # bağlantısız sayım
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_memory import resolve_qdrant_url
from src.rag.yetenek_agi import KOLEKSIYON, VEKTOR_BOYUTU, kart_envanteri, kisit_envanteri
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

VARSAYILAN_URL = "http://localhost:6333"


def tokenizer_kur(vocab_path: str, lexicon_path: str) -> KristalTokenizer:
    """Gateway create_default ile aynı tokenizer hattı (anka vocab 33.114)."""
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    return KristalTokenizer(compiler, vocab)


def main() -> int:
    ayarla = argparse.ArgumentParser(
        description="yetenek kartları envanterini 'yetenek_agi' koleksiyonuna ekim (idempotent upsert)")
    ayarla.add_argument("--url", type=str, default=None,
                        help="Qdrant URL'i (ör. http://192.168.1.9:6333); yoksa QDRANT_URL env, o da yoksa localhost:6333")
    ayarla.add_argument("--collection", type=str, default=KOLEKSIYON)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--timeout", type=float, default=30.0,
                        help="Uzak Qdrant istemci zaman aşımı saniyesi (varsayılan 30)")
    ayarla.add_argument("--dry-run", action="store_true",
                        help="Bağlantı KURMAZ: kart sayısını ve alan dağılımını basar")
    args = ayarla.parse_args()

    hedef = resolve_qdrant_url(args.url)
    if hedef is None:
        hedef = ("localhost", 6333)
    hedef_url = args.url or __import__("os").environ.get("QDRANT_URL", VARSAYILAN_URL)
    print(f"[EKIM] hedef Qdrant: {hedef[0]}:{hedef[1]} (url={hedef_url!r})")
    print(f"[EKIM] koleksiyon: {args.collection} | vektör boyutu: {VEKTOR_BOYUTU}")

    kartlar = list(kart_envanteri()) + list(kisit_envanteri())
    alanlar = {}
    for k in kartlar:
        alanlar[k.etki_alani] = alanlar.get(k.etki_alani, 0) + 1
    print(f"[EKIM] kart envanteri: {len(kartlar)} kart "
          f"({len(kart_envanteri())} yetenek + {len(kisit_envanteri())} kısıt) "
          f"| alanlar: {alanlar}")
    if args.collection != KOLEKSIYON:
        print(f"[EKIM][HATA] yazma yetkisi YALNIZ '{KOLEKSIYON}'; '{args.collection}' REDDEDİLDİ")
        return 1

    if args.dry_run:
        print("[EKIM] --dry-run: bağlantı KURULMADI, yazma YOK")
        for k in kartlar[:3]:
            print(f"  örnek kart {k.kimlik!r} (alan={k.etki_alani}, ağırlık={k.agirlik}) "
                  f"kanit={k.meta.get('kanit')!r}")
        return 0

    tokenizer = tokenizer_kur(args.vocab, args.lexicon)
    print(f"[EKIM] tokenizer hazır (vocab={len(tokenizer.vocab.stoi)})")

    # Uzak Qdrant aralıklı erişilebilir olabilir (LAN; ölçülen kalıp) →
    # geçici-ağ hatasında KONTROLLÜ yeniden-dene (toplam 3 deneme, 5s aralık).
    from src.rag.yetenek_agi import YetenekAgi
    ag = None
    son_hata = None
    for deneme in range(1, 4):
        try:
            ag = YetenekAgi(host=hedef[0], port=hedef[1], timeout=args.timeout,
                            tokenizer=tokenizer)
            break
        except Exception as e:  # noqa: BLE001 — hata mesajı raporlanır
            son_hata = e
            print(f"[EKIM][DENEME {deneme}/3] bağlantı hatası: {type(e).__name__}: {e}")
            if deneme < 3:
                time.sleep(5)
    if ag is None:
        print(f"[EKIM][HATA] 3 denemede bağlanılamadı: {hedef[0]}:{hedef[1]} — DUR (uydurma yazma YOK)")
        print(f"[EKIM][HATA] son hata: {son_hata}")
        return 1
    print(f"[EKIM] bağlandı: {ag.memory.storage_type}")

    onceki = ag.kart_sayisi()
    print(f"[EKIM] ekim-öncesi kart sayısı: {onceki}")

    sonuc = ag.tohumla(kartlar=kartlar)
    print(f"[EKIM] upsert: {sonuc['yazilan']} kart yazıldı "
          f"(benzersiz nokta: {sonuc['benzersiz']})")

    # Read-back doğrulama
    sonraki = ag.kart_sayisi()
    print(f"[EKIM] ekim-SONRASI kart sayısı: {sonraki} (öncesi: {onceki})")
    for p in ag.kartlari_oku(limit=3):
        print(f"  [KANIT] kimlik={p.get('kimlik')!r} alan={p.get('etki_alani')!r} "
              f"ağırlık={p.get('agirlik')} kanit={p.get('kanit')!r}")

    if sonraki < len(kartlar):
        print(f"[EKIM][HATA] read-back {sonraki} < beklenen {len(kartlar)} — DOĞRULAMA BAŞARISIZ")
        return 1
    print(f"[EKIM] TAMAM: {sonuc['yazilan']} kart upsert, koleksiyonda {sonraki} nokta "
          f"(≥{len(kartlar)} doğrulandı)")
    ag.memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
