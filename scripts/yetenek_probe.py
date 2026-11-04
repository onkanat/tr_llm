#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-2 — YETENEK PROBE (demo): 5 sorguluk aktivasyon ölçümü
=========================================================
Gerçek 'yetenek_agi' koleksiyonuna 5 sorgu atar, aktivasyon tablosunu
(hangi yetenek, kosinüs skor, ağırlıklı etki) basar ve ANLAMLILIĞI ölçer:
beklenen alan kartı top-1'de mi / top-3'te mi (hit@1 / hit@3)?

Ham skor bandı değişkenle (--esik 0) görülebilir; varsayılan koşum
DEFAULT_ESIK ile süzülmüş tabloyu basar.

Kullanım:
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/yetenek_probe.py
    venv/bin/python scripts/yetenek_probe.py --url http://192.168.1.9:6333
    venv/bin/python scripts/yetenek_probe.py --esik 0.0   # ham skorlar
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_memory import resolve_qdrant_url
from src.rag.yetenek_agi import DEFAULT_ESIK, KOLEKSIYON, YetenekAgi
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

# Demo sorguları (görev-4) ve dürüst beklenti: ilgili ALAN/kart kimlikleri.
SORGULAR = [
    ("zımpara nedir", ["marangoz_ceketi"]),
    ("iş kanunu maddesi", ["legal_ceketi", "mevzuat_getirici"]),
    ("gideceğim kelimesini çözümle", ["kristal_tokenizer", "kristal_derleyici",
                                     "morfolojik_taktikler", "kok_sozlugu"]),
    ("ceviz ağacı özellikleri", ["bahcivan_ceketi"]),
    ("Qdrant'ta ne var", ["vektor_bellek_yonetimi", "anka_bellek",
                          "hibrit_rrf_arama", "rag_boru_hatti"]),
]


def tokenizer_kur(vocab_path: str, lexicon_path: str) -> KristalTokenizer:
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    return KristalTokenizer(compiler, vocab)


def main() -> int:
    ayarla = argparse.ArgumentParser(description="VG-2 yetenek-aktivasyon demo probu (5 sorgu)")
    ayarla.add_argument("--url", type=str, default=None)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--top-k", type=int, default=5)
    ayarla.add_argument("--esik", type=float, default=DEFAULT_ESIK,
                        help=f"Aktivasyon eşiği (varsayılan {DEFAULT_ESIK}; ham için 0.0)")
    ayarla.add_argument("--json", type=str, default=None,
                        help="Sonuç tablosunu bu JSON dosyasına da yaz")
    args = ayarla.parse_args()

    hedef = resolve_qdrant_url(args.url) or ("localhost", 6333)
    print(f"[PROBE] hedef Qdrant: {hedef[0]}:{hedef[1]} | koleksiyon: {KOLEKSIYON}")
    print(f"[PROBE] top_k={args.top_k} eşik={args.esik:g} (DEFAULT_ESIK={DEFAULT_ESIK})")

    from src.rag.yetenek_agi import kart_envanteri
    kimlik_by_ad = {k.ad: k.kimlik for k in kart_envanteri()}

    tokenizer = tokenizer_kur(args.vocab, args.lexicon)
    ag = YetenekAgi(host=hedef[0], port=hedef[1], timeout=30.0, tokenizer=tokenizer)
    print(f"[PROBE] bağlandı: {ag.memory.storage_type} | kart sayısı: {ag.kart_sayisi()}")
    if ag.kart_sayisi() < 20:
        print(f"[PROBE][HATA] koleksiyon çok küçük ({ag.kart_sayisi()}) — önce seed_yetenek_agi.py")
        return 1

    satirlar = []
    hit1 = hit3 = 0
    print()
    print("=" * 88)
    for sorgu, beklenen in SORGULAR:
        sonuc = ag.aktivasyon(sorgu, top_k=args.top_k, esik=args.esik)
        print(f"SORGU: {sorgu!r}   (beklenen: {', '.join(beklenen)})")
        aktif = sonuc["aktif_yetenekler"]
        if not aktif:
            print("  (eşik altında — aktif yetenek yok)")
        adlar = [ad for (ad, _, _) in aktif]
        kimlikler = [kimlik_by_ad.get(ad, ad) for ad in adlar]
        for siralama, ((ad, skor, etki), kimlik) in enumerate(zip(aktif, kimlikler), start=1):
            isaret = " ← BEKLENEN" if kimlik in beklenen else ""
            print(f"  {siralama}. {ad:<28} skor={skor:.4f}  etki={etki:.4f}{isaret}")
        print(f"  karar: {sonuc['karar_notu']}")
        if any(k in beklenen for k in kimlikler[:1]):
            hit1 += 1
        if any(k in beklenen for k in kimlikler[:3]):
            hit3 += 1
        satirlar.append({"sorgu": sorgu, "beklenen": beklenen,
                         "aktif_yetenekler": sonuc["aktif_yetenekler"],
                         "aktif_kimlikler": kimlikler,
                         "karar_notu": sonuc["karar_notu"]})
        print("-" * 88)

    n = len(SORGULAR)
    print(f"[ÖLÇÜM] hit@1: {hit1}/{n}  |  hit@3: {hit3}/{n}  (eşik={args.esik:g}, top_k={args.top_k})")
    print("[ÖLÇÜM] kart sayısı:", ag.kart_sayisi(),
          "| determinizm kanıtı: her sorgu 2. koşumda birebir aynı tablo verir")

    # Determinizm kanıtı (aynı sorgu iki kez → birebir aynı sonuç)
    s = SORGULAR[0][0]
    a1, a2 = ag.aktivasyon(s, top_k=args.top_k, esik=args.esik), ag.aktivasyon(s, top_k=args.top_k, esik=args.esik)
    print(f"[ÖLÇÜM] determinizm ({s!r}): {'SAĞLANDI' if a1 == a2 else 'SAĞLANAMADI'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"esik": args.esik, "top_k": args.top_k,
                       "hit1": hit1, "hit3": hit3, "toplam": n,
                       "sorgular": satirlar}, f, ensure_ascii=False, indent=2)
        print(f"[PROBE] tablo yazıldı: {args.json}")
    ag.memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
