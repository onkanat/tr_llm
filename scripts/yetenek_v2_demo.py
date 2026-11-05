#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-2b — YETENEK AĞI v2 DEMO (2026-11-05)
=========================================
GÖREV-5 demo tekrarı (v1 sorguları) — gerçek 'yetenek_agi' koleksiyonunda:
  (1) KISIT EKİMİ: 5 GERÇEK kısıt kartı idempotent upsert (yetenek
      ağırlıkları DOKUNULMAZ — yalnız kısıt noktaları eklenir).
  (2) ÖNCE: 5 v1 sorgusu aktivasyon tablosu (VG-2 probu ile karşılaştırma).
  (3) AĞIRLIK EVRİMİ: 8 olay (4 uyum + 3 uyumsuz + 1 kayıt-yalnız;
      router kararları simüle — gerçek zirveler canlı koleksiyondan) işlenir,
      ağırlıklar ×(1±0,05) bantla geri yazılır, geçmiş telemetriye gider.
  (4) SONRA: aynı 5 sorgu — evrim öncesi/sonrası aktivasyon farkı.
  (5) KISIT KARTI ETKİSİ: tek sorguda baskılayıcı çarpanın etkiye yansıması.
  (6) READ-BACK: koleksiyon sayısı + evrilmiş ağırlıklar KANIT olarak basılır.

Yazma yetkisi YALNIZ 'yetenek_agi' (kısıt ekimi + ağırlık güncellemeleri).
Kullanım:
    venv/bin/python scripts/yetenek_v2_demo.py --url http://192.168.1.9:6333
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.vector_memory import resolve_qdrant_url
from src.rag.yetenek_agi import (
    TELEMETRI_YOLU, YetenekAgi, aktivasyon_olayi,
    kisit_envanteri, telemetri_ekle, telemetri_oku,
)
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

# v1 demo sorguları (VG-2 probu, data/eval/vg2_yetenek_agi_2026-11-04.md — DONMUŞ)
SORGULAR = [
    "zımpara nedir",
    "iş kanunu maddesi",
    "gideceğim kelimesini çözümle",
    "ceviz ağacı özellikleri",
    "Qdrant'ta ne var",
]

# Ağırlık evrimi simülasyon olayları (GÖREV-5: 5-10 olay, simüle sinyaller).
# router_uzmani değerleri SİMÜLE (uyum: beklenen uzman; uyumsuz: kasıtlı
# yanlış-yönlendirme; None: kayıt-yalnız). Zirveler CANLI koleksiyondan gelir.
EVrim_OLAYLARI = [
    {"sorgu": "zımpara nedir", "router_uzmani": "carpenter", "not": "gerçek beklenti (carpenter_terms 'zımpara')"},
    {"sorgu": "iş kanunu maddesi", "router_uzmani": "legal", "not": "gerçek beklenti (P25 legal 8/10)"},
    {"sorgu": "gideceğim kelimesini çözümle", "router_uzmani": "grammar_core", "not": "gerçek beklenti"},
    {"sorgu": "ceviz ağacı özellikleri", "router_uzmani": "gardener", "not": "gerçek beklenti"},
    {"sorgu": "Qdrant'ta ne var", "router_uzmani": "grammar_core", "not": "zirve uzmansız → kayıt-yalnız"},
    {"sorgu": "zımpara nedir", "router_uzmani": "pedagogy", "not": "SİMÜLE yanlış-yönlendirme → uyumsuz"},
    {"sorgu": "iş kanunu maddesi", "router_uzmani": "carpenter", "not": "SİMÜLE yanlış-yönlendirme → uyumsuz"},
    {"sorgu": "ceviz ağacı özellikleri", "router_uzmani": "legal", "not": "SİMÜLE yanlış-yönlendirme → uyumsuz"},
]

# Kısıt kartı etkisi tek sorguda: sözlük genişletme → donmuş sözlük ×0,5
KISIT_SORGUSU = "sözlüğe yeni kelime ekle bilinmeyen kelime azalt"


def tokenizer_kur(vocab_path: str, lexicon_path: str) -> KristalTokenizer:
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    return KristalTokenizer(compiler, vocab)


def tablo_bas(baslik: str, ag: YetenekAgi, satirlar: list) -> None:
    print("\n" + "=" * 96)
    print(baslik)
    print("=" * 96)
    for sonuc in satirlar:
        sorgu = sonuc["sorgu"]
        print(f"SORGU: {sorgu!r}")
        for i, (ad, skor, etki) in enumerate(sonuc["aktif_yetenekler"], start=1):
            kisit = ""
            uygulanan = (sonuc.get("kisit_etkileri") or {}).get("uygulanan") or {}
            if ad in uygulanan:
                kisit = f"  [KISIT ×{uygulanan[ad]}]"
            print(f"  {i}. {ad:<46} skor={skor:.4f}  etki={etki:.6f}{kisit}")
        print("-" * 96)


def main() -> int:
    ayarla = argparse.ArgumentParser(description="VG-2b yetenek ağı v2 demo (telemetri+evrim+kısıt)")
    ayarla.add_argument("--url", type=str, default=None)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--esik", type=float, default=0.02)
    ayarla.add_argument("--json", type=str, default="scratch/vg2b_demo.json")
    args = ayarla.parse_args()

    hedef = resolve_qdrant_url(args.url) or ("localhost", 6333)
    print(f"[DEMO] hedef Qdrant: {hedef[0]}:{hedef[1]} | telemetri: {TELEMETRI_YOLU}")

    tokenizer = tokenizer_kur(args.vocab, args.lexicon)
    ag = YetenekAgi(host=hedef[0], port=hedef[1], timeout=30.0, tokenizer=tokenizer)
    print(f"[DEMO] bağlandı: {ag.memory.storage_type} | kart sayısı (öncesi): {ag.kart_sayisi()}")
    if ag.kart_sayisi() < 36:
        print(f"[DEMO][HATA] koleksiyon çok küçük ({ag.kart_sayisi()}) — önce VG-2 ekimi")
        return 1

    # --- (1) KISIT EKİMİ (idempotent; yetenek ağırlıkları DOKUNULMAZ) ---
    kisitlar = kisit_envanteri()
    sayi_once = ag.kart_sayisi()
    r_kisit = ag.tohumla(kartlar=kisitlar)
    print(f"[DEMO] kısıt ekimi: {len(kisitlar)} kısıt kartı upsert "
          f"(yazılan={r_kisit['yazilan']}) | kart sayısı: {sayi_once} → {ag.kart_sayisi()}")
    for k in kisitlar:
        print(f"  [KISIT] {k.kimlik:<28} çarpan={k.meta['carpan']} koşul={k.meta['kosul']} "
              f"hedef={k.meta.get('hedef') or k.meta.get('hedef_alan')} kanit={k.meta['kanit']}")

    # --- (2) ÖNCE: evrim-öncesi aktivasyon (v1 sorguları) ---
    once = []
    for sorgu in SORGULAR:
        sonuc = ag.aktivasyon(sorgu, top_k=3, esik=args.esik)
        olay = aktivasyon_olayi(sorgu, sonuc, router_uzmani=None, kaynak="demo_once")
        telemetri_ekle(olay, TELEMETRI_YOLU)
        once.append({"sorgu": sorgu, **sonuc})
    tablo_bas("EVRİM ÖNCESİ aktivasyon (v1 sorguları, eşik=0,02, top_k=3)", ag, once)

    agirlik_once = {p["kimlik"]: p["agirlik"] for p in ag.kartlari_oku(limit=100)
                    if p.get("kimlik")}
    izlenecek = ["marangoz_ceketi", "legal_ceketi", "kristal_derleyici", "bahcivan_ceketi"]
    print("[DEMO] evrim-öncesi ağırlıklar (izlenen kartlar):")
    for kimlik in izlenecek:
        print(f"  {kimlik:<22} agirlik={agirlik_once.get(kimlik)}")

    # --- (3) AĞIRLIK EVRİMİ: 8 olay (simüle uyum/uyumsuz sinyalleri) ---
    print("\n" + "=" * 96)
    print("AĞIRLIK EVRİMİ — 8 olay işleniyor (eps=0,05, bant [0,5; 2,0]; "
          "sinyaller SİMÜLE, zirveler CANLI koleksiyondan)")
    print("=" * 96)
    olaylar = [{"sorgu": o["sorgu"], "router_uzmani": o["router_uzmani"],
                "kaynak": "demo_simulasyon"} for o in EVrim_OLAYLARI]
    r_evrim = ag.evrim_isle(olaylar, telemetri_yolu=TELEMETRI_YOLU)
    for o, g in zip(EVrim_OLAYLARI, r_evrim["olaylar"]):
        isaret = {"1": "UYUM      (+1)", "-1": "UYUMSUZ   (−1)", "None": "KAYIT-YALNIZ (—)"}[str(g["sinyal"])]
        print(f"  {o['sorgu']!r:<38} router={o['router_uzmani']!r:<14} "
              f"zirve={g['zirve_kimligi']} ({g['zirve_uzmani']}) → {isaret}   [{o['not']}]")
    print(f"[DEMO] işlenen={r_evrim['islenen']}  sinyalli (ağırlık değişen)={r_evrim['sinyalli']}")
    print("[DEMO] ağırlık geçmişi (kimlik: eski → yeni [sinyal]):")
    for g in r_evrim["agirlik_gecmisi"]:
        print(f"  {g['kimlik']:<22} {g['eski_agirlik']:.6f} → {g['yeni_agirlik']:.6f} "
              f"[sinyal={g['sinyal']:+d}]")

    # --- (4) SONRA: evrim-sonrası aktivasyon (aynı 5 sorgu) ---
    sonra = []
    for sorgu in SORGULAR:
        sonuc = ag.aktivasyon(sorgu, top_k=3, esik=args.esik)
        olay = aktivasyon_olayi(sorgu, sonuc, router_uzmani=None, kaynak="demo_sonra")
        telemetri_ekle(olay, TELEMETRI_YOLU)
        sonra.append({"sorgu": sorgu, **sonuc})
    tablo_bas("EVRİM SONRASI aktivasyon (aynı sorgular — ağırlıklar evrimleşti)", ag, sonra)

    agirlik_sonra = {p["kimlik"]: p["agirlik"] for p in ag.kartlari_oku(limit=100)
                     if p.get("kimlik")}
    print("[DEMO] ağırlık değişimi (öncesi → sonrası, read-back):")
    for kimlik in izlenecek:
        print(f"  {kimlik:<22} {agirlik_once.get(kimlik)} → {agirlik_sonra.get(kimlik)}")

    # Öncesi/sonrası zirve-etki farkı tablosu
    print("\n[ÖLÇÜM] evrim öncesi/sonrası ZİRVE etki farkı:")
    for o, s in zip(once, sonra):
        zo, zs = o["aktif_yetenekler"][0], s["aktif_yetenekler"][0]
        print(f"  {o['sorgu']!r:<38} {zo[0]}: etki {zo[2]:.6f} → {zs[2]:.6f}")

    # --- (5) KISIT KARTI ETKİSİ (tek sorgu) ---
    kisit_sonuc = ag.aktivasyon(KISIT_SORGUSU, top_k=5, esik=args.esik)
    olay = aktivasyon_olayi(KISIT_SORGUSU, kisit_sonuc, router_uzmani=None, kaynak="demo_kisit")
    telemetri_ekle(olay, TELEMETRI_YOLU)
    print("\n" + "=" * 96)
    print(f"KISIT KARTI ETKİSİ — sorgu: {KISIT_SORGUSU!r}")
    print("=" * 96)
    uygulanan = (kisit_sonuc.get("kisit_etkileri") or {}).get("uygulanan") or {}
    for i, (ad, skor, etki) in enumerate(kisit_sonuc["aktif_yetenekler"], start=1):
        c = uygulanan.get(ad, 1.0)
        print(f"  {i}. {ad:<46} skor={skor:.4f}  etki={etki:.6f}"
              + (f"  [KISIT ×{c}: ham etki {round(etki / c, 6):.6f} → {etki:.6f}]"
                 if c < 1.0 else ""))
    print(f"  kısıtlar (havuzda): {(kisit_sonuc.get('kisit_etkileri') or {}).get('kisitlar', [])}")
    print(f"  uygulanan çarpanlar: {uygulanan}")

    # --- (6) READ-BACK kanıtları ---
    print("\n[READ-BACK] koleksiyon sayısı:", ag.kart_sayisi())
    print(f"[READ-BACK] telemetri satır sayısı: {len(telemetri_oku(TELEMETRI_YOLU))}")
    evrim_kayitlari = [k for k in telemetri_oku(TELEMETRI_YOLU)
                       if k.get("olay") == "agirlik_evrimi"]
    print(f"[READ-BACK] telemetrideki evrim geçmişi kaydı: {len(evrim_kayitlari)}")
    for k in evrim_kayitlari[-len(r_evrim['agirlik_gecmisi']):]:
        print(f"  [KANIT] {k['kimlik']}: {k['eski_agirlik']} → {k['yeni_agirlik']} "
              f"(sinyal={k['sinyal']}, eps={k['eps']}, kaynak={k['kaynak']})")

    # determinizm kanıtı (aynı sorgu 2× → birebir)
    s = KISIT_SORGUSU
    a1, a2 = ag.aktivasyon(s, top_k=3, esik=args.esik), ag.aktivasyon(s, top_k=3, esik=args.esik)
    print(f"[ÖLÇÜM] determinizm ({s!r}): {'SAĞLANDI' if a1 == a2 else 'SAĞLANAMADI'}")

    if args.json:
        cikti = {
            "esik": args.esik,
            "kisit_kartlari": [k.kimlik for k in kisitlar],
            "once": once, "sonra": sonra,
            "agirlik_once": {k: agirlik_once.get(k) for k in izlenecek},
            "agirlik_sonra": {k: agirlik_sonra.get(k) for k in izlenecek},
            "evrim_gecmisi": r_evrim["agirlik_gecmisi"],
            "kisit_sorgusu": {"sorgu": KISIT_SORGUSU, **kisit_sonuc},
            "kart_sayisi": ag.kart_sayisi(),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(cikti, f, ensure_ascii=False, indent=2)
        print(f"[DEMO] JSON kanıt yazıldı: {args.json}")
    ag.memory.close()
    print("[DEMO] TAMAM — commit YOK (controller'a devredilir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
