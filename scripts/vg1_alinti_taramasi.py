#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-1.2 — BİREBİR ALINTI TARAMASI + FARK TABLOSU (öncesi/sonrası)
================================================================
data/eval/vg1_uretim_{oncesi,sonrasi}.json üretim kanıtlarını ve
data/eval/vg1_retrieval_{oncesi,sonrasi}.json retrieval kanıtlarını okuyup:
- donmuş sorgu başına ÖNCE/SONRA: kaynak koleksiyon, skor, koşullanma,
  üretilen metin, enjekte edilen belge;
- BİREBİR ALINTI ölçümü: üretilen metin ile enjekte edilen belge arasında
  en-uzun ortak sözcük dizisi (LCS) ve paylaşılan 4-gram sayısı;
- madde-numarası/kavram geçiş taraması (4857, madde, hafta, saat…).
"""
import json
import re
import sys
from pathlib import Path

ONC = json.load(open("data/eval/vg1_uretim_oncesi.json", encoding="utf-8"))
SON = json.load(open("data/eval/vg1_uretim_sonrasi.json", encoding="utf-8"))
R_ONC = json.load(open("data/eval/vg1_retrieval_oncesi.json", encoding="utf-8"))
R_SON = json.load(open("data/eval/vg1_retrieval_sonrasi.json", encoding="utf-8"))


def kelimeler(s):
    return re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9']+", s.lower())


def en_uzun_ortak_dizi(a, b):
    """Sözcük dizileri arasında en uzun ortak ardışık dizinin (uzunluk, örnek)."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0, ""
    onceki = [0] * (lb + 1)
    en_iyi, son_idx = 0, 0
    for i in range(1, la + 1):
        simdi = [0] * (lb + 1)
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                simdi[j] = onceki[j - 1] + 1
                if simdi[j] > en_iyi:
                    en_iyi, son_idx = simdi[j], i
        onceki = simdi
    ornek = " ".join(a[son_idx - en_iyi: son_idx]) if en_iyi else ""
    return en_iyi, ornek


def ngram_say(a, b, n=4):
    """Paylaşılan n-gram sayısı (birebir alıntı yoğunluğu)."""
    ga = {" ".join(a[i:i + n]) for i in range(len(a) - n + 1)}
    gb = {" ".join(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(ga & gb)


cikti = {"sorgular": []}
for sorgu in SON["sorgular"]:
    o = ONC["sorgular"][sorgu]
    s = SON["sorgular"][sorgu]
    doc = s.get("rag_document", "") or ""
    yanit_k = kelimeler(s.get("response_text", ""))
    doc_k = kelimeler(doc)
    lcs, lcs_ornek = en_uzun_ortak_dizi(yanit_k, doc_k)
    n4 = ngram_say(yanit_k, doc_k, 4)
    yanit = s.get("response_text", "")
    kavramlar = ["4857", "madde", "hafta", "saat", "kırkbeş", "bildirim", "sözleşme"]
    satir = {
        "sorgu": sorgu,
        "once": {
            "kaynak": o.get("source_collection"), "skor": o.get("rag_score"),
            "kosullanma": o.get("conditioned"),
            "yanit": o.get("response_text"),
        },
        "sonra": {
            "kaynak": s.get("source_collection"), "skor": s.get("rag_score"),
            "kosullanma": s.get("conditioned"),
            "yanit": yanit,
            "belge_bas": doc[:300],
        },
        "birebir_alinti": {
            "en_uzun_ortak_sozcuk_dizisi": lcs,
            "ornek": lcs_ornek,
            "paylasilan_4gram": n4,
        },
        "kavram_gecisi": {k: (k.lower() in yanit.lower()) for k in kavramlar},
    }
    cikti["sorgular"].append(satir)

Path("data/eval").mkdir(parents=True, exist_ok=True)
with open("data/eval/vg1_fark_tablosu.json", "w", encoding="utf-8") as f:
    json.dump(cikti, f, ensure_ascii=False, indent=1)

for satir in cikti["sorgular"]:
    print("=" * 100)
    print("SORGU :", satir["sorgu"])
    o, s, b = satir["once"], satir["sonra"], satir["birebir_alinti"]
    print(f"  ÖNCE : kaynak={o['kaynak']} skor={o['skor']} koşullanma={o['kosullanma']}")
    print(f"         yanıt: {o['yanit'][:200]!r}")
    print(f"  SONRA: kaynak={s['kaynak']} skor={s['skor']} koşullanma={s['kosullanma']}")
    print(f"         belge: {s['belge_bas'][:160]!r}")
    print(f"         yanıt: {s['yanit'][:200]!r}")
    print(f"  ALINTI: en-uzun-ortak={b['en_uzun_ortak_sozcuk_dizisi']} sözcük "
          f"({b['ornek']!r}), 4-gram={b['paylasilan_4gram']}")
    print(f"  KAVRAM: {satir['kavram_gecisi']}")
print("\nyazıldı: data/eval/vg1_fark_tablosu.json")
