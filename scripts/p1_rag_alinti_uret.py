#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 — RAG-KOŞULLANMALI SFT: BİREBİR ALINTI VERİSİ ÜRETİCİSİ (GÖREV 1+2)
=====================================================================
Amaç: modele, <BELGE> zarfıyla enjekte edilen belgeyi ALINTILAYARAK (birebir
öbek + kaynak gösterimiyle) cevaplama öğretmek. VG-1'in ölçülmüş zemini:
retrieval düzeldi (İHBAR → 4857 M8 @0,70; süre → M41 @0,833) ama taban model
çıktıya taşımıyor (en-uzun-ortak-dizi 1 sözcük, 0 4-gram).

KAYNAKLAR (yerel dosyalar; Qdrant KULLANILMAZ):
- data/kulliyat_v2/gts_definitions.txt  (MADDE<TAB>anlam; 128.327 satır)
- data/legal_korpus/*.txt               (madde satırlı 16 kanun)

KAYIT ŞABLONU (iş emri):
  instruction: 'Belgeye dayanarak yanıtla.'
  input:  SORU + belge zarfı  — KANONİK prompt sözleşmesi biçiminde:
          build_rag_input(doc, soru) = '<BELGE> {doc} </BELGE> {soru}'
          (src/llm/prompt_contract.py donmuş sözleşmesi; K1/VG-1 üretim rig'i
          bu zarfı kurar → eğitim-üretim biçim birebir örtüşür)
  output: alıntılı yanıt — belgeden BİREBİR ≥1 öbek (4+ sözcük) + kaynak
          gösterimi (kanun+k madde no / sözlük maddesi)

BLOCK-128 UYUMU: her kaydın TAM zarfı (render_example) tokenizer ile ölçülür;
>117 morfem olan kayıt ATILIR (FastSampleAlignedDataset'in block-128 orta-
kırpma dalı hiç ateşlenmez). Uzun maddeler BÖLÜNMEZ — yalnız KISA maddeler
seçilir (gövde ≤42 morfem). UNK=0 zorunlu; UNK'lu kayıt atılır (tur-5 dersi).

Doğrulamalar (programatik):
- birebir-alıntı: output ile doc arasında en-uzun ortak SÖZCÜK dizisi ≥4
  (vg1_alinti_taraması kelimeler()/LCS kalıbı; boşluk-normalize);
- kaynak gösterimi: output'ta 'Madde {n}' (kanun) veya madde başı (GTS);
- UNK ölçümü ve morfem dağılımı montaj JSON'una yazılır.

Çıktılar:
- data/pedagogy/rag_alinti_sft.jsonl           (kayıt dosyası; son-50 = VAL)
- data/eval/p1_rag_alinti_montaj_{TARIH}.json  (montaj kanıtı + sha256)
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)
os.chdir(KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import render_example, build_rag_input

TARIH = time.strftime("%Y-%m-%d")
INSTRUCTION = "Belgeye dayanarak yanıtla."
MORF_TAVAN = 120          # tam zarf tavanı (block-128: x≤127 → orta-kırpma DALI HİÇ ateşlenmez)
GTS_ANLAM_MIN, GTS_ANLAM_MAKS = 4, 30   # anlam morf bandı
KANUN_GOVDE_MAKS = 45                    # madde gövdesi morf tavanı (BÖLME YOK; kısmi alıntı)
KANUN_T1_MAKS = 30                       # ≤30: tam gövde alıntısı; 31-45: ilk cümle alıntısı
HEDEF_GTS, HEDEF_KANUN = 310, 210        # toplam 520 (son-50 val dâhil)


def kelimeler(s: str):
    """vg1_alinti_taramasi kalıbı: sözcük normalizasyonu."""
    return re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9']+", s.lower())


def en_uzun_ortak_dizi(a, b):
    """Sözcük dizileri arasında en uzun ortak ardışık dizi (uzunluk, örnek)."""
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
    return en_iyi, (" ".join(a[son_idx - en_iyi: son_idx]) if en_iyi else "")


class Olcer:
    """Tokenizer ölçüm katmanı: morf sayısı, UNK tespiti."""

    def __init__(self):
        lexicon = LexiconManager()
        lexicon.load_from_tsv("data/lexicon/roots.tsv")
        compiler = CrystalCompiler(lexicon, build_default_graph())
        vocab = Vocabulary()
        vocab.load("data/rebuild/vocab_anka_r1_33114.json")
        self.tok = KristalTokenizer(compiler, vocab)
        self.vocab = vocab

    def morf(self, s: str) -> int:
        t = self.tok.decode(self.tok.encode(s))
        return len(t.split()) if t.strip() else 0

    def unk_var(self, s: str) -> bool:
        return "<UNK>" in self.tok.decode(self.tok.encode(s)).split()

    def zarf_morf(self, rec: dict) -> int:
        """TAM eğitim zarfı morf sayısı (render_example; dataset full_seq'e ≈)."""
        tam = render_example(rec["instruction"], rec["input"], rec["output"])
        return len(self.tok.encode(tam))

    def zarf_unk(self, rec: dict) -> bool:
        tam = render_example(rec["instruction"], rec["input"], rec["output"])
        return "<UNK>" in self.tok.decode(self.tok.encode(tam)).split()


# ---------------------------------------------------------------- GTS kartları
def gts_havuz(olc: Olcer):
    havuz = []
    gorulen = set()
    with open("data/kulliyat_v2/gts_definitions.txt", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            madde, anlam = line.rstrip("\n").split("\t", 1)
            madde = madde.strip()
            if not madde or madde.lower() in gorulen:
                continue
            gorulen.add(madde.lower())
            if len(madde) > 28 or len(madde.split()) > 3 or any(c.isdigit() for c in madde):
                continue
            if not re.search(r"[a-zçğıöşü]", madde):   # kısaltma/başlık-satırı değil
                continue
            anlam = re.sub(r"\s+", " ", anlam).strip()
            if len(anlam) < 12 or anlam.lower().startswith("bk."):
                continue
            if olc.unk_var(madde) or olc.unk_var(anlam):
                continue
            # birebir-alıntı kapısı sözcük-düzeyi LCS≥4 ile ölçülür (vg1 rig'i):
            # alıntılanacak anlam ≥4 SÖZCÜK olmalı (morf sayısı tek başına yetmez)
            if len(kelimeler(anlam)) < 4:
                continue
            n = olc.morf(anlam)
            if not (GTS_ANLAM_MIN <= n <= GTS_ANLAM_MAKS):
                continue
            havuz.append((madde, anlam, n))
    return havuz


GTS_SORULAR = [
    "{x} nedir?",
    "{x} ne demek?",
    "Sözlükte {x} nasıl tanımlanır?",
    "{x} kelimesinin sözlükteki anlamı nedir?",
]
# v4 (koşum-6): sorular v1'in madde-numaralı/özgün stiline DÖNDÜ. Ölçülmüş gerekçe:
# v3'ün numarasız-GENEL soru şablonları ("Bu belgede hangi kural vardır?") K3
# sonda sızıntısı yaptı (fidan 45→9 morf, "Belgedeki sözlere göre:" kalıbı) —
# genel soru register'ı, genel sonda sorgularına yakınsıyor. v1 soru stili +
# 12 epok koşumunda K3 ÖLÇÜLMÜŞ GEÇMİŞTİ (40/41/42 morf). Numarasız sorunun
# K1'e katkısı yok: endüksiyon probu gereği ailede kopya devresi yok (K1 bu
# yapıda değişmez), K3 maliyeti ise ölçülmüş.
GTS_YANITLAR = [
    "Belgeye göre {x} şöyle tanımlanır: {a} (kaynak: sözlük maddesi {x}).",
    "Sözlükte {x} şu şekilde tanımlanır: {a} (kaynak: sözlük maddesi {x}).",
    "Belgedeki tanım şudur: {a} (kaynak: sözlük maddesi {x}).",
]


def gts_kart(madde, anlam, i):
    q = GTS_SORULAR[i % len(GTS_SORULAR)].format(x=madde)
    a = GTS_YANITLAR[i % len(GTS_YANITLAR)].format(x=madde, a=anlam)
    doc = f"Sözlük maddesi {madde}: {anlam}"
    return {
        "instruction": INSTRUCTION,
        "input": build_rag_input(doc, q),
        "output": a,
        "_tip": "gts", "_doc": doc, "_kaynak": madde,
    }


# ------------------------------------------------------------- kanun kartları
MAD_RE = re.compile(r"^Madde (\d+)\s*-\s*(.*)$")
BASLIK_RE = re.compile(r"^[^a-zçğıöşü]*$")
JUNK_GOVDE = re.compile(r"yerine işlenmiştir|yürürlükten kaldırıl|md\.\s*\)\s*$")


def kanun_ayristir(yol):
    """Bir korpus dosyasını (kanun_adı, kanun_no, [ (madde_no, gövde) ]) ayıştırır."""
    kanun_adi, kanun_no, maddeler = "", os.path.basename(yol).split("_")[0], []
    ham = [l.rstrip("\n") for l in open(yol, encoding="utf-8")]
    aktif = None
    for s in ham:
        if s.startswith("# KANUN ADI:"):
            kanun_adi = s.split(":", 1)[1].strip()
            continue
        if s.startswith("#"):
            continue
        if not s.strip():
            continue
        m = MAD_RE.match(s)
        if m:
            aktif = [int(m.group(1)), [m.group(2).strip()]]
            maddeler.append(aktif)
            continue
        if aktif is not None:
            aktif[1].append(s.strip())
    temiz = []
    for no, gov in maddeler:
        satirlar = []
        for s in gov:
            if BASLIK_RE.match(s):          # BİRİNCİ BÖLÜM / GENEL HÜKÜMLER vb.
                break
            if re.match(r"^\((?:Ek fıkra|Değişik|Mülga|Derlenen madde)", s):
                continue
            satirlar.append(s)
        govde = re.sub(r"\[\d+\]", "", " ".join(satirlar))
        govde = re.sub(r"\s+", " ", govde).strip(" ;:")
        if len(govde) < 15 or "." not in govde:
            continue
        if JUNK_GOVDE.search(govde):
            continue
        temiz.append((no, govde))
    return kanun_adi, kanun_no, temiz


KANUN_SORULAR = [
    "{ad} {n}. maddede ne yazılıdır?",
    "{ad} Madde {n} ne diyor?",
    "{ad} {n}. maddede hangi kural yer alır?",
]
# v4: madde-numaralı soru stili (v1) — numarasız-general şablonlar K3'e sızdı.


def ilk_cumle(olc, govde):
    """Gövdenin ilk cümlesi; morf sayısı >22 ise ilk 8 sözcüklük birebir pencere."""
    m = re.match(r"^(.*?[.])\s", govde + " ")
    c = m.group(1) if m else govde
    if len(kelimeler(c)) >= 4 and olc.morf(c) <= 22:
        return c
    ks = govde.split(" ")
    return " ".join(ks[:8])


def kanun_kart(olc, kanun_adi, kanun_no, no, govde, govde_morf, i):
    ad = kanun_adi
    q = KANUN_SORULAR[i % len(KANUN_SORULAR)].format(ad=ad, n=no)
    doc = f"{ad} (Sayı: {kanun_no}) Madde {no} - {govde}"
    # v3: v1 çerçeve çıktıları (koşum-1'de K2+K3 GEÇTİ; koşum-2 restart şablonu
    # K3 sızıntısı yaptı ve kopya da kurmadı — endüksiyon probu gereği ailede
    # kopya devresi yok; çerçeve+alıntı öğretilebilir, birebir kopya öğretilemez)
    if govde_morf <= KANUN_T1_MAKS:
        a = f"{ad} Madde {no} uyarınca: {govde} (kaynak: Madde {no})."
    else:
        alinti = ilk_cumle(olc, govde)
        a = f"{ad} Madde {no}: {alinti} (kaynak: Madde {no})."
    return {
        "instruction": INSTRUCTION,
        "input": build_rag_input(doc, q),
        "output": a,
        "_tip": "kanun", "_doc": doc, "_kaynak": f"Madde {no}",
    }


# ------------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser(description="P1 RAG-alıntı SFT veri üreticisi")
    p.add_argument("--out", default="data/pedagogy/rag_alinti_sft.jsonl")
    p.add_argument("--meta", default=f"data/eval/p1_rag_alinti_montaj_{TARIH}.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    olc = Olcer()
    print(f"[MONTAJ] tokenizer hazır ({time.time()-t0:.1f} sn)", flush=True)

    # --- GTS
    havuz = gts_havuz(olc)
    print(f"[MONTAJ] GTS havuz (morf {GTS_ANLAM_MIN}-{GTS_ANLAM_MAKS}, UNK'suz): {len(havuz)}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(havuz)
    gts_kayitlar, gts_atik = [], {"unk": 0, "uzun": 0}
    for idx, (madde, anlam, n) in enumerate(havuz):
        if len(gts_kayitlar) >= HEDEF_GTS:
            break
        kart = gts_kart(madde, anlam, idx)
        if olc.zarf_unk(kart):
            gts_atik["unk"] += 1
            continue
        if olc.zarf_morf(kart) > MORF_TAVAN:
            gts_atik["uzun"] += 1
            continue
        gts_kayitlar.append(kart)
    print(f"[MONTAJ] GTS kart: {len(gts_kayitlar)} (atık: {gts_atik})", flush=True)

    # --- KANUN (yasalarda dönüşümlü seçim: tek kanun baskın olmaz)
    kanun_ayristirilmis = []
    for yol in sorted(glob.glob("data/legal_korpus/*.txt")):
        ad, no, maddeler = kanun_ayristir(yol)
        if not ad or not maddeler:
            continue
        kisa = []
        for no_m, govde in maddeler:
            n = olc.morf(govde)
            if n <= KANUN_GOVDE_MAKS:
                kisa.append((no_m, govde, n))
        rng.shuffle(kisa)
        kanun_ayristirilmis.append((ad, no, kisa))
        print(f"[MONTAJ] {no} {ad}: kısa aday={len(kisa)}", flush=True)

    kanun_kayitlar, kanun_atik = [], {"unk": 0, "uzun": 0, "alinti": 0}
    imlec = [0] * len(kanun_ayristirilmis)   # her kanunun sıradaki aday imleci (dönüşümlü seçim)
    while len(kanun_kayitlar) < HEDEF_KANUN:
        eklenen = False
        for law_idx, (ad, no, kisa) in enumerate(kanun_ayristirilmis):
            if len(kanun_kayitlar) >= HEDEF_KANUN:
                break
            while imlec[law_idx] < len(kisa):
                no_m, govde, n = kisa[imlec[law_idx]]
                imlec[law_idx] += 1
                kart = kanun_kart(olc, ad, no, no_m, govde, n, len(kanun_kayitlar))
                lcs, _ = en_uzun_ortak_dizi(kelimeler(kart["output"]), kelimeler(kart["_doc"]))
                if olc.zarf_unk(kart):
                    kanun_atik["unk"] += 1
                elif olc.zarf_morf(kart) > MORF_TAVAN:
                    kanun_atik["uzun"] += 1
                elif lcs < 4:
                    kanun_atik["alinti"] += 1
                else:
                    kanun_kayitlar.append(kart)
                    eklenen = True
                    break
        if not eklenen:
            break
    print(f"[MONTAJ] kanun kart: {len(kanun_kayitlar)} (atık: {kanun_atik})", flush=True)

    # --- doğrulama: birebir alıntı + kaynak gösterimi
    tumu = gts_kayitlar + kanun_kayitlar
    alinti_hata, kaynak_hata = [], []
    for k in tumu:
        y = kelimeler(k["output"])
        d = kelimeler(k["_doc"])
        lcs, ornek = en_uzun_ortak_dizi(y, d)
        if lcs < 4:
            alinti_hata.append({"tip": k["_tip"], "kaynak": k["_kaynak"], "lcs": lcs})
        if k["_tip"] == "kanun" and k["_kaynak"].lower() not in k["output"].lower():
            kaynak_hata.append({"tip": k["_tip"], "kaynak": k["_kaynak"]})
        if k["_tip"] == "gts" and k["_kaynak"].lower() not in k["output"].lower():
            kaynak_hata.append({"tip": k["_tip"], "kaynak": k["_kaynak"]})
    print(f"[MONTAJ] alıntı denetimi: {len(tumu)-len(alinti_hata)}/{len(tumu)} GEÇTİ "
          f"(hata {len(alinti_hata)}); kaynak denetimi hata: {len(kaynak_hata)}", flush=True)
    assert not alinti_hata and not kaynak_hata, "MONTAJ_DUR: alıntı/kaynak denetimi"

    # --- sıralama + val kuyruğu (son-50 bu SETE özgü; iki tip de temsil edilsin)
    son50_tip = []
    for deneme in range(100):
        rng2 = random.Random(137 + deneme)
        tumu_s = tumu[:]
        rng2.shuffle(tumu_s)
        son50_tip = [k["_tip"] for k in tumu_s[-50:]]
        if son50_tip.count("gts") >= 12 and son50_tip.count("kanun") >= 12:
            break
    tumu_s = [{kk: v for kk, v in k.items() if not kk.startswith("_")} for k in tumu_s]

    with open(args.out, "w", encoding="utf-8") as f:
        for k in tumu_s:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()

    # --- montaj kanıtı
    morflar = sorted(olc.zarf_morf(k) for k in tumu_s)
    meta = {
        "gorev": "P1 RAG-alıntı SFT montaj",
        "tarih": TARIH,
        "yapi": "v4 — v1 soru/çıktı stili (madde-numaralı sorular; koşum-6) + v1-v3 arası filtre "
                "iyileştirmeleri (sözcük≥4, gövde tavan 45, zarf tavan 120)",
        "cikti": args.out,
        "sha256": sha,
        "toplam_kayit": len(tumu_s),
        "gts_kart": len(gts_kayitlar),
        "kanun_kart": len(kanun_kayitlar),
        "val_kuyruk": "son-50 (bu sete özgü; pure_base_sft ile karışmaz)",
        "val_n": 50,
        "val_bilesim": {"gts": son50_tip.count("gts"), "kanun": son50_tip.count("kanun")},
        "train_n": len(tumu_s) - 50,
        "pencere_onerisi_n": max(1, (len(tumu_s) - 50) // 4),
        "unk_kayit": 0,
        "morf_istatistik": {"min": morflar[0], "medyan": morflar[len(morflar)//2],
                             "maks": morflar[-1], "tavan": MORF_TAVAN},
        "alinti_denetim_orani": f"{len(tumu_s)-len(alinti_hata)}/{len(tumu_s)}",
        "atik": {"gts": gts_atik, "kanun": kanun_atik},
        "kanun_dagilimi": {no: sum(1 for k in kanun_kayitlar if f"(Sayı: {no})" in k["_doc"])
                           for _, no, _ in kanun_ayristirilmis},
        "kaynaklar": ["data/kulliyat_v2/gts_definitions.txt", "data/legal_korpus/*.txt"],
        "kural": "commit YOK; Qdrant YOK; yerel dosyalar",
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[MONTAJ] yazıldı: {args.out} ({len(tumu_s)} kayıt)")
    print(f"[MONTAJ] sha256: {sha}")
    print(f"[MONTAJ] meta: {args.meta}")
    print(f"[MONTAJ] süre: {time.time()-t0:.1f} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
