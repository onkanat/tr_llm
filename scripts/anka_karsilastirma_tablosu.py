#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ANKA · KARŞILAŞTIRMA TABLOSU — ham JSON'lardan, ELLE SAYI YAZILMADAN (plan §Doğrulama #6).

NEDEN BETİK: T-0096'nın en pahalı dersi, elle yazılmış bir eşiğin (B = −5,0; gerçeği +5,0)
gerekçe metnini ters çevirmesiydi. Bu araç, karşılaştırma satırlarını **dosyadan okur**;
hiçbir sayı burada saklanmaz.

Okunan kaynaklar (hepsi kanonik `evaluate_carpenter_anka.py` şeması):
  · `ceket_ekseni` : ezber · tutarsızlık · ROUGE-L · içerik kesişimi
  · `A_ekseni` / `A_ekseni_taban` : Wikipedia CE (dil-modeli sağlığı) ve unutma
  · `B_ekseni` : noktalama top-1
  · `hukum` : ilan edilmiş kapı kararları (betik YENİDEN HESAPLAMAZ, kaydı okur)

ppl, `scripts/taban_akicilik_tanisi.py` çıktılarından (varsa) `--ppl` ile iliştirilir.

Kullanım:
  venv/bin/python scripts/anka_karsilastirma_tablosu.py            # kayıtlı tarifler
  venv/bin/python scripts/anka_karsilastirma_tablosu.py --sade     # dar tablo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

D = "data/eval"
# (etiket, kanonik JSON, ppl JSON ya da None, not)
TARIFLER: List[Tuple[str, str, Optional[str], str]] = [
    ("taban anka_a1r",        f"{D}/anka_r23_taban_yetenek_2026-09-21.json",
     f"{D}/anka_r34_akicilik_taban_2026-09-22.json", "ön-eğitilmiş taban"),
    ("modül @1000 (r30)",     f"{D}/anka_r30_cikis_kafasi_yetenek_2026-09-21.json", None, "modül-only, +lm_head"),
    ("modül @3000 (r32)",     f"{D}/anka_r32_adim_yetenek_2026-09-22.json", None, "modül-only, 3× adım"),
    ("seg_6 tek başına",      "scratch/t0096_kos/eval_seg_6.json",
     f"{D}/anka_r34_akicilik_seg6_2026-09-22.json", "tam ince ayar (yetenek)"),
    ("seg_6 + onarıcı modül", f"{D}/anka_r33_yetkin_temel_yetenek_2026-09-22.json",
     f"{D}/anka_r34_akicilik_seg6_modul_2026-09-22.json", "onarım"),
    ("seg_1 tek başına",      "scratch/t0096_kos/eval_seg_1.json",
     f"{D}/anka_r34_akicilik_seg1_2026-09-22.json", "tam ince ayar @1000"),
    ("seg_1 + modül",         f"{D}/anka_r34_son_tur_yetenek_2026-09-22.json", None, "r34"),
    ("anka_i1 lr2e-4 (DÜŞTÜ)", f"{D}/anka_i1_faz1_seg1_yetenek_2026-09-22.json", None,
     "talimat aşaması, kapı segment 1'de düştü"),
    ("anka_i1 lr2e-5 seg_1",   f"{D}/anka_i1_lr2e5_seg1_yetenek_2026-09-22.json", None,
     "talimat aşaması, lr 10× düşürüldü"),
    ("anka_i1 lr2e-5 nihai",   f"{D}/anka_i1_lr2e5_yetenek_2026-09-22.json", None,
     "talimat aşaması, 6×500 adım (koşum sonunda)"),
]


def oku(yol: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(yol):
        return None
    try:
        return json.load(open(yol, encoding="utf-8"))
    except Exception:
        return None


def satir(etiket: str, kanonik: str, ppl_yolu: Optional[str], not_: str) -> Dict[str, Any]:
    d = oku(kanonik)
    if d is None:
        return {"etiket": etiket, "var": False, "not": not_}
    c = d.get("ceket_ekseni", {})
    h = d.get("hukum", {})
    A = d.get("A_ekseni", {})
    Ab = d.get("A_ekseni_taban")
    s: Dict[str, Any] = {
        "etiket": etiket, "var": True, "not": not_,
        "rouge": c.get("rouge_l_ort"), "tutarsizlik": c.get("tutarsizlik_orani"),
        "kesisim": c.get("kesisim_orani"), "ezber": c.get("ezber_orani"),
        "ce": A.get("CE_ort"), "ppl_ce": round(math.exp(A["CE_ort"]), 2) if A.get("CE_ort") else None,
        "ce_taban": Ab.get("CE_ort") if Ab else None,
        "a_artis": h.get("A_artis_yuzde"), "b_dusus": h.get("B_dusus_puan"),
        "unutma_gec": h.get("unutma_gec"),
        "hukum_var": bool(h),
    }
    if s["ce"] and s["ce_taban"]:
        s["ce_artis"] = round(100.0 * (s["ce"] / s["ce_taban"] - 1.0), 2)
    p = oku(ppl_yolu) if ppl_yolu else None
    s["ppl"] = p.get("ppl") if p else None
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Anka karşılaştırma tablosu (ham JSON'dan)")
    ap.add_argument("--sade", action="store_true")
    a = ap.parse_args()

    satirlar = [satir(*t) for t in TARIFLER]
    var = [s for s in satirlar if s["var"]]
    yok = [s for s in satirlar if not s["var"]]

    print("=" * 108)
    print(" ANKA · KARŞILAŞTIRMA TABLOSU — ham JSON'lardan üretildi (elle sayı yok)")
    print("=" * 108)
    print("%-24s %-8s %-8s %-8s %-7s %-9s %-8s %-9s %s" % (
        "tarif", "ROUGE", "tutarsz", "kesişim", "ezber", "CE(ppl)", "A artış", "B düşüş", "ppl"))
    print("-" * 108)
    for s in var:
        if a.sade:
            print("%-24s %-8s %-8s %-8s" % (s["etiket"], _f(s["rouge"], 4), _f(s["tutarsizlik"], 1),
                                            _f(s["kesisim"], 1)))
            continue
        print("%-24s %-8s %-8s %-8s %-7s %-9s %-8s %-9s %s" % (
            s["etiket"], _f(s["rouge"], 4), _f(s["tutarsizlik"], 1), _f(s["kesisim"], 1),
            _f(s["ezber"], 1),
            ("%.4f" % s["ce"]) if s.get("ce") else "—",
            ("%+.2f%%" % s["a_artis"]) if s.get("a_artis") is not None else ("%+.2f%%" % s["ce_artis"]) if s.get("ce_artis") is not None else "—",
            ("%+.2f" % s["b_dusus"]) if s.get("b_dusus") is not None else "—",
            ("%.2f" % s["ppl"]) if s.get("ppl") else "—"))
    print("-" * 108)
    # İLAN EDİLMİŞ KAPILAR — koddan okunur, elle yazılmaz
    import scripts.evaluate_carpenter_anka as ECA
    print(" ilan edilmiş eşikler (kaynaktan): ezber<%.0f%% · tutarsızlık<%.0f%% · ROUGE≥%.2f · "
          "kesişim≥%.0f%% · A artış≤+%.0f%% · B düşüş≤%.1f" % (
              ECA.ESIK_EZBER, ECA.ESIK_TUTARSIZ, ECA.ESIK_ROUGE, ECA.ESIK_KESISIM,
              ECA.ESIK_A_ARTIS, ECA.ESIK_B_DUSUS))
    print(" A ekseni dil-modeli sağlığıdır: taban CE 3,5352 (ppl 34,30); "
          "ilan edilmiş tavan CE %.4f (ppl %.2f)" % (
              3.5351739511825144 * 1.1, math.exp(3.5351739511825144 * 1.1)))
    if yok:
        print("\n ÖLÇÜLEMEDİ (dosya yok — 'değişti' DEĞİL, okunamadı):")
        for s in yok:
            print("   · %s" % s["etiket"])
    return 0


def _f(x: Any, n: int) -> str:
    if x is None:
        return "—"
    return ("%%.%df" % n) % x


if __name__ == "__main__":
    sys.exit(main() or 0)
