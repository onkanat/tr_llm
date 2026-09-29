#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0197 Faz-2B — üretim-saf onarım (BETİKTEN; CPU-only).

Zincir: T-0196 curated çıktısı (sha-çıpalı) ÜSTÜNE 3-eylem:
P1 dil-dışlama (İngilizce-baskın >50% ASCII, ≥6 kelime — eğitimden atla);
P2 zarf-canon ('Usta Cevabı: Teknik Çözüm:' tek-önek; 3 map BETİKTEN);
P3 meta-prompt regex-pano → atla. P4 = rapor-kolon (birebir-×≥20
kısa-kapalı-etiket). Çıktı data/eval/t0197_uretim_saf.jsonl + meta."""
import json
import os
import re
import sys
import time

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

T0196_META = os.path.join(KÖK, "data/eval/t0196_curation_meta.json")
T0196_JSONL = os.path.join(KÖK, "data/eval/t0196_kulliyat_arz_dengeli.jsonl")
OUT_PATH = os.path.join(KÖK, "data/eval/t0197_uretim_saf.jsonl")
OUT_META = os.path.join(KÖK, "data/eval/t0197_onarim_meta.json")
CANON = "Usta Cevabı: Teknik Çözüm:"
PANO = ("<INSTRUCTION>", "<OUTPUT>", "The user", "<|", "Sen Türk dili")


def en_baskin(s: str) -> bool:
    w = s.split()
    if len(w) < 6:
        return False
    return sum(1 for x in w if re.fullmatch(r"[A-Za-z]+", x)) / len(w) > 0.5


def zarf_canon(o: str):
    """P2 canonic-eylemi: 3-form → tek-ön-ek; içerik bit-birebir."""
    if o.startswith("Usta Cevabı: Teknik Çözüm:"):
        return o, None
    if o.startswith("Usta Cevabı: "):
        return CANON + o[len("Usta Cevabı: "):], "tek_usta_birlestir"
    if o.startswith("Teknik Çözüm: "):
        return "Usta Cevabı: " + o, "tek_teknik_birlestir"
    return o, None


def main() -> int:
    import hashlib
    meta0 = json.load(open(T0196_META, encoding="utf-8"))
    sha0 = hashlib.sha256(open(T0196_JSONL, "rb").read()).hexdigest()
    if sha0 != meta0["cikti_sha256"]:
        print(f"T0197_ONARIM_DUR: T-0196 curated sha uymaz {sha0}", flush=True)
        return 2
    recs = [json.loads(l) for l in open(T0196_JSONL, encoding="utf-8")]
    n0 = len(recs)

    disla_dil = 0
    disla_meta = 0
    zarf_map = {"ikili_degismedi": 0, "tek_usta_birlestir": 0, "tek_teknik_birlestir": 0}
    out_recs = []
    for r in recs:
        o = r.get("output", "")
        if en_baskin(o):
            disla_dil += 1
            continue
        if any(p in o for p in PANO):
            disla_meta += 1
            continue
        yeni, tip = zarf_canon(o)
        if tip:
            zarf_map[tip] += 1
        r2 = dict(r)
        r2["output"] = yeni
        out_recs.append(r2)
    n1 = len(out_recs)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_sha = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()

    # ---- P4: birebir-×≥20 kısa-kapalı-etiket rapor-kolonu (eğitimde KALIR) ----
    say = {}
    for r in out_recs:
        if len(r["output"].split()) <= 6:
            say[r["output"]] = say.get(r["output"], 0) + 1
    p4 = [(o, c) for o, c in say.items() if c >= 20]
    p4_n = sum(c for _, c in p4)
    p4_top10 = sorted(p4, key=lambda t: -t[1])[:10]

    zarf_değişen = zarf_map["tek_usta_birlestir"] + zarf_map["tek_teknik_birlestir"]
    tablo = {
        "task": "T-0197", "faz": "Faz-2B", "betik": "onarım",
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "t0196_kaynak": T0196_JSONL, "t0196_sha256": sha0, "n_onesi": n0,
        "P1_dil_disla": disla_dil, "P3_meta_atla": disla_meta,
        "P2_zarf_canon": zarf_map, "zarf_degisen": zarf_değişen,
        "n_sonrasi": n1,
        "P4_x20_kisa_etiket": {"farkli_output": len(p4), "satir": p4_n,
                                "top10": [{"x": c, "out": o[:80]} for o, c in p4_top10]},
        "cikti": OUT_PATH, "cikti_sha256": out_sha,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(tablo, f, ensure_ascii=False, indent=2)
    print("[ONARIM]", json.dumps(tablo, ensure_ascii=False)[:420], flush=True)
    print(f"T0197_ONARIM_GECTI n={n0}->{n1} (dil={disla_dil} meta={disla_meta} "
          f"zarf-canon={zarf_değişen})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())