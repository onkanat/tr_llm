#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0198 Faz-3 — kabul-ölçümü (BETİKTEN; MPS; hiza-ağırlıklı onarım modeli).

T-0197 kalıbı birebir arm-A (`calis_arm` YENİDEN-KULLANIM): N=100/seed-42
eşli; train_4grams ORİJİNAL regen-train'den. BETİKTEN-ek-kapılar (İLAN-beyanlı):
yüklemsiz-pir (<28), erken-kes-pir (<17), dongu-payı (çıpa 3/100), dil-hizalı
alt-küme ayrı-özet (T-0197 çıpası İNG n=16 RAW 0,0467 ↔ TÜRKÇE n=84 RAW 0,1078).
Kapı-cümlesi (İLAN): Ezber ≤5 VE Tutarsızlık <28 VE decomp ≥0,35 VE yüklemsiz-pir
<28 VE erken-kes-pir <17 → T0198_ONARIM_GECTI rc=0; aksi DUR rc=2."""
import json
import os
import random
import sys
import time
import torch

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict
from scripts.evaluate_anka_b1_5 import DEVICE, DATA_DIR, SPLITS_DIR, compute_sha256
from scripts.t0194_giyim_testi import calis_arm, kelime  # halef — yalnız-OKU
from scripts.t0197_uretim_saf_onarim import en_baskin  # T-0197 P1 yüklem — birebir
from scripts.train_t0198_hiza import HALEF_SHA  # halef-çıpa — yalnız-OKU (history: t0198_egitim-hükümde)

ESIK = {"ezber_max": 5.0, "tutarsizlik_max": 28.0, "decomp_rouge_min": 0.35,
         "yuklemsiz_pir_max": 28.0, "erken_kes_pir_max": 17.0}
CIPA = {"tutarsizlik": 28.0, "yuklemsiz": 28.0, "erken_kes": 17.0, "dongu": 3.0,
         "decomp_rouge": 0.1165, "raw_rouge": 0.0981,
         "dil_hizali": {"ing_hedef": {"n": 16, "raw_ort": 0.0467, "decomp_ort": 0.0581},
                         "turkce_ref": {"n": 84, "raw_ort": 0.1078, "decomp_ort": 0.1276}}}
ONARIM_PATH = os.path.join(KÖK, "data/anka_b1_5_hiza.pt")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0198_faz3_hukum_2026-09-29.json")


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    eg_hukum = json.load(open(os.path.join(KÖK, "data/eval/t0198_egitim_hukum_2026-09-29.json"), encoding="utf-8"))
    if eg_hukum.get("hukum") != "T0198_EGITIM_GECTI":
        print(f"T0198_FAZ3_DUR: eğitim hükmü {eg_hukum.get('hukum')}", flush=True)
        return 2
    if DEVICE.type != "mps":
        print(f"T0198_FAZ3_DUR: MPS yok ({DEVICE})", flush=True)
        return 2
    onarim_sha = compute_sha256(ONARIM_PATH)
    if onarim_sha != eg_hukum["best"]["sha256"]:
        print(f"T0198_FAZ3_DUR: onarım sha uymaz {onarim_sha}", flush=True)
        return 2
    if compute_sha256(os.path.join(KÖK, "data/anka_b1_5_best.pt")) != HALEF_SHA:
        print("T0198_FAZ3_DUR: halef sha değişti", flush=True)
        return 2

    train_rec = [json.loads(l) for l in open(os.path.join(SPLITS_DIR, "train.jsonl"), encoding="utf-8")]
    test_records = [json.loads(l) for l in open(os.path.join(SPLITS_DIR, "test.jsonl"), encoding="utf-8")]
    eval_records = random.Random(42).sample(test_records, min(100, len(test_records)))

    train_4grams = set()
    for d in train_rec:
        w = kelime(d.get("output", ""))
        for i in range(len(w) - 3):
            train_4grams.add(tuple(w[i:i + 4]))

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "rebuild", "vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    decompiler = MorphemeDecompiler(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(ONARIM_PATH, map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print("[MODELI] onarım=anka_b1_5_hiza yüklendi", flush=True)

    kayit, ozet = calis_arm("A", eval_records, tokenizer, vocab, model, train_4grams,
                            decompiler, vocab.stoi.get("<EOS>", 3), vocab.stoi.get("</OUTPUT>", 9), [0])
    print("OZET:", json.dumps(ozet, ensure_ascii=False), flush=True)

    # ---- BETİKTEN-ek-kapılar (T-0197 İLAN-beyanı birebir) ----
    yuklemsiz = sum(r["yuklem_yok"] for r in kayit) * 100.0 / len(kayit)
    dongu = sum(r["dongu"] for r in kayit) * 100.0 / len(kayit)
    erken_kes = sum(1 for r in kayit
                     if not r["decomp_text"].rstrip().endswith((".", "!", "?", ":"))) * 100.0 / len(kayit)

    ing_raw, ing_dec, tr_raw, tr_dec, ing_n, tr_n = [], [], [], [], 0, 0
    for kk in kayit:
        hedef = eval_records[kk["idx"]].get("output", "")
        if en_baskin(hedef):
            ing_raw.append(kk["rouge_l_raw"])
            ing_dec.append(kk["rouge_l_decomp"])
            ing_n += 1
        else:
            tr_raw.append(kk["rouge_l_raw"])
            tr_dec.append(kk["rouge_l_decomp"])
            tr_n += 1

    def ort(x: list) -> float:
        return sum(x) / len(x) if x else 0.0

    dil = {"ing_hedef": {"n": ing_n, "raw_ort": ort(ing_raw), "decomp_ort": ort(ing_dec)},
           "turkce_ref": {"n": tr_n, "raw_ort": ort(tr_raw), "decomp_ort": ort(tr_dec)}}
    print("DIL-HIZALI:", json.dumps(dil, ensure_ascii=False), flush=True)

    ezber = ozet["memorization_rate"]
    tutarsiz = ozet["incoherence_rate"]
    kapi = {"ezber_le5": ezber <= ESIK["ezber_max"],
             "tutarsizlik_lt28": tutarsiz < ESIK["tutarsizlik_max"],
             "decomp_ge0_35": ozet["mean_rouge_l_decomp"] >= ESIK["decomp_rouge_min"],
             "yuklemsiz_pir_lt28": yuklemsiz < ESIK["yuklemsiz_pir_max"],
             "erken_kes_pir_lt17": erken_kes < ESIK["erken_kes_pir_max"]}
    gecti = all(kapi.values())
    hukum = "T0198_ONARIM_GECTI" if gecti else "T0198_ONARIM_DUR"
    donem = {
        "task": "T-0198", "hukum": hukum, "rc": 0 if gecti else 2,
        "esikler": ESIK,
        "olculen": {"ezber_rate": ezber, "tutarsizlik_rate": tutarsiz,
                     "decomp_rouge": ozet["mean_rouge_l_decomp"], "raw_rouge": ozet["mean_rouge_l"],
                     "yuklemsiz_pir": yuklemsiz, "erken_kes_pir": erken_kes, "dongu_pir": dongu},
        "kapi_dersleri": kapi,
        "dil_hizali_alt_kume": dil,
        "cipa_t0197": CIPA,
        "cipa_t0194_kol_a": {"decomp": 0.1325, "tutarsizlik": 22.0, "raw_rouge": 0.1058},
        "egitim_hukum": eg_hukum.get("hukum"), "egitim_sha256": onarim_sha,
        "history_path": "data/eval/t0198_training_history.json",
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ilan": "data/eval/t0198_egitim_ilan_2026-09-29.md",
    }
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    with open(os.path.join(KÖK, "data/eval/t0198_faz3_kayitlar.json"), "w", encoding="utf-8") as f:
        json.dump({"ozet": ozet, "kayitlar": kayit, "dil_hizali": dil}, f, indent=2, ensure_ascii=False)
    print(f"ESIK: ezber={ezber:.2f}<=5 {kapi['ezber_le5']} | tutarsiz={tutarsiz:.2f}<28 "
          f"{kapi['tutarsizlik_lt28']} | decomp={ozet['mean_rouge_l_decomp']:.4f}>=0.35 "
          f"{kapi['decomp_ge0_35']} | yuklemsiz={yuklemsiz:.2f}<28 "
          f"{kapi['yuklemsiz_pir_lt28']} | erken-kes={erken_kes:.2f}<17 "
          f"{kapi['erken_kes_pir_lt17']}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())