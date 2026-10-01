#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0203 Faz-3 — Polish (anka_t0203_polish.pt) kabul ve akıcılık ölçümü.

Eşikler:
- Ezber <= 5.0
- Tutarsızlık < 28.0 (>= 30 otomatik-DUR)
- Decomp ROUGE-L >= 0.35
- Yüklemsiz < 28.0
- Erken kesme < 17.0
M1-ÇEVRİM:
- scripts/taban_akicilik_tanisi.py --model data/anka_t0203_polish.pt --device mps
  -> data/eval/t0203_m1_polish.json (hedef ppl < 40.0 akıcılık koruma kanıtı).
"""
import json
import os
import random
import subprocess
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
from scripts.t0194_giyim_testi import calis_arm, kelime

ESIK = {
    "ezber_max": 5.0,
    "tutarsizlik_max": 28.0,
    "decomp_rouge_min": 0.35,
    "yuklemsiz_pir_max": 28.0,
    "erken_kes_pir_max": 17.0
}
CIPA = {
    "tutarsizlik_t0192": 22.0,
    "tutarsizlik_t0197_esigi": 28.0,
    "zincir_son_t0198": 30.0,
    "tutarsizlik_t0202_b05": 35.0,
    "halef_ppl_beyan": 50.75,
    "t0202_b05_ppl": 33.12,
    "decomp_rouge": 0.1165,
    "decomp_rouge_t0202": 0.1493
}
MODEL_PATH = os.path.join(KÖK, "data/anka_t0203_polish.pt")
EXPECTED_SHA = "4f4183caec729f61eea4489c4faf2f14dbb7ed7b8e0843dabeeba5e97d862b2a"
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0203_faz3_hukum_2026-10-01.json")
KAYIT_PATH = os.path.join(KÖK, "data/eval/t0203_faz3_kayitlar.json")
AKICILIK_BETIK = os.path.join(KÖK, "scripts", "taban_akicilik_tanisi.py")
M1_OUTPUT = os.path.join(KÖK, "data/eval/t0203_m1_polish.json")


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)

    if DEVICE.type != "mps":
        print(f"T0203_FAZ3_DUR: MPS yok ({DEVICE})", flush=True)
        return 2

    if not os.path.exists(MODEL_PATH):
        print(f"T0203_FAZ3_DUR: model dosyası bulunamadı: {MODEL_PATH}", flush=True)
        return 2

    model_sha = compute_sha256(MODEL_PATH)
    if model_sha != EXPECTED_SHA:
        print(f"T0203_FAZ3_DUR: model sha uymaz: {model_sha} != {EXPECTED_SHA}", flush=True)
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

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print(f"[MODEL] onarım=anka_t0203_polish yüklendi (sha={model_sha[:16]}...)", flush=True)

    kayit, ozet = calis_arm("A", eval_records, tokenizer, vocab, model, train_4grams,
                            decompiler, vocab.stoi.get("<EOS>", 3), vocab.stoi.get("</OUTPUT>", 9), [0])
    print("OZET:", json.dumps(ozet, ensure_ascii=False), flush=True)

    yuklemsiz = sum(r["yuklem_yok"] for r in kayit) * 100.0 / len(kayit)
    dongu = sum(r["dongu"] for r in kayit) * 100.0 / len(kayit)
    erken_kes = sum(1 for r in kayit if not r["decomp_text"].rstrip().endswith((".", "!", "?", ":"))) * 100.0 / len(kayit)

    ezber = ozet["memorization_rate"]
    tutarsiz = ozet["incoherence_rate"]

    kapilar = {
        "ezber_le5": ezber <= ESIK["ezber_max"],
        "tutarsizlik_lt28": tutarsiz < ESIK["tutarsizlik_max"],
        "tutarsizlik_ge30_otomatik_dur": tutarsiz < CIPA["zincir_son_t0198"],
        "decomp_ge0_35": ozet["mean_rouge_l_decomp"] >= ESIK["decomp_rouge_min"],
        "yuklemsiz_pir_lt28": yuklemsiz < ESIK["yuklemsiz_pir_max"],
        "erken_kes_pir_lt17": erken_kes < ESIK["erken_kes_pir_max"]
    }
    basari_beyani = {
        "tutarsizlik_le22_t0192": tutarsiz <= CIPA["tutarsizlik_t0192"],
        "beyan": "kapı DEĞİL — zincir-çıpa %22'ye dönüş-beyanı"
    }

    # ---- M1-ÇEVRİM: Taban Akıcılık Ölçümü ----
    m1 = None
    olcum = subprocess.run([sys.executable, AKICILIK_BETIK, "--model", MODEL_PATH,
                            "--device", "mps", "--output", M1_OUTPUT],
                           capture_output=True, text=True, cwd=KÖK)
    if olcum.returncode == 0:
        m1 = json.load(open(M1_OUTPUT, encoding="utf-8"))
        print(f"[M1-ÇEVRİM] polish ppl={m1['ppl']} ce={m1['ce_ort']} "
              f"(beklenen ppl < 50.75: {m1['ppl'] < 50.75})", flush=True)
    else:
        print(f"[M1-ÇEVRİM-ISTISNA] rc={olcum.returncode} {olcum.stderr.strip()[-200:]}", flush=True)

    gecti = all(kapilar.values())
    hukum = "T0203_ONARIM_GECTI" if gecti else "T0203_ONARIM_DUR"
    donem = {
        "task": "T-0203",
        "hukum": hukum,
        "rc": 0 if gecti else 2,
        "model": "data/anka_t0203_polish.pt",
        "sha256": model_sha,
        "esikler": ESIK,
        "olculen": {
            "ezber_rate": ezber,
            "tutarsizlik_rate": tutarsiz,
            "decomp_rouge": ozet["mean_rouge_l_decomp"],
            "raw_rouge": ozet["mean_rouge_l"],
            "yuklemsiz_pir": yuklemsiz,
            "erken_kes_pir": erken_kes,
            "dongu_pir": dongu
        },
        "kapi_dersleri": kapilar,
        "basari_beyani": basari_beyani,
        "m1_cevrim": {
            "kayit": M1_OUTPUT,
            "ppl": m1["ppl"] if m1 else None,
            "ce_ort": m1["ce_ort"] if m1 else None,
            "halef_ppl_beyan": CIPA["halef_ppl_beyan"],
            "ppl_lt_halef": (m1["ppl"] < 50.75) if m1 else None,
            "kayit_turu": "kapı DEĞİL — taban-akıcılık ÇEVRİM-kanıtı"
        },
        "cipa_tablo": CIPA,
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    with open(KAYIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"ozet": ozet, "kayitlar": kayit, "m1_cevrim": donem["m1_cevrim"]}, f, indent=2, ensure_ascii=False)

    print(f"ESIK: ezber={ezber:.2f}<=5 {kapilar['ezber_le5']} | "
          f"tutarsiz={tutarsiz:.2f}<28 {kapilar['tutarsizlik_lt28']} | "
          f"decomp={ozet['mean_rouge_l_decomp']:.4f}>=0.35 {kapilar['decomp_ge0_35']} | "
          f"yuklemsiz={yuklemsiz:.2f}<28 {kapilar['yuklemsiz_pir_lt28']} | "
          f"erken-kes={erken_kes:.2f}<17 {kapilar['erken_kes_pir_lt17']} | "
          f"basari-beyani<=22 {basari_beyani['tutarsizlik_le22_t0192']}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())
