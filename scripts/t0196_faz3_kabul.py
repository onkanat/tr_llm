#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0196 Faz-3 — kabul-ölçümü (BETİKTEN; MPS; onarım modeli).

T-0192 kalıbı birebir arm-A (`calis_arm` YENİDEN-KULLANIM): N=100/seed-42
eşli; train_4grams ORİJİNAL regen-train'den (T-0192 yüzeyi birebir).
Kapılar: Ezber %≤5 VE Tutarsızlık <%5 VE decomp-ROUGE ≥0,35 →
T0196_ONARIM_GECTI rc=0; aksi DUR rc=2 (üç eşik AYRI beyanlı).
Çıpa-karşılaştırma: T-0194 kol-A 0,1325/%22/0,1058."""
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
from scripts.train_t0196_arz_onarim import HALEF_SHA, HISTORY_PATH

ESIK = {"ezber_max": 5.0, "tutarsizlik_max": 5.0, "decomp_rouge_min": 0.35}
ONARIM_PATH = os.path.join(KÖK, "data/anka_b1_5_arz.pt")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0196_faz3_hukum_2026-09-29.json")


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    eg_hukum = json.load(open(os.path.join(KÖK, "data/eval/t0196_egitim_hukum_2026-09-29.json"), encoding="utf-8"))
    if eg_hukum.get("hukum") != "T0196_EGITIM_GECTI":
        print(f"T0196_FAZ3_DUR: eğitim hükmü {eg_hukum.get('hukum')}", flush=True)
        return 2
    if DEVICE.type != "mps":
        print(f"T0196_FAZ3_DUR: MPS yok ({DEVICE})", flush=True)
        return 2
    onarim_sha = compute_sha256(ONARIM_PATH)
    if onarim_sha != eg_hukum["best"]["sha256"]:
        print(f"T0196_FAZ3_DUR: onarım sha uymaz {onarim_sha}", flush=True)
        return 2
    if compute_sha256(os.path.join(KÖK, "data/anka_b1_5_best.pt")) != HALEF_SHA:
        print("T0196_FAZ3_DUR: halef sha değişti", flush=True)
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
    print("[MODELI] onarım=anka_b1_5_arz yüklendi", flush=True)

    kayit, ozet = calis_arm("A", eval_records, tokenizer, vocab, model, train_4grams,
                            decompiler, vocab.stoi.get("<EOS>", 3), vocab.stoi.get("</OUTPUT>", 9), [0])
    print("OZET:", json.dumps(ozet, ensure_ascii=False), flush=True)

    ezber = ozet["memorization_rate"]
    tutarsiz = ozet["incoherence_rate"]
    decomp = ozet["mean_rouge_l_decomp"]
    kapi = {"ezber_le5": ezber <= ESIK["ezber_max"],
             "tutarsizlik_lt5": tutarsiz < ESIK["tutarsizlik_max"],
             "decomp_ge0_35": decomp >= ESIK["decomp_rouge_min"]}
    gecti = all(kapi.values())
    hukum = "T0196_ONARIM_GECTI" if gecti else "T0196_ONARIM_DUR"
    donem = {
        "task": "T-0196", "hukum": hukum, "rc": 0 if gecti else 2,
        "esikler": ESIK, "olculen": {"ezber_rate": ezber, "tutarsizlik_rate": tutarsiz,
                                     "decomp_rouge": decomp, "raw_rouge": ozet["mean_rouge_l"]},
        "kapi_dersleri": kapi,
        "cipa_t0194_kol_a": {"decomp": 0.1325, "tutarsizlik": 22.0, "raw_rouge": 0.1058},
        "egitim_hukum": eg_hukum.get("hukum"), "egitim_sha256": onarim_sha,
        "history_path": HISTORY_PATH,
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ilan": "data/eval/t0196_onarim_ilan_2026-09-29.md",
    }
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    with open(os.path.join(KÖK, "data/eval/t0196_faz3_kayitlar.json"), "w", encoding="utf-8") as f:
        json.dump({"ozet": ozet, "kayitlar": kayit}, f, indent=2, ensure_ascii=False)
    print(f"ESIK: ezber={ezber:.2f}<=5 {ezber <= 5} | tutarsiz={tutarsiz:.2f}<5 {tutarsiz < 5} | "
          f"decomp={decomp:.4f}>=0.35 {decomp >= 0.35}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())