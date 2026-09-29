#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0196 K1 ön-kontrol — halef modelde Kol-A sayım == BEKLENEN_A.

T-0192 kalıbı birebir (scripts/t0194_giyim_testi.calis_arm YENİDEN-
KULLANIM; halef betik dokunulmaz). Hedef: değerlendirme-yığınını
eğitim-ÖNCESİ kanıtla — aksi DUR rc=2, eğitim BAŞLAMAZ."""
import json
import os
import random
import sys
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
from scripts.evaluate_anka_b1_5 import DEVICE, compute_sha256
from scripts.t0194_giyim_testi import BEKLENEN_A, calis_arm, kelime  # halef — yalnız-OKU

HALEF_SHA = "e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7"
SPLITS_DIR = os.path.join(KÖK, "data/b1_5_splits")


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    if DEVICE.type != "mps":
        print(f"T0196_K1_DUR: MPS yok ({DEVICE})", flush=True)
        return 2
    if compute_sha256(os.path.join(KÖK, "data/anka_b1_5_best.pt")) != HALEF_SHA:
        print("T0196_K1_DUR: halef sha uymaz", flush=True)
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
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    decompiler = MorphemeDecompiler(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(os.path.join(KÖK, "data/anka_b1_5_best.pt"), map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)

    _, ozet = calis_arm("A", eval_records, tokenizer, vocab, model, train_4grams,
                        decompiler, vocab.stoi.get("<EOS>", 3), vocab.stoi.get("</OUTPUT>", 9), [0])
    bek = BEKLENEN_A
    for k, bek_d in (("n", bek["n"]), ("memorization_rate", bek["memorization_rate"]),
                     ("incoherence_rate", bek["incoherence_rate"]),
                     ("conditioning_rate", bek["conditioning_rate"]),
                     ("mean_rouge_l", bek["mean_rouge_l"])):
        fark = abs(ozet[k] - bek_d)
        if fark > (1e-6 if k == "mean_rouge_l" else 1e-9):
            print(f"T0196_K1_DUR: {k} beklenen {bek_d} olculen {ozet[k]}", flush=True)
            return 2
    print("K1: A-arm sayim BEKLENEN_A birebir — PASS", flush=True)
    print("K1_GECTI", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())