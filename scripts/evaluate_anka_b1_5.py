#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0192 — B1.5 test-evaluasyonu (Anka-çıpa; T-0190 regen-test).

Katman-4 üretim-kalitesi birebir eski yüzey (evaluate_batch_generation
import'la; elle-parça-zarf + argmax + max_new=128). Katman-1 MCQ bu
turda açılmaz (aday dosyaları eski 13-Eyl split'inin ürünü — İLAN-2).
Çıktı data/eval/t0192_final_evaluation_report.json — donmuş dizindeki
eski report EZİLMEZ."""
import hashlib
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
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.evaluate_b1_5_rigorous import evaluate_batch_generation  # tarihsel betik dokunulmaz
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict


def compute_sha256(yol: str) -> str:
    with open(yol, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

CIPE = {
    "data/anka_b1_5_best.pt": "e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7",
    "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
    "data/b1_5_splits/test.jsonl": "f106e7d2c7854ea653a6039ac6fc2578ee7b8e2d36926b2763bc7ae2e2926260",
    "data/b1_5_splits/val.jsonl": "eb96241534e71ae12329de84998cfdc04aaaa50fb8cbcd6236a922d32812f6a4",
    "data/b1_5_splits/train.jsonl": "c2d8480b866f9fd0b57d7e671eb091c91e5c80cca5a4f133b62e61cdafaae39e",
}

EŞİK = {"ezber_orani_max": 10.0, "tutarsizlik_orani_max": 5.0, "rouge_l_min": 0.35}
DATA_DIR = os.path.join(KÖK, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError(f"T0192_DUR: MPS yok ({DEVICE}) — sandbox DIŞI koşum şart")


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    for yol, bek in CIPE.items():
        ölç = compute_sha256(os.path.join(KÖK, yol))
        if ölç != bek:
            print(f"T0192_DUR: çıpa-uyuşmazlığı {yol}: {ölç}")
            return 2
    print("[CIPI] 6/6 çıpa BETİKTEN teyit — PASS", flush=True)

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "rebuild", "vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(os.path.join(KÖK, "data/anka_b1_5_best.pt"), map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print("[MODELI] anka_b1_5_best yüklendi (sha çıpa BETİKTEN)", flush=True)

    rapor = evaluate_batch_generation(
        model, tokenizer, vocab,
        os.path.join(SPLITS_DIR, "test.jsonl"),
        os.path.join(SPLITS_DIR, "train.jsonl"),
        n_samples=100,
    )
    rapor["damga"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rapor["ilan"] = "data/eval/t0192_test_ilan_2026-09-29.md"
    rapor["model_sha256"] = CIPE["data/anka_b1_5_best.pt"]
    rapor["test_sha256"] = CIPE["data/b1_5_splits/test.jsonl"]

    print("RAPOR_ANAHTARLARI:", sorted(rapor.keys()), flush=True)
    with open(os.path.join(KÖK, "data/eval/t0192_final_evaluation_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print("[RAPOR] data/eval/t0192_final_evaluation_report.json yazildi", flush=True)

    # Eşik-hüküm — İLAN-önceden-sabit, alan-adları tarihsel yüzey birebir:
    #   Ezber <%10  VE  Tutarsızlık <%5  VE  ROUGE-L >=0,35 → T0192_TEST_GECTİ rc=0
    ezber = float(rapor["memorization_rate"])
    tutarsizlik = float(rapor["incoherence_rate"])
    rouge = float(rapor["mean_rouge_l"])
    ezber_ok = ezber < EŞİK["ezber_orani_max"]
    tutarsizlik_ok = tutarsizlik < EŞİK["tutarsizlik_orani_max"]
    rouge_ok = rouge >= EŞİK["rouge_l_min"]
    hukum = "T0192_TEST_GECTİ" if (ezber_ok and tutarsizlik_ok and rouge_ok) else "T0192_TEST_DUR"
    print(f"ESIK_EZBER: %{ezber:.2f} < %10.0 → {'PASS' if ezber_ok else 'DUS'}", flush=True)
    print(f"ESIK_TUTARSIZLIK: %{tutarsizlik:.2f} < %5.0 → {'PASS' if tutarsizlik_ok else 'DUS'}", flush=True)
    print(f"ESIK_ROUGE_L: {rouge:.4f} >= 0.35 → {'PASS' if rouge_ok else 'DUS'}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if hukum == "T0192_TEST_GECTİ" else 2


if __name__ == "__main__":
    sys.exit(main())