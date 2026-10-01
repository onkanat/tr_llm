#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0204 — Saf SFT Format Parlatması ve Son Eşik Kapanışı (MPS).

- Başlangıç Modeli: data/anka_t0203_polish.pt
- Veri: 5.491 SFT kaydı (FastSampleAlignedDataset)
- Batch: 16, Accum: 4 -> 86 opt-adım
- LR: peak_lr=4e-6, min_lr=1e-6, warmup=10
- Çıktı: data/anka_t0204_pure.pt
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict
from scripts.train_step_b1_5_rigorous import (
    FastSampleAlignedDataset,
    evaluate_val_loss,
    compute_sha256,
)
from scripts.t0200_mutasyon_kanit import (
    pencere_ce_olc,
    yukle_checkpoint,
)

INIT_MODEL_PATH = os.path.join(KÖK, "data/anka_t0203_polish.pt")
INIT_EXPECTED_SHA = "4f4183caec729f61eea4489c4faf2f14dbb7ed7b8e0843dabeeba5e97d862b2a"
SFT_PATH = os.path.join(KÖK, "data/eval/t0197_uretim_saf.jsonl")
HISTORY_PATH = os.path.join(KÖK, "data/eval/t0204_training_history.json")
EGITIM_HUKUM_PATH = os.path.join(KÖK, "data/eval/t0204_egitim_hukum_2026-10-01.json")
BEST_MODEL_PATH = os.path.join(KÖK, "data/anka_t0204_pure.pt")

BLOCK = 128
PEAK_LR = 4e-6
MIN_LR = 1e-6
WARMUP = 10
BATCH = 16
ACCUM = 4
TOTAL_OPT_STEPS = math.ceil(5491 / (BATCH * ACCUM))  # 86 adım

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError("T0204_DUR: MPS yok — koşum sandbox DIŞI olmalı")


def get_lr(step: int, total_steps: int) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, total_steps - WARMUP)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if not os.path.exists(INIT_MODEL_PATH):
        print(f"T0204_DUR: Başlangıç modeli bulunamadı: {INIT_MODEL_PATH}", flush=True)
        return 2

    init_sha = compute_sha256(INIT_MODEL_PATH)
    if init_sha != INIT_EXPECTED_SHA:
        print(f"T0204_DUR: Init sha uymaz: {init_sha} != {INIT_EXPECTED_SHA}", flush=True)
        return 2

    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    yukle_checkpoint(INIT_MODEL_PATH, model)
    model.to(DEVICE)
    model.train()

    sft_records = [json.loads(l) for l in open(SFT_PATH, encoding="utf-8")]
    ds = FastSampleAlignedDataset(sft_records, tokenizer, vocab, model, block_size=BLOCK)
    print(f"[VERI] Saf SFT kayıtları: {len(ds)} — opt_adim: {TOTAL_OPT_STEPS}", flush=True)

    val_ds = torch.load(os.path.join(KÖK, "data/b1_5_splits/anka_fast_ds_val.pt"), weights_only=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    train_loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)

    init_val, init_ppl = evaluate_val_loss(model, val_loader)
    init_ce = pencere_ce_olc(model, DEVICE)
    print(f"[INIT] val_loss {init_val:.4f} (PPL {init_ppl:.2f}) | pencere-CE {init_ce:.4f}", flush=True)

    history = {
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ilan": "data/eval/t0204_egitim_ilan_2026-10-01.md",
        "recete": {
            "epochs": 1, "batch": BATCH, "accum": ACCUM,
            "peak_lr": PEAK_LR, "min_lr": MIN_LR, "warmup": WARMUP,
            "ds_n": len(ds), "opt_steps": TOTAL_OPT_STEPS
        },
        "init": {"val_loss": init_val, "val_ppl": init_ppl, "pencere_ce": init_ce},
        "history": []
    }

    global_step = 0
    t0 = time.time()
    model.train()

    for b_idx, (x, y, sm) in enumerate(train_loader):
        x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
        _, loss = model(x, y, sm)
        (loss / ACCUM).backward()

        if (b_idx + 1) % ACCUM == 0 or (b_idx + 1) == len(train_loader):
            global_step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(global_step, TOTAL_OPT_STEPS)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.mps.empty_cache()

            if global_step % 20 == 0 or global_step == 1:
                print(f"  adım {global_step}/{TOTAL_OPT_STEPS} | loss {loss.item():.4f} | lr {pg['lr']:.7f}", flush=True)

    ep_val, ep_ppl = evaluate_val_loss(model, val_loader)
    ep_ce = pencere_ce_olc(model, DEVICE)
    torch.mps.empty_cache()

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    best_sha = compute_sha256(BEST_MODEL_PATH)
    dur = time.time() - t0

    history["history"].append({
        "epoch": 1, "val_loss": ep_val, "val_ppl": ep_ppl,
        "pencere_ce": ep_ce, "duration": dur, "sha256": best_sha
    })
    history["best"] = {
        "path": BEST_MODEL_PATH, "sha256": best_sha,
        "val_loss": ep_val, "val_ppl": ep_ppl, "pencere_ce": ep_ce
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[BEST] val_loss: {ep_val:.4f} (init {init_val:.4f}, Δ={ep_val - init_val:+.4f}) | "
          f"pencere-CE: {ep_ce:.4f} (init {init_ce:.4f}) -> {BEST_MODEL_PATH} sha={best_sha[:16]}...", flush=True)

    kapilar = {
        "val_loss_dustu_veya_korundu": ep_val <= init_val + 0.01,
        "pencere_ce_korundu": ep_ce < 3.75,
        "opt_adim_tamamlandi": global_step >= TOTAL_OPT_STEPS - 2
    }
    gecti = all(kapilar.values())
    hukum = "T0204_EGITIM_GECTI" if gecti else "T0204_EGITIM_DUR"
    hukum_dict = {
        "task": "T-0204", "hukum": hukum, "rc": 0 if gecti else 2,
        "kapilar": kapilar,
        "init": {"val_loss": init_val, "pencere_ce": init_ce},
        "best": history["best"],
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(EGITIM_HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(hukum_dict, f, indent=2, ensure_ascii=False)

    for k, v in kapilar.items():
        print(f"  KAPI {k}: {v}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())
