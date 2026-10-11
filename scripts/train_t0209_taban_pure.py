#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0209 — Saf Akıcı Taban Model Eğitimi (MPS).

Marangozluk ve diğer dikey mesleki verilerden arındırılmış saf Türkçe gramer,
akıl yürütme ve RAG formatı taban modeli.
- Başlangıç ağırlığı: data/anka_t0201_b05.pt
- Öğretmen: data/anka_base_v2.pt (KL-çıpa, β=0.2)
- Veri: 4.498 Saf SFT + 1.124 Pretrain Pencere (4:1 karışım) = 5.622 örnek
- Optimizasyon: batch=16, accum=2 -> 176 opt-adım (1 epoch)
- LR: peak_lr=1e-5, min_lr=2e-6, warmup=15
- Çıktı: data/anka_taban_pure.pt
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.train_step_b1_5_rigorous import (
    FastSampleAlignedDataset,
    evaluate_val_loss,
    compute_sha256,
)
from scripts.t0200_mutasyon_kanit import (
    PencereDataset,
    pencere_ce_olc,
    kl_pencere_kayip,
    yukle_checkpoint,
)

INIT_MODEL_PATH = os.path.join(KÖK, "data/anka_t0201_b05.pt")
HOCA_PATH = os.path.join(KÖK, "data/anka_base_v2.pt")
SFT_PATH = os.path.join(KÖK, "data/pedagogy/pure_base_sft.jsonl")
BIN_PATH = os.path.join(KÖK, "data/anka_a1r_pretrain.bin")
HISTORY_PATH = os.path.join(KÖK, "data/eval/t0209_taban_pure_history.json")
EGITIM_HUKUM_PATH = os.path.join(KÖK, "data/eval/t0209_taban_pure_hukum_2026-10-02.json")
BEST_MODEL_PATH = os.path.join(KÖK, "data/anka_taban_pure.pt")

BLOCK = 128
BETA = 0.2
PEAK_LR = 1e-5
MIN_LR = 2e-6
WARMUP = 15
BATCH = 16
ACCUM = 2
SFT_N = 4498
PENCERE_N = 1124  # 4:1 SFT lehine karışım (4498 SFT : 1124 Pencere)
DS_N_BEKLENEN = SFT_N + PENCERE_N  # 5622
TOTAL_OPT_STEPS = math.ceil(DS_N_BEKLENEN / (BATCH * ACCUM))  # 176

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError("T0209_DUR: MPS yok — koşum sandbox DIŞI olmalı")


class PureTabanKarısımDataset:
    def __init__(self, sft_records, tokenizer, vocab, model, device: torch.device):
        self.sft = FastSampleAlignedDataset(sft_records, tokenizer, vocab, model, block_size=BLOCK)
        self.pencere = PencereDataset(BIN_PATH, model, device, adet=PENCERE_N, seed=42)

    def __len__(self) -> int:
        return len(self.sft) + len(self.pencere)

    def __getitem__(self, idx: int):
        if idx < len(self.sft):
            x, y, sm = self.sft[idx]
            return (x, y, sm, torch.tensor(0.0))
        return self.pencere[idx - len(self.sft)]


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
        print(f"T0209_DUR: Başlangıç modeli bulunamadı: {INIT_MODEL_PATH}", flush=True)
        return 2
    if not os.path.exists(HOCA_PATH):
        print(f"T0209_DUR: Hoca modeli bulunamadı: {HOCA_PATH}", flush=True)
        return 2
    if not os.path.exists(SFT_PATH):
        print(f"T0209_DUR: SFT veri kümesi bulunamadı: {SFT_PATH}", flush=True)
        return 2

    print("=== T-0209 SAF AKICI TABAN MODEL EĞİTİMİ (MPS) ===", flush=True)
    t_basla = time.time()

    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    tokenizer = KristalTokenizer(compiler, vocab)
    vocab_size = len(vocab.stoi)

    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    yukle_checkpoint(INIT_MODEL_PATH, model)
    model.to(DEVICE)

    hoca = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    yukle_checkpoint(HOCA_PATH, hoca)
    hoca.to(DEVICE)
    hoca.eval()
    for p in hoca.parameters():
        p.requires_grad = False

    sft_records = []
    with open(SFT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sft_records.append(json.loads(line))

    print(f"Yüklenen Saf SFT: {len(sft_records)} (Beklenen: {SFT_N})", flush=True)
    val_records = sft_records[-150:]
    train_records = sft_records[:-150]

    val_dataset = FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=BLOCK)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    init_val_ce, init_ppl = evaluate_val_loss(model, val_loader)
    init_pencere_ce = pencere_ce_olc(model, DEVICE)
    print(f"[BAŞLANGIÇ] Val CE: {init_val_ce:.6f} | Pencere CE: {init_pencere_ce:.6f}", flush=True)

    karısım_ds = PureTabanKarısımDataset(train_records, tokenizer, vocab, model, DEVICE)
    train_loader = DataLoader(karısım_ds, batch_size=BATCH, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=MIN_LR, weight_decay=0.01)

    opt_step = 0
    accum_count = 0
    history = []
    running_loss = 0.0

    model.train()
    optimizer.zero_grad()

    for epoch in range(1):
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                x, y, sm, _ = batch
                is_pencere = False
            else:
                x, y, sm, kl_loss = batch
                is_pencere = True

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sm = sm.to(DEVICE)

            logits, _ = model(x)
            V = logits.size(-1)
            loss_token = F.cross_entropy(logits.view(-1, V), y.view(-1), reduction="none").view_as(sm)
            ce_loss = (loss_token * sm).sum() / (sm.sum() + 1e-8)

            if is_pencere and BETA > 0.0:
                with torch.no_grad():
                    hoca_logits, _ = hoca(x)
                hoca_probs = F.softmax(hoca_logits, dim=-1)
                student_lprobs = F.log_softmax(logits, dim=-1)
                kl_elem = F.kl_div(student_lprobs, hoca_probs, reduction="none").sum(dim=-1)
                kl_val = (kl_elem * sm).sum() / (sm.sum() + 1e-8)
                loss = ce_loss + BETA * kl_val
            else:
                loss = ce_loss

            loss_scaled = loss / ACCUM
            loss_scaled.backward()
            running_loss += loss.item()
            accum_count += 1

            if accum_count == ACCUM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr_curr = get_lr(opt_step, TOTAL_OPT_STEPS)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_curr
                optimizer.step()
                optimizer.zero_grad()

                opt_step += 1
                accum_count = 0
                step_loss = running_loss / ACCUM
                running_loss = 0.0

                if opt_step % 20 == 0 or opt_step == TOTAL_OPT_STEPS:
                    print(f"Adım [{opt_step}/{TOTAL_OPT_STEPS}] | Loss: {step_loss:.4f} | LR: {lr_curr:.2e}", flush=True)

                history.append({"step": opt_step, "loss": step_loss, "lr": lr_curr})

    final_val_ce, final_ppl = evaluate_val_loss(model, val_loader)
    final_pencere_ce = pencere_ce_olc(model, DEVICE)
    toplam_sure = time.time() - t_basla

    print(f"\n[BİTİŞ] Final Val CE: {final_val_ce:.6f} | Final Pencere CE: {final_pencere_ce:.6f} | Süre: {toplam_sure:.1f} sn", flush=True)

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    best_sha = compute_sha256(BEST_MODEL_PATH)
    print(f"[MODEL_KAYDEDİLDİ] {BEST_MODEL_PATH} (sha256: {best_sha})", flush=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    hukum = {
        "durum": "BASARILI",
        "model_path": BEST_MODEL_PATH,
        "sha256": best_sha,
        "init_val_ce": init_val_ce,
        "final_val_ce": final_val_ce,
        "init_pencere_ce": init_pencere_ce,
        "final_pencere_ce": final_pencere_ce,
        "opt_steps": opt_step,
        "sure_sn": round(toplam_sure, 1),
    }
    with open(EGITIM_HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
