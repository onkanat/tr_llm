#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: FAZ B2 RAG-SENTEZ PİLOT MODEL EĞİTİMİ
=================================================================
1. `data/rag_pilot/train_800.jsonl` (800 kayıt) üzerinde ince ayar (fine-tuning).
2. `block_size = 256` (128 -> 256 bağlam ölçekleme, 512 ekstrapolasyon riskinden kaçınma).
3. Causal Prompt Masking (-100) ile sadece `<OUTPUT>` tokenleri üzerinde gradyan.
4. `data/rag_pilot/val_100.jsonl` üzerinde her epoch başı ve sonu loss/ppl izleme.
5. Minimum Val Loss checkpoint'i `data/rag_pilot/kristal_rag_pilot_best.pt` olarak kaydeder.
6. Ön-Kayıtlı Kural: Val_final <= Val_init * 0.90 kontrolü (FAIL-A vs FAIL-B ayrımı için).
"""

import os
import sys
import json
import time
import math
import shutil
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")

TRAIN_PATH = os.path.join(PILOT_DIR, "train_800.jsonl")
VAL_PATH = os.path.join(PILOT_DIR, "val_100.jsonl")
INIT_CKPT_PATH = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
BEST_CKPT_PATH = os.path.join(PILOT_DIR, "kristal_rag_pilot_best.pt")


class FastRagPilotDataset(Dataset):
    """Precomputes x, y, and sign_mask tensors for RAG fine-tuning."""
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: KristalTokenizer,
        vocab: Vocabulary,
        model: KristalLM,
        block_size: int = 256
    ):
        self.block_size = block_size
        self.pad_id = vocab.stoi.get("<PAD>", 1)
        self.bos_id = vocab.stoi.get("<BOS>", 2)
        self.eos_id = vocab.stoi.get("<EOS>", 3)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for rec in records:
            inst = rec.get("instruction", "").strip()
            inp = rec.get("input", "").strip()
            belge = rec.get("belge", "").strip()
            out = rec.get("output", "").strip()

            parts = []
            if inst:
                parts.extend(["<INSTRUCTION>", inst, "</INSTRUCTION>"])
            parts.extend(["<INPUT>", inp, "</INPUT>"])
            if belge:
                parts.extend(["<BELGE>", belge, "</BELGE>"])
            parts.append("<OUTPUT>")
            prompt_str = " ".join(parts)

            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == self.eos_id:
                prompt_ids = prompt_ids[:-1]

            out_ids = tokenizer.encode(out)
            if out_ids and out_ids[0] == self.bos_id:
                out_ids = out_ids[1:]
            if not out_ids or out_ids[-1] != self.eos_id:
                out_ids.append(self.eos_id)

            full_seq = prompt_ids + out_ids
            P = len(prompt_ids)
            if len(full_seq) < 3:
                continue

            # Crop if sequence exceeds block_size + 1
            if len(full_seq) > self.block_size + 1:
                excess = len(full_seq) - (self.block_size + 1)
                if P - 4 > excess:
                    prompt_ids = prompt_ids[:2] + prompt_ids[2 + excess:]
                    P = len(prompt_ids)
                    full_seq = prompt_ids + out_ids
                else:
                    full_seq = full_seq[-(self.block_size + 1):]
                    P = max(1, P - excess)

            x_ids = full_seq[:-1]
            y_ids = full_seq[1:]

            # Mask prompt tokens with -100 so gradients only compute on output
            y_masked = []
            for i, target_token in enumerate(y_ids):
                if i < P - 1:
                    y_masked.append(-100)
                else:
                    y_masked.append(target_token)

            # Pad to block_size
            curr_len = len(x_ids)
            if curr_len < self.block_size:
                pad_len = self.block_size - curr_len
                x_ids = x_ids + [self.pad_id] * pad_len
                y_masked = y_masked + [-100] * pad_len
            else:
                x_ids = x_ids[:self.block_size]
                y_masked = y_masked[:self.block_size]

            x_tensor = torch.tensor(x_ids, dtype=torch.long)
            y_tensor = torch.tensor(y_masked, dtype=torch.long)
            sign_mask = model.embedding.compute_sign_mask(x_tensor.unsqueeze(0))[0]
            self.samples.append((x_tensor, y_tensor, sign_mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def evaluate_val_loss(model: KristalLM, val_loader: DataLoader) -> Tuple[float, float]:
    """Evaluates validation loss and perplexity across the full val set."""
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, sign_mask in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sign_mask = sign_mask.to(DEVICE)
            _, loss = model(x, y, sign_mask)
            if loss is not None:
                losses.append(loss.item())

    model.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl


def main():
    print("=" * 75)
    print(" FAZ B2: RAG-SENTEZ PİLOT MODEL EĞİTİMİ (MPS / CUDA / CPU)")
    print("=" * 75)
    print(f"Cihaz: {DEVICE}")

    # 1. Sözlük, Lexicon, Tokenizer Yükleme
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    print(f"Kelime Dağarcığı (Vocab): {len(vocab.stoi):,} token")

    # 2. Model Yükleme (kristal_b1_5_best.pt)
    model = KristalLM(
        vocab_size=len(vocab.stoi),
        n_embd=768,
        vocab=vocab,
        block_size=4096,
        n_layer=6,
        n_head=6
    )
    if not os.path.exists(INIT_CKPT_PATH):
        raise FileNotFoundError(f"Başlangıç checkpoint'i bulunamadı: {INIT_CKPT_PATH}")
    try:
        sd = torch.load(INIT_CKPT_PATH, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Başlangıç checkpoint'i yüklenirken hata oluştu ({INIT_CKPT_PATH}): {e}") from e

    sha256_val = compute_sha256(INIT_CKPT_PATH)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(INIT_CKPT_PATH)} sha256={sha256_val} anahtar={len(sd)}")
    load_res = model.load_state_dict(sd, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(f"[SOYAGACI_UYARI] strict=False ile yüklendi: eksik={len(load_res.missing_keys)}, fazla={len(load_res.unexpected_keys)}")
        if load_res.missing_keys:
            print(f"  * Eksik anahtarlar: {load_res.missing_keys[:5]}{'...' if len(load_res.missing_keys) > 5 else ''}")
        if load_res.unexpected_keys:
            print(f"  * Fazla anahtarlar: {load_res.unexpected_keys[:5]}{'...' if len(load_res.unexpected_keys) > 5 else ''}")
    else:
        print("[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).")
    print(f"B1.5 Checkpoint Yüklendi: '{INIT_CKPT_PATH}'")

    model.to(DEVICE)

    # 3. Veri Kümelerini Yükle
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]

    print(f"Eğitim Verisi: {len(train_records)} örnek")
    print(f"Doğrulama Verisi: {len(val_records)} örnek")

    block_size = 256
    train_cache = os.path.join(PILOT_DIR, "train_fast_ds.pt")
    val_cache = os.path.join(PILOT_DIR, "val_fast_ds.pt")

    if os.path.exists(train_cache):
        print(f"Önbellekten Yükleniyor: {train_cache}...", flush=True)
        train_ds = torch.load(train_cache, weights_only=False)
    else:
        print(f"Eğitim Kümesi Tokenize Ediliyor (block_size={block_size})...", flush=True)
        train_ds = FastRagPilotDataset(train_records, tokenizer, vocab, model, block_size=block_size)
        torch.save(train_ds, train_cache)
        print(f"Train Dataset Önbelleğe Alındı: {train_cache}", flush=True)

    if os.path.exists(val_cache):
        print(f"Önbellekten Yükleniyor: {val_cache}...", flush=True)
        val_ds = torch.load(val_cache, weights_only=False)
    else:
        print(f"Doğrulama Kümesi Tokenize Ediliyor (block_size={block_size})...", flush=True)
        val_ds = FastRagPilotDataset(val_records, tokenizer, vocab, model, block_size=block_size)
        torch.save(val_ds, val_cache)
        print(f"Val Dataset Önbelleğe Alındı: {val_cache}", flush=True)

    batch_size = 8
    grad_accum_steps = 2  # Effective batch = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    epochs = 8
    total_opt_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = 30
    peak_lr = 5e-5  # Koruyucu ince ayar oranı (morphology unlearning önleme)
    min_lr = 5e-6

    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    print(f"\nEğitim Parametreleri:")
    print(f"  Epoch Sayısı      : {epochs}")
    print(f"  Batch Boyutu      : {batch_size} (Grad Accum: {grad_accum_steps} -> Efektif Batch: {batch_size * grad_accum_steps})")
    print(f"  Toplam Opt Adımı  : {total_opt_steps}")
    print(f"  Öğrenme Oranı     : Peak {peak_lr} -> Min {min_lr} (Warmup: {warmup_steps} adım)")
    print("-" * 75)

    # Başlangıç Val Loss
    init_val_loss, init_val_ppl = evaluate_val_loss(model, val_loader)
    print(f"Eğitim Öncesi Başlangıç Val Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}", flush=True)

    global_step = 0
    checkpoint_history = []
    best_val_loss = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        ep_t0 = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0

        for b_idx, (x, y, sign_mask) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sign_mask = sign_mask.to(DEVICE)

            logits, loss = model(x, y, sign_mask)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            accum_loss += loss.item()

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(train_loader):
                global_step += 1
                cur_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                step_loss = accum_loss / grad_accum_steps
                epoch_losses.append(step_loss)
                accum_loss = 0.0

                if global_step % 10 == 0 or global_step == 1:
                    recent = float(np.mean(epoch_losses[-10:])) if epoch_losses else step_loss
                    print(f"  Epoch {ep}/{epochs} | Adım {global_step:3d}/{total_opt_steps} | Train Loss: {recent:.4f} | LR: {cur_lr:.6f}", flush=True)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()

        ep_dt = time.time() - ep_t0
        ep_train_loss = float(np.mean(epoch_losses))
        ep_val_loss, ep_val_ppl = evaluate_val_loss(model, val_loader)

        ep_ckpt_path = os.path.join(PILOT_DIR, f"kristal_rag_pilot_ep{ep}.pt")
        torch.save(model.state_dict(), ep_ckpt_path)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            best_flag = "(*) EN İYİ CHECKPOINT"
        else:
            best_flag = ""

        checkpoint_history.append({
            "epoch": ep,
            "train_loss": ep_train_loss,
            "val_loss": ep_val_loss,
            "val_ppl": ep_val_ppl,
            "path": ep_ckpt_path,
            "duration": ep_dt
        })

        print(f"\n>>> [Epoch {ep}/{epochs} Sonu]: Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} (PPL: {ep_val_ppl:.2f}) | {best_flag} | Süre: {ep_dt:.1f}s")

    # Değerlendirme ve Ön-Kayıtlı Eşik Denetimi
    print("\n" + "=" * 75)
    print(" FAZ B2 ÖN-KAYITLI EĞİTİM BAŞARI KRİTERİ DENETİMİ")
    print("=" * 75)
    print(f"Başlangıç Val Loss : {init_val_loss:.4f}")
    print(f"Final En İyi Val Loss: {best_val_loss:.4f}")
    loss_ratio = best_val_loss / init_val_loss
    loss_drop_pct = (1.0 - loss_ratio) * 100.0
    print(f"Val Loss Değişimi   : %{loss_drop_pct:+.2f} (Oran: {loss_ratio:.4f})")

    if loss_ratio <= 0.90:
        fit_diagnosis = "EĞİTİM BAŞARILI (Val Loss >= %10 azaldı -> Model RAG-sentez hedefine uyum sağlıyor)"
        status_category = "TRAIN_PASS"
    else:
        fit_diagnosis = "YETERSİZ UYUM / FAIL-B RİSKİ (Val Loss %10'dan az düştü -> Temsil yetersizliği veya aşırı regülarizasyon)"
        status_category = "TRAIN_STAGNANT"

    print(f"Teşhis              : {fit_diagnosis}")
    print(f"En İyi Ağırlık      : {BEST_CKPT_PATH}")

    history_summary = {
        "init_val_loss": init_val_loss,
        "best_val_loss": best_val_loss,
        "loss_ratio": loss_ratio,
        "loss_drop_pct": loss_drop_pct,
        "fit_diagnosis": fit_diagnosis,
        "status_category": status_category,
        "best_checkpoint": BEST_CKPT_PATH,
        "epochs": checkpoint_history
    }

    history_file = os.path.join(PILOT_DIR, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_summary, f, indent=2, ensure_ascii=False)
    print(f"Eğitim geçmişi kaydedildi: {history_file}\n")


if __name__ == "__main__":
    main()
