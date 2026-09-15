#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: ADIM B1.5 TİTİZ MODEL EĞİTİMİ VE VAL SEÇİMİ
=====================================================================
1. `data/b1_5_splits/train.jsonl` (13,009 kayıt) üzerinde eğitim koşar.
2. Causal Prompt Masking (-100) ve Örnek Düzeyinde Hizalama.
3. Test kümesine (`test.jsonl`) EĞİTİMDE KESİNLİKLE DOKUNULMAZ (Sıfır Sızıntı).
4. Her epoch sonunda `val.jsonl` (1,622 kayıt) üzerinde Val Loss ve Val Perplexity ölçülür.
5. Minimum Val Loss'a sahip checkpoint otomatik seçilir -> `data/kristal_b1_5_best.pt`.
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
from src.llm.dataset import load_jsonl_records
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import compute_completion_logp, wilson_ci, resize_state_dict
from src.llm.prompt_contract import render_prompt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")

TRAIN_PATH = os.path.join(SPLITS_DIR, "train.jsonl")
VAL_PATH = os.path.join(SPLITS_DIR, "val.jsonl")


class FastSampleAlignedDataset(Dataset):
    """Precomputes x, y, and sign_mask tensors for high training throughput."""
    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: KristalTokenizer,
        vocab: Vocabulary,
        model: KristalLM,
        block_size: int = 128
    ):
        self.block_size = block_size
        self.pad_id = vocab.stoi.get("<PAD>", 1)
        self.bos_id = vocab.stoi.get("<BOS>", 2)
        self.eos_id = vocab.stoi.get("<EOS>", 3)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        for rec in records:
            inst = rec.get("instruction", "").strip()
            inp = rec.get("input", "").strip()
            out = rec.get("output", "").strip()
            
            prompt_str = render_prompt(inst, inp)
            
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
            
            y_masked = []
            for i, target_token in enumerate(y_ids):
                if i < P - 1:
                    y_masked.append(-100)
                else:
                    y_masked.append(target_token)
                    
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


def evaluate_val_loss(model: KristalLM, val_loader: DataLoader, max_batches: int = 50) -> Tuple[float, float]:
    """Evaluates validation loss and perplexity."""
    model.eval()
    losses = []
    with torch.no_grad():
        for b_idx, (x, y, sign_mask) in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sign_mask = sign_mask.to(DEVICE)
            _, loss = model(x, y, sign_mask)
            losses.append(loss.item())
            
    model.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl


def main():
    print("===================================================================")
    print("KRİSTAL-VEKTÖREL: ADIM B1.5 TİTİZ MODEL EĞİTİMİ (MPS / CPU)")
    print("===================================================================")
    print(f"Hesaplama Cihazı: {DEVICE}")
    
    # 1. Load Vocab, Lexicon, Tokenizer
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    print(f"Kelime Dağarcığı (Vocab): {len(vocab.stoi):,} token")
    
    # 2. Model Initialization & Weight Transfer
    model = KristalLM(
        vocab_size=len(vocab.stoi), 
        n_embd=768, 
        vocab=vocab, 
        block_size=4096, 
        n_layer=6, 
        n_head=6
    )
    
    prev_checkpoints = [
        os.path.join(DATA_DIR, "kristal_model_step_b1_final.pt"),
        os.path.join(DATA_DIR, "kristal_model_step_a_final.pt"),
        os.path.join(DATA_DIR, "kristal_model.pt")
    ]
    primary_ckpt = prev_checkpoints[0]
    loaded_ckpt = None

    for idx, ckpt in enumerate(prev_checkpoints):
        if not os.path.exists(ckpt):
            continue
        if idx > 0:
            print(f"[SOYAGACI_UYARI] BIRINCIL YOK ({primary_ckpt}), GERI DUSULDU -> {ckpt}")
        try:
            sd = torch.load(ckpt, map_location="cpu")
        except Exception as e:
            # Var olan checkpoint dosyası yüklenemiyorsa yutulamaz; bozulma gizlenemez (BULGU 2b)
            raise RuntimeError(f"Checkpoint dosyası mevcut fakat yüklenemedi ({ckpt}): {e}") from e

        sha256_val = compute_sha256(ckpt)
        print(f"[SOYAGACI] yuklenen={os.path.abspath(ckpt)} sha256={sha256_val} anahtar={len(sd)}")
        new_sd = resize_state_dict(model, sd)
        load_res = model.load_state_dict(new_sd, strict=False)
        if load_res.missing_keys or load_res.unexpected_keys:
            print(f"[SOYAGACI_UYARI] strict=False ile yüklendi: eksik={len(load_res.missing_keys)}, fazla={len(load_res.unexpected_keys)}")
            if load_res.missing_keys:
                print(f"  * Eksik anahtarlar: {load_res.missing_keys[:5]}{'...' if len(load_res.missing_keys) > 5 else ''}")
            if load_res.unexpected_keys:
                print(f"  * Fazla anahtarlar: {load_res.unexpected_keys[:5]}{'...' if len(load_res.unexpected_keys) > 5 else ''}")
        else:
            print("[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).")
        print(f"Model Ağırlıkları Aktarıldı: '{ckpt}'")
        loaded_ckpt = ckpt
        break

    if loaded_ckpt is None:
        raise FileNotFoundError(f"Hiçbir ata checkpoint bulunamadı! Denenen zincir: {prev_checkpoints} (sessiz sıfırdan başlama engellendi)")

    model.to(DEVICE)
    
    # 3. Load Datasets
    train_records = load_jsonl_records([TRAIN_PATH])
    val_records = load_jsonl_records([VAL_PATH])
    print(f"Eğitim Kümesi: {len(train_records):,} kayıt | Val Kümesi: {len(val_records):,} kayıt")
    print(f"NOT: Test kümesine eğitim süresince kesinlikle dokunulmamaktadır.")
    
    block_size = 128
    train_cache = os.path.join(SPLITS_DIR, "train_fast_ds.pt")
    val_cache = os.path.join(SPLITS_DIR, "val_fast_ds.pt")
    
    if os.path.exists(train_cache):
        print(f"Önbellekten Yükleniyor: {train_cache}...", flush=True)
        train_ds = torch.load(train_cache, weights_only=False)
    else:
        print("Eğitim Kümesi Hızlı Tokenize Ediliyor (Train block_size=128)...", flush=True)
        train_ds = FastSampleAlignedDataset(train_records, tokenizer, vocab, model, block_size=block_size)
        torch.save(train_ds, train_cache)
        print(f"Train Tensorları Kaydedildi: {train_cache}", flush=True)
        
    if os.path.exists(val_cache):
        print(f"Önbellekten Yükleniyor: {val_cache}...", flush=True)
        val_ds = torch.load(val_cache, weights_only=False)
    else:
        print("Doğrulama Kümesi Tokenize Ediliyor (Val)...", flush=True)
        val_ds = FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=block_size)
        torch.save(val_ds, val_cache)
        print(f"Val Tensorları Kaydedildi: {val_cache}", flush=True)
        
    batch_size = 16
    grad_accum_steps = 2  # Effective batch = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    epochs = 3
    total_opt_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = 50
    peak_lr = 3e-4
    min_lr = 1e-5
    
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    print(f"\nEğitim Parametreleri: {epochs} Epoch, {total_opt_steps} Adım (Batch: {batch_size}, Accum: {grad_accum_steps})")
    print("-" * 75)
    
    # Baseline validation loss
    init_loss, init_ppl = evaluate_val_loss(model, val_loader)
    print(f"Eğitim Öncesi Başlangıç Val Loss: {init_loss:.4f} | Perplexity: {init_ppl:.2f}", flush=True)
    
    global_step = 0
    checkpoint_history = []
    
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
                
                if global_step % 20 == 0 or global_step == 1:
                    recent = float(np.mean(epoch_losses[-20:])) if epoch_losses else step_loss
                    print(f"  Epoch {ep}/{epochs} | Adım {global_step:4d}/{total_opt_steps} | Train Loss: {recent:.4f} | LR: {cur_lr:.6f}", flush=True)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                        
        ep_dt = time.time() - ep_t0
        ep_train_loss = float(np.mean(epoch_losses))
        ep_val_loss, ep_val_ppl = evaluate_val_loss(model, val_loader)
        
        ckpt_path = os.path.join(DATA_DIR, f"kristal_b1_5_epoch{ep}.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        checkpoint_history.append({
            "epoch": ep,
            "train_loss": ep_train_loss,
            "val_loss": ep_val_loss,
            "val_ppl": ep_val_ppl,
            "path": ckpt_path,
            "duration": ep_dt
        })
        
        print(f"\n>>> [Epoch {ep} Sonu]: Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} (PPL: {ep_val_ppl:.2f}) | Süre: {ep_dt:.1f}s")
        print(f"    Checkpoint Kaydedildi: {ckpt_path}\n", flush=True)

    # Checkpoint Selection
    print("===================================================================")
    print("CHECKPOINT SEÇİMİ (Ön-Kayıtlı Kural: Minimum Validation Loss)")
    print("===================================================================")
    for c in checkpoint_history:
        print(f"  Epoch {c['epoch']}: Train Loss = {c['train_loss']:.4f} | Val Loss = {c['val_loss']:.4f} | PPL = {c['val_ppl']:.2f}")
        
    best_epoch = min(checkpoint_history, key=lambda c: c["val_loss"])
    best_path = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
    shutil.copyfile(best_epoch["path"], best_path)
    shutil.copyfile(best_epoch["path"], os.path.join(DATA_DIR, "kristal_model.pt"))
    
    print(f"\nEN İYİ CHECKPOINT SEÇİLDİ: Epoch {best_epoch['epoch']} (Val Loss: {best_epoch['val_loss']:.4f})")
    print(f"  -> {best_path} dosyasına kopyalandı.")
    
    summary_path = os.path.join(SPLITS_DIR, "training_history.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": checkpoint_history,
            "best_epoch": best_epoch["epoch"],
            "best_val_loss": best_epoch["val_loss"],
            "best_checkpoint": best_path
        }, f, indent=2)
        
    print(f"Eğitim Özeti Kaydedildi: {summary_path}\n")


if __name__ == "__main__":
    main()
