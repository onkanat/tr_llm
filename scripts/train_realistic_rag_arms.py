#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: FAZ B2 GERÇEKÇİ RAG ÇOK KOLLU MODEL EĞİTİMİ (KOL A / B / C)
=============================================================================
Bu betik; 93M parametreli KristalLM modelini 3 ayrıştırıcı kol üzerinde eğitir:

- Kol A: Standart Morfemik Tokenizer (<PROPER_NOUN>) + 3.000 Doğal Veri
- Kol B: İki Katmanlı Hibrit Tokenizer (<ENT>, <CAP>, <ALL_CAPS>) + 3.000 Doğal Veri
- Kol C: İki Katmanlı Hibrit Tokenizer + 3.000 Örnek (%30 Counterfactual Prior-Kırıcı)

Eğitim Standartları:
- block_size = 512
- Causal Prompt Masking (-100) ile sadece <OUTPUT> tokenleri üzerinde gradyan
- Cosine LR Schedule (5e-5 -> 5e-6), warmup 50 adım
- Her epoch başı/sonu val.jsonl üzerinde validation loss ve perplexity takibi
- Minimum Val Loss veren checkpoint'i kaydetme
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
import random
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import render_prompt
from scripts.train_step_demo import KristalLM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REALISTIC_DIR = os.path.join(DATA_DIR, "realistic_rag")

BASE_VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
EXP_VOCAB_PATH = os.path.join(DATA_DIR, "vocab_entity.json")
BASE_CKPT_PATH = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
EXP_CKPT_PATH = os.path.join(DATA_DIR, "kristal_b1_5_entity_ready.pt")

VAL_PATH = os.path.join(REALISTIC_DIR, "val.jsonl")


class RealisticRagDataset(Dataset):
    """Precomputes causal prompt-masked tensors for RAG fine-tuning."""
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

            # Mask prompt tokens with -100 so loss is computed strictly on response
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_val_loss(model: KristalLM, val_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    with torch.no_grad():
        for x, y, sm in val_loader:
            x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
            logits, _ = model(x, sign_mask=sm)
            active_mask = (y != -100)
            n_active = active_mask.sum().item()
            if n_active > 0:
                B, T, V = logits.shape
                loss = loss_fn(logits.view(-1, V), y.view(-1))
                total_loss += loss.item()
                total_tokens += n_active

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def train_single_arm(
    arm_name: str,
    train_path: str,
    val_path: str,
    vocab_path: str,
    ckpt_path: str,
    literal_entity: bool,
    epochs: int = 5,
    lr: float = 5e-5,
    min_lr: float = 5e-6,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    block_size: int = 256
) -> Dict[str, Any]:
    print(f"\n{'='*70}", flush=True)
    print(f"   EĞİTİM BAŞLIYOR: {arm_name} (literal_entity_mode={literal_entity})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Cihaz: {DEVICE} | block_size: {block_size} | Efektif Batch: {batch_size * grad_accum_steps} | Epochs: {epochs}", flush=True)

    # 1. Derleyici ve Tokenizer
    lex = LexiconManager()
    roots_path = os.path.join(DATA_DIR, "lexicon", "roots.tsv")
    if os.path.exists(roots_path):
        lex.load_from_tsv(roots_path)
    else:
        raise FileNotFoundError(f"Leksikon kök dosyası bulunamadı: {roots_path}")
    graph = build_default_graph()
    compiler = CrystalCompiler(lex, graph)

    vocab = Vocabulary()
    vocab.load(vocab_path, freeze=True)
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=literal_entity)

    print(f"Vocab Boyutu: {len(vocab.stoi):,} token", flush=True)

    # 2. Model Yükleme
    model = KristalLM(
        vocab_size=len(vocab.stoi),
        n_embd=768,
        vocab=vocab,
        block_size=4096,
        n_layer=6,
        n_head=6
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Başlangıç ağırlık checkpoint'i bulunamadı: {ckpt_path}")
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Başlangıç checkpoint'i yüklenirken hata oluştu ({ckpt_path}): {e}") from e

    sha256_val = compute_sha256(ckpt_path)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(ckpt_path)} sha256={sha256_val} anahtar={len(sd)}", flush=True)
    load_res = model.load_state_dict(sd, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(f"[SOYAGACI_UYARI] strict=False ile yüklendi: eksik={len(load_res.missing_keys)}, fazla={len(load_res.unexpected_keys)}", flush=True)
        if load_res.missing_keys:
            print(f"  * Eksik anahtarlar: {load_res.missing_keys[:5]}{'...' if len(load_res.missing_keys) > 5 else ''}", flush=True)
        if load_res.unexpected_keys:
            print(f"  * Fazla anahtarlar: {load_res.unexpected_keys[:5]}{'...' if len(load_res.unexpected_keys) > 5 else ''}", flush=True)
    else:
        print("[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).", flush=True)
    model.to(DEVICE)
    print(f"Başlangıç Ağırlıkları Yüklendi: '{ckpt_path}'", flush=True)

    # 3. Veri Setlerini Yükle ve Tokenize Et
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(l) for l in f]
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(l) for l in f]

    print(f"Eğitim Örnek Sayısı: {len(train_records):,} | Doğrulama Örnek Sayısı: {len(val_records):,}", flush=True)
    print(f"[{arm_name}] Veri kümesi tensörleri hazırlanıyor (block_size={block_size})...", flush=True)

    t0_ds = time.time()
    train_ds = RealisticRagDataset(train_records, tokenizer, vocab, model, block_size=block_size)
    val_ds = RealisticRagDataset(val_records, tokenizer, vocab, model, block_size=block_size)
    print(f"[{arm_name}] Tensörler hazırlandı ({time.time()-t0_ds:.1f} saniye).", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Başlangıç Val Kaybı
    print(f"[{arm_name}] Başlangıç Val Kaybı hesaplanıyor...", flush=True)
    init_val_loss, init_val_ppl = evaluate_val_loss(model, val_loader)
    print(f"Başlangıç (Pre-train) Val Loss: {init_val_loss:.4f} (PPL: {init_val_ppl:.2f})", flush=True)

    # 5. Optimizer ve LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = 50

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_val_loss = init_val_loss
    best_ckpt_path = os.path.join(DATA_DIR, f"kristal_rag_{arm_name.lower().replace(' ', '_')}_best.pt")

    opt_step = 0
    cur_lr = lr
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        train_tokens_accum = 0
        optimizer.zero_grad()

        for batch_idx, (x, y, sm) in enumerate(train_loader):
            x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
            logits, _ = model(x, sign_mask=sm)

            B, T, V = logits.shape
            loss = loss_fn(logits.view(-1, V), y.view(-1))
            loss = loss / grad_accum_steps
            loss.backward()

            active_tokens = (y != -100).sum().item()
            train_loss_accum += (loss.item() * grad_accum_steps * active_tokens)
            train_tokens_accum += active_tokens

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                cur_lr = get_lr(opt_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1
                if DEVICE.type == "mps" and opt_step % 25 == 0:
                    torch.mps.empty_cache()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                cur_loss = train_loss_accum / max(1, train_tokens_accum)
                print(f"  [Epoch {epoch}/{epochs}] Adım {batch_idx+1}/{len(train_loader)} | Train Loss: {cur_loss:.4f} | LR: {cur_lr:.2e}", flush=True)

        epoch_train_loss = train_loss_accum / max(1, train_tokens_accum)
        val_loss, val_ppl = evaluate_val_loss(model, val_loader)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

        ep_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | {'[BEST SAVED]' if is_best else ''}", flush=True)

    elapsed = time.time() - start_time
    loss_reduction = (best_val_loss / init_val_loss)

    print(f"\n{arm_name} Eğitimi Tamamlandı ({elapsed/60:.1f} dakika).", flush=True)
    print(f"  * Başlangıç Val Loss: {init_val_loss:.4f} -> Final Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"  * Kayıp Oranı (Val_final / Val_init): {loss_reduction:.4f} (Hedef <= 0.90)", flush=True)
    print(f"  * En İyi Checkpoint: '{best_ckpt_path}'", flush=True)

    # Bellek temizliği (MPS bellek sızıntısını önleme)
    del model
    del optimizer
    del train_ds
    del val_ds
    del train_loader
    del val_loader
    import gc
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return {
        "arm": arm_name,
        "init_val_loss": init_val_loss,
        "final_val_loss": best_val_loss,
        "loss_reduction": loss_reduction,
        "best_ckpt": best_ckpt_path,
        "epochs": epochs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="all", choices=["A", "B", "C", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    train_nat_path = os.path.join(REALISTIC_DIR, "train_natural_3000.jsonl")
    train_cf_path = os.path.join(REALISTIC_DIR, "train_cf_3000.jsonl")

    results = {}

    # KOL A
    if args.arm in ("A", "all"):
        res_a = train_single_arm(
            arm_name="arm_A",
            train_path=train_nat_path,
            val_path=VAL_PATH,
            vocab_path=BASE_VOCAB_PATH,
            ckpt_path=BASE_CKPT_PATH,
            literal_entity=False,
            epochs=args.epochs,
            block_size=args.block_size
        )
        results["arm_A"] = res_a

    # KOL B
    if args.arm in ("B", "all"):
        res_b = train_single_arm(
            arm_name="arm_B",
            train_path=train_nat_path,
            val_path=VAL_PATH,
            vocab_path=EXP_VOCAB_PATH,
            ckpt_path=EXP_CKPT_PATH,
            literal_entity=True,
            epochs=args.epochs,
            block_size=args.block_size
        )
        results["arm_B"] = res_b

    # KOL C
    if args.arm in ("C", "all"):
        res_c = train_single_arm(
            arm_name="arm_C",
            train_path=train_cf_path,
            val_path=VAL_PATH,
            vocab_path=EXP_VOCAB_PATH,
            ckpt_path=EXP_CKPT_PATH,
            literal_entity=True,
            epochs=args.epochs,
            block_size=args.block_size
        )
        results["arm_C"] = res_c

    summary_file = os.path.join(REALISTIC_DIR, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nÖzet rapor kaydedildi: '{summary_file}'")


if __name__ == "__main__":
    main()
