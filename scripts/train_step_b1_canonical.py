#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADIM B1 EĞİTİMİ: KANONİK ŞEMA VE RAKAM/NOKTALAMA HİZALAMASI
===========================================================
- data/pedagogy_canonical/*.jsonl üzerinden eğitim.
- 0-9 tekil basamak ve noktalama tokenları aktif.
- Örnek düzeyinde hizalı padding (<PAD>) ve prompt hedef maskesi (-100).
- Şablon hash'li sızıntısız train/val ayrımı.
- Her epoch sonunda Katmanlı (High School, Tarih, Carpenter) hem SEEN hem HELD-OUT MCQ ölçümü.
"""

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys
import time
import math
import json
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.dataset import SampleAlignedDataset, template_aware_train_val_split, load_jsonl_records
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import compute_completion_logp, wilson_ci, resize_state_dict
from src.llm.prompt_contract import render_prompt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def evaluate_val_loss(model: KristalLM, val_loader: DataLoader, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sign_mask = model.embedding.compute_sign_mask(x.cpu()).to(DEVICE)
            _, loss = model(x, y, sign_mask)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def evaluate_stratified_mcq(model: KristalLM, tokenizer: KristalTokenizer, vocab: Vocabulary, test_pools: Dict[str, list], K: int = 10):
    model.eval()
    results = {}
    eos_id = vocab.stoi.get("<EOS>", 3)
    bos_id = vocab.stoi.get("<BOS>", 2)
    
    for stratum_name, samples in test_pools.items():
        if not samples: continue
        all_answers = [a for _, _, a in samples]
        unique_answers = list(dict.fromkeys(all_answers))
        
        correct = 0
        ranks = []
        for inst, q, true_ans in samples:
            distractors = random.sample([a for a in unique_answers if a.strip() != true_ans.strip()], K - 1)
            candidates = [true_ans] + distractors
            
            prompt_str = render_prompt(inst, q)
            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == eos_id: prompt_ids = prompt_ids[:-1]
            
            scores = []
            for cand in candidates:
                cand_ids = tokenizer.encode(cand)
                if cand_ids and cand_ids[0] == bos_id: cand_ids = cand_ids[1:]
                if cand_ids and cand_ids[-1] == eos_id: cand_ids = cand_ids[:-1]
                lp = compute_completion_logp(model, prompt_ids, cand_ids)
                scores.append(lp)
                
            true_rank = int(np.where(np.argsort(scores)[::-1] == 0)[0][0]) + 1
            ranks.append(true_rank)
            if true_rank == 1: correct += 1
            
        acc = (correct / len(samples)) * 100
        ci_l, ci_h = wilson_ci(correct, len(samples))
        mean_r = float(np.mean(ranks))
        results[stratum_name] = {'acc': acc, 'ci_l': ci_l, 'ci_h': ci_h, 'mean_rank': mean_r, 'n': len(samples)}
        
    model.train()
    return results


def main():
    print("=" * 70)
    print(" ADIM B1: KANONİK ŞEMA VE RAKAM/NOKTALAMA HİZALANMIŞ MODEL EĞİTİMİ")
    print("=" * 70)
    print(f"Hesaplama Cihazı: {DEVICE}")
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    canonical_paths = [
        'data/pedagogy_canonical/high_school_canonical.jsonl',
        'data/pedagogy_canonical/middle_school_canonical.jsonl',
        'data/pedagogy_canonical/turk_tarihi_canonical.jsonl',
        'data/pedagogy_canonical/literature_canonical.jsonl',
        'data/pedagogy_canonical/parenting_canonical.jsonl',
        'data/pedagogy_canonical/carpenter_canonical.jsonl',
    ]
    
    all_records = load_jsonl_records(canonical_paths)
    print(f"Toplam Kanonik Kayıt: {len(all_records):,}")
    
    # Template-aware split: Train vs Held-Out
    train_records, heldout_records = template_aware_train_val_split(all_records, val_ratio=0.1, seed=42)
    print(f"Şablon-Farkında Ayrım: {len(train_records):,} Train | {len(heldout_records):,} Held-Out Val")
    
    # Training subset: 15,000 balanced records
    rng = random.Random(42)
    rng.shuffle(train_records)
    b1_train = train_records[:15000]
    rng.shuffle(heldout_records)
    b1_val = heldout_records[:1500]
    
    block_size = 256
    train_ds = SampleAlignedDataset(b1_train, tokenizer, vocab, block_size=block_size)
    val_ds = SampleAlignedDataset(b1_val, tokenizer, vocab, block_size=block_size)
    print(f"Dataset hazır: {len(train_ds)} train, {len(val_ds)} val.")
    
    micro_batch_size = 8
    grad_accum_steps = 4
    train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False)
    
    # Build Stratified Test Pools: SEEN (from train) and HELDOUT (from heldout)
    def extract_pools(records, limit=50):
        pools = {'HighSchool': [], 'History': [], 'Carpenter': []}
        for r in records:
            inst = r.get('instruction', '')
            q = r.get('input', '')
            a = r.get('output', '')
            if not (q and a and len(a) > 10): continue
            if 'ahşap' in inst.lower() or 'marangoz' in inst.lower():
                if len(pools['Carpenter']) < limit: pools['Carpenter'].append((inst, q, a))
            elif 'tarih' in inst.lower():
                if len(pools['History']) < limit: pools['History'].append((inst, q, a))
            elif 'fen' in inst.lower() or 'lise' in inst.lower():
                if len(pools['HighSchool']) < limit: pools['HighSchool'].append((inst, q, a))
        return pools
        
    seen_pools = extract_pools(b1_train, limit=50)
    heldout_pools = extract_pools(heldout_records, limit=50)
    
    # Load Model (resume from step A final)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    load_path = 'data/kristal_model_step_a_final.pt'
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Birincil checkpoint bulunamadı (sessiz sıfırdan başlama engellendi): {load_path}")
    try:
        sd = torch.load(load_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Birincil checkpoint yüklenirken hata oluştu ({load_path}): {e}") from e

    sha256_val = compute_sha256(load_path)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(load_path)} sha256={sha256_val} anahtar={len(sd)}")
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
    print(f"Model '{load_path}' başarıyla yüklendi.")
    model.to(DEVICE)
    
    epochs = 3
    total_opt_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = 100
    peak_lr = 3e-4
    min_lr = 1e-5
    
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    print(f"\nEğitim Başlıyor: {epochs} Epoch, Toplam {total_opt_steps} Adım (Micro: {micro_batch_size}, Accum: {grad_accum_steps})")
    print("-" * 70)
    
    global_step = 0
    t0 = time.time()
    
    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        ep_t0 = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for b_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sign_mask = model.embedding.compute_sign_mask(x.cpu()).to(DEVICE)
            
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
                
                epoch_losses.append(accum_loss / grad_accum_steps)
                accum_loss = 0.0
                
                if global_step % 50 == 0:
                    print(f"  Epoch {ep}/{epochs} | Adım {global_step:4d}/{total_opt_steps} | Train Loss: {np.mean(epoch_losses[-50:]):.4f} | LR: {cur_lr:.6f}", flush=True)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                        
        ep_dt = time.time() - ep_t0
        ep_train_loss = float(np.mean(epoch_losses))
        ep_val_loss = evaluate_val_loss(model, val_loader)
        
        print(f"\n>>> [Epoch {ep} Sonu]: Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} | Süre: {ep_dt:.1f}s")
        
        # Stratified MCQ: SEEN vs HELDOUT
        print("    Katmanlı MCQ Ölçülüyor (Seen & Held-Out)...", flush=True)
        res_seen = evaluate_stratified_mcq(model, tokenizer, vocab, seen_pools)
        res_held = evaluate_stratified_mcq(model, tokenizer, vocab, heldout_pools)
        
        print(f"    [SEEN KÜMESİ MCQ]:")
        for k, v in res_seen.items():
            print(f"      * {k:<12}: %{v['acc']:4.1f} [{v['ci_l']:.1f}, {v['ci_h']:.1f}] (Sıra: {v['mean_rank']:.2f})")
        print(f"    [HELD-OUT KÜMESİ MCQ]:")
        for k, v in res_held.items():
            print(f"      * {k:<12}: %{v['acc']:4.1f} [{v['ci_l']:.1f}, {v['ci_h']:.1f}] (Sıra: {v['mean_rank']:.2f})")
        print("-" * 70, flush=True)
        
        # Save checkpoint
        torch.save(model.state_dict(), f"data/kristal_model_step_b1_ep{ep}.pt")
        
    total_dt = time.time() - t0
    print("=" * 70)
    print(f"ADIM B1 TAMAMLANDI ({total_dt:.1f} saniye)!")
    print("=" * 70)
    torch.save(model.state_dict(), "data/kristal_model_step_b1_final.pt")
    print("Final model 'data/kristal_model_step_b1_final.pt' olarak kaydedildi.")

if __name__ == '__main__':
    main()
