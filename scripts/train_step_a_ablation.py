#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADIM A ABLASYON EĞİTİMİ: LOADER & PROMPT MASKELEME
===================================================
Mevcut (arındırılmamış) veriler üzerinde yalnızca:
1. Örnek-düzeyinde hizalı padding (<PAD>).
2. İstem pozisyonlarına -100 hedef maskesi.
3. Şablon imzası (template skeleton hash) ile sızıntısız train/val ayrımı.
4. 3 Epoch bütçesi, LR warmup + cosine decay.
5. Her epoch sonunda N=300 MCQ (Wilson 95% GA) ve Morfoloji Regresyon testi (E7).
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
from scripts.evaluate_mcq_conditioning import compute_completion_logp, wilson_ci
from src.llm.prompt_contract import render_prompt, resize_state_dict

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


def evaluate_morphology_regression(model: KristalLM, tokenizer: KristalTokenizer, vocab: Vocabulary) -> float:
    test_path = "tests/morphology_regression_100.json"
    if not os.path.exists(test_path):
        return 0.0
    with open(test_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    correct = 0
    model.eval()
    eos_id = vocab.stoi.get("<EOS>", 3)
    
    for item in samples:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        true_out = item.get("output", "").strip()
        
        # Format prompt via canonical contract
        prompt_str = render_prompt(inst, inp)
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]
            
        true_ids = tokenizer.encode(true_out)
        if true_ids and true_ids[0] == 2: true_ids = true_ids[1:]
        if true_ids and true_ids[-1] == eos_id: true_ids = true_ids[:-1]
        
        # Model generate greedily up to len(true_ids) + 5
        curr = list(prompt_ids)
        with torch.no_grad():
            for _ in range(min(15, len(true_ids) + 3)):
                x = torch.tensor([curr[-256:]], dtype=torch.long, device=DEVICE)
                res = model(x)
                logits = res[0] if isinstance(res, (tuple, list)) else res
                nxt = torch.argmax(logits[0, -1, :]).item()
                if nxt == eos_id or nxt == vocab.stoi.get("</OUTPUT>", 9):
                    break
                curr.append(nxt)
                
        pred_out_ids = curr[len(prompt_ids):]
        pred_str = tokenizer.decode(pred_out_ids).replace("<BOS>", "").replace("<EOS>", "").strip()
        true_str = tokenizer.decode(true_ids).replace("<BOS>", "").replace("<EOS>", "").strip()
        
        if pred_str.strip() == true_str.strip():
            correct += 1
            
    model.train()
    return (correct / len(samples)) * 100


def run_quick_mcq(model: KristalLM, tokenizer: KristalTokenizer, vocab: Vocabulary, test_samples: list, unique_answers: list, K: int = 10):
    model.eval()
    correct = 0
    ranks = []
    eos_id = vocab.stoi.get("<EOS>", 3)
    bos_id = vocab.stoi.get("<BOS>", 2)
    
    for inst, q, true_ans in test_samples:
        eligible = [a for a in unique_answers if a.strip() != true_ans.strip() and abs(len(a) - len(true_ans)) < 300]
        if len(eligible) < K - 1:
            eligible = [a for a in unique_answers if a.strip() != true_ans.strip()]
        distractors = random.sample(eligible, K - 1)
        candidates = [true_ans] + distractors
        
        prompt_str = render_prompt(inst, q)
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]
            
        scores = []
        for cand in candidates:
            cand_ids = tokenizer.encode(cand)
            if cand_ids and cand_ids[0] == bos_id: cand_ids = cand_ids[1:]
            if cand_ids and cand_ids[-1] == eos_id: cand_ids = cand_ids[:-1]
            lp = compute_completion_logp(model, prompt_ids, cand_ids)
            scores.append(lp)
            
        sorted_indices = np.argsort(scores)[::-1]
        true_rank = int(np.where(sorted_indices == 0)[0][0]) + 1
        ranks.append(true_rank)
        if true_rank == 1:
            correct += 1
            
    model.train()
    acc = (correct / len(test_samples)) * 100
    ci_low, ci_high = wilson_ci(correct, len(test_samples))
    mean_r = float(np.mean(ranks))
    return acc, ci_low, ci_high, mean_r


def main():
    print("=" * 70)
    print(" ADIM A ABLASYON: LOADER & PROMPT MASKELEME DENEYİ")
    print("=" * 70)
    print(f"Hesaplama Cihazı: {DEVICE}")
    
    # 1. Vocab & Compiler
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    # 2. Load Existing Raw Records
    raw_paths = [
        'data/pedagogy/high_school_foundation_dataset.jsonl',
        'data/pedagogy/middle_school_chat.jsonl',
        'data/pedagogy/turk_tarihi_sft.jsonl',
        'data/pedagogy/literature_poetry_dataset.jsonl',
        'data/pedagogy/parenting_dataset.jsonl',
        'data/pedagogy/chat_conversations.jsonl',
        'data/pedagogy/carpenter_specialization_dataset.jsonl',
        'data/pedagogy/rag_dataset.jsonl',
    ]
    
    print("\nMevcut veri kümeleri yükleniyor...")
    all_records = load_jsonl_records(raw_paths)
    print(f"Toplam ham kayıt sayısı: {len(all_records):,}")
    
    # Template-aware split (E2)
    train_records, val_records = template_aware_train_val_split(all_records, val_ratio=0.1, seed=42)
    print(f"Şablon-Farkında Ayrım: {len(train_records):,} Train | {len(val_records):,} Val")
    
    # Subsample for Ablation Epochs: Keep a high-leverage 12,000-sample training set
    # balancing across domains so 3 epochs run in ~20 minutes.
    rng = random.Random(42)
    rng.shuffle(train_records)
    ablation_train = train_records[:12000]
    rng.shuffle(val_records)
    ablation_val = val_records[:1200]
    print(f"Ablasyon Alt Kümesi: {len(ablation_train):,} Train | {len(ablation_val):,} Val")
    
    block_size = 256
    print("\nSampleAlignedDataset oluşturuluyor (Prompt maskeleme: -100)...", flush=True)
    train_ds = SampleAlignedDataset(ablation_train, tokenizer, vocab, block_size=block_size)
    val_ds = SampleAlignedDataset(ablation_val, tokenizer, vocab, block_size=block_size)
    print(f"Dataset hazır: {len(train_ds)} train örneği, {len(val_ds)} val örneği.")
    
    micro_batch_size = 8
    grad_accum_steps = 4
    train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False)
    
    # Prepare MCQ Test Samples (N=300)
    print("\nMCQ N=300 Test Havuzu Hazırlanıyor...")
    eval_pool = []
    hs_path = "data/pedagogy/high_school_foundation_dataset.jsonl"
    if os.path.exists(hs_path):
        with open(hs_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                q, a, inst = d.get("input", "").strip(), d.get("output", "").strip(), d.get("instruction", "").strip()
                if q and a and len(a) > 10: eval_pool.append((inst, q, a))
    hist_path = "data/pedagogy/turk_tarihi_sft.jsonl"
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                inst, a = d.get("instruction", "").strip(), d.get("output", "").strip()
                if inst and a and len(a) > 10 and not any(m in inst.lower() for m in ["tarih -", "page", "dizin"]):
                    eval_pool.append(("Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.", inst, a))
                    if len(eval_pool) >= 600: break
    rng.shuffle(eval_pool)
    unique_answers = list(dict.fromkeys([a for _, _, a in eval_pool]))
    mcq_test_samples = eval_pool[:300]
    print(f"MCQ Örnek Sayısı: {len(mcq_test_samples)} (Tekil Cevap: {len(unique_answers)})")
    
    # Load Base Model
    print("\nBase Model Yükleniyor...")
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    base_ckpt = "data/kristal_model.pt"
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Birincil baz model checkpoint bulunamadı (sessiz sıfırdan başlama engellendi): {base_ckpt}")
    try:
        sd = torch.load(base_ckpt, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Birincil baz model checkpoint yüklenirken hata oluştu ({base_ckpt}): {e}") from e

    sha256_val = compute_sha256(base_ckpt)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(base_ckpt)} sha256={sha256_val} anahtar={len(sd)}")
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
    print(f"{base_ckpt} başarıyla yüklendi.")
    model.to(DEVICE)
    
    # Baseline Measurements
    print("\n[0. Adım - Baz Hat Ölçümü]")
    base_val_loss = evaluate_val_loss(model, val_loader)
    base_morph_acc = evaluate_morphology_regression(model, tokenizer, vocab)
    print(f"  * Başlangıç Val Loss (Maskeli): {base_val_loss:.4f}")
    print(f"  * Başlangıç Morfoloji Doğruluğu (E7): %{base_morph_acc:.1f}")
    print("  * Başlangıç MCQ (N=300): %9.3 [95% GA: %6.5, %13.2] (Önceki ölçümden biliniyor)")
    
    # E1: Budget & Schedule
    epochs = 3
    total_optimizer_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = 100
    peak_lr = 3e-4
    min_lr = 1e-5
    
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    print(f"\nEğitim Başlıyor: {epochs} Epoch, Toplam {total_optimizer_steps} Optimizasyon Adımı (Micro-Batch: {micro_batch_size}, Accum: {grad_accum_steps}, Efektif Batch: {micro_batch_size * grad_accum_steps})")
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
                    print(f"  Epoch {ep}/{epochs} | Adım {global_step:4d}/{total_optimizer_steps} | Train Loss: {np.mean(epoch_losses[-50:]):.4f} | LR: {cur_lr:.6f}", flush=True)
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                
        ep_dt = time.time() - ep_t0
        ep_train_loss = float(np.mean(epoch_losses))
        ep_val_loss = evaluate_val_loss(model, val_loader)
        
        print(f"\n>>> [Epoch {ep} Sonu]: Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} | Süre: {ep_dt:.1f}s")
        
        # Intermediate MCQ and Morphology check
        print("    MCQ Testi Koşuluyor (N=300)...", flush=True)
        mcq_acc, mcq_ci_low, mcq_ci_high, mcq_rank = run_quick_mcq(model, tokenizer, vocab, mcq_test_samples, unique_answers)
        morph_acc = evaluate_morphology_regression(model, tokenizer, vocab)
        
        print(f"    >> Epoch {ep} MCQ Top-1 Doğruluğu: %{mcq_acc:.1f} [95% GA: %{mcq_ci_low:.1f}, %{mcq_ci_high:.1f}] | Ort. Sıra: {mcq_rank:.2f}")
        print(f"    >> Epoch {ep} Morfoloji Doğruluğu (E7): %{morph_acc:.1f}")
        print("-" * 70, flush=True)
        
        # Save checkpoint
        ckpt_path = f"data/kristal_model_step_a_ep{ep}.pt"
        torch.save(model.state_dict(), ckpt_path)
        
    total_dt = time.time() - t0
    print("=" * 70)
    print(f"ADIM A ABLASYONU TAMAMLANDI ({total_dt:.1f} saniye)!")
    print("=" * 70)
    
    # Save final model as candidate
    torch.save(model.state_dict(), "data/kristal_model_step_a_final.pt")
    print("Model 'data/kristal_model_step_a_final.pt' olarak kaydedildi.")

if __name__ == '__main__':
    main()
