#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCQ CONDITIONING TEST (Çoktan Seçmeli Koşullanma Testi)
Evaluates whether KristalLM is conditioned on the INPUT or predicting unconditionally.
For each (instruction, input, output) test sample:
Evaluates log-likelihood of True Output vs K-1 Random Distractors (K=10).
Random chance = 10% (1/K).
"""

import os
import sys
import json
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

from src.llm.prompt_contract import render_prompt, resize_state_dict

def load_eval_model(checkpoint_path: str, vocab: Vocabulary) -> KristalLM:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Değerlendirme checkpoint'i bulunamadı: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Checkpoint dosyası mevcut fakat yüklenemedi ({checkpoint_path}): {e}") from e

    from scripts.train_step_b1_5_rigorous import compute_sha256
    sha256_val = compute_sha256(checkpoint_path)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(checkpoint_path)} sha256={sha256_val} anahtar={len(state)}")

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    new_sd = resize_state_dict(model, state)
    model.load_state_dict(new_sd)
    model.to(DEVICE)
    model.eval()
    return model

def compute_completion_logp(model: KristalLM, prompt_ids: list, completion_ids: list) -> float:
    """Computes average log-probability of completion_ids conditioned on prompt_ids."""
    full_seq = prompt_ids + completion_ids
    if len(full_seq) > 256:
        # Keep prompt tail + completion
        full_seq = full_seq[-256:]
        prompt_len = max(1, len(full_seq) - len(completion_ids))
    else:
        prompt_len = len(prompt_ids)
        
    x = torch.tensor([full_seq], dtype=torch.long, device=DEVICE)
    sign_mask = model.embedding.compute_sign_mask(x).to(DEVICE)
    
    with torch.no_grad():
        res = model(x, sign_mask=sign_mask)
        logits = res[0] if isinstance(res, (tuple, list)) else res
        
        # Log-probs for target tokens (from index prompt_len-1 to end-1)
        # logits[:, i, :] predicts token at full_seq[i+1]
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_indices = full_seq[prompt_len:]
        if not target_indices:
            return -999.0
            
        token_logps = []
        for i, target_id in enumerate(target_indices):
            pred_idx = prompt_len - 1 + i
            if pred_idx < log_probs.size(1):
                lp = log_probs[0, pred_idx, target_id].item()
                token_logps.append(lp)
                
    return float(np.mean(token_logps)) if token_logps else -999.0

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculates Wilson 95% score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return float(max(0.0, center - margin) * 100), float(min(1.0, center + margin) * 100)

def run_mcq_test(target_n: int = 300):
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    tokenizer = KristalTokenizer(compiler, vocab)
    
    print("Modeller Yükleniyor...")
    base_model = load_eval_model("data/kristal_model.pt", vocab)
    carpenter_model = load_eval_model("data/kristal_carpenter_model.pt", vocab)
    
    eval_pool = []
    
    # 1. High school foundation
    hs_path = "data/pedagogy/high_school_foundation_dataset.jsonl"
    if os.path.exists(hs_path):
        with open(hs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    q = d.get("input", "").strip()
                    a = d.get("output", "").strip()
                    inst = d.get("instruction", "").strip()
                    if q and a and len(a) > 10:
                        eval_pool.append((inst, q, a))
                        
    # 2. History (real history questions)
    hist_path = "data/pedagogy/turk_tarihi_sft.jsonl"
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    inst = d.get("instruction", "").strip()
                    a = d.get("output", "").strip()
                    if inst and a and len(a) > 10 and not any(m in inst.lower() for m in ["tarih -", "page", "dizin"]):
                        eval_pool.append(("Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.", inst, a))
                        if len(eval_pool) >= 600:
                            break
                            
    random.seed(42)
    random.shuffle(eval_pool)
    
    # Deduplicate answer pool
    unique_answers = list(dict.fromkeys([a for _, _, a in eval_pool]))
    
    actual_n = min(target_n, len(eval_pool))
    test_samples = eval_pool[:actual_n]
    
    print(f"Toplam Test Örneği: {len(test_samples)} (Tekil Yanıt Havuzu: {len(unique_answers)})")
    print(f"Uzunluk Normalizasyonu: Token başına ortalama log-olasılık (sum(log_p) / n_tokens)")
    
    K = 10
    
    for model_name, model in [("Base Model (kristal_model.pt)", base_model), ("Carpenter Model (kristal_carpenter_model.pt)", carpenter_model)]:
        correct_count = 0
        rank_positions = []
        
        print(f"\n======================================================================")
        print(f"MCQ TESTİ: {model_name} (N={actual_n}, K={K}, Null Şans=%{100/K:.1f})")
        print(f"======================================================================")
        
        for idx, (inst, q, true_ans) in enumerate(test_samples):
            # Select K-1 deduped distractors
            eligible_distractors = [a for a in unique_answers if a.strip() != true_ans.strip() and abs(len(a) - len(true_ans)) < 300]
            if len(eligible_distractors) < K - 1:
                eligible_distractors = [a for a in unique_answers if a.strip() != true_ans.strip()]
            distractors = random.sample(eligible_distractors, K - 1)
            candidates = [true_ans] + distractors
            
            # Format prompt via canonical contract
            prompt_str = render_prompt(inst, q)
            prompt_ids = tokenizer.encode(prompt_str)
            eos_id = vocab.stoi.get("<EOS>", -1)
            if prompt_ids and prompt_ids[-1] == eos_id:
                prompt_ids = prompt_ids[:-1]
                
            scores = []
            for cand in candidates:
                cand_ids = tokenizer.encode(cand)
                # Strip BOS/EOS
                bos_id = vocab.stoi.get("<BOS>", -1)
                if cand_ids and cand_ids[0] == bos_id: cand_ids = cand_ids[1:]
                if cand_ids and cand_ids[-1] == eos_id: cand_ids = cand_ids[:-1]
                
                # Length-normalized average log-prob
                lp = compute_completion_logp(model, prompt_ids, cand_ids)
                scores.append(lp)
                
            # True candidate is at index 0
            sorted_indices = np.argsort(scores)[::-1]
            true_rank = int(np.where(sorted_indices == 0)[0][0]) + 1
            rank_positions.append(true_rank)
            
            if true_rank == 1:
                correct_count += 1
                
            if (idx + 1) % 50 == 0 or (idx + 1) == actual_n:
                curr_acc = (correct_count / (idx + 1)) * 100
                ci_low, ci_high = wilson_ci(correct_count, idx + 1)
                print(f"  Örnek {idx+1:3d}/{actual_n} | Doğruluk: %{curr_acc:5.1f} [95% GA: %{ci_low:.1f}, %{ci_high:.1f}] | Ort. Sıra: {np.mean(rank_positions):.2f}")
                
        final_acc = (correct_count / len(test_samples)) * 100
        mean_rank = float(np.mean(rank_positions))
        final_ci_low, final_ci_high = wilson_ci(correct_count, len(test_samples))
        
        print(f"\n>> {model_name} SONUCU:")
        print(f"   * MCQ Doğruluğu (Top-1): {correct_count}/{len(test_samples)} (%{final_acc:.1f})")
        print(f"   * Wilson 95% Güven Aralığı: [%{final_ci_low:.1f}, %{final_ci_high:.1f}]")
        print(f"   * Ortalama Sıralama:        {mean_rank:.2f} / {K}")
        print(f"   * Teorik Rastgele Şans:     %{100/K:.1f} (Beklenen Ortalama Sıra: {(K+1)/2:.1f})")
        
        if final_ci_low <= (100 / K) <= final_ci_high:
            print(f"   * İSTATİSTİKSEL KARAR: Model doğruluğu 95% GA içinde rastgele şans eşiğini (%{100/K:.1f}) kapsıyor (İstatistiki olarak şanstan farksız)!")
        elif final_acc > (100 / K):
            print(f"   * İSTATİSTİKSEL KARAR: Model rastgele şansın üzerinde istatistiksel olarak anlamlı koşullanma gösteriyor!")
        else:
            print(f"   * İSTATİSTİKSEL KARAR: Model rastgele şansın altında kalıyor (Şablon/ceza kilitlenmesi)!")

if __name__ == "__main__":
    n = 300
    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    run_mcq_test(target_n=n)
