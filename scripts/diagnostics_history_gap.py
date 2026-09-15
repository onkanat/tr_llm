#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: TÜRK TARİHİ < TABAN TANI BETİĞİ (PAKET 1)
===========================================================
Araştırma Sorusu:
Şans seviyesinin (+1.08 basamak) üzerindeki zayıf sinyal,
yüksek LM prior'a sahip jenerik tarih şablonları tarafından mı maskeleniyor,
yoksa model girdiye semantik olarak hiç koşullanmıyor mu (içerik sinyali sıfır mı)?

Ölçümler:
1. Koşullu Log-Olasılık: log P(Aday | Soru)
2. Boş Prompt Dil Modeli Öncülü (LM Prior): log P(Aday)
3. Noktasal Karşılıklı Bilgi (PMI): PMI = log P(Aday | Soru) - log P(Aday)
4. Rank Histogramı ve Top-1, Top-3, Top-5 Doğrulukları (logP vs PMI)
5. Aday Uzunluğu (Token Sayısı) ile logP Korelasyonu
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")
CANDS_PATH = os.path.join(SPLITS_DIR, "test_candidates_hard.jsonl")
LEX_PATH = os.path.join(SPLITS_DIR, "lexical_preds_hard.json")
OUTPUT_JSON = os.path.join(SPLITS_DIR, "history_gap_diagnostics.json")


def compute_token_logps(model: KristalLM, prompt_ids: List[int], cand_ids: List[int]) -> List[float]:
    """Computes per-token log-probabilities of cand_ids given prompt_ids."""
    full_seq = prompt_ids + cand_ids
    if len(full_seq) > 4096:
        full_seq = full_seq[-4096:]
        prompt_len = max(0, len(prompt_ids) - (len(prompt_ids) + len(cand_ids) - 4096))
    else:
        prompt_len = len(prompt_ids)
        
    x = torch.tensor([full_seq], dtype=torch.long, device=DEVICE)
    sign_mask = model.embedding.compute_sign_mask(x.cpu()).to(DEVICE)
    
    with torch.no_grad():
        logits, _ = model(x, sign_mask=sign_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        
    target_indices = full_seq[prompt_len:]
    if not target_indices:
        return []
        
    token_lps = []
    for i, target_id in enumerate(target_indices):
        pred_idx = prompt_len - 1 + i
        if pred_idx < log_probs.size(1):
            token_lps.append(log_probs[0, pred_idx, target_id].item())
            
    return token_lps


def main():
    print("===================================================================")
    print("KRİSTAL-VEKTÖREL: TÜRK TARİHİ < TABAN TANISI (PAKET 1)")
    print("===================================================================")
    print(f"Cihaz: {DEVICE}")
    
    # 1. Load Vocab, Compiler, Tokenizer
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    bos_id = vocab.stoi.get("<BOS>", 2)
    eos_id = vocab.stoi.get("<EOS>", 3)
    out_start_id = vocab.stoi.get("<OUTPUT>", 8)
    
    # Empty prompt representing unconditional LM generation: "<BOS> <OUTPUT>"
    empty_prompt_ids = [bos_id, out_start_id]
    
    # 2. Load Model Checkpoint
    model_path = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(DATA_DIR, "kristal_model.pt")
        
    print(f"Model Yükleniyor: {model_path}")
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(model_path, map_location="cpu")
    new_sd = resize_state_dict(model, sd)
    model.load_state_dict(new_sd, strict=False)
    model.to(DEVICE)
    model.eval()
    print("Model Hazır.\n")
    
    # 3. Load Candidates and Lexical Baseline
    with open(CANDS_PATH, "r", encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f if l.strip()]
        
    with open(LEX_PATH, "r", encoding="utf-8") as f:
        lex_map = {p["question_id"]: p["is_correct"] for p in json.load(f)}
        
    tt_records = [r for r in all_records if r.get("stratum") == "turk_tarihi"]
    print(f"Değerlendirilecek Türk Tarihi Soru Sayısı: {len(tt_records)}")
    
    logp_ranks = []
    pmi_ranks = []
    lengths = []
    cand_logps = []
    
    n_both_correct = 0
    n_model_correct_base_wrong = 0
    n_base_correct_model_wrong = 0
    n_both_wrong = 0
    
    sample_diagnostics = []
    
    for idx, item in enumerate(tt_records):
        qid = item["question_id"]
        inp = item["input"]
        inst = item.get("instruction", "")
        cands = item["candidates"]
        true_idx = item["correct_index"]
        lex_correct = lex_map.get(qid, 0)
        
        # Build Prompt
        parts = []
        if inst:
            parts.extend(["<INSTRUCTION>", inst, "</INSTRUCTION>"])
        parts.extend(["<INPUT>", inp, "</INPUT>", "<OUTPUT>"])
        prompt_str = " ".join(parts)
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]
            
        cand_cond_lps = []
        cand_prior_lps = []
        cand_pmis = []
        cand_token_lens = []
        
        for c_text in cands:
            c_ids = tokenizer.encode(c_text)
            if c_ids and c_ids[0] == bos_id: c_ids = c_ids[1:]
            if c_ids and c_ids[-1] == eos_id: c_ids = c_ids[:-1]
            
            cand_token_lens.append(len(c_ids))
            
            # 1. Conditional log-prob: log P(Cand | Prompt)
            cond_lps = compute_token_logps(model, prompt_ids, c_ids)
            mean_cond_lp = float(np.mean(cond_lps)) if cond_lps else -999.0
            cand_cond_lps.append(mean_cond_lp)
            
            # 2. Prior log-prob: log P(Cand | Empty)
            prior_lps = compute_token_logps(model, empty_prompt_ids, c_ids)
            mean_prior_lp = float(np.mean(prior_lps)) if prior_lps else -999.0
            cand_prior_lps.append(mean_prior_lp)
            
            # 3. PMI (Conditional - Prior)
            cand_pmis.append(mean_cond_lp - mean_prior_lp)
            
        # LogP Rank of True Answer
        sorted_cond = sorted(cand_cond_lps, reverse=True)
        true_cond_lp = cand_cond_lps[true_idx]
        rank_cond = sorted_cond.index(true_cond_lp) + 1
        logp_ranks.append(rank_cond)
        
        # PMI Rank of True Answer
        sorted_pmi = sorted(cand_pmis, reverse=True)
        true_pmi = cand_pmis[true_idx]
        rank_pmi = sorted_pmi.index(true_pmi) + 1
        pmi_ranks.append(rank_pmi)
        
        # Track 2x2
        model_correct = 1 if (rank_cond == 1) else 0
        if model_correct == 1 and lex_correct == 1: n_both_correct += 1
        elif model_correct == 1 and lex_correct == 0: n_model_correct_base_wrong += 1
        elif model_correct == 0 and lex_correct == 1: n_base_correct_model_wrong += 1
        else: n_both_wrong += 1
        
        lengths.extend(cand_token_lens)
        cand_logps.extend(cand_cond_lps)
        
        if idx < 5:
            pred_idx_cond = int(np.argmax(cand_cond_lps))
            pred_idx_pmi = int(np.argmax(cand_pmis))
            sample_diagnostics.append({
                "question": inp[:80],
                "true_answer": cands[true_idx][:100],
                "cond_rank": rank_cond,
                "pmi_rank": rank_pmi,
                "picked_cond": cands[pred_idx_cond][:80],
                "picked_pmi": cands[pred_idx_pmi][:80],
                "true_cond_lp": true_cond_lp,
                "true_prior_lp": cand_prior_lps[true_idx],
                "true_pmi": true_pmi
            })
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(tt_records):
            print(f"  İlerleyiş: {idx + 1:3d}/{len(tt_records)} soru analiz edildi...", flush=True)

    # 4. Statistical Summary
    n_tot = len(tt_records)
    mean_rank_cond = float(np.mean(logp_ranks))
    mean_rank_pmi = float(np.mean(pmi_ranks))
    
    top1_cond = sum(1 for r in logp_ranks if r == 1) / n_tot * 100.0
    top3_cond = sum(1 for r in logp_ranks if r <= 3) / n_tot * 100.0
    top5_cond = sum(1 for r in logp_ranks if r <= 5) / n_tot * 100.0
    
    top1_pmi = sum(1 for r in pmi_ranks if r == 1) / n_tot * 100.0
    top3_pmi = sum(1 for r in pmi_ranks if r <= 3) / n_tot * 100.0
    top5_pmi = sum(1 for r in pmi_ranks if r <= 5) / n_tot * 100.0
    
    # Length vs LogP Pearson correlation
    corr = float(np.corrcoef(lengths, cand_logps)[0, 1])
    
    print("\n" + "=" * 80)
    print("TÜRK TARİHİ TANI RAPORU SONUÇLARI (N=300)")
    print("=" * 80)
    print(f"{'Metrik':35s} | {'Standart log P':18s} | {'PMI (Prior-Düzeltmeli)':20s} | {'Şans Tabanı'}")
    print("-" * 80)
    print(f"{'Ortalama Sıra (1-10)':35s} | {mean_rank_cond:6.2f} / 10        | {mean_rank_pmi:6.2f} / 10           | 5.50 / 10")
    print(f"{'Top-1 Doğruluk':35s} | %{top1_cond:6.2f}            | %{top1_pmi:6.2f}               | %10.00")
    print(f"{'Top-3 Doğruluk':35s} | %{top3_cond:6.2f}            | %{top3_pmi:6.2f}               | %30.00")
    print(f"{'Top-5 Doğruluk':35s} | %{top5_cond:6.2f}            | %{top5_pmi:6.2f}               | %50.00")
    print("-" * 80)
    print(f"Uzunluk (Token) vs Log-Olasılık Korelasyonu (r): {corr:.4f}")
    print(f"2x2 Tablo: n11={n_both_correct}, n10={n_model_correct_base_wrong}, n01={n_base_correct_model_wrong}, n00={n_both_wrong}")
    print("=" * 80)
    
    # Interpret findings
    if top1_pmi > top1_cond + 10.0:
        diagnosis = "LM Prior Maskelemesi: Modelin içerik sinyali var ancak jenerik şablonların LM prior'ı tarafından maskeleniyor."
    elif top1_pmi <= top1_cond + 2.0:
        diagnosis = "Zayıf / Sıfır İçerik Sinyali: Model girdiye anlamsal olarak koşullanmıyor; içerik sinyali baştan yetersiz."
    else:
        diagnosis = "Kısmi Prior Etkisi: Modelde zayıf bir sinyal var, prior düzeltmesi marjinal bir toparlanma sağlıyor."
        
    print(f"\nTEŞHİS BULGUSU:\n>>> {diagnosis}\n")
    
    # Save Report
    results = {
        "n_total": n_tot,
        "mean_rank_logp": mean_rank_cond,
        "mean_rank_pmi": mean_rank_pmi,
        "top1_logp": top1_cond,
        "top3_logp": top3_cond,
        "top5_logp": top5_cond,
        "top1_pmi": top1_pmi,
        "top3_pmi": top3_pmi,
        "top5_pmi": top5_pmi,
        "length_logp_correlation": corr,
        "contingency_table": {
            "n11": n_both_correct,
            "n10": n_model_correct_base_wrong,
            "n01": n_base_correct_model_wrong,
            "n00": n_both_wrong
        },
        "diagnosis": diagnosis,
        "sample_diagnostics": sample_diagnostics,
        "logp_rank_histogram": dict(Counter(logp_ranks)),
        "pmi_rank_histogram": dict(Counter(pmi_ranks))
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Tanı Sonuçları JSON Kaydedildi: {OUTPUT_JSON}\n")


if __name__ == "__main__":
    main()
