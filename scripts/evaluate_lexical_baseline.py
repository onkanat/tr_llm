#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: SÖZCÜKSEL-ÖRTÜŞME TABAN ÇİZGİSİ MOTORU (LEXICAL BASELINE)
==========================================================================
1. Yalnızca `data/b1_5_splits/train.jsonl` kümesi üzerinde IDF matrisini fit eder (Sıfır Test Sızıntısı).
2. Diskte sabitlenmiş `test_candidates_hard.jsonl` (Birincil) ve `test_candidates_random.jsonl` (İkincil)
   dosyalarını okur.
3. Soru ile adaylar arasındaki morfem/sözcük TF-IDF örtüşmesini hesaplar ve en yüksek örtüşmeli adayı seçer.
4. Her test öğesi için ikili (0/1) tahmin vektörünü kaydeder (`lexical_preds_hard.json` ve `lexical_preds_random.json`).
5. Katman bazında doğruluk ve Wilson %95 Güven Aralığını raporlar.
"""

import os
import re
import json
import math
from collections import Counter
from typing import List, Dict, Set, Tuple
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")

TRAIN_PATH = os.path.join(SPLITS_DIR, "train.jsonl")
HARD_CANDS_PATH = os.path.join(SPLITS_DIR, "test_candidates_hard.jsonl")
RANDOM_CANDS_PATH = os.path.join(SPLITS_DIR, "test_candidates_random.jsonl")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Calculates Wilson 95% Confidence Interval for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denominator
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return p, lower, upper


def get_tokens(text: str) -> Set[str]:
    """Cleans text and extracts morpheme/word tokens."""
    return set(re.findall(r"[\w']+", text.lower(), flags=re.UNICODE))


def fit_train_idf(train_path: str) -> Dict[str, float]:
    """
    Fits IDF weights STRICTLY on train.jsonl.
    Zero leakage into test evaluation!
    """
    print(f"Train IDF Eğitiliyor: {train_path}...")
    doc_counts = Counter()
    total_docs = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # Combine input and output as document
            toks = get_tokens(d.get("input", "") + " " + d.get("output", ""))
            for t in toks:
                doc_counts[t] += 1
            total_docs += 1
            
    # IDF with add-1 smoothing
    idf = {}
    for term, cnt in doc_counts.items():
        idf[term] = math.log((total_docs + 1.0) / (cnt + 1.0)) + 1.0
        
    print(f"Train Doküman Sayısı: {total_docs:,} | Sözlükteki Terim Sayısı: {len(idf):,}")
    return idf


def evaluate_candidates(candidates_path: str, idf: Dict[str, float], output_pred_path: str, label: str):
    print(f"\n===================================================================")
    print(f"SÖZCÜKSEL TABAN DEĞERLENDİRMESİ: {label.upper()}")
    print(f"Aday Dosyası: {candidates_path}")
    print(f"===================================================================")
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
        
    by_stratum = Counter()
    correct_by_stratum = Counter()
    predictions = []
    
    default_idf = math.log(len(idf) + 1.0) if len(idf) > 0 else 1.0
    
    for item in records:
        stratum = item.get("stratum", "unknown")
        q_text = item["input"]
        q_tokens = get_tokens(q_text)
        candidates = item["candidates"]
        correct_idx = item["correct_index"]
        
        # Score each candidate using TF-IDF weighted overlap
        cand_scores = []
        for c_idx, c_text in enumerate(candidates):
            c_tokens = get_tokens(c_text)
            inter = q_tokens & c_tokens
            score = sum(idf.get(t, default_idf) for t in inter)
            cand_scores.append(score)
            
        # Select best candidate (argmax)
        best_cand_idx = int(np.argmax(cand_scores))
        is_correct = (best_cand_idx == correct_idx)
        
        by_stratum[stratum] += 1
        if is_correct:
            correct_by_stratum[stratum] += 1
            
        predictions.append({
            "question_id": item["question_id"],
            "stratum": stratum,
            "correct_index": correct_idx,
            "predicted_index": best_cand_idx,
            "is_correct": 1 if is_correct else 0
        })
        
    # Save predictions
    with open(output_pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        
    print(f"\nTahminler Kaydedildi: {output_pred_path} ({len(predictions)} öğe)")
    print(f"\n{'Katman (Stratum)':20s} | {'N':5s} | {'Doğru':5s} | {'Doğruluk':8s} | {'Wilson %95 GA':18s}")
    print("-" * 65)
    
    total_n = len(records)
    total_corr = sum(p["is_correct"] for p in predictions)
    
    for st in sorted(by_stratum.keys()):
        n = by_stratum[st]
        k = correct_by_stratum[st]
        p, lo, hi = wilson_ci(k, n)
        print(f"{st:20s} | {n:5d} | {k:5d} | %{p*100:6.2f} | [%{lo*100:5.1f}, %{hi*100:5.1f}]")
        
    tot_p, tot_lo, tot_hi = wilson_ci(total_corr, total_n)
    print("-" * 65)
    print(f"{'GENEL TOPLAM':20s} | {total_n:5d} | {total_corr:5d} | %{tot_p*100:6.2f} | [%{tot_lo*100:5.1f}, %{tot_hi*100:5.1f}]\n")
    return predictions


def main():
    # 1. Fit Train IDF strictly on train.jsonl
    idf = fit_train_idf(TRAIN_PATH)
    
    # 2. Evaluate Primary: Hard-Negative Distractors
    hard_preds_path = os.path.join(SPLITS_DIR, "lexical_preds_hard.json")
    evaluate_candidates(HARD_CANDS_PATH, idf, hard_preds_path, "Birincil: Zor-Negatif (Hard Negatives)")
    
    # 3. Evaluate Secondary: Random Distractors
    rand_preds_path = os.path.join(SPLITS_DIR, "lexical_preds_random.json")
    evaluate_candidates(RANDOM_CANDS_PATH, idf, rand_preds_path, "İkincil (Sanity): Rastgele Çeldirici (Random)")


if __name__ == "__main__":
    main()
