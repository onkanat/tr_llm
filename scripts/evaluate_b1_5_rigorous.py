#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: ADIM B1.5 TİTİZ DEĞERLENDİRME VE ÖN-KAYITLI RAPORLAMA
================================================================================
1. Katman 1: Eşleştirilmiş McNemar Testi (Birincil: Zor-Negatif, İkincil: Rastgele).
   - Diskteki sabit aday kümelerini (`test_candidates_hard.jsonl` ve `test_candidates_random.jsonl`) çözer.
   - Diskteki Sözcüksel Taban tahminleri ile eşleştirerek McNemar chi2 ve p-değerini hesaplar.
2. Katman 4: Toplu Üretim Kalite Metrikleri (N=100 held-out test):
   - Ezber Oranı (Train ile >= %90 4-gram örtüşme, Eşik: < %10)
   - Tutarsızlık Oranı (Yüklemsiz / dağılma, Eşik: < %5)
   - Koşullanma Skoru (ROUGE-L >= 0.35)
3. Niteliksel Değerlendirme (Test 1, 2, 3) ve Ön-Kayıtlı Standart Tablo Üretimi.
"""

import os
import sys
import re
import json
import math
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import compute_completion_logp, wilson_ci, resize_state_dict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")

HARD_CANDS_PATH = os.path.join(SPLITS_DIR, "test_candidates_hard.jsonl")
RANDOM_CANDS_PATH = os.path.join(SPLITS_DIR, "test_candidates_random.jsonl")
HARD_LEXICAL_PREDS = os.path.join(SPLITS_DIR, "lexical_preds_hard.json")
RANDOM_LEXICAL_PREDS = os.path.join(SPLITS_DIR, "lexical_preds_random.json")
TRAIN_PATH = os.path.join(SPLITS_DIR, "train.jsonl")
TEST_PATH = os.path.join(SPLITS_DIR, "test.jsonl")


# ---------------------------------------------------------
# 1. İSTATİSTİKSEL FONKSİYONLAR: MCNEMAR TESTİ
# ---------------------------------------------------------

def compute_mcnemar_test(model_preds: List[int], baseline_preds: List[int]) -> Tuple[float, float, Dict[str, int]]:
    """
    Computes McNemar's paired test with continuity correction.
    model_preds: list of 0/1 correctness
    baseline_preds: list of 0/1 correctness
    """
    assert len(model_preds) == len(baseline_preds), "Length mismatch in paired test!"
    
    n11 = sum(1 for m, b in zip(model_preds, baseline_preds) if m == 1 and b == 1) # both correct
    n10 = sum(1 for m, b in zip(model_preds, baseline_preds) if m == 1 and b == 0) # model correct, base wrong
    n01 = sum(1 for m, b in zip(model_preds, baseline_preds) if m == 0 and b == 1) # model wrong, base correct
    n00 = sum(1 for m, b in zip(model_preds, baseline_preds) if m == 0 and b == 0) # both wrong
    
    table = {"n11": n11, "n10": n10, "n01": n01, "n00": n00}
    discordant = n10 + n01
    
    if discordant == 0:
        return 0.0, 1.0, table
        
    # Continuity corrected Chi-square
    chi2 = ((abs(n10 - n01) - 1.0) ** 2) / float(discordant)
    
    # Exact two-tailed binomial p-value
    k = min(n10, n01)
    # p = 2 * sum_{i=0}^k (n choose i) * 0.5^n
    p_val = 0.0
    for i in range(k + 1):
        p_val += math.comb(discordant, i) * (0.5 ** discordant)
    p_val = min(1.0, 2.0 * p_val)
    
    return float(chi2), float(p_val), table


# ---------------------------------------------------------
# 2. MODEL MCQ ÇIKARIM VE ADAY PUANLAMA
# ---------------------------------------------------------

def evaluate_model_on_candidates(
    model: KristalLM,
    tokenizer: KristalTokenizer,
    vocab: Vocabulary,
    candidates_path: str,
    lexical_preds_path: str,
    test_label: str
) -> Dict[str, Any]:
    print(f"\n===================================================================")
    print(f"KATMAN 1 MCQ DEĞERLENDİRMESİ: {test_label.upper()}")
    print(f"Aday Kümesi: {candidates_path}")
    print(f"===================================================================")
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
        
    with open(lexical_preds_path, "r", encoding="utf-8") as f:
        lexical_preds = json.load(f)
        
    lex_map = {p["question_id"]: p["is_correct"] for p in lexical_preds}
    
    bos_id = vocab.stoi.get("<BOS>", 2)
    eos_id = vocab.stoi.get("<EOS>", 3)
    
    model.eval()
    by_stratum_model = defaultdict(list)
    by_stratum_base = defaultdict(list)
    all_model_preds = []
    all_base_preds = []
    
    with torch.no_grad():
        for idx, item in enumerate(records):
            qid = item["question_id"]
            stratum = item.get("stratum", "unknown")
            inst = item.get("instruction", "")
            inp = item["input"]
            candidates = item["candidates"]
            true_idx = item["correct_index"]
            
            # Format Prompt
            parts = []
            if inst:
                parts.extend(["<INSTRUCTION>", inst, "</INSTRUCTION>"])
            parts.extend(["<INPUT>", inp, "</INPUT>", "<OUTPUT>"])
            prompt_str = " ".join(parts)
            
            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == eos_id:
                prompt_ids = prompt_ids[:-1]
                
            # Compute completion log-likelihood for each candidate
            scores = []
            for cand_text in candidates:
                cand_ids = tokenizer.encode(cand_text)
                if cand_ids and cand_ids[0] == bos_id:
                    cand_ids = cand_ids[1:]
                if cand_ids and cand_ids[-1] == eos_id:
                    cand_ids = cand_ids[:-1]
                lp = compute_completion_logp(model, prompt_ids, cand_ids)
                scores.append(lp)
                
            pred_idx = int(np.argmax(scores))
            is_correct = 1 if (pred_idx == true_idx) else 0
            base_correct = lex_map.get(qid, 0)
            
            by_stratum_model[stratum].append(is_correct)
            by_stratum_base[stratum].append(base_correct)
            all_model_preds.append(is_correct)
            all_base_preds.append(base_correct)
            
            if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
                print(f"  İlerleyiş: {idx + 1:3d}/{len(records)} soru değerlendirildi...", flush=True)

    # Report McNemar Results per Stratum
    print("\n" + "=" * 90)
    print(f"{'Katman (Stratum)':15s} | {'N':5s} | {'Model Acc':10s} | {'Lexical Acc':11s} | {'McNemar chi2':12s} | {'p-değeri':10s} | {'Ön-Kayıtlı Karar'}")
    print("-" * 90)
    
    results_summary = {}
    for st in sorted(by_stratum_model.keys()):
        m_list = by_stratum_model[st]
        b_list = by_stratum_base[st]
        n = len(m_list)
        m_acc = sum(m_list) / n * 100.0
        b_acc = sum(b_list) / n * 100.0
        chi2, p_val, tbl = compute_mcnemar_test(m_list, b_list)
        
        # Pre-registered ruling rule
        if st in ["high_school", "literature", "middle_school"]:
            decision = "TANIMLAYICI (Hüküm Yok: N<300)"
        elif p_val < 0.05 and m_acc > b_acc:
            decision = "BAŞARILI (p < 0.05, Model > Taban)"
        elif abs(m_acc - b_acc) <= 5.0:
            decision = "MARJİNAL (Model ≈ Taban)"
        else:
            decision = "BAŞARISIZ (Model <= Taban)"
            
        print(f"{st:15s} | {n:5d} | %{m_acc:6.2f}    | %{b_acc:6.2f}     | {chi2:10.2f}   | {p_val:8.4f}   | {decision}")
        results_summary[st] = {
            "n": n, "model_acc": m_acc, "base_acc": b_acc,
            "chi2": chi2, "p_val": p_val, "table": tbl, "decision": decision
        }
        
    tot_n = len(all_model_preds)
    tot_m_acc = sum(all_model_preds) / tot_n * 100.0
    tot_b_acc = sum(all_base_preds) / tot_n * 100.0
    tot_chi2, tot_p_val, tot_tbl = compute_mcnemar_test(all_model_preds, all_base_preds)
    
    tot_decision = "BAŞARILI (p < 0.05)" if (tot_p_val < 0.05 and tot_m_acc > tot_b_acc) else "MARJİNAL / BAŞARISIZ"
    print("-" * 90)
    print(f"{'GENEL TOPLAM':15s} | {tot_n:5d} | %{tot_m_acc:6.2f}    | %{tot_b_acc:6.2f}     | {tot_chi2:10.2f}   | {tot_p_val:8.4f}   | {tot_decision}")
    print("=" * 90 + "\n")
    
    results_summary["overall"] = {
        "n": tot_n, "model_acc": tot_m_acc, "base_acc": tot_b_acc,
        "chi2": tot_chi2, "p_val": tot_p_val, "table": tot_tbl, "decision": tot_decision
    }
    return results_summary


# ---------------------------------------------------------
# 3. KATMAN 4: TOPLU ÜRETİM KALİTE METRİKLERİ (N=100)
# ---------------------------------------------------------

def rouge_l_score(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    """Computes ROUGE-L based on Longest Common Subsequence."""
    m = len(cand_tokens)
    n = len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
        
    # LCS table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    lcs_len = lcs[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    if prec + rec == 0:
        return 0.0
    return (2.0 * prec * rec) / (prec + rec)


def evaluate_batch_generation(
    model: KristalLM,
    tokenizer: KristalTokenizer,
    vocab: Vocabulary,
    test_path: str,
    train_path: str,
    n_samples: int = 100
) -> Dict[str, Any]:
    print("===================================================================")
    print(f"KATMAN 4: TOPLU ÜRETİM KALİTE METRİKLERİ (Held-Out N={n_samples})")
    print("===================================================================")
    
    # 1. Load Training Outputs for Memorization Check
    print("Eğitim Çıktıları Yükleniyor (Ezber Denetimi İçin)...")
    train_4grams = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            words = re.findall(r"[\w']+", d.get("output", "").lower())
            for i in range(len(words) - 3):
                train_4grams.add(tuple(words[i:i+4]))
                
    print(f"Eğitim Kümesindeki Eşsiz 4-gram Sayısı: {len(train_4grams):,}")
    
    # 2. Sample Held-Out Test Items
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(l) for l in f]
        
    rng = random.Random(RANDOM_SEED)
    eval_records = rng.sample(test_records, min(n_samples, len(test_records)))
    
    bos_id = vocab.stoi.get("<BOS>", 2)
    eos_id = vocab.stoi.get("<EOS>", 3)
    out_start_id = vocab.stoi.get("<OUTPUT>", 8)
    out_end_id = vocab.stoi.get("</OUTPUT>", 9)
    
    PREDICATE_TAGS = ("TENSE_", "COPULA_")
    
    memorized_count = 0
    incoherent_count = 0
    rouge_scores = []
    content_overlap_count = 0
    
    generated_samples = []
    
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(eval_records):
            inst = item.get("instruction", "")
            inp = item["input"]
            ref_out = item["output"]
            
            parts = []
            if inst:
                parts.extend(["<INSTRUCTION>", inst, "</INSTRUCTION>"])
            parts.extend(["<INPUT>", inp, "</INPUT>", "<OUTPUT>"])
            prompt_str = " ".join(parts)
            
            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == eos_id:
                prompt_ids = prompt_ids[:-1]
                
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
            
            # Autoregressive generation (deterministic argmax)
            gen_ids = []
            max_new = 128
            curr_x = input_tensor
            
            for _ in range(max_new):
                sign_mask = model.embedding.compute_sign_mask(curr_x.cpu()).to(DEVICE)
                logits, _ = model(curr_x, sign_mask=sign_mask)
                next_token = int(torch.argmax(logits[0, -1, :]).item())
                
                if next_token in (eos_id, out_end_id):
                    break
                gen_ids.append(next_token)
                next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
                curr_x = torch.cat([curr_x, next_tensor], dim=1)
                if curr_x.size(1) >= 256:
                    break
                    
            gen_tokens = [vocab.decode(tid) for tid in gen_ids]
            gen_text = " ".join(gen_tokens)
            
            # --- Metrik 1: Ezber Denetimi (4-gram overlap >= 90%) ---
            gen_words = re.findall(r"[\w']+", gen_text.lower())
            cand_4grams = [tuple(gen_words[i:i+4]) for i in range(len(gen_words) - 3)]
            
            if cand_4grams:
                matched_4grams = sum(1 for fg in cand_4grams if fg in train_4grams)
                overlap_ratio = matched_4grams / len(cand_4grams)
            else:
                overlap_ratio = 0.0
                
            is_memorized = (overlap_ratio >= 0.90)
            if is_memorized:
                memorized_count += 1
                
            # --- Metrik 2: Tutarsızlık Denetimi (Yüklem ve Tekrar Döngüsü) ---
            # Has predicate tag in last 5 tokens?
            last_tokens = gen_tokens[-5:] if len(gen_tokens) >= 5 else gen_tokens
            has_predicate = any(tok.startswith(PREDICATE_TAGS) for tok in last_tokens)
            
            # Check 3x consecutive repeating 2-grams
            has_repetition_loop = False
            if len(gen_tokens) >= 6:
                for i in range(len(gen_tokens) - 5):
                    if gen_tokens[i:i+2] == gen_tokens[i+2:i+4] == gen_tokens[i+4:i+6]:
                        has_repetition_loop = True
                        break
                        
            is_incoherent = (not has_predicate) or has_repetition_loop
            if is_incoherent:
                incoherent_count += 1
                
            # --- Metrik 3: Koşullanma ve ROUGE-L ---
            ref_words = re.findall(r"[\w']+", ref_out.lower())
            r_l = rouge_l_score(gen_words, ref_words)
            rouge_scores.append(r_l)
            
            q_words = set(re.findall(r"[\w']+", inp.lower()))
            common_content = len(q_words & set(gen_words))
            if common_content >= 2:
                content_overlap_count += 1
                
            if idx < 5:
                generated_samples.append({
                    "idx": idx + 1,
                    "input": inp[:80],
                    "generated": gen_text[:120],
                    "reference": ref_out[:120],
                    "rouge_l": r_l,
                    "memorized": is_memorized,
                    "incoherent": is_incoherent
                })

    mem_rate = (memorized_count / len(eval_records)) * 100.0
    incoh_rate = (incoherent_count / len(eval_records)) * 100.0
    mean_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    cond_rate = (content_overlap_count / len(eval_records)) * 100.0
    
    print("\nTOPLU ÜRETİM KALİTE RAPORU:")
    print("-" * 75)
    print(f"1. Ezber Oranı (Train ile >= %90 4-gram) : %{mem_rate:5.2f} (Ön-Kayıtlı Eşik: < %10.0)")
    print(f"2. Tutarsızlık Oranı (Yüklemsiz / Döngü): %{incoh_rate:5.2f} (Ön-Kayıtlı Eşik: <  %5.0)")
    print(f"3. Koşullanma Skoru (ROUGE-L Ortalaması): {mean_rouge:6.4f}  (Ön-Kayıtlı Eşik: >= 0.35)")
    print(f"4. Soruyla İçerik Kesişimi (>=2 kök)   : %{cond_rate:5.2f} (Ön-Kayıtlı Eşik: >= %80.0)")
    print("-" * 75)
    
    # Pre-registered ruling for Layer 4
    mem_pass = mem_rate < 10.0
    incoh_pass = incoh_rate < 5.0
    rouge_pass = mean_rouge >= 0.35
    layer_4_decision = "BAŞARILI" if (mem_pass and incoh_pass and rouge_pass) else "BAŞARISIZ"
    print(f"KATMAN 4 HÜKMÜ: {layer_4_decision} (Ezber={mem_pass}, Tutarsızlık={incoh_pass}, ROUGE={rouge_pass})\n")
    
    print("Örnek Üretimler (İlk 3 Test):")
    for s in generated_samples[:3]:
        print(f"  [Örnek {s['idx']}] Soru: {s['input']}")
        print(f"    Model Çıktısı : {s['generated']}")
        print(f"    Referans Yanıt: {s['reference']}")
        print(f"    ROUGE-L: {s['rouge_l']:.4f} | Ezber: {s['memorized']} | Tutarsız: {s['incoherent']}\n")
        
    return {
        "n": len(eval_records),
        "memorization_rate": mem_rate,
        "incoherence_rate": incoh_rate,
        "mean_rouge_l": mean_rouge,
        "conditioning_rate": cond_rate,
        "decision": layer_4_decision,
        "samples": generated_samples
    }


# ---------------------------------------------------------
# 4. ANA YÜRÜTME
# ---------------------------------------------------------

def main():
    print("===================================================================")
    print("KRİSTAL-VEKTÖREL: ADIM B1.5 TİTİZ DOĞRULAMA VE RAPORLAMA")
    print("===================================================================")
    print(f"Hesaplama Cihazı: {DEVICE}")
    
    # 1. Load Vocab, Lexicon, Tokenizer
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    # 2. Load Selected Best Checkpoint
    best_model_path = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
    if not os.path.exists(best_model_path):
        # Fallback to model.pt if best.pt not generated yet
        best_model_path = os.path.join(DATA_DIR, "kristal_model.pt")
        
    print(f"Değerlendirilecek Model Checkpoint'i: {best_model_path}")
    model = KristalLM(
        vocab_size=len(vocab.stoi), 
        n_embd=768, 
        vocab=vocab, 
        block_size=4096, 
        n_layer=6, 
        n_head=6
    )
    sd = torch.load(best_model_path, map_location="cpu")
    new_sd = resize_state_dict(model, sd)
    model.load_state_dict(new_sd, strict=False)
    model.to(DEVICE)
    print("Model Ağırlıkları Başarıyla Yüklendi.")
    
    # 3. Katman 1 (Birincil: Hard-Negative Distractors)
    res_hard = evaluate_model_on_candidates(
        model, tokenizer, vocab, 
        HARD_CANDS_PATH, HARD_LEXICAL_PREDS, 
        "Birincil Test: Zor-Negatif (Hard Negatives)"
    )
    
    # 4. Katman 1 (İkincil: Random Distractors)
    res_rand = evaluate_model_on_candidates(
        model, tokenizer, vocab, 
        RANDOM_CANDS_PATH, RANDOM_LEXICAL_PREDS, 
        "İkincil Sanity Testi: Rastgele Çeldirici (Random)"
    )
    
    # 5. Katman 4: Toplu Üretim Kalite Metrikleri (N=100)
    res_gen = evaluate_batch_generation(
        model, tokenizer, vocab,
        TEST_PATH, TRAIN_PATH, n_samples=100
    )
    
    # 6. Save Complete Final Evaluation JSON
    final_report_path = os.path.join(SPLITS_DIR, "final_evaluation_report.json")
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "katman_1_hard_negatives": res_hard,
            "katman_1_random_distractors": res_rand,
            "katman_4_batch_generation": res_gen
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Nihai Rapor JSON Kaydedildi: {final_report_path}")
    print("Adım B1.5 Titiz Doğrulama Süreci Tamamlandı! ✓\n")


if __name__ == "__main__":
    main()
