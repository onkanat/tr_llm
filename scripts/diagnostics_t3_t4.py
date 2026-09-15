#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T3 & T4 TEŞHİS DENEYİ: KATMANLI VE FORMAT-KARŞILAŞTIRMALI MCQ
============================================================
T3: Kaynak bazında katmanlı MCQ (high_school, turk_tarihi, carpenter).
T4: Format A (Eğitim: Soru INSTRUCTION'da) vs Format B (Konuşlandırma: Soru INPUT'ta).
"""

import os
import sys
import json
import random
import numpy as np
import torch
from typing import Tuple, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import compute_completion_logp, wilson_ci, load_eval_model, DEVICE
from src.llm.prompt_contract import render_prompt


def run_t3_t4():
    print("=" * 75)
    print(" T3 & T4 TEŞHİS DENEYİ: KATMANLI VE FORMAT-KARŞILAŞTIRMALI MCQ")
    print("=" * 75)
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    model_path = 'data/kristal_model_step_a_final.pt'
    if not os.path.exists(model_path):
        model_path = 'data/kristal_model.pt'
    print(f"Değerlendirilen Model: {model_path}")
    model = load_eval_model(model_path, vocab)
    
    # 1. High School pool
    hs_pool = []
    with open('data/pedagogy/high_school_foundation_dataset.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q, a, inst = d.get('input', '').strip(), d.get('output', '').strip(), d.get('instruction', '').strip()
            if q and a and len(a) > 10:
                hs_pool.append(('high_school', inst, q, a))
                
    # 2. Turk Tarihi pool
    hist_pool = []
    with open('data/pedagogy/turk_tarihi_sft.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            # In raw turk_tarihi, instruction was question, input was empty
            raw_inst = d.get('instruction', '').strip()
            raw_inp = d.get('input', '').strip()
            a = d.get('output', '').strip()
            if raw_inst and a and len(a) > 10 and not any(m in raw_inst.lower() for m in ['tarih -', 'page', 'dizin']):
                hist_pool.append(('turk_tarihi', raw_inst, a))
                
    # 3. Carpenter pool
    carp_pool = []
    with open('data/pedagogy/carpenter_specialization_dataset.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            # In raw carpenter, instruction had question or prompt
            raw_inst = d.get('instruction', '').strip()
            a = d.get('output', '').strip()
            if raw_inst and a and len(a) > 10:
                carp_pool.append(('carpenter', raw_inst, a))
                
    rng = random.Random(42)
    rng.shuffle(hs_pool)
    rng.shuffle(hist_pool)
    rng.shuffle(carp_pool)
    
    N_per_stratum = 50
    hs_samples = hs_pool[:N_per_stratum]
    hist_samples = hist_pool[:N_per_stratum]
    carp_samples = carp_pool[:N_per_stratum]
    
    # Answer pools for distractors
    all_answers_hs = [a for _, _, _, a in hs_pool]
    all_answers_hist = [a for _, _, a in hist_pool]
    all_answers_carp = [a for _, _, a in carp_pool]
    
    K = 10
    eos_id = vocab.stoi.get('<EOS>', 3)
    bos_id = vocab.stoi.get('<BOS>', 2)
    
    def eval_stratum(samples, answer_pool, role_name, is_hs=False):
        # We test both Format A (Question in INSTRUCTION) and Format B (Question in INPUT)
        results = {'Format_A': {'correct': 0, 'ranks': []}, 'Format_B': {'correct': 0, 'ranks': []}}
        unique_answers = list(dict.fromkeys(answer_pool))
        
        for item in samples:
            if is_hs:
                _, inst_raw, q_raw, true_ans = item
                q = q_raw
                role = inst_raw
            else:
                _, q_raw, true_ans = item
                q = q_raw
                role = role_name
                
            distractors = random.sample([a for a in unique_answers if a.strip() != true_ans.strip()], K - 1)
            candidates = [true_ans] + distractors
            
            # Prepare candidate token IDs
            cand_token_ids = []
            for cand in candidates:
                c_ids = tokenizer.encode(cand)
                if c_ids and c_ids[0] == bos_id: c_ids = c_ids[1:]
                if c_ids and c_ids[-1] == eos_id: c_ids = c_ids[:-1]
                cand_token_ids.append(c_ids)
                
            # Format A: Question in INSTRUCTION (<INSTRUCTION> {q} </INSTRUCTION> <INPUT> </INPUT> <OUTPUT>)
            p_a_str = render_prompt(q, "")
            p_a_ids = tokenizer.encode(p_a_str)
            if p_a_ids and p_a_ids[-1] == eos_id: p_a_ids = p_a_ids[:-1]
            
            # Format B: Deployment (<INSTRUCTION> {role} </INSTRUCTION> <INPUT> {q} </INPUT> <OUTPUT>)
            p_b_str = render_prompt(role, q)
            p_b_ids = tokenizer.encode(p_b_str)
            if p_b_ids and p_b_ids[-1] == eos_id: p_b_ids = p_b_ids[:-1]
            
            # Score Format A
            scores_a = [compute_completion_logp(model, p_a_ids, c_ids) for c_ids in cand_token_ids]
            rank_a = int(np.where(np.argsort(scores_a)[::-1] == 0)[0][0]) + 1
            results['Format_A']['ranks'].append(rank_a)
            if rank_a == 1: results['Format_A']['correct'] += 1
            
            # Score Format B
            scores_b = [compute_completion_logp(model, p_b_ids, c_ids) for c_ids in cand_token_ids]
            rank_b = int(np.where(np.argsort(scores_b)[::-1] == 0)[0][0]) + 1
            results['Format_B']['ranks'].append(rank_b)
            if rank_b == 1: results['Format_B']['correct'] += 1
            
        return results

    print("\n1. High School Katmanı Ölçülüyor...")
    res_hs = eval_stratum(hs_samples, all_answers_hs, 'Lise fen ve temel bilimler uzmanı olarak açıkla.', is_hs=True)
    
    print("2. Türk Tarihi Katmanı Ölçülüyor...")
    res_hist = eval_stratum(hist_samples, all_answers_hist, 'Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.')
    
    print("3. Carpenter Katmanı Ölçülüyor...")
    res_carp = eval_stratum(carp_samples, all_answers_carp, 'Ahşap ve marangozluk uzmanı olarak cevapla.')
    
    print("\n" + "=" * 75)
    print(" T3 & T4 SONUÇ TABLOSU")
    print("=" * 75)
    print(f"{'Katman (Stratum)':<18} | {'Format A (Eğitim: Soru INSTRUCTION)':<35} | {'Format B (Konuşlandırma: Soru INPUT)':<35}")
    print("-" * 95)
    
    for name, res in [('Lise Temel Bilim', res_hs), ('Türk Tarihi 1931', res_hist), ('Marangozluk', res_carp)]:
        n_tot = len(res['Format_A']['ranks'])
        acc_a = (res['Format_A']['correct'] / n_tot) * 100
        mean_r_a = np.mean(res['Format_A']['ranks'])
        ci_a_l, ci_a_h = wilson_ci(res['Format_A']['correct'], n_tot)
        
        acc_b = (res['Format_B']['correct'] / n_tot) * 100
        mean_r_b = np.mean(res['Format_B']['ranks'])
        ci_b_l, ci_b_h = wilson_ci(res['Format_B']['correct'], n_tot)
        
        str_a = f"%{acc_a:4.1f} [{ci_a_l:.1f}, {ci_a_h:.1f}] (Sıra: {mean_r_a:.2f})"
        str_b = f"%{acc_b:4.1f} [{ci_b_l:.1f}, {ci_b_h:.1f}] (Sıra: {mean_r_b:.2f})"
        print(f"{name:<18} | {str_a:<35} | {str_b:<35}")
        
    print("=" * 75)

if __name__ == '__main__':
    run_t3_t4()
