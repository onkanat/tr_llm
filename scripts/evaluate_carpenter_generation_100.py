#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: CARPENTER N=100 ÜRETİM KALİTE TESTİ (PAKET 2)
==============================================================
Hedef:
N=25 rastgele örneklemde ölçülen ROUGE-L 0.3735 ve Tutarsızlık %4.00
başarısının varyans olmadığını, N=100 held-out marangozluk örneği üzerinde
istatistiksel olarak doğrulamak.

Ölçülecek Metrikler:
1. Ezber Oranı (Train ile >= %90 4-gram, Ön-Kayıtlı Eşik: < %10.0)
2. Tutarsızlık Oranı (Yüklemsiz / Döngü, Ön-Kayıtlı Eşik: < %5.0)
3. Koşullanma Skoru (ROUGE-L Ortalaması, Ön-Kayıtlı Eşik: >= 0.35)
4. Soruyla İçerik Kesişimi (>=2 kök, Ön-Kayıtlı Eşik: >= %80.0)
"""

import os
import sys
import re
import json
import random
import numpy as np
import torch
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict, wilson_ci
from scripts.evaluate_b1_5_rigorous import rouge_l_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")
TEST_PATH = os.path.join(SPLITS_DIR, "test.jsonl")
TRAIN_PATH = os.path.join(SPLITS_DIR, "train.jsonl")
OUTPUT_JSON = os.path.join(SPLITS_DIR, "carpenter_generation_100_results.json")


def main():
    print("===================================================================")
    print("KRİSTAL-VEKTÖREL: CARPENTER N=100 HELD-OUT ÜRETİM TESTİ (PAKET 2)")
    print("===================================================================")
    print(f"Cihaz: {DEVICE}")
    
    # 1. Load Vocab, Lexicon, Decompiler, Tokenizer
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)
    
    bos_id = vocab.stoi.get("<BOS>", 2)
    eos_id = vocab.stoi.get("<EOS>", 3)
    out_end_id = vocab.stoi.get("</OUTPUT>", 9)
    PREDICATE_TAGS = ("TENSE_", "COPULA_")
    
    import argparse
    parser = argparse.ArgumentParser(description="Carpenter generation evaluation")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    # 2. Load Model
    if args.model:
        model_path = args.model
    else:
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
    
    # 3. Load Train 4-grams for Memorization Check
    print("Eğitim 4-gram'ları yükleniyor...")
    train_4grams = set()
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            words = re.findall(r"[\w']+", d.get("output", "").lower())
            for i in range(len(words) - 3):
                train_4grams.add(tuple(words[i:i+4]))
    print(f"Eğitimdeki Eşsiz 4-gram Sayısı: {len(train_4grams):,}")
    
    # 4. Filter Held-Out Carpenter Records and Sample N=100
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        all_test = [json.loads(l) for l in f if l.strip()]
        
    carpenter_test = [r for r in all_test if "Ahşap" in r.get("instruction", "")]
    print(f"Toplam Held-Out Marangozluk Kaydı: {len(carpenter_test)}")
    
    rng = random.Random(RANDOM_SEED)
    sample_records = rng.sample(carpenter_test, min(100, len(carpenter_test)))
    print(f"Örneklenen Değerlendirme Kaydı: {len(sample_records)}\n")
    
    memorized_count = 0
    incoherent_count = 0
    content_overlap_count = 0
    rouge_scores = []
    
    qualitative_samples = []
    
    with torch.no_grad():
        for idx, item in enumerate(sample_records):
            inp = item["input"]
            ref_out = item["output"]
            inst = item.get("instruction", "")
            
            parts = []
            if inst:
                parts.extend(["<INSTRUCTION>", inst, "</INSTRUCTION>"])
            parts.extend(["<INPUT>", inp, "</INPUT>", "<OUTPUT>"])
            prompt_str = " ".join(parts)
            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == eos_id:
                prompt_ids = prompt_ids[:-1]
                
            curr_x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
            gen_ids = []
            
            for _ in range(128):
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
            gen_morphemic = " ".join(gen_tokens)
            
            # Decompile to surface Turkish
            try:
                gen_surface = decompiler.decompile_sentence(gen_morphemic)
            except Exception:
                gen_surface = gen_morphemic
                
            gen_words = re.findall(r"[\w']+", gen_morphemic.lower())
            ref_words = re.findall(r"[\w']+", ref_out.lower())
            
            # 1. Memorization Check
            cand_4grams = [tuple(gen_words[i:i+4]) for i in range(len(gen_words) - 3)]
            is_mem = False
            if cand_4grams:
                m_count = sum(1 for fg in cand_4grams if fg in train_4grams)
                if (m_count / len(cand_4grams)) >= 0.90:
                    is_mem = True
            if is_mem:
                memorized_count += 1
                
            # 2. Incoherence Check
            last_tokens = gen_tokens[-5:] if len(gen_tokens) >= 5 else gen_tokens
            has_predicate = any(tok.startswith(PREDICATE_TAGS) for tok in last_tokens)
            has_loop = False
            if len(gen_tokens) >= 6:
                for i in range(len(gen_tokens) - 5):
                    if gen_tokens[i:i+2] == gen_tokens[i+2:i+4] == gen_tokens[i+4:i+6]:
                        has_loop = True
                        break
            is_incoh = (not has_predicate) or has_loop
            if is_incoh:
                incoherent_count += 1
                
            # 3. ROUGE-L
            r_l = rouge_l_score(gen_words, ref_words)
            rouge_scores.append(r_l)
            
            # 4. Content Overlap
            q_words = set(re.findall(r"[\w']+", inp.lower()))
            has_overlap = len(q_words & set(gen_words)) >= 2
            if has_overlap:
                content_overlap_count += 1
                
            if idx < 5:
                qualitative_samples.append({
                    "idx": idx + 1,
                    "input": inp,
                    "generated_morphemes": gen_morphemic[:120],
                    "generated_surface": gen_surface[:150],
                    "reference": ref_out[:150],
                    "rouge_l": r_l,
                    "is_memorized": is_mem,
                    "is_incoherent": is_incoh,
                    "has_overlap": has_overlap
                })
                
            if (idx + 1) % 25 == 0 or (idx + 1) == len(sample_records):
                print(f"  İlerleyiş: {idx + 1:3d}/{len(sample_records)} marangozluk örneği üretildi...", flush=True)

    # 5. Report Summary
    n = len(sample_records)
    mem_rate = (memorized_count / n) * 100.0
    incoh_rate = (incoherent_count / n) * 100.0
    mean_rouge = float(np.mean(rouge_scores))
    median_rouge = float(np.median(rouge_scores))
    overlap_rate = (content_overlap_count / n) * 100.0
    
    # 95% Wilson CIs
    mem_low, mem_high = wilson_ci(memorized_count, n)
    incoh_low, incoh_high = wilson_ci(incoherent_count, n)
    overlap_low, overlap_high = wilson_ci(content_overlap_count, n)
    
    # ROUGE normal approx CI
    r_std = float(np.std(rouge_scores))
    rouge_ci_low = max(0.0, mean_rouge - 1.96 * (r_std / np.sqrt(n)))
    rouge_ci_high = min(1.0, mean_rouge + 1.96 * (r_std / np.sqrt(n)))
    
    print("\n" + "=" * 80)
    print("CARPENTER HELD-OUT N=100 TOPLU ÜRETİM RAPORU")
    print("=" * 80)
    print(f"1. Ezber Oranı (Train ile >= %90 4-gram) : %{mem_rate:5.2f} [{mem_low:4.1f}%, {mem_high:4.1f}%]  (Eşik: < %10.0)")
    print(f"2. Tutarsızlık Oranı (Yüklemsiz / Döngü): %{incoh_rate:5.2f} [{incoh_low:4.1f}%, {incoh_high:4.1f}%]  (Eşik: <  %5.0)")
    print(f"3. Koşullanma Skoru (ROUGE-L Ortalaması): {mean_rouge:6.4f} [{rouge_ci_low:.4f}, {rouge_ci_high:.4f}] (Eşik: >= 0.35)")
    print(f"   ROUGE-L Medyanı                      : {median_rouge:6.4f}")
    print(f"4. Soruyla İçerik Kesişimi (>=2 kök)   : %{overlap_rate:5.2f} [{overlap_low:4.1f}%, {overlap_high:4.1f}%]  (Eşik: >= %80.0)")
    print("-" * 80)
    
    mem_pass = mem_rate < 10.0
    incoh_pass = incoh_rate < 5.0
    rouge_pass = mean_rouge >= 0.35
    overlap_pass = overlap_rate >= 80.0
    
    status = "BAŞARILI" if (mem_pass and incoh_pass and rouge_pass and overlap_pass) else (
        "KISMEN BAŞARILI" if (mem_pass and incoh_pass and rouge_pass) else "BAŞARISIZ"
    )
    print(f"NİHAİ CARPENTER ÜRETİM HÜKMÜ: {status}")
    print(f"(Ezber: {mem_pass}, Tutarsızlık: {incoh_pass}, ROUGE: {rouge_pass}, Kesişim: {overlap_pass})\n")
    
    print("Resmî Decompile Canlı Çıktı Örnekleri (Tam Cümle):")
    for s in qualitative_samples[:3]:
        print(f"  [Örnek {s['idx']}] Soru: {s['input']}")
        print(f"    Decompile Çıktı: {s['generated_surface']}")
        print(f"    Referans Yanıt : {s['reference']}")
        print(f"    ROUGE-L: {s['rouge_l']:.4f} | Tutarsız: {s['is_incoherent']} | Kesişim: {s['has_overlap']}\n")
        
    results = {
        "n": n,
        "memorization_rate": mem_rate,
        "incoherence_rate": incoh_rate,
        "mean_rouge_l": mean_rouge,
        "median_rouge_l": median_rouge,
        "rouge_ci": [rouge_ci_low, rouge_ci_high],
        "content_overlap_rate": overlap_rate,
        "decision": status,
        "samples": qualitative_samples
    }
    
    target_out_json = args.output if args.output else OUTPUT_JSON
    with open(target_out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Sonuçlar Kaydedildi: {target_out_json}\n")


if __name__ == "__main__":
    main()
