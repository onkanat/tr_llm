#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import random
import torch
import numpy as np

# Ensure workspace root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalDataset, KristalLM

from src.llm.prompt_contract import resize_state_dict

def generate_response(model, vocab, prompt_tokens, max_new_tokens=15, device='cpu', repetition_penalty=1.5):
    model.eval()
    generated = list(prompt_tokens)
    eos_id = vocab.stoi.get("<EOS>", -1)
    output_end_id = vocab.stoi.get("</OUTPUT>", -1)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :]
            
            output_tokens = generated[len(prompt_tokens):]
            if repetition_penalty > 1.0 and output_tokens:
                window_tokens = output_tokens[-12:]
                for token_id in set(window_tokens):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty
                        
            pred_id = torch.argmax(logits).item()
            generated.append(pred_id)
            if pred_id == eos_id or pred_id == output_end_id:
                break
                
    response_tokens = generated[len(prompt_tokens):]
    # Filter out stop tokens from output
    clean_tokens = [tid for tid in response_tokens if tid not in (eos_id, output_end_id)]
    return clean_tokens


def evaluate_perplexity_on_binary(model, dataset, device, num_batches=30, batch_size=32):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
            sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
            x = x_cpu.to(device)
            targets = y_cpu.to(device)
            sign_mask = sign_mask_cpu.to(device)
            
            _, loss = model(x, targets, sign_mask)
            losses.append(loss.item())
            
    avg_loss = float(np.mean(losses))
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl

def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL MODELİN EĞİTİM VERİSİNE GÖRE DEĞERLENDİRİLMESİ")
    print("=" * 70)

    # 1. Device Setup
    device = torch.device("cpu")
    print(f"Cihaz: {device}", flush=True)

    # 2. Load Vocab, Lexicon, Compiler, Decompiler
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}", flush=True)

    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)

    # 3. Load Model
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    model_path = 'data/kristal_model.pt'
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası '{model_path}' bulunamadı!", flush=True)
        return
        
    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
    state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model '{model_path}' adresinden başarıyla yüklendi.\n", flush=True)

    # =========================================================================
    # A. KÜLLİYAT & DERİNLEŞTİRİLMİŞ EĞİTİM BİNARY VERİSİ PPL ÖLÇÜMÜ
    # =========================================================================
    for bin_path, desc in [('data/train.bin', 'Genel Eğitim Verisi (train.bin)'),
                           ('data/train_deep_sft.bin', 'Derinleştirilmiş SFT Verisi (train_deep_sft.bin)')]:
        if os.path.exists(bin_path):
            print("-" * 60)
            print(f" [BÖLÜM] {desc} Perplexity Ölçümü")
            print("-" * 60)
            block_size = 128 if 'deep' in bin_path else 64
            dataset = KristalDataset(bin_path, block_size=block_size)
            avg_loss, ppl = evaluate_perplexity_on_binary(model, dataset, device, num_batches=25, batch_size=32)
            print(f"  -> Örneklenen Batch Sayısı: 25 (Batch Boyutu: 32)")
            print(f"  -> Ortalama Kayıp (Cross-Entropy Loss): {avg_loss:.4f}")
            print(f"  -> Şaşkınlık (Perplexity - PPL):        {ppl:.2f}\n")

    # =========================================================================
    # B. SFT DERİNLEŞTİRİLMİŞ EĞİTİM VERİSİ GÖREV DOĞRULUK ANALİZİ
    # =========================================================================
    eval_jsonl_paths = [
        ('data/pedagogy/parenting_deep_dataset.jsonl', 40, "Derinleştirilmiş Ebeveynlik (10 Morfolojik Görev)"),
        ('data/pedagogy/carpenter_specialization_dataset.jsonl', 10, "Marangozluk Alan Uzmanlığı (Carpenter AI)"),
    ]
    
    for jsonl_path, sample_size, dataset_title in eval_jsonl_paths:
        if not os.path.exists(jsonl_path):
            continue
            
        print("-" * 60)
        print(f" [SFT DOĞRULUK TESTİ] {dataset_title}")
        print("-" * 60)
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
            
        random.seed(42)
        sample_n = min(sample_size, len(all_lines))
        sampled_lines = random.sample(all_lines, sample_n)
        
        output_start_id = vocab.stoi.get("<OUTPUT>", -1)
        task_stats = {}
        exact_matches = 0
        results_to_print = []
        
        for idx, line in enumerate(sampled_lines):
            record = json.loads(line)
            instruction = record.get("instruction", "")
            inp = record.get("input", "")
            expected_output = record.get("output", "")
            
            # Identify task type
            if "carpenter" in jsonl_path:
                task_type = "MARANGOZLUK"
            elif any(w in instruction.lower() for w in ["marangoz", "ahşap", "ağaç", "mobilya"]):
                task_type = "MARANGOZLUK"
            elif "kök" in instruction:
                task_type = "KÖK_BULMA"
            elif "hâl" in instruction or "durum" in instruction:
                task_type = "HÂL_EKİ"
            elif "zaman" in instruction or "kip" in instruction:
                task_type = "KİP_ZAMAN"
            elif "çoğul" in instruction or "PLURAL" in instruction:
                task_type = "ÇOĞUL_EKİ"
            elif "iyelik" in instruction or "aitlik" in instruction:
                task_type = "İYELİK_EKİ"
            elif "olumsuzluk" in instruction:
                task_type = "OLUMSUZLUK"
            elif "yeterlilik" in instruction:
                task_type = "YETERLİLİK"
            elif "yapım" in instruction:
                task_type = "YAPIM_EKİ"
            elif "ayrıştır" in instruction or "morfem" in instruction:
                task_type = "SEGMENTASYON"
            elif "sentezle" in instruction:
                task_type = "SENTEZLEME"
            elif "tanım" in instruction:
                task_type = "SÖZLÜK_TANIMI"
            else:
                task_type = "ALAN_UZMANLIĞI"
                
            if task_type not in task_stats:
                task_stats[task_type] = {"total": 0, "correct": 0}
            task_stats[task_type]["total"] += 1
            
            prompt_obj = {
                "instruction": instruction,
                "input": inp,
                "output": ""
            }
            prompt_str = json.dumps(prompt_obj, ensure_ascii=False)
            prompt_ids = tokenizer.encode(prompt_str)
            
            if output_start_id in prompt_ids:
                cut_idx = prompt_ids.index(output_start_id) + 1
                eval_prompt_ids = prompt_ids[:cut_idx]
            else:
                eval_prompt_ids = prompt_ids
                
            max_tokens = 35 if task_type == "MARANGOZLUK" else 10
            pred_token_ids = generate_response(model, vocab, eval_prompt_ids, max_new_tokens=max_tokens, device=device)
            pred_tags = [vocab.decode(tid) for tid in pred_token_ids]
            pred_str = " ".join(pred_tags)
            
            decomp_output = decompiler.decompile_sentence(pred_str)
            
            expected_ids = tokenizer.encode(expected_output)
            expected_tags = [vocab.decode(tid) for tid in expected_ids if tid not in (vocab.stoi.get("<BOS>", -1), vocab.stoi.get("<EOS>", -1))]
            expected_tags_str = " ".join(expected_tags)
            
            is_match = (pred_str.strip() == expected_tags_str.strip()) or (expected_tags_str in pred_str)
            
            # Task-specific heuristic matches
            if task_type in ("ÇOĞUL_EKİ", "OLUMSUZLUK", "YETERLİLİK"):
                if "Evet" in expected_output and "evet" in pred_str.lower():
                    is_match = True
                elif "Hayır" in expected_output and "hayır" in pred_str.lower():
                    is_match = True
                    
            if task_type == "KÖK_BULMA":
                exp_root = expected_output.replace("ROOT:", "").strip().lower()
                if exp_root in pred_str.lower():
                    is_match = True
                    
            if task_type == "HÂL_EKİ":
                exp_cases = [tok for tok in expected_tags if tok.startswith("CASE_")]
                if exp_cases and any(c in pred_tags for c in exp_cases):
                    is_match = True

            if task_type == "KİP_ZAMAN":
                exp_tenses = [tok for tok in expected_tags if tok.startswith("TENSE_")]
                if exp_tenses and any(t in pred_tags for t in exp_tenses):
                    is_match = True

            if task_type == "İYELİK_EKİ":
                exp_poss = [tok for tok in expected_tags if tok.startswith("POSS_")]
                if exp_poss and any(p in pred_tags for p in exp_poss):
                    is_match = True

            if task_type == "YAPIM_EKİ":
                exp_deriv = [tok for tok in expected_tags if tok.startswith("DERIV_")]
                if exp_deriv and any(d in pred_tags for d in exp_deriv):
                    is_match = True
                    
            if task_type in ("SENTEZLEME", "SEGMENTASYON"):
                if expected_output.strip().lower() in decomp_output.strip().lower() or expected_tags_str in pred_str:
                    is_match = True

            if task_type == "MARANGOZLUK":
                decomp_lower = decomp_output.lower()
                exp_lower = expected_output.lower()
                carpentry_terms = ["kullanılmalıdır", "uygundur", "ahşap", "ağaç", "rende", "kumpas", "tutkal", "gürgen", "meşe", "yağ", "zıvana", "kırlangıç", "cila", "kereste", "nem", "lif", "kurutma", "kaplama", "fırın", "çatlak", "tabak", "parça", "halka", "teknik", "çözüm", "rapor", "usta", "eğe", "iskarpela", "gönye", "planya", "işkence", "freze"]
                has_domain_term = any(w in decomp_lower for w in carpentry_terms)
                exp_words = [w.strip(".,;:\"'!?") for w in exp_lower.split() if len(w) >= 4]
                shared_words = [w for w in exp_words if w in decomp_lower]
                if (has_domain_term and len(shared_words) >= 1) or len(shared_words) >= 2 or ("kullanılmalıdır" in decomp_lower and any(w in decomp_lower for w in exp_words)):
                    is_match = True


            if is_match:
                exact_matches += 1
                task_stats[task_type]["correct"] += 1
                
            if idx < 10:
                results_to_print.append({
                    "task": task_type,
                    "input": inp,
                    "expected": expected_output,
                    "pred_morphemes": pred_str,
                    "decompiled": decomp_output,
                    "is_correct": is_match
                })
                
        print(f"  -> Test Edilen Rastgele Örnek Sayısı: {sample_size}")
        print(f"  -> Genel SFT Doğruluğu: %{ (exact_matches / sample_size) * 100:.2f} ({exact_matches}/{sample_size})")
        print("\n  Görev Bazlı Başarı Oranları:")
        for t_name, stats in task_stats.items():
            acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"    * {t_name:<12}: %{acc:6.2f} ({stats['correct']}/{stats['total']})")
            
        print("\n" + "=" * 70)
        print(" ÖRNEK GİRDİ - BEKLENEN VE MODEL ÇIKTILARI KARŞILAŞTIRMASI")
        print("=" * 70)
        for r in results_to_print:
            status_icon = "✅" if r["is_correct"] else "❌"
            print(f"{status_icon} [{r['task']}] Kelime: '{r['input']}'")
            print(f"   Beklenen Çıktı:   {r['expected']}")
            print(f"   Model Morfemleri: {r['pred_morphemes']}")
            print(f"   Decompile Çıktı:  {r['decompiled']}")
            print("-" * 50)

    print("\n" + "=" * 70)
    print(" TEST VE DEĞERLENDİRME SÜRECİ TAMAMLANDI.")
    print("=" * 70)

if __name__ == '__main__':
    main()
