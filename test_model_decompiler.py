#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from src.compiler.decompiler import MorphemeDecompiler

# ANSI Color Codes for pretty output
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_GRAY = "\033[90m"

def color_token(tag: str) -> str:
    if tag in ["<BOS>", "<EOS>", "<PAD>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>"]:
        return f"{C_RED}{C_BOLD}{tag}{C_RESET}"
    if tag == "<UNK>":
        return f"{C_RED}{C_BOLD}{tag}{C_RESET}"
    if tag == "<NUMBER>":
        return f"{C_YELLOW}{tag}{C_RESET}"
    if tag == "<PROPER_NOUN>":
        return f"{C_GREEN}{C_BOLD}{tag}{C_RESET}"
    
    suffix_prefixes = ("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_", "DERIV_")
    if tag.startswith(suffix_prefixes) or tag in ("PLURAL", "POTENTIAL", "NEG", "IMPOTENTIAL_NEG"):
        return f"{C_CYAN}{tag}{C_RESET}"
    
    return f"{C_GREEN}{tag}{C_RESET}"

def generate_tokens(model, vocab, prompt_tokens, max_new_tokens=30, device='cpu', temperature=0.0, repetition_penalty=1.5, repetition_window=10):
    """Generates next tokens autoregressively from prompt_tokens with repetition penalty."""
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
                window_tokens = output_tokens[-repetition_window:]
                for token_id in set(window_tokens):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty
                        
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                pred_id = torch.multinomial(probs, num_samples=1).item()
            else:
                pred_id = torch.argmax(logits).item()
                
            generated.append(pred_id)
            if pred_id == eos_id or pred_id == output_end_id:
                break
    return generated

def main():
    print(f"{C_MAGENTA}{C_BOLD}" + "=" * 70)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: MODEL ÇIKTISINI YENİDEN YAPILANDIRMA TESTİ")
    print("=" * 70 + f"{C_RESET}")

    # 1. Device Setup
    device_name = "cpu"
    if "--mps" in sys.argv and torch.backends.mps.is_available():
        device_name = "mps"
    elif "--device" in sys.argv:
        device_name = sys.argv[sys.argv.index("--device") + 1]
    device = torch.device(device_name)
    print(f"Cihaz:      {C_CYAN}{device}{C_RESET}", flush=True)

    # 2. Load Vocabulary
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    if not os.path.exists(vocab_path):
        print(f"{C_RED}Hata: {vocab_path} bulunamadı!{C_RESET}")
        return
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük:     {C_CYAN}{vocab_size} morfem{C_RESET}")

    # 3. Load Lexicon, Graph & Compiler
    lexicon = LexiconManager()
    lexicon_path = 'data/lexicon/roots.tsv'
    lexicon.load_from_tsv(lexicon_path)
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)

    # 4. Load Model
    model_path = 'data/kristal_model.pt'
    if not os.path.exists(model_path):
        print(f"{C_RED}Hata: Model dosyası '{model_path}' bulunamadı!{C_RESET}")
        return
        
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    state_dict = torch.load(model_path, map_location=device)
    # Filter out mismatching rotary embed buffers
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model:      {C_CYAN}{model_path} (Yüklendi){C_RESET}")
    print("=" * 70)

    # 5. Part 1: Normal Generation & Decompilation
    print(f"\n{C_BOLD}--- BÖLÜM 1: OTOMATİK CÜMLE TAMAMLAMA VE DECOMPİLATİON ---{C_RESET}")
    
    completion_prompts = [
        "Akmayan su kımıldanmayan",
        "Yarın okula gideceğim",
        "Durgun su ölümü",
        "Kitabı okudum ve temizledim"
    ]
    
    for prompt in completion_prompts:
        print(f"\n{C_BLUE}Girdi Metni:    {C_RESET}'{prompt}'")
        
        # Tokenize prompt
        token_ids = tokenizer.encode(prompt)
        # Strip trailing EOS if present for prompt continuation
        if token_ids and token_ids[-1] == vocab.stoi.get("<EOS>", -1):
            prompt_tokens = token_ids[:-1]
        else:
            prompt_tokens = token_ids
            
        decoded_prompt_morphemes = [vocab.decode(tid) for tid in prompt_tokens]
        print(f"Girdi Morfem:   " + " ".join([color_token(t) for t in decoded_prompt_morphemes]))
        
        # Generate completion
        generated_ids = generate_tokens(model, vocab, prompt_tokens, max_new_tokens=20, device=device, temperature=0.0)
        
        # Decode whole sequence
        decoded_output_morphemes = [vocab.decode(tid) for tid in generated_ids]
        print(f"Model Çıktı:    " + " ".join([color_token(t) for t in decoded_output_morphemes]))
        
        # Decompile the whole sentence
        crystal_tags_str = " ".join(decoded_output_morphemes)
        decompiled_sentence = decompiler.decompile_sentence(crystal_tags_str)
        print(f"{C_GREEN}Decompile Edilmiş Cümle: {C_RESET}{C_BOLD}{decompiled_sentence}{C_RESET}")

    # 6. Part 2: SFT Task Generation & Decompilation
    print(f"\n{C_BOLD}--- BÖLÜM 2: SFT GÖREVLERİ TAHMİNİ VE DECOMPİLATİON ---{C_RESET}")
    
    sft_prompts = [
        {"instruction": "Kelimedeki kök morfemini bul.", "input": "kitaplarda", "output": ""},
        {"instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.", "input": "okula", "output": ""},
        {"instruction": "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.", "input": "çocuklar", "output": ""},
        {"instruction": "Kelimedeki eylemin zamanını veya kipini tespit et.", "input": "gideceğim", "output": ""}
    ]
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    
    for case in sft_prompts:
        print(f"\n{C_BLUE}Görev:          {C_RESET}'{case['instruction']}' | {C_BLUE}Kelime: {C_RESET}'{case['input']}'")
        
        # Encode JSON SFT prompt
        raw_prompt = json.dumps(case)
        token_ids = tokenizer.encode(raw_prompt)
        
        # Slice up to <OUTPUT>
        if output_start_id in token_ids:
            output_idx = token_ids.index(output_start_id)
            prompt_tokens = token_ids[:output_idx + 1]
        else:
            prompt_tokens = token_ids
            
        # Generate output tokens
        generated_ids = generate_tokens(model, vocab, prompt_tokens, max_new_tokens=15, device=device, temperature=0.0)
        
        # Extract the model's response after <OUTPUT>
        response_tokens = generated_ids[len(prompt_tokens):]
        response_morphemes = [vocab.decode(tid) for tid in response_tokens]
        print(f"Model Cevabı:   " + " ".join([color_token(t) for t in response_morphemes]))
        
        # Decompile the answer if it contains linguistic morphemes
        response_tags_str = " ".join(response_morphemes)
        decompiled_response = decompiler.decompile_sentence(response_tags_str)
        if decompiled_response.strip():
            print(f"{C_GREEN}Decompile Edilmiş Cevap: {C_RESET}{C_BOLD}{decompiled_response}{C_RESET}")
        else:
            # If no surface forms could be reconstructed (e.g. just raw tags or text), show cleaned string
            clean_response = response_tags_str.replace("<EOS>", "").replace("</OUTPUT>", "").strip()
            print(f"{C_GREEN}Metinsel Model Cevabı:   {C_RESET}{C_BOLD}{clean_response}{C_RESET}")

    print("\n" + "=" * 70)
    print(" DECOMPİLATİON MODEL TEST SÜRECİ TAMAMLANDI")
    print("=" * 70)

if __name__ == '__main__':
    main()
