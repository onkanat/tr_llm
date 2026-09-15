#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPERIMENT A: INSTRUCTION CHANNEL ISOLATION
Fixed checkpoint: data/kristal_carpenter_model.pt
Fixed question: 'Zıvana bağlantısı nasıl yapılır?'
5 distinct instructions to test if output shifts with instruction.
Includes bug fixes for repetition penalty (sign-aware) and max_tokens=128.
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from src.compiler.decompiler import MorphemeDecompiler
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

def generate(model: KristalLM, tokenizer: KristalTokenizer, prompt_str: str, max_tokens: int = 128, temp: float = 0.0) -> str:
    input_ids = tokenizer.encode(prompt_str)
    eos_id = tokenizer.vocab.stoi.get("<EOS>", -1)
    if input_ids and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]
        
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    masked_tags = [
        "<PAD>", "<BOS>", "<INSTRUCTION>", "</INSTRUCTION>",
        "<INPUT>", "</INPUT>", "<OUTPUT>", "<PROPER_NOUN>", "<UNK>", "<NUMBER>"
    ]
    forbidden_ids = set()
    for tag in masked_tags:
        if tag in tokenizer.vocab.stoi:
            forbidden_ids.add(tokenizer.vocab.stoi[tag])
            
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= 256 else x[:, -256:]
            sign_mask = model.embedding.compute_sign_mask(x_cond).to(DEVICE)
            res = model(x_cond, sign_mask=sign_mask)
            logits = res[0] if isinstance(res, (tuple, list)) else res
            next_logits = logits[:, -1, :].clone()
            
            for fid in forbidden_ids:
                next_logits[:, fid] = -float("inf")
                
            # Sign-aware repetition penalty
            if len(generated) > 0:
                recent_ids = generated[-32:]
                for tid in set(recent_ids):
                    val = next_logits[0, tid].item()
                    if val > 0:
                        next_logits[0, tid] /= 1.3
                    else:
                        next_logits[0, tid] *= 1.3
                    
            if temp == 0.0:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temp
                val, _ = torch.topk(next_logits, min(10, next_logits.size(-1)))
                next_logits[next_logits < val[:, [-1]]] = -float("inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                
            tok_id = next_tok.item()
            if tok_id == eos_id:
                break
            generated.append(tok_id)
            x = torch.cat((x, next_tok), dim=1)
            
    return tokenizer.decode(generated).strip()

def run_exp_a():
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)
    
    ckpt_path = "data/kristal_carpenter_model.pt"
    model = load_eval_model(ckpt_path, vocab)
    
    question = "Zıvana bağlantısı nasıl yapılır?"
    q_tok = tokenizer.decode(tokenizer.encode(question)).replace("<BOS>", "").replace("<EOS>", "").strip()
    
    instructions = [
        ("1. Ahsap_Uzmani", "Ahşap ve marangozluk uzmanı olarak cevapla."),
        ("2. Fen_Uzmani", "Temel bilimler uzmanı olarak açıkla."),
        ("3. Tarih_Uzmani", "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla."),
        ("4. Bos_Talimat", ""),
        ("5. Anlamsiz_Talimat", "Fransızca cevapla.")
    ]
    
    print("=" * 80)
    print(f"DENEY A: TALİMAT KANALI İZOLASYONU")
    print(f"Model: {ckpt_path}")
    print(f"Sabit Soru: '{question}' (Morfemler: {q_tok})")
    print("=" * 80)
    
    outputs = []
    for label, inst in instructions:
        prompt = render_prompt(inst, question)
            
        raw_out = generate(model, tokenizer, prompt, max_tokens=128, temp=0.0)
        dec = decompiler.decompile_sentence(raw_out, capitalize=True)
        outputs.append((label, prompt, raw_out, dec))
        
        print(f"\n[{label}] (Talimat: '{inst}')")
        print(f"  Prompt:    {prompt}")
        print(f"  Decompile: {dec}")
        print(f"  Ham Çıktı: {raw_out[:90]}...")
        
    print("\n" + "=" * 80)
    print("DENEY A ÇIKTI KARŞILAŞTIRMASI:")
    print("=" * 80)
    for label, _, raw_out, dec in outputs:
        print(f"  * {label:22}: {dec[:75]}...")

if __name__ == "__main__":
    run_exp_a()
