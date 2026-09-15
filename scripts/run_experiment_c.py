#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPERIMENT C: CHANNEL-ALIGNED INFERENCE (C1 MEMORIZATION vs C2 GENERALIZATION)
Uses the EXACT training schema: Question in INSTRUCTION, INPUT is EMPTY.
Model: data/kristal_carpenter_model.pt (frozen, original weights)

C1: Exact training instruction strings from carpenter_specialization_dataset.jsonl
C2: Paraphrased instruction strings (never seen in training)
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

def resize_state_dict(model: torch.nn.Module, old_state_dict: dict) -> dict:
    new_state_dict = model.state_dict()
    for k, v in old_state_dict.items():
        if k in new_state_dict:
            if v.shape != new_state_dict[k].shape:
                if len(v.shape) == 2:
                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])] = v[:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])]
                elif len(v.shape) == 1:
                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0])] = v[:min(v.shape[0], new_state_dict[k].shape[0])]
            else:
                new_state_dict[k] = v
    return new_state_dict

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

def run_exp_c():
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)
    
    ckpt_path = "data/kristal_carpenter_model.pt"
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    state = torch.load(ckpt_path, map_location="cpu")
    new_sd = resize_state_dict(model, state)
    model.load_state_dict(new_sd)
    model.to(DEVICE)
    model.eval()
    
    # -------------------------------------------------------------
    # C1: EXACT TRAINING PROMPTS (Soru INSTRUCTION içinde, INPUT boş)
    # -------------------------------------------------------------
    c1_cases = [
        (
            "C1_1_Dolgu_Vernigi",
            "Ahşap atölyesi güvenlik ve imalat rehberi: Ahşabın son kat yüzeyinde dolgu verniği nasıl tatbik edilmelidir?",
            "Dolgu verniği uygulamasında ahşabın açık gözeneklerini doldurarak son kat cila için cam gibi pürüzsüz ve homojen bir alt zemin hazırlayan astar verniktir."
        ),
        (
            "C1_2_Kurt_Disi",
            "Geleneksel ahşap zanaatı sorusu: Usta marangoz olarak kurt dişi birleştirme geçme tekniğinin püf noktalarını açıkla.",
            "Kurt dişi birleştirme uygulamasında açılı zikzak dişlerle ahşap parçaların boylamasına eklenerek sonsuz kereste elde edilmesini sağlayan endüstriyel eklemedir."
        ),
        (
            "C1_3_Kayin_Agaci",
            "Marangozluk alan uzmanlığı: Marangozlukta kayın ağacının özellikleri ve kullanım alanları nelerdir?",
            "Kayın ağacı: Oldukça sert ve sıkı liflidir; buharlama yöntemiyle kolayca bükülerek Thonet tarzı sandalye ve bükme mobilyalarda idealdir."
        )
    ]
    
    print("=" * 80)
    print("DENEY C1: EZBER TESTİ (Birebir Eğitimdeki İstemler, INPUT Boş)")
    print("=" * 80)
    
    c1_results = []
    for cid, inst_text, expected in c1_cases:
        i_tok = tokenizer.decode(tokenizer.encode(inst_text)).replace("<BOS>", "").replace("<EOS>", "").strip()
        # Prompt exactly as trained: question in INSTRUCTION, empty INPUT
        prompt = f"<BOS> <INSTRUCTION> {i_tok} </INSTRUCTION> <INPUT> </INPUT> <OUTPUT>"
        raw_out = generate(model, tokenizer, prompt, max_tokens=128, temp=0.0)
        dec = decompiler.decompile_sentence(raw_out, capitalize=True)
        c1_results.append((cid, inst_text, expected, dec, raw_out))
        
        print(f"\n[{cid}]")
        print(f"  İstem (INSTRUCTION): {inst_text}")
        print(f"  Beklenen Cevap:      {expected}")
        print(f"  Model Çıktısı:       {dec}")
        print(f"  Ham Çıktı:           {raw_out[:90]}...")
        
    # -------------------------------------------------------------
    # C2: PARAPHRASED PROMPTS (Eğitimde Geçmeyen Parafrazlar, INPUT Boş)
    # -------------------------------------------------------------
    c2_cases = [
        (
            "C2_1_Dolgu_Vernigi_Parafraz",
            "Ahşap atölyesi güvenlik ve imalat rehberi: Masif ahşapta dolgu verniği uygulaması nasıl yapılır?",
            "Dolgu verniği uygulamasında ahşabın açık gözeneklerini doldurarak son kat cila için..."
        ),
        (
            "C2_2_Kurt_Disi_Parafraz",
            "Geleneksel ahşap zanaatı sorusu: Ahşapta kurt dişi geçme nasıl yapılır ve nelere dikkat edilmelidir?",
            "Kurt dişi birleştirme uygulamasında açılı zikzak dişlerle ahşap parçaların boylamasına..."
        ),
        (
            "C2_3_Kayin_Agaci_Parafraz",
            "Marangozluk alan uzmanlığı: Mobilyada kayın ağacının genel özellikleri ve kullanım yerleri nelerdir?",
            "Kayın ağacı: Oldukça sert ve sıkı liflidir..."
        )
    ]
    
    print("\n" + "=" * 80)
    print("DENEY C2: GENELLEME TESTİ (Parafraz İstemler, INPUT Boş)")
    print("=" * 80)
    
    c2_results = []
    for cid, inst_text, expected in c2_cases:
        i_tok = tokenizer.decode(tokenizer.encode(inst_text)).replace("<BOS>", "").replace("<EOS>", "").strip()
        prompt = f"<BOS> <INSTRUCTION> {i_tok} </INSTRUCTION> <INPUT> </INPUT> <OUTPUT>"
        raw_out = generate(model, tokenizer, prompt, max_tokens=128, temp=0.0)
        dec = decompiler.decompile_sentence(raw_out, capitalize=True)
        c2_results.append((cid, inst_text, expected, dec, raw_out))
        
        print(f"\n[{cid}]")
        print(f"  Parafraz İstem:      {inst_text}")
        print(f"  Beklenen Cevap:      {expected}")
        print(f"  Model Çıktısı:       {dec}")
        print(f"  Ham Çıktı:           {raw_out[:90]}...")
        
    print("\n" + "=" * 80)
    print("DENEY C ÖZET KARŞILAŞTIRMASI:")
    print("=" * 80)
    print("C1 (Ezber):")
    for cid, _, _, dec, _ in c1_results:
        print(f"  * {cid:25}: {dec[:75]}...")
    print("\nC2 (Genelleme):")
    for cid, _, _, dec, _ in c2_results:
        print(f"  * {cid:25}: {dec[:75]}...")

if __name__ == "__main__":
    run_exp_c()
