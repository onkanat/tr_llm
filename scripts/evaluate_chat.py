#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: SOHBET (CHAT) ÇIKARIM VE DOĞRULAMA TESTİ
=================================================================
Bu betik, eğitilmiş KristalLM modelinin Türkçe sohbet yeteneklerini
çeşitli diyalog senaryolarında test eder. Üretilen morfem dizilerini
MorphemeDecompiler ile doğal Türkçe yüzey biçimine dönüştürür.
"""

import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM

def resize_state_dict(model, old_state_dict):
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

def generate_chat_response(model, tokenizer, decompiler, prompt_text: str, instruction: str = "", max_tokens: int = 30, device: str = 'cpu') -> dict:
    vocab = tokenizer.vocab
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    output_end_id = vocab.stoi.get("</OUTPUT>", -1)
    
    if not instruction:
        instruction = "Kullanıcı ile doğal ve samimi bir Türkçe diyalog yürüt."
        
    prompt_obj = {
        "instruction": instruction,
        "input": prompt_text,
        "output": ""
    }
    prompt_str = json.dumps(prompt_obj, ensure_ascii=False)
    prompt_ids = tokenizer.encode(prompt_str)
    
    if output_start_id in prompt_ids:
        cut_idx = prompt_ids.index(output_start_id) + 1
        eval_prompt_ids = prompt_ids[:cut_idx]
    else:
        eval_prompt_ids = prompt_ids
        
    generated = list(eval_prompt_ids)
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :]
            
            # Repetition penalty on recently generated tokens
            recent = generated[len(eval_prompt_ids):][-10:]
            for tok_id in set(recent):
                if logits[tok_id] > 0:
                    logits[tok_id] /= 1.4
                else:
                    logits[tok_id] *= 1.4
                    
            pred_id = torch.argmax(logits).item()
            generated.append(pred_id)
            if pred_id in (eos_id, output_end_id):
                break
                
    response_tokens = generated[len(eval_prompt_ids):]
    clean_tokens = [tid for tid in response_tokens if tid not in (eos_id, output_end_id)]
    
    pred_tags = [vocab.decode(tid) for tid in clean_tokens]
    pred_str = " ".join(pred_tags)
    decompiled_text = decompiler.decompile_sentence(pred_str)
    
    return {
        "input": prompt_text,
        "instruction": instruction,
        "raw_tags": pred_tags,
        "morpheme_str": pred_str,
        "decompiled": decompiled_text
    }

def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL MODEL: TÜRKÇE SOHBET DOĞRULAMA TESTİ")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"Cihaz: {device}")

    # 1. Vocab & Compiler Setup
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    print(f"Sözlük Yüklendi. Boyut: {len(vocab.stoi)} token.")

    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)

    # 2. Model Loading
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    model_path = 'data/kristal_model.pt'
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası '{model_path}' bulunamadı!")
        return

    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
    state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model '{model_path}' adresinden başarıyla yüklendi.\n")

    # 3. Test Dialogue Scenarios
    test_queries = [
        # Selamlaşma
        ("Merhaba!", "Kullanıcı ile doğal, kibar ve samimi bir Türkçe diyalog yürüt."),
        ("Selam dostum nasılsın?", "Kullanıcı ile doğal, kibar ve samimi bir Türkçe diyalog yürüt."),
        ("Günaydın!", "Kullanıcı ile doğal, kibar ve samimi bir Türkçe diyalog yürüt."),
        
        # Asistan & Rehberlik
        ("Bana yardımcı olabilir misin?", "Kullanıcıya yardımcı, yapıcı ve rehberlik eden bir asistan olarak cevap ver."),
        ("Sen ne yapabilirsin?", "Kullanıcıya yardımcı, yapıcı ve rehberlik eden bir asistan olarak cevap ver."),
        ("Bana bir kitap önerir misin?", "Kullanıcıya yardımcı, yapıcı ve rehberlik eden bir asistan olarak cevap ver."),
        
        # Empati & Yaşam
        ("Bugün çok yoruldum.", "Kullanıcıya anlayışlı, empatik ve destekleyici bir dille yanıt ver."),
        ("Sınavımdan çok yüksek not aldım!", "Kullanıcıya içten ve motive edici bir dille karşılık ver."),
        
        # Merak & Bilim
        ("Ağaçlar kışın neden yaprak döker?", "Soruyu açık, anlaşılır ve eğitici bir Türkçe ile yanıtla."),
        ("Gökyüzü neden mavidir?", "Soruyu açık, anlaşılır ve eğitici bir Türkçe ile yanıtla."),
        
        # Nezaket & Kapanış
        ("Çok teşekkür ederim, çok yardımcı oldun.", "Kullanıcıya nazik, sıcak ve saygılı bir dille karşılık ver."),
        ("Görüşmek üzere, hoşça kal!", "Kullanıcıya nazik, sıcak ve saygılı bir dille karşılık ver.")
    ]

    print("-" * 70)
    print(" SOHBET DİYALOGLARI VE MODEL CEVAPLARI")
    print("-" * 70)

    for i, (query, inst) in enumerate(test_queries, 1):
        res = generate_chat_response(model, tokenizer, decompiler, query, instruction=inst, max_tokens=25, device=device)
        print(f"\n[Diyalog #{i}]")
        print(f"  👤 Kullanıcı:  {query}")
        print(f"  🧠 Morfemler:  {res['morpheme_str']}")
        print(f"  🤖 Türkçe:     {res['decompiled']}")

    print("\n" + "=" * 70)
    print(" SOHBET DEĞERLENDİRME TESTİ TAMAMLANDI.")
    print("=" * 70)

if __name__ == '__main__':
    main()
