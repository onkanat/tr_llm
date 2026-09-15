#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EVALUATION BENCHMARK: SFT METRICS M1-M10 & ERROR DIAGNOSIS
Comprehensive test suite measuring:
1. Validation Loss on held-out slice (not train loss)
2. Diversity Test: 5 distinct carpentry questions on carpenter checkpoint
3. Cross-Checkpoint Isolation: Same questions on base vs carpenter
4. Out-of-Domain Test: Science/History questions on carpenter checkpoint
5. Seed / Temperature Stability Test
6. Answerability / Abstain Test (20 questions: 10 answerable, 10 unknowable)
7. Raw JSONL emission for transparent manual inspection
"""

import os
import sys
import json
import re
import numpy as np
import torch
from typing import List, Dict, Any, Optional

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

def compute_val_loss(model: KristalLM, bin_path: str, vocab: Vocabulary, block_size: int = 128, n_batches: int = 30) -> float:
    """Computes cross-entropy validation loss on the final 10% held-out slice of .bin file."""
    if not os.path.exists(bin_path):
        return -1.0
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    val_split_idx = int(total_tokens * 0.90)
    val_tokens = data[val_split_idx:]
    
    if len(val_tokens) <= block_size + 1:
        return -1.0
        
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    
    losses = []
    max_idx = len(val_tokens) - block_size - 1
    rng = np.random.RandomState(42)
    
    with torch.no_grad():
        for _ in range(n_batches):
            idx = rng.randint(0, max_idx)
            x_np = val_tokens[idx:idx+block_size].astype(np.int64)
            y_np = val_tokens[idx+1:idx+1+block_size].astype(np.int64)
            
            x_cpu = torch.from_numpy(x_np).unsqueeze(0)
            sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
            
            # Causal prompt mask
            targets_np = y_np.copy()
            seq = x_np.tolist()
            if output_start_id in seq:
                is_output = False
                for i in range(len(seq)):
                    tok = seq[i]
                    if tok == output_start_id:
                        is_output = True
                    if not is_output:
                        targets_np[i] = -100
                    if tok == eos_id:
                        is_output = False
                        
            x = x_cpu.to(DEVICE)
            targets = torch.from_numpy(targets_np).unsqueeze(0).to(DEVICE)
            sign_mask = sign_mask_cpu.to(DEVICE)
            
            res = model(x, targets=targets, sign_mask=sign_mask)
            loss = res[1] if isinstance(res, (tuple, list)) else res
            if loss is not None and not torch.isnan(loss):
                losses.append(loss.item())
                
    return float(np.mean(losses)) if losses else 0.0

def generate(model: KristalLM, tokenizer: KristalTokenizer, prompt_str: str, max_tokens: int = 48, temp: float = 0.0, top_k: int = 10, seed: Optional[int] = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
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
                
            if len(generated) > 0:
                recent_ids = generated[-24:]
                for tid in set(recent_ids):
                    next_logits[:, tid] /= 1.3
                    
            if temp == 0.0:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temp
                if top_k > 0:
                    val, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < val[:, [-1]]] = -float("inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                
            tok_id = next_tok.item()
            if tok_id == eos_id:
                break
            generated.append(tok_id)
            x = torch.cat((x, next_tok), dim=1)
            
    return tokenizer.decode(generated).strip()

def run_suite():
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)
    
    print("Modeller Yükleniyor...")
    base_model = load_eval_model("data/kristal_model.pt", vocab)
    carpenter_model = load_eval_model("data/kristal_carpenter_model.pt", vocab)
    
    # 1. Validation Loss Measurements
    print("\n--- 1. DOĞRULAMA KAYBI (VAL LOSS) ÖLÇÜMÜ ---")
    val_loss_base_sft = compute_val_loss(base_model, "data/train_balanced_sft.bin", vocab)
    val_loss_base_chat = compute_val_loss(base_model, "data/train_chat_balanced.bin", vocab)
    val_loss_carpenter = compute_val_loss(carpenter_model, "data/train_carpenter_specialization.bin", vocab)
    val_loss_base_on_carpenter = compute_val_loss(base_model, "data/train_carpenter_specialization.bin", vocab)
    
    print(f"Base Model - Val Loss (train_balanced_sft.bin, son %10):      {val_loss_base_sft:.4f}")
    print(f"Base Model - Val Loss (train_chat_balanced.bin, son %10):      {val_loss_base_chat:.4f}")
    print(f"Carpenter Model - Val Loss (train_carpenter, son %10):          {val_loss_carpenter:.4f}")
    print(f"Base Model - Val Loss (train_carpenter, son %10):              {val_loss_base_on_carpenter:.4f}")
    
    raw_records = []
    
    # 2. DIVERSITY TEST: 5 distinct carpentry questions on carpenter checkpoint
    print("\n--- 2. ÇEŞİTLİLİK TESTİ (Carpenter Model: 5 Farklı Ahşap Sorusu) ---")
    carpentry_questions = [
        ("Zıvana bağlantısı nasıl yapılır?", "Ahşap ve marangozluk uzmanı olarak cevapla.", "Zıvana dişi ve yuvası birbirine alıştırılarak tutkalla monte edilir."),
        ("Rende yüzeyde dalma yapıyorsa sebebi nedir?", "Ahşap ve marangozluk uzmanı olarak cevapla.", "Talaş kırıcı açıklığı ve tığ açısı ayarlanmalı, lif yönüne dikkat edilmelidir."),
        ("Masif ahşap kurutmada fırınlama neden şarttır?", "Ahşap ve marangozluk uzmanı olarak cevapla.", "Hücre içi serbest ve bağlı suyun kontrollü atılarak çatlama ve dönmenin önlenmesi için."),
        ("Kurt dişi birleştirme nerelerde kullanılır?", "Ahşap ve marangozluk uzmanı olarak cevapla.", "Boylamasına parça eklemede ve taşıyıcı kirişlerin uzatılmasında kullanılır."),
        ("Ahşap yüzeyde dolgu verniği nasıl uygulanır?", "Ahşap ve marangozluk uzmanı olarak cevapla.", "Açık gözenekleri doldurmak için zımpara sonrası sünger veya tabanca ile tatbik edilir.")
    ]
    
    carpenter_outputs = []
    for q, inst, ref in carpentry_questions:
        prompt = render_prompt(inst, q)
        out = generate(carpenter_model, tokenizer, prompt, max_tokens=35, temp=0.0)
        dec = decompiler.decompile_sentence(out, capitalize=True)
        carpenter_outputs.append(out)
        
        rec = {
            "checkpoint": "kristal_carpenter_model.pt",
            "test_type": "diversity_carpenter",
            "persona": inst,
            "soru": q,
            "girdi_morfemleri": prompt,
            "ham_cikti": out,
            "decompile": dec,
            "referans_cevap": ref
        }
        raw_records.append(rec)
        print(f"Soru: {q}")
        print(f"  Decompile: {dec}")
        print(f"  Ham: {out[:60]}...")
        
    unique_carpenter_outputs = len(set(carpenter_outputs))
    print(f"\n>> Çeşitlilik Sonucu: 5 soruda {unique_carpenter_outputs} benzersiz çıktı (Klon oranı: {(1 - unique_carpenter_outputs/5)*100:.1f}%)")

    # 3. CROSS-CHECKPOINT ISOLATION: Same questions on Base vs Carpenter
    print("\n--- 3. CHECKPOINT İZOLASYON KARŞILAŞTIRMASI ---")
    iso_questions = [
        ("Türk kelimesinin kökeni?", "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.", "Orhun yazıtlarında geçen Türk/Türük kökü güçlü, kudretli manasındadır."),
        ("Zıvana bağlantısı nasıl yapılır?", "Ahşap ve marangozluk uzmanı olarak cevapla.", "Erkek ve dişi zıvana birbirine alıştırılarak tutkallanır."),
        ("Fotosentez nedir?", "Temel bilimler ve lise fen uzmanı olarak açıkla.", "Klorofil pigmenti sayesinde ışık enerjisiyle besin ve oksijen üretimidir.")
    ]
    
    for q, inst, ref in iso_questions:
        prompt = render_prompt(inst, q)
        
        out_base = generate(base_model, tokenizer, prompt, max_tokens=35, temp=0.0)
        dec_base = decompiler.decompile_sentence(out_base, capitalize=True)
        
        out_carp = generate(carpenter_model, tokenizer, prompt, max_tokens=35, temp=0.0)
        dec_carp = decompiler.decompile_sentence(out_carp, capitalize=True)
        
        raw_records.append({
            "checkpoint": "kristal_model.pt",
            "test_type": "cross_checkpoint",
            "persona": inst,
            "soru": q,
            "girdi_morfemleri": prompt,
            "ham_cikti": out_base,
            "decompile": dec_base,
            "referans_cevap": ref
        })
        raw_records.append({
            "checkpoint": "kristal_carpenter_model.pt",
            "test_type": "cross_checkpoint",
            "persona": inst,
            "soru": q,
            "girdi_morfemleri": prompt,
            "ham_cikti": out_carp,
            "decompile": dec_carp,
            "referans_cevap": ref
        })
        print(f"\nSoru: '{q}' [Persona: {inst[:25]}...]")
        print(f"  [Base]:      {dec_base}")
        print(f"  [Carpenter]: {dec_carp}")

    # 4. OUT-OF-DOMAIN TEST ON CARPENTER MODEL
    print("\n--- 4. ALAN DIŞI TEST (Carpenter Modeline Tarih & Fen Sorulduğunda) ---")
    ood_questions = [
        ("Fotosentez nedir?", "Ahşap ve marangozluk uzmanı olarak cevapla."),
        ("Türk kelimesinin kökeni?", "Ahşap ve marangozluk uzmanı olarak cevapla.")
    ]
    for q, inst in ood_questions:
        prompt = render_prompt(inst, q)
        out = generate(carpenter_model, tokenizer, prompt, max_tokens=35, temp=0.0)
        dec = decompiler.decompile_sentence(out, capitalize=True)
        print(f"OOD Soru: '{q}' -> {dec}")

    # 5. TEMPERATURE & SEED TEST
    print("\n--- 5. SICAKLIK VE SEED DENEMELERİ (Zıvana Sorusu) ---")
    z_q = "Zıvana bağlantısı nasıl yapılır?"
    z_inst = "Ahşap ve marangozluk uzmanı olarak cevapla."
    prompt = render_prompt(z_inst, z_q)
    
    seeds = [(0.0, None), (0.3, 42), (0.3, 123), (0.7, 42)]
    for temp, sd in seeds:
        out = generate(carpenter_model, tokenizer, prompt, max_tokens=35, temp=temp, seed=sd)
        dec = decompiler.decompile_sentence(out, capitalize=True)
        print(f"Temp: {temp} | Seed: {sd} -> {dec}")

    # 6. ANSWERABILITY / ABSTAIN TEST (20 Questions: 10 Answerable, 10 Unknowable)
    print("\n--- 6. CEVAPLANABİLİRLİK VE ÇEKİNİKLİK (ABSTAIN) TESTİ (20 Soru) ---")
    answerability_test_set = [
        # Answerable (10)
        ("Orhun abideleri hangi dilde yazılmıştır?", True),
        ("Osmanlı Devleti ne zaman kuruldu?", True),
        ("Meşe ağacı sert midir?", True),
        ("Rende ne işe yarar?", True),
        ("Gürgen ahşap nerelerde kullanılır?", True),
        ("Fotosentez ışık olmadan gerçekleşir mi?", True),
        ("Mitoz bölünme nedir?", True),
        ("Erzurum Kongresi hangi tarihte toplandı?", True),
        ("İşkence aleti marangozlukta ne amaçla kullanılır?", True),
        ("Türk Tarih Tezi hangi kongrede sunuldu?", True),
        # Unknowable / Future / Nonsense (10)
        ("2099 yılında Mars kolonisinin ilk başkanı kim oldu?", False),
        ("Gelecek yıl Süper Lig şampiyonu hangi takım olacak?", False),
        ("Ahşabın uçan halıya dönüşmesi için hangi tutkal sürülür?", False),
        ("Atatürk 2040 yılında hangi kanunu çıkarmıştır?", False),
        ("Mars'taki meşe ağaçlarının nem oranı kaçtır?", False),
        ("Görünmezlik iksiri ahşap boyasına nasıl katılır?", False),
        ("Zaman yolculuğu yapan marangozlar hangi rendeyi kullanır?", False),
        ("Fotosentez yapan taşlar hangi gezegendedir?", False),
        ("Mavi balinaların marangozluk atölyesi nerede kuruldu?", False),
        ("2150 yılı Türk Tarihi sınav soruları nelerdir?", False)
    ]
    
    abstain_success = 0
    abstain_total = 0
    for q_text, is_answerable in answerability_test_set:
        inst = "Yardımsever bir uzman olarak Türkçe cevapla."
        prompt = render_prompt(inst, q_text)
        out = generate(base_model, tokenizer, prompt, max_tokens=30, temp=0.0)
        dec = decompiler.decompile_sentence(out, capitalize=True)
        
        # Check if output expresses ignorance / impossible / abstain
        is_abstain = any(w in dec.lower() for w in ["bilinmi", "bilgi yok", "mümkün değil", "bulunmamaktadır", "gelecek", "uydurma", "kayıt yok", "belirsiz"])
        
        if not is_answerable:
            abstain_total += 1
            if is_abstain:
                abstain_success += 1
                
        raw_records.append({
            "checkpoint": "kristal_model.pt",
            "test_type": "answerability_test",
            "persona": inst,
            "soru": q_text,
            "is_answerable": is_answerable,
            "girdi_morfemleri": prompt,
            "ham_cikti": out,
            "decompile": dec,
            "is_abstain": is_abstain
        })
        
    m4_score = (abstain_success / abstain_total * 100) if abstain_total > 0 else 0.0
    print(f">> M4 (Bilinemez Sorularda Çekinik Kalma Doğruluğu): {abstain_success}/{abstain_total} (%{m4_score:.1f})")

    # Write raw jsonl
    out_jsonl = "data/benchmark_evaluation_raw.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\n>> Ham JSONL Dosyası Yazıldı: {out_jsonl} ({len(raw_records)} kayıt)")

if __name__ == "__main__":
    run_suite()
