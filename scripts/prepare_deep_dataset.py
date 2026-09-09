#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def tokenize_jsonl(filepath: str, tokenizer: KristalTokenizer, max_samples: int = None) -> list:
    records = []
    if not os.path.exists(filepath):
        print(f"Uyarı: '{filepath}' bulunamadı, atlanıyor.")
        return records
        
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
        
    if max_samples and len(lines) > max_samples:
        random.seed(42)
        lines = random.sample(lines, max_samples)
        
    for line in lines:
        try:
            item = json.loads(line)
            raw_prompt = json.dumps(item, ensure_ascii=False)
            token_ids = tokenizer.encode(raw_prompt)
            if len(token_ids) > 2: # Has content
                records.append(token_ids)
        except Exception:
            continue
            
    print(f"  * {os.path.basename(filepath):<36}: {len(records):,} kayıt tokenize edildi.")
    return records

def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DERİNLEŞTİRİLMİŞ EĞİTİM VERİ DERLEME")
    print("=" * 70)

    # 1. Setup Compiler & Tokenizer
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    initial_vocab_size = len(vocab.stoi)
    print(f"Başlangıç Sözlük Boyutu: {initial_vocab_size} token.")

    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    print("\n[1] Alt Veri Kümeleri Yükleniyor ve Tokenize Ediliyor:")
    
    # 2. Tokenize sub-datasets
    # A. Derinleştirilmiş Ebeveynlik (Morfolojik 10 Görev)
    parenting_records = tokenize_jsonl('data/pedagogy/parenting_deep_dataset.jsonl', tokenizer, max_samples=40000)
    
    # B. GTS Sözlük ve Anlamsal Tanımlar
    semantics_records = tokenize_jsonl('data/pedagogy/lexical_semantics_dataset.jsonl', tokenizer, max_samples=25000)
    
    # C. Marangozluk Alan Uzmanlığı (Carpenter AI) - 10x oversample
    carpenter_raw = tokenize_jsonl('data/pedagogy/carpenter_specialization_dataset.jsonl', tokenizer)
    carpenter_records = carpenter_raw * 10
    print(f"  * [Oversampled] Marangozluk Uzmanlığı: {len(carpenter_records):,} kayıt.")
    
    # D. Bebeklik (Kavramsal Sınırlar)
    infancy_records = tokenize_jsonl('data/pedagogy/infancy_dataset.jsonl', tokenizer, max_samples=7500)
    
    # E. Samimi ve Derin Türkçe Sohbet (Chat Conversations & Middle School)
    deep_chat_records = tokenize_jsonl('data/pedagogy/chat_conversations.jsonl', tokenizer, max_samples=6000)
    chat_raw = tokenize_jsonl('data/pedagogy/middle_school_chat.jsonl', tokenizer)
    chat_records = deep_chat_records + (chat_raw * 5)
    print(f"  * Toplam Sohbet (Chat) Kayıt Sayısı: {len(chat_records):,} kayıt.")

    # F. İnteraktif Self-RAG Kullanım Görevleri
    rag_records = tokenize_jsonl('data/pedagogy/rag_interactive_dataset.jsonl', tokenizer, max_samples=6500)

    # 3. Save Vocabulary Update
    vocab.save('data/vocab.json')
    print(f"\n[2] Sözlük Güncellendi: {initial_vocab_size} -> {len(vocab.stoi)} token.")

    # 4. Combine and Shuffle
    all_records = (
        parenting_records + 
        semantics_records + 
        carpenter_records + 
        infancy_records + 
        chat_records + 
        rag_records
    )
    
    print(f"\n[3] Toplam Derinleştirilmiş SFT Kayıt Sayısı: {len(all_records):,}")
    print("Kayıtlar karıştırılıyor (shuffling)...")
    random.seed(42)
    random.shuffle(all_records)

    # 5. Pack and Pad into Binary File (block_size = 128)
    block_size = 128
    pad_id = vocab.stoi.get("<PAD>", 1)
    
    print(f"\n[4] Tokenlar Sabit Blok Boyutuna ({block_size}) Hizalanıyor ve Paketleniyor...")
    flat_tokens = []
    for rec in all_records:
        if len(rec) < block_size:
            padded_rec = rec + [pad_id] * (block_size - len(rec))
        else:
            padded_rec = rec[:block_size]
        flat_tokens.extend(padded_rec)

    output_bin = 'data/train_deep_sft.bin'
    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin)
    
    meta_info = {
        "total_records": len(all_records),
        "total_tokens": len(flat_tokens),
        "block_size": block_size,
        "vocab_size": len(vocab.stoi),
        "dataset_composition": {
            "parenting_deep": len(parenting_records),
            "lexical_semantics": len(semantics_records),
            "carpenter_specialization": len(carpenter_records),
            "infancy": len(infancy_records),
            "chat": len(chat_records),
            "rag": len(rag_records)
        }
    }
    
    meta_path = output_bin + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(" DERİNLEŞTİRİLMİŞ VERİ SETİ DERLEMESİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)
    print(f"Çıktı Binary Dosyası: {output_bin} ({file_size_mb:.2f} MB)")
    print(f"Toplam Token Sayısı : {len(flat_tokens):,}")
    print(f"Meta Veri Dosyası   : {meta_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
