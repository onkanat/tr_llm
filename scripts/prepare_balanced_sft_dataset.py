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

def main():
    print("=" * 60)
    print(" DENGELİ SFT VERİ SETİ HAZIRLAMA (DENGELİ VERİ KARIŞTIRMA)")
    print("=" * 60)

    # 1. Initialize Vocabulary and Tokenizer
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    print(f"Sözlük yüklendi: {len(vocab.stoi)} token.")

    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Load Scientific DPO Chosen tokens
    scientific_bin_path = 'data/train_all_chosen.bin'
    if not os.path.exists(scientific_bin_path):
        print(f"Hata: {scientific_bin_path} bulunamadı!")
        return
    
    print("Bilimsel SFT tokenleri yükleniyor...")
    scientific_tokens = np.fromfile(scientific_bin_path, dtype=np.uint16)
    print(f"  -> Bilimsel SFT token sayısı: {len(scientific_tokens)}")

    # Split scientific tokens back into records (using BOS as separator)
    # BOS is 2, EOS is 3
    scientific_records = []
    current_record = []
    for token in scientific_tokens:
        if token == 2 and current_record:
            scientific_records.append(current_record)
            current_record = []
        current_record.append(int(token))
    if current_record:
        scientific_records.append(current_record)
    print(f"  -> Bölünen bilimsel kayıt sayısı: {len(scientific_records)}")

    # 3. Load and Tokenize Parenting SFT tasks
    parenting_path = 'data/pedagogy/parenting_dataset.jsonl'
    if not os.path.exists(parenting_path):
        print(f"Hata: {parenting_path} bulunamadı!")
        return

    print("Ebeveynlik (Parenting SFT) görevleri yükleniyor ve tokenize ediliyor...")
    parenting_records = []
    with open(parenting_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Render into SFT structured format and encode
                raw_prompt = json.dumps(item)
                token_ids = tokenizer.encode(raw_prompt)
                parenting_records.append(token_ids)
    print(f"  -> Benzersiz ebeveynlik kayıt sayısı: {len(parenting_records)}")

    # 4. Oversample Parenting SFT tasks to balance the dataset
    oversample_factor = 10
    oversampled_parenting = parenting_records * oversample_factor
    print(f"  -> Oversampling sonrası ebeveynlik kayıt sayısı: {len(oversampled_parenting)}")

    # 5. Load and Tokenize Turk Tarihi SFT tasks
    history_sft_path = 'data/pedagogy/turk_tarihi_sft.jsonl'
    history_sft_records = []
    if os.path.exists(history_sft_path):
        print("Tarih SFT görevleri yükleniyor ve tokenize ediliyor...")
        with open(history_sft_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        raw_prompt = json.dumps(item, ensure_ascii=False)
                        token_ids = tokenizer.encode(raw_prompt)
                        if len(token_ids) > 2:
                            history_sft_records.append(token_ids)
                    except Exception:
                        pass
        print(f"  -> Benzersiz tarih SFT kayıt sayısı: {len(history_sft_records)}")

    # 6. Combine and Shuffle
    all_records = scientific_records + oversampled_parenting + history_sft_records
    print(f"  -> Toplam birleşik kayıt sayısı: {len(all_records)}")
    
    print("Kayıtlar karıştırılıyor (shuffling)...")
    random.seed(42)
    random.shuffle(all_records)

    # 6. Flatten and save to binary file
    flat_tokens = []
    for rec in all_records:
        flat_tokens.extend(rec)

    output_bin_path = 'data/train_balanced_sft.bin'
    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin_path)
    print(f"  -> Dengeli SFT binary dosyası kaydedildi: {output_bin_path}")
    print(f"  -> Toplam birleşik token sayısı: {len(flat_tokens)}")
    
    # Calculate percentage
    parenting_token_count = sum(len(r) for r in oversampled_parenting)
    scientific_token_count = sum(len(r) for r in scientific_records)
    print(f"  -> Bilimsel Token Oranı:  {scientific_token_count / len(flat_tokens) * 100:.2f}%")
    print(f"  -> Ebeveynlik Token Oranı: {parenting_token_count / len(flat_tokens) * 100:.2f}%")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
