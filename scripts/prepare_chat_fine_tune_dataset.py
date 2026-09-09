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
    print(" SAMİMİ ORTAOKUL SOHBET SFT VERİ SETİ DERLEME")
    print("=" * 60)

    # 1. Initialize Vocabulary and Tokenizer
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    print(f"Sözlük yüklendi: {len(vocab.stoi)} token.")

    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Load and Tokenize Middle School Chat SFT data
    chat_path = 'data/pedagogy/middle_school_chat.jsonl'
    if not os.path.exists(chat_path):
        print(f"Hata: {chat_path} bulunamadı! Lütfen önce generatorü çalıştırın.")
        return

    print("Ortaokul sohbet verileri yükleniyor ve tokenize ediliyor...")
    chat_records = []
    with open(chat_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                raw_prompt = json.dumps(item)
                token_ids = tokenizer.encode(raw_prompt)
                chat_records.append(token_ids)
    print(f"  -> Benzersiz sohbet kayıt sayısı: {len(chat_records)}")

    # Oversample chat data (repeat 5 times to make it a balanced feature)
    oversampled_chat = chat_records * 5
    print(f"  -> Oversampling sonrası sohbet kayıt sayısı: {len(oversampled_chat)}")

    # 3. Load and Tokenize Parenting tasks (subset of 1,500 records)
    parenting_path = 'data/pedagogy/parenting_dataset.jsonl'
    parenting_records = []
    if os.path.exists(parenting_path):
        print("Ebeveynlik (Parenting SFT) görevleri yükleniyor...")
        with open(parenting_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    raw_prompt = json.dumps(item)
                    token_ids = tokenizer.encode(raw_prompt)
                    parenting_records.append(token_ids)
        # Settle to a max subset of 1500 to keep the dataset balanced
        random.seed(42)
        if len(parenting_records) > 1500:
            parenting_records = random.sample(parenting_records, 1500)
        print(f"  -> Seçilen ebeveynlik kayıt sayısı: {len(parenting_records)}")
    else:
        print("Uyarı: Ebeveynlik veri kümesi bulunamadı, atlanıyor.")

    # 4. Load Scientific SFT data (subset of 1,500 records)
    scientific_bin_path = 'data/train_all_chosen.bin'
    scientific_records = []
    if os.path.exists(scientific_bin_path):
        print("Bilimsel SFT tokenleri yükleniyor...")
        scientific_tokens = np.fromfile(scientific_bin_path, dtype=np.uint16)
        
        # Split into records
        current_record = []
        for token in scientific_tokens:
            if token == 2 and current_record:
                scientific_records.append(current_record)
                current_record = []
            current_record.append(int(token))
        if current_record:
            scientific_records.append(current_record)
            
        if len(scientific_records) > 1500:
            random.seed(42)
            scientific_records = random.sample(scientific_records, 1500)
        print(f"  -> Seçilen bilimsel kayıt sayısı: {len(scientific_records)}")
    else:
        print("Uyarı: Bilimsel veri kümesi bulunamadı, atlanıyor.")

    # 4.5. Load and Tokenize RAG tasks (subset of 1,500 records)
    rag_path = 'data/pedagogy/rag_dataset.jsonl'
    rag_records = []
    if os.path.exists(rag_path):
        print("RAG (Retrieval-Augmented Generation) görevleri yükleniyor...")
        with open(rag_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    raw_prompt = json.dumps(item)
                    token_ids = tokenizer.encode(raw_prompt)
                    rag_records.append(token_ids)
        # Settle to a max subset of 6000 to keep the dataset balanced
        random.seed(42)
        if len(rag_records) > 6000:
            rag_records = random.sample(rag_records, 6000)
        print(f"  -> Seçilen RAG kayıt sayısı: {len(rag_records)}")

    else:
        print("Uyarı: RAG veri kümesi bulunamadı, atlanıyor.")

    # Save vocabulary to persist any new tokens added during tokenization
    vocab.save('data/vocab.json')
    print(f"Sözlük güncellendi ve 'data/vocab.json' dosyasına kaydedildi (Yeni boyut: {len(vocab.stoi)}).")

    # 5. Combine and Shuffle
    all_records = oversampled_chat + parenting_records + scientific_records + rag_records
    print(f"  -> Toplam birleşik kayıt sayısı: {len(all_records)}")
    
    print("Kayıtlar karıştırılıyor (shuffling)...")
    random.seed(42)
    random.shuffle(all_records)

    # 6. Flatten, pad to 128 and save to binary file
    flat_tokens = []
    pad_id = vocab.stoi.get("<PAD>", 1)
    for rec in all_records:
        if len(rec) < 128:
            padded_rec = rec + [pad_id] * (128 - len(rec))
        else:
            padded_rec = rec[:128]
        flat_tokens.extend(padded_rec)

    output_bin_path = 'data/train_chat_sft.bin'
    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin_path)
    print(f"  -> Sohbet odaklı SFT binary dosyası kaydedildi: {output_bin_path}")
    print(f"  -> Toplam birleşik token sayısı (padded): {len(flat_tokens)}")
    
    # Calculate percentage
    chat_token_count = sum(len(r) for r in oversampled_chat)
    parenting_token_count = sum(len(r) for r in parenting_records)
    scientific_token_count = sum(len(r) for r in scientific_records)
    rag_token_count = sum(len(r) for r in rag_records)
    print(f"  -> Sohbet Token Oranı:   {chat_token_count / len(flat_tokens) * 100:.2f}%")
    print(f"  -> Ebeveynlik Token Oranı: {parenting_token_count / len(flat_tokens) * 100:.2f}%")
    print(f"  -> Bilimsel Token Oranı:  {scientific_token_count / len(flat_tokens) * 100:.2f}%")
    print(f"  -> RAG Token Oranı:        {rag_token_count / len(flat_tokens) * 100:.2f}%")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
