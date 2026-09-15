#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: DENGELİ SOHBET (CHAT) SFT VERİ DERLEME
===============================================================
Bu betik, sohbet ağırlıklı (%65 Chat, %20 Ebeveynlik/Morfoloji, %10 GTS Sözlük, %5 Alan Uzmanlığı)
dengeli bir ince ayar veri kümesi (`data/train_chat_balanced.bin`) üretir.
Bu sayede model yeni sohbet yetenekleri kazanırken eski morfolojik yeteneklerini unutmaz.
"""

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
            if len(token_ids) > 2:
                records.append(token_ids)
        except Exception:
            continue
            
    print(f"  * {os.path.basename(filepath):<36}: {len(records):,} kayıt tokenize edildi.")
    return records

def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DENGELİ SOHBET EĞİTİM VERİSİ DERLEME")
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

    print("\n[1] Dengeli Sohbet Bileşenleri Yükleniyor:")

    # A. Zenginleştirilmiş Sohbet Verisi (Chat Conversations)
    chat_records = tokenize_jsonl('data/pedagogy/chat_conversations.jsonl', tokenizer, max_samples=6000)

    # B. Ortaokul Samimi Sohbet (3x oversampled)
    ms_chat_raw = tokenize_jsonl('data/pedagogy/middle_school_chat.jsonl', tokenizer)
    ms_chat_records = ms_chat_raw * 3

    # C. Morfolojik Ebeveynlik (Dengeleme için 2,500 kayıt)
    parenting_records = tokenize_jsonl('data/pedagogy/parenting_deep_dataset.jsonl', tokenizer, max_samples=2500)

    # D. GTS Anlamsal Sözlük (1,500 kayıt)
    semantics_records = tokenize_jsonl('data/pedagogy/lexical_semantics_dataset.jsonl', tokenizer, max_samples=1500)

    # E. Marangozluk Alan Uzmanlığı (2,500 kayıt)
    carpenter_raw = tokenize_jsonl('data/pedagogy/carpenter_specialization_dataset.jsonl', tokenizer)
    carpenter_records = carpenter_raw[:2500]

    # F. İnteraktif Self-RAG Kullanım Görevleri (2,500 kayıt)
    rag_records = tokenize_jsonl('data/pedagogy/rag_interactive_dataset.jsonl', tokenizer, max_samples=2500)

    # G. Klasik Sadık Belge Alıntılama (RAG Grounding - 3,500 kayıt)
    classic_rag_records = tokenize_jsonl('data/pedagogy/rag_dataset.jsonl', tokenizer, max_samples=3500)

    # H. 1931 Türk Tarihi Çok Turlu Sohbet Külliyatı (3,000 kayıt)
    history_chat_records = tokenize_jsonl('data/pedagogy/turk_tarihi_chat.jsonl', tokenizer, max_samples=3000)

    # I. 1931 Türk Tarihi SFT ve Persona Görevleri (2,500 kayıt)
    history_sft_records = tokenize_jsonl('data/pedagogy/turk_tarihi_sft.jsonl', tokenizer, max_samples=2500)

    # 2. Save Vocabulary Update if any new token emerged
    vocab.save('data/vocab.json')
    print(f"\n[2] Sözlük Kontrolü: {initial_vocab_size} -> {len(vocab.stoi)} token.")

    # 3. Combine and Shuffle
    all_records = (
        chat_records +
        ms_chat_records +
        parenting_records +
        semantics_records +
        carpenter_records +
        rag_records +
        classic_rag_records +
        history_chat_records +
        history_sft_records
    )

    print(f"\n[3] Toplam Dengeli Sohbet & RAG SFT Kayıt Sayısı: {len(all_records):,}")
    print("Kayıtlar karıştırılıyor (shuffling)...")
    random.seed(42)
    random.shuffle(all_records)

    # 4. Pack into fixed block_size = 128
    block_size = 128
    pad_id = vocab.stoi.get("<PAD>", 1)

    print(f"\n[4] Tokenlar Blok Boyutuna ({block_size}) Hizalanıyor ve Paketleniyor...")
    flat_tokens = []
    for rec in all_records:
        if len(rec) < block_size:
            padded_rec = rec + [pad_id] * (block_size - len(rec))
        else:
            padded_rec = rec[:block_size]
        flat_tokens.extend(padded_rec)

    output_bin = 'data/train_chat_balanced.bin'
    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin)

    meta_info = {
        "total_records": len(all_records),
        "total_tokens": len(flat_tokens),
        "block_size": block_size,
        "vocab_size": len(vocab.stoi),
        "dataset_composition": {
            "chat_conversations": len(chat_records),
            "middle_school_chat": len(ms_chat_records),
            "parenting_deep": len(parenting_records),
            "lexical_semantics": len(semantics_records),
            "carpenter_specialization": len(carpenter_records),
            "rag_interactive": len(rag_records),
            "classic_rag": len(classic_rag_records),
            "turk_tarihi_1931_chat": len(history_chat_records)
        }
    }

    meta_path = output_bin + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(" DENGELİ SOHBET EĞİTİM VERİSİ DERLEMESİ TAMAMLANDI!")
    print("=" * 70)
    print(f"Çıktı Binary Dosyası: {output_bin} ({file_size_mb:.2f} MB)")
    print(f"Toplam Token Sayısı : {len(flat_tokens):,}")
    print(f"Meta Veri Dosyası   : {meta_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
