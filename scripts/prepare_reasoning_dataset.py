#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: MORFEMİK AKIL YÜRÜTME (CoT) VERİ DERLEME
===================================================================
Bu betik; 'data/pedagogy/cot_vault.jsonl' kasasında izole edilen derin
öğretmen akıl yürütme (Chain-of-Thought) ve düşünce zinciri kayıtlarını
okuyarak Evre 5 (Quiet-STaR / Morphemic Reasoning) eğitimi için
'data/train_reasoning_cot.bin' ikili eğitim veri setini üretir.

Biçim:
<BOS> <INSTRUCTION> {instruction} </INSTRUCTION> <INPUT> {query} </INPUT> <OUTPUT> <DUSUNCE> {thought_trace} </DUSUNCE> {final_answer} </OUTPUT> <EOS>
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary


def format_reasoning_sample(item: dict) -> dict:
    """
    Transforms a raw CoT vault record into the Kristal structured prompt dictionary.
    """
    instruction = item.get("instruction", "Soruyu derinlemesine akıl yürüterek açıkla.").strip()
    query = item.get("query", "").strip()
    thought = item.get("thought_trace", "").strip()
    answer = item.get("final_answer", "").strip()
    
    output_text = f"<DUSUNCE> {thought} </DUSUNCE> {answer}".strip()
    
    return {
        "instruction": instruction,
        "input": query,
        "output": output_text
    }


def compile_reasoning_dataset(
    vault_path: str = "data/pedagogy/cot_vault.jsonl",
    output_bin: str = "data/train_reasoning_cot.bin",
    vocab_path: str = "data/vocab.json",
    lexicon_path: str = "data/lexicon/roots.tsv",
    min_thought_length: int = 15
):
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL: MORFEMİK AKIL YÜRÜTME (CoT) VERİ DERLEME (TRAIN PREP)")
    print("=" * 70)

    # 1. Setup Compiler & Tokenizer
    vocab = Vocabulary()
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Boyutu: {vocab_size} morfem tokeni.")

    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Check and Read Vault
    if not os.path.exists(vault_path):
        print(f"Bilgi: '{vault_path}' bulunamadı. Boş bir kasa oluşturuluyor.")
        os.makedirs(os.path.dirname(vault_path), exist_ok=True)
        with open(vault_path, "w", encoding="utf-8") as f:
            pass

    records = []
    skipped = 0
    with open(vault_path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                data = json.loads(line_s)
                thought = data.get("thought_trace", "")
                if len(thought) < min_thought_length:
                    skipped += 1
                    continue
                formatted = format_reasoning_sample(data)
                raw_prompt = json.dumps(formatted, ensure_ascii=False)
                token_ids = tokenizer.encode(raw_prompt)
                if len(token_ids) > 4:
                    records.append(token_ids)
            except Exception as e:
                skipped += 1
                continue

    print(f"  * Kasa Dosyası: {vault_path}")
    print(f"  * Geçerli Akıl Yürütme Örneği: {len(records):,} adet (Atlanan yetersiz kayıt: {skipped}).")

    if not records:
        print("Uyarı: Kasada henüz yeterli akıl yürütme verisi birikmedi.")
        print("Pedagojik Süpervizör çalıştıkça CoT kasası otomatik olarak dolacaktır.")
        return

    # 3. Serialization to Binary uint16
    print("\n[3] İkili (Binary uint16) Formata Serileştiriliyor...")
    flat_tokens = []
    for rec in records:
        flat_tokens.extend(rec)

    total_tokens = len(flat_tokens)
    print(f"  * Toplam Morfem Token Sayısı: {total_tokens:,} token.")

    token_array = np.array(flat_tokens, dtype=np.uint16)
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)
    token_array.tofile(output_bin)
    file_size_kb = os.path.getsize(output_bin) / 1024
    print(f"  * Çıktı Dosyası: {output_bin} ({file_size_kb:.2f} KB)")

    # 4. Metadata
    meta = {
        "dataset_name": "train_reasoning_cot",
        "source_vault": vault_path,
        "total_tokens": total_tokens,
        "total_samples": len(records),
        "vocab_size": vocab_size,
        "min_thought_length": min_thought_length
    }
    meta_path = output_bin + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  * Meta Veri Kaydedildi: {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile CoT vault to binary reasoning dataset.")
    parser.add_argument("--vault", type=str, default="data/pedagogy/cot_vault.jsonl", help="Path to cot_vault.jsonl")
    parser.add_argument("--output", type=str, default="data/train_reasoning_cot.bin", help="Output .bin file path")
    parser.add_argument("--min-len", type=int, default=15, help="Minimum thought length")
    args = parser.parse_args()

    compile_reasoning_dataset(
        vault_path=args.vault,
        output_bin=args.output,
        min_thought_length=args.min_len
    )
