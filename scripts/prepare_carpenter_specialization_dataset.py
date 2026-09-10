#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: EVRE 4 - ALAN UZMANLAŞMASI (CARPENTER SPECIALIZATION)
================================================================================
Pedagojik Gelişim Evreleri:
  1. Evre: Bebeklik (Kavramsal Ontoloji)
  2. Evre: Ebeveynlik (Morfolojik Yetkinlik)
  3. Evre: Lise Temel Eğitimi (Genel Kültür, Fen, Edebiyat, Mantık)
  4. Evre: Alan Uzmanlaşması (Örn: Carpenter AI / Ahşap & Marangozluk Teknolojisi)

Bu betik; Lise Temel Eğitimini tamamlamış ana model üzerine giydirilecek olan
Ahşap ve Marangozluk Dikey Uzmanlık Modülü için `data/train_carpenter_specialization.bin`
veri kümesini hazırlar. Unutmayı (Catastrophic Forgetting) önlemek için küçük bir
morfolojik ve lise çıpası (anchoring) ile harmanlar.
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

    with open(filepath, "r", encoding="utf-8") as f:
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

    print(f"  * {os.path.basename(filepath):<38}: {len(records):,} kayıt tokenize edildi.")
    return records


def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL: EVRE 4 ALAN UZMANLAŞMASI (CARPENTER AI) VERİ DERLEME")
    print("=" * 70)

    # 1. Setup Compiler & Tokenizer
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Boyutu: {vocab_size} morfem tokeni.")

    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    print("\n[1] Ahşap Uzmanlığı ve Çıpa Verileri Yükleniyor:")

    # A. Ahşap ve Marangozluk Uzmanlığı (Doğal Soru-Cevap + SFT Normalizasyonu)
    carpenter_records = []
    carp_path = "data/pedagogy/carpenter_specialization_dataset.jsonl"
    if os.path.exists(carp_path):
        with open(carp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    raw_s = json.dumps(d, ensure_ascii=False)
                    tids = tokenizer.encode(raw_s)
                    if len(tids) > 2:
                        carpenter_records.append(tids)

                    # SFT Standart Komut Çifti ("Ahşap uzmanı olarak cevapla.")
                    inst = d.get("instruction", "")
                    q = d.get("input", "")
                    if not q and ":" in inst:
                        q = inst.split(":", 1)[1].strip()
                    if q:
                        norm_d = {
                            "instruction": "Ahşap uzmanı olarak cevapla.",
                            "input": q,
                            "output": d.get("output", "")
                        }
                        norm_tids = tokenizer.encode(json.dumps(norm_d, ensure_ascii=False))
                        if len(norm_tids) > 2:
                            carpenter_records.append(norm_tids)
                except Exception:
                    continue
    print(f"  * Ahşap & Marangozluk Kayıtları: {len(carpenter_records):,} adet.")

    # B. Unutmayı Önleyici Çıpa (Anti-Forgetting Anchor): Az miktarda morfoloji ve lise temeli
    anchor_parenting = tokenize_jsonl("data/pedagogy/parenting_deep_dataset.jsonl", tokenizer, max_samples=1500)
    anchor_foundation = tokenize_jsonl("data/pedagogy/high_school_foundation_dataset.jsonl", tokenizer, max_samples=700)

    # 2. Harmanlama
    all_records = carpenter_records + anchor_parenting + anchor_foundation
    random.seed(42)
    random.shuffle(all_records)
    print(f"\n[2] Toplam Harmanlanan Uzmanlık Verisi: {len(all_records):,} adet.")

    # 3. İkili Serileştirme
    output_bin = "data/train_carpenter_specialization.bin"
    flat_tokens = []
    boundaries = []
    curr_offset = 0

    for r in all_records:
        flat_tokens.extend(r)
        boundaries.append({
            "offset": curr_offset,
            "length": len(r)
        })
        curr_offset += len(r)

    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin)

    meta_path = output_bin + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "record_count": len(all_records),
            "total_tokens": len(flat_tokens),
            "boundaries": boundaries
        }, f, ensure_ascii=False)

    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"\n[3] Uzmanlık İkili Dosyası Oluşturuldu: {output_bin} ({size_mb:.2f} MB, {len(flat_tokens):,} token)")
    print("=" * 70)


if __name__ == "__main__":
    main()
