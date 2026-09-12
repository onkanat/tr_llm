#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: PEDAGOJİ VE TEMEL LİSE EĞİTİMİ VERİ DERLEME
=======================================================================
Bu betik; temel modelin eğitimi için Pedagoji (10 Morfolojik Görev),
Bebeklik (Kavramsal Ontoloji), Temel Lise Eğitimi (Fen, Edebiyat, Tarih, Coğrafya)
ve Anlamsal Sözlük verilerini derleyerek `data/train_pedagogy_highschool.bin` dosyasını üretir.
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
    print(" KRİSTAL-VEKTÖREL: PEDAGOJİ & LİSE EĞİTİMİ VERİ DERLEME (TRAIN PREP)")
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

    print("\n[1] Pedagojik ve Temel Lise Alt Kümeleri Yükleniyor:")

    # A. Derinleştirilmiş Ebeveynlik (10 Temel Morfolojik Görev)
    parenting_records = tokenize_jsonl("data/pedagogy/parenting_deep_dataset.jsonl", tokenizer, max_samples=20000)

    # B. Temel Lise Eğitimi (Fen, Edebiyat, Tarih, Coğrafya, Mantık) - 10x oversampled
    high_school_raw = tokenize_jsonl("data/pedagogy/high_school_foundation_dataset.jsonl", tokenizer)
    high_school_records = high_school_raw * 10
    print(f"  * [Oversampled] Temel Lise Müfredatı: {len(high_school_records):,} kayıt.")

    # C. Bebeklik (Kavramsal Temeller & Ontoloji - Dengeli)
    infancy_records = tokenize_jsonl("data/pedagogy/infancy_dataset.jsonl", tokenizer, max_samples=4000)

    # D. GTS Semantik Sözlük ve Anlamsal Tanımlar
    semantics_records = tokenize_jsonl("data/pedagogy/lexical_semantics_dataset.jsonl", tokenizer, max_samples=10000)

    # E. Self-RAG & Ajan İletişimi
    rag_records = tokenize_jsonl("data/pedagogy/rag_dataset.jsonl", tokenizer, max_samples=3500)

    # F. Edebiyat ve Şiir Özel Külliyatı (10x oversampled)
    lit_path = "data/pedagogy/literature_poetry_dataset.jsonl"
    lit_records = []
    if os.path.exists(lit_path):
        lit_raw = tokenize_jsonl(lit_path, tokenizer)
        lit_records = lit_raw * 10
        print(f"  * [Oversampled] Türkçe Edebiyat & Şiir: {len(lit_records):,} kayıt.")

    # G. 1931 Türk Tarihi Ders Kitapları Sentetik SFT Külliyatı
    history_path = "data/pedagogy/turk_tarihi_sft.jsonl"
    history_records = []
    if os.path.exists(history_path):
        history_records = tokenize_jsonl(history_path, tokenizer, max_samples=6500)
        print(f"  * 1931 Türk Tarihi SFT Külliyatı: {len(history_records):,} kayıt.")

    # 2. Bütünleştirme ve Karıştırma
    # DİKKAT: Ahşap/Marangozluk uzmanlığı (Evre 4), Lise Temel Eğitimi (Evre 3) bittikten
    # sonra dikey bir modül olarak ayrı derlenir ve eğitilir (scripts/prepare_carpenter_specialization_dataset.py).
    all_records = (
        parenting_records + 
        high_school_records + 
        infancy_records + 
        semantics_records + 
        rag_records + 
        lit_records +
        history_records
    )
    random.seed(42)
    random.shuffle(all_records)
    print(f"  * Toplam Örnek Sayısı: {len(all_records):,} adet.")

    # 3. İkili Diziye Dönüştürme
    print("\n[3] İkili (Binary uint16) Formata Serileştiriliyor...")
    flat_tokens = []
    for rec in all_records:
        flat_tokens.extend(rec)

    total_tokens = len(flat_tokens)
    print(f"  * Toplam Morfem Token Sayısı: {total_tokens:,} token.")

    token_array = np.array(flat_tokens, dtype=np.uint16)
    output_bin = "data/train_pedagogy_highschool.bin"
    token_array.tofile(output_bin)
    file_size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"  * Çıktı Dosyası: {output_bin} ({file_size_mb:.2f} MB)")

    # 4. Meta Veri Kaydı
    block_size = 64
    meta = {
        "dataset_name": "train_pedagogy_highschool",
        "created_at": "2026-09-12",
        "total_tokens": total_tokens,
        "total_samples": len(all_records),
        "block_size": block_size,
        "vocab_size": vocab_size,
        "distribution": {
            "parenting_morphology": len(parenting_records),
            "high_school_foundation": len(high_school_records),
            "infancy_ontology": len(infancy_records),
            "lexical_semantics": len(semantics_records),
            "rag_communication": len(rag_records),
            "literature_poetry": len(lit_records),
            "turk_tarihi_1931": len(history_records)
        }
    }
    meta_path = output_bin + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  * Meta Veri Kaydedildi: {meta_path}")
    print("=" * 70)
    print(" HAZIRLIK TAMAMLANDI: Model artık bu ikili dosya ile doğrudan eğitilebilir:")
    print(f" ./venv/bin/python -u train.py --data {output_bin} --steps 300 --batch-size 16 --device cpu")
    print("=" * 70)


if __name__ == "__main__":
    main()
