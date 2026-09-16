#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DENGELİ SFT VERİ SETİ HAZIRLAMA (DENGELİ VERİ KARIŞTIRMA)
=========================================================
"""

import os
import sys
import json
import random
from typing import Dict, Tuple, List
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.frozen_guard import is_frozen_path


def check_output_path(output_bin: str, allow_frozen_write: bool = False) -> None:
    """Çıktı yolunun donmuş olup olmadığını denetler. allow_frozen_write=False ise RuntimeError fırlatır."""
    if is_frozen_path(output_bin) and not allow_frozen_write:
        raise RuntimeError(
            f"Donmus yola yazma engellendi: {output_bin} (allow_frozen_write=False)"
        )


def tokenize_jsonl(
    filepath: str,
    tokenizer: KristalTokenizer,
    max_samples: int = None,
    min_tokens: int = 2
) -> Tuple[List[List[int]], Dict[str, int]]:
    """JSONL kütüğünü satır satır okuyup tokenize eder ve 8 sayaçlı tam muhasebe istatistiği döner."""
    stats = {
        "missing": False,
        "raw_lines": 0,
        "sampled": 0,
        "json_errors": 0,
        "encode_errors": 0,
        "lines_ok": 0,
        "short_skipped": 0,
        "kept": 0,
    }
    records: List[List[int]] = []

    if not os.path.exists(filepath):
        stats["missing"] = True
        print(f"Uyarı: '{filepath}' bulunamadı, atlanıyor.")
        return records, stats

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    stats["raw_lines"] = len(lines)

    if max_samples and len(lines) > max_samples:
        random.seed(42)
        lines = random.sample(lines, max_samples)

    stats["sampled"] = len(lines)
    logged_errors = 0

    for line_idx, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except Exception as e:
            stats["json_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[tokenize_jsonl] {os.path.basename(filepath)}:{line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        try:
            raw_prompt = json.dumps(item, ensure_ascii=False)
            token_ids = tokenizer.encode(raw_prompt)
        except Exception as e:
            stats["encode_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[tokenize_jsonl] {os.path.basename(filepath)}:{line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        stats["lines_ok"] += 1

        if len(token_ids) <= min_tokens:
            stats["short_skipped"] += 1
        else:
            records.append(token_ids)
            stats["kept"] += 1

    # D4 Özdeşlik denetimi
    assert stats["sampled"] == stats["json_errors"] + stats["encode_errors"] + stats["lines_ok"], (
        f"Özdeşlik hatası ({filepath}): {stats['sampled']} != "
        f"{stats['json_errors']} + {stats['encode_errors']} + {stats['lines_ok']}"
    )

    print(f"  * {os.path.basename(filepath):<36}: {len(records):,} kayıt tokenize edildi.")
    return records, stats


def main(
    vocab_path: str = "data/rebuild/vocab_base_32852.json",
    literal_entity_mode: bool = True,
    allow_frozen_write: bool = False,
) -> None:
    print("=" * 60)
    print(" DENGELİ SFT VERİ SETİ HAZIRLAMA (DENGELİ VERİ KARIŞTIRMA)")
    print("=" * 60)

    output_bin_path = "data/train_balanced_sft.bin"
    # D2: Çıktı yolu denetimi (yazmadan önce)
    check_output_path(output_bin_path, allow_frozen_write)

    # 1. Initialize Vocabulary and Tokenizer
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"Sözlük yüklendi: {len(vocab.stoi)} token ({vocab_path}).")

    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=literal_entity_mode)

    source_stats: Dict[str, dict] = {}

    # 2. Load Scientific DPO Chosen tokens
    # D3-ISTISNA 2: Bu kaynak JSONL değildir; D4 satır özdeşliği uygulanmaz.
    scientific_bin_path = "data/train_all_chosen.bin"
    if not os.path.exists(scientific_bin_path):
        print(f"Hata: {scientific_bin_path} bulunamadı!")
        return

    print("Bilimsel SFT tokenleri yükleniyor...")
    scientific_tokens = np.fromfile(scientific_bin_path, dtype=np.uint16)
    print(f"  -> Bilimsel SFT token sayısı: {len(scientific_tokens)}")

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

    source_stats["scientific_dpo_chosen"] = {
        "input_tokens": len(scientific_tokens),
        "records_split": len(scientific_records),
        "kept": len(scientific_records),
    }

    # 3. Load and Tokenize Parenting SFT tasks
    # D3-ISTISNA 2: parenting_dataset.jsonl min_tokens=0 ile çağrılır (len>2 süzgeci taşımıyor davranışı korunur)
    parenting_records, source_stats["parenting_sft"] = tokenize_jsonl(
        "data/pedagogy/parenting_dataset.jsonl", tokenizer, min_tokens=0
    )
    print(f"  -> Benzersiz ebeveynlik kayıt sayısı: {len(parenting_records)}")

    # 4. Oversample Parenting SFT tasks to balance the dataset
    oversample_factor = 10
    oversampled_parenting = parenting_records * oversample_factor
    print(f"  -> Oversampling sonrası ebeveynlik kayıt sayısı: {len(oversampled_parenting)}")

    # 5. Load and Tokenize Turk Tarihi SFT tasks (min_tokens=2 varsayılan)
    history_sft_records, source_stats["turk_tarihi_sft"] = tokenize_jsonl(
        "data/pedagogy/turk_tarihi_sft.jsonl", tokenizer, min_tokens=2
    )
    print(f"  -> Benzersiz tarih SFT kayıt sayısı: {len(history_sft_records)}")

    # 6. Combine and Shuffle
    all_records = scientific_records + oversampled_parenting + history_sft_records
    print(f"  -> Toplam birleşik kayıt sayısı: {len(all_records)}")

    print("Kayıtlar karıştırılıyor (shuffling)...")
    random.seed(42)
    random.shuffle(all_records)

    # 7. Flatten and save to binary file
    flat_tokens = []
    for rec in all_records:
        flat_tokens.extend(rec)

    # D2: Çıktı dosyasını yazmadan hemen önce tekrar kontrol
    check_output_path(output_bin_path, allow_frozen_write)

    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin_path)
    print(f"  -> Dengeli SFT binary dosyası kaydedildi: {output_bin_path}")
    print(f"  -> Toplam birleşik token sayısı: {len(flat_tokens)}")

    total_accounting = {
        "raw_lines": sum(st.get("raw_lines", 0) for st in source_stats.values()),
        "sampled": sum(st.get("sampled", 0) for st in source_stats.values()),
        "json_errors": sum(st.get("json_errors", 0) for st in source_stats.values()),
        "encode_errors": sum(st.get("encode_errors", 0) for st in source_stats.values()),
        "lines_ok": sum(st.get("lines_ok", 0) for st in source_stats.values()),
        "short_skipped": sum(st.get("short_skipped", 0) for st in source_stats.values()),
        "kept": sum(st.get("kept", 0) for st in source_stats.values()),
    }

    meta_path = output_bin_path + ".meta.json"
    meta_info = {
        "total_records": len(all_records),
        "total_tokens": len(flat_tokens),
        "vocab_path": vocab_path,
        "vocab_size": len(vocab.stoi),
        "literal_entity_mode": literal_entity_mode,
        "parenting_oversample_factor": oversample_factor,
        "record_accounting": total_accounting,
        "source_stats": source_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    # Calculate percentage
    parenting_token_count = sum(len(r) for r in oversampled_parenting)
    scientific_token_count = sum(len(r) for r in scientific_records)
    print(f"  -> Bilimsel Token Oranı:  {scientific_token_count / len(flat_tokens) * 100:.2f}%")
    print(f"  -> Ebeveynlik Token Oranı: {parenting_token_count / len(flat_tokens) * 100:.2f}%")
    print(f"  -> Meta veri kaydedildi: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
