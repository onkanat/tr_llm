#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: EVRE 4 ALAN UZMANLAŞMASI (CARPENTER AI) VERİ DERLEME
=====================================================================
Bu betik; Lise Temel Eğitimini tamamlamış ana model üzerine giydirilecek olan
Ahşap ve Marangozluk Dikey Uzmanlık Modülü için `data/train_carpenter_specialization.bin`
veri kümesini hazırlar.
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

    print(f"  * {os.path.basename(filepath):<38}: {len(records):,} kayıt tokenize edildi.")
    return records, stats


def tokenize_carpenter_jsonl(
    filepath: str,
    tokenizer: KristalTokenizer
) -> Tuple[List[List[int]], Dict[str, int]]:
    """Ahşap ve marangozluk uzmanlık kütüğünü (ham + normalize SFT) tokenize eder ve 9 sayaçlı muhasebe döner."""
    stats = {
        "missing": False,
        "raw_lines": 0,
        "sampled": 0,
        "json_errors": 0,
        "encode_errors": 0,
        "normalize_encode_errors": 0,
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
    stats["sampled"] = len(lines)
    logged_errors = 0

    for line_idx, line in enumerate(lines, 1):
        try:
            d = json.loads(line)
        except Exception as e:
            stats["json_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[carpenter] {line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        try:
            raw_s = json.dumps(d, ensure_ascii=False)
            tids = tokenizer.encode(raw_s)
        except Exception as e:
            stats["encode_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[carpenter] {line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        stats["lines_ok"] += 1

        # 1. Ham kayıt
        if len(tids) > 2:
            records.append(tids)
            stats["kept"] += 1
        else:
            stats["short_skipped"] += 1

        # 2. SFT Standart Komut Çifti ("Ahşap uzmanı olarak cevapla.")
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
            try:
                norm_tids = tokenizer.encode(json.dumps(norm_d, ensure_ascii=False))
                if len(norm_tids) > 2:
                    records.append(norm_tids)
                    stats["kept"] += 1
                else:
                    stats["short_skipped"] += 1
            except Exception as e:
                # İkinci kodlama hatası (normalize SFT)
                stats["normalize_encode_errors"] += 1
                if logged_errors < 5:
                    sys.stderr.write(f"[carpenter-sft] {line_idx} {type(e).__name__}: {e}\n")
                    logged_errors += 1

    # D4 Özdeşlik denetimi (satır/hata/ok üçlüsü)
    assert stats["sampled"] == stats["json_errors"] + stats["encode_errors"] + stats["lines_ok"], (
        f"Özdeşlik hatası (carpenter): {stats['sampled']} != "
        f"{stats['json_errors']} + {stats['encode_errors']} + {stats['lines_ok']}"
    )
    # Ek invariant: satır başına en fazla 2 kayıt üretilebilir
    assert stats["kept"] <= 2 * stats["lines_ok"], (
        f"Kayıt tavan aşımı: kept ({stats['kept']}) > 2 * lines_ok ({stats['lines_ok']})"
    )

    return records, stats


def main(
    vocab_path: str = "data/rebuild/vocab_base_32852.json",
    literal_entity_mode: bool = True,
    allow_frozen_write: bool = False,
) -> None:
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL: EVRE 4 ALAN UZMANLAŞMASI (CARPENTER AI) VERİ DERLEME")
    print("=" * 70)

    output_bin = "data/train_carpenter_specialization.bin"
    # D2: Çıktı yolu denetimi (yazmadan önce)
    check_output_path(output_bin, allow_frozen_write)

    # 1. Setup Compiler & Tokenizer
    vocab = Vocabulary()
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Boyutu: {vocab_size} morfem tokeni ({vocab_path}).")

    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=literal_entity_mode)

    print(f"\n[1] Ahşap Uzmanlığı ve Çıpa Verileri Yükleniyor (literal_entity_mode={literal_entity_mode}):")

    source_stats: Dict[str, dict] = {}

    # A. Ahşap ve Marangozluk Uzmanlığı (Doğal Soru-Cevap + SFT Normalizasyonu)
    # D3-ISTISNA (carpenter): Satır başına 2 kayıt üretebilir, sayım olay düzeyinde yapılır.
    carp_path = "data/pedagogy/carpenter_specialization_dataset.jsonl"
    carpenter_records, carp_stats = tokenize_carpenter_jsonl(carp_path, tokenizer)
    source_stats["carpenter_specialization"] = carp_stats
    print(f"  * Ahşap & Marangozluk Kayıtları: {len(carpenter_records):,} adet.")

    # B. Unutmayı Önleyici Çıpa (Anti-Forgetting Anchor): Az miktarda morfoloji ve lise temeli
    anchor_parenting, source_stats["anchor_parenting"] = tokenize_jsonl(
        "data/pedagogy/parenting_deep_dataset.jsonl", tokenizer, max_samples=1500
    )
    anchor_foundation, source_stats["anchor_foundation"] = tokenize_jsonl(
        "data/pedagogy/high_school_foundation_dataset.jsonl", tokenizer, max_samples=700
    )

    # 2. Harmanlama
    all_records = carpenter_records + anchor_parenting + anchor_foundation
    random.seed(42)
    random.shuffle(all_records)
    print(f"\n[2] Toplam Harmanlanan Uzmanlık Verisi: {len(all_records):,} adet.")

    # 3. İkili Serileştirme
    check_output_path(output_bin, allow_frozen_write)

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

    total_accounting = {
        "raw_lines": sum(st.get("raw_lines", 0) for st in source_stats.values()),
        "sampled": sum(st.get("sampled", 0) for st in source_stats.values()),
        "json_errors": sum(st.get("json_errors", 0) for st in source_stats.values()),
        "encode_errors": sum(st.get("encode_errors", 0) for st in source_stats.values()),
        "normalize_encode_errors": sum(st.get("normalize_encode_errors", 0) for st in source_stats.values()),
        "lines_ok": sum(st.get("lines_ok", 0) for st in source_stats.values()),
        "short_skipped": sum(st.get("short_skipped", 0) for st in source_stats.values()),
        "kept": sum(st.get("kept", 0) for st in source_stats.values()),
    }

    meta_path = output_bin + ".meta.json"
    meta_info = {
        "record_count": len(all_records),
        "total_records": len(all_records),
        "total_tokens": len(flat_tokens),
        "vocab_path": vocab_path,
        "vocab_size": vocab_size,
        "literal_entity_mode": literal_entity_mode,
        "boundaries": boundaries,
        "record_accounting": total_accounting,
        "source_stats": source_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"\n[3] Uzmanlık İkili Dosyası Oluşturuldu: {output_bin} ({size_mb:.2f} MB, {len(flat_tokens):,} token)")
    print("=" * 70)


if __name__ == "__main__":
    main()
