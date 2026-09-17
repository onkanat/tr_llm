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


def build_composition(parts: Dict[str, int]) -> Dict[str, int]:
    """Sohbet dengeli veri kümesi bileşenlerini derler ve döndürür.
    
    Tüm 9 kaynağın kayıt sayılarını tam olarak içerir.
    Öz-tutarlılık: Bileşen toplamı toplam kayıt sayısına eşit olmalıdır.
    """
    composition = {
        "chat_conversations": parts.get("chat_conversations", 0),
        "middle_school_chat": parts.get("middle_school_chat", 0),
        "parenting_deep": parts.get("parenting_deep", 0),
        "lexical_semantics": parts.get("lexical_semantics", 0),
        "carpenter_specialization": parts.get("carpenter_specialization", 0),
        "rag_interactive": parts.get("rag_interactive", 0),
        "classic_rag": parts.get("classic_rag", 0),
        "turk_tarihi_1931_chat": parts.get("turk_tarihi_1931_chat", 0),
        "turk_tarihi_sft": parts.get("turk_tarihi_sft", 0),
    }
    return composition


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
    output_bin: str = "data/train_chat_balanced.bin",
    exclude_sources: list = None,
    vocab_path: str = "data/rebuild/vocab_base_32852.json",
    literal_entity_mode: bool = True,
    allow_frozen_write: bool = False,
) -> None:
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DENGELİ SOHBET EĞİTİM VERİSİ DERLEME")
    print("=" * 70)

    exclude_set = set(exclude_sources) if exclude_sources else set()
    if exclude_set:
        print(f"Hariç Tutulan Kaynaklar: {sorted(list(exclude_set))}")

    # D2: Çıktı yolu denetimi (yazmadan önce)
    check_output_path(output_bin, allow_frozen_write)

    # 1. Setup Compiler & Tokenizer
    vocab = Vocabulary()
    vocab.load(vocab_path)
    initial_vocab_size = len(vocab.stoi)
    print(f"Başlangıç Sözlük Boyutu: {initial_vocab_size} token ({vocab_path}).")

    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=literal_entity_mode)

    print(f"\n[1] Dengeli Sohbet Bileşenleri Yükleniyor (literal_entity_mode={literal_entity_mode}):")

    source_stats: Dict[str, dict] = {}
    all_records = []
    raw_parts: Dict[str, int] = {}

    # A. Zenginleştirilmiş Sohbet Verisi (Chat Conversations)
    if "chat_conversations" not in exclude_set:
        chat_records, source_stats["chat_conversations"] = tokenize_jsonl(
            "data/pedagogy/chat_conversations.jsonl", tokenizer, max_samples=6000
        )
        all_records.extend(chat_records)
        raw_parts["chat_conversations"] = len(chat_records)

    # B. Ortaokul Samimi Sohbet (3x oversampled)
    if "middle_school_chat" not in exclude_set:
        ms_chat_raw, ms_stats = tokenize_jsonl("data/pedagogy/middle_school_chat.jsonl", tokenizer)
        ms_chat_records = ms_chat_raw * 3
        source_stats["middle_school_chat"] = ms_stats
        all_records.extend(ms_chat_records)
        raw_parts["middle_school_chat"] = len(ms_chat_records)

    # C. Morfolojik Ebeveynlik (Dengeleme için 2,500 kayıt)
    if "parenting_deep" not in exclude_set:
        parenting_records, source_stats["parenting_deep"] = tokenize_jsonl(
            "data/pedagogy/parenting_deep_dataset.jsonl", tokenizer, max_samples=2500
        )
        all_records.extend(parenting_records)
        raw_parts["parenting_deep"] = len(parenting_records)

    # D. GTS Anlamsal Sözlük (1,500 kayıt)
    if "lexical_semantics" not in exclude_set:
        semantics_records, source_stats["lexical_semantics"] = tokenize_jsonl(
            "data/pedagogy/lexical_semantics_dataset.jsonl", tokenizer, max_samples=1500
        )
        all_records.extend(semantics_records)
        raw_parts["lexical_semantics"] = len(semantics_records)

    # E. Marangozluk Alan Uzmanlığı (2,500 kayıt)
    if "carpenter_specialization" not in exclude_set:
        carpenter_raw, source_stats["carpenter_specialization"] = tokenize_jsonl(
            "data/pedagogy/carpenter_specialization_dataset.jsonl", tokenizer
        )
        carpenter_records = carpenter_raw[:2500]
        all_records.extend(carpenter_records)
        raw_parts["carpenter_specialization"] = len(carpenter_records)

    # F. İnteraktif Self-RAG Kullanım Görevleri (2,500 kayıt)
    if "rag_interactive" not in exclude_set:
        rag_records, source_stats["rag_interactive"] = tokenize_jsonl(
            "data/pedagogy/rag_interactive_dataset.jsonl", tokenizer, max_samples=2500
        )
        all_records.extend(rag_records)
        raw_parts["rag_interactive"] = len(rag_records)

    # G. Klasik Sadık Belge Alıntılama (RAG Grounding - 3,500 kayıt)
    if "classic_rag" not in exclude_set:
        classic_rag_records, source_stats["classic_rag"] = tokenize_jsonl(
            "data/pedagogy/rag_dataset.jsonl", tokenizer, max_samples=3500
        )
        all_records.extend(classic_rag_records)
        raw_parts["classic_rag"] = len(classic_rag_records)

    # H. 1931 Türk Tarihi Çok Turlu Sohbet Külliyatı (3,000 kayıt)
    if "turk_tarihi_1931_chat" not in exclude_set:
        history_chat_records, source_stats["turk_tarihi_1931_chat"] = tokenize_jsonl(
            "data/pedagogy/turk_tarihi_chat.jsonl", tokenizer, max_samples=3000
        )
        all_records.extend(history_chat_records)
        raw_parts["turk_tarihi_1931_chat"] = len(history_chat_records)

    # I. 1931 Türk Tarihi SFT ve Persona Görevleri (2,500 kayıt)
    if "turk_tarihi_sft" not in exclude_set:
        history_sft_records, source_stats["turk_tarihi_sft"] = tokenize_jsonl(
            "data/pedagogy/turk_tarihi_sft.jsonl", tokenizer, max_samples=2500
        )
        all_records.extend(history_sft_records)
        raw_parts["turk_tarihi_sft"] = len(history_sft_records)

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

    # D2: Çıktı dosyasını yazmadan hemen önce tekrar kontrol
    check_output_path(output_bin, allow_frozen_write)

    out_dir = os.path.dirname(output_bin)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    arr = np.array(flat_tokens, dtype=np.uint16)
    arr.tofile(output_bin)

    composition = build_composition(raw_parts)

    # Öz-tutarlılık assert'i: tüm 9 bileşenin toplamı toplam kayıt sayısına eşit olmalıdır
    assert sum(composition.values()) == len(all_records), (
        f"Öz-tutarlılık hatası: Bileşenler toplamı ({sum(composition.values())}) "
        f"toplam kayıt sayısına ({len(all_records)}) eşit değil!"
    )

    # Toplam kayıt muhasebesi
    total_accounting = {
        "raw_lines": sum(st.get("raw_lines", 0) for st in source_stats.values()),
        "sampled": sum(st.get("sampled", 0) for st in source_stats.values()),
        "json_errors": sum(st.get("json_errors", 0) for st in source_stats.values()),
        "encode_errors": sum(st.get("encode_errors", 0) for st in source_stats.values()),
        "lines_ok": sum(st.get("lines_ok", 0) for st in source_stats.values()),
        "short_skipped": sum(st.get("short_skipped", 0) for st in source_stats.values()),
        "kept": sum(st.get("kept", 0) for st in source_stats.values()),
    }

    meta_info = {
        "total_records": len(all_records),
        "total_tokens": len(flat_tokens),
        "block_size": block_size,
        "vocab_path": vocab_path,
        "vocab_size": len(vocab.stoi),
        "literal_entity_mode": literal_entity_mode,
        "excluded_sources": sorted(list(exclude_set)),
        "active_sources": sorted(list(raw_parts.keys())),
        "dataset_composition": composition,
        "record_accounting": total_accounting,
        "source_stats": source_stats,
    }

    meta_path = output_bin + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(" DENGELİ SOHBET EĞİTİM VERİSİ DERLEMESİ TAMAMLANDI!")
    print("=" * 70)
    print(f"Çıktı Binary Dosyası: {output_bin} ({file_size_mb:.2f} MB)")
    print(f"Toplam Token Sayısı : {len(flat_tokens):,}")
    print(f"Meta Veri Dosyası   : {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare balanced chat SFT dataset")
    parser.add_argument("--output", type=str, default="data/train_chat_balanced.bin", help="Output binary file path")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated source keys to exclude")
    parser.add_argument("--vocab-path", type=str, default="data/rebuild/vocab_base_32852.json", help="Vocabulary path")
    parser.add_argument("--literal-entity-mode", action="store_true", default=True, help="Enable literal entity mode")
    parser.add_argument("--no-literal-entity-mode", dest="literal_entity_mode", action="store_false")
    parser.add_argument("--allow-frozen-write", action="store_true", default=False, help="Allow writing to frozen path")
    args = parser.parse_args()

    exc_list = [s.strip() for s in args.exclude.split(",") if s.strip()] if args.exclude else None

    main(
        output_bin=args.output,
        exclude_sources=exc_list,
        vocab_path=args.vocab_path,
        literal_entity_mode=args.literal_entity_mode,
        allow_frozen_write=args.allow_frozen_write
    )
