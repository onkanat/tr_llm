#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: İKİLİ EĞİTİM VERİ KÜMELERİNİ YENİDEN DERLEME
============================================================
Temizlenmiş JSONL kütüklerini uint16 binary formatına (.bin) dönüştürür.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_goal_pipeline import compile_jsonl_to_bin

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"


def main():
    print(f"{C_BOLD}{C_GREEN}======================================================================")
    print("  KRİSTAL-VEKTÖREL: ARINMIŞ İKİLİ EĞİTİM VERİLERİNİ DERLEME")
    print("======================================================================\n" + C_RESET)

    t0 = time.time()

    # 1. Balanced SFT Dataset
    print(f"{C_BOLD}1. data/train_balanced_sft.bin Derleniyor...{C_RESET}")
    compile_jsonl_to_bin(
        jsonl_paths=[
            "data/pedagogy/high_school_foundation_dataset.jsonl",
            "data/pedagogy/middle_school_chat.jsonl",
            "data/pedagogy/turk_tarihi_sft.jsonl",
            "data/pedagogy/literature_poetry_dataset.jsonl",
            "data/pedagogy/infancy_dataset.jsonl",
            "data/pedagogy/parenting_dataset.jsonl",
            "data/pedagogy/arena_base_accumulated.jsonl",
            "data/future_train_vector.jsonl"
        ],
        output_bin="data/train_balanced_sft.bin",
        vocab_path="data/rebuild/vocab_base_32852.json",
        literal_entity_mode=True,
        block_size=64,
        allow_frozen_write=True
    )

    # 2. Chat Balanced SFT Dataset
    print(f"\n{C_BOLD}2. data/train_chat_balanced.bin Derleniyor...{C_RESET}")
    compile_jsonl_to_bin(
        jsonl_paths=[
            "data/pedagogy/chat_conversations.jsonl",
            "data/pedagogy/turk_tarihi_chat.jsonl",
            "data/pedagogy/arena_base_accumulated.jsonl",
            "data/future_train_vector.jsonl"
        ],
        output_bin="data/train_chat_balanced.bin",
        vocab_path="data/rebuild/vocab_base_32852.json",
        literal_entity_mode=True,
        block_size=64,
        allow_frozen_write=True
    )

    # 3. Carpenter Specialization Dataset
    print(f"\n{C_BOLD}3. data/train_carpenter_specialization.bin Derleniyor...{C_RESET}")
    compile_jsonl_to_bin(
        jsonl_paths=[
            "data/pedagogy/carpenter_specialization_dataset.jsonl",
            "data/pedagogy/arena_carpenter_accumulated.jsonl"
        ],
        output_bin="data/train_carpenter_specialization.bin",
        vocab_path="data/rebuild/vocab_base_32852.json",
        literal_entity_mode=True,
        block_size=64,
        allow_frozen_write=True
    )

    dt = time.time() - t0
    print(f"\n{C_BOLD}{C_GREEN}Tüm ikili veri kümeleri başarıyla derlendi ({dt:.1f} sn)!{C_RESET}\n")


if __name__ == "__main__":
    main()
