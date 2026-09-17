#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_replay_mix.py

T-0060: F4-v2 ceket yeniden eğitimi için replay karışımı oluşturucu.
Ceket verisi (train_carpenter_specialization.bin) ile replay kütlesini
(train_chat_balanced.bin) 128 token'lık bloklar düzeyinde [ceket, ceket, ceket, replay]
deseniyle (3:1 oranında, %25 replay) birleştirir.

Deterministik, saf fonksiyon yapısında ve tip ipuçları ile donatılmıştır.
"""

import os
import sys
import json
import argparse
import hashlib
from typing import Tuple, Dict, Any, List
import numpy as np


def compute_sha256(path: str) -> str:
    """Verilen dosyanın 64 karakterlik tam SHA-256 özetini hesaplar."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_replay_mix(
    jacket_path: str,
    replay_path: str,
    output_path: str,
    block_size: int = 128,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ceket ve replay ikili verilerini 3:1 blok deseniyle birleştirir.
    Artan ceket token'larını dosya sonuna ekler.
    Replay bloklarını replay veri kümesinin tamamına eşit aralıklarla yayar.
    """
    if not os.path.exists(jacket_path):
        raise FileNotFoundError(f"Ceket dosyası bulunamadı: {jacket_path}")
    if not os.path.exists(replay_path):
        raise FileNotFoundError(f"Replay dosyası bulunamadı: {replay_path}")

    jacket_sha = compute_sha256(jacket_path)
    replay_sha = compute_sha256(replay_path)

    j_data = np.memmap(jacket_path, dtype=np.uint16, mode="r")
    r_data = np.memmap(replay_path, dtype=np.uint16, mode="r")

    n_j_tokens = len(j_data)
    n_r_tokens = len(r_data)

    n_j_blocks = n_j_tokens // block_size
    rem_j_tokens = n_j_tokens % block_size

    # Her 3 ceket bloğuna 1 replay bloğu (ceil(n_j_blocks / 3))
    n_r_blocks = (n_j_blocks + 2) // 3

    total_avail_r_blocks = n_r_tokens // block_size
    if n_r_blocks > total_avail_r_blocks:
        raise ValueError(f"Yetersiz replay bloğu: gereken {n_r_blocks}, mevcut {total_avail_r_blocks}")

    # Replay bloklarını replay dosyasının tamamına eşit aralıklarla yay
    # linspace ile indeksleri belirle (deterministik)
    replay_indices = np.round(np.linspace(0, total_avail_r_blocks - 1, n_r_blocks)).astype(np.int64)

    # Toplam çıktı token sayısı
    total_mix_tokens = (n_j_blocks + n_r_blocks) * block_size + rem_j_tokens

    # Çıktı dosyasını disk üzerinde oluştur
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_mmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(total_mix_tokens,))

    curr_j_block = 0
    curr_r_idx = 0
    out_pos = 0

    # [ceket, ceket, ceket, replay] deseni
    while curr_j_block < n_j_blocks:
        # En fazla 3 ceket bloğu ekle
        chunk_j = min(3, n_j_blocks - curr_j_block)
        for _ in range(chunk_j):
            src_start = curr_j_block * block_size
            src_end = src_start + block_size
            out_mmap[out_pos:out_pos + block_size] = j_data[src_start:src_end]
            out_pos += block_size
            curr_j_block += 1

        # 1 replay bloğu ekle
        if curr_r_idx < n_r_blocks:
            r_block_idx = replay_indices[curr_r_idx]
            src_start = r_block_idx * block_size
            src_end = src_start + block_size
            out_mmap[out_pos:out_pos + block_size] = r_data[src_start:src_end]
            out_pos += block_size
            curr_r_idx += 1

    # Artan ceket token'larını sona ekle
    if rem_j_tokens > 0:
        src_start = n_j_blocks * block_size
        out_mmap[out_pos:out_pos + rem_j_tokens] = j_data[src_start:src_start + rem_j_tokens]
        out_pos += rem_j_tokens

    out_mmap.flush()
    del out_mmap

    out_sha = compute_sha256(output_path)

    replay_tokens_count = n_r_blocks * block_size
    jacket_tokens_count = n_j_tokens
    measured_replay_ratio = float(replay_tokens_count / total_mix_tokens)

    meta = {
        "dataset_name": "train_f4_replay_mix",
        "jacket_source": {
            "path": jacket_path,
            "sha256": jacket_sha,
            "total_tokens": int(n_j_tokens)
        },
        "replay_source": {
            "path": replay_path,
            "sha256": replay_sha,
            "total_tokens": int(n_r_tokens)
        },
        "mix_parameters": {
            "block_size": int(block_size),
            "pattern": "[jacket, jacket, jacket, replay]",
            "seed": int(seed),
            "jacket_blocks": int(n_j_blocks),
            "replay_blocks": int(n_r_blocks),
            "residual_jacket_tokens": int(rem_j_tokens)
        },
        "output_stats": {
            "path": output_path,
            "sha256": out_sha,
            "total_tokens": int(total_mix_tokens),
            "jacket_tokens": int(jacket_tokens_count),
            "replay_tokens": int(replay_tokens_count),
            "measured_replay_ratio_pct": round(measured_replay_ratio * 100.0, 4)
        }
    }

    meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Build replay mix binary dataset")
    parser.add_argument("--jacket", type=str, default="data/train_carpenter_specialization.bin", help="Jacket binary dataset")
    parser.add_argument("--replay", type=str, default="data/train_chat_balanced.bin", help="Replay binary dataset")
    parser.add_argument("--output", type=str, default="data/train_f4_replay_mix.bin", help="Output binary dataset")
    parser.add_argument("--block-size", type=int, default=128, help="Block size for pattern")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    print(f"Replay Karışımı Oluşturuluyor:")
    print(f"  Ceket : {args.jacket}")
    print(f"  Replay: {args.replay}")
    print(f"  Çıktı : {args.output}")

    meta = build_replay_mix(
        jacket_path=args.jacket,
        replay_path=args.replay,
        output_path=args.output,
        block_size=args.block_size,
        seed=args.seed
    )

    print("\nKarışım Tamamlandı:")
    print(f"  Toplam Token : {meta['output_stats']['total_tokens']:,}")
    print(f"  Ceket Token  : {meta['output_stats']['jacket_tokens']:,}")
    print(f"  Replay Token : {meta['output_stats']['replay_tokens']:,}")
    print(f"  Ölçülen Oran : %{meta['output_stats']['measured_replay_ratio_pct']:.2f}")
    print(f"  SHA-256      : {meta['output_stats']['sha256']}")
    print(f"  Meta JSON    : {args.output}.meta.json")


if __name__ == "__main__":
    main()
