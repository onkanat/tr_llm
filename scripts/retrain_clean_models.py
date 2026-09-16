#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: TEMİZ MODEL VE UZMANLIK EĞİTİM ORKESTRATÖRÜ
============================================================
Arındırılmış veri kümeleriyle:
1. Temel Model:
   - Stage 1: Pretraining (200 adım, train.bin)
   - Stage 2: Dengeli SFT (200 adım, train_balanced_sft_v2.bin)
   - Stage 3: Chat SFT (200 adım, train_chat_balanced.bin)
   - Stage 4: DPO Tercih Hizalama (100 adım, dpo_all_tokenized.jsonl)
2. Marangozluk Modeli:
   - Uzmanlık Eğitimi (150 adım, train_carpenter_specialization.bin)
"""

import os
import sys
import time
import shutil
import subprocess
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.frozen_guard import check_frozen_save_path

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"


def run_cmd(cmd_list, desc):
    print(f"\n{C_CYAN}[Komut]: {' '.join(cmd_list)}{C_RESET} ({desc})", flush=True)
    t0 = time.time()
    subprocess.run(cmd_list, check=True)
    dt = time.time() - t0
    print(f"{C_GREEN}[Tamamlandı]: {desc} ({dt:.1f} sn){C_RESET}\n", flush=True)
    return dt


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    python_bin = sys.executable

    print(f"{C_BOLD}{C_MAGENTA}======================================================================")
    print("  KRİSTAL-VEKTÖREL: ARINMIŞ MODEL EĞİTİM BORU HATTI")
    print("======================================================================")
    print(f"Cihaz: {device}")
    print("======================================================================\n" + C_RESET, flush=True)

    t_total = time.time()

    start_stage = 1
    allow_frozen_write = "--allow-frozen-write" in sys.argv
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--resume-stage", "--stage") and arg_idx + 1 < len(sys.argv):
            start_stage = int(sys.argv[arg_idx + 1])

    # Backup current models before retraining if starting at stage 1
    if start_stage == 1 and os.path.exists("data/kristal_model.pt"):
        check_frozen_save_path("data/kristal_model_pre_clean.pt", allow_frozen_write=allow_frozen_write)
        shutil.copyfile("data/kristal_model.pt", "data/kristal_model_pre_clean.pt")
        print("Mevcut model data/kristal_model_pre_clean.pt olarak yedeklendi.", flush=True)

    # -------------------------------------------------------------------------
    # 1. Stage-1: Pretraining (from scratch)
    # -------------------------------------------------------------------------
    if start_stage <= 1:
        stage1_cmd = [
            python_bin, "train.py",
            "--device", device,
            "--data", "data/train.bin",
            "--steps", "200",
            "--from-scratch",
            "--save-path", "data/kristal_model.pt"
        ]
        if allow_frozen_write:
            stage1_cmd.append("--allow-frozen-write")
        check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
        run_cmd(stage1_cmd, "Stage-1 Pretraining (200 adım)")

    # -------------------------------------------------------------------------
    # 2. Stage-2: Balanced SFT Fine-Tuning
    # -------------------------------------------------------------------------
    if start_stage <= 2:
        stage2_cmd = [
            python_bin, "train.py",
            "--device", device,
            "--data", "data/train_balanced_sft_v2.bin",
            "--steps", "200",
            "--load-path", "data/kristal_model.pt",
            "--save-path", "data/kristal_model.pt"
        ]
        if allow_frozen_write:
            stage2_cmd.append("--allow-frozen-write")
        check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
        run_cmd(stage2_cmd, "Stage-2 Dengeli SFT (200 adım)")

    # -------------------------------------------------------------------------
    # 3. Stage-3: Chat SFT Fine-Tuning
    # -------------------------------------------------------------------------
    if start_stage <= 3:
        stage3_cmd = [
            python_bin, "train.py",
            "--device", device,
            "--data", "data/train_chat_balanced.bin",
            "--steps", "200",
            "--batch-size", "16",
            "--load-path", "data/kristal_model.pt",
            "--save-path", "data/kristal_model.pt"
        ]
        if allow_frozen_write:
            stage3_cmd.append("--allow-frozen-write")
        check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
        run_cmd(stage3_cmd, "Stage-3 Chat SFT (200 adım)")

    # -------------------------------------------------------------------------
    # 4. Stage-4: DPO Alignment (CPU)
    # -------------------------------------------------------------------------
    if start_stage <= 4:
        check_frozen_save_path("data/kristal_model_sft.pt", allow_frozen_write=allow_frozen_write)
        shutil.copyfile("data/kristal_model.pt", "data/kristal_model_sft.pt")
        stage4_cmd = [
            python_bin, "train_dpo.py",
            "--steps", "30"
        ]
        if allow_frozen_write:
            stage4_cmd.append("--allow-frozen-write")
        check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
        run_cmd(stage4_cmd, "Stage-4 DPO Tercih Hizalama (30 adım)")

    print(f"\n{C_BOLD}{C_GREEN}>>> TEMEL MODEL EĞİTİMİ TAMAMLANDI: data/kristal_model.pt <<<{C_RESET}\n", flush=True)

    # -------------------------------------------------------------------------
    # 5. Carpenter Specialization Training
    # -------------------------------------------------------------------------
    if start_stage <= 5:
        stage5_cmd = [
            python_bin, "train.py",
            "--device", device,
            "--base-model", "data/kristal_model.pt",
            "--output-model", "data/kristal_carpenter_model.pt",
            "--data", "data/train_carpenter_specialization.bin",
            "--steps", "150",
            "--batch-size", "16",
            "--lr", "0.0003"
        ]
        if allow_frozen_write:
            stage5_cmd.append("--allow-frozen-write")
        check_frozen_save_path("data/kristal_carpenter_model.pt", allow_frozen_write=allow_frozen_write)
        run_cmd(stage5_cmd, "Marangozluk Uzmanlık Eğitimi (150 adım)")

    dt_all = time.time() - t_total
    print(f"\n{C_BOLD}{C_GREEN}======================================================================")
    print(f"  TÜM MODEL EĞİTİMLERİ BAŞARIYLA TAMAMLANDI ({dt_all:.1f} sn)!")
    print("  1. data/kristal_model.pt (Temiz Temel Model)")
    print("  2. data/kristal_carpenter_model.pt (Marangozluk Modeli)")
    print("======================================================================\n{C_RESET}", flush=True)


if __name__ == "__main__":
    main()
