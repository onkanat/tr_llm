#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0129 / G1: train.py CLI argümanları ve optimizer politikası fail-closed testleri.

Kabul Kriterleri (T-0129):
1. train.py --steps olmadan koşum: rc != 0 ve hata mesajı 'steps belirtilmedi' içeriği
   (SFT ve pretrain yollarında ayrı ayrı).
2. --steps X --toplam-adim Y çelişkisi: X > Y durumunda rc != 0 ve RuntimeError.
3. Optimizer yan dosyası VAR + --load-optimizer yok: rc != 0 (yalnız --optimizer-fresh ile geçer);
   üç durumda da tek satır görünür beyan.
4. Yan dosya yok: Tek satır görünür beyan ('[OPTIMIZER] BEYAN: Yan dosya yok').
"""
import subprocess
import sys
from pathlib import Path
import pytest

KOK = Path(__file__).resolve().parent.parent
BETIK = KOK / "train.py"
VOCAB = "data/rebuild/vocab_anka_r1_33114.json"
VERI = "data/train.bin"

ORTAK = [
    "--device", "cpu",
    "--data", VERI,
    "--vocab", VOCAB,
    "--batch-size", "1",
    "--block-size", "32",
    "--seed", "1234",
]


def _kosum(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BETIK), *args],
        cwd=str(KOK),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_steps_verilmezse_pretrain_fail_closed(tmp_path):
    """Kriter 2a: Pretrain yolunda --steps verilmezse fail-closed RuntimeError."""
    save_path = tmp_path / "m.pt"
    res = _kosum(*ORTAK, "--pretrain", "--from-scratch", "--save-path", str(save_path))
    assert res.returncode != 0, f"rc={res.returncode}, sıfır dönmemeliydi"
    err = res.stderr + res.stdout
    assert "steps belirtilmedi" in err, f"Hata mesajında 'steps belirtilmedi' bulunamadı:\n{err}"
    assert "DURDURULDU" in err


def test_steps_verilmezse_sft_fail_closed(tmp_path):
    """Kriter 2b: SFT yolunda --steps verilmezse fail-closed RuntimeError."""
    save_path = tmp_path / "m.pt"
    res = _kosum(*ORTAK, "--from-scratch", "--save-path", str(save_path))
    assert res.returncode != 0, f"rc={res.returncode}, sıfır dönmemeliydi"
    err = res.stderr + res.stdout
    assert "steps belirtilmedi" in err, f"Hata mesajında 'steps belirtilmedi' bulunamadı:\n{err}"
    assert "DURDURULDU" in err


def test_steps_toplam_adim_celiskisi_fail_closed(tmp_path):
    """Kriter 3a (regresyon): --steps X > --toplam-adim Y çelişkisi: rc != 0 ve RuntimeError."""
    save_path = tmp_path / "m.pt"
    res = _kosum(
        *ORTAK,
        "--pretrain",
        "--from-scratch",
        "--steps", "10",
        "--toplam-adim", "5",
        "--save-path", str(save_path),
    )
    assert res.returncode != 0, f"rc={res.returncode}, çelişkide sıfır dönmemeliydi"
    err = res.stderr + res.stdout
    assert "DURDURULDU" in err
    assert "Cosine periyodu adım sayısından küçük olamaz" in err or "çelişkili" in err or "çelişkisi" in err


def test_steps_toplam_adim_ters_yon_celiskisi_fail_closed(tmp_path):
    """Kriter 3b (G1 ONARIM): --steps X < --toplam-adim Y çelişkisi: rc != 0 ve RuntimeError."""
    save_path = tmp_path / "m.pt"
    res = _kosum(
        *ORTAK,
        "--pretrain",
        "--from-scratch",
        "--steps", "10",
        "--toplam-adim", "200",
        "--save-path", str(save_path),
    )
    assert res.returncode != 0, f"rc={res.returncode}, ters yön çelişkide sıfır dönmemeliydi"
    err = res.stderr + res.stdout
    assert "DURDURULDU" in err
    assert "erken biter" in err or "çelişkisi" in err


def test_optimizer_yan_dosya_var_load_yok_fail_closed(tmp_path):
    """Kriter 4a: Yan dosya var + --load-optimizer yok -> rc != 0 (fail-closed)."""
    m1 = tmp_path / "m1.pt"
    r1 = _kosum(
        *ORTAK,
        "--pretrain",
        "--from-scratch",
        "--steps", "2",
        "--save-path", str(m1),
        "--save-optimizer",
    )
    assert r1.returncode == 0, f"1. koşum düştü:\n{r1.stderr}"
    assert (tmp_path / "m1.pt.opt.pt").exists(), "Yan dosya oluşmadı"

    m2 = tmp_path / "m2.pt"
    # Devam koşumu: m1 yüklenecek, yan dosya var ama --load-optimizer ve --optimizer-fresh YOK
    r2 = _kosum(
        *ORTAK,
        "--pretrain",
        "--steps", "2",
        "--load-path", str(m1),
        "--save-path", str(m2),
    )
    assert r2.returncode != 0, f"Yan dosya varken bayraksız koşum fail-closed olmalıydı (rc={r2.returncode})"
    err = r2.stderr + r2.stdout
    assert "DURDURULDU" in err
    assert "--optimizer-fresh" in err


def test_optimizer_yan_dosya_var_fresh_ile_gecer(tmp_path):
    """Kriter 4b: Yan dosya var + --optimizer-fresh var -> rc == 0 ve beyan basılır."""
    m1 = tmp_path / "m1.pt"
    r1 = _kosum(
        *ORTAK,
        "--pretrain",
        "--from-scratch",
        "--steps", "2",
        "--save-path", str(m1),
        "--save-optimizer",
    )
    assert r1.returncode == 0, f"1. koşum düştü:\n{r1.stderr}"

    m2 = tmp_path / "m2.pt"
    r2 = _kosum(
        *ORTAK,
        "--pretrain",
        "--steps", "2",
        "--load-path", str(m1),
        "--save-path", str(m2),
        "--optimizer-fresh",
    )
    assert r2.returncode == 0, f"--optimizer-fresh ile koşum geçmeliydi:\n{r2.stderr}"
    out = r2.stdout
    assert "[OPTIMIZER] BEYAN: Yan dosya mevcut" in out
    assert "--optimizer-fresh" in out


def test_optimizer_yan_dosya_yok_tek_satir_beyan(tmp_path):
    """Kriter 4c: Yan dosya yokken tek satır görünür beyan basılır."""
    save_path = tmp_path / "m.pt"
    res = _kosum(
        *ORTAK,
        "--pretrain",
        "--from-scratch",
        "--steps", "2",
        "--save-path", str(save_path),
    )
    assert res.returncode == 0, f"Koşum düştü:\n{res.stderr}"
    out = res.stdout
    assert "[OPTIMIZER] BEYAN: Yan dosya yok" in out
