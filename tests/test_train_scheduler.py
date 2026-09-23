#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2/Aşama 1: train.py scheduler'ının (get_lr) birim testi.

Kanonik kaynak: train.py:get_lr — B1 şablonunun (scripts/train_step_b1_canonical.py:201-242)
train.py'ye taşınmış hâli; KOPYA YASAK kuralı gereği testler kanonik kaynaktan import eder.

İddia sınırı: burada yalnız LR EĞRİSİ sınanır (sayısal matematik); eğitimdeki etkisi
(CE eğrisi) koşum ölçümüdür ve birim test sayılmaz.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import get_lr  # noqa: E402


def test_sifirdan_scheduler_yok_sabit_peak():
    """warmup=0, toplam=0: her adım peak (scheduler kapalı davranışın eşleniği)."""
    for step in (1, 5, 100, 10_000):
        assert get_lr(step, 2e-4, 0, 0, 1e-6) == 2e-4


def test_warmup_artis_monoton():
    """warmup eğrisi: 0→peak artar, peak'i AŞMAZ, ilk adım peak'in 1/warmup payı."""
    peak = 3e-4
    lrler = [get_lr(s, peak, 100, 0, 1e-6) for s in range(1, 101)]
    assert all(b > a for a, b in zip(lrler, lrler[1:])), "warmup monoton artmalı"
    assert abs(lrler[0] - peak / 100) < 1e-12
    assert lrler[-1] == peak, "warmup sonunda peak'e ulaşmalı, aşmamalı"
    # warmup sonrası (toplam ilan edilmedi) peak'te KALIR (sabit):
    assert get_lr(150, peak, 100, 0, 1e-6) == peak
    assert get_lr(10_000, peak, 100, 0, 1e-6) == peak


def test_cosine_yarida_ortada_bitista_min():
    """cosine: yarı yolda (peak+min)/2; bitişte tam min_lr (B1 formülü)."""
    peak, min_l = 2e-4, 1e-6
    yarida = get_lr(50, peak, 0, 100, min_l)
    assert abs(yarida - (peak + min_l) / 2) < 1e-12, yarida
    bitiste = get_lr(100, peak, 0, 100, min_l)
    assert abs(bitiste - min_l) < 1e-15, bitiste


def test_cosine_toplami_asan_adimlar_min_lrde_kalir():
    """İLERLEME TUZAĞI KAPANDI: toplam_adim aşıldığında şablonun kosinüsü geri
    YÜKSELIYORDU (progress 2.0'da peak'e döner); kırpılan sürüm min_lr'de DURUR."""
    peak, min_l = 2e-4, 1e-6
    for adim in (100, 101, 150, 10_000):
        lr = get_lr(adim, peak, 10, 100, min_l)
        assert abs(lr - min_l) < 1e-12, f"adım {adim}: lr {lr} ≠ min_lr"


def test_warmup_ve_cosine_birlikte():
    """warmup 100 + toplam 1000: ilk adım küçük, 100. adım peak, bitişte min."""
    peak, min_l = 2e-4, 1e-6
    assert get_lr(1, peak, 100, 1000, min_l) == peak / 100
    assert get_lr(100, peak, 100, 1000, min_l) == peak
    # 550. adım: progress = (550-100)/(1000-100) = 0.5 ⇒ (peak+min)/2
    orta = get_lr(550, peak, 100, 1000, min_l)
    assert abs(orta - (peak + min_l) / 2) < 1e-12
    assert get_lr(1000, peak, 100, 1000, min_l) == min_l


def test_kosum_kapisindan_ayni_egri_devam_edebilir():
    """Carry senaryosu: yan dosyadaki scheduler durumuyla (warmup/toplam/min_lr)
    kurulan eğri, aynı adımda aynı lr'yi verir (kalıcılık — eğri yeniden tepeye
    başlamamalı)."""
    peak, min_l = 1e-4, 1e-6
    e = {"warmup_steps": 100, "toplam_adim": 3000, "min_lr": min_l}
    adim = 2000  # ilk koşum bu adımda kaydetti
    lr_kayitli = get_lr(adim, peak, **e)
    # devam koşumu aynı parametrelerle (payload'dan) kurar:
    lr_devam = get_lr(adim + 1, peak, e["warmup_steps"], e["toplam_adim"], e["min_lr"])
    assert lr_devam < lr_kayitli, "devam eğrisi düşmeye devam etmeli (monoton cosine)"
    assert lr_devam > min_l * 0.99, "devam eğrisi bitişten önce min_lr'ye çakılmamalı"