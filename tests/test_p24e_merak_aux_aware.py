#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P24e Kalem-4: ΔH sinyalinin aux çağına uyarlanması — 'aux_aware' OP-İN modu (TDD).

ZEMİN (p24c, çıpalı envanter): delta_h@−0,85 çürüdü (J +0,113, CI 0 içerir;
aux 32/107 flip, belirsiz 28/28 koşullandı — ayrım öldü). OFFLINE tarama
(p24c 107 ham kaydı; bootstrap 1000, kategori-tabakalı, tohum 20261104):
0-DIŞLAYAN kaynak-koşullu kural VAR —
  aux→post_h≥2,25 | birincil→ΔH≥−0,75 : J=+0,297 CI [+0,091;+0,516] tetik %28
  (her iki kol aktif: aux kolunda 11, birincil kolunda 3 belirsiz tetiklenir;
  plato geniş: tA∈[2,25;2,5], tB=−0,75 çevresi J +0,262…+0,297).
Karşı aday F2 (cond→post_h≥2,75 | cond-değil→ΔH≥0,0): J=+0,322 CI
[+0,116;+0,515] ama ikinci kol VAKUM (cond=False kalan 16 sorgunun dH≥0
olanı 0) → raporda alternatif olarak kalır.

SÖZLEŞME: 'aux_aware' modu context_is_aux ZORUNLU; aux kolunda mutlak
post-entropi ≥ tau_aux, birincil koloda ΔH ≥ tau_primary (entropy_pre
ZORUNLU). Varsayılan mod DEĞİŞMEZ: 'entropy'/τ=2,5 birebir; yeni
parametreler varsayılan motorda yok sayılır.
"""
import pytest
import torch

from src.rag.merak import CuriosityEngine


# ------------------------------------------------ varsayılan birebir (BIT-ÖZDEŞ)
def test_varsayilan_mod_yeni_parametreleri_yok_sayar():
    """'entropy' (default) motoru: tau_aux/tau_primary/context_is_aux verilse
    bile mutlak entropi tetiği birebir (P2.6 donmuş davranış korunur)."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                             tau_aux=0.5, tau_primary=0.0)
    assert not engine.detect_epistemic_gap(torch.tensor([2.4999]),
                                            context_is_aux=torch.tensor([True])).item()
    assert engine.detect_epistemic_gap(torch.tensor([2.5]),
                                       context_is_aux=torch.tensor([False])).item()
    # entropy_pre de yok sayılır (P2.6 mevcut testle aynı iddia)
    assert not engine.detect_epistemic_gap(torch.tensor([2.4999]),
                                          entropy_pre=torch.tensor([0.0])).item()


def test_gecersiz_trigger_signal_hala_reddedilir():
    with pytest.raises(ValueError):
        CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='foo')


def test_aux_aware_gecersiz_deger_hala_reddedilir_uyarida_listelenir():
    with pytest.raises(ValueError) as ei:
        CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='nebula')
    assert "aux_aware" in str(ei.value)


# ------------------------------------------------ aux_aware: kapılar (fail-closed)
def test_aux_aware_context_is_aux_zorunlu():
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware')
    with pytest.raises(ValueError):
        engine.detect_epistemic_gap(torch.tensor([4.0]))


def test_aux_aware_birincil_kol_entropy_pre_zorunlu():
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware')
    with pytest.raises(ValueError):
        engine.detect_epistemic_gap(torch.tensor([4.0]), context_is_aux=torch.tensor([False]))


# ------------------------------------------------ aux_aware: dondurulmuş ölçüm
def test_aux_aware_dondurulmus_esikler():
    """Eşikler p24c çıpalı-envanter ölçümüyle donduruldu: tau_aux=2,25
    (post_h), tau_primary=−0,75 (ΔH). Drift-guard."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware')
    assert engine.tau_aux == pytest.approx(2.25)
    assert engine.tau_primary == pytest.approx(-0.75)


def test_aux_aware_aux_kolu_post_entropi_esigi():
    """aux kol: tetik post-entropinin MUTLAK eşiğidir (ΔH aux çağında bilgi
    taşımaz — p24c §6). entropy_pre yok sayılır."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware')
    # p24c #91 ölçümü: aux kaynağa H_post 4,21 → tetik (post_h 4,21 ≥ 2,25)
    assert engine.detect_epistemic_gap(torch.tensor([4.21]),
                                       context_is_aux=torch.tensor([True])).item()
    # H_post 2,24 < 2,25 → tetik yok; entropy_pre (yüksek) sonucu DEĞİŞTİRMEZ
    assert not engine.detect_epistemic_gap(torch.tensor([2.24]),
                                           entropy_pre=torch.tensor([9.0]),
                                           context_is_aux=torch.tensor([True])).item()
    # tam eşik (>= semantiği mevcut modlarla aynı)
    assert engine.detect_epistemic_gap(torch.tensor([2.25]),
                                       context_is_aux=torch.tensor([True])).item()


def test_aux_aware_birincil_kolu_delta_h_esigi():
    """birincil kol: tetik (H − H_pre) >= −0,75 — ΔH birincil zincirde yaşıyor
    (aux çağı uyarlamasının amacı bu)."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware')
    # ΔH = 4,0 − 4,5 = −0,5 ≥ −0,75 → tetik
    assert engine.detect_epistemic_gap(torch.tensor([4.0]),
                                       entropy_pre=torch.tensor([4.5]),
                                       context_is_aux=torch.tensor([False])).item()
    # ΔH = 4,0 − 5,0 = −1,0 < −0,75 → tetik yok
    assert not engine.detect_epistemic_gap(torch.tensor([4.0]),
                                           entropy_pre=torch.tensor([5.0]),
                                           context_is_aux=torch.tensor([False])).item()
    # tam eşik: ΔH = −0,75 → tetik
    assert engine.detect_epistemic_gap(torch.tensor([4.25]),
                                       entropy_pre=torch.tensor([5.0]),
                                       context_is_aux=torch.tensor([False])).item()


def test_aux_aware_cagiran_ozel_esik_verebilir():
    """Eşikler parametredir (çıpalı ölçümün VARSAYILANI; envanter değişince
    yeniden ölçümle çağıran günceller)."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware',
                            tau_aux=6.0, tau_primary=0.0)
    assert not engine.detect_epistemic_gap(torch.tensor([5.99]),
                                           context_is_aux=torch.tensor([True])).item()
    assert engine.detect_epistemic_gap(torch.tensor([6.0]),
                                       context_is_aux=torch.tensor([True])).item()
    # ΔH = 2.0 − 2.0 = 0.0 >= 0.0 (tam eşik) → tetik; ΔH = 1.9 − 2.0 = −0.1 < 0.0 → yok
    assert engine.detect_epistemic_gap(torch.tensor([2.0]),
                                       entropy_pre=torch.tensor([2.0]),
                                       context_is_aux=torch.tensor([False])).item()
    assert not engine.detect_epistemic_gap(torch.tensor([1.9]),
                                           entropy_pre=torch.tensor([2.0]),
                                           context_is_aux=torch.tensor([False])).item()


# ------------------------------------------------ forward entegresi
def test_aux_aware_forward_entegresi():
    """forward(..., context_is_aux=...) aux_aware tetiğine akar; unk_token_id
    None + has_unk False ile sinyal tetiği izole edilir."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, trigger_signal='aux_aware')
    hidden = torch.zeros(1, 32)
    # ln(10) ~ 2.303 ≥ 2.25 → aux kol tetik
    _, fire_aux, _ = engine(hidden, torch.zeros(1, 10), unk_token_id=None,
                            context_is_aux=torch.tensor([True]))
    assert fire_aux.item()
    # ln(10) ~ 2.303; ΔH = 2.303 − 2.9 = −0.597 ≥ −0.75 → birincil kol tetik
    _, fire_prim, _ = engine(hidden, torch.zeros(1, 10), unk_token_id=None,
                             entropy_pre=torch.tensor([2.9]),
                             context_is_aux=torch.tensor([False]))
    assert fire_prim.item()
    # ΔH = 2.303 − 3.5 = −1.197 < −0.75 → tetik yok
    _, fire_prim2, _ = engine(hidden, torch.zeros(1, 10), unk_token_id=None,
                              entropy_pre=torch.tensor([3.5]),
                              context_is_aux=torch.tensor([False]))
    assert not fire_prim2.item()
