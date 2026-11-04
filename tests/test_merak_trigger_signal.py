#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.6 Kalem-1: CuriosityEngine trigger_signal OPT-IN modları (TDD).

Üç sinyal modu:
  'entropy'  (DEFAULT, mevcut donmuş davranış): mutlak H(z) >= tau
  'delta_h'  : (H(z) - H(z)_pre) >= tau          — entropy_pre ZORUNLU
  'post_h'   : mutlak H(z) >= tau, çağıran sözleşmesi: argüman POST-retrieval entropisi

Varsayılan DEĞİŞMEZ: trigger_signal verilmeden kurulan motor, entropy_pre
verilse bile mevcut mutlak-entropi tetiğiyle birebir davranır.
"""
import math
import pytest
import torch

from src.rag.merak import CuriosityEngine


def _uniform_logits(n_vocab: int, batch: int = 1) -> torch.Tensor:
    """Bilinçli entropi kontrolü: üniform softmax -> H = ln(n_vocab)."""
    return torch.zeros(batch, n_vocab)


# ---------------------------------------------------------------- RED-1
def test_invalid_trigger_signal_rejected():
    with pytest.raises(ValueError):
        CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                        trigger_signal='foo')


# ---------------------------------------------------------------- entropy (DEFAULT)
def test_entropy_mode_is_absolute_and_ignores_entropy_pre():
    """DEFAULT mod: tetik mutlak entropi üzerinedir; entropy_pre verilse bile yok sayılır."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                             trigger_signal='entropy')
    below = torch.tensor([2.4999])
    exact = torch.tensor([2.5000])
    above = torch.tensor([2.5001])
    assert not engine.detect_epistemic_gap(below).item()
    assert engine.detect_epistemic_gap(exact).item()
    assert engine.detect_epistemic_gap(above).item()
    # entropy_pre verildiğinde de aynı mutlak davranış (geriye dönük uyumluluk)
    pre = torch.tensor([9.0])
    assert not engine.detect_epistemic_gap(below, entropy_pre=pre).item()
    assert engine.detect_epistemic_gap(above, entropy_pre=pre).item()


def test_default_engine_unchanged_without_entropy_pre():
    """Yeni param hiç kullanılmadığında mevcut davranış birebir (varsayılan sabit)."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5)
    assert engine.detect_epistemic_gap(torch.tensor([2.4999])).item() is False
    assert engine.detect_epistemic_gap(torch.tensor([2.5])).item() is True


# ---------------------------------------------------------------- delta_h
def test_delta_h_mode_triggers_on_entropy_gain_not_absolute():
    """delta_h modu: tetik (H_now - H_pre) >= tau; mutlak H_now >= tau tek başına yetmez."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                             trigger_signal='delta_h')
    h_now = torch.tensor([4.0])
    # ΔH = 4.0 - 3.0 = 1.0 < 2.5 -> tetik YOK (mutlak 4.0 >= 2.5 olsa bile)
    assert not engine.detect_epistemic_gap(h_now, entropy_pre=torch.tensor([3.0])).item()
    # ΔH = 4.0 - 1.0 = 3.0 >= 2.5 -> tetik VAR
    assert engine.detect_epistemic_gap(h_now, entropy_pre=torch.tensor([1.0])).item()
    # ΔH = 4.0 - 1.5 = 2.5 (tam eşik) -> tetik VAR (>= semantics, mevcut eşikle aynı)
    assert engine.detect_epistemic_gap(h_now, entropy_pre=torch.tensor([1.5])).item()


def test_delta_h_mode_requires_entropy_pre():
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                            trigger_signal='delta_h')
    with pytest.raises(ValueError):
        engine.detect_epistemic_gap(torch.tensor([4.0]))


def test_delta_h_mode_via_forward():
    """forward entegresi: forward(..., entropy_pre=...) delta_h tetiğine akar."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                            trigger_signal='delta_h')
    hidden = torch.zeros(1, 32)
    # üniform 20 sözlük -> H = ln(20) ~ 3.00; unk_token_id=None: sinyal tetiği izole
    logits = _uniform_logits(20)
    _, needs_retrieval, _ = engine(hidden, logits, unk_token_id=None,
                                   entropy_pre=torch.tensor([1.0]))
    # ΔH = 3.00 - 1.0 = 2.0 < 2.5 -> tetik YOK
    assert not needs_retrieval.item()
    _, needs_retrieval, _ = engine(hidden, logits, unk_token_id=None,
                                   entropy_pre=torch.tensor([0.2]))
    # ΔH = 3.00 - 0.2 = 2.8 >= 2.5 -> tetik VAR
    assert needs_retrieval.item()


# ---------------------------------------------------------------- post_h
def test_post_h_mode_absolute_post_entropy():
    """post_h modu: tetik POST-retrieval entropisinin mutlak eşiğidir; entropy_pre yok sayılır."""
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                             trigger_signal='post_h')
    assert not engine.detect_epistemic_gap(torch.tensor([2.49]),
                                           entropy_pre=torch.tensor([9.0])).item()
    assert engine.detect_epistemic_gap(torch.tensor([2.51]),
                                       entropy_pre=torch.tensor([0.0])).item()


def test_post_h_mode_via_forward():
    engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5,
                             trigger_signal='post_h')
    hidden = torch.zeros(1, 32)
    # ln(12) ~ 2.485 < 2.5 -> tetik YOK; ln(13) ~ 2.565 >= 2.5 -> tetik VAR
    # unk_token_id=None: üniform dağılımdaki p(<UNK>) 2. tetikleyicisini ateşlemesin
    _, needs_low, _ = engine(hidden, _uniform_logits(12), unk_token_id=None)
    _, needs_high, _ = engine(hidden, _uniform_logits(13), unk_token_id=None)
    assert not needs_low.item()
    assert needs_high.item()
    assert math.isclose(engine.calculate_entropy(_uniform_logits(13)).item(),
                        math.log(13), rel_tol=1e-4)