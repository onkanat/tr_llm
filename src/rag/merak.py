#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class CuriosityEngine(nn.Module):
    """
    MERAK (Curiosity Engine / Epistemic Gap Detector):
    Quantifies internal model uncertainty (Shannon Entropy H(z)).
    When uncertainty exceeds threshold tau (H(z) > tau), synthesizes a targeted
    Curiosity Vector (q_merak) to guide RAG ocean navigation and resolve ambiguity.
    OPT-IN trigger signals (P2.6): trigger_signal='entropy' (DEFAULT, frozen
    absolute-entropy trigger, bit-identical to previous behavior) | 'delta_h'
    (entropy gain: (H(z) - entropy_pre) >= tau, entropy_pre REQUIRED) |
    'post_h' (absolute threshold on the POST-retrieval entropy the caller
    passes in). The frozen tau=2.5 default is NOT changed by this parameter.
    OPT-IN 'aux_aware' (P24e Kalem-4, 2026-11-05): kaynak-koşullu eşik —
    aux kaynaklı bağlamda (context_is_aux=True) POST entropi >= tau_aux;
    birincil zincirde (context_is_aux=False) ΔH = (H(z) − entropy_pre) >=
    tau_primary (entropy_pre REQUIRED). Gerekçe (p24c çıpalı-envanter
    ölçümü, data/eval/p24c_esik_dagilim_2026-11-04.json): aux ekimi ΔH'nin
    ayrım gücünü öldürdü (belirsiz 28/28 koşullandı; delta_h@−0,85 J +0,113,
    CI 0 içerir) — ΔH yalnız BİRİNCİL zincirde bilgi taşır, aux'ta post_h
    taşır. Dondurulmuş ölçülmüş eşikler (107 sorgu; bootstrap 1000,
    kategori-tabakalı, tohum 20261104): tau_aux=2,25 / tau_primary=−0,75 →
    J=+0,297, CI [+0,091;+0,516] (0'ı DIŞLAR), tetik %28.
    ÇIPALAMA KURALI (bkz. p24c HÜKÜM §2): bu eşikler ölçüldükleri Qdrant
    envanterine çıpalıdır (anka_bellek 46 / muhakeme_bellek 91 /
    simulasyon_bellek 4.073 / gts_bellek 128.327 / kanun_madde 4.068 /
    yetenek_agi 36). ENVANTER DEĞİŞİNCE — yeni koleksiyon ekimi, mevcut
    koleksiyona ekim/silme, aux zinciri değişikliği — eşikler YÜRÜRLÜKTEN
    DÜŞER: yeniden ölçüm + yeniden çıpalam yapılmadan kullanılmaz.
    """
    def __init__(self, hidden_dim: int = 768, curiosity_dim: int = 768, tau: float = 2.5,
                 trigger_signal: str = 'entropy',
                 tau_aux: float = 2.25, tau_primary: float = -0.75):
        super().__init__()
        if trigger_signal not in ('entropy', 'delta_h', 'post_h', 'aux_aware'):
            raise ValueError(
                f"Geçersiz trigger_signal: {trigger_signal!r}; "
                "geçerli değerler: 'entropy' | 'delta_h' | 'post_h' | 'aux_aware'")
        self.hidden_dim = hidden_dim
        self.curiosity_dim = curiosity_dim
        self.tau = tau
        self.trigger_signal = trigger_signal
        # P24e aux_aware eşikleri — dondurulmuş ölçüm değerleri (çıpalama
        # kuralı yukarıda); çağıran yeniden ölçümle GÜNCELLEYEBİLİR.
        self.tau_aux = float(tau_aux)
        self.tau_primary = float(tau_primary)
        
        # Linear projection to synthesize curiosity vector from hidden state z
        self.q_proj = nn.Linear(hidden_dim, curiosity_dim)
        self.norm = nn.LayerNorm(curiosity_dim)

    def calculate_entropy(self, logits: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        Calculates Shannon Entropy H(z) = - sum(p * log(p)) over next-token logits.
        Args:
            logits: Tensor of shape (batch_size, vocab_size) or (batch_size, seq_len, vocab_size)
        Returns:
            entropy: Tensor of shape (batch_size,) or (batch_size, seq_len)
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def detect_epistemic_gap(self, entropy: torch.Tensor,
                            entropy_pre: Optional[torch.Tensor] = None,
                            context_is_aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns boolean mask indicating where the trigger signal exceeds
        threshold tau (tau >= 2.5).

        trigger_signal modes (P2.6 opt-in / P24e Kalem-4):
          'entropy' (DEFAULT) — absolute H(z) >= tau; entropy_pre and
          context_is_aux are IGNORED.
          'delta_h' — entropy gain (H(z) - entropy_pre) >= tau; entropy_pre is
          REQUIRED (ValueError when absent).
          'post_h' — absolute H(z) >= tau where the caller passes the
          POST-retrieval entropy; entropy_pre is IGNORED.
          'aux_aware' (P24e) — kaynak-koşullu: context_is_aux REQUIRED
          (ValueError when absent). True → absolute H(z) >= tau_aux
          (post-retrieval entropy; entropy_pre IGNORED — ΔH aux çağında
          bilgi taşımaz, p24c §6); False → (H(z) - entropy_pre) >=
          tau_primary (entropy_pre REQUIRED — ΔH birincil zincirde yaşar).
        """
        if self.trigger_signal == 'delta_h':
            if entropy_pre is None:
                raise ValueError(
                    "trigger_signal='delta_h' entropy_pre gerektirir: "
                    "H(z)_pre olmaksızın ΔH tanımsızdır")
            return (entropy - entropy_pre) >= self.tau
        if self.trigger_signal == 'aux_aware':
            if context_is_aux is None:
                raise ValueError(
                    "trigger_signal='aux_aware' context_is_aux gerektirir: "
                    "kaynak (aux/birincil) bilinmeden kaynak-koşullu eşik tanımsızdır")
            if bool(context_is_aux):
                return entropy >= self.tau_aux
            if entropy_pre is None:
                raise ValueError(
                    "trigger_signal='aux_aware' birincil zincirde entropy_pre gerektirir: "
                    "H(z)_pre olmaksızın ΔH tanımsızdır")
            return (entropy - entropy_pre) >= self.tau_primary
        return entropy >= self.tau

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        has_unk: bool = False,
        unk_token_id: Optional[int] = 1,
        entropy_pre: Optional[torch.Tensor] = None,
        context_is_aux: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Last layer hidden states z (batch_size, hidden_dim)
            logits: Next token prediction logits (batch_size, vocab_size)
            has_unk: Boolean flag indicating presence of <UNK> morpheme in input context
            unk_token_id: Token ID for <UNK> to detect predictive epistemic gap
            entropy_pre: Optional pre-retrieval entropy H(z)_pre; consumed only by
            trigger_signal='delta_h' and 'aux_aware' birincil kolu (P2.6/P24e
            opt-in), ignored otherwise
            context_is_aux: Optional source flag (True: aux koleksiyondan gelen
            bağlam; False: birincil ANKA zinciri); consumed only by
            trigger_signal='aux_aware' (P24e opt-in), ignored otherwise
        Returns:
            entropy: Calculated Shannon entropy H(z)
            needs_retrieval: Boolean tensor (signal >= tau or has_unk)
            q_merak: Synthesized curiosity vector (batch_size, curiosity_dim)
        """
        entropy = self.calculate_entropy(logits)
        needs_retrieval = self.detect_epistemic_gap(
            entropy, entropy_pre=entropy_pre, context_is_aux=context_is_aux)
        
        # Epistemic trigger 1: Context contains <UNK> morpheme
        if has_unk:
            needs_retrieval = torch.ones_like(needs_retrieval, dtype=torch.bool)
            entropy = entropy + self.tau + 1.0  # Epistemic curiosity boost strictly exceeding tau

        # Epistemic trigger 2: Model prediction assigns non-trivial probability to <UNK>
        if unk_token_id is not None and logits.shape[-1] > unk_token_id:
            probs = torch.softmax(logits, dim=-1)
            unk_p = probs[..., unk_token_id]
            predicts_unk = unk_p > 0.05
            if predicts_unk.any():
                needs_retrieval = needs_retrieval | predicts_unk
        
        q_merak = self.norm(self.q_proj(hidden_states))
        return entropy, needs_retrieval, q_merak

