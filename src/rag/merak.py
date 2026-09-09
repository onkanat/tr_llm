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
    """
    def __init__(self, hidden_dim: int = 768, curiosity_dim: int = 768, tau: float = 2.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.curiosity_dim = curiosity_dim
        self.tau = tau
        
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

    def detect_epistemic_gap(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Returns boolean mask indicating where entropy exceeds threshold tau.
        """
        return entropy > self.tau

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Last layer hidden states z (batch_size, hidden_dim)
            logits: Next token prediction logits (batch_size, vocab_size)
        Returns:
            entropy: Calculated Shannon entropy H(z)
            needs_retrieval: Boolean tensor (H(z) > tau)
            q_merak: Synthesized curiosity vector (batch_size, curiosity_dim)
        """
        entropy = self.calculate_entropy(logits)
        needs_retrieval = self.detect_epistemic_gap(entropy)
        
        q_merak = self.norm(self.q_proj(hidden_states))
        return entropy, needs_retrieval, q_merak
