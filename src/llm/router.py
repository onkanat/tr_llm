#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class TriModalRouter(nn.Module):
    """
    Faz 4: Tri-Modal Router & Dynamic MoE Gate
    Combines 3 independent channels:
      e_route = W_p(P) + W_m(q_merak) + W_r(ctx_rag)
      Gate_Weights = Softmax(TopK(e_route * W_g, k=2))
    """
    def __init__(
        self,
        prompt_dim: int = 768,
        merak_dim: int = 768,
        rag_dim: int = 768,
        router_dim: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        expert_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.merak_dim = merak_dim
        self.rag_dim = rag_dim
        self.router_dim = router_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        self.expert_names = expert_names or [f"expert_{i}" for i in range(num_experts)]
        
        # Linear projections for each modality
        self.w_p = nn.Linear(prompt_dim, router_dim)
        self.w_m = nn.Linear(merak_dim, router_dim)
        self.w_r = nn.Linear(rag_dim, router_dim)
        
        self.layer_norm = nn.LayerNorm(router_dim)
        
        # Gating projection from router representation to expert logits
        self.w_g = nn.Linear(router_dim, num_experts, bias=False)

    def forward(
        self,
        prompt_vec: torch.Tensor,
        merak_vec: Optional[torch.Tensor] = None,
        rag_vec: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            prompt_vec: Embedding from input prompt P (batch_size, prompt_dim)
            merak_vec: Curiosity vector q_merak (batch_size, merak_dim) [Optional]
            rag_vec: Retrieved context embedding ctx_rag (batch_size, rag_dim) [Optional]
            
        Returns:
            topk_weights: Normalized softmax weights for top_k experts (batch_size, top_k)
            topk_indices: Indices of selected experts (batch_size, top_k)
            e_route: Blended routing latent vector (batch_size, router_dim)
        """
        batch_size = prompt_vec.shape[0]
        device = prompt_vec.device
        
        # 1. Project prompt
        e_route = self.w_p(prompt_vec)
        
        # 2. Add merak projection if present
        if merak_vec is not None:
            e_route = e_route + self.w_m(merak_vec)
            
        # 3. Add RAG projection if present
        if rag_vec is not None:
            e_route = e_route + self.w_r(rag_vec)
            
        e_route = self.layer_norm(e_route)
        
        # 4. Gating logits
        gate_logits = self.w_g(e_route) # (batch_size, num_experts)
        
        # 5. Top-K selection
        topk_logits, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)
        
        return topk_weights, topk_indices, e_route

    def get_selected_expert_names(self, topk_indices: torch.Tensor) -> List[List[str]]:
        """Utility to map expert indices back to human-readable expert names."""
        results = []
        for row in topk_indices.tolist():
            results.append([self.expert_names[idx] for idx in row])
        return results
