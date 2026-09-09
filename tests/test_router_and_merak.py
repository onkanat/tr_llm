import pytest
import torch
import math

from src.rag.merak import CuriosityEngine
from src.llm.router import TriModalRouter

def test_curiosity_engine_entropy():
    engine = CuriosityEngine(hidden_dim=128, curiosity_dim=128, tau=2.0)
    
    # 1. Peaked distribution (low entropy)
    peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    low_entropy = engine.calculate_entropy(peaked_logits)
    assert low_entropy.item() < 0.5
    assert not engine.detect_epistemic_gap(low_entropy).item()
    
    # 2. Uniform distribution (high entropy = ln(4) approx 1.386)
    uniform_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]) # 10 classes, ln(10) ~ 2.302
    high_entropy = engine.calculate_entropy(uniform_logits)
    assert high_entropy.item() > 2.0
    assert engine.detect_epistemic_gap(high_entropy).item()

def test_curiosity_engine_forward():
    hidden_dim = 64
    vocab_size = 100
    engine = CuriosityEngine(hidden_dim=hidden_dim, curiosity_dim=hidden_dim, tau=2.0)
    
    batch_size = 3
    hidden_states = torch.randn(batch_size, hidden_dim)
    logits = torch.randn(batch_size, vocab_size)
    
    entropy, needs_retrieval, q_merak = engine(hidden_states, logits)
    
    assert entropy.shape == (batch_size,)
    assert needs_retrieval.shape == (batch_size,)
    assert q_merak.shape == (batch_size, hidden_dim)

def test_tri_modal_router():
    router = TriModalRouter(
        prompt_dim=64,
        merak_dim=64,
        rag_dim=64,
        router_dim=32,
        num_experts=4,
        top_k=2,
        expert_names=["grammar_core", "pedagogy", "carpenter", "legal"]
    )
    
    batch_size = 2
    prompt = torch.randn(batch_size, 64)
    merak = torch.randn(batch_size, 64)
    rag = torch.randn(batch_size, 64)
    
    # Forward with all modalities
    weights, indices, e_route = router(prompt, merak, rag)
    
    assert weights.shape == (batch_size, 2)
    assert indices.shape == (batch_size, 2)
    assert e_route.shape == (batch_size, 32)
    
    # Verify weights are normalized (sum to ~1.0)
    for w in weights:
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)
        
    # Verify expert name mapping
    names = router.get_selected_expert_names(indices)
    assert len(names) == batch_size
    assert len(names[0]) == 2
    assert all(n in router.expert_names for n in names[0])

def test_tri_modal_router_partial_inputs():
    router = TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2)
    prompt = torch.randn(1, 64)
    
    # Prompt-only forward
    weights, indices, _ = router(prompt)
    assert weights.shape == (1, 2)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
