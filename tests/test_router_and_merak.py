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


def test_vector_memory_hybrid_operations():
    import tempfile
    from src.rag.vector_memory import VectorMemory
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm = VectorMemory(collection_name="test_epistemic_coll", vector_size=64, storage_path=tmp_dir, host=None)
        
        # Add test doc
        dense = [0.1] * 64
        sparse = {"indices": [1, 5], "values": [0.5, 0.8]}
        meta = {"crystal_tags": "test kök CASE_DAT", "token_ids": [10, 20]}
        
        vm.add_documents_batch(
            texts=["Test ahşap belgesi."],
            dense_vectors=[dense],
            sparse_vectors=[sparse],
            metadatas=[meta]
        )
        
        assert vm.get_document_count() == 1
        
        # Recall
        results = vm.hybrid_recall(dense, sparse, top_k=1, query_tags="test")
        assert len(results) >= 1
        assert "Test ahşap belgesi" in results[0]["text"]


def test_epistemic_curiosity_loop_records_to_future_train():
    import tempfile
    import os
    import json
    from src.rag.epistemic_agent import EpistemicCuriosityAgent
    from src.rag.vector_memory import VectorMemory
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        future_file = os.path.join(tmp_dir, "future_train_vector.jsonl")
        
        # Create minimal mock vocabulary & tokenizer
        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3, "unk": 4, "test": 5}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx):
                return self.itos.get(idx, "unk")
                
        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text):
                return [0, 5, 2]
            def decode(self, ids):
                return "test"
                
        # Mock model that outputs uniform logits (high entropy = doesn't understand)
        class MockHighEntropyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 64)
            def forward(self, x, return_hidden_states=False):
                batch_size, seq_len = x.shape
                # Uniform logits across 10 tokens -> Shannon entropy ~ ln(10) = 2.30 > tau(1.5)
                logits = torch.ones(batch_size, seq_len, 10)
                hidden = torch.randn(batch_size, seq_len, 64)
                if return_hidden_states:
                    return logits, None, hidden
                return logits, None
                
        # Mock VectorMemory returning score >= 0.85
        class MockVectorMemory:
            collection_name = "mock_kristal"
            def hybrid_recall(self, dense, sparse, top_k=1, query_tags=""):
                return [{
                    "text": "Yüksek kaliteli ahşap bilgisi.",
                    "score": 0.92, # >= 0.85
                    "has_root_match": True,
                    "metadata": {"crystal_tags": "ahşap bilgi POSS_3SG", "token_ids": [5, 1]}
                }]
                
        agent = EpistemicCuriosityAgent(
            model=MockHighEntropyModel(),
            tokenizer=MockTokenizer(),
            memory=MockVectorMemory(),
            curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=1.5),
            router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
            tau=1.5,
            similarity_threshold=0.85,
            future_train_path=future_file,
            device="cpu"
        )
        
        result = agent.process_query("test sorusu?", force_rag=True)
        
        assert result["is_high_similarity"] is True
        assert result["epistemic_failure"] is True
        assert result["future_train_recorded"] is True
        assert os.path.exists(future_file)
        
        with open(future_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["similarity_score"] == 0.92
            assert entry["reason"] == "epistemic_gap_unresolved_high_similarity"
            assert "Yüksek kaliteli ahşap bilgisi" in entry["rag_document"]


def test_epistemic_curiosity_loop_skips_low_similarity():
    import tempfile
    import os
    from src.rag.epistemic_agent import EpistemicCuriosityAgent
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        future_file = os.path.join(tmp_dir, "future_train_vector.jsonl")
        
        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "unk")
        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text): return [0, 2]
            def decode(self, ids): return "test"
        class MockHighEntropyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 64)
            def forward(self, x, return_hidden_states=False):
                logits = torch.ones(x.shape[0], x.shape[1], 10)
                hidden = torch.randn(x.shape[0], x.shape[1], 64)
                return (logits, None, hidden) if return_hidden_states else (logits, None)
                
        # Return low score 0.72 (< 0.85)
        class MockLowScoreMemory:
            collection_name = "mock_kristal"
            def hybrid_recall(self, dense, sparse, top_k=1, query_tags=""):
                return [{
                    "text": "Düşük uyumlu doküman.",
                    "score": 0.72,
                    "has_root_match": True,
                    "metadata": {"crystal_tags": "düşük", "token_ids": [1]}
                }]
                
        agent = EpistemicCuriosityAgent(
            model=MockHighEntropyModel(),
            tokenizer=MockTokenizer(),
            memory=MockLowScoreMemory(),
            curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=1.5),
            router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
            tau=1.5,
            similarity_threshold=0.85,
            future_train_path=future_file,
            device="cpu"
        )
        
        result = agent.process_query("test sorusu?", force_rag=True)
        assert result["is_high_similarity"] is False
        assert result["future_train_recorded"] is False
        assert not os.path.exists(future_file)


def test_epistemic_curiosity_loop_skips_when_model_understands():
    import tempfile
    import os
    from src.rag.epistemic_agent import EpistemicCuriosityAgent
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        future_file = os.path.join(tmp_dir, "future_train_vector.jsonl")
        
        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3, "bilgi": 4}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "bilgi")
        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text): return [0, 4, 2]
            def decode(self, ids): return "bilgi"
            
        # Mock model that is very confident / understands (peaked logits -> entropy < 0.1)
        class MockConfidentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 64)
            def forward(self, x, return_hidden_states=False):
                logits = torch.zeros(x.shape[0], x.shape[1], 10)
                logits[:, :, 4] = 20.0 # Strongly peaked on token 4
                hidden = torch.randn(x.shape[0], x.shape[1], 64)
                return (logits, None, hidden) if return_hidden_states else (logits, None)
                
        class MockHighScoreMemory:
            collection_name = "mock_kristal"
            def hybrid_recall(self, dense, sparse, top_k=1, query_tags=""):
                return [{
                    "text": "Mükemmel ahşap dokümanı.",
                    "score": 0.95, # >= 0.85
                    "has_root_match": True,
                    "metadata": {"crystal_tags": "ahşap", "token_ids": [4]}
                }]
                
        agent = EpistemicCuriosityAgent(
            model=MockConfidentModel(),
            tokenizer=MockTokenizer(),
            memory=MockHighScoreMemory(),
            curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=1.5),
            router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
            tau=1.5,
            similarity_threshold=0.85,
            future_train_path=future_file,
            device="cpu"
        )
        
        result = agent.process_query("test sorusu?", force_rag=True)
        assert result["is_high_similarity"] is True
        # Model understood, so epistemic failure is False
        assert result["epistemic_failure"] is False
        assert result["future_train_recorded"] is False
        assert not os.path.exists(future_file)

