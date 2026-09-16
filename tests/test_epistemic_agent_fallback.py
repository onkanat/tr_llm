import os
import tempfile
import json
import torch
from src.rag.epistemic_agent import EpistemicCuriosityAgent
from src.rag.merak import CuriosityEngine
from src.llm.router import TriModalRouter


def test_epistemic_curiosity_loop_empty_doc_fallback():
    """
    Test epistemic agent when retrieved doc has empty text (text="") but match_score >= 0.85.
    Must gracefully fallback to doc_crystal_tags or morpheme_output without raising NameError.
    """
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

        class MockEmptyDocMemory:
            collection_name = "mock_kristal"
            def hybrid_recall(self, dense, sparse, top_k=1, query_tags=""):
                return [{
                    "text": "", # Empty document text
                    "score": 0.90, # >= 0.85 high similarity
                    "has_root_match": True,
                    "metadata": {"crystal_tags": "kristal etiketler", "token_ids": [1]}
                }]

        agent = EpistemicCuriosityAgent(
            model=MockHighEntropyModel(),
            tokenizer=MockTokenizer(),
            memory=MockEmptyDocMemory(),
            curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=1.5),
            router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
            tau=1.5,
            similarity_threshold=0.85,
            future_train_path=future_file,
            device="cpu"
        )

        result = agent.process_query("test sorusu?", force_rag=True)

        assert result is not None
        assert os.path.exists(future_file)
        with open(future_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["output"] == "kristal etiketler"
