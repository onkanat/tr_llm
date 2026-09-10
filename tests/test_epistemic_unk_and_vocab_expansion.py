#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import torch
import torch.nn as nn
from src.rag.merak import CuriosityEngine
from src.llm.tokenizer import Vocabulary
from src.gateway.retrain_pipeline import expand_model_vocabulary


class TestEpistemicUnkAndVocabExpansion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_curiosity_engine_triggers_on_unk(self):
        engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5)
        hidden = torch.zeros(1, 32)
        # Construct completely certain logits (entropy ~ 0.0)
        logits = torch.zeros(1, 100)
        logits[0, 5] = 100.0  # token 5 is chosen with ~100% probability

        # Without UNK: needs_retrieval must be False
        entropy, needs_retrieval, _ = engine(hidden, logits, has_unk=False)
        self.assertFalse(needs_retrieval.item())
        self.assertLess(entropy.item(), 0.1)

        # With UNK: needs_retrieval MUST be True, regardless of low entropy!
        entropy_unk, needs_retrieval_unk, _ = engine(hidden, logits, has_unk=True)
        self.assertTrue(needs_retrieval_unk.item())
        self.assertGreater(entropy_unk.item(), engine.tau)

    def test_curiosity_engine_triggers_on_predictive_unk(self):
        engine = CuriosityEngine(hidden_dim=32, curiosity_dim=32, tau=2.5)
        hidden = torch.zeros(1, 32)
        logits = torch.zeros(1, 100)
        # Give token 1 (<UNK>) significant probability
        logits[0, 1] = 10.0
        logits[0, 2] = 10.0

        _, needs_retrieval, _ = engine(hidden, logits, has_unk=False, unk_token_id=1)
        self.assertTrue(needs_retrieval.item())

    def test_vocabulary_register_new_tokens(self):
        vocab = Vocabulary()
        initial_size = len(vocab.stoi)
        vocab.freeze()

        # Normal add_token should do nothing when frozen
        vocab.add_token("mitokondri")
        self.assertEqual(len(vocab.stoi), initial_size)

        # register_new_tokens should safely register even if frozen
        new_ids = vocab.register_new_tokens(["mitokondri", "ribozom", "kuantum"])
        self.assertEqual(len(new_ids), 3)
        self.assertEqual(len(vocab.stoi), initial_size + 3)
        self.assertTrue(vocab.frozen)  # Re-frozen after registration
        self.assertIn("mitokondri", vocab.stoi)
        self.assertEqual(vocab.decode(new_ids[0]), "mitokondri")

    def test_expand_model_vocabulary_weight_surgery(self):
        # Mock model with embedding and lm_head
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, n_embd):
                super().__init__()
                self.embedding = nn.Sequential()
                # Emulate KristalLM structure: model.embedding.embedding
                self.embedding.embedding = nn.Embedding(vocab_size, n_embd)
                self.lm_head = nn.Linear(n_embd, vocab_size)
                self.vocab_size = vocab_size

            def forward(self, x):
                return self.lm_head(self.embedding.embedding(x))

        vocab = Vocabulary()
        base_vocab_size = len(vocab.stoi)
        n_embd = 16
        model = SimpleLM(vocab_size=base_vocab_size, n_embd=n_embd)

        # Record original weights of token 5
        original_token5_emb = model.embedding.embedding.weight[5].clone()
        original_token5_head = model.lm_head.weight[5].clone()

        # Expand vocabulary with 3 new morphemes
        new_words = ["mitokondri", "ribozom", "kloroplast"]
        model, new_ids = expand_model_vocabulary(model, vocab, new_words)

        self.assertEqual(model.vocab_size, base_vocab_size + 3)
        self.assertEqual(model.embedding.embedding.weight.shape[0], base_vocab_size + 3)
        self.assertEqual(model.lm_head.weight.shape[0], base_vocab_size + 3)

        # ZERO-FORGETTING VERIFICATION: Existing token 5 weights must match 100%
        self.assertTrue(torch.equal(model.embedding.embedding.weight[5], original_token5_emb))
        self.assertTrue(torch.equal(model.lm_head.weight[5], original_token5_head))

        # NEW TOKEN FORWARD PASS: Inference on new tokens must run without IndexError!
        new_token_tensor = torch.tensor([[new_ids[0], new_ids[1], new_ids[2]]], dtype=torch.long)
        out = model(new_token_tensor)
        self.assertEqual(out.shape, (1, 3, base_vocab_size + 3))


if __name__ == "__main__":
    unittest.main()
