#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_moe_adversarial_stress.py
====================================
Milestone M4 Challenger 1: Adversarial Stress Test Suite for MoE Dynamic Routing in AgentGateway.

Targeted Test Categories:
1. Carpentry prompts (zıvana geçme, rendeleme, meşe kereste, etc.) -> strictly 'carpenter'
2. Gardening prompts (domates budama, aşı kalemi, gübreleme, etc.) -> strictly 'gardener'
3. Pedagogy prompts (ilkokul seviyesinde anlat, pedagojik rehberlik, etc.) -> strictly 'pedagogy'
4. Cross-domain ambiguity and blend resolution
5. Zero embeddings, extreme values, and empty/whitespace inputs
6. Mock vs Live mode architectural consistency and execution dispatch
"""

import os
import unittest
import torch
import torch.nn as nn

from src.gateway.agent_gateway import AgentGateway
from src.rag.vector_memory import VectorMemory
from src.llm.router import TriModalRouter
from src.rag.merak import CuriosityEngine


class TestMoEAdversarialStress(unittest.TestCase):
    """Adversarial stress test harness for AgentGateway MoE Dynamic Routing."""

    @classmethod
    def setUpClass(cls):
        """Initializes live AgentGateway once for live checkpoint test methods if checkpoints exist."""
        cls.live_checkpoints_present = all(os.path.exists(p) for p in [
            "data/anka_taban_pure.pt",
            "data/anka_marangoz.pt",
            "data/anka_bahcivan.pt",
            "data/anka_router_v2.pt",
            "data/rebuild/vocab_anka_r1_33114.json"
        ])
        if cls.live_checkpoints_present:
            cls.live_gw = AgentGateway.create_default(
                model_path="data/anka_taban_pure.pt",
                vocab_path="data/rebuild/vocab_anka_r1_33114.json",
                router_state_path="data/anka_router_v2.pt",
                expert_model_paths={
                    "carpenter": "data/anka_marangoz.pt",
                    "gardener": "data/anka_bahcivan.pt"
                },
                device="cpu",
                storage_path=None,
                host=None
            )
        else:
            cls.live_gw = None

    def test_carpentry_prompts_route_strictly_to_carpenter(self):
        """Verifies that diverse carpentry prompts route strictly to the 'carpenter' expert."""
        if not self.live_checkpoints_present:
            self.skipTest("Live checkpoints not available for live routing verification.")

        carpentry_queries = [
            "zıvana geçme nasıl yapılır?",
            "ahşap rendeleme ve tesviye adımları nelerdir?",
            "meşe kereste fırınlama ve kurutma süreci nasıl işletilir?",
            "kırlangıç kuyruğu köşe birleştirme tekniği",
            "marangozlukta planya ve zımpara kullanımı",
            "gürgen masif panel tutkallama püf noktaları"
        ]

        for q in carpentry_queries:
            with self.subTest(query=q):
                res = self.live_gw.ask(q, mode="DIRECT")
                active = res.get("active_expert")
                top = res.get("router_experts", [])
                self.assertEqual(
                    active, "carpenter",
                    f"Carpentry query '{q}' did not route strictly to 'carpenter'! Got: active={active}, top={top}"
                )
                self.assertIn(
                    "carpenter", top,
                    f"'carpenter' not present in router_experts top-k for '{q}'. Got: {top}"
                )

    def test_gardening_prompts_route_strictly_to_gardener(self):
        """Verifies that diverse gardening prompts route strictly to the 'gardener' expert."""
        if not self.live_checkpoints_present:
            self.skipTest("Live checkpoints not available for live routing verification.")

        gardening_queries = [
            "domates budama ve koltuk alma işlemi nasıl yapılır?",
            "aşı kalemi nasıl yapılır?",
            "fidanlarda aşı kalemi nasıl hazırlanır?",
            "organik gübreleme ve toprak hazırlığı adımları",
            "patlıcan ve biber fidelerinde mildiyö tedavisi",
            "salatalık fidesi sulama aralıkları ve gübreleme"
        ]

        for q in gardening_queries:
            with self.subTest(query=q):
                res = self.live_gw.ask(q, mode="DIRECT")
                active = res.get("active_expert")
                top = res.get("router_experts", [])
                self.assertEqual(
                    active, "gardener",
                    f"Gardening query '{q}' did not route strictly to 'gardener'! Got: active={active}, top={top}"
                )
                self.assertIn(
                    "gardener", top,
                    f"'gardener' not present in router_experts top-k for '{q}'. Got: {top}"
                )

    def test_pedagogy_prompts_route_strictly_to_pedagogy(self):
        """Verifies that diverse pedagogy and educational prompts route strictly to the 'pedagogy' expert."""
        if not self.live_checkpoints_present:
            self.skipTest("Live checkpoints not available for live routing verification.")

        pedagogy_queries = [
            "bu konuyu ilkokul seviyesinde anlatır mısın?",
            "pedagojik rehberlik ilkeleri ve öğrenci takibi",
            "ortaokul düzeyinde fen bilgisi ders anlatımı",
            "nazım birimi ve kafiye şeması tahlili nasıl öğretilir?",
            "türk tarihi ve milli mücadele dönemi ders planı",
            "ilkokul öğrencileri için okuma anlama etkinlikleri"
        ]

        for q in pedagogy_queries:
            with self.subTest(query=q):
                res = self.live_gw.ask(q, mode="DIRECT")
                active = res.get("active_expert")
                top = res.get("router_experts", [])
                self.assertEqual(
                    active, "pedagogy",
                    f"Pedagogy query '{q}' did not route strictly to 'pedagogy'! Got: active={active}, top={top}"
                )
                self.assertIn(
                    "pedagogy", top,
                    f"'pedagogy' not present in router_experts top-k for '{q}'. Got: {top}"
                )

    def test_cross_domain_ambiguity_and_blend_behavior(self):
        """Verifies behavior on cross-domain ambiguous queries (blended carpentry, gardening, pedagogy)."""
        if not self.live_checkpoints_present:
            self.skipTest("Live checkpoints not available for live routing verification.")

        # 1. Carpentry + Gardening: wood planter for tomato seedling
        q_blend_1 = "Ahşap saksıda domates fidesi yetiştirirken gübreleme nasıl yapılır?"
        res1 = self.live_gw.ask(q_blend_1, mode="DIRECT")
        top1 = res1.get("router_experts", [])
        # Both gardener and carpenter should appear in top-2 experts
        self.assertTrue(
            "gardener" in top1 and "carpenter" in top1,
            f"Expected both 'gardener' and 'carpenter' in top experts for '{q_blend_1}', got {top1}"
        )

        # 2. Carpentry + Gardening: oak timber fence for garden
        q_blend_2 = "Bahçe çiti için meşe kerestesi rendeleme adımları"
        res2 = self.live_gw.ask(q_blend_2, mode="DIRECT")
        top2 = res2.get("router_experts", [])
        self.assertTrue(
            "carpenter" in top2 and "gardener" in top2,
            f"Expected both 'carpenter' and 'gardener' in top experts for '{q_blend_2}', got {top2}"
        )

        # 3. Pedagogy + Carpentry: teaching woodworking to students
        q_blend_3 = "Öğrencilere ahşap oyma ve marangozluk nasıl öğretilir?"
        res3 = self.live_gw.ask(q_blend_3, mode="DIRECT")
        top3 = res3.get("router_experts", [])
        self.assertIn(
            "carpenter", top3,
            f"Expected 'carpenter' in top experts for '{q_blend_3}', got {top3}"
        )

    def test_zero_embeddings_and_edge_cases_in_router(self):
        """Verifies numerical stability of TriModalRouter with zero vectors, NaN avoidance, and extreme inputs."""
        router = TriModalRouter(
            prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256,
            num_experts=5, top_k=2,
            expert_names=["grammar_core", "pedagogy", "carpenter", "gardener", "legal"]
        )

        # Test A: All zero inputs
        prompt_zero = torch.zeros(1, 768)
        merak_zero = torch.zeros(1, 768)
        rag_zero = torch.zeros(1, 768)
        w_zero, idx_zero, e_zero = router(prompt_zero, merak_zero, rag_zero)

        self.assertFalse(torch.isnan(w_zero).any().item(), "Zero input produced NaN in topk weights")
        self.assertFalse(torch.isnan(e_zero).any().item(), "Zero input produced NaN in e_route")
        self.assertAlmostEqual(w_zero.sum().item(), 1.0, places=4, msg="Top-K weights must sum to 1.0")
        self.assertEqual(idx_zero.shape, (1, 2), "Selected indices shape mismatch")

        # Test B: Extreme values (1e4)
        prompt_extreme = torch.full((1, 768), 10000.0)
        w_ext, idx_ext, e_ext = router(prompt_extreme)
        self.assertFalse(torch.isnan(w_ext).any().item(), "Extreme input produced NaN in topk weights")
        self.assertAlmostEqual(w_ext.sum().item(), 1.0, places=4, msg="Extreme input weights must sum to 1.0")

        # Test C: Empty, whitespace, punctuation, and random string inputs on AgentGateway
        if self.live_checkpoints_present:
            for strange_input in ["", "   ", "\n\t", "???!!!", "xyz123abc"]:
                with self.subTest(strange_input=strange_input):
                    res = self.live_gw.ask(strange_input, mode="DIRECT")
                    self.assertIn("active_expert", res)
                    self.assertIn("router_experts", res)
                    self.assertIsInstance(res["active_expert"], str)
                    self.assertGreater(len(res["router_experts"]), 0)

    def test_mock_vs_live_consistency_and_dispatch(self):
        """Verifies that mock and live modes maintain identical interface contracts and strict dispatch."""
        class TrackingModel(nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.call_count = 0
                self.embedding = nn.Embedding(30, 64)

            def forward(self, x, return_hidden_states=False):
                if not return_hidden_states:
                    self.call_count += 1
                b, s = x.shape
                logits = torch.ones(b, s, 30)
                hidden = torch.randn(b, s, 64)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3, "test": 4}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "test")

        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text): return [0, 4, 2]
            def decode(self, ids): return " ".join(self.vocab.decode(i) for i in ids)

        class MockDecompiler:
            def decompile_sentence(self, tags, capitalize=True): return tags

        mock_base = TrackingModel("mock_base")
        mock_carp = TrackingModel("mock_carp")
        mock_gard = TrackingModel("mock_gard")

        mock_gw = AgentGateway(
            model=mock_base,
            tokenizer=MockTokenizer(),
            decompiler=MockDecompiler(),
            memory=VectorMemory(collection_name="mock_b1", vector_size=64, storage_path=None, host=None),
            general_memory=VectorMemory(collection_name="mock_b2", vector_size=64, storage_path=None, host=None),
            expert_models={
                "carpenter": mock_carp,
                "gardener": mock_gard
            },
            device="cpu"
        )

        # Verify automatic default mapping for pedagogy and grammar_core
        self.assertIn("pedagogy", mock_gw.expert_models)
        self.assertIn("grammar_core", mock_gw.expert_models)
        self.assertIs(mock_gw.expert_models["pedagogy"], mock_base)
        self.assertIs(mock_gw.expert_models["grammar_core"], mock_base)

        # Configure deterministic routing to carpenter
        router = TriModalRouter(
            prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32,
            num_experts=5, top_k=2,
            expert_names=["grammar_core", "pedagogy", "carpenter", "gardener", "legal"]
        )
        with torch.no_grad():
            router.w_p.weight.zero_()
            router.w_p.bias.zero_()
            router.w_m.weight.zero_()
            router.w_m.bias.zero_()
            router.w_r.weight.zero_()
            router.w_r.bias.zero_()
            router.w_g.weight.zero_()
            router.w_g.weight[2, 0] = 10.0  # Force carpenter logit high

        mock_gw.epistemic_agent.router = router
        mock_gw.epistemic_agent.curiosity_engine = CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5)

        res_mock = mock_gw.ask("Zıvana testi", mode="DIRECT")
        self.assertEqual(res_mock["active_expert"], "carpenter")
        self.assertEqual(mock_carp.call_count, 1)
        self.assertEqual(mock_gard.call_count, 0)
        self.assertEqual(mock_base.call_count, 0)

        # Check response schema fields match expected contract
        expected_keys = {
            "query", "instruction", "mode", "response_text", "morphemes",
            "entropy_pre", "entropy_post", "needs_retrieval", "retrieval_triggered",
            "rag_document", "source_collection", "rag_score", "conditioned",
            "is_high_similarity", "router_experts", "active_expert",
            "epistemic_failure", "future_train_recorded", "future_train_path", "timestamp"
        }
        self.assertTrue(expected_keys.issubset(set(res_mock.keys())))


if __name__ == "__main__":
    unittest.main()
