#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_moe_gateway_routing.py
==================================
Milestone M4: AgentGateway Canlı ve Mock MoE Çıkarımı, Dinamik Uzman Yönlendirmesi,
Çift Çıktılı Ayrıştırıcı ve CoT Kasa Güvenliği Doğrulama Testleri.
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock
import torch

from src.gateway.agent_gateway import AgentGateway
from src.gateway.pedagogical_supervisor import (
    extract_cot_and_card,
    record_to_cot_vault,
    _CANONICAL_PROD_VAULT
)
from src.rag.vector_memory import VectorMemory
from src.llm.router import TriModalRouter
from src.rag.merak import CuriosityEngine


class TestMoEGatewayRouting(unittest.TestCase):
    """Verifies mock and live MoE dynamic routing on AgentGateway and dual parser contracts."""

    def test_mock_moe_dynamic_routing_three_domains(self):
        """Verifies that AgentGateway dispatches queries to the exact expert models in mock mode."""
        class MockVocab:
            stoi = {
                "<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3,
                "ahşap": 4, "kırlangıç": 5, "domates": 6, "mildiyö": 7,
                "nazım": 8, "şiir": 9, "bilgi": 10
            }
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "bilgi")

        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text):
                text_l = text.lower()
                ids = [0]
                if "kırlangıç" in text_l or "ahşap" in text_l:
                    ids.extend([4, 5])
                elif "domates" in text_l or "mildiyö" in text_l:
                    ids.extend([6, 7])
                elif "nazım" in text_l or "şiir" in text_l:
                    ids.extend([8, 9])
                else:
                    ids.append(10)
                ids.append(2)
                return ids
            def decode(self, ids): return " ".join(self.vocab.decode(i) for i in ids)

        class MockDecompiler:
            def decompile_sentence(self, tags_str, capitalize=True):
                return f"Decompiled: {tags_str}"

        class TrackingMockModel(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.generation_called = False
                self.embedding = torch.nn.Embedding(20, 64)

            def forward(self, x, return_hidden_states=False):
                if not return_hidden_states:
                    self.generation_called = True
                b, s = x.shape
                logits = torch.ones(b, s, 20)
                hidden = torch.randn(b, s, 64)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        base_model = TrackingMockModel("base_pedagogy")
        carpenter_model = TrackingMockModel("expert_carpenter")
        gardener_model = TrackingMockModel("expert_gardener")

        # 5-expert TriModalRouter: indices [0: grammar_core, 1: pedagogy, 2: carpenter, 3: gardener, 4: legal]
        # Weights configured so domain priors or base weights route to the desired expert
        router = TriModalRouter(
            prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32,
            num_experts=5, top_k=2,
            expert_names=["grammar_core", "pedagogy", "carpenter", "gardener", "legal"]
        )
        with torch.no_grad():
            router.w_p.weight.zero_()
            router.w_p.bias.zero_()
            router.w_p.bias[0] = 5.0
            router.w_m.weight.zero_()
            router.w_m.bias.zero_()
            router.w_r.weight.zero_()
            router.w_r.bias.zero_()
            # Pedagogy (class 1) receives slight default bias, easily overridden by carpenter/gardener domain priors
            router.w_g.weight.zero_()
            router.w_g.weight[1, 0] = 0.5

        # Memory mock (in-memory)
        mem = VectorMemory(collection_name="mock_anka_bellek", vector_size=64, storage_path=None, host=None)
        gen_mem = VectorMemory(collection_name="mock_sim_bellek", vector_size=64, storage_path=None, host=None)

        gw = AgentGateway(
            model=base_model,
            tokenizer=MockTokenizer(),
            decompiler=MockDecompiler(),
            memory=mem,
            general_memory=gen_mem,
            expert_models={
                "carpenter": carpenter_model,
                "gardener": gardener_model,
                "pedagogy": base_model
            },
            device="cpu"
        )
        gw.epistemic_agent.router = router
        gw.epistemic_agent.curiosity_engine = CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5)

        # 1. Carpentry Query: domain prior triggers class 2 ('carpenter')
        res_carp = gw.ask("Kırlangıç kuyruğu masif ahşapta nasıl açılır?", mode="DIRECT")
        self.assertEqual(res_carp["active_expert"], "carpenter")
        self.assertIn("carpenter", res_carp["router_experts"])
        self.assertTrue(carpenter_model.generation_called)
        self.assertFalse(gardener_model.generation_called)
        self.assertFalse(base_model.generation_called)

        # Reset flags
        carpenter_model.generation_called = False
        gardener_model.generation_called = False
        base_model.generation_called = False

        # 2. Gardening Query: domain prior triggers class 3 ('gardener')
        res_gard = gw.ask("Domates fidesinde mildiyö hastalığı nasıl tedavi edilir?", mode="DIRECT")
        self.assertEqual(res_gard["active_expert"], "gardener")
        self.assertIn("gardener", res_gard["router_experts"])
        self.assertTrue(gardener_model.generation_called)
        self.assertFalse(carpenter_model.generation_called)
        self.assertFalse(base_model.generation_called)

        # Reset flags
        carpenter_model.generation_called = False
        gardener_model.generation_called = False
        base_model.generation_called = False

        # 3. Pedagogy Query: default routing routes to class 1 ('pedagogy')
        res_ped = gw.ask("Nazım birimi ve şiir tahlili nedir?", mode="DIRECT")
        self.assertEqual(res_ped["active_expert"], "pedagogy")
        self.assertIn("pedagogy", res_ped["router_experts"])
        self.assertTrue(base_model.generation_called)
        self.assertFalse(carpenter_model.generation_called)
        self.assertFalse(gardener_model.generation_called)

    def test_live_moe_routing_with_real_checkpoints(self):
        """Verifies live inference with anka_taban_pure.pt, anka_marangoz.pt, anka_bahcivan.pt, and anka_router_v2.pt."""
        checkpoints = [
            "data/anka_taban_pure.pt",
            "data/anka_marangoz.pt",
            "data/anka_bahcivan.pt",
            "data/anka_router_v2.pt",
            "data/rebuild/vocab_anka_r1_33114.json"
        ]
        for cp in checkpoints:
            if not os.path.exists(cp):
                self.skipTest(f"Live checkpoint not found: {cp}")

        gw = AgentGateway.create_default(
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

        # 1. Carpentry -> carpenter jacket
        res_carp = gw.ask("Kırlangıç kuyruğu masif ahşapta nasıl açılır?", mode="DIRECT")
        self.assertEqual(res_carp["active_expert"], "carpenter")
        self.assertIn("carpenter", res_carp["router_experts"])
        self.assertIsNotNone(res_carp["response_text"])

        # 2. Gardening -> gardener jacket
        res_gard = gw.ask("Domates fidesinde mildiyö hastalığı nasıl tedavi edilir?", mode="DIRECT")
        self.assertEqual(res_gard["active_expert"], "gardener")
        self.assertIn("gardener", res_gard["router_experts"])
        self.assertIsNotNone(res_gard["response_text"])

        # 3. Pedagogy -> pure base model
        res_ped = gw.ask("Soru: Nazım birimi nedir ve temel nazım birimleri nelerdir?", mode="DIRECT")
        self.assertEqual(res_ped["active_expert"], "pedagogy")
        self.assertIn("pedagogy", res_ped["router_experts"])
        self.assertIsNotNone(res_ped["response_text"])

    def test_dual_parser_all_five_tag_patterns(self):
        """Verifies extract_cot_and_card across dual-tagged, card-only, thought-only, short-card, and procedural cards."""
        # Case 1: Dual-tagged
        dual_text = (
            "<DUSUNCE>\n"
            "Kullanıcı ahşap özelliklerini soruyor.\n"
            "Gürgen lif yapısı son derece sert ve dayanıklıdır.\n"
            "</DUSUNCE>\n"
            "<BILGI_KARTI>\n"
            "Gürgen ağacı yüksek mukavemeti ve sert dokusu nedeniyle mobilyacılıkta tercih edilir.\n"
            "</BILGI_KARTI>"
        )
        th1, cd1 = extract_cot_and_card(dual_text)
        self.assertIsNotNone(th1)
        self.assertIsNotNone(cd1)
        self.assertIn("Kullanıcı ahşap özelliklerini", th1)
        self.assertIn("Gürgen ağacı yüksek mukavemeti", cd1)
        self.assertNotIn("<DUSUNCE>", th1)
        self.assertNotIn("<BILGI_KARTI>", cd1)

        # Case 2: Card-only (CRITICAL: must not synthesize phantom '<BILGI_KARTI>\n</BILGI_KARTI>' thought trace)
        card_only_text = (
            "<BILGI_KARTI>\n"
            "Gürgen ağacı yüksek mukavemeti ve sert dokusu nedeniyle mobilyacılıkta tercih edilir.\n"
            "</BILGI_KARTI>"
        )
        th2, cd2 = extract_cot_and_card(card_only_text)
        self.assertIsNone(th2, f"Card-only input must have thought_trace=None, got {repr(th2)}")
        self.assertIsNotNone(cd2)
        self.assertIn("Gürgen ağacı yüksek mukavemeti", cd2)
        self.assertNotIn("<BILGI_KARTI>", cd2)

        # Case 3: Thought-only
        thought_only_text = (
            "<DUSUNCE>\n"
            "Kullanıcı burada yalnızca içsel bir akıl yürütme istemi yönlendirmiştir.\n"
            "Deklaratif bir bilgi kartı üretilmemelidir.\n"
            "</DUSUNCE>"
        )
        th3, cd3 = extract_cot_and_card(thought_only_text)
        self.assertIsNotNone(th3)
        self.assertIn("akıl yürütme istemi", th3)
        self.assertIsNone(cd3, f"Thought-only input must have clean_card=None, got {repr(cd3)}")

        # Case 4: Short card (< 15 chars)
        short_card_text = (
            "<DUSUNCE>Kısa düşünce adımı burada yer alıyor.</DUSUNCE>\n"
            "<BILGI_KARTI>Kısa metin.</BILGI_KARTI>"
        )
        th4, cd4 = extract_cot_and_card(short_card_text)
        self.assertIsNotNone(th4)
        self.assertIsNone(cd4, f"Short card (< 15 chars) must not leak raw XML tags; got {repr(cd4)}")

        # Case 5: Procedural card
        proc_text = (
            "<DUSUNCE>\n"
            "Marangozlukta ahşap hazırlama adımlarını sıralıyoruz.\n"
            "</DUSUNCE>\n"
            "<BILGI_KARTI>\n"
            "Adım 1: Gürgen ağacı gönyesinde hassas biçimde kesilir.\n"
            "Adım 2: Fırınlanmış kereste rendelenerek pürüzsüz yüzey elde edilir.\n"
            "</BILGI_KARTI>"
        )
        th5, cd5 = extract_cot_and_card(proc_text)
        self.assertIsNotNone(th5)
        self.assertIsNotNone(cd5)
        self.assertIn("Adım 1: Gürgen ağacı", cd5)
        self.assertIn("Adım 2: Fırınlanmış", cd5)
        self.assertNotIn("<BILGI_KARTI>", cd5)

    def test_allow_prod_vault_write_fail_closed_falsy_values(self):
        """Verifies that setting ALLOW_PROD_VAULT_WRITE to falsy strings strictly raises RuntimeError."""
        old_val = os.environ.get("ALLOW_PROD_VAULT_WRITE")
        try:
            for falsy in ["0", "false", "no", "False", "NO", "   0   ", ""]:
                os.environ["ALLOW_PROD_VAULT_WRITE"] = falsy
                with self.assertRaises(RuntimeError, msg=f"Should raise RuntimeError for ALLOW_PROD_VAULT_WRITE={repr(falsy)}"):
                    record_to_cot_vault(
                        query="Güvenlik sorgusu",
                        instruction="Yönerge",
                        thought_trace="Düşünce adımı güvenlik doğrulaması",
                        final_answer="Nihai yanıt güvenlik doğrulaması",
                        vault_path=_CANONICAL_PROD_VAULT
                    )
        finally:
            if old_val is not None:
                os.environ["ALLOW_PROD_VAULT_WRITE"] = old_val
            else:
                os.environ.pop("ALLOW_PROD_VAULT_WRITE", None)


if __name__ == "__main__":
    unittest.main()
