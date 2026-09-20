#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock
import urllib.request
import threading
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.agent_gateway import AgentGateway
from src.gateway.retrain_pipeline import RetrainPipeline
from src.gateway.pedagogical_supervisor import PedagogicalSupervisor
from src.rag.vector_memory import VectorMemory


class TestAgentGateway(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.future_file = os.path.join(self.tmp_dir.name, "future_train_vector.jsonl")
        self.archive_file = os.path.join(self.tmp_dir.name, "future_train_archive.jsonl")
        self.bin_file = os.path.join(self.tmp_dir.name, "test_finetune.bin")

        # Mock Vocab & Tokenizer
        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3, "ahşap": 4, "bilgi": 5, "unk": 6}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "unk")

        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text): return [0, 4, 5, 2]
            def decode(self, ids): return "ahşap bilgi"

        class MockDecompiler:
            def decompile_sentence(self, tags_str): return "ahşap bilgisi"

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 768)
            def forward(self, x, return_hidden_states=False):
                b, s = x.shape
                logits = torch.ones(b, s, 10)
                hidden = torch.randn(b, s, 768)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        self.memory = VectorMemory(
            collection_name="test_gw_kristal",
            vector_size=768,
            storage_path=self.tmp_dir.name,
            host=None
        )

        self.gateway = AgentGateway(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            decompiler=MockDecompiler(),
            memory=self.memory,
            future_train_path=self.future_file,
            device="cpu"
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_gateway_ask_returns_structured_telemetry(self):
        res = self.gateway.ask("Kırlangıç kuyruğu nedir?", mode="RAG")
        self.assertIn("response_text", res)
        self.assertIn("entropy_pre", res)
        self.assertIn("entropy_post", res)
        self.assertIn("rag_score", res)
        self.assertIn("epistemic_failure", res)
        self.assertIn("future_train_recorded", res)

    def test_gateway_inject_knowledge_and_check(self):
        inject_res = self.gateway.inject_knowledge(
            text="Kırlangıç kuyruğu mukavemetli bir köşe birleştirmedir.",
            target_collection="kristal_bellek"
        )
        self.assertEqual(inject_res["status"], "success")
        self.assertGreaterEqual(inject_res["total_documents"], 1)

        # Check retrieval
        check_res = self.gateway.check_memory("Kırlangıç kuyruğu", target_collection="kristal_bellek")
        self.assertGreater(len(check_res), 0)
        self.assertIn("Kırlangıç kuyruğu", check_res[0]["text"])

    def test_pedagogical_supervisor_evaluation_and_step(self):
        isolated_vault = os.path.join(self.tmp_dir.name, "isolated_supervisor_vault.jsonl")
        supervisor = PedagogicalSupervisor(
            gateway=self.gateway,
            vault_path=isolated_vault
        )
        
        # Test heuristic quality evaluation
        sat, msg = supervisor.evaluate_response_quality("ahşap mobilya ve köşe birleştirme", ["köşe", "ahşap"])
        self.assertTrue(sat)
        
        unsat, u_msg = supervisor.evaluate_response_quality("alakasız bir cevap", ["köşe", "zıvana"])
        self.assertFalse(unsat)

        # Mock teacher call to isolate from external network/API and test CoT recording safely
        teacher_output = (
            "<DUSUNCE>\n"
            "Kırlangıç kuyruğu çekmece kasalarında ve masif sandık köşelerinde yüksek mukavemet sağlar.\n"
            "</DUSUNCE>\n"
            "<BILGI_KARTI>\n"
            "Kırlangıç kuyruğu geçme, çekmecelerde ve sandıklarda yüksek çekme direnci için kullanılır.\n"
            "</BILGI_KARTI>"
        )
        supervisor.call_gemini = MagicMock(return_value=teacher_output)
        supervisor.gemini_api_key = "dummy_key"

        # Test single supervision step
        probe = {
            "query": "Kırlangıç kuyruğu nerelerde kullanılır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "Kırlangıç kuyruğu çekmecelerde ve sandıklarda kullanılır.",
            "expected_keywords": ["çekmece", "sandık"]
        }
        step_result = supervisor.execute_supervision_step(probe, auto_inject=True)
        self.assertIn("first_attempt", step_result)
        self.assertIn("is_satisfactory", step_result)
        self.assertIn("knowledge_injected", step_result)
        self.assertTrue(os.path.exists(isolated_vault))
        with open(isolated_vault, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        vault_data = json.loads(lines[0])
        self.assertEqual(vault_data["query"], "Kırlangıç kuyruğu nerelerde kullanılır?")
        self.assertIn("çekmece kasalarında", vault_data["thought_trace"])

    def test_retrain_pipeline_compile(self):
        # Create 2 mock records in future_train_vector.jsonl
        rec1 = {"instruction": "Belgeye göre cevapla.", "input": "belge: test sorgu: test", "output": "ahşap"}
        rec2 = {"instruction": "Belgeye göre cevapla.", "input": "belge: zıvana sorgu: nedir", "output": "zıvana"}
        with open(self.future_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec1) + "\n")
            f.write(json.dumps(rec2) + "\n")

        pipeline = RetrainPipeline(
            future_train_path=self.future_file,
            archive_path=self.archive_file,
            vocab_path="data/rebuild/vocab_anka_r1_33114.json",
            output_bin_path=self.bin_file
        )
        self.assertEqual(pipeline.get_pending_count(), 2)

        bin_path, count = pipeline.compile_backlog_to_bin(block_size=32, oversample_factor=2)
        self.assertEqual(count, 2)
        self.assertTrue(os.path.exists(bin_path))
        self.assertTrue(os.path.exists(bin_path + ".meta.json"))

        # Test archive helper
        pipeline._archive_processed_records()
        self.assertEqual(pipeline.get_pending_count(), 0)
        self.assertTrue(os.path.exists(self.archive_file))

    def test_gateway_http_server_endpoints(self):
        port = 18099
        server = self.gateway.create_http_server(host="127.0.0.1", port=port)
        self.assertIsNotNone(server)
        self.assertEqual(server.server_port, port)
        server.server_close()

        # Test status endpoint payload
        status_data = self.gateway.get_status()
        self.assertEqual(status_data["status"], "online")
        self.assertIn("kristal_bellek_docs", status_data)

        # Test inject payload
        inject_data = self.gateway.inject_knowledge(text="Test sunucu bilgisi.", target_collection="kristal_bellek")
        self.assertEqual(inject_data["status"], "success")

        # Test query payload
        query_data = self.gateway.ask(query="Test sunucu sorusu?")
        self.assertIn("response_text", query_data)

    def test_canary_create_default_raises_on_missing_model(self):
        # Bu test, model checkpoint'i bulunamadığında sessizce rastgele modelle açılmak yerine
        # gürültülü bir biçimde FileNotFoundError fırlatıldığını doğrular.
        # T-0087: `vocab_path` artık ZORUNLU; kapi model yoluna ULASMADAN durmasin diye
        # gecerli sozluk ACIKCA verilir.
        with self.assertRaises(FileNotFoundError):
            AgentGateway.create_default(
                model_path="/tmp/nonexistent_model_file_canary.pt",
                vocab_path="data/rebuild/vocab_anka_r1_33114.json"
            )

    def test_canary_create_default_varsayilanlar_kaldirildi(self):
        """T-0087 (FAIL-CLOSED): `create_default` varsayilanlari KALDIRILDI.

        Argumansiz cagri SESSIZCE bayat sozluge (data/vocab.json, 31.357) veya olu
        checkpoint yoluna DUSMEZ; RuntimeError ile durur. Sozluk verilip model
        verilmezse de durur (iki kapi BAGIMSIZ olarak sinanir).
        """
        with self.assertRaises(RuntimeError):
            AgentGateway.create_default()
        with self.assertRaises(RuntimeError):
            AgentGateway.create_default(model_path="data/anka_a1r.pt")

    def test_canary_retrain_pipeline_varsayilanlar_kaldirildi(self):
        """T-0087 (FAIL-CLOSED): `RetrainPipeline` sozluk/model/save varsayilanlari KALDIRILDI.

        Sozluk yuklemesi ve egitim ZORUNLU arguman verilmeden BASLAMAZ: eksik argumanda
        `compile_backlog_to_bin` hicbir sey YAZMAZ (bkz. K3).
        """
        pipeline = RetrainPipeline(
            future_train_path=self.future_file,
            archive_path=self.archive_file,
            output_bin_path=self.bin_file
        )
        with self.assertRaises(RuntimeError):
            pipeline.compile_backlog_to_bin(block_size=32)
        with self.assertRaises(RuntimeError):
            pipeline.run_training(steps=1)
        self.assertFalse(os.path.exists(self.bin_file))

    def test_canary_load_eval_model_raises_on_missing_checkpoint(self):
        # Bu test, değerlendiricilerde model checkpoint'i bulunamadığında
        # sessizce rastgele model döndürmek yerine FileNotFoundError fırlatıldığını doğrular.
        from src.llm.tokenizer import Vocabulary
        from scripts.evaluate_mcq_conditioning import load_eval_model as load_mcq
        from scripts.run_experiment_a import load_eval_model as load_exp_a
        from scripts.evaluate_sft_benchmarks import load_eval_model as load_sft
        vocab = Vocabulary()
        vocab.load("data/vocab.json")
        for name, fn in [("mcq", load_mcq), ("exp_a", load_exp_a), ("sft", load_sft)]:
            with self.assertRaises(FileNotFoundError, msg=f"{name} did not raise FileNotFoundError"):
                fn("/tmp/nonexistent_eval_checkpoint.pt", vocab)


if __name__ == "__main__":
    unittest.main()
