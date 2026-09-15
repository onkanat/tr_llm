#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for CoT Vault, Dual-Output Parser, and Reasoning Isolation.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock

from src.gateway.pedagogical_supervisor import (
    extract_cot_and_card,
    record_to_cot_vault,
    PedagogicalSupervisor,
    _CANONICAL_PROD_VAULT
)
from scripts.prepare_reasoning_dataset import format_reasoning_sample


class TestCoTVaultAndDualParser(unittest.TestCase):

    def test_extract_cot_and_card_with_explicit_tags(self):
        raw_text = (
            "<DUSUNCE>\n"
            "Kullanıcı gürgen ağacının özelliklerini ve kullanım alanlarını soruyor.\n"
            "Gürgen lif yapısı sert, esnekliği az olan dayanıklı bir ağaçtır.\n"
            "Alet sapları, masif mobilya ve zemin kaplamalarında kullanılır.\n"
            "</DUSUNCE>\n"
            "<BILGI_KARTI>\n"
            "Gürgen ağacı yüksek mukavemeti ve sert dokusu nedeniyle çekiç, keser gibi el aletlerinin saplarında "
            "ve ağır hizmet tipi masif doğramalarda tercih edilir.\n"
            "</BILGI_KARTI>"
        )

        thought, card = extract_cot_and_card(raw_text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("Kullanıcı gürgen ağacının", thought)
        self.assertIn("Gürgen ağacı yüksek mukavemeti", card)
        self.assertNotIn("<DUSUNCE>", thought)
        self.assertNotIn("<BILGI_KARTI>", card)

    def test_extract_cot_and_card_with_think_tags(self):
        raw_text = (
            "<think>\n"
            "The user wants to know about acik istiare in Turkish literature.\n"
            "Acik istiare is a metaphor where only the vehicle (kendisine benzetilen) is explicitly stated.\n"
            "</think>\n"
            "Açık istiare, benzetme ögelerinden yalnızca kendisine benzetilenin kullanılmasıyla yapılan edebi sanattır."
        )

        thought, card = extract_cot_and_card(raw_text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("The user wants to know", thought)
        self.assertIn("Açık istiare", card)
        self.assertNotIn("The user wants to know", card)

    def test_extract_cot_and_card_with_english_preamble(self):
        raw_text = (
            "We need to create a 2-3 sentence concise, pedagogically verified info card.\n"
            "The user says: Gürgen ağacından ne yapalım?\n"
            "Gürgen ağacı; sertliği ve darbelere karşı direnci sayesinde kesme tahtası, fırın küreği ve alet sapı yapımında kullanılır."
        )

        thought, card = extract_cot_and_card(raw_text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("We need to create", thought)
        self.assertIn("Gürgen ağacı", card)
        self.assertNotIn("We need to create", card)

    def test_record_to_cot_vault(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = os.path.join(tmpdir, "test_vault.jsonl")

            query = "Gürgen ağacından ne yapılır?"
            instruction = "ahşap uzmanı olarak cevapla."
            thought = "Gürgen sert yapılıdır ve sürtünmeye dayanıklıdır."
            final_answer = "Gürgen ağacı alet sapları ve tezgahlarda kullanılır."

            success = record_to_cot_vault(
                query=query,
                instruction=instruction,
                thought_trace=thought,
                final_answer=final_answer,
                source="test_teacher",
                vault_path=vault_path
            )

            self.assertTrue(success)
            self.assertTrue(os.path.exists(vault_path))

            with open(vault_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                data = json.loads(lines[0])
                self.assertEqual(data["query"], query)
                self.assertEqual(data["thought_trace"], thought)
                self.assertEqual(data["final_answer"], final_answer)
                self.assertEqual(data["source"], "test_teacher")

    def test_format_reasoning_sample(self):
        item = {
            "instruction": "Akıl yürüterek cevapla.",
            "query": "Açık istiare nedir?",
            "thought_trace": "İstiare teşbihin tek ögeyle yapılmasıdır.",
            "final_answer": "Açık istiare sadece kendisine benzetilenle yapılır."
        }

        formatted = format_reasoning_sample(item)
        self.assertEqual(formatted["instruction"], "Akıl yürüterek cevapla.")
        self.assertEqual(formatted["input"], "Açık istiare nedir?")
        self.assertEqual(
            formatted["output"],
            "<DUSUNCE> İstiare teşbihin tek ögeyle yapılmasıdır. </DUSUNCE> Açık istiare sadece kendisine benzetilenle yapılır."
        )

    def test_pedagogical_supervisor_isolation_flow(self):
        # Mock gateway
        mock_gateway = MagicMock()
        mock_gateway.ask.return_value = {
            "response_text": "yetersiz yanıt",
            "epistemic_failure": True,
            "rag_score": 0.40
        }
        mock_gateway.check_memory.return_value = [{"score": 0.95}]
        mock_gateway.inject_knowledge.return_value = {"status": "success"}
        mock_gateway.inject_reasoning_trace.return_value = {"status": "success"}

        with tempfile.TemporaryDirectory() as tmpdir:
            isolated_vault = os.path.join(tmpdir, "isolated_cot_vault.jsonl")
            supervisor = PedagogicalSupervisor(
                gateway=mock_gateway,
                enrich_rag=True,
                vault_path=isolated_vault
            )

            # Mock teacher call returning dual CoT and Card
            teacher_output = (
                "<DUSUNCE>\n"
                "Açık istiare teşbih sanatının bir türüdür. Sadece benzeyen değil kendisine benzetilen söylenir.\n"
                "</DUSUNCE>\n"
                "<BILGI_KARTI>\n"
                "Açık istiare, teşbihte temel ögelerden yalnızca kendisine benzetilenin anıldığı istiare türüdür.\n"
                "</BILGI_KARTI>"
            )
            supervisor.call_gemini = MagicMock(return_value=teacher_output)
            supervisor.gemini_api_key = "dummy_key"

            probe = {
                "query": "Açık istiare nedir?",
                "instruction": "Lise edebiyat dersi kapsamında açıkla.",
                "target_collection": "simulasyon_bellek",
                "knowledge_to_inject": "Açık istiarede yalnızca kendisine benzetilen kullanılır.",
                "expected_keywords": ["açık", "istiare", "benzetilen"]
            }

            res = supervisor.execute_supervision_step(probe, auto_inject=True)

            # Assert declarative injection got ONLY the clean card
            mock_gateway.inject_knowledge.assert_called_once()
            args, kwargs = mock_gateway.inject_knowledge.call_args
            injected_text = kwargs["text"]
            self.assertIn("Açık istiare, teşbihte temel ögelerden", injected_text)
            self.assertNotIn("<DUSUNCE>", injected_text)
            self.assertNotIn("Açık istiare teşbih sanatının bir türüdür.", injected_text)

            # Assert reasoning trace was injected to muhakeme_bellek
            mock_gateway.inject_reasoning_trace.assert_called_once()
            r_args, r_kwargs = mock_gateway.inject_reasoning_trace.call_args
            self.assertEqual(r_kwargs["query"], "Açık istiare nedir?")
            self.assertIn("kendisine benzetilen söylenir", r_kwargs["thought_text"])

            # Assert isolated vault was written and contains the CoT trace (isolation confirmed)
            self.assertTrue(os.path.exists(isolated_vault))
            with open(isolated_vault, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            vault_data = json.loads(lines[0])
            self.assertEqual(vault_data["query"], "Açık istiare nedir?")
            self.assertIn("Açık istiare teşbih sanatının bir türüdür.", vault_data["thought_trace"])

    def test_record_to_cot_vault_blocks_prod_write_in_tests_cwd_repo_root(self):
        # (G1-i) CWD=repo koku + varsayılan/göreli yol -> RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            record_to_cot_vault(
                query="Tehlikeli sorgu",
                instruction="Yönerge",
                thought_trace="Düşünce adımı",
                final_answer="Nihai yanıt",
                vault_path="data/pedagogy/cot_vault.jsonl"
            )
        self.assertIn("Güvenlik İhlali", str(ctx.exception))

    def test_record_to_cot_vault_blocks_prod_write_in_tests_cwd_other(self):
        # (G1-ii) CWD=başka bir dizin + mutlak kanonik üretim yolu -> YİNE RuntimeError (fail-open koruması)
        old_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            with self.assertRaises(RuntimeError) as ctx:
                record_to_cot_vault(
                    query="Tehlikeli sorgu CWD disi",
                    instruction="Yönerge",
                    thought_trace="Düşünce adımı",
                    final_answer="Nihai yanıt",
                    vault_path=_CANONICAL_PROD_VAULT
                )
            self.assertIn("Güvenlik İhlali", str(ctx.exception))
        finally:
            os.chdir(old_cwd)

    def test_record_to_cot_vault_flat_filename_in_tmpdir(self):
        # (G2) vault_path dizin bileşeni içermiyorsa (düz dosya adı) sessizce False dönmez, başarıyla yazar
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                res = record_to_cot_vault(
                    query="Düz dosya sorgusu",
                    instruction="Yönerge",
                    thought_trace="Düşünce adımı",
                    final_answer="Nihai yanıt",
                    vault_path="flat_vault.jsonl"
                )
                self.assertTrue(res)
                self.assertTrue(os.path.exists("flat_vault.jsonl"))
            finally:
                os.chdir(old_cwd)

    def test_record_to_cot_vault_f4_default_vault_path_none_cwd_tmp(self):
        # (F4) CWD=/tmp + vault_path=None -> hedef kanonik MUTLAK yol olur ve test ortamında RuntimeError fırlatır
        old_cwd = os.getcwd()
        old_env_cot = os.environ.pop("COT_VAULT_PATH", None)
        try:
            os.chdir("/tmp")
            with self.assertRaises(RuntimeError) as ctx:
                record_to_cot_vault(
                    query="Tehlikeli sorgu CWD tmp vault None",
                    instruction="Yönerge",
                    thought_trace="Düşünce adımı",
                    final_answer="Nihai yanıt",
                    vault_path=None
                )
            self.assertIn("Güvenlik İhlali", str(ctx.exception))
            self.assertIn(_CANONICAL_PROD_VAULT, str(ctx.exception))

            # COT_VAULT_PATH verilmişse ona dokunulmadığı da gösterilir
            with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
                os.environ["COT_VAULT_PATH"] = tf.name
                res = record_to_cot_vault(
                    query="Güvenli özel env sorgusu",
                    instruction="Yönerge",
                    thought_trace="Düşünce adımı",
                    final_answer="Nihai yanıt",
                    vault_path=None
                )
                self.assertTrue(res)
                self.assertTrue(os.path.exists(tf.name))
        finally:
            if old_env_cot is not None:
                os.environ["COT_VAULT_PATH"] = old_env_cot
            else:
                os.environ.pop("COT_VAULT_PATH", None)
            os.chdir(old_cwd)

    def test_record_to_cot_vault_f5_unwritable_path_does_not_raise(self):
        # (F5) Yazılamayan bir vault_path çağrıyı DÜŞÜRMEZ, loglanır ve False döner
        unwritable_path = "/nonexistent_protected_root_dir_12345/vault.jsonl"
        res = record_to_cot_vault(
            query="Yazılamayan yol sorgusu",
            instruction="Yönerge",
            thought_trace="Düşünce adımı",
            final_answer="Nihai yanıt",
            vault_path=unwritable_path
        )
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
