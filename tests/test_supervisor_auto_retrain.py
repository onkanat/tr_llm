#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P2.3 ONARIM-1: Süpervizör --auto-retrain varsayılan boru hattı sözleşmesi.
=========================================================================
T-0087 (FAIL-CLOSED) `RetrainPipeline`'dan model_path/save_path/vocab_path
varsayılanlarını kaldırdı; `PedagogicalSupervisor`'ün argümansız
`RetrainPipeline()` varsayılanı bu yüzden eşik aşıldığında `run_training`'de
RuntimeError ile ölüyordu (P2.3'te ölçüldü). Sözleşme: varsayılan boru hattı
AÇIK kanonik yollarla KONFİGÜRE kurulur, yollar env/CLI ile geçilebilir ve
çıktı asla taban checkpoint'in üzerine yazmaz.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.pedagogical_supervisor import (
    PedagogicalSupervisor,
    varsayilan_otonom_retrain_pipeline,
    OTONOM_RETRAIN_MODEL_PATH,
    OTONOM_RETRAIN_SAVE_PATH,
    OTONOM_RETRAIN_VOCAB_PATH,
)

KANONIK_VOCAB = "data/rebuild/vocab_anka_r1_33114.json"


class TestSupervisorAutoRetrain(unittest.TestCase):

    ENV_KEYS = [
        "RETRAIN_MODEL_PATH", "RETRAIN_SAVE_PATH", "RETRAIN_VOCAB_PATH",
        "RETRAIN_FUTURE_PATH", "RETRAIN_ARCHIVE_PATH", "RETRAIN_BIN_PATH",
    ]

    def setUp(self):
        self._old_env = {k: os.environ.get(k) for k in self.ENV_KEYS}
        for k in self.ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _mock_gateway():
        gw = MagicMock()
        gw.ask.return_value = {
            "response_text": "test kavram cevabı",
            "epistemic_failure": False,
            "rag_score": 0.9,
        }
        return gw

    def test_varsayilan_pipeline_kanonik_acik_yollarla_kurulur(self):
        """Argümansız süpervizör, T-0087 uyumlu KONFİGÜRE pipeline alır (None yollar DEĞİL)."""
        sup = PedagogicalSupervisor(gateway=self._mock_gateway())
        p = sup.retrain_pipeline
        self.assertEqual(p.model_path, OTONOM_RETRAIN_MODEL_PATH)
        self.assertEqual(p.model_path, "data/anka_taban_pure.pt")
        self.assertEqual(p.save_path, OTONOM_RETRAIN_SAVE_PATH)
        self.assertEqual(p.save_path, "data/anka_otonom_retrain_out.pt")
        self.assertEqual(p.vocab_path, OTONOM_RETRAIN_VOCAB_PATH)
        self.assertEqual(p.vocab_path, KANONIK_VOCAB)
        self.assertNotEqual(
            os.path.realpath(p.save_path), os.path.realpath(p.model_path),
            "Otonom retrain çıktısı taban checkpoint'in ÜZERİNE yazmamalıdır."
        )
        # data/*.pt donmuş desendir; ayrı çıktıya yazabilmek için --allow-frozen-write eşdeğeri
        self.assertTrue(p.allow_frozen_write)

    def test_retrain_yollari_env_ile_gecirilebilir(self):
        """RETRAIN_*_PATH ortam değişkenleri kanonik sabitleri geçersiz kılar."""
        os.environ["RETRAIN_MODEL_PATH"] = "data/anka_a1r.pt"
        os.environ["RETRAIN_SAVE_PATH"] = "data/anka_test_out.pt"
        os.environ["RETRAIN_VOCAB_PATH"] = KANONIK_VOCAB
        sup = PedagogicalSupervisor(gateway=self._mock_gateway())
        self.assertEqual(sup.retrain_pipeline.model_path, "data/anka_a1r.pt")
        self.assertEqual(sup.retrain_pipeline.save_path, "data/anka_test_out.pt")
        self.assertEqual(sup.retrain_pipeline.vocab_path, KANONIK_VOCAB)

    def test_save_path_taban_ustune_yazamaz(self):
        """save_path == model_path (donmuş taban) kurulumda fail-closed durur."""
        with self.assertRaises(RuntimeError):
            varsayilan_otonom_retrain_pipeline(save_path=OTONOM_RETRAIN_MODEL_PATH)
        os.environ["RETRAIN_SAVE_PATH"] = OTONOM_RETRAIN_MODEL_PATH
        with self.assertRaises(RuntimeError):
            varsayilan_otonom_retrain_pipeline()
        # eşdeğer yol yazımı (./data/... vs data/...) de yakalanır
        with self.assertRaises(RuntimeError):
            varsayilan_otonom_retrain_pipeline(
                model_path="./data/anka_taban_pure.pt",
                save_path="data/anka_taban_pure.pt",
            )

    def test_auto_retrain_adim_acik_yollarla_egitir(self):
        """E2E: auto_retrain=True + eşik dolu → run_training artık RuntimeError
        değil; train.py çağrısı AÇIK --load-path/--save-path/--vocab taşır."""
        with tempfile.TemporaryDirectory() as tmp:
            future = os.path.join(tmp, "future_train_vector.jsonl")
            archive = os.path.join(tmp, "future_train_archive.jsonl")
            binpath = os.path.join(tmp, "train_future_finetune.bin")
            model = os.path.join(tmp, "taban.pt")
            save = os.path.join(tmp, "otonom_out.pt")
            with open(model, "wb") as f:  # T-0148 5A ön-varlık kapısı
                f.write(b"")
            with open(future, "w", encoding="utf-8") as f:
                for _ in range(5):  # retrain_threshold varsayılanı 5
                    f.write(json.dumps({
                        "instruction": "Belgeye göre cevapla.",
                        "input": "sorgu: test",
                        "output": "yanıt",
                    }) + "\n")

            os.environ["RETRAIN_MODEL_PATH"] = model
            os.environ["RETRAIN_SAVE_PATH"] = save
            os.environ["RETRAIN_VOCAB_PATH"] = KANONIK_VOCAB
            os.environ["RETRAIN_FUTURE_PATH"] = future
            os.environ["RETRAIN_ARCHIVE_PATH"] = archive
            os.environ["RETRAIN_BIN_PATH"] = binpath

            sup = PedagogicalSupervisor(gateway=self._mock_gateway())
            probe = {
                "query": "Test sorusu?",
                "instruction": "test",
                "target_collection": "simulasyon_bellek",
                "knowledge_to_inject": "",
                "expected_keywords": ["test"],
            }
            with patch("src.gateway.retrain_pipeline.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="egitim tamam")
                step_log = sup.execute_supervision_step(probe, auto_inject=True, auto_retrain=True)

            retr = step_log.get("retrain_executed")
            self.assertIsNotNone(retr, "auto_retrain tetiklenmedi — backlog eşiği dolmuş olmalı")
            self.assertEqual(retr["status"], "success", retr)
            cmd = mock_run.call_args[0][0]
            self.assertIn("--load-path", cmd)
            self.assertIn(model, cmd)
            self.assertIn("--save-path", cmd)
            self.assertIn(save, cmd)
            self.assertNotIn(OTONOM_RETRAIN_MODEL_PATH, cmd)  # taban YALNIZCA yüklenir
            self.assertIn("--vocab", cmd)
            self.assertIn(KANONIK_VOCAB, cmd)
            self.assertIn("--allow-frozen-write", cmd)
            # başarılı eğitim birikimi arşive taşır
            self.assertTrue(os.path.exists(archive))


if __name__ == "__main__":
    unittest.main()