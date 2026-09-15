#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests and canaries for B1.5 dataset split leakage and import_examples safety.
T-0012: Closed-class skeleton dead code elimination and zero-template-leakage enforcement.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_b1_5_datasets import split_stratum_3way


class TestB15SplitLeakage(unittest.TestCase):

    def test_canary_split_stratum_groups_template_variants_together(self):
        """
        Canary Test: Two records sharing the same answer but varying in template/question
        must land on the SAME side of the split (both in train, both in val, or both in test).
        They must NEVER leak across train and test/val.
        """
        words = [
            "elma", "armut", "kiraz", "visne", "karpuz", "kavun", "cilek", "muz",
            "seftali", "kayisi", "erik", "incir", "uzum", "nar", "ayva", "portakal",
            "mandalina", "greyfurt", "limon", "dut", "bogurtlen", "ahududu", "yabanmersini",
            "kizilcik", "avokado", "ananas", "mango", "papaya", "kivi", "hindistancevizi"
        ]
        
        shared_answer = "Zıvana ahşap parçaları 90 derece açıyla birleştiren geleneksel bir yöntemdir."
        record_template_a = {
            "instruction": "Açıkla",
            "input": "Şablon A: Ahşap zanaatında zıvana bağlantısı nasıl yapılır?",
            "output": shared_answer
        }
        record_template_b = {
            "instruction": "Açıkla",
            "input": "Şablon B: Ahşap işlerinde zıvana geçme tekniği nedir?",
            "output": shared_answer
        }
        
        records = [record_template_a, record_template_b]
        for w in words:
            records.append({
                "instruction": "Soru",
                "input": f"Benzersiz {w} soru metni?",
                "output": f"Benzersiz {w} cevap metni."
            })
            
        # Run split_stratum_3way with seed=0, val_ratio=0.20, test_ratio=0.20
        train, val, test, stats = split_stratum_3way(
            records, stratum_name="canary_stratum", val_ratio=0.20, test_ratio=0.20, seed=0
        )
        
        # Determine split placement of template A and template B
        split_a = "train" if any(r["input"] == record_template_a["input"] for r in train) else (
            "val" if any(r["input"] == record_template_a["input"] for r in val) else "test"
        )
        split_b = "train" if any(r["input"] == record_template_b["input"] for r in train) else (
            "val" if any(r["input"] == record_template_b["input"] for r in val) else "test"
        )
        
        self.assertEqual(
            split_a, split_b,
            f"Sızıntı tespit edildi! Aynı cevabın şablon varyantları farklı bölmelere dağıldı: A={split_a}, B={split_b}"
        )

    def test_canary_import_examples_raises_on_missing_vocab(self):
        """
        Canary Test: scripts/import_examples.py must noisily raise FileNotFoundError
        if data/vocab.json is missing, rather than silently proceeding with an empty vocab.
        """
        from scripts.import_examples import import_gts_examples
        
        orig_exists = os.path.exists
        def mock_exists(path):
            if path == 'data/vocab.json':
                return False
            return orig_exists(path)
            
        with patch('os.path.exists', side_effect=mock_exists):
            with self.assertRaises(FileNotFoundError):
                import_gts_examples(max_examples=1)


if __name__ == "__main__":
    unittest.main()
