import unittest
import os
from src.llm.curriculum import CurriculumGenerator

class TestCurriculumGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = CurriculumGenerator()
        self.test_jsonl = "test_curriculum.jsonl"

    def tearDown(self):
        if os.path.exists(self.test_jsonl):
            os.remove(self.test_jsonl)

    def test_infancy_data(self):
        data = self.generator.generate_infancy_data()
        self.assertGreater(len(data), 0)
        self.assertIn("instruction", data[0])
        self.assertIn("input", data[0])
        self.assertIn("output", data[0])
        # Verify it contains contrastive examples
        self.assertIn("Pozitif", data[0]["output"])
        self.assertIn("Negatif", data[0]["output"])

    def test_parenting_data(self):
        data = self.generator.generate_parenting_data()
        self.assertGreater(len(data), 0)
        # Should contain compiler/morphotactics terminology
        self.assertTrue(any("ROOT:" in item["output"] or "TENSE_" in item["output"] for item in data))

    def test_specialization_data(self):
        # Test valid domain
        data = self.generator.generate_specialization_data("marangoz")
        self.assertGreater(len(data), 0)
        self.assertIn("zımpara", data[0]["output"].lower())

        # Test invalid/empty domain
        empty_data = self.generator.generate_specialization_data("uzay_mühendisliği")
        self.assertEqual(len(empty_data), 0)

    def test_export_to_jsonl(self):
        data = self.generator.generate_parenting_data()
        self.generator.export_to_jsonl(data, self.test_jsonl)
        
        self.assertTrue(os.path.exists(self.test_jsonl))
        
        # Read back and verify
        with open(self.test_jsonl, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(data))
            first_item = eval(lines[0])
            self.assertEqual(first_item["instruction"], data[0]["instruction"])

    def test_export_to_prompt_file(self):
        data = self.generator.generate_parenting_data()
        prompt_path = "test_curriculum.prompts.txt"
        try:
            self.generator.export_to_prompt_file(data, prompt_path)
            self.assertTrue(os.path.exists(prompt_path))
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("<INSTRUCTION>", content)
                self.assertIn("<INPUT>", content)
                self.assertIn("<OUTPUT>", content)
        finally:
            if os.path.exists(prompt_path):
                os.remove(prompt_path)

if __name__ == '__main__':
    unittest.main()
