import unittest
import os
from src.compiler.lexicon import LexiconManager

class TestLexiconManager(unittest.TestCase):
    def setUp(self):
        self.lexicon = LexiconManager()
        self.test_tsv = "test_roots.tsv"
        # Create a temporary TSV file for testing
        with open(self.test_tsv, "w", encoding="utf-8") as f:
            f.write("lemma\tpos\tattributes\n")
            f.write("gel\tVERB\t-\n")
            f.write("kitap\tNOUN\tVOICING\n")
            f.write("göz\tNOUN\t-\n")
            f.write("git\tVERB\tVOICING\n")
        self.lexicon.load_from_tsv(self.test_tsv)

    def tearDown(self):
        if os.path.exists(self.test_tsv):
            os.remove(self.test_tsv)

    def test_find_exact_stem(self):
        stems = self.lexicon.find_stems("geliyorum")
        self.assertEqual(len(stems), 1)
        self.assertEqual(stems[0][0], "gel")
        self.assertEqual(stems[0][1]["lemma"], "gel")
        self.assertEqual(stems[0][1]["pos"], "VERB")

    def test_find_voiced_stem(self):
        # 'kitab' is the prefix in 'kitabım'
        stems = self.lexicon.find_stems("kitabım")
        self.assertEqual(len(stems), 1)
        self.assertEqual(stems[0][0], "kitab")
        self.assertEqual(stems[0][1]["lemma"], "kitap")
        self.assertEqual(stems[0][1]["attributes"], "VOICING")

    def test_find_voiced_verb_stem(self):
        # 'gid' is the prefix in 'gidiyorum' -> root is 'git'
        stems = self.lexicon.find_stems("gidiyorum")
        self.assertEqual(len(stems), 1)
        self.assertEqual(stems[0][0], "gid")
        self.assertEqual(stems[0][1]["lemma"], "git")

    def test_no_stem_found(self):
        stems = self.lexicon.find_stems("bilgisayar")
        self.assertEqual(len(stems), 0)

    def test_multiple_stems(self):
        # Let's add an overlapping root manually for testing
        self.lexicon._insert("ge", {"lemma": "ge", "pos": "NOUN", "attributes": "-"})
        stems = self.lexicon.find_stems("geliyorum")
        # Should find both 'ge' and 'gel'
        self.assertEqual(len(stems), 2)
        lemmas = [s[1]["lemma"] for s in stems]
        self.assertIn("ge", lemmas)
        self.assertIn("gel", lemmas)

if __name__ == '__main__':
    unittest.main()
