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

    def test_case_alias_ikiz_satiri(self):
        # T-0147 A-2: TSV'de büyük-harfli özel-ad satırı + küçük-harfli gerçek
        # lemma satırı aynı trie-düğümünde birleştiğinde (ikiz-satır), seçim
        # sırası KANONİK (bayraksız) entry'den yanadır — koşum-1 P1
        # BIT_UYUMSUZ-deseni ('Meşrutiyet'im' ≠ 'meşrutiyetim') davranış-onarımıyla
        # kapanır; veri (TSV) değişmez, bayrak kaynak-kopyayı işaretler.
        with open(self.test_tsv, "a", encoding="utf-8") as f:
            f.write("Meşrutiyet\tNOUN\t-\n")
            f.write("meşrutiyet\tNOUN\t-\n")
        self.lexicon.load_from_tsv(self.test_tsv)
        stems = self.lexicon.find_stems("meşrutiyet")
        self.assertEqual(len(stems), 2)
        # kanonik küçük-harfli satır ÖNDE (deterministik; TSV-sırasına değil
        # bayrağa bağlı — _insert kanonik-girişi alias'ın önüne alır)
        self.assertEqual(stems[0][1]["lemma"], "meşrutiyet")
        self.assertFalse(stems[0][1].get("is_case_alias", False))
        # düz-lower ikiz-KOPYASI alias-bayraklıdır (lemma'sı büyük-harfli asıldır)
        self.assertEqual(stems[1][1]["lemma"], "Meşrutiyet")
        self.assertTrue(stems[1][1].get("is_case_alias", False))

if __name__ == '__main__':
    unittest.main()
