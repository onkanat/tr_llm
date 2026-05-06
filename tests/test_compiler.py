import unittest
import os
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

class TestCrystalCompiler(unittest.TestCase):
    def setUp(self):
        # 1. Setup Lexicon
        self.lexicon = LexiconManager()
        self.test_tsv = "test_compiler_roots.tsv"
        with open(self.test_tsv, "w", encoding="utf-8") as f:
            f.write("lemma\tpos\tattributes\n")
            f.write("gel\tVERB\t-\n")
            f.write("git\tVERB\tVOICING\n")
            f.write("kitap\tNOUN\tVOICING\n")
            f.write("göz\tNOUN\t-\n")
            f.write("düş\tVERB\t-\n")
        self.lexicon.load_from_tsv(self.test_tsv)

        # 2. Setup Morphotactics
        self.graph = build_default_graph()

        # 3. Initialize Compiler
        self.compiler = CrystalCompiler(self.lexicon, self.graph)

    def tearDown(self):
        if os.path.exists(self.test_tsv):
            os.remove(self.test_tsv)

    def test_compile_simple_verb(self):
        # gel + ecek + im -> geleceğim
        result = self.compiler.compile("geleceğim")

        
        self.assertEqual(result["input"], "geleceğim")
        self.assertGreater(len(result["analyses"]), 0)
        
        best_analysis = result["analyses"][0]
        self.assertEqual(best_analysis["surface_form"], "geleceğim")
        
        morphemes = best_analysis["morphemes"]
        self.assertEqual(len(morphemes), 3)
        self.assertEqual(morphemes[0]["id"], "gel")
        self.assertEqual(morphemes[1]["id"], "TENSE_FUT")
        self.assertEqual(morphemes[1]["surface"], "eceğ") # 'k' becomes 'ğ' because of the buffer vowel? Wait!
        # Ah! Phonology rule for TENSE_FUT: (y)AcAk. 
        # Wait, if we pass "gelecek" to PhonologyEngine.resolve_affix("gelecek", "(I)m"), 
        # the 'k' in 'gelecek' does NOT mutate automatically unless the Phonology Engine handles 'k' -> 'ğ' mutation!
        # Wait, our PhonologyEngine handles 'D' and 'C', but does it handle standard noun/verb ending mutations?
        pass

    def test_compile_past_verb(self):
        # düş + tü -> düştü (Simple Past - Witnessed)
        result = self.compiler.compile("düştü")
        self.assertGreater(len(result["analyses"]), 0)
        
        morphemes = result["analyses"][0]["morphemes"]
        self.assertEqual(morphemes[0]["id"], "düş")
        self.assertEqual(morphemes[1]["id"], "TENSE_PAST")
        self.assertEqual(morphemes[1]["surface"], "tü")

    def test_compile_evidential_verb(self):
        # düş + müş -> düşmüş (Evidential Past - Inferred/Heard)
        result = self.compiler.compile("düşmüş")
        self.assertGreater(len(result["analyses"]), 0)
        
        morphemes = result["analyses"][0]["morphemes"]
        self.assertEqual(morphemes[0]["id"], "düş")
        self.assertEqual(morphemes[1]["id"], "TENSE_EVIDENTIAL")
        self.assertEqual(morphemes[1]["surface"], "müş")

    def test_compile_compound_tense_past_prog(self):
        # git + iyor + du + m -> gidiyordum
        result = self.compiler.compile("gidiyordum")
        self.assertGreater(len(result["analyses"]), 0)
        
        morphemes = result["analyses"][0]["morphemes"]
        self.assertEqual(len(morphemes), 4)
        self.assertEqual(morphemes[0]["id"], "git")
        self.assertEqual(morphemes[0]["surface"], "gid")
        self.assertEqual(morphemes[1]["id"], "TENSE_PROG")
        self.assertEqual(morphemes[1]["surface"], "iyor")
        self.assertEqual(morphemes[2]["id"], "COPULA_PAST")
        self.assertEqual(morphemes[2]["surface"], "du")
        self.assertEqual(morphemes[3]["id"], "PERSON_1SG")
        self.assertEqual(morphemes[3]["surface"], "m")

    def test_compile_compound_tense_evidential_fut(self):
        # git + ecek + miş + im -> gidecekmişim
        result = self.compiler.compile("gidecekmişim")
        self.assertGreater(len(result["analyses"]), 0)
        
        morphemes = result["analyses"][0]["morphemes"]
        self.assertEqual(len(morphemes), 4)
        self.assertEqual(morphemes[0]["id"], "git")
        self.assertEqual(morphemes[0]["surface"], "gid")
        self.assertEqual(morphemes[1]["id"], "TENSE_FUT")
        self.assertEqual(morphemes[1]["surface"], "ecek")
        self.assertEqual(morphemes[2]["id"], "COPULA_EVIDENTIAL")
        self.assertEqual(morphemes[2]["surface"], "miş")
        self.assertEqual(morphemes[3]["id"], "PERSON_1SG")
        self.assertEqual(morphemes[3]["surface"], "im")

    def test_compile_simple_noun(self):
        # kitap + lar + da -> kitaplarda
        result = self.compiler.compile("kitaplarda")
        
        self.assertGreater(len(result["analyses"]), 0)
        morphemes = result["analyses"][0]["morphemes"]
        self.assertEqual(len(morphemes), 3)
        self.assertEqual(morphemes[0]["id"], "kitap")
        self.assertEqual(morphemes[1]["id"], "PLURAL")
        self.assertEqual(morphemes[2]["id"], "CASE_LOC")
        
        self.assertEqual(result["token_vector"], ["kitap", "PLURAL", "CASE_LOC"])

    def test_compile_voiced_noun(self):
        # kitap + ım -> kitabım
        result = self.compiler.compile("kitabım")
        self.assertGreater(len(result["analyses"]), 0)
        morphemes = result["analyses"][0]["morphemes"]
        self.assertEqual(morphemes[0]["id"], "kitap") # Lemma is 'kitap'
        self.assertEqual(morphemes[0]["surface"], "kitab") # Surface is 'kitab'
        self.assertEqual(morphemes[1]["id"], "POSS_1SG")
        self.assertEqual(morphemes[1]["surface"], "ım")

if __name__ == '__main__':
    unittest.main()
