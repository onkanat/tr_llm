import unittest
from src.compiler.phonology import PhonologyEngine

class TestPhonologyEngine(unittest.TestCase):
    
    def test_major_vowel_harmony(self):
        self.assertEqual(PhonologyEngine.resolve_affix("kitap", "lAr")[1], "lar")
        self.assertEqual(PhonologyEngine.resolve_affix("göz", "lAr")[1], "ler")
        self.assertEqual(PhonologyEngine.resolve_affix("araba", "DA")[1], "da")
        self.assertEqual(PhonologyEngine.resolve_affix("ev", "DAn")[1], "den")

    def test_minor_vowel_harmony(self):
        self.assertEqual(PhonologyEngine.resolve_affix("kız", "(I)m")[1], "ım")
        self.assertEqual(PhonologyEngine.resolve_affix("ev", "(I)m")[1], "im")
        self.assertEqual(PhonologyEngine.resolve_affix("yol", "(I)m")[1], "um")
        self.assertEqual(PhonologyEngine.resolve_affix("göz", "(I)m")[1], "üm")

    def test_buffer_consonants(self):
        self.assertEqual(PhonologyEngine.resolve_affix("oku", "(y)AcAk")[1], "yacak")
        self.assertEqual(PhonologyEngine.resolve_affix("yürü", "(y)AcAk")[1], "yecek")
        self.assertEqual(PhonologyEngine.resolve_affix("gel", "(y)AcAk")[1], "ecek")
        self.assertEqual(PhonologyEngine.resolve_affix("yaz", "(y)AcAk")[1], "acak")
        
        self.assertEqual(PhonologyEngine.resolve_affix("araba", "(s)I")[1], "sı")
        self.assertEqual(PhonologyEngine.resolve_affix("ev", "(s)I")[1], "i")

    def test_buffer_vowels(self):
        self.assertEqual(PhonologyEngine.resolve_affix("kitap", "(I)m")[1], "ım")
        self.assertEqual(PhonologyEngine.resolve_affix("araba", "(I)m")[1], "m")

    def test_consonant_mutation(self):
        self.assertEqual(PhonologyEngine.resolve_affix("yap", "DI")[1], "tı")
        self.assertEqual(PhonologyEngine.resolve_affix("git", "DI")[1], "ti")
        self.assertEqual(PhonologyEngine.resolve_affix("yaz", "DI")[1], "dı")
        self.assertEqual(PhonologyEngine.resolve_affix("gel", "DI")[1], "di")
        
        self.assertEqual(PhonologyEngine.resolve_affix("Türk", "CA")[1], "çe")
        self.assertEqual(PhonologyEngine.resolve_affix("İngiliz", "CA")[1], "ce")
        
    def test_complex_templates(self):
        self.assertEqual(PhonologyEngine.resolve_affix("gel", "Iyor")[1], "iyor")
        self.assertEqual(PhonologyEngine.resolve_affix("yaz", "Iyor")[1], "ıyor")
        self.assertEqual(PhonologyEngine.resolve_affix("koş", "Iyor")[1], "uyor")
        self.assertEqual(PhonologyEngine.resolve_affix("gör", "Iyor")[1], "üyor")

if __name__ == '__main__':
    unittest.main()
