import pytest
from src.compiler.phonology import PhonologyEngine

@pytest.mark.parametrize("stem, template, attributes, expected_stem, expected_affix", [
    # 1. Vowel Harmony (A-Type)
    ("kol", "lAr", "-", "kol", "lar"),
    ("gel", "lAr", "-", "gel", "ler"),
    
    # 2. Vowel Harmony (I-Type)
    ("kitap", "(I)m", "-", "kitap", "ım"),
    ("kedi", "(I)m", "-", "kedi", "m"),
    ("göz", "(I)m", "-", "göz", "üm"),
    ("gül", "(I)m", "-", "gül", "üm"),
    
    # 3. Voicing (Ünsüz Yumuşaması)
    ("kitap", "(I)m", "VOICING", "kitab", "ım"),
    ("ağaç", "(y)A", "VOICING", "ağac", "a"),
    ("kanat", "(I)m", "VOICING", "kanad", "ım"),
    ("renk", "(I)", "VOICING", "reng", "i"),
    
    # 4. Vowel Drop (Hece Düşmesi)
    ("ağız", "(I)", "VOWEL_DROP", "ağz", "ı"),
    ("burun", "(I)m", "VOWEL_DROP", "burn", "um"),
    ("oğul", "(I)", "VOWEL_DROP", "oğl", "u"),
    
    # 5. Consonant Hardening (Sertleşme)
    ("kitap", "DA", "-", "kitap", "ta"),
    ("ağaç", "DA", "-", "ağaç", "ta"),
    ("gel", "DA", "-", "gel", "de"),
    ("yap", "DI", "-", "yap", "tı"),
    
    # 6. Buffers (y, s, n, ş)
    ("araba", "(y)A", "-", "araba", "ya"),
    ("anne", "(s)I", "-", "anne", "si"),
    ("odada", "(n)In", "-", "odada", "nın"),
    ("iki", "(ş)Ar", "-", "iki", "şer"),
    
    # 7. Complex: Voicing + Vowel Drop (If any exists)
    # Most Turkish words do one or the other, but let's check
    
    # 8. Derivational Affixes
    ("göz", "lIk", "VOICING", "göz", "lük"), # Suffix itself has voicing for future
    ("gözlük", "(I)", "VOICING", "gözlüğ", "ü"), # Derivation result voiced
    
    # 9. Case + 3rd Person Possessive (n buffer)
    ("odası", "nDA", "-", "odası", "nda"),
    ("evi", "nA", "-", "evi", "ne"),
])
def test_phonology_engine_golden(stem, template, attributes, expected_stem, expected_affix):
    mutated_stem, resolved_affix = PhonologyEngine.resolve_affix(stem, template, attributes)
    assert mutated_stem == expected_stem
    assert resolved_affix == expected_affix

if __name__ == "__main__":
    pytest.main([__file__])
