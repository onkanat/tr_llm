import re

class PhonologyEngine:
    VOWELS = set("aeıioöuüAEIİOÖUÜ")
    BACK_VOWELS = set("aıouAIOU")
    FRONT_VOWELS = set("eiöüEİÖÜ")
    ROUNDED_VOWELS = set("oöuüOÖUÜ")
    UNROUNDED_VOWELS = set("aeıiAEIİ")
    UNVOICED_CONSONANTS = set("fstkçşhpFSTKÇŞHP")

    @classmethod
    def _get_last_vowel(cls, word: str) -> str:
        for char in reversed(word):
            if char in cls.VOWELS:
                return char
        return 'a' # Fallback

    @classmethod
    def _ends_with_vowel(cls, word: str) -> bool:
        if not word: return False
        return word[-1] in cls.VOWELS

    @classmethod
    def _ends_with_unvoiced(cls, word: str) -> bool:
        if not word: return False
        return word[-1] in cls.UNVOICED_CONSONANTS

    @classmethod
    def resolve_affix(cls, stem: str, template: str) -> tuple[str, str]:
        """
        Resolves an abstract affix template into its surface form based on the stem.
        Returns a tuple: (mutated_stem, resolved_affix)
        This handles cases where adding a vowel suffix mutates the stem (e.g. ecek + im -> eceğ + im)
        """
        if not template:
            return stem, ""

        resolved = ""
        mutated_stem = stem
        last_vowel = cls._get_last_vowel(stem)
        ends_with_vowel = cls._ends_with_vowel(stem)
        ends_with_unvoiced = cls._ends_with_unvoiced(stem)
        
        # Determine harmonies
        is_back = last_vowel in cls.BACK_VOWELS
        is_rounded = last_vowel in cls.ROUNDED_VOWELS

        def apply_stem_mutation(stem_to_mutate):
            """Applies consonant mutation to the end of the stem when followed by a vowel."""
            if not stem_to_mutate: return stem_to_mutate
            last_char = stem_to_mutate[-1]
            voicing_map = {'p': 'b', 'ç': 'c', 't': 'd', 'k': 'ğ'}
            if last_char in voicing_map:
                return stem_to_mutate[:-1] + voicing_map[last_char]
            return stem_to_mutate

        i = 0
        
        while i < len(template):
            char = template[i]

            # Handle optional buffer characters e.g., (y), (I)
            if char == '(' and i + 2 < len(template) and template[i+2] == ')':
                buffer_char = template[i+1]
                i += 3
                
                if buffer_char in "ysnş":
                    if ends_with_vowel:
                        resolved += buffer_char
                        ends_with_vowel = False
                elif buffer_char in "IA":
                    if not ends_with_vowel:
                        if buffer_char == 'I':
                            resolved += cls._resolve_I(is_back, is_rounded)
                        elif buffer_char == 'A':
                            resolved += 'a' if is_back else 'e'
                        ends_with_vowel = True
                continue

            # Handle abstract placeholders
            if char == 'A':
                resolved += 'a' if is_back else 'e'
                ends_with_vowel = True
            elif char == 'I':
                resolved += cls._resolve_I(is_back, is_rounded)
                ends_with_vowel = True
            elif char == 'D':
                resolved += 't' if ends_with_unvoiced else 'd'
                ends_with_vowel = False
                ends_with_unvoiced = False
            elif char == 'C':
                resolved += 'ç' if ends_with_unvoiced else 'c'
                ends_with_vowel = False
                ends_with_unvoiced = False
            else:
                # Normal character
                resolved += char
                ends_with_vowel = char in cls.VOWELS
                ends_with_unvoiced = char in cls.UNVOICED_CONSONANTS
            i += 1

        if resolved and resolved[0] in cls.VOWELS:
            mutated_stem = apply_stem_mutation(mutated_stem)

        return mutated_stem, resolved

    @classmethod
    def _resolve_I(cls, is_back: bool, is_rounded: bool) -> str:
        if is_back and not is_rounded:
            return 'ı'
        elif is_back and is_rounded:
            return 'u'
        elif not is_back and not is_rounded:
            return 'i'
        elif not is_back and is_rounded:
            return 'ü'
        return 'ı' # Fallback
