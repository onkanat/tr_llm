import re

class PhonologyEngine:
    # D3-C (T-0080): düzeltme işaretli ünlüler kümede YOKTU ⇒ `_ends_with_vowel`
    # onları ÜNSÜZ sayıyordu ⇒ 'askerî'+DAT 'askerîe' (doğrusu 'askerîye'),
    # 'hâlâ'+DAT 'hâlâa' (doğrusu 'hâlâya'). 152 lemma düzeltme işaretiyle BİTER.
    #
    # ⚠️ İKİ EKSEN AYRILIR (ölçüldü, ilk denemem YANLIŞTI): harfi `VOWELS`'a
    # eklemek ünlü-sonluluk/tampon eksenini DÜZELTİR, ama UYUM eksenini BOZAR.
    # `_get_last_vowel` düzeltme işaretini görürse: 'efkâr'+iyelik 'efkârı' olur
    # (doğrusu 'efkâri'; eski kod önceki 'e'yi bulup ÖN veriyordu = doğru).
    # Ölçülen tablo (â uyum bilgisi TAŞIMAZ — D1'deki önek/uzatma beraberliğiyle
    # aynı yapı): 'efkâr'→ön · 'kâr'→art · 'hâl'→ön; üçü de aynı yapıda, zıt uyum.
    # ⇒ düzeltme işaretleri YALNIZ VOWELS'a girer; `HARMONY_VOWELS` onları ATLAR,
    # böylece ESKİ uyum davranışı birebir korunur ve `hâl` ADIYLA BİLİNEN İSTİSNA
    # olarak Faz 1'de lemma niteliğiyle çözülür — sessizce yanlış tarafa yazılmaz.
    VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
    HARMONY_VOWELS = set("aeıioöuüAEIİOÖUÜ")
    BACK_VOWELS = set("aıouAIOU")
    FRONT_VOWELS = set("eiöüEİÖÜ")
    ROUNDED_VOWELS = set("oöuüOÖUÜ")
    UNROUNDED_VOWELS = set("aeıiAEIİ")
    UNVOICED_CONSONANTS = set("fstkçşhpFSTKÇŞHP")

    @classmethod
    def _get_last_vowel(cls, word: str) -> str:
        # HARMONY_VOWELS: düzeltme işaretleri ATLANIR (bkz. yukarıdaki ölçüm).
        for char in reversed(word):
            if char in cls.HARMONY_VOWELS:
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
    def mutate_stem(cls, stem: str, attributes: str, next_is_vowel: bool) -> str:
        """
        Applies phonetic mutations to the stem based on its attributes and the next character.
        """
        if not stem or not next_is_vowel:
            return stem

        mutated = stem
        
        # 1. VOWEL_DROP (Hece Düşmesi) e.g., ağız -> ağz-ı
        if "VOWEL_DROP" in attributes:
            if len(stem) >= 3:
                vowel_indices = [i for i, char in enumerate(stem) if char in cls.VOWELS]
                if len(vowel_indices) >= 2:
                    last_vowel_idx = vowel_indices[-1]
                    if last_vowel_idx == len(stem) - 2:
                        mutated = stem[:last_vowel_idx] + stem[last_vowel_idx+1:]
        
        # 2. VOICING (Ünsüz Yumuşaması) e.g., kitap -> kitab-ı
        if "VOICING" in attributes:
            last_char = mutated[-1]
            voicing_map = {'p': 'b', 'ç': 'c', 't': 'd', 'k': 'ğ'}
            if last_char in voicing_map:
                new_char = voicing_map[last_char]
                if last_char == 'k' and len(mutated) >= 2 and mutated[-2].lower() == 'n':
                    new_char = 'g'
                mutated = mutated[:-1] + new_char

        # 3. GEMINATION (Ünsüz İkizleşmesi) e.g., hak -> hakk-ı, his -> hiss-i,
        # af -> aff-ı, zam -> zamm-ı, ret -> redd-i (VOICING + GEMINATION birlikte:
        # ÖNCE yumuşama sonra ikizleşme ⇒ 'ret'->'red'->'redd'). Nitelik sözlükten
        # gelir (Faz 1 verisi): ikizleşme EK ile değil LEMMA ile belirlenir, bu
        # yüzden kural değil veri olarak taşınır. Ünlüyle biten gövde ikizlenmez.
        if "GEMINATION" in attributes and mutated:
            son = mutated[-1]
            if son not in cls.VOWELS:
                mutated = mutated + son
        
        return mutated

    @classmethod
    def resolve_affix(cls, stem: str, template: str, attributes: str = "-",
                       lemma_attributes: str = "-") -> tuple[str, str]:
        """
        Resolves an abstract affix template into its surface form based on the stem.
        Returns a tuple: (mutated_stem, resolved_affix)

        `attributes`: bir ÖNCEKİ morfemin (çoğunlukla gövdenin) işlemsel
        nitelikleri (VOWEL_DROP/VOICING/GEMINATION...) — mutate_stem'ı sürer.
        `lemma_attributes`: KÖKÜN sözleşme nitelikleri (T2.2). Yalnız uyum
        eksenini (is_back) etkiler: FRONT_HARMONY/BACK_HARMONY, ödünçleme
        gövdesinin son ünlüsünün YALNITTIĞI uyumu veriyle düzeltir
        ('ihtimal'→'ihtimali', 'hâl'→'hâli' sınıfı; kural hard-code edilmez).
        Nitelik yoksa ('-') davranış bit-özdeş kalır.
        """
        if not template:
            return stem, ""

        last_vowel = cls._get_last_vowel(stem)
        is_back = last_vowel in cls.BACK_VOWELS
        is_rounded = last_vowel in cls.ROUNDED_VOWELS

        # T2.3 — -(I)yor ünlü kaynaşması (yalnız bu şablonda VE ünlü-sonlu
        # gövdede): gövdenin son ünlüsü DÜZ (a/e/ı/i) ise ünlü düşer, tampon-y
        # kalkar (anla→anl+"Iyor"→anlıyor); YUVARLAK (o/ö/u/ü) ise I düşer,
        # tampon-y kalır (oku→oku+"(y)or"→okuyor). Uyum, KALAN gövdenin son
        # ünlüsünden YENİDEN hesaplanır: 'oyna'→'oyn' dönüşünde uyum taşıyıcısı
        # düşen 'a' değil kalan 'o'dur (-ıyor değil -uyor).
        #
        # K1 (Faz-2) — ünlüsüz-kalan gövde kaynaşması: 2-karakter gövde
        # ('de', 'ye') kırpılınca ÜNLÜ KALMAZ ('d', 'y'); yeniden hesap
        # _get_last_vowel'in 'a' fallback'ına düşer ve olmayan art-uyumlu
        # yüzey üretir ('dıyor'/'yıyor'). Kırpma sonrası gövdede ünlü
        # KALMADIYSA kırpma ÖNCESİ is_back/is_rounded KORUNUR (yeniden hesap
        # yapılmaz); ünlü kaldıysa mevcut yeniden hesap sürer.
        if template == "(y)Iyor" and cls._ends_with_vowel(stem) and len(stem) >= 2:
            if last_vowel in cls.UNROUNDED_VOWELS:
                stem = stem[:-1]
                template = "Iyor"
                if any(ch in cls.HARMONY_VOWELS for ch in stem):
                    last_vowel = cls._get_last_vowel(stem)
                    is_back = last_vowel in cls.BACK_VOWELS
                    is_rounded = last_vowel in cls.ROUNDED_VOWELS
                # else: gövde ünlüsüz kaldı (de→d, ye→y) — kırpma öncesi
                # is_back/is_rounded korunur ('e'→ön→'i' → 'diyor'/'yiyor').
            else:
                template = "(y)or"
                last_vowel = cls._get_last_vowel(stem)
                is_back = last_vowel in cls.BACK_VOWELS
                is_rounded = last_vowel in cls.ROUNDED_VOWELS

        # T2.2 — lemma niteliği uyum eksenini ÇEVİRİR (gövde ünlüsünden sonra).
        # Basit tutulur: yalnız is_back döner; is_rounded gövdenin son ünlüsünden
        # kalır (FRONT_HARMONY + yuvarlak-son-ünlü birleşimi sözlükte yok).
        if "FRONT_HARMONY" in lemma_attributes:
            is_back = False
        elif "BACK_HARMONY" in lemma_attributes:
            is_back = True
        
        # Determine current state for hardening/buffer
        current_ends_unvoiced = cls._ends_with_unvoiced(stem)
        current_ends_vowel = cls._ends_with_vowel(stem)

        resolved = ""
        i = 0
        starts_with_vowel = False
        
        # Peek to see if it starts with a vowel or a vowel-inducing buffer
        if template.startswith('('):
            if len(template) >= 3 and template[2] == ')':
                buffer_char = template[1]
                if buffer_char in "IA" and not current_ends_vowel:
                    starts_with_vowel = True
                elif buffer_char in "ysnş" and current_ends_vowel:
                    starts_with_vowel = False
        elif template[0] in "AI":
            starts_with_vowel = True
        elif template[0] in cls.VOWELS:
            starts_with_vowel = True

        while i < len(template):
            char = template[i]

            if char == '(' and i + 2 < len(template) and template[i+2] == ')':
                buffer_char = template[i+1]
                i += 3
                if buffer_char in "ysnş":
                    if current_ends_vowel:
                        resolved += buffer_char
                        current_ends_vowel = False
                        current_ends_unvoiced = False
                elif buffer_char in "IA":
                    if not current_ends_vowel:
                        vowel_char = cls._resolve_I(is_back, is_rounded) if buffer_char == 'I' else ('a' if is_back else 'e')
                        resolved += vowel_char
                        current_ends_vowel = True
                        current_ends_unvoiced = False
                        is_back = vowel_char in cls.BACK_VOWELS
                        is_rounded = vowel_char in cls.ROUNDED_VOWELS
                continue

            if char == 'A':
                vowel_char = 'a' if is_back else 'e'
                resolved += vowel_char
                current_ends_vowel = True
                current_ends_unvoiced = False
                is_back = vowel_char in cls.BACK_VOWELS
                is_rounded = vowel_char in cls.ROUNDED_VOWELS
            elif char == 'I':
                vowel_char = cls._resolve_I(is_back, is_rounded)
                resolved += vowel_char
                current_ends_vowel = True
                current_ends_unvoiced = False
                is_back = vowel_char in cls.BACK_VOWELS
                is_rounded = vowel_char in cls.ROUNDED_VOWELS
            elif char == 'D':
                resolved += 't' if current_ends_unvoiced else 'd'
                current_ends_vowel = False
                current_ends_unvoiced = resolved[-1] in cls.UNVOICED_CONSONANTS
            elif char == 'C':
                resolved += 'ç' if current_ends_unvoiced else 'c'
                current_ends_vowel = False
                current_ends_unvoiced = resolved[-1] in cls.UNVOICED_CONSONANTS
            else:
                resolved += char
                current_ends_vowel = char in cls.VOWELS
                current_ends_unvoiced = char in cls.UNVOICED_CONSONANTS
                if current_ends_vowel:
                    is_back = char in cls.BACK_VOWELS
                    is_rounded = char in cls.ROUNDED_VOWELS
            i += 1

        mutated_stem = stem
        if starts_with_vowel or (resolved and resolved[0] in cls.VOWELS):
             mutated_stem = cls.mutate_stem(stem, attributes, True)

        return mutated_stem, resolved

    @classmethod
    def _resolve_I(cls, is_back: bool, is_rounded: bool) -> str:
        if is_back and not is_rounded: return 'ı'
        if is_back and is_rounded: return 'u'
        if not is_back and not is_rounded: return 'i'
        return 'ü'
