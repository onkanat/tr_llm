#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from src.compiler.core import CrystalCompiler
from src.compiler.phonology import PhonologyEngine

META_TAGS = {
    "root", "case", "ten", "plural", "evet", "hayır", "var", "yok",
    "belge:", "sorgu:", "tanım:", "cevap:",
    "<ARA>", "</ARA>", "<BELGE>", "</BELGE>", "<DUSUNCE>", "</DUSUNCE>", "ara", "belge", "düşünce"
}

SUFFIX_PREFIXES = (
    "TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_", "DERIV_", "REL_", "VOICE_", "IMPOTENTIAL_"
)
SUFFIX_EXACTS = {"PLURAL", "POTENTIAL", "NEG", "IMPOTENTIAL_NEG", "REL_ki"}

PAST_PERSON_TEMPLATES = {
    "PERSON_1SG": "m",
    "PERSON_2SG": "n",
    "PERSON_3SG": "",
    "PERSON_1PL": "k",
    "PERSON_2PL": "nIz",
    "PERSON_3PL": "lAr"
}

COMMON_FALLBACKS = {
    "PLURAL": {"template": "lAr", "attributes": "-"},
    "POTENTIAL": {"template": "(y)Abil", "attributes": "-"},
    "NEG": {"template": "mA", "attributes": "-"},
    "IMPOTENTIAL_NEG": {"template": "(y)AmA", "attributes": "-"},
    "REL_ki": {"template": "ki", "attributes": "-"},
    "COPULA_PAST": {"template": "(y)DI", "attributes": "-"},
    "COPULA_EVIDENTIAL": {"template": "(y)mIş", "attributes": "-"},
    "COPULA_COND": {"template": "(y)sA", "attributes": "-"},
    "COPULA_AORIST": {"template": "DIr", "attributes": "-"},
}

class MorphemeDecompiler:
    """
    decomp0 (Crystal Decompiler / Phonetic Synthesizer):
    Applies the phonetic harmony function f(·) (vowel harmony, consonant mutation,
    buffer letters) to generated morpheme sequences to reconstruct fluent Turkish surface text.
    """
    def __init__(self, compiler: CrystalCompiler, vocab: Optional[Vocabulary] = None):
        self.compiler = compiler
        self.vocab = vocab
        
        # T-0147 A-1: affix_id → şablon LİSTESİ (çok-şablonlu id'ler için).
        # Şablonlar şablon-dizgesine göre SIRALI (deterministik tie-break —
        # graph ekleme-sırası değil); aynı (template, attributes) tekrarı
        # tek girişte dedupe edilir (ör. CASE_INS tek-şablon re-use bit-özdeş).
        self.affix_info: Dict[str, List[Dict[str, Any]]] = {}
        for state, transitions in compiler.graph.transitions.items():
            for trans in transitions:
                bilgi = {
                    "template": trans.affix_template,
                    "attributes": trans.attributes
                }
                liste = self.affix_info.setdefault(trans.affix_id, [])
                if bilgi not in liste:
                    liste.append(bilgi)
        for affix_id in self.affix_info:
            self.affix_info[affix_id].sort(key=lambda b: b["template"])

    def is_suffix(self, tag: str) -> bool:
        """Returns True if the tag represents an affix/suffix."""
        return tag.startswith(SUFFIX_PREFIXES) or tag in SUFFIX_EXACTS

    def decompile_tags(self, tags: List[str]) -> str:
        """
        Decompiles/reconstructs a list of morpheme tags back into a Turkish surface word.
        Uses phonology rules to correctly resolve vowel harmony and consonant changes.
        """
        if not tags:
            return ""
            
        clean_tags = [t for t in tags if not (t.startswith("<") and t.endswith(">"))]
        if not clean_tags:
            return ""
            
        # If single tag and it is an affix (e.g. CASE_DAT), return as-is
        if len(clean_tags) == 1 and self.is_suffix(clean_tags[0]):
            return clean_tags[0]

        # 1. Identify root lemma (first tag)
        root_lemma = clean_tags[0]
        
        # If the root is a meta keyword, don't affix
        # T-0103 düzeltmesi: 'ara' hem meta marker hem GERÇEK lexicon
        # köküdür (ara-). Meta-kökü yalnız takip eden tag'lerin TÜMÜ
        # suffix DEĞİLSE ham bırak; hepsi suffix ise normal affix yoluna
        # gir (ör. 'ara POSS_3SG CASE_LOC_N' → 'arasında').
        if root_lemma.lower() in META_TAGS:
            kalan = clean_tags[1:]
            if not all(self.is_suffix(t) for t in kalan):
                return " ".join(clean_tags)
            
        stems = self.compiler.lexicon.find_stems(root_lemma)
        root_entry = None
        for stem_prefix, entry in stems:
            if entry['lemma'] == root_lemma:
                root_entry = entry
                break
                
        is_placeholder = root_lemma in ("[Özel İsim]", "[sayı]", "[?]") or root_lemma.isdigit()
        is_proper_noun = bool(root_lemma and (root_lemma[0].isupper() or root_lemma.isupper()) and not root_lemma.startswith("["))
        # T-0147 A-2: seçilen kök-entry özel-ad İKİZ-KOPYAsı ise (is_case_alias)
        # kanonik asıl küçük-harfli lemma'dır → apostrof-dalı tetiklenmez.
        if root_entry and root_entry.get('is_case_alias'):
            is_proper_noun = False
        if is_placeholder:
            root_surface = root_lemma
            root_attrs = "-"
        elif is_proper_noun and not root_entry:
            root_surface = root_lemma
            root_attrs = "-"
        elif not root_entry:
            root_surface = root_lemma
            root_attrs = "-"
        else:
            root_surface = root_lemma
            root_attrs = root_entry.get('attributes', '-')
            
        current_surface = root_surface
        current_attrs = root_attrs
        
        # 2. Iteratively attach affixes using PhonologyEngine
        for idx in range(1, len(clean_tags)):
            affix_id = clean_tags[idx]
            stem_for_phonology = "isim" if is_placeholder and idx == 1 else current_surface
            adaylar = self.affix_info.get(affix_id)
            on_cozum = None
            if adaylar is None:
                info = COMMON_FALLBACKS.get(affix_id, {"template": "", "attributes": "-"})
            elif len(adaylar) == 1:
                # T-0147 A-1: tek-şablonlu id → mevcut davranış bit-özdeş.
                info = adaylar[0]
            else:
                # T-0147 A-1 (çözümle-seç): çok-şablonlu id (ör. GERUND_KEN).
                # Adaylar kurulumda şablon-dizgesine göre sorted; PhonologyEngine
                # ile çözülen yüzeyi derleyicinin compile'ı KABUL EDEN ilk
                # şablon seçilir (deterministik). Hiçbiri kabul edilmezse
                # mevcut davranış (ilk şablon) — fail-open değil, kayıt İLAN'lı.
                for aday in adaylar:
                    m_stem, r_affix = PhonologyEngine.resolve_affix(
                        stem_for_phonology, aday["template"], current_attrs
                    )
                    if (is_placeholder or is_proper_noun) and idx == 1:
                        aday_yuzey = f"{root_lemma}'{r_affix}"
                    else:
                        aday_yuzey = m_stem + r_affix
                    try:
                        aday_paket = self.compiler.compile(aday_yuzey)
                    except Exception:
                        continue
                    if aday_paket["analyses"]:
                        on_cozum = (aday, m_stem, r_affix)
                        break
                info = on_cozum[0] if on_cozum else adaylar[0]

            template = info["template"]

            # Special case for past tense person endings
            prev_affix = clean_tags[idx - 1]
            if affix_id.startswith("PERSON_") and prev_affix in ("TENSE_PAST", "COPULA_PAST"):
                if affix_id in PAST_PERSON_TEMPLATES:
                    template = PAST_PERSON_TEMPLATES[affix_id]

            if on_cozum is not None:
                # Çözümle-seç yolunda resolve_affix ZATEN koşuldu — tekrarlanmaz.
                mutated_stem, resolved_affix = on_cozum[1], on_cozum[2]
            else:
                mutated_stem, resolved_affix = PhonologyEngine.resolve_affix(
                    stem_for_phonology,
                    template,
                    current_attrs
                )
            
            if (is_placeholder or is_proper_noun) and idx == 1:
                current_surface = f"{root_lemma}'{resolved_affix}"
            else:
                current_surface = mutated_stem + resolved_affix
            current_attrs = info.get("attributes", "-")
            
        return current_surface

    def decompile_sentence(self, crystal_tags_str: str, capitalize: bool = False) -> str:
        """
        Decompiles a full sequence represented as a space-separated string of morpheme tags.
        Identifies word boundaries, metadata keywords, and decompiles each word individually.
        """
        if not crystal_tags_str:
            return ""
            
        tokens = crystal_tags_str.split()
        reconstructed_words = []
        current_word_tags = []

        in_entity = False
        entity_chars = []
        all_caps = False
        cap_next = False

        for tag_i, tag in enumerate(tokens):
            if tag == "<ENT>":
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                in_entity = True
                entity_chars = []
                all_caps = False
                cap_next = False
                continue

            if in_entity:
                if tag == "</ENT>":
                    surface_word = "".join(entity_chars)
                    current_word_tags.append(surface_word)
                    in_entity = False
                    continue
                if tag == "<ALL_CAPS>":
                    all_caps = True
                    continue
                if tag == "<CAP>":
                    cap_next = True
                    continue
                
                char_str = tag
                if all_caps:
                    char_str = char_str.upper()
                elif cap_next:
                    char_str = char_str.upper()
                    cap_next = False
                entity_chars.append(char_str)
                continue

            if tag == "<PROPER_NOUN>":
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                current_word_tags.append("[Özel İsim]")
                continue

            if tag == "<UNK>":
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                current_word_tags.append("[?]")
                continue

            if tag == "<NUMBER>":
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                current_word_tags.append("[sayı]")
                continue

            # Skip control tokens
            if tag.startswith("<") and tag.endswith(">"):
                continue

            # Digit handling
            if tag.isdigit():
                if current_word_tags:
                    if current_word_tags[0].isdigit():
                        current_word_tags[0] += tag
                        continue
                    else:
                        reconstructed_words.append(self.decompile_tags(current_word_tags))
                        current_word_tags = []
                if reconstructed_words and (reconstructed_words[-1][-1].isdigit() or reconstructed_words[-1].endswith("-")):
                    reconstructed_words[-1] = reconstructed_words[-1] + tag
                else:
                    reconstructed_words.append(tag)
                continue

            # Punctuation handling
            if tag in {".", ",", "?", "!", ":", ";", ")"}:
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                if reconstructed_words:
                    reconstructed_words[-1] = reconstructed_words[-1] + tag
                else:
                    reconstructed_words.append(tag)
                continue

            if tag == "-":
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                if reconstructed_words:
                    reconstructed_words[-1] = reconstructed_words[-1] + "-"
                else:
                    reconstructed_words.append("-")
                continue

            if tag == "(":
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                reconstructed_words.append(tag)
                continue
                
            is_suf = self.is_suffix(tag)
            is_meta = (tag.lower() in META_TAGS) and not is_suf

            if is_meta:
                # T-0103: 'ara' hem meta marker hem GERÇEK lexicon köküdür.
                # Ardından suffix gelen meta-kök gerçek kelimedir
                # ('ara POSS_3SG CASE_LOC_N' → 'arasında') — normal kök
                # dalına düşer; değilse eski davranış (tek başına marker).
                sonraki = tokens[tag_i + 1] if tag_i + 1 < len(tokens) else ""
                if not self.is_suffix(sonraki):
                    if current_word_tags:
                        reconstructed_words.append(self.decompile_tags(current_word_tags))
                        current_word_tags = []
                    reconstructed_words.append(tag)
                    continue
                
            if not is_suf:
                if current_word_tags:
                    reconstructed_words.append(self.decompile_tags(current_word_tags))
                    current_word_tags = []
                current_word_tags.append(tag)
            else:
                if not current_word_tags:
                    # Suffix appeared without root; emit tag directly
                    reconstructed_words.append(tag)
                else:
                    current_word_tags.append(tag)
                    
        if current_word_tags:
            reconstructed_words.append(self.decompile_tags(current_word_tags))
            
        text = " ".join(reconstructed_words).strip()
        if text and capitalize:
            # Capitalize first character respecting Turkish dotted/undotted I
            first_ch = text[0]
            if first_ch == 'i':
                first_ch_upper = 'İ'
            elif first_ch == 'ı':
                first_ch_upper = 'I'
            else:
                first_ch_upper = first_ch.upper()
            text = first_ch_upper + text[1:]
            
        return text
