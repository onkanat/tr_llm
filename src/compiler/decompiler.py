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
        
        # Build mapping from affix semantic ID/tag to surface form templates & attributes
        self.affix_info: Dict[str, Dict[str, Any]] = {}
        for state, transitions in compiler.graph.transitions.items():
            for trans in transitions:
                if trans.affix_id not in self.affix_info:
                    self.affix_info[trans.affix_id] = {
                        "template": trans.affix_template,
                        "attributes": trans.attributes
                    }

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
        if root_lemma.lower() in META_TAGS:
            return " ".join(clean_tags)
            
        stems = self.compiler.lexicon.find_stems(root_lemma)
        root_entry = None
        for stem_prefix, entry in stems:
            if entry['lemma'] == root_lemma:
                root_entry = entry
                break
                
        is_placeholder = root_lemma in ("[Özel İsim]", "[sayı]", "[?]") or root_lemma.isdigit()
        is_proper_noun = bool(root_lemma and (root_lemma[0].isupper() or root_lemma.isupper()) and not root_lemma.startswith("["))
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
            info = self.affix_info.get(affix_id)
            if info is None:
                info = COMMON_FALLBACKS.get(affix_id, {"template": "", "attributes": "-"})
                
            template = info["template"]
            
            # Special case for past tense person endings
            prev_affix = clean_tags[idx - 1]
            if affix_id.startswith("PERSON_") and prev_affix in ("TENSE_PAST", "COPULA_PAST"):
                if affix_id in PAST_PERSON_TEMPLATES:
                    template = PAST_PERSON_TEMPLATES[affix_id]
                    
            stem_for_phonology = "isim" if is_placeholder and idx == 1 else current_surface
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
        
        for tag in tokens:
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
