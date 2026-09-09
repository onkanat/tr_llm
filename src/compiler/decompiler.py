#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from src.compiler.core import CrystalCompiler
from src.compiler.phonology import PhonologyEngine

META_TAGS = {
    "root", "case", "ten", "plural", "evet", "hayır", "var", "yok",
    "belge:", "sorgu:", "tanım:", "cevap:",
    "<ARA>", "</ARA>", "<BELGE>", "</BELGE>", "ara", "belge"
}

SUFFIX_PREFIXES = (
    "TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_", "DERIV_"
)
SUFFIX_EXACTS = {"PLURAL", "POTENTIAL", "NEG", "IMPOTENTIAL_NEG"}

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
    "IMPOTENTIAL_NEG": {"template": "(y)AmA", "attributes": "-"}
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
                
        if not root_entry:
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
                    
            mutated_stem, resolved_affix = PhonologyEngine.resolve_affix(
                current_surface,
                template,
                current_attrs
            )
            
            current_surface = mutated_stem + resolved_affix
            current_attrs = info.get("attributes", "-")
            
        return current_surface

    def decompile_sentence(self, crystal_tags_str: str) -> str:
        """
        Decompiles a full sequence represented as a space-separated string of morpheme tags.
        Identifies word boundaries, metadata keywords, and decompiles each word individually.
        """
        if not crystal_tags_str:
            return ""
            
        tokens = crystal_tags_str.split()
        reconstructed_words = []
        current_word_tags = []
        
        for tag in tokens:
            # Skip control tokens
            if tag.startswith("<") and tag.endswith(">"):
                continue
                
            is_suf = self.is_suffix(tag)
            is_meta = tag.lower() in META_TAGS
            
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
            
        return " ".join(reconstructed_words)
