#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary


class TestTwoLayerTokenizer(unittest.TestCase):
    def setUp(self):
        self.lex = LexiconManager()
        roots_path = "data/lexicon/roots.tsv"
        if os.path.exists(roots_path):
            self.lex.load_from_tsv(roots_path)
        else:
            raise FileNotFoundError(f"Leksikon kök dosyası bulunamadı: {roots_path}")
        self.graph = build_default_graph()
        self.compiler = CrystalCompiler(self.lex, self.graph)
        self.vocab = Vocabulary()
        self.vocab.load("data/vocab.json", freeze=True)
        self.tok_morpheme = KristalTokenizer(self.compiler, self.vocab, literal_entity_mode=False)
        self.tok_hybrid = KristalTokenizer(self.compiler, self.vocab, literal_entity_mode=True)
        self.decompiler = MorphemeDecompiler(self.compiler, self.vocab)

    def test_morpheme_mode_uses_proper_noun(self):
        text = "Şuppiluliuma Kargamış'a gitti."
        enc = self.tok_morpheme.encode(text)
        dec = self.tok_morpheme.decode(enc)
        self.assertIn("<PROPER_NOUN>", dec)
        self.assertNotIn("<ENT>", dec)

    def test_hybrid_mode_encodes_unseen_entity(self):
        text = "Şuppiluliuma Kargamış'a gitti."
        enc = self.tok_hybrid.encode(text)
        dec = self.tok_hybrid.decode(enc)
        self.assertIn("<ENT>", dec)
        self.assertIn("<CAP>", dec)
        self.assertIn("ş", dec)
        self.assertIn("u", dec)
        self.assertIn("p", dec)

    def test_all_caps_acronym_encoding(self):
        text = "TBMM ve NASA heyeti"
        enc = self.tok_hybrid.encode(text)
        dec = self.tok_hybrid.decode(enc)
        self.assertIn("<ALL_CAPS>", dec)
        self.assertIn("t", dec)
        self.assertIn("b", dec)
        self.assertIn("m", dec)

    def test_entity_decompile_roundtrip(self):
        text = "Şuppiluliuma'nın emriyle Kargamış'tan dönüldü."
        enc = self.tok_hybrid.encode(text)
        dec = self.tok_hybrid.decode(enc)
        decompiled = self.decompiler.decompile_sentence(dec)
        self.assertIn("Şuppiluliuma'nın", decompiled)
        self.assertIn("Kargamış'tan", decompiled)

    def test_acronym_decompile_roundtrip(self):
        text = "TBMM binası"
        enc = self.tok_hybrid.encode(text)
        dec = self.tok_hybrid.decode(enc)
        decompiled = self.decompiler.decompile_sentence(dec)
        self.assertIn("TBMM", decompiled)

    def test_canary_lexicon_sensitive_morphology(self):
        # Bu test sözlük yüklü olmadığında (<UNK>) düşer, sözlük yüklü olduğunda geçer.
        text = "dubaracılık"
        enc = self.tok_morpheme.encode(text)
        dec = self.tok_morpheme.decode(enc)
        self.assertIn("dubara", dec)
        self.assertIn("DERIV_CI", dec)
        self.assertIn("DERIV_lIk", dec)
        self.assertNotIn("<UNK>", dec)


if __name__ == "__main__":
    unittest.main()
