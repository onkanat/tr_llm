#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2.1 — et/fehmet fiillerine VOICING ('edeceğim' ailesi).

Veri düzeltmesinin TDD kanıtı: data/lexicon/roots.tsv içindeki
`et VERB` (satır 14676) ve `fehmet VERB` (satır 15417) satırlarına
VOICING niteliği eklenmeden 'edeceğim' / 'fehmedilmez' çözülmez —
find_stems'nin voicing-bakışı (lexicon.py reverse_voicing_map) yalnız
VOICING nitelikli köklere açılır. `et NOUN` (satır 14675, 'yiyecek')
DOKUNULMAZ: 'etmek/etti' üzerinden kökün fiil olduğunu doğrular.
"""

import os

from src.llm.tokenizer import Vocabulary, KristalTokenizer
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

VOCAB_PATH = os.path.join('data', 'rebuild', 'vocab_anka_r1_33114.json')
LEXICON_PATH = os.path.join('data', 'lexicon', 'roots.tsv')


def _build():
    """Skill'in kanonik fixture'ı: üretim hattı (agent_gateway) gibi."""
    lex = LexiconManager()
    lex.load_from_tsv(LEXICON_PATH)
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)
    compiler = CrystalCompiler(lex, build_default_graph(), vocab=vocab)
    tokenizer = KristalTokenizer(compiler, vocab)
    return compiler, tokenizer


compiler, tokenizer = _build()


def _morf(word):
    """Kelimenin ÇÖZÜM UZAYINDAKİ morf id'leri (tüm analizlerin birleşimi).

    'etti' ölçümü (T2.1 RED fazı): NOUN `et` (satır 14675, dokunulmaz) ile
    VERB `et` 2-morfem BERABERE kalır ve stable sıra NOUN copula analizini
    ([et, COPULA_PAST]) en iyi analizin ÖNÜNE koyar — bu ÖNCEDEN VAR OLAN
    davranıştır, VOICING eklemek değiştirmez (TSV satır sırası korunur).
    Regresyon guard'ının işi VERB analizinin ([et, TENSE_PAST]) çözüm uzayı
    içinde VAR OLMAKTA kalmasıdır; en iyi-analiz seçimi test 1'de
    (`test_edecegim_ayrisir`, küme-eşitliği) ayrıca sabitlenir.
    """
    r = compiler.compile(word)
    ids = set()
    for a in r.get("analyses") or []:
        ids.update(m["id"] for m in a["morphemes"])
    return ids


def test_edecegim_ayrisir():
    r = compiler.compile("edeceğim")
    m = set(r.get("token_vector") or [])
    assert m == {"et", "TENSE_FUT", "PERSON_1SG"}


def test_etmek_etti_regresyon():
    assert "et" in _morf("etmek") and "INF_mAk" in _morf("etmek")
    assert "et" in _morf("etti") and "TENSE_PAST" in _morf("etti")


def test_fehmedilmez_ayrisir():
    m = _morf("fehmedilmez")
    assert {"fehmet", "VOICE_PASS_Il", "TENSE_AORIST_NEG"} <= set(m)