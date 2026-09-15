#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pytest
from src.llm.tokenizer import Vocabulary, KristalTokenizer
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

def test_morphology_regression_integrity():
    json_path = os.path.join(os.path.dirname(__file__), 'morphology_regression_100.json')
    assert os.path.exists(json_path), 'morphology_regression_100.json bulunamadı!'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert len(data) == 100, f'Beklenen 100 örnek, bulunan: {len(data)}'
    
    tasks = set()
    for item in data:
        assert 'instruction' in item and item['instruction']
        assert 'input' in item and item['input']
        assert 'output' in item and item['output']
        tasks.add(item['instruction'])
        
    assert len(tasks) == 4, f'4 temel görev bekleniyor, bulunan: {tasks}'

def test_morphology_regression_tokenization():
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    lex = LexiconManager()
    lex.load_from_tsv('data/lexicon/roots.tsv')
    comp = CrystalCompiler(lex, build_default_graph())
    tok = KristalTokenizer(comp, vocab)
    
    json_path = os.path.join(os.path.dirname(__file__), 'morphology_regression_100.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data[:10]:
        enc_in = tok.encode(item['input'])
        enc_out = tok.encode(item['output'])
        assert len(enc_in) >= 2
        assert len(enc_out) >= 2
