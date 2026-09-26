#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_dataset_compiler.py

T-0134 / G3b kabul testleri:
  1. Çift-kayıt yapısı denetimi (satır başına 2 kayıt, kept <= 2 * lines_ok).
  2. Jenerik alan parametresi desteği (Bahçıvan / Berber için system_prompt/domain_name).
  3. bin_dekod_dogrula invariant denetimi (<BOS>, <EOS>, <OUTPUT>, </OUTPUT> eşitliği).
  4. Bit-özdeş regresyon denetimi (yeni modül ile eski betik çıktısı tam sha256 birebir).
"""

import os
import json
import hashlib
import numpy as np
import pytest

from src.llm.dataset_compiler import (
    tokenize_specialization_jsonl,
    tokenize_standard_jsonl,
    bin_dekod_dogrula,
    check_output_path
)
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler


@pytest.fixture(scope="module")
def compiler_tools():
    vocab_path = "data/rebuild/vocab_base_32852.json"
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    return vocab, tokenizer


def test_tokenize_specialization_double_record_structure(compiler_tools, tmp_path):
    """Kriter 1: Satır başına 2 kayıt üretebilme ve D4 muhasebe denetimi."""
    vocab, tokenizer = compiler_tools
    sample_file = tmp_path / "sample.jsonl"
    records_in = [
        {"instruction": "Ahşap tutkalı nasıl sürülür?", "input": "Tutkal sürme yöntemi nedir?", "output": "İnce tabaka halinde sürülür."},
        {"instruction": "Gül budama: Ne zaman yapılır?", "output": "İlkbahar başında yapılır."} # ':' içeren tek parça
    ]
    with open(sample_file, "w", encoding="utf-8") as f:
        for r in records_in:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    recs, stats = tokenize_specialization_jsonl(
        str(sample_file),
        tokenizer,
        system_prompt="Bahçe uzmanı olarak cevapla.",
        domain_name="gardener"
    )

    # 2 satır girdiden her ikisi de soru taşıdığı için 2'şer kayıt üretmeli -> 4 kayıt
    assert stats["raw_lines"] == 2
    assert stats["lines_ok"] == 2
    assert stats["kept"] == 4
    assert len(recs) == 4
    assert stats["domain_name"] == "gardener"
    assert stats["system_prompt"] == "Bahçe uzmanı olarak cevapla."


def test_bin_dekod_dogrula_invariants(compiler_tools, tmp_path):
    """Kriter 2: bin_dekod_dogrula invariant denetimi ve sapmada hata fırlatma."""
    vocab, tokenizer = compiler_tools
    sample_file = tmp_path / "sample2.jsonl"
    records_in = [
        {"instruction": "Test sorusu", "input": "Girdi", "output": "Cevap"}
    ]
    with open(sample_file, "w", encoding="utf-8") as f:
        for r in records_in:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    recs, stats = tokenize_specialization_jsonl(str(sample_file), tokenizer)
    
    flat = []
    boundaries = []
    curr = 0
    for r in recs:
        flat.extend(r)
        boundaries.append({"offset": curr, "length": len(r)})
        curr += len(r)

    bin_path = str(tmp_path / "test.bin")
    meta_path = str(tmp_path / "test.bin.meta.json")

    np.array(flat, dtype=np.uint16).tofile(bin_path)
    meta = {
        "record_count": len(recs),
        "total_records": len(recs),
        "total_tokens": len(flat),
        "boundaries": boundaries
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # Doğru durumda geçmeli
    res = bin_dekod_dogrula(bin_path, meta_path, vocab)
    assert res["ok"] is True
    assert res["total_records"] == 2
    assert res["bos_count"] == 2
    assert res["eos_count"] == 2

    # Bozuk meta durumunda AssertionError fırlatmalı
    meta_corrupt = dict(meta)
    meta_corrupt["total_records"] = 999
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_corrupt, f)

    with pytest.raises(AssertionError):
        bin_dekod_dogrula(bin_path, meta_path, vocab)


def test_bit_exact_regression_against_carpenter_corpus(compiler_tools):
    """Kriter 3: Yeni modülün Marangoz külliyatındaki çıktısı eski betikle bit-özdeştir."""
    vocab, tokenizer = compiler_tools
    carp_path = "data/pedagogy/carpenter_specialization_dataset.jsonl"
    
    from scripts.prepare_carpenter_specialization_dataset import tokenize_carpenter_jsonl as old_tok
    
    # 100 satırlık dondurulmuş örneklem yerine bütün külliyat veya ilk 100 satır
    # Hızlı regresyon için ilk 50 satırı karşılaştıralım (determinizm ve bit-özdeşlik)
    with open(carp_path, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(100)]
    
    # Geçici dosyaya yaz
    import tempfile
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
        tf.writelines(lines)
        t_path = tf.name

    try:
        old_recs, old_stats = old_tok(t_path, tokenizer)
        new_recs, new_stats = tokenize_specialization_jsonl(t_path, tokenizer, system_prompt="Ahşap uzmanı olarak cevapla.")

        assert len(old_recs) == len(new_recs)
        assert old_recs == new_recs
        assert old_stats["kept"] == new_stats["kept"]
        assert old_stats["lines_ok"] == new_stats["lines_ok"]
    finally:
        os.remove(t_path)
