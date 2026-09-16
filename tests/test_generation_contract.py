#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_generation_contract.py

K3 Kanarya Testi: Üretim kopyalarındaki yasaklı etiketler (masked_tags) sözleşmesini denetler.
AST ayrıştırması kullanarak kod yürütmeden 4 üretim dosyasını inceler:
1. scripts/run_experiment_a.py
2. scripts/run_experiment_b.py
3. scripts/run_experiment_c.py
4. scripts/evaluate_sft_benchmarks.py

Kural: 4 dosyadaki masked_tags listeleri BİREBİR aynı olmalıdır.
Parametrelendirme: GENERATION_CONTRACT_FILES ortam değişkeni tanımlıysa o dosyaları inceler;
böylece gerçek dosyalara dokunmadan mutant pozitif kontrolü çalıştırılabilir.
"""

import os
import ast
import json
import hashlib
from typing import List, Dict, Tuple
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CANONICAL_FILES = [
    "scripts/run_experiment_a.py",
    "scripts/run_experiment_b.py",
    "scripts/run_experiment_c.py",
    "scripts/evaluate_sft_benchmarks.py",
]

EXPECTED_CANONICAL_TAGS = [
    "<PAD>", "<BOS>", "<INSTRUCTION>", "</INSTRUCTION>",
    "<INPUT>", "</INPUT>", "<OUTPUT>", "<PROPER_NOUN>", "<UNK>", "<NUMBER>"
]


def extract_masked_tags(filepath: str) -> List[str]:
    """AST kullanarak bir Python dosyasındaki masked_tags atamasını çıkarır."""
    abs_path = filepath if os.path.isabs(filepath) else os.path.join(REPO_ROOT, filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Hedef dosya bulunamadı: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=abs_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "masked_tags":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        tags = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                tags.append(elt.value)
                        return tags

    raise ValueError(f"masked_tags ataması AST içinde bulunamadı: {filepath}")


def assert_masked_tags_identical(filepaths: List[str]) -> Tuple[bool, Dict[str, List[str]]]:
    """Verilen dosya listesindeki masked_tags tanımlarının özdeş olduğunu doğrular."""
    extracted = {}
    hashes = {}
    for p in filepaths:
        tags = extract_masked_tags(p)
        extracted[p] = tags
        tag_hash = hashlib.sha256(json.dumps(tags).encode("utf-8")).hexdigest()
        hashes[p] = tag_hash

    unique_hashes = set(hashes.values())
    if len(unique_hashes) != 1:
        details = "\n".join([f"  {p}: {extracted[p]} (hash: {hashes[p][:12]})" for p in filepaths])
        raise AssertionError(f"masked_tags listeleri AYRIŞTI! Dosyalar arasında fark var:\n{details}")

    return True, extracted


def test_canonical_masked_tags_contract():
    """
    4 kanonik üretim dosyasının masked_tags sözleşmesini doğrular.
    GENERATION_CONTRACT_FILES ortam değişkeni varsa o dosya listesini kullanır.
    """
    env_files = os.environ.get("GENERATION_CONTRACT_FILES")
    if env_files:
        files_to_check = [f.strip() for f in env_files.split(",") if f.strip()]
    else:
        files_to_check = CANONICAL_FILES

    ok, extracted = assert_masked_tags_identical(files_to_check)
    assert ok is True

    # Standart kanonik dosyalarda beklenen etiketler doğrulanır
    if not env_files:
        first_file = CANONICAL_FILES[0]
        assert extracted[first_file] == EXPECTED_CANONICAL_TAGS
        assert "</OUTPUT>" not in extracted[first_file], "</OUTPUT> mevcut kanonik sözleşmede serbest bırakılmıştır"


def test_contract_detects_divergence_mutant(tmp_path):
    """
    Mutant pozitif kontrolü: Gerçek dosyaları değiştirmeden geçici bir mutant dosya
    üretir ve ayrışmanın AssertionError ile yakalandığını kanıtlar.
    """
    # 3 kanonik dosya + 1 geçici mutant dosya
    clean_file = os.path.join(REPO_ROOT, CANONICAL_FILES[0])
    with open(clean_file, "r", encoding="utf-8") as f:
        clean_content = f.read()

    # Mutant: masked_tags içine '</OUTPUT>' ekle
    mutant_content = clean_content.replace(
        '"<NUMBER>"',
        '"<NUMBER>", "</OUTPUT>"'
    )
    assert '"</OUTPUT>"' in mutant_content, "Mutasyon uygulanamadı"

    mutant_file = str(tmp_path / "mutant_experiment.py")
    with open(mutant_file, "w", encoding="utf-8") as f:
        f.write(mutant_content)

    test_files = [
        CANONICAL_FILES[0],
        CANONICAL_FILES[1],
        CANONICAL_FILES[2],
        mutant_file
    ]

    with pytest.raises(AssertionError, match="masked_tags listeleri AYRIŞTI"):
        assert_masked_tags_identical(test_files)
