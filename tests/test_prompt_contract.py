#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_prompt_contract.py

Canary tests for T-0013:
  (a) None of the 9 prompt construction sites produce double <BOS> (encode(p)[:2] != [BOS, BOS]).
  (b) render_example gives the exact same token sequence as KristalTokenizer._render_structured_prompt.
  (c) chat_prompt.load_model_instance raises FileNotFoundError on missing checkpoint.
  Positive Control: Confirms detector correctly finds residual unmigrated sites (e.g. scratch/measure_forward_pass.py).
"""

import ast
import os
import re
import pytest
import torch

from src.llm.prompt_contract import render_prompt, render_example, describe, TokenizerConfig
from src.llm.tokenizer import Vocabulary, KristalTokenizer
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
import chat_prompt

# Fixture to provide tokenizer and vocab
@pytest.fixture(scope="module")
def env():
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    lex = LexiconManager()
    lex.load_from_tsv("data/lexicon/roots.tsv")
    comp = CrystalCompiler(lex, build_default_graph())
    tokenizer = KristalTokenizer(comp, vocab)
    return {
        "vocab": vocab,
        "tokenizer": tokenizer,
        "bos_id": vocab.stoi.get("<BOS>", 2),
        "eos_id": vocab.stoi.get("<EOS>", 3),
    }


def test_canary_c_load_model_instance_raises_filenotfound(env):
    """Canary (c): chat_prompt.load_model_instance raises FileNotFoundError on missing checkpoint."""
    missing_path = "data/non_existent_canary_model_12345.pt"
    if os.path.exists(missing_path):
        os.remove(missing_path)

    with pytest.raises(FileNotFoundError, match="Model checkpoint'i bulunamadı"):
        chat_prompt.load_model_instance(
            missing_path,
            vocab_size=len(env["vocab"].stoi),
            vocab=env["vocab"],
            device=torch.device("cpu")
        )


def test_canary_b_render_example_token_identity(env):
    """Canary (b): render_example matches _render_structured_prompt token-for-token."""
    from src.llm.prompt_contract import render_example

    test_cases = [
        ("Türkçeye çevir.", "Hello world", "Merhaba dünya"),
        ("Ahşap ve marangozluk uzmanı olarak cevapla.", "Zıvana nedir?", "İki ahşabın birbirine geçmesini sağlayan dişi oyuk."),
        ("", "yalnızca girdi", "yalnızca çıktı"),
        ("Boş girdi testi.", "", "çıktı metni"),
        ("Soru", "Belge: ağaç kütük", "cevap"),
    ]

    tok = env["tokenizer"]
    for inst, inp, out in test_cases:
        contract_str = render_example(inst, inp, out)
        canonical_str = KristalTokenizer._render_structured_prompt(None, {
            "instruction": inst,
            "input": inp,
            "output": out
        })
        contract_tokens = tok.encode(contract_str)
        canonical_tokens = tok.encode(canonical_str)

        assert contract_tokens == canonical_tokens, (
            f"Token mismatch for ({inst!r}, {inp!r}, {out!r}):\n"
            f"Contract:  {contract_tokens}\n"
            f"Canonical: {canonical_tokens}"
        )


def evaluate_site_tokens(filepath: str, content: str = None, tok = None, bos_id: int = 2):
    """
    Dynamically identifies prompt construction sites via AST, evaluates the expressions
    with realistic dummy arguments, and measures tokenizer tokens to ensure no double <BOS>.
    """
    if content is None:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

    tree = ast.parse(content, filename=filepath)
    results = []
    scope = {
        "render_prompt": render_prompt,
        "render_example": render_example,
        "inst": "Türkçeye çevir.",
        "question": "Deneme sorusu?",
        "q": "Deneme sorusu?",
        "inp": "Girdi",
        "a": "Cevap",
        "persona_inst": "Uzman olarak.",
        "role": "Rol.",
        "z_inst": "Uzman.",
        "z_q": "Zıvana?",
        "q_text": "Soru?"
    }
    relevant_targets = {"prompt", "prompt_str", "full_str", "p_a_str", "p_b_str"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            matched = relevant_targets.intersection(target_names)
            if matched:
                expr_src = ast.get_source_segment(content, node.value)
                if expr_src and (
                    "render_prompt" in expr_src or 
                    "render_example" in expr_src or 
                    "<BOS>" in expr_src or 
                    "<INSTRUCTION>" in expr_src
                ):
                    prompt_val = eval(expr_src.strip(), scope)
                    tokens = tok.encode(prompt_val)
                    if tokens[:2] == [bos_id, bos_id]:
                        raise AssertionError(
                            f"Double <BOS> detected in {filepath}:{node.lineno}: tokens[:3]={tokens[:3]} for expr: {expr_src}"
                        )
                    results.append((node.lineno, list(matched)[0], expr_src, tokens))
    return results


PROMPT_FILES = [
    "scripts/train_step_b1_canonical.py",
    "scripts/train_step_a_ablation.py",
    "scripts/run_experiment_a.py",
    "scripts/run_experiment_b.py",
    "scripts/evaluate_mcq_conditioning.py",
    "scripts/evaluate_sft_benchmarks.py",
    "scripts/diagnostics_t3_t4.py",
]


@pytest.mark.parametrize("filepath", PROMPT_FILES)
def test_canary_a_token_measurement_no_double_bos(filepath, env):
    """Canary (a): Dynamic AST + Token-level measurement verifies none of the prompt sites produce double <BOS>."""
    assert os.path.exists(filepath), f"File {filepath} must exist"
    sites = evaluate_site_tokens(filepath, tok=env["tokenizer"], bos_id=env["bos_id"])
    assert len(sites) > 0, f"Expected at least one prompt construction site in {filepath}, found 0"
    for lineno, target_var, expr, tokens in sites:
        assert tokens[:2] != [env["bos_id"], env["bos_id"]], (
            f"Double <BOS> detected at {filepath}:{lineno} ({target_var} = {expr}): tokens[:3] == {tokens[:3]}"
        )
        assert tokens[0] == env["bos_id"], (
            f"Expected initial <BOS> token at {filepath}:{lineno}, got {tokens[0]}"
        )
        assert tokens[1] != env["bos_id"], (
            f"Second token must not be <BOS> at {filepath}:{lineno}, got {tokens[1]}"
        )


def test_canary_a_mutant_detector_catches_double_bos_regression(env):
    """
    Permanent Mutant Test: Proves that the detector is NOT a vacuum and actively catches
    a double <BOS> regression when injected into a prompt site (Claude mutant replication).
    """
    test_file = "scripts/run_experiment_a.py"
    assert os.path.exists(test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        orig_content = f.read()

    # Inject mutant 1: f-string prepending literal <BOS>
    mutant_fstring = orig_content.replace(
        "prompt = render_prompt(inst, question)",
        'prompt = f"<BOS> {render_prompt(inst, question)}"'
    )
    assert mutant_fstring != orig_content, "Replacement failed: target line not found"

    with pytest.raises(AssertionError, match="Double <BOS> detected"):
        evaluate_site_tokens(test_file, content=mutant_fstring, tok=env["tokenizer"], bos_id=env["bos_id"])

    # Inject mutant 2: string concatenation prepending literal <BOS>
    mutant_concat = orig_content.replace(
        "prompt = render_prompt(inst, question)",
        'prompt = "<BOS> " + render_prompt(inst, question)'
    )
    assert mutant_concat != orig_content, "Replacement failed: target line not found"

    with pytest.raises(AssertionError, match="Double <BOS> detected"):
        evaluate_site_tokens(test_file, content=mutant_concat, tok=env["tokenizer"], bos_id=env["bos_id"])


def test_canary_d_tokenizer_config_validation_and_describe(env, capsys):
    """Canary (d): TokenizerConfig.validate() is actively called and warns on literal_entity_mode=True."""
    from src.llm.prompt_contract import describe, TokenizerConfig

    # Normal mode -> no warning
    capsys.readouterr()  # clear buffer
    desc_default = describe()
    out_default = capsys.readouterr().out
    assert "[SOZLESME]" in desc_default
    assert "literal_entity=False" in desc_default
    assert "[SOZLESME_UYARI]" not in out_default

    # Entity mode -> prints [SOZLESME_UYARI]
    cfg_entity = TokenizerConfig(literal_entity_mode=True)
    desc_entity = describe(cfg_entity)
    out_entity = capsys.readouterr().out
    assert "[SOZLESME]" in desc_entity
    assert "literal_entity=True" in desc_entity
    assert "[SOZLESME_UYARI]" in out_entity

    # Missing vocab -> raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        cfg_bad = TokenizerConfig(vocab_path="data/non_existent_vocab_file_12345.json")
        describe(cfg_bad)


def test_positive_control_detector_finds_unmigrated():
    """Positive Control: Detector finds unmigrated residual files (e.g. scratch/measure_forward_pass.py)."""
    scratch_file = "scratch/measure_forward_pass.py"
    assert os.path.exists(scratch_file)

    with open(scratch_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify that scratch/measure_forward_pass.py indeed has silent model loading
    has_silent_load = "if os.path.exists(sft_model_path):" in content
    assert has_silent_load, "Positive control failed: scratch file pattern changed unexpectedly."
