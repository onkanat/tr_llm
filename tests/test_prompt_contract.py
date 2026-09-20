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

# T-0088: GUNCEL kulliyat sozlugu. Test bu yolu ACIKCA gecirir; bayat varsayilan YOK.
# (env fixture'i hala `data/vocab.json` yukler; o ayri ve ACIK bir maddedir.)
GUNCEL_VOCAB = "data/rebuild/vocab_anka_r1_33114.json"

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
        "q_text": "Soru?",
        "inst_text": "Uzman olarak."
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
    "scripts/run_experiment_c.py",
]


@pytest.mark.parametrize("filepath", PROMPT_FILES)
def test_canary_a_token_measurement_no_double_bos(filepath, env):
    """Canary (a): Dynamic AST + Token-level measurement verifies none of the prompt sites produce double <BOS>."""
    assert os.path.exists(filepath), f"File {filepath} must exist"
    sites = evaluate_site_tokens(filepath, tok=env["tokenizer"], bos_id=env["bos_id"])
    assert len(sites) > 0, f"Expected at least one prompt construction site in {filepath}, found 0"
    if filepath == "scripts/run_experiment_c.py":
        assert len(sites) == 2, f"Expected exactly 2 prompt construction sites in {filepath}, found {len(sites)}"
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


def test_canary_a_run_experiment_c_mutant_detector(env):
    """
    Mutant Test for run_experiment_c.py:
    1. Proves that double <BOS> regression is caught if literal <BOS> is reintroduced.
    2. Proves that site loss is caught if one of the 2 sites disappears.
    """
    test_file = "scripts/run_experiment_c.py"
    assert os.path.exists(test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        orig_content = f.read()

    # Normal state: exactly 2 sites found
    normal_sites = evaluate_site_tokens(test_file, content=orig_content, tok=env["tokenizer"], bos_id=env["bos_id"])
    assert len(normal_sites) == 2, f"Expected exactly 2 sites in normal run_experiment_c.py, got {len(normal_sites)}"

    # Mutant 1: double <BOS> injected into first site
    mutant_bos = orig_content.replace(
        'prompt = render_prompt(inst_text, "")',
        'prompt = f"<BOS> {render_prompt(inst_text, \'\')}"',
        1
    )
    assert mutant_bos != orig_content
    with pytest.raises(AssertionError, match="Double <BOS> detected"):
        evaluate_site_tokens(test_file, content=mutant_bos, tok=env["tokenizer"], bos_id=env["bos_id"])

    # Mutant 2: site loss (one site deleted/renamed)
    mutant_loss = orig_content.replace(
        'prompt = render_prompt(inst_text, "")',
        'ignored_prompt = "dummy"',
        1
    )
    assert mutant_loss != orig_content
    sites_after_loss = evaluate_site_tokens(test_file, content=mutant_loss, tok=env["tokenizer"], bos_id=env["bos_id"])
    assert len(sites_after_loss) == 1, f"Expected 1 site after removing one, got {len(sites_after_loss)}"


def test_canary_d_tokenizer_config_validation_and_describe(env, capsys):
    """Canary (d): TokenizerConfig.validate() is actively called and warns on literal_entity_mode=True."""
    from src.llm.prompt_contract import describe, TokenizerConfig

    # Normal mode -> no warning. T-0088: config artik ZORUNLU (bayat varsayilan KALDIRILDI).
    capsys.readouterr()  # clear buffer
    desc_default = describe(TokenizerConfig(vocab_path=GUNCEL_VOCAB))
    out_default = capsys.readouterr().out
    assert "[SOZLESME]" in desc_default
    assert "literal_entity=False" in desc_default
    assert "[SOZLESME_UYARI]" not in out_default

    # Entity mode -> prints [SOZLESME_UYARI]
    cfg_entity = TokenizerConfig(vocab_path=GUNCEL_VOCAB, literal_entity_mode=True)
    desc_entity = describe(cfg_entity)
    out_entity = capsys.readouterr().out
    assert "[SOZLESME]" in desc_entity
    assert "literal_entity=True" in desc_entity
    assert "[SOZLESME_UYARI]" in out_entity

    # Missing vocab -> raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        cfg_bad = TokenizerConfig(vocab_path="data/non_existent_vocab_file_12345.json")
        describe(cfg_bad)


def test_canary_e_tokenizer_config_fail_closed(env):
    """T-0088 fail-closed kanaryası: sözlük yolu VERİLMEZSE sessizce bayat varsayılana
    düşülmez. Düzeltmeden ÖNCE bu çağrı `data/vocab.json` (BAYAT, 31.357) kabul ederdi
    ve hiçbir uyarı vermezdi; kapı artık çağrı anında durur."""
    # 1) Argümansız TokenizerConfig geçersiz
    with pytest.raises(ValueError):
        TokenizerConfig().validate()

    # 2) Argümansız describe() durur (eskiden SESSİZCE bayat sözlüğe düşüyordu)
    with pytest.raises(ValueError):
        describe()

    # 3) POZİTİF KONTROL: geçerli yol verilince çalışır — kanarya kendi kendini
    #    çürütmesin (yoksa "her zaman duruyor" da kapıyı geçerdi).
    assert "[SOZLESME]" in describe(TokenizerConfig(vocab_path=GUNCEL_VOCAB))


def test_positive_control_detector_finds_unmigrated():
    """Positive Control: Detector finds unmigrated residual files (e.g. scratch/measure_forward_pass.py)."""
    scratch_file = "scratch/measure_forward_pass.py"
    assert os.path.exists(scratch_file)

    with open(scratch_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify that scratch/measure_forward_pass.py indeed has silent model loading
    has_silent_load = "if os.path.exists(sft_model_path):" in content
    assert has_silent_load, "Positive control failed: scratch file pattern changed unexpectedly."


def is_resize_state_dict_silent(fn_ast: ast.FunctionDef, source_code: str) -> bool:
    """
    Analyzes an AST FunctionDef of resize_state_dict.
    Uses structural analysis (F3):
    Returns True if the function modifies/writes weights (assignments, pad, copy_, slice, etc.)
    WITHOUT emitting any print, warning, logger alert, or raise.
    """
    has_warning_or_error = False
    writes_weights = False

    for node in ast.walk(fn_ast):
        if isinstance(node, ast.Call):
            func = node.func
            # Check for print(...)
            if isinstance(func, ast.Name) and func.id == "print":
                has_warning_or_error = True
            # Check for warnings.warn(...) or logger calls
            elif isinstance(func, ast.Attribute) and func.attr in ("warn", "warning", "error", "critical"):
                has_warning_or_error = True
            # Check for mutating tensor calls or pad
            elif isinstance(func, ast.Attribute) and func.attr in ("copy_", "pad", "resize_", "narrow", "slice"):
                writes_weights = True
            elif isinstance(func, ast.Name) and func.id in ("pad",):
                writes_weights = True
        elif isinstance(node, ast.Raise):
            has_warning_or_error = True
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            # Assignments such as state_dict[k] = ..., d[k][:min(...)] = ...
            writes_weights = True

    # If it writes weights but has no warning, print, or raise -> SILENT
    return writes_weights and not has_warning_or_error


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def audit_repo_resize_state_dict_definitions(repo_root=None, return_unparseable=False):
    """Scans all python files in repo for resize_state_dict definitions."""
    if repo_root is None:
        repo_root = _REPO_ROOT
    definitions = []
    unparseable = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "__pycache__", "build", "dist")]
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.normpath(os.path.join(root, file))
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == "resize_state_dict":
                        silent = is_resize_state_dict_silent(node, content)
                        definitions.append({
                            "path": path,
                            "line": node.lineno,
                            "is_silent": silent
                        })
            except Exception as e:
                # F7: DO NOT silently pass! Record unparseable file explicitly.
                unparseable.append({
                    "path": path,
                    "error": str(e)
                })

    if return_unparseable:
        return definitions, unparseable
    if unparseable:
        # F7: Unparseable files cause canary failure (RED) so broken files cannot hide silent definitions
        raise SyntaxError(f"audit_repo_resize_state_dict_definitions failed to parse {len(unparseable)} files: {unparseable}")
    return definitions


def test_canary_e_no_silent_resize_state_dict_in_repo():
    """Canary (e): Repo contains ZERO silent resize_state_dict definitions."""
    defs = audit_repo_resize_state_dict_definitions()
    silent_defs = [d for d in defs if d["is_silent"]]
    assert len(silent_defs) == 0, f"Found {len(silent_defs)} silent resize_state_dict definitions in repo: {silent_defs}"

    # Also verify that the 3 target files now import canonical resize_state_dict
    target_files = [
        "scripts/test_rag_interactive.py",
        "scripts/evaluate_chat.py",
        "scripts/evaluate_on_train_data.py"
    ]
    for tf in target_files:
        assert os.path.exists(tf)
        with open(tf, "r", encoding="utf-8") as f:
            content = f.read()
        assert "from src.llm.prompt_contract import resize_state_dict" in content, f"{tf} must import canonical resize_state_dict"
        assert not re.search(r"^\s*def resize_state_dict", content, re.MULTILINE), f"{tf} must not define local resize_state_dict"


def test_canary_e_silent_resize_mutant_detector():
    """
    Mutant Test for Canary (e) - F2:
    Proves that if an old silent resize_state_dict implementation is reintroduced
    into the repo filesystem (under scratch/), audit_repo_resize_state_dict_definitions()
    discovers the physical file, flags it as silent, and causes the canary assertion to fail (RED).
    Cleans up the mutant file in a finally block.
    """
    import time
    scratch_dir = os.path.join(_REPO_ROOT, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    mutant_path = os.path.join(scratch_dir, f"_mutant_silent_resize_{os.getpid()}_{time.time_ns()}.py")

    mutant_code = (
        "def resize_state_dict(model, old_state_dict):\n"
        "    new_state_dict = model.state_dict()\n"
        "    for k, v in old_state_dict.items():\n"
        "        if k in new_state_dict:\n"
        "            if v.shape != new_state_dict[k].shape:\n"
        "                if len(v.shape) == 2:\n"
        "                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])] = v[:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])]\n"
        "            else:\n"
        "                new_state_dict[k] = v\n"
        "    return new_state_dict\n"
    )

    try:
        with open(mutant_path, "w", encoding="utf-8") as f:
            f.write(mutant_code)

        # Actively run audit on repo containing the mutant file
        defs = audit_repo_resize_state_dict_definitions()
        mutant_defs = [d for d in defs if os.path.realpath(d["path"]) == os.path.realpath(mutant_path)]
        assert len(mutant_defs) == 1, f"Expected mutant file {mutant_path} to be discovered by audit, found: {mutant_defs}"
        assert mutant_defs[0]["is_silent"] is True, f"Expected mutant to be classified as silent, got {mutant_defs}"

        # Verify canary failure condition (must be RED when mutant is present)
        silent_in_repo = [d for d in defs if d["is_silent"]]
        assert len(silent_in_repo) > 0, "Canary should turn RED when silent mutant exists in repo!"
    finally:
        if os.path.exists(mutant_path):
            os.remove(mutant_path)

    # Post-cleanup: verify zero silent definitions remain
    defs_after = audit_repo_resize_state_dict_definitions()
    silent_after = [d for d in defs_after if d["is_silent"]]
    assert len(silent_after) == 0, f"Expected 0 silent definitions after cleanup, found {silent_after}"


def test_canary_e_structural_silent_detector_probes():
    """
    F3 Validation:
    Proves that is_resize_state_dict_silent classifies BOTH:
      (i) historical min() pattern
      (ii) .size() + pad pattern
    as silent (RED).
    """
    # Probe (i): historical min() pattern
    probe1_code = (
        "def resize_state_dict(model, old_state_dict):\n"
        "    new_state_dict = model.state_dict()\n"
        "    for k, v in old_state_dict.items():\n"
        "        if k in new_state_dict:\n"
        "            if v.shape != new_state_dict[k].shape:\n"
        "                if len(v.shape) == 2:\n"
        "                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])] = v[:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])]\n"
        "            else:\n"
        "                new_state_dict[k] = v\n"
        "    return new_state_dict\n"
    )
    tree1 = ast.parse(probe1_code)
    fn1 = next(n for n in ast.walk(tree1) if isinstance(n, ast.FunctionDef))
    assert is_resize_state_dict_silent(fn1, probe1_code) is True, "Probe 1 (min() pattern) must be classified as silent"

    # Probe (ii): .size() + pad pattern (lacks min( or shape tokens)
    probe2_code = (
        "def resize_state_dict(model, old_state_dict):\n"
        "    import torch.nn.functional as F\n"
        "    new_state_dict = model.state_dict()\n"
        "    for k, v in old_state_dict.items():\n"
        "        if k in new_state_dict:\n"
        "            diff = new_state_dict[k].size(0) - v.size(0)\n"
        "            new_state_dict[k] = F.pad(v, (0, 0, 0, diff))\n"
        "    return new_state_dict\n"
    )
    tree2 = ast.parse(probe2_code)
    fn2 = next(n for n in ast.walk(tree2) if isinstance(n, ast.FunctionDef))
    assert is_resize_state_dict_silent(fn2, probe2_code) is True, "Probe 2 (.size() + pad pattern) must be classified as silent"


def test_canary_e_unparseable_probe_fails_canary():
    """
    F7 Validation:
    Proves that a probe file containing a syntax error and a silent resize_state_dict
    is NOT silently passed; it causes audit_repo_resize_state_dict_definitions to flag it
    as unparseable and raise SyntaxError (RED).
    """
    import time
    scratch_dir = os.path.join(_REPO_ROOT, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    probe_path = os.path.join(scratch_dir, f"_unparseable_probe_{os.getpid()}_{time.time_ns()}.py")

    broken_code = (
        "def resize_state_dict(model, old_state_dict):\n"
        "    new = model.state_dict()\n"
        "    new['w'] = old_state_dict['w'][:min(10, 20)]\n"
        "    return new\n"
        "def intentional_syntax_error(:\n"
    )

    try:
        with open(probe_path, "w", encoding="utf-8") as f:
            f.write(broken_code)

        # Calling audit without return_unparseable must raise SyntaxError (RED)
        with pytest.raises(SyntaxError, match="failed to parse"):
            audit_repo_resize_state_dict_definitions()

        # Calling audit with return_unparseable returns the unparseable file in the report
        defs, unparseable = audit_repo_resize_state_dict_definitions(return_unparseable=True)
        unparseable_paths = [os.path.realpath(u["path"]) for u in unparseable]
        assert os.path.realpath(probe_path) in unparseable_paths, f"Expected {probe_path} to be in unparseable list, got: {unparseable}"
    finally:
        if os.path.exists(probe_path):
            os.remove(probe_path)

    # After cleanup, no unparseable files remain
    defs_clean, unparseable_clean = audit_repo_resize_state_dict_definitions(return_unparseable=True)
    assert len(unparseable_clean) == 0, f"Expected 0 unparseable files after cleanup, got: {unparseable_clean}"



