#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite: Milestone M3 Challenger 2
========================================================
Rigorous empirical challenge of:
1. Hybrid Search RRF score ceiling, absence of 0.75 clipping, and monotonic strict ordering in [0.0, 1.0].
2. Thought-step filtering and step penalties: XML thinking tags (<think>, </think>, <thought>, <reasoning>, <dusunce>),
   English CoT phrases ("we need to", "first let's"), and Turkish CoT phrases ("düşünelim", "adım adım").
3. Structural tag isolation (<ENT>, <CAP>, <BELGE>, <PROPER_NOUN>, <NUMBER>) from morphological roots.
4. Curiosity threshold edge cases around tau = 2.500 (2.499 vs 2.500 vs 2.501) and future_train_vector.jsonl JSON schema compliance.
"""

import os
import json
import tempfile
from datetime import datetime
import pytest
import torch

from src.rag.hybrid_search import (
    RRF_SCORE_MAX,
    reciprocal_rank_fusion,
    normalize_rrf_score,
    filter_thought_steps,
    extract_distinctive_roots,
    compute_root_match_penalty,
    filter_and_rank_candidates,
)
from src.rag.merak import CuriosityEngine
from src.rag.epistemic_agent import EpistemicCuriosityAgent
from src.llm.router import TriModalRouter


# ==============================================================================
# Task 1: Hybrid Search RRF Score Ceiling & Monotonic Ordering
# ==============================================================================

def test_rrf_ceiling_no_clipping_above_075():
    """Verify that scores above 0.75 are NOT clipped to 1.0 and maintain distinct resolution."""
    assert RRF_SCORE_MAX == 1.0

    scores_to_test = [0.75, 0.78, 0.8333333333, 0.90, 0.95, 1.0]
    normalized = [normalize_rrf_score(s) for s in scores_to_test]

    # Verify each score retains its proportional value without being flattened
    for raw, norm in zip(scores_to_test, normalized):
        assert pytest.approx(norm, 1e-5) == raw, f"Score {raw} was altered to {norm}"

    # Verify strict ascending order
    for i in range(len(normalized) - 1):
        assert normalized[i] < normalized[i + 1], f"Scores {normalized[i]} and {normalized[i+1]} are not strictly ordered!"


def test_rrf_monotonic_strict_ordering_permutations():
    """Verify documents with dual top-ranks, single top-ranks, and lower ranks

    produce monotonically strictly ordered normalized scores in [0.0, 1.0].
    """
    list_a = ["c1", "c2", "c3", "c6", "c4", "c5", "c7", "c8", "c_dummy", "c9"]
    list_b = ["c1", "c4", "c2", "c5", "c3"]
    # c1: pos 0 (1/2) + pos 0 (1/2) = 1.0
    # c2: pos 1 (1/3) + pos 2 (1/4) = 0.5833
    # c3: pos 2 (1/4) + pos 4 (1/6) = 0.4167
    # c4: pos 4 (1/6) + pos 1 (1/3) = 0.5000
    # c5: pos 5 (1/7) + pos 3 (1/5) = 0.3429
    # c6: pos 3 (1/5) + None = 0.2000
    # c7: pos 6 (1/8) + None = 0.1250
    # c8: pos 7 (1/9) + None = 0.1111
    # c9: pos 9 (1/11) + None = 0.0909

    scores = reciprocal_rank_fusion([list_a, list_b], k=2)
    normalized = {cid: normalize_rrf_score(sc) for cid, sc in scores.items()}

    # All normalized scores must be in [0.0, 1.0]
    for cid, val in normalized.items():
        assert 0.0 <= val <= 1.0, f"Score {val} for {cid} out of [0.0, 1.0] bounds"

    # Strictly monotonic order check:
    assert normalized["c1"] > normalized["c2"] > normalized["c4"] > normalized["c3"] > normalized["c5"] > normalized["c6"] > normalized["c7"] > normalized["c8"] > normalized["c9"]
    assert normalized["c1"] == 1.0


def test_rrf_edge_case_inputs():
    """Verify RRF bounds handling: empty lists, single list, negative inputs, overflow."""
    assert reciprocal_rank_fusion([], k=2) == {}
    assert normalize_rrf_score(0.0) == 0.0
    assert normalize_rrf_score(-1.0) == 0.0
    assert normalize_rrf_score(2.5, max_score=1.0) == 1.0
    assert normalize_rrf_score(0.5, max_score=0.0) == 0.0


# ==============================================================================
# Task 2: Thought-Step Filtering & Penalty (Adversarial Challenge)
# ==============================================================================

@pytest.mark.parametrize("pure_text,tag_name", [
    ("<think>This is pure internal chain of thought.</think>", "<think>"),
    ("<thought>Thinking deeply about the user request.</thought>", "<thought>"),
    ("<reasoning>Step 1: formulate answer. Step 2: finalize.</reasoning>", "<reasoning>"),
    ("<dusunce>Öğrenciye pedagojik yanıt taslağı hazırlıyorum.</dusunce>", "<dusunce>"),
    ("</think>unclosed residual reasoning without opening</think>", "</think>"),
])
def test_thought_filtering_pure_thought_tags(pure_text, tag_name):
    """Test pure thought documents containing XML tags are rejected (is_pure_thought=True, penalty=0.0)."""
    cleaned, is_pure, is_contam, penalty = filter_thought_steps(pure_text)
    assert is_pure is True, f"Failed to reject pure thought tag {tag_name}: {pure_text}"
    assert is_contam is True, f"Failed to flag contamination in {tag_name}: {pure_text}"
    assert penalty == 0.0, f"Expected penalty 0.0, got {penalty} for {tag_name}"


@pytest.mark.parametrize("pure_text,phrase_name", [
    ("The user wants information.\nWe need to answer concisely.\nSo that should be enough.", "we need to"),
    ("First let's analyze the problem carefully.\nThen the question will be answered.", "first let's"),
    ("Önce nasıl cevap vereceğimizi düşünelim.\nİlk taslak hazırlanıyor.", "düşünelim"),
    ("Bu problemi adım adım inceleyelim.\nSon kontrol yapılıyor.", "adım adım"),
])
def test_thought_filtering_pure_thought_cot_phrases(pure_text, phrase_name):
    """Test pure thought documents containing English & Turkish CoT phrases are rejected."""
    cleaned, is_pure, is_contam, penalty = filter_thought_steps(pure_text)
    assert is_pure is True, f"Failed to reject pure thought CoT phrase '{phrase_name}': {pure_text}"
    assert is_contam is True, f"Failed to flag contamination in '{phrase_name}': {pure_text}"
    assert penalty == 0.0, f"Expected penalty 0.0, got {penalty} for '{phrase_name}'"


@pytest.mark.parametrize("text,marker", [
    (
        "<think>Ahşap birleşimleri hakkında düşünelim.</think>\n"
        "Kırlangıç kuyruğu zıvana birleştirme ahşap mobilyada yüksek mukavemet sağlar.",
        "<think>"
    ),
    (
        "</think>\n"
        "Meşe ağacı yüksek lif yoğunluğu sebebiyle suya ve çürümeye karşı dirençlidir.",
        "</think>"
    ),
    (
        "<reasoning>Analyze wood properties</reasoning>\n"
        "Gürgen ağacı sert dokusuyla tornacılıkta ve alet saplarında yaygın kullanılır.",
        "<reasoning>"
    ),
    (
        "<dusunce>Pedagojik açıklama</dusunce>\n"
        "Ceviz ağacı estetik desenleri ve işlenme kolaylığıyla masa yapımında tercih edilir.",
        "<dusunce>"
    ),
    (
        "We need to clarify the timber drying process.\n"
        "Fırın kurutma işlemi kerestenin nem oranını yüzde sekiz ile on iki arasına indirir.",
        "We need to"
    ),
    (
        "First let's inspect the joint tolerances.\n"
        "Kavelalı birleştirmelerde delik çapı kavela kalınlığı ile tam uyumlu olmalıdır.",
        "First let's"
    ),
    (
        "Önce yapıştırıcı türünü düşünelim.\n"
        "Poliüretan tutkallar neme maruz kalan dış mekan ahşap işlerinde kullanılır.",
        "düşünelim"
    ),
    (
        "Bu uygulamayı adım adım takip edelim.\n"
        "Masif ahşap yüzeyler zımparalama sonrası bezir yağı ile koruma altına alınır.",
        "adım adım"
    ),
])
def test_thought_filtering_contaminated_documents_receive_step_penalty(text, marker):
    """Test factual substantive documents containing thought markers receive step_penalty (0.5)

    and have thought markers stripped, preserving substantive text.
    """
    cleaned, is_pure, is_contam, penalty = filter_thought_steps(text, step_penalty=0.5)
    assert is_pure is False, f"Document with substantive facts was incorrectly marked pure thought: {text}"
    assert is_contam is True, f"Document with marker '{marker}' was not flagged as contaminated: {text}"
    assert penalty == 0.5, f"Expected step penalty 0.5, got {penalty} for marker '{marker}'"
    assert marker.lower() not in cleaned.lower(), f"Marker '{marker}' was not stripped from cleaned text: {cleaned}"


# ==============================================================================
# Task 3: Structural Tag Isolation
# ==============================================================================

def test_structural_tag_isolation_extract_distinctive_roots():
    """Verify that structural tags (<ENT>, <CAP>, <BELGE>, <PROPER_NOUN>, <NUMBER>)

    are never extracted as morphological roots.
    """
    tags_string = (
        "<BOS> <ENT> </ENT> <CAP> </CAP> <BELGE> </BELGE> <PROPER_NOUN> </PROPER_NOUN> "
        "<NUMBER> </NUMBER> <PAD> <UNK> <LITERAL> </LITERAL> <EOS> "
        "marangoz meşe DERIV_lIK CASE_DAT TENSE_AORIST PLURAL"
    )

    roots = extract_distinctive_roots(tags_string)

    forbidden_tokens = {
        "<ent>", "</ent>", "<cap>", "</cap>", "<belge>", "</belge>",
        "<proper_noun>", "</proper_noun>", "<number>", "</number>",
        "<bos>", "<eos>", "<pad>", "<unk>", "<literal>", "</literal>",
        "ent", "cap", "belge", "proper_noun", "number", "bos", "eos", "pad", "unk", "literal"
    }

    leakage = roots.intersection(forbidden_tokens)
    assert len(leakage) == 0, f"Structural tags leaked into morphological roots: {leakage}"

    assert "marangoz" in roots
    assert "meşe" in roots
    assert "deriv_lik" not in roots
    assert "case_dat" not in roots


def test_query_with_only_structural_tags_triggers_fail_closed_penalty():
    """If a query consists solely of structural tags with no distinctive roots,

    it must receive a 0.05 fail-closed penalty when tested against documents with substantive roots.
    """
    structural_query = "<ENT> <CAP> <BELGE> </BELGE> </CAP> </ENT> <NUMBER> <PROPER_NOUN>"
    query_roots = extract_distinctive_roots(structural_query)
    assert query_roots == set()

    doc_roots = {"kırlangıç", "zıvana", "ahşap"}
    raw_score = 1.0

    penalized_score, has_match, matching = compute_root_match_penalty(raw_score, query_roots, doc_roots)
    assert pytest.approx(penalized_score, 1e-4) == 0.05
    assert has_match is False
    assert len(matching) == 0


# ==============================================================================
# Task 4: Curiosity Threshold Edge Cases & Schema Compliance
# ==============================================================================

def test_curiosity_engine_tau_boundary_deterministic():
    """Test Shannon entropy around tau = 2.500: 2.499 vs 2.500 vs 2.501.

    Must trigger deterministically:
    - 2.499 < 2.5 -> False
    - 2.500 >= 2.5 -> True
    - 2.501 >= 2.5 -> True
    """
    engine = CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5)

    e_below = torch.tensor([2.4999])
    e_exact = torch.tensor([2.5000])
    e_above = torch.tensor([2.5001])

    assert engine.detect_epistemic_gap(e_below).item() is False, "Entropy 2.4999 should NOT trigger curiosity!"
    assert engine.detect_epistemic_gap(e_exact).item() is True, "Entropy 2.5000 MUST trigger curiosity!"
    assert engine.detect_epistemic_gap(e_above).item() is True, "Entropy 2.5001 MUST trigger curiosity!"


def test_epistemic_agent_tau_boundary_and_jsonl_schema():
    """Test EpistemicCuriosityAgent at tau=2.500 boundary and verify future_train_vector.jsonl

    JSON schema compliance across all 14 required fields.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        future_file = os.path.join(tmp_dir, "future_train_vector.jsonl")

        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3, "ahşap": 4}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "ahşap")

        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text): return [0, 4, 2]
            def decode(self, ids): return "ahşap"

        class ControlledEntropyModel(torch.nn.Module):
            def __init__(self, target_entropy: float):
                super().__init__()
                self.target_entropy = target_entropy
                self.embedding = torch.nn.Embedding(10, 64)
            def forward(self, x, return_hidden_states=False):
                batch_size, seq_len = x.shape
                logits = torch.zeros(batch_size, seq_len, 10)
                hidden = torch.randn(batch_size, seq_len, 64)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        class HighScoreMemory:
            collection_name = "anka_bellek"
            def hybrid_recall(self, dense, sparse, top_k=1, query_tags=""):
                return [{
                    "text": "Meşe ağacı yüksek mukavemetli masif ahşaptır.",
                    "score": 0.92,  # >= 0.85
                    "has_root_match": True,
                    "metadata": {"crystal_tags": "meşe ağaç ahşap", "token_ids": [4]}
                }]

        # Sub-case A: entropy_post = 2.499 (< 2.500) -> NO epistemic failure, NO record
        agent_below = EpistemicCuriosityAgent(
            model=ControlledEntropyModel(2.499),
            tokenizer=MockTokenizer(),
            memory=HighScoreMemory(),
            curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5),
            router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
            tau=2.5,
            similarity_threshold=0.85,
            future_train_path=future_file,
            device="cpu"
        )
        agent_below.generate_tokens = lambda prompt_tokens, **kwargs: ([4, 1], 2.499)
        res_below = agent_below.process_query("Ahşap nedir?", force_rag=True)

        assert res_below["epistemic_failure"] is False
        assert res_below["future_train_recorded"] is False
        assert not os.path.exists(future_file)

        # Sub-case B: entropy_post = 2.500 (== 2.500) -> Epistemic failure, WRITES record
        agent_exact = EpistemicCuriosityAgent(
            model=ControlledEntropyModel(2.500),
            tokenizer=MockTokenizer(),
            memory=HighScoreMemory(),
            curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5),
            router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
            tau=2.5,
            similarity_threshold=0.85,
            future_train_path=future_file,
            device="cpu"
        )
        agent_exact.generate_tokens = lambda prompt_tokens, **kwargs: ([4, 1], 2.500)
        res_exact = agent_exact.process_query("Ahşap nedir?", force_rag=True)

        assert res_exact["epistemic_failure"] is True
        assert res_exact["future_train_recorded"] is True
        assert os.path.exists(future_file)

        # Sub-case C: Verify JSON Schema of recorded line
        with open(future_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1

        record = json.loads(lines[0])
        required_fields = [
            "instruction", "input", "output", "model_failed_output",
            "decompiled_output", "rag_document", "retrieval_collection",
            "similarity_score", "similarity_threshold", "entropy_pre",
            "entropy_post", "tau", "reason", "timestamp"
        ]

        for field in required_fields:
            assert field in record, f"Missing required JSON schema field '{field}' in future_train_vector.jsonl!"

        # Value and type checks
        assert record["tau"] == 2.5
        assert record["similarity_score"] >= 0.85
        assert record["reason"] == "epistemic_gap_unresolved_high_similarity"
        assert record["retrieval_collection"] == "anka_bellek"
        assert isinstance(record["entropy_post"], (int, float))
        # ISO timestamp format check
        dt = datetime.fromisoformat(record["timestamp"])
        assert dt is not None
