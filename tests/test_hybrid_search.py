#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Birim Testleri: Hibrit Arama & Sıralama (hybrid_search.py)
===========================================================
Reciprocal Rank Fusion (RRF), düşünce adımı (CoT) filtreleme ve cezalandırma,
yapısal etiket (<...>) ayıklama ve morfolojik kök kısıtlarını doğrular.
"""

import pytest
from src.rag.hybrid_search import (
    RRF_SCORE_MAX,
    reciprocal_rank_fusion,
    normalize_rrf_score,
    filter_thought_steps,
    extract_distinctive_roots,
    compute_root_match_penalty,
    filter_and_rank_candidates,
)


def test_reciprocal_rank_fusion_scoring():
    """RRF k=2 formülüne göre doğru skorlama yapar."""
    dense_list = ["doc1", "doc2", "doc3"]
    sparse_list = ["doc1", "doc3", "doc4"]

    scores = reciprocal_rank_fusion([dense_list, sparse_list], k=2)

    # doc1 her iki listede de 1. sırada (pos=0): 1/(0+2) + 1/(0+2) = 0.5 + 0.5 = 1.0
    assert pytest.approx(scores["doc1"], 1e-4) == 1.0

    # doc3 dense'de 3. sırada (pos=2: 1/4 = 0.25), sparse'da 2. sırada (pos=1: 1/3 = 0.3333): 0.5833
    assert pytest.approx(scores["doc3"], 1e-4) == (0.25 + 1.0 / 3.0)

    # doc2 yalnız dense'de 2. sırada (pos=1): 1/3 = 0.3333
    assert pytest.approx(scores["doc2"], 1e-4) == 1.0 / 3.0

    # doc4 yalnız sparse'da 3. sırada (pos=2): 1/4 = 0.25
    assert pytest.approx(scores["doc4"], 1e-4) == 0.25


def test_normalize_rrf_score_calibrated_ceiling():
    """RRF skor normalizasyonu 0.75'te kırpılmaz, [0.0, 1.0] tavan aralığını korur."""
    assert RRF_SCORE_MAX == 1.0

    # Mükemmel eşleşme (rank 1 dense + rank 1 sparse = 1.0)
    assert pytest.approx(normalize_rrf_score(1.0), 1e-4) == 1.0

    # Yakın eşleşme (rank 1 dense + rank 2 sparse = 0.8333)
    s_833 = 0.5 + 1.0 / 3.0
    assert pytest.approx(normalize_rrf_score(s_833), 1e-4) == s_833

    # Rank 1 dense + rank 3 sparse = 0.75
    assert pytest.approx(normalize_rrf_score(0.75), 1e-4) == 0.75

    # Skor ayrımı korunur (düzleşme YOK)
    assert normalize_rrf_score(1.0) > normalize_rrf_score(s_833) > normalize_rrf_score(0.75)

    # Sınırlar
    assert normalize_rrf_score(-0.1) == 0.0
    assert normalize_rrf_score(1.5) == 1.0


def test_filter_thought_steps_pure_thought_rejection():
    """Saf düşünce adımı içeren belgeler elenir (is_pure_thought=True, penalty=0.0)."""
    pure_think = "<think>We need to think step by step about carpentry.</think>"
    _, is_pure, is_contam, penalty = filter_thought_steps(pure_think)
    assert is_pure is True
    assert is_contam is True
    assert penalty == 0.0

    pure_dusunce = "<dusunce>Öğretmen rolündeyim, öğrenciye cevap taslağı hazırlıyorum.</dusunce>"
    _, is_pure, is_contam, penalty = filter_thought_steps(pure_dusunce)
    assert is_pure is True
    assert penalty == 0.0

    pure_english_cot = (
        "The user says: Give me info.\n"
        "We need to answer concisely.\n"
        "Here is the draft."
    )
    _, is_pure, is_contam, penalty = filter_thought_steps(pure_english_cot)
    assert is_pure is True
    assert penalty == 0.0


def test_filter_thought_steps_mild_contamination_step_penalty():
    """Düşünce adımıyla hafif kirlenmiş fakat olgusal bilgi taşıyan metne adım cezası uygulanır."""
    mild_doc = (
        "<think>Kırlangıç kuyruğu hakkında düşünelim.</think>\n"
        "Kırlangıç kuyruğu ahşap birleştirmede yüksek mukavemet sağlayan kadim bir tekniktir."
    )
    cleaned, is_pure, is_contam, penalty = filter_thought_steps(mild_doc, step_penalty=0.5)
    assert is_pure is False
    assert is_contam is True
    assert penalty == 0.5
    assert "<think>" not in cleaned
    assert "Kırlangıç kuyruğu ahşap birleştirmede" in cleaned

    turkish_cot_doc = (
        "Adım 1: Soru marangozlukla ilgili.\n"
        "Meşe ağacı yüksek lif yoğunluğu sebebiyle suya ve çürümeye karşı son derece dayanıklıdır."
    )
    cleaned2, is_pure2, is_contam2, penalty2 = filter_thought_steps(turkish_cot_doc, step_penalty=0.5)
    assert is_pure2 is False
    assert is_contam2 is True
    assert penalty2 == 0.5
    assert "Adım 1" not in cleaned2
    assert "Meşe ağacı yüksek lif yoğunluğu" in cleaned2


def test_filter_thought_steps_clean_text():
    """Olgusal temiz Türkçe metin hiçbir ceza veya elemeye uğramaz."""
    clean_text = "Ceviz ağacı mobilya yapımında estetik desenleri ve sert dokusuyla tercih edilir."
    cleaned, is_pure, is_contam, penalty = filter_thought_steps(clean_text)
    assert is_pure is False
    assert is_contam is False
    assert penalty == 1.0
    assert cleaned == clean_text


def test_extract_distinctive_roots_tag_exclusion():
    """Tüm yapısal etiketler (<...>), çekim ekleri ve bağlaç kökler ayıklanır."""
    tags = (
        "<BOS> <ENT> <CAP> marangoz DERIV_lIK <BELGE> meşe ağaç CASE_DAT "
        "</BELGE> <PROPER_NOUN> masa <NUMBER> <SYMBOL> <EOS>"
    )
    roots = extract_distinctive_roots(tags)

    # Yapısal etiketlerin hiçbiri kök olarak dönmemeli
    assert "<ent>" not in roots
    assert "<cap>" not in roots
    assert "<belge>" not in roots
    assert "</belge>" not in roots
    assert "<bos>" not in roots
    assert "<eos>" not in roots
    assert "<proper_noun>" not in roots

    # Türetim ve çekim ekleri ayıklanmalı
    assert "deriv_lik" not in roots
    assert "case_dat" not in roots

    # Anlamsal kökler korunmalı
    assert "marangoz" in roots
    assert "meşe" in roots
    assert "ağaç" in roots
    assert "masa" in roots


def test_compute_root_match_penalty_scenarios():
    """Kök eşleşme oranı senaryoları cezayı doğru hesaplar."""
    query_roots = {"kırlangıç", "kuyruk", "zıvana"}

    # Tam eşleşme (3/3 >= 50%) -> cezasız
    doc_roots_full = {"kırlangıç", "kuyruk", "zıvana", "ahşap"}
    score, has_match, matched = compute_root_match_penalty(1.0, query_roots, doc_roots_full)
    assert score == 1.0
    assert has_match is True
    assert matched == {"kırlangıç", "kuyruk", "zıvana"}

    # Kısmi eşleşme (1/3 < 50%) -> (0.3 + 0.7 * 1/3) = 0.5333
    doc_roots_partial = {"kırlangıç", "tutkal", "vida"}
    score, has_match, matched = compute_root_match_penalty(1.0, query_roots, doc_roots_partial)
    expected = 1.0 * (0.3 + 0.7 * (1.0 / 3.0))
    assert pytest.approx(score, 1e-4) == expected
    assert has_match is True
    assert matched == {"kırlangıç"}

    # Sıfır eşleşme -> 0.05 ceza
    doc_roots_none = {"bahçe", "toprak", "tohum"}
    score, has_match, matched = compute_root_match_penalty(1.0, query_roots, doc_roots_none)
    assert pytest.approx(score, 1e-4) == 0.05
    assert has_match is False
    assert len(matched) == 0

    # OOV / Boş sorgu kökü -> belge kök içeriyorsa 0.05 fail-closed ceza
    score_oov, has_match_oov, _ = compute_root_match_penalty(1.0, set(), doc_roots_full)
    assert pytest.approx(score_oov, 1e-4) == 0.05
    assert has_match_oov is False


def test_filter_and_rank_candidates_integration():
    """filter_and_rank_candidates saf düşünceyi eler, kirlenmişi cezalandırır, temiz eşleşeni öne çıkarır."""
    class MockPoint:
        def __init__(self, score, text, crystal_tags):
            self.score = score
            self.payload = {"text": text, "crystal_tags": crystal_tags}

    candidates = [
        # 1. Saf düşünce: elenmeli
        MockPoint(1.0, "<think>We need to analyze the wood type.</think>", "<think> ağaç"),
        # 2. Kısmi kirlenmiş ama kök eşleşen: cezalandırılmalı (0.9 * 0.5 = 0.45)
        MockPoint(0.9, "<think>Düşünce</think> Kırlangıç kuyruğu zıvana birleştirme mukavemeti.", "kırlangıç kuyruk zıvana"),
        # 3. Temiz ve tam kök eşleşen: en üstte olmalı (0.8 * 1.0 = 0.8)
        MockPoint(0.8, "Kırlangıç kuyruğu ahşap işçiliğinde zıvana mukavemetini en üst seviyeye çıkarır.", "kırlangıç kuyruk zıvana ahşap"),
    ]

    results = filter_and_rank_candidates(
        scored_points=candidates,
        query_tags="<ENT> kırlangıç kuyruk zıvana </ENT>",
        top_k=3,
        step_penalty=0.5,
    )

    # Saf düşünce elendiği için 2 aday kalmalı
    assert len(results) == 2

    # Temiz aday 1. sırada olmalı
    assert "ahşap işçiliğinde" in results[0]["text"]
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["has_root_match"] is True

    # Kirlenmiş aday 2. sırada olmalı
    assert "mukavemeti." in results[1]["text"]
