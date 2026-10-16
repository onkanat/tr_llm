#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kristal-Vektörel Mimarisi: Hibrit Arama & Sıralama Modülü (Hybrid Search)
========================================================================
Dense (KristalEmbedding semantik) ve Sparse (BM25 leksikal) vektörlerin
Reciprocal Rank Fusion (RRF) ile birleştirilmesi, düşünce adımı (CoT)
filtrelemesi/cezası, morfolojik kök ayrıştırması ve kalibre edilmiş skor
normalizasyonunu saf fonksiyonlar (pure functions) olarak sağlar.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# RRF skor kalibrasyonu:
# Qdrant'ın 2-kollu (dense + sparse) RRF algoritmasında k=2 sıralama sabitiyle
# her kolun 1. sıradaki adayı için skor: 1 / (1 + 2 - 1) = 0.5'tir.
# Her iki kolda da 1. sırada olan mükemmel eşleşmenin teorik tavan skoru:
# 0.5 + 0.5 = 1.0'dır.
# Eski 0.75 sabiti tavanı yapay olarak kırpıp üst %25'lik dilimi (0.75 - 1.0)
# 1.0'a düzleştirdiğinden, kalibre edilmiş teorik tavan RRF_SCORE_MAX = 1.0 olarak tanımlanır.
RRF_SCORE_MAX: float = 1.0

# Morfolojik kök çıkarımında göz ardı edilecek fonksiyonel / bağlaç kökler
COMMON_TURKISH_STOP_ROOTS: Set[str] = {
    "su", "bir", "ve", "de", "da", "ki", "o", "bu", "şu", "ama", "ile",
    "en", "daha", "her", "şey", "için", "ol", "et", "yap",
    "ne", "kim", "nasıl", "neden", "niçin", "hangi", "nere", "kaç", "mı", "mi", "mu", "mü"
}

# Çekim eki ve türetim önekleri
INFLECTION_PREFIXES: Tuple[str, ...] = (
    "TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_"
)

INFLECTION_TAGS: Set[str] = {"PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"}

# Düşünce zinciri (CoT) blok ve satır desenleri
THOUGHT_BLOCK_PATTERNS = [
    r'</?(?:think|thought|reasoning|dusunce)>.*?</?(?:think|thought|reasoning|dusunce)>',
]

COT_LINE_PATTERNS = [
    # English CoT markers
    r'\bthe user\b',
    r'\bwe need to\b',
    r'\bthey want\b',
    r'\blet\'s produce\b',
    r'\bhere is\b',
    r'\bso that should\b',
    r'\bmake it\b',
    r'\bfirst,?\s+let\'?s?\b',
    r'\bin this case\b',
    r'\bconcise, pedagogically\b',
    r'\bto answer this\b',
    r'\bthen the question\b',
    # Turkish CoT markers
    r'\b(?:dusunce|düşünce|dusunelim|düşünelim)\b',
    r'\bderinlemesine düşünce\b',
    r'\bpedagojik kontrol\b',
    r'\b(?:adım|step)\s*\d+\b',
    r'\badım\s+adım\b',
    r'\bhedef kitlem(?:iz)?\b',
    r'\böğretmen rolü(?:ndeyim)?\b',
    r'\b(?:ilk\s+)?taslak\b',
    r'\bson kontrol\b',
]

_THOUGHT_BLOCK_REGEX = re.compile('|'.join(THOUGHT_BLOCK_PATTERNS), re.DOTALL | re.IGNORECASE)
_COT_LINE_REGEX = re.compile('|'.join(COT_LINE_PATTERNS), re.IGNORECASE)
_ANY_THOUGHT_TAG_REGEX = re.compile(r'</?(?:think|thought|reasoning|dusunce)>', re.IGNORECASE)
_OPENING_THOUGHT_REGEX = _ANY_THOUGHT_TAG_REGEX


def reciprocal_rank_fusion(
    ranked_lists: List[List[Any]],
    k: int = 2
) -> Dict[Any, float]:
    """Computes Reciprocal Rank Fusion (RRF) scores across multiple ranked lists.
    
    Formula: score(d) = sum_{list} 1 / (rank_i(d) + k)
    where rank_i is 1-based (i.e. pos + 1), equivalent to 1 / (pos + k + 1 - 1) = 1 / (pos + k).
    With k=2 and two branches, theoretical maximum for an item ranked #1 in both is:
    1 / (0 + 2) + 1 / (0 + 2) = 0.5 + 0.5 = 1.0.
    """
    scores: Dict[Any, float] = {}
    for r_list in ranked_lists:
        for pos, item in enumerate(r_list):
            item_id = item[0] if isinstance(item, (tuple, list)) else item
            score = 1.0 / (pos + k)
            scores[item_id] = scores.get(item_id, 0.0) + score
    return scores


def normalize_rrf_score(score: float, max_score: float = RRF_SCORE_MAX) -> float:
    """Normalizes raw RRF score to [0.0, 1.0] interval without premature clipping.
    Theoretical maximum for 2-branch RRF with k=2 is 1.0.
    """
    if max_score <= 0.0:
        return 0.0
    return min(1.0, max(0.0, float(score / max_score)))


def filter_thought_steps(
    text: str,
    step_penalty: float = 0.5
) -> Tuple[str, bool, bool, float]:
    """Filters thought steps and determines contamination status.
    
    Handles:
    - XML/HTML thought blocks (<think>, <thought>, <reasoning>, <dusunce>)
    - English and Turkish CoT reasoning markers
    
    Returns:
        Tuple of:
        - cleaned_text (str): Content with thought blocks/lines stripped
        - is_pure_thought (bool): True if text was completely thought/reasoning without substantive facts
        - is_contaminated (bool): True if any thought artifacts were detected
        - penalty_multiplier (float): 0.0 for pure thought (drop), step_penalty for mild contamination, 1.0 for clean
    """
    if not text:
        return "", False, False, 1.0

    has_thought_block = bool(_THOUGHT_BLOCK_REGEX.search(text) or _ANY_THOUGHT_TAG_REGEX.search(text))
    has_cot_marker = bool(_COT_LINE_REGEX.search(text))

    # Strip thought blocks
    stripped = _THOUGHT_BLOCK_REGEX.sub('', text)
    # Strip residual tags if unclosed
    stripped = re.sub(r'</?(?:think|thought|reasoning|dusunce)>', '', stripped, flags=re.IGNORECASE)

    # Strip line-by-line CoT
    clean_lines = []
    has_line_contamination = False
    for line in stripped.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        if _COT_LINE_REGEX.search(line_s):
            has_line_contamination = True
            continue
        clean_lines.append(line_s)

    cleaned_text = '\n'.join(clean_lines).strip()
    is_contaminated = has_thought_block or has_cot_marker or has_line_contamination

    # Check if remaining substantive content exists
    # If cleaned text is empty or has fewer than 10 alphanumeric chars, it is pure thought
    substantive_chars = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9]', '', cleaned_text)
    if len(substantive_chars) < 10 and is_contaminated:
        is_pure_thought = True
        penalty = 0.0
    elif is_contaminated:
        is_pure_thought = False
        penalty = step_penalty
    else:
        is_pure_thought = False
        penalty = 1.0

    return cleaned_text if cleaned_text else text, is_pure_thought, is_contaminated, penalty


def extract_distinctive_roots(
    tags_str: Optional[str],
    common_roots: Optional[Set[str]] = None
) -> Set[str]:
    """Extracts distinctive lexical roots from morpheme crystal tags.
    
    Excludes:
    - ALL structural and special tags (<...>, e.g., <BOS>, <EOS>, <PAD>, <UNK>,
      <ENT>, </ENT>, <CAP>, </CAP>, <BELGE>, </BELGE>, <PROPER_NOUN>, <LITERAL>, etc.)
    - Derivational morphemes (DERIV_...)
    - Inflectional affixes (CASE_..., TENSE_..., PERSON_..., PLURAL, etc.)
    - Common functional/stop roots ('ve', 'bir', 'için', question words, etc.)
    """
    if not tags_str:
        return set()

    stopwords = common_roots if common_roots is not None else COMMON_TURKISH_STOP_ROOTS
    distinctive_roots: Set[str] = set()

    for m in tags_str.split():
        m_strip = m.strip()
        if not m_strip:
            continue
        # Exclude ALL XML-style / structural tags
        if m_strip.startswith("<") and m_strip.endswith(">"):
            continue
        if m_strip.startswith("DERIV_"):
            continue
        if m_strip.startswith(INFLECTION_PREFIXES) or m_strip in INFLECTION_TAGS:
            continue
        root = m_strip.lower()
        if root not in stopwords:
            distinctive_roots.add(root)

    return distinctive_roots


def compute_root_match_penalty(
    score: float,
    query_roots: Set[str],
    doc_roots: Set[str]
) -> Tuple[float, bool, Set[str]]:
    """Calculates root match constraint score and penalty.
    
    Returns:
        (penalized_score, has_root_match, matching_roots)
    """
    if not query_roots:
        # T-0147 A-3 (OOV fail-closed): empty distinctive query roots
        # receives severe penalty if doc has roots (query couldn't produce any valid root)
        if doc_roots:
            return score * 0.05, False, set()
        return score, True, set()

    matching_roots = query_roots.intersection(doc_roots)
    match_ratio = len(matching_roots) / len(query_roots)

    if match_ratio == 0:
        return score * 0.05, False, set()
    elif match_ratio < 0.5:
        return score * (0.3 + 0.7 * match_ratio), True, matching_roots
    else:
        return score, True, matching_roots


def filter_and_rank_candidates(
    scored_points: List[Any],
    query_tags: Optional[str] = None,
    top_k: int = 3,
    step_penalty: float = 0.5,
    rrf_score_max: float = RRF_SCORE_MAX
) -> List[Dict[str, Any]]:
    """Filters, penalizes, normalizes and ranks candidate points from Qdrant RRF retrieval.
    
    Args:
        scored_points: List of Qdrant ScoredPoint objects or dicts
        query_tags: Crystal tags string of query for root extraction
        top_k: Number of top results to return
        step_penalty: Multiplier applied to mildly contaminated thought documents
        rrf_score_max: Theoretical maximum for score normalization
        
    Returns:
        List of dicts with keys: 'score', 'text', 'metadata', 'has_root_match', 'matching_roots'
    """
    distinctive_query_roots = extract_distinctive_roots(query_tags)

    results = []
    for sp in scored_points:
        raw_score = float(sp.score if hasattr(sp, "score") else sp.get("score", 0.0))
        payload = sp.payload if hasattr(sp, "payload") else sp.get("payload", {})
        if payload is None:
            payload = {}

        text = payload.get("text", "")
        doc_tags_str = payload.get("crystal_tags", "")

        # 1. Thought Step & CoT Contamination Filter
        cleaned_text, is_pure_thought, is_contaminated, cot_multiplier = filter_thought_steps(
            text, step_penalty=step_penalty
        )
        if is_pure_thought:
            # Drop pure thought candidates
            continue

        score = raw_score * cot_multiplier

        # 2. Token-Type Constraint / Root Match Penalty
        has_root_match = True
        matching_roots: Set[str] = set()
        if query_tags and doc_tags_str:
            doc_roots = extract_distinctive_roots(doc_tags_str)
            score, has_root_match, matching_roots = compute_root_match_penalty(
                score, distinctive_query_roots, doc_roots
            )

        # 3. Calibrated Normalization
        normalized_score = normalize_rrf_score(score, max_score=rrf_score_max)

        results.append({
            "score": normalized_score,
            "text": text,
            "metadata": payload,
            "has_root_match": has_root_match,
            "matching_roots": list(matching_roots)
        })

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
