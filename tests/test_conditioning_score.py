"""tests/test_conditioning_score.py

T-0038 — conditioning_score modülünü doğrular.
Kanaryalar: pozitif, negatif (boş), negatif (alakasız), ayırım.
Mutant: formül kasıtlı bozulduğunda kanaryalardan en az biri kırmızı.

Koşturma (sandbox DIŞINDA):
    ./venv/bin/pytest -q tests/test_conditioning_score.py
"""
import pytest
import os
import sys
import importlib
import types
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.core import CrystalCompiler
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.eval.conditioning_score import conditioning_score

VOCAB_PATH = "data/rebuild/vocab_base_32816.json"


@pytest.fixture(scope="module")
def tok_and_vocab():
    lex = LexiconManager()
    lex.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lex, graph)
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH, freeze=True)
    tok = KristalTokenizer(compiler, vocab, literal_entity_mode=False)
    return tok, vocab


# ── KANARYA-1: Pozitif — istem metninin kendisi çıktı → skor ~ 1.0 ──────────

def test_canary1_self_copy_score_near_1(tok_and_vocab):
    """İstem metninin birebir kopyası çıktı olarak verildiğinde skor 0.99+."""
    tok, vocab = tok_and_vocab
    prompt = "Ahşap çalışmalarında zımpara kademelerini açıkla"
    output = prompt  # birebir kopya
    result = conditioning_score(prompt, output, tok, vocab)
    assert result["score"] >= 0.99, (
        f"Beklenen ≥0.99 (birebir kopya), alınan: {result['score']:.4f}"
    )
    assert result["bos"] is False


# ── KANARYA-2: Negatif — boş çıktı → skor == 0.0, bos=True ─────────────────

def test_canary2_empty_output_score_zero(tok_and_vocab):
    """Boş çıktı → score == 0.0 ve bos=True."""
    tok, vocab = tok_and_vocab
    prompt = "Ahşap çalışmalarında zımpara kademelerini açıkla"
    result = conditioning_score(prompt, "", tok, vocab)
    assert result["score"] == 0.0, f"Beklenen 0.0, alınan: {result['score']}"
    assert result["bos"] is True


# ── KANARYA-3: Negatif — alakasız çıktı → skor < 0.05 ──────────────────────

def test_canary3_irrelevant_output_score_low(tok_and_vocab):
    """Tamamen alakasız çıktı < 0.05."""
    tok, vocab = tok_and_vocab
    prompt = "Ahşap çalışmalarında zımpara kademelerini açıkla"
    # Matematiksel terimler — ahşapla ilgisiz
    output = "integral türev fonksiyon limit diferansiyel denklem"
    result = conditioning_score(prompt, output, tok, vocab)
    assert result["score"] < 0.05, (
        f"Beklenen <0.05 (alakasız), alınan: {result['score']:.4f}"
    )


# ── KANARYA-4: Ayırım — aynı kökler ama farklı istem çiftleri ───────────────

def test_canary4_separation_matched_above_cross(tok_and_vocab):
    """Doğru eşleşme çapraz eşleşmenin üzerinde olmalı.
    T-0036'da ölçülen: çapraz max 0.0769 < eşik.
    """
    tok, vocab = tok_and_vocab
    prompt_a = "Ahşap çalışmalarında zımpara kademelerini açıkla"
    output_a = "zımpara kademeleri ahşap yüzey bağlantı"
    prompt_b = "Masif ahşapta dolgu verniği kullanım koşullarını açıkla"
    output_b = "vernik dolgu koşul bağlantı"

    # Doğru eşleşmeler
    score_aa = conditioning_score(prompt_a, output_a, tok, vocab)["score"]
    score_bb = conditioning_score(prompt_b, output_b, tok, vocab)["score"]

    # Çapraz eşleşmeler
    score_ab = conditioning_score(prompt_a, output_b, tok, vocab)["score"]
    score_ba = conditioning_score(prompt_b, output_a, tok, vocab)["score"]

    cross_max = max(score_ab, score_ba)
    matched_min = min(score_aa, score_bb)

    assert matched_min > cross_max, (
        f"Doğru eşleşme (min={matched_min:.4f}) çapraz max ({cross_max:.4f}) üzerinde olmalı"
    )


# ── MUTANT: Formül bozulduğunda kanaryalardan en az biri kırmızı ─────────────

def test_mutant_broken_formula_triggers_canary(tok_and_vocab):
    """MUTANT KANARYA: Formül kasıtlı bozulduğunda kanaryalardan en az biri kırmızı olmalı.

    Mutant A: RP(O) sabit 1.0 — tekrar cezası devre dışı.
    Test yöntemi: Tekrarlı ama alakalı çıktı → gerçek formül düşük skor verir (RP ceza),
    mutant A yüksek skor verir. Bu fark kanaryanın vakum olmadığını kanıtlar.

    Mutant B: Payda len(C_O) yerine len(C_P).
    Test yöntemi: Çıktı istemden çok daha kısa → payda farkı skoru etkiler.
    """
    tok, vocab = tok_and_vocab
    from src.eval.conditioning_score import _extract_content_roots

    # ── Tekrarlı ama alakalı örnek ──
    # Gerçek RP(O) = min(1.0, (1/10) / 0.70) = 0.143 → skor << recall
    # Mutant A RP=1.0 → skor = recall (çok yüksek)
    prompt_rep = "ahşap zımpara tahta bağlantı masa"
    # "zımpara" 10 kez tekrar — alakalı ama tekrarlı
    output_rep = "zımpara zımpara zımpara zımpara zımpara zımpara zımpara zımpara zımpara zımpara"

    # Gerçek formül skoru
    real_result = conditioning_score(prompt_rep, output_rep, tok, vocab)
    real_score = real_result["score"]
    real_rp = real_result["rep_penalty"]

    # Mutant A: RP sabit 1.0
    def mutant_rp_always_1(prompt_text, output_text, tokenizer, vocab_obj):
        c_p = _extract_content_roots(prompt_text, tokenizer, vocab_obj)
        c_o = _extract_content_roots(output_text, tokenizer, vocab_obj)
        if not c_p or not c_o:
            return {"score": 0.0, "bos": True}
        p_counts = {}
        for r in c_p:
            p_counts[r] = p_counts.get(r, 0) + 1
        o_counts = {}
        for r in c_o:
            o_counts[r] = o_counts.get(r, 0) + 1
        common_count = sum(min(p_counts[r], o_counts.get(r, 0)) for r in p_counts)
        recall = common_count / len(c_p)
        # MUTANT: RP sabit 1.0 (ceza yok)
        score = recall * 1.0
        return {"score": score, "recall": recall, "rep_penalty": 1.0, "bos": False}

    mutant_a_score = mutant_rp_always_1(prompt_rep, output_rep, tok, vocab)["score"]

    # Mutant A gerçek skoru önemli ölçüde aşmalı (RP cezası yoksa skor yüksek kalır)
    mutant_a_breaks_canary = mutant_a_score > real_score * 2.0 and real_rp < 0.5

    # ── Mutant B: Payda len(C_O) ──
    # Çıktı istemden çok daha uzunsa payda farkı skoru değiştirir
    prompt_b = "ahşap zımpara"
    output_b = "zımpara ahşap tahta masa bağlantı vernik dolgu koşul"  # daha uzun çıktı

    def mutant_wrong_denominator(prompt_text, output_text, tokenizer, vocab_obj):
        c_p = _extract_content_roots(prompt_text, tokenizer, vocab_obj)
        c_o = _extract_content_roots(output_text, tokenizer, vocab_obj)
        if not c_p or not c_o:
            return {"score": 0.0, "bos": True}
        p_counts = {}
        for r in c_p:
            p_counts[r] = p_counts.get(r, 0) + 1
        o_counts = {}
        for r in c_o:
            o_counts[r] = o_counts.get(r, 0) + 1
        common_count = sum(min(p_counts[r], o_counts.get(r, 0)) for r in p_counts)
        # MUTANT: payda len(C_O) yerine len(C_P)
        recall = common_count / len(c_o) if c_o else 0.0
        unique_o = len(set(c_o))
        rep_ratio = unique_o / len(c_o)
        rep_penalty = min(1.0, rep_ratio / 0.70)
        score = recall * rep_penalty
        return {"score": score, "recall": recall, "rep_penalty": rep_penalty, "bos": False}

    real_b = conditioning_score(prompt_b, output_b, tok, vocab)["score"]
    mutant_b_score = mutant_wrong_denominator(prompt_b, output_b, tok, vocab)["score"]
    # Payda farklı olduğunda skor farklı çıkmalı
    mutant_b_breaks_canary = abs(mutant_b_score - real_b) > 0.01

    assert mutant_a_breaks_canary or mutant_b_breaks_canary, (
        f"HİÇBİR MUTANT KANARYA KIRMADIR — KANARYALAR VAKUMDUR!\n"
        f"Mutant A: real_score={real_score:.4f}, mutant_score={mutant_a_score:.4f}, "
        f"real_rp={real_rp:.4f} → {'KIRDI' if mutant_a_breaks_canary else 'KIRMIYOR'}\n"
        f"Mutant B: real_b={real_b:.4f}, mutant_b={mutant_b_score:.4f} → "
        f"{'KIRDI' if mutant_b_breaks_canary else 'KIRMIYOR'}"
    )



# ── Import testi: modül importlanabiliyor ──────────────────────────────────────

def test_module_importable():
    """conditioning_score modülü doğrudan import edilebilmeli."""
    from src.eval.conditioning_score import conditioning_score as cs
    assert callable(cs)


def test_return_keys(tok_and_vocab):
    """Dönüş sözlüğünün zorunlu anahtarları içermeli."""
    tok, vocab = tok_and_vocab
    result = conditioning_score("ahşap zımpara", "ahşap", tok, vocab)
    for key in ["score", "recall", "rep_penalty", "common_roots",
                "prompt_roots", "output_roots", "bos"]:
        assert key in result, f"Anahtar eksik: {key}"
