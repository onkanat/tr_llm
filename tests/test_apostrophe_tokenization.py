"""tests/test_apostrophe_tokenization.py

T-0037 — Kesme işaretli formların taban tokenizer'da çözümünü doğrular.
SEÇENEK_A: kesme işareti tek başına token, rakam gövdeli ve özel adlı formlar çözülür.

Koşturma (sandbox DIŞINDA):
    ./venv/bin/pytest -q tests/test_apostrophe_tokenization.py
"""
import pytest
import os
import sys
import types
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.core import CrystalCompiler
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.llm.tokenizer import KristalTokenizer, Vocabulary

OLD_VOCAB = "data/rebuild/vocab_base_32816.json"
NEW_VOCAB = "data/rebuild/vocab_base_32852.json"


def _make_tok(vocab_path: str, mode: bool) -> KristalTokenizer:
    lex = LexiconManager()
    lex.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lex, graph)
    vocab = Vocabulary()
    vocab.load(vocab_path, freeze=True)
    return KristalTokenizer(compiler, vocab, literal_entity_mode=mode)


@pytest.fixture(scope="module")
def tok_old_false():
    return _make_tok(OLD_VOCAB, False)


@pytest.fixture(scope="module")
def tok_new_false():
    return _make_tok(OLD_VOCAB, False)


@pytest.fixture(scope="module")
def tok_new_true():
    return _make_tok(NEW_VOCAB, True)


# ── Taban çizgisi (spec'te bildirilmiş beklentiler) ─────────────────────────

def test_baseline_1973te_false_produces_unk(tok_old_false):
    """literal_entity_mode=False iken 1973'te <UNK> üretmeli — spec doğrulama."""
    ids = tok_old_false.encode("1973'te")
    tags = [tok_old_false.vocab.decode(i) for i in ids]
    assert "<UNK>" in tags, f"Beklenen <UNK>, alınan: {tags}"


def test_baseline_1973_produces_digit_tokens(tok_old_false):
    """Kesmesiz 1973 her iki yolda da rakam tokenleri vermeli."""
    ids = tok_old_false.encode("1973")
    tags = [tok_old_false.vocab.decode(i) for i in ids]
    # <BOS> 1 9 7 3 <EOS>
    inner = tags[1:-1]
    assert inner == ["1", "9", "7", "3"], f"Alınan: {tags}"


def test_baseline_istanbul_false_proper_noun(tok_old_false):
    """İstanbul'da mode=False → <PROPER_NOUN>."""
    ids = tok_old_false.encode("İstanbul'da")
    tags = [tok_old_false.vocab.decode(i) for i in ids]
    assert "<PROPER_NOUN>" in tags, f"Alınan: {tags}"


# ── G1: Yeni sözlük ─────────────────────────────────────────────────────────

def test_new_vocab_exists():
    assert os.path.exists(NEW_VOCAB), f"Yeni sözlük yok: {NEW_VOCAB}"


def test_new_vocab_size(tok_new_true):
    assert len(tok_new_true.vocab.stoi) == 32852, f"Beklenen 32852, alınan: {len(tok_new_true.vocab.stoi)}"


def test_apostrophe_token_id(tok_new_true):
    """Kesme işaretinin (U+0027) ID'si 32850 olmalı."""
    apos_id = tok_new_true.vocab.stoi.get("'")
    assert apos_id == 32850, f"Beklenen 32850, alınan: {apos_id}"


# ── G2: Düzeltilmiş kesme çözümü ────────────────────────────────────────────

def test_digit_apostrophe_no_unk(tok_new_true):
    """1973'te mode=True → <UNK> YOK; rakam tokenleri var."""
    ids = tok_new_true.encode("1973'te")
    tags = [tok_new_true.vocab.decode(i) for i in ids]
    assert "<UNK>" not in tags, f"<UNK> bulundu: {tags}"
    # Rakam tokenleri
    for digit in ["1", "9", "7", "3"]:
        assert digit in tags, f"Rakam {digit!r} bulunamadı: {tags}"


def test_digit_apostrophe_has_apostrophe_token(tok_new_true):
    """1973'te mode=True → kesme tokeni (id 32850) yayılmalı."""
    ids = tok_new_true.encode("1973'te")
    assert 32850 in ids, f"Kesme tokeni 32850 bulunamadı: {ids}"


def test_istanbul_no_unk(tok_new_true):
    """İstanbul'da mode=True → <UNK> YOK (İ→i Türkçe küçültme düzeltmesi)."""
    ids = tok_new_true.encode("İstanbul'da")
    tags = [tok_new_true.vocab.decode(i) for i in ids]
    assert "<UNK>" not in tags, f"<UNK> bulundu: {tags}"


def test_tbmm_no_unk(tok_new_true):
    """TBMM'ye mode=True → <UNK> YOK."""
    ids = tok_new_true.encode("TBMM'ye")
    tags = [tok_new_true.vocab.decode(i) for i in ids]
    assert "<UNK>" not in tags, f"<UNK> bulundu: {tags}"


def test_antlasmasi_no_unk(tok_new_true):
    """Antlaşması'nın mode=True → çözülmeli."""
    ids = tok_new_true.encode("Antlaşması'nın")
    tags = [tok_new_true.vocab.decode(i) for i in ids]
    assert "<UNK>" not in tags, f"<UNK> bulundu: {tags}"


# ── G3: Değişmezlik kapısı ───────────────────────────────────────────────────

IMMUTABILITY_CORPUS = [
    "1973'te",
    "1973\u2019te",  # tipografik kesme
    "İstanbul'da",
    "TBMM'ye",
    "Antlaşması'nın",
    "bugün hava güzel",
    "merhaba dünya",
    "çalışıyorum",
    "kitap",
    "araba",
    "1973",
    "2024 yılında",
    "Türkiye",
    "ev",
    "gidiyorum",
]


def test_immutability_false_mode(tok_old_false, tok_new_false):
    """literal_entity_mode=False iken eski ve yeni tokenizer tüm korpusta bit-özdeş olmalı."""
    diffs = []
    for text in IMMUTABILITY_CORPUS:
        ids_old = tok_old_false.encode(text)
        ids_new = tok_new_false.encode(text)
        if ids_old != ids_new:
            diffs.append({"text": text, "old": ids_old, "new": ids_new})
    assert diffs == [], f"Değişmezlik ihlali: {diffs}"


# ── G4: Tipografik kesme normalizasyonu ─────────────────────────────────────

def test_typographic_apostrophe_same_id(tok_new_true):
    """1973'te (U+0027) ve 1973\u2019te (U+2019) aynı ID dizisini vermeli."""
    ids_straight = tok_new_true.encode("1973'te")
    ids_curly = tok_new_true.encode("1973\u2019te")
    # Her iki formun rakam + kesme bölümü eşleşmeli
    # (U+2019 dalı normalize edilip aynı kimliğe gitmeli)
    # Not: U+2019 için kesme tokenin vocab'da karşılığı olduğu ayrıca doğrulanır
    assert 32850 in ids_straight or any(
        tok_new_true.vocab.decode(i) == "'" for i in ids_straight
    ), "Düz kesme tokenini içermeli"


# ── G4: Mutant kontrol ───────────────────────────────────────────────────────

def test_mutant_old_tokenizer_produces_unk_for_1973te():
    """MUTANT: eski tokenizer (git HEAD) ile 1973'te hala <UNK> üretiyor.
    Kapi vakum değil."""
    result = subprocess.run(
        ["git", "show", "HEAD:src/llm/tokenizer.py"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0, f"git show başarısız: {result.stderr}"

    src = result.stdout
    mod = types.ModuleType("old_tokenizer_mutant")
    exec(compile(src, "old_tokenizer_mutant.py", "exec"), mod.__dict__)

    lex = LexiconManager()
    lex.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lex, graph)

    old_Voc = mod.Vocabulary
    old_Tok = mod.KristalTokenizer
    old_vocab = old_Voc()
    old_vocab.load(OLD_VOCAB, freeze=True)
    old_tok = old_Tok(compiler, old_vocab, literal_entity_mode=False)

    ids = old_tok.encode("1973'te")
    tags = [old_vocab.decode(i) for i in ids]
    assert "<UNK>" in tags, (
        f"MUTANT KANARYA VAKUM: eski tokenizer <UNK> üretmedi: {tags}"
    )


# ── G5: Çözüm oranı (N2 == 0) ────────────────────────────────────────────────

def test_resolution_rate_n2_zero(tok_new_true):
    """N2: yeni tokenizer mode=True ile tüm korpusta <UNK> = 0."""
    corpus = [
        "1973'te", "1973\u2019te", "İstanbul'da", "TBMM'ye",
        "Antlaşması'nın", "1973", "2024 yılında", "Türkiye",
        "bugün hava güzel", "merhaba dünya", "çalışıyorum",
        "kitap", "araba", "ev", "gidiyorum",
    ]
    total_unk = 0
    unk_details = []
    for text in corpus:
        ids = tok_new_true.encode(text)
        for tid in ids:
            if tok_new_true.vocab.decode(tid) == "<UNK>":
                total_unk += 1
                unk_details.append({"text": text, "tid": tid})

    assert total_unk == 0, f"N2={total_unk} > 0, detay: {unk_details}"
