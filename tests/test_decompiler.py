import os
import sys
import pytest

# Ensure workspace root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import Vocabulary

@pytest.fixture(scope="module")
def decompiler():
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    return MorphemeDecompiler(compiler, vocab)

def test_decompiler_basic_words(decompiler):
    test_cases = [
        (["kitap", "PLURAL", "CASE_LOC"], "kitaplarda"),
        (["okul", "CASE_DAT"], "okula"),
        (["git", "TENSE_FUT", "PERSON_1SG"], "gideceğim"),
        (["çocuk", "PLURAL"], "çocuklar"),
        (["ev", "POSS_1SG", "CASE_ABL"], "evimden"),
        (["gel", "NEG", "TENSE_PAST", "PERSON_2SG"], "gelmedin"),
        (["yaz", "POTENTIAL", "TENSE_PROG", "PERSON_1PL"], "yazabiliyoruz"),
        (["oku", "PART_DIk", "POSS_3SG"], "okuduğu"),
    ]
    for tags, expected in test_cases:
        reconstructed = decompiler.decompile_tags(tags)
        assert reconstructed == expected, f"Failed for {tags}: expected {expected}, got {reconstructed}"

def test_decompiler_sentences(decompiler):
    sentence_tags = "kitap POSS_3SG oku TENSE_PAST PERSON_1SG ve temizle TENSE_PAST PERSON_1SG"
    expected = "kitabı okudum ve temizledim"
    result = decompiler.decompile_sentence(sentence_tags)
    assert result == expected, f"Expected '{expected}', got '{result}'"

def test_decompiler_copula_past_buffer(decompiler):
    test_cases = [
        (["tanımlan", "INF_mAk", "CASE_LOC", "COPULA_PAST"], "tanımlanmaktaydı"),
        (["göster", "INF_mAk", "CASE_LOC", "COPULA_PAST"], "göstermekteydi"),
        (["gel", "TENSE_EVIDENTIAL", "COPULA_PAST"], "gelmişti"),
        (["ev", "CASE_LOC", "COPULA_PAST"], "evdeydi"),
    ]
    for tags, expected in test_cases:
        result = decompiler.decompile_tags(tags)
        assert result == expected, f"Failed for {tags}: expected {expected}, got {result}"

def test_decompiler_placeholders(decompiler):
    sentence_tags = "göster INF_mAk CASE_LOC COPULA_PAST <UNK> ve <NUMBER> ile <PROPER_NOUN>"
    expected = "göstermekteydi [?] ve [sayı] ile [Özel İsim]"
    result = decompiler.decompile_sentence(sentence_tags)
    assert result == expected, f"Expected '{expected}', got '{result}'"

