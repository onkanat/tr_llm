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


# --- Faz-2 inceleme K3: PAST_PERSON override, çözümle-seç dalından ÖNCE ------
# RED (04.11.2026, düzeltmeden önce): override, çok-şablonlu PERSON id'lerinde
# çözümle-seç dalının ARDINDAN uygulanıyordu. Dal 'geldi'+'(I)z' → 'geldiz'
# adayını compile kabul ediyor (donmuş grafta [gel,TENSE_PAST,PERSON_1PL]
# çözümünün yüzeyi 'geldiz'dir) ve on_cozum yolunu override sonradan
# DEĞİŞTİREMİYORDU. Ölçülen regresyonlar: 'geldiz'/'gelmediz'/'gelmiştiz'/
# 'geliyorduz'/'gelmeliydiz' (doğruları 'geldik' sınıfı).
@pytest.mark.parametrize("tags,beklenen", [
    (["gel", "TENSE_PAST", "PERSON_1PL"], "geldik"),
    (["gel", "NEG", "TENSE_PAST", "PERSON_1PL"], "gelmedik"),
    (["gel", "TENSE_EVIDENTIAL", "COPULA_PAST", "PERSON_1PL"], "gelmiştik"),
    (["gel", "TENSE_PROG", "COPULA_PAST", "PERSON_1PL"], "geliyorduk"),
    (["gel", "TENSE_NECESS", "COPULA_PAST", "PERSON_1PL"], "gelmeliydik"),
])
def test_past_person_override_onceligi_k3(decompiler, tags, beklenen):
    yuzey = decompiler.decompile_tags(tags)
    assert yuzey == beklenen, (
        f"{tags}: beklenen {beklenen!r}, gelen {yuzey!r} — PAST_PERSON "
        f"override çözümle-seç dalına yenik düşüyor")


def test_past_person_yuzeyleri_hala_derlenir_k3(decompiler):
    """K3 düzeltmesi yalnız decompiler'dadır; derleyici tarafı DEĞİŞMEMELİ.

    Ölçülen donmuş-gerçek (04.11.2026): grafta PERSON_1PL için 'k' şablonu
    YOKTUR ('(I)z'/'(y)Iz'); bu yüzden 'geldik' [gel, PART_DIk] olarak
    çözülür, [gel, TENSE_PAST, PERSON_1PL] çözümü ise (olmayan) 'geldiz'
    yüzeyindedir. Pin: her iki yüzey de HÂLÂ çözülür — decompiler
    düzeltmesi compile tarafını bozmaz."""
    for kelime, beklenen_idler in [
        ("geldik", ["gel", "PART_DIk"]),
        ("gelmedik", ["gel", "NEG", "PART_DIk"]),
        ("geldiz", ["gel", "TENSE_PAST", "PERSON_1PL"]),
    ]:
        paket = decompiler.compiler.compile(kelime)
        idler = [[m["id"] for m in a["morphemes"]]
                 for a in paket.get("analyses") or []]
        assert beklenen_idler in idler, (
            f"{kelime!r}: {beklenen_idler} çözüm uzayından düştü: {idler}")


# --- Faz-2 inceleme Ö1: çözümle-seç adayı KÖK-BAĞLI tercih ------------------
# RED (04.11.2026, düzeltmeden önce): ['iyi','PERSON_1SG'] → 'iyim'. Sıralı
# adayların ilki '(I)m' → 'iyim' compile KABUL ediliyordu, ama [iyi,
# POSS_1SG] olarak — çözüm zinciri İYELİK köklü, PERSON değil. Doğru aday
# '(y)Im' → 'iyiyim' → [iyi, PERSON_1SG]. Düzeltme: aday, compile çözüm
# uzayında root_lemma'dan başlayan ve şimdiye kadarki etiket dizisiyle
# birebir eşleşen analiz TAŞIYORSA tercih edilir.
@pytest.mark.parametrize("tags,beklenen", [
    (["iyi", "PERSON_1SG"], "iyiyim"),
    (["hasta", "PERSON_1SG"], "hastayım"),
])
def test_kisi_secimi_kok_bagli_tercih_o1(decompiler, tags, beklenen):
    yuzey = decompiler.decompile_tags(tags)
    assert yuzey == beklenen, (
        f"{tags}: beklenen {beklenen!r}, gelen {yuzey!r} — aday seçimi "
        f"iyelik-köklü zinciri yanlışlıkla tercih ediyor")


@pytest.mark.parametrize("tags,beklenen", [
    (["iyi", "PERSON_2SG"], "iyisin"),
    (["ev", "CASE_LOC", "PERSON_1SG"], "evdeyim"),
    (["nasıl", "PERSON_2SG"], "nasılsın"),
])
def test_kisi_secimi_yesiller_korunur_o1(decompiler, tags, beklenen):
    yuzey = decompiler.decompile_tags(tags)
    assert yuzey == beklenen, f"{tags}: yeşil davranış düştü: {yuzey!r}"

