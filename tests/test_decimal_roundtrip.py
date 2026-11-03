"""tests/test_decimal_roundtrip.py

Faz-1 / T1.2 — Ondalık birleştirme: "3.14" → "3. 14" hatası.
decompile_sentence digit bloğu 'rakam . rakam' örüntüsünü tek sayı kelimesi
olarak birleştirmeli. Üretim hattı (chat_prompt.py / agent_gateway.py /
epistemic_agent.py): ids → tokenizer.decode(ids) → decompile_sentence(...).

Koşturma:
    venv/bin/python -m pytest tests/test_decimal_roundtrip.py -q
"""
import os, sys, pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

@pytest.fixture(scope="module")
def fx():
    lex = LexiconManager(); lex.load_from_tsv("data/lexicon/roots.tsv")
    vocab = Vocabulary(); vocab.load("data/rebuild/vocab_anka_r1_33114.json")
    comp = CrystalCompiler(lex, build_default_graph())
    # DELTA: kontrolör fixture'ıyla aynı, tek fark literal_entity_mode=True
    # (test_u2019_typographic.py ile ortak blok; bu test için mode etkisizdir).
    return KristalTokenizer(comp, vocab, literal_entity_mode=True), comp, vocab

def test_ondalik_roundtrip(fx):
    t, _, _ = fx
    ids = t.encode("Daire 3.14 metrekaredir.")
    # DELTA (gövde uyarlaması): kontrolör spec'inde cikti = t.decode(ids)
    # doğrudan metin varsayılmıştı; KristalTokenizer.decode etiket dizisi
    # döndürür ("3 . 1 4"). Metin, üretim hattındaki gibi
    # decompiler.decompile_sentence(tokenizer.decode(ids)) ile üretilir.
    # Assert'ler kontrolör spec'iyle birebir aynıdır.
    dec = MorphemeDecompiler(t.compiler, t.vocab)
    cikti = dec.decompile_sentence(t.decode(ids))
    metin = " ".join(cikti.split())
    assert "3.14" in metin and "3. 14" not in metin

def test_ondalik_virgul_roundtrip(fx):
    """Faz-1 kalite-inceleme: Türkçe ondalık VİRGÜL. 'Sıcaklık 3,14 derece.'
    eski davranışta '3, 14' üretiyordu ('rakam , rakam' etiket görünümü
    birleşmiyordu). Beklenti: noktadaki lookahead mantığı virgül için de
    çalışır → '3,14' tek sayı kelimesi olarak birleşir."""
    t, _, _ = fx
    ids = t.encode("Sıcaklık 3,14 derece.")
    dec = MorphemeDecompiler(t.compiler, t.vocab)
    cikti = dec.decompile_sentence(t.decode(ids))
    metin = " ".join(cikti.split())
    assert "3,14" in metin and "3, 14" not in metin, (
        f"Ondalık virgül birleşmedi: {metin!r}"
    )

# Kasıtlı takaslar — '.' ve ',' birleşmesinin BİLİNÇLİ sınırlamaları.
# Etiket akışında 'rakam . rakam' / 'rakam , rakam' görünümü ondalık sayı ile
# cümle/listeyi ayırt EDİLEMEZ (bkz. decompiler.py digit bloğu yorum bloğu):
# tokenizer tüm rakamları tek-hane, nokta/virgülü ayrı token yayar; bağlam
# etiketlerde yoktur. Bu testler mevcut davranışı sabitler (pin): bilinçli
# takas olduğundan değişmemeli; değişirse yorum bloğuyla birlikte güncellenmelidir.
KASITLI_TAKASLAR = [
    # Takas: virgülle ayrılmış rakam listesi ondalık gibi birleşir (veri kaybı yok).
    ("1, 2 ve 3", "1,2 ve 3"),
    # Takas ('.', mevcut): noktayla biten sayıdan sonra gelen yeni sayı birleşir.
    ("3.14. 5 saniye geçti.", "3.14.5 saniye geçti."),
    # Kusur ('.', mevcut): zincir kırılınca ara boşluk girer — veri kaybı yok.
    ("192.168.1.9", "192.168.1. 9"),
]

@pytest.mark.parametrize("girdi,beklenen", KASITLI_TAKASLAR)
def test_digit_birlestirme_kasitli_takaslar(fx, girdi, beklenen):
    t, _, _ = fx
    ids = t.encode(girdi)
    dec = MorphemeDecompiler(t.compiler, t.vocab)
    cikti = dec.decompile_sentence(t.decode(ids))
    metin = " ".join(cikti.split())
    assert metin == beklenen, f"{girdi!r}: beklenen {beklenen!r}, alınan {metin!r}"