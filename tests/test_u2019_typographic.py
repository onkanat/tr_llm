"""tests/test_u2019_typographic.py

Faz-1 / T1.1 — U+2019 (tipografik kesme) encode regex'ine alınır.
Beklenti: 'Türkçe’dir' (U+2019) ile 'Türkçe'dir' (U+0027) aynı ID dizisini
üretmeli; kesme tokeni (id 32850) ve ek çözümü (COPULA_AORIST) yaşamlı kalmalı.

Koşturma:
    venv/bin/python -m pytest tests/test_u2019_typographic.py -q
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
    # DELTA (kontrolör fixture'ına uyarlama): test_u2019_kesme_tokeni_yasiyor
    # "'" tokeninin YAYILMASINI gerektirir; bu yalnız literal_entity_mode=True
    # dalında olur (tokenizer.py L266 koşulu + L291-293 apostrof tokeni).
    # mode=False ile "'" hiçbir koşulda üretilmez (probe ile doğrulandı).
    return KristalTokenizer(comp, vocab, literal_entity_mode=True), comp, vocab

def test_u2019_ascii_ile_ayni_id_dizisi(fx):
    t, _, _ = fx
    assert t.encode("Türkçe'dir") == t.encode("Türkçe’dir")

def test_u2019_ascii_ile_ayni_id_dizisi_mode_false(fx):
    """Faz-1 kalite-inceleme pini: mode=False'ta da U+2019 ve U+0027 kesme
    aynı ID dizisini üretmeli. Bu yolda kesme sözcükten silinip genel
    derleme yolu işletilir (apostrof dalı hiç girilmez); iki yazım da
    aynı 'Türkçeedir' gövdesine düşer — kalite-inceleme probe'uyla
    doğrulanmış mevcut davranış, regresyona karşı sabitlenir."""
    _, comp, vocab = fx
    t_false = KristalTokenizer(comp, vocab, literal_entity_mode=False)
    assert t_false.encode("Türkçe'dir") == t_false.encode("Türkçe’dir"), (
        "mode=False'ta tipografik ve düz kesme farklı ID dizisi üretti"
    )

def test_u2019_kesme_tokeni_yasiyor(fx):
    t, _, vocab = fx
    ids = t.encode("Türkçe’dir")
    morf = [vocab.decode(i) for i in ids]
    assert "'" in morf and "COPULA_AORIST" in morf

@pytest.mark.parametrize("kelime", ["'Merhaba'", "’Merhaba’"])
def test_kesme_cevreli_kelime_icerigi_kaybolmaz(fx, kelime):
    """Faz-1 kalite-inceleme (boş gövde + kelime kaybı): kesme işareti
    kelimeyi çerçeveliyorsa (ör. 'Merhaba' / ’Merhaba’ — alıntı/tırnak
    kullanımı) apostrof dalında stem boş kalır; eski davranışta <UNK>
    üretilip kelime içeriği tamamen kaybolurdu ('[?]'). Beklenti: kesmeler
    soyulup kelime genel derleme yoluna düşmeli — 'Merhaba' kökü
    ('merhaba') morf listesinde yaşamalı, <UNK> olmamalı."""
    t, comp, vocab = fx
    ids = t.encode(kelime)
    morf = [vocab.decode(i) for i in ids]
    assert "<UNK>" not in morf, f"{kelime!r}: <UNK> üretildi: {morf}"
    assert "merhaba" in morf, f"{kelime!r}: 'Merhaba' kökü/derlemesi yok: {morf}"
    # Uçtan uca: üretim hattı gibi metin de üretilip içerik korunmalı.
    dec = MorphemeDecompiler(comp, vocab)
    metin = dec.decompile_sentence(t.decode(ids))
    assert "merhaba" in metin, f"{kelime!r}: metin içeriği kayboldu: {metin!r}"