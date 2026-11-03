#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2.3 — -(I)yor ünlü kaynaşması (anlıyor/arıyor/okuyor/oynuyor sınıfı).

Ünlü-sonlu fiil gövdeleri -(I)yor'a katılırken gövde ünlüsü düşer:
  anla+ıyor → anlıyor · oku+yor → okuyor · oyna+uyor → oynuyor.
İki tamamlayıcı mekanizma:
  a) resolve_affix: "(y)Iyor" ünlü-sonlu gövdeye gelirse — son ünlü düz
     (a/e) ise gövde kırpılır ve tampon-y kalkar (anla→anl+Iyor); yuvarlak
     (o/ö/u/ü) ise I düşer, tampon-y kalır (oku→oku+(y)or).
  b) find_stems: yüzey kırpılmış gövdeyle (anl) yürüyüp 'ı/i' gördüğünde,
     tam lemma (anla, VERB) geri-bakışla kök adayı döner (yüzey gövdesi
     matched_prefix'tir; kalan 'ıyor' DFS'te (y)Iyor ile eşleşir).

Faz-2 K1 — de/ye sınıfı (ünlüsüz-kalan gövde kaynaşması): 2-karakter
gövde ('de', 'ye') kırpılınca ÜNLÜ KALMAZ ('d', 'y'); _get_last_vowel
fallback'ı 'a' verir → art-uyumlu olmayan 'dıyor'/'yıyor' üretilir ve
compile yalnız olmayan yüzeyi kabul ederdi ('diyor'/'yiyor' → []). İKİ
YARIM, İKİSİ DE ŞART:
  (a) phonology: kırpma sonrası gövdede ünlü kalmadıysa kırpma ÖNCESİ
      is_back/is_rounded KORUNUR (yeniden hesap yapılmaz); ünlü kaldıysa
      mevcut yeniden hesap sürer ('oyna'→'oyn' -uyor doğrusu bozulmaz).
  (b) veri: roots.tsv'de de/ye VERB satırlarına FRONT_HARMONY (çapa #3)
      — parser yolunu açar (retro-lookback 'd'+FRONT_HARMONY → I→'i').

Faz-2 K2 — NEG+(I)yor compiler'da ölü: 'anlamıyor' → [] (yol
'anlama'+(y)Iyor kaynaşmasında budanıyor: 'anlamıyor'.startswith('anlama')
False). Düzeltme core.py budama adımında: new_string a/e ile bitiyorsa
target.startswith(new_string[:-1]) adayı da KABUL edilir; bir sonraki
adımda (y)Iyor ya kırpıp eşleşir ya yol ölür.

Faz-2 Ö2 — 'arıyor' NOUN homograf kilidi ('arı' kökü çözüm uzayında
OLMAMALI; T2.3 retro-lookback pos=VERB süzer) ve modül-kapsamı fixture
(sözlük modül başına BİR kez yüklenir — param başına yeniden yükleme yok).

Test semantiği — BİLİNÇLİ: ÇÖZÜM UZAYI (tüm `analyses`) sınanır, en-iyi
analiz değil. Homograf köklerde eşit-morfemli yollar skor-tie'da kalır ve
TSV satır sırası en-iyiyi belirleyebilir (bilinen 'etti' dersi); bu testin
iddiası kök+TENSE_PROG YOLUNUN VAR olduğudur.
"""
import pytest
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler


@pytest.fixture(scope='module')
def comp():
    """Modül-kapsamı derleyici fixture'ı (Ö2): 52k-satırlık TSV param
    başına değil, modül başına BİR kez yüklenir."""
    lex = LexiconManager()
    lex.load_from_tsv('data/lexicon/roots.tsv')
    return CrystalCompiler(lex, build_default_graph())


@pytest.fixture(scope='module')
def dec(comp):
    return MorphemeDecompiler(comp)


def _id_dizileri(paket):
    return [[m['id'] for m in a['morphemes']]
            for a in (paket.get('analyses') or [])]


def _analiz_var(paket, kok, ek):
    """Çözüm uzayında `kok` köküyle başlayıp `ek` içeren analiz var mı?"""
    for analiz in paket.get('analyses') or []:
        morfemler = analiz['morphemes']
        if morfemler[0]['id'] == kok and any(m['id'] == ek for m in morfemler[1:]):
            return True
    return False


@pytest.mark.parametrize("yuzey,kok", [
    ("anlıyor", "anla"),
    ("arıyor", "ara"),
    ("başlıyor", "başla"),
    ("bekliyor", "bekle"),
    ("atlıyor", "atla"),
    ("söylüyor", "söyle"),
    ("oynuyor", "oyna"),
    ("okuyor", "oku"),
])
def test_kaynasma_ayrisir(comp, yuzey, kok):
    paket = comp.compile(yuzey)
    assert paket.get('analyses'), f"{yuzey} hiç çözülemedi"
    assert _analiz_var(paket, kok, 'TENSE_PROG'), (
        f"{yuzey}: çözüm uzayında {kok}+TENSE_PROG yok; analizler="
        f"{_id_dizileri(paket)}")


@pytest.mark.parametrize("yuzey", ["geliyor", "bakıyor", "duyuyor"])
def test_regresyon_var_olan(comp, yuzey):
    """Ünsüz-sonlu gövdeli -(I)yor yüzeyleri halihazırda çözülür — korunmalı."""
    paket = comp.compile(yuzey)
    analyses = paket.get('analyses') or []
    assert analyses, f"{yuzey} hiç çözülemedi"
    assert paket['best_surface'] == yuzey, (
        f"{yuzey}: best_surface bozuldu: {paket['best_surface']!r}")
    assert 'TENSE_PROG' in [m['id'] for m in analyses[0]['morphemes']], (
        f"{yuzey}: en-iyi analizde TENSE_PROG yok: "
        f"{[m['id'] for m in analyses[0]['morphemes']]}")


# --- Faz-2 K1: de/ye — ünlüsüz-kalan gövde kaynaşması ------------------------

@pytest.mark.parametrize("kok,beklenen", [
    ("de", "diyor"),
    ("ye", "yiyor"),
])
def test_de_ye_prog_donusturme_k1(dec, kok, beklenen):
    """decompile_tags([kök, TENSE_PROG]) doğru (önlü) yüzey üretmeli.

    RED (04.11.2026, düzeltmeden önce): 'dıyor'/'yıyor' — kırpma sonrası
    ünlüsüz kalan gövdede _get_last_vowel 'a' fallback'ına düşüyor."""
    yuzey = dec.decompile_tags([kok, "TENSE_PROG"])
    assert yuzey == beklenen, (
        f"decompile_tags([{kok!r}, 'TENSE_PROG']): beklenen {beklenen!r}, "
        f"gelen {yuzey!r}")


@pytest.mark.parametrize("yuzey,kok", [
    ("diyor", "de"),
    ("yiyor", "ye"),
])
def test_de_ye_prog_cozum_uzayinda_k1(comp, yuzey, kok):
    """compile('diyor') uzayında [de, TENSE_PROG] VAR olmalı.

    RED (04.11.2026): [] — yalnız 'dıyor' (olmayan yüzey) kabul ediliyordu.
    roots.tsv de/ye VERB satırlarına FRONT_HARMONY eklendi (çapa #3):
    retro-lookback 'd'+FRONT_HARMONY → I→'i' → 'iyor' eşleşir."""
    paket = comp.compile(yuzey)
    idler = _id_dizileri(paket)
    assert [kok, 'TENSE_PROG'] in idler, (
        f"{yuzey}: çözüm uzayında {kok}+TENSE_PROG yok; analizler={idler}")


@pytest.mark.parametrize("yuzey,kok", [
    ("dıyor", "de"),
    ("yıyor", "ye"),
])
def test_olmayan_art_uyumlu_yuzey_red_k1(comp, yuzey, kok):
    """'dıyor'/'yıyor' (olmayan art-uyumlu yüzeyler) de/ye üzerinden
    çözülmemeli. RED (04.11.2026): compile('dıyor') uzayında
    [de, TENSE_PROG] VARDI (yanlış kabul). 'da'/'ya' homograf kökleri
    ayrı lemmalardır — pin yalnız de/ye içindir."""
    paket = comp.compile(yuzey)
    idler = _id_dizileri(paket)
    assert [kok, 'TENSE_PROG'] not in idler, (
        f"{yuzey}: olmayan yüzey {kok}+TENSE_PROG olarak çözülüyor: {idler}")


@pytest.mark.parametrize("kok,beklenen", [
    ("oyna", "oynuyor"),
    ("oku", "okuyor"),
    ("anla", "anlıyor"),
    ("söyle", "söylüyor"),
    ("ara", "arıyor"),
])
def test_kaynasma_regresyon_k1a(dec, kok, beklenen):
    """K1a regesyonu: ünlü KALAN kırpılmış gövde mevcut yeniden hesapla
    sürer — 'oyna'→'oyn' taşıyıcısı kalan 'o' (-uyor), 'oku' yuvarlak dal."""
    yuzey = dec.decompile_tags([kok, "TENSE_PROG"])
    assert yuzey == beklenen, (
        f"decompile_tags([{kok!r}, 'TENSE_PROG']): beklenen {beklenen!r}, "
        f"gelen {yuzey!r}")


# --- Faz-2 K2: NEG+(I)yor compiler'da ölü yol -------------------------------

@pytest.mark.parametrize("yuzey,beklenen_idler", [
    ("anlamıyor", ["anla", "NEG", "TENSE_PROG"]),
    ("istemiyor", ["iste", "NEG", "TENSE_PROG"]),
    ("bilmiyor", ["bil", "NEG", "TENSE_PROG"]),
    ("görmüyor", ["gör", "NEG", "TENSE_PROG"]),
    ("aramıyor", ["ara", "NEG", "TENSE_PROG"]),
    ("oynamıyor", ["oyna", "NEG", "TENSE_PROG"]),
    ("okumuyor", ["oku", "NEG", "TENSE_PROG"]),
    ("anlamıyorlar", ["anla", "NEG", "TENSE_PROG", "PERSON_3PL"]),
])
def test_neg_prog_cozulur_k2(comp, yuzey, beklenen_idler):
    """RED (04.11.2026, düzeltmeden önce): hepsi → [] — kök+NEG gövdesi
    'anlama' budağında kesiliyor, çünkü 'anlamıyor'.startswith('anlama')
    False (kaynaşmada NEG ünlüsü düşer). Budama gevşetmesi: a/e biten
    new_string için new_string[:-1] önek adayı da kabul edilir."""
    paket = comp.compile(yuzey)
    idler = _id_dizileri(paket)
    assert beklenen_idler in idler, (
        f"{yuzey}: çözüm uzayında {beklenen_idler} yok; analizler={idler}")


@pytest.mark.parametrize("yuzey", ["geliyor", "anlıyor", "okuyor"])
def test_prog_regresyon_neg_sonrasi_k2(comp, yuzey):
    """K2 gevşetmesi NEG'siz -(I)yor yüzeylerini bozmaz."""
    paket = comp.compile(yuzey)
    idler = _id_dizileri(paket)
    assert any('TENSE_PROG' in dizi for dizi in idler), (
        f"{yuzey}: TENSE_PROG yolu düştü: {idler}")


# --- Faz-2 Ö2: 'arıyor' NOUN homograf kilidi ---------------------------------

def test_ariyor_noun_homograf_kilidi_o2(comp):
    """'arıyor' çözüm uzayında 'arı' (NOUN homograf) köküyle başlayan
    analiz YOK — T2.3 retro-lookback pos=VERB süzer. Bugün doğru; buraya
    kilitlenir ('arı' NOUN satırı olsa da yol açılmaz)."""
    paket = comp.compile("arıyor")
    idler = _id_dizileri(paket)
    assert idler, "'arıyor' hiç çözülemedi"
    arili = [dizi for dizi in idler if dizi[0] == "arı"]
    assert not arili, f"'arıyor' uzayında NOUN homograf 'arı' yolu: {arili}"
    assert ["ara", "TENSE_PROG"] in idler, (
        f"'arıyor': ara+TENSE_PROG yolu yok: {idler}")