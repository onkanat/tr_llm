#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2.4 — İsim-soylu ek-fiil şahsı + ADJ→isim türetimleri (morfoloji).

Kapsam (build_default_graph, 'Noun Copula' bloğu sonrası):
  * ADJ_ROOT / NOUN_POST_CASE → PERSON_1SG/2SG/1PL/2PL (iyiyim, nasılsın,
    güzelsin, evdeyim). PERSON_* kimlikleri fiil yolundakiyle AYNIDIR —
    sözlük donmuş (33.114), yeni morfem jetonu YOK.
  * ADJ_ROOT → DERIV_lIk/lI/sIz/CI → NOUN_ROOT (hayranlık, güzellik,
    güzelsiz). DERIV_* kimlikleri isim yolundakiyle AYNIDIR.
  * Çıplak NOUN_ROOT'a şahıs BİLİNÇLİ olarak eklenmedi: 'evim' iyelik
    (POSS_1SG) ile çözülmeli, PERSON_1SG çözüm uzayında görünmemeli.

Çözüm-uzayı semantiği: beklentiler TÜM `analyses` morf-id dizileri
içinde aranır — homograf (güzel ADJ+NOUN) ve skor-tie hallerinde yolun
VARLIĞI sınanır; en iyi `token_vector` üzerinden değil (skill: tr-llm).
"""

import pytest

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler


@pytest.fixture(scope='module')
def comp():
    lex = LexiconManager()
    lex.load_from_tsv('data/lexicon/roots.tsv')
    return CrystalCompiler(lex, build_default_graph())


def _analizler(comp, kelime):
    sonuc = comp.compile(kelime)
    return list(sonuc.get('analyses') or [])


def _id_dizileri(comp, kelime):
    return [[m['id'] for m in a['morphemes']] for a in _analizler(comp, kelime)]


def test_iyiyim_nasilsin_guzelsin(comp):
    """ADJ kökten ek-fiil şahsı; (y) tamponu ünlü-sonlu gövdeyi taşır."""
    # iyi + (y)Im -> 'iyiyim' (ünlü-sonlu -> tampon y)
    iyiyim = _analizler(comp, 'iyiyim')
    assert ['iyi', 'PERSON_1SG'] in [[m['id'] for m in a['morphemes']] for a in iyiyim]
    yol = next(a for a in iyiyim if [m['id'] for m in a['morphemes']] == ['iyi', 'PERSON_1SG'])
    assert yol['surface_form'] == 'iyiyim'
    assert yol['morphemes'][1]['surface'] == 'yim'

    # nasıl + (y)sIn -> 'nasılsın' (ünsüz-sonlu -> tampon yok)
    nasilsin = _analizler(comp, 'nasılsın')
    assert ['nasıl', 'PERSON_2SG'] in [[m['id'] for m in a['morphemes']] for a in nasilsin]
    yol = next(a for a in nasilsin if [m['id'] for m in a['morphemes']] == ['nasıl', 'PERSON_2SG'])
    assert yol['surface_form'] == 'nasılsın'
    assert yol['morphemes'][1]['surface'] == 'sın'

    # güzel + (y)sIn -> 'güzelsin'
    guzelsin = _analizler(comp, 'güzelsin')
    assert ['güzel', 'PERSON_2SG'] in [[m['id'] for m in a['morphemes']] for a in guzelsin]
    yol = next(a for a in guzelsin if [m['id'] for m in a['morphemes']] == ['güzel', 'PERSON_2SG'])
    assert yol['surface_form'] == 'güzelsin'
    assert yol['morphemes'][1]['surface'] == 'sin'

    # ev + CASE_LOC + PERSON_1SG -> 'evdeyim' (case sonrası şahıs)
    evdeyim = _analizler(comp, 'evdeyim')
    assert ['ev', 'CASE_LOC', 'PERSON_1SG'] in [
        [m['id'] for m in a['morphemes']] for a in evdeyim]
    yol = next(a for a in evdeyim
               if [m['id'] for m in a['morphemes']] == ['ev', 'CASE_LOC', 'PERSON_1SG'])
    assert yol['surface_form'] == 'evdeyim'
    assert yol['morphemes'][1]['surface'] == 'de'
    assert yol['morphemes'][2]['surface'] == 'yim'

    # Ünlü-sonlu gövde ailesi: (y) tamponu yalnız ünlü-ünlü boşluğunda
    # (iyiyim/iyiyiz); s-ile başlayan ekte tampon YOK (iyisin/iyisiniz) —
    # '(y)sIn' şablonu yanlış 'iyiysin' yüzeyi üretirdi (decompiler T-0147
    # çok-şablon dalı da PERSON_2SG'yi bozar; o yüzden fiil-şablonu birebir).
    for kelime, bek in [('iyiyiz', ['iyi', 'PERSON_1PL']),
                        ('iyisin', ['iyi', 'PERSON_2SG']),
                        ('iyisiniz', ['iyi', 'PERSON_2PL'])]:
        yollar = _analizler(comp, kelime)
        assert bek in [[m['id'] for m in a['morphemes']] for a in yollar], kelime
        assert any(a['surface_form'] == kelime for a in yollar), kelime


def test_evim_iyelik_regresyon(comp):
    """'evim' iyelik içerir; şahıs (PERSON_*) çözüm uzayında YOKTUR.

    Çıplak NOUN_ROOT'a şahıs eki bilinçli olarak bağlanmadı: aksi hâlde
    'evim' [ev, POSS_1SG] ile [ev, PERSON_1SG] ikilemesine düşerdi.
    """
    idler = _id_dizileri(comp, 'evim')
    assert idler, "'evim' çözülemedi"
    assert ['ev', 'POSS_1SG'] in idler
    tum_morflar = {mid for dizi in idler for mid in dizi}
    assert 'PERSON_1SG' not in tum_morflar, (
        f"'evim' çözüm uzayında PERSON_1SG göründü: {idler}")


def test_adj_den_isim_turetimi(comp):
    """ADJ → isim yapım ekleri; DERIV_* kimlikleri isim yolundakiyle aynı."""
    # hayran + lIk -> 'hayranlık'
    assert ['hayran', 'DERIV_lIk'] in _id_dizileri(comp, 'hayranlık')
    # güzel + lIk -> 'güzellik'
    assert ['güzel', 'DERIV_lIk'] in _id_dizileri(comp, 'güzellik')
    # güzel + sIz -> 'güzelsiz'
    assert ['güzel', 'DERIV_sIz'] in _id_dizileri(comp, 'güzelsiz')
    # yüzey bütünlüğü: türemiş kelime tek parça geri gelmeli
    for kelime in ('hayranlık', 'güzellik', 'güzelsiz'):
        yollar = _analizler(comp, kelime)
        assert yollar, f"{kelime!r} çözülemedi"
        assert any(a['surface_form'] == kelime for a in yollar), kelime