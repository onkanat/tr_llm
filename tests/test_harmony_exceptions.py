#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2.5 — â/uzatma uyum istisnaları: imkân/hâl/ihtimal LEMMA niteliğiyle.

Zemin (phonology.py üstbilgi ölçümü, T-0080): 'â' UYUM bilgisi TAŞIMAZ —
HARMONY_VOWELS düzeltme işaretini atlar; '_get_last_vowel' gövdede başka
uyum ünlüsü yoksa 'a' fallback'ine düşer. Ölçülen tablo: 'efkâr'→ön ·
'kâr'→art · 'hâl'→ön — aynı biçimsel yapı, ZIT uyum. Bu istisnalar KURAL
hard-code edilmeden VERİYLE çözülür (T2.2 altyapısı: resolve_affix(...,
lemma_attributes)):

  imkân   NOUN  BACK_HARMONY  → imkân   + POSS_3SG/CASE_ACC 'ı' (imkânı)
  hâl     NOUN  FRONT_HARMONY → hâl     + POSS_3SG          'i' (hâli)
  ihtimal NOUN  FRONT_HARMONY → ihtimal + POSS_3SG          'i' (ihtimali)

RED ölçümü (04.11.2026, nitelik EKLENMEDEN önce): üçü de 0 analiz —
imkân 'i' nedeniyle ÖN veriyor ('imkâni'), hâl fallback 'a' ile ART
('hâlı'), ihtimal 'a' ile ART ('ihtimalı'). 'kâr' NOUN nitelik ALMAZ
(ölçüm: art): 'â' fallback 'a' mevcut ART uyumu birebir korur —
test_kar_regresyonu bunu sabitler. 'hal' (satır 18349, NOKTA'sız) ve
'ihtimal' ADV (satır 20176) DOKUNULMAZ.
"""

import os

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

LEXICON_PATH = os.path.join('data', 'lexicon', 'roots.tsv')


def _derleyici():
    """Compiler fixture'ı (test_morphology_regression._derleyici ile aynı)."""
    lex = LexiconManager()
    lex.load_from_tsv(LEXICON_PATH)
    return CrystalCompiler(lex, build_default_graph())


compiler = _derleyici()


def _yollar(word):
    """Kelimenin ÇÖZÜM UZAYI: her analizin morf-id kümesi (frozenset)."""
    r = compiler.compile(word)
    return {frozenset(m["id"] for m in a["morphemes"])
            for a in (r.get("analyses") or [])}


def test_harmony_istisnalari():
    # imkânı → imkân + Poss/Case ART uyumla (BACK_HARMONY niteliği):
    # 'imkân' gövdesinin son uyum ünlüsü 'i' (ÖN) — nitelik çevirir → 'ı'.
    yollar = _yollar("imkânı")
    assert frozenset({"imkân", "POSS_3SG"}) in yollar, (
        "imkânı: imkân+POSS_3SG çözüm uzayında yok — BACK_HARMONY işlemedi")
    assert frozenset({"imkân", "CASE_ACC"}) in yollar

    # hâli → hâl + POSS_3SG ÖN uyumla (FRONT_HARMONY): 'â' uyum taşımaz,
    # fallback 'a' ART verirdi ('hâlı') — nitelik ÖN'e çevirir → 'i'.
    yollar = _yollar("hâli")
    assert frozenset({"hâl", "POSS_3SG"}) in yollar, (
        "hâli: hâl+POSS_3SG çözüm uzayında yok — FRONT_HARMONY işlemedi")
    assert frozenset({"hâl", "CASE_ACC"}) in yollar

    # ihtimali → ihtimal + POSS_3SG ÖN (FRONT_HARMONY): gövde 'a' ile biter
    # (ART beklenir) ama sözlüksel gerçek ÖN → 'ihtimali'.
    yollar = _yollar("ihtimali")
    assert frozenset({"ihtimal", "POSS_3SG"}) in yollar, (
        "ihtimali: ihtimal+POSS_3SG çözüm uzayında yok — FRONT_HARMONY işlemedi")
    assert frozenset({"ihtimal", "CASE_ACC"}) in yollar


def test_kar_regresyonu():
    # kârı → kâr + POSS_3SG: nitelik YOK — 'â' fallback 'a' mevcut ART
    # uyumu kalır (ölçüm: kâr→art). Veri düzeltmesi bu davranışı BOZMAZ.
    yollar = _yollar("kârı")
    assert frozenset({"kâr", "POSS_3SG"}) in yollar, (
        "kârı: kâr+POSS_3SG DÜŞTÜ — kâr satırına nitelik karışmış olmalı")
    assert frozenset({"kâr", "CASE_ACC"}) in yollar
    assert compiler.compile("kârı").get("token_vector") == ["kâr", "POSS_3SG"]