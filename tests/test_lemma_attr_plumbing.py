#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2.2 — lemma niteliklerinin fonolojiye taşınması (altyapı).

resolve_affix'e `lemma_attributes` parametresi eklenir: kökün (lemma'nın)
FRONT_HARMONY / BACK_HARMONY nitelikleri, gövdenin son ünlüsünden hesaplanan
uyum eksenini EZER. Amaç: 'ihtimal' (Arapça ödünçleme, son ünlü 'a' ama ekte
ÖN uyum: ihtimali, *ihtimalı) sınıfını VERİ ile çözmek — kural hard-code
edilmez. Sözlük donmuş: bu test yalnız imzayı ve eksen-çevirmeyi sınar,
roots.tsv'ye dokunmaz.

Davranış sözleşmesi: nitelik yoksa ('-') sonuç birebir aynı kalır.
"""
from src.compiler.phonology import PhonologyEngine


def test_front_harmony_niteligi_uyumu_cevirir():
    # 'ihtimal' son ünlü 'a' → arkalık beklenir ('ı'); FRONT_HARMONY önü zorlar.
    _, resolved = PhonologyEngine.resolve_affix("ihtimal", "(s)I", "-", "FRONT_HARMONY")
    assert resolved == "i"


def test_nitelik_yoksa_davranis_ayni():
    # lemma_attributes='-' (varsayılan) mevcut davranışı bit-özdeş korur.
    a = PhonologyEngine.resolve_affix("ihtimal", "(s)I", "-")
    b = PhonologyEngine.resolve_affix("ihtimal", "(s)I", "-", "-")
    assert a == b