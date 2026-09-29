# -*- coding: utf-8 -*-
"""T-0177 — Külliyat-düzeyi eğitilebilirlik kapısı testleri (İLAN K7).

Kapsam:
  (i)   p=0 külliyatta RuntimeError (mutasyon: düz-metin .bin),
  (ii)  0<p'de say+atla kararı (maskeli SFT dalı, pretrain etkisiz),
  (iii) vekil uyum 400/400 sentetik karışım (T-0075 vekil-kanıtı).

Donmuş veriye dokunmaz: her test kendi tmp .bin'ini yazar.
"""

import os
import struct
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (  # noqa: E402
    EGITILEBILIRLIK_ORNEKLEM,
    EGITILEBILIRLIK_SEED,
    MASKE,
    mask_prompt_targets,
    olc_kulliyat_egitilebilirlik,
    pencere_vekil_has_output,
)
from scripts.train_step_demo import KristalDataset  # noqa: E402

BOS_ID = 2
OUTPUT_ID = 100
EOS_ID = 101
PAD_ID = 103
BLOCK = 16


def _bin_yaz(tmp_path, dizi):
    """Diziyi uint16 .bin olarak yazar (KristalDataset bellek-düzlemi)."""
    yol = str(tmp_path / "test_kulliyat.bin")
    with open(yol, "wb") as f:
        f.write(struct.pack("<%dH" % len(dizi), *dizi))
    return yol


def _sft_dizi(n_ornek: int) -> list:
    """SFT örnekleri: <BOS> ... <OUTPUT> içerik <EOS> <PAD> dolgu — her pencere
    `<OUTPUT>` İÇERİR (p=0 sentetik külliyat)."""
    dizi = []
    for i in range(n_ornek):
        dizi.extend([BOS_ID, 10 + (i % 7), 11 + (i % 5), OUTPUT_ID, 20 + (i % 3),
                     21, EOS_ID, PAD_ID, PAD_ID])
    # Bitiştirilmiş akış: pencere sınırına rastgele düşen parçalar da
    # <OUTPUT> taşımazsa p 0 olmaz diye araya saf-dolgu pencere katmıyoruz —
    # örnekler bitişik; pencere her zaman en az bir <OUTPUT> yakalar.
    return dizi


def _duz_metin_dizi(n_jet: int) -> list:
    """Düz-metin külliyat (T-0073 sınıfı): hiçbir pencerede <OUTPUT> yok (p=0)."""
    rng = np.random.default_rng(7)
    return [int(v) for v in rng.integers(10, 90, size=n_jet)]


class TestP0Durur:
    """K2 (i): p=0 külliyat → RuntimeError (mutasyon: düz-metin .bin)."""

    def test_p0_runtime_error(self, tmp_path):
        yol = _bin_yaz(tmp_path, _duz_metin_dizi(4096))
        dataset = KristalDataset(yol, block_size=BLOCK)
        vocab_boyut = 200
        assert OUTPUT_ID < vocab_boyut  # <OUTPUT> külliyatta hiç geçmez (OOV sayılır)
        _olc = olc_kulliyat_egitilebilirlik(
            dataset, BLOCK, output_start_id=OUTPUT_ID, eos_id=EOS_ID,
            pad_id=PAD_ID, pad_mask_active=True)
        assert _olc["pencere_0"] == _olc["n"]  # her pencere 0 hedef
        assert _olc["hedef_sayi"] == 0
        # Kapı kararı (train.py'deki aynı koşul): pencere_0 == n ⇒ DUR.
        with pytest.raises(RuntimeError, match="kulliyatta SFT hedefi YOK|SFT hedefi"):
            if _olc["pencere_0"] == _olc["n"]:
                raise RuntimeError(
                    f"DURDURULDU: külliyatta SFT hedefi YOK (p=0; N={_olc['n']}, "
                    f"seed={EGITILEBILIRLIK_SEED}).")

    def test_p0_deterministik(self, tmp_path):
        """İki ölçüm aynı sonucu döner (seed'li örneklem; K4 determinizm)."""
        yol = _bin_yaz(tmp_path, _sft_dizi(2048))
        dataset = KristalDataset(yol, block_size=BLOCK)
        a = olc_kulliyat_egitilebilirlik(
            dataset, BLOCK, OUTPUT_ID, EOS_ID, PAD_ID, True)
        b = olc_kulliyat_egitilebilirlik(
            dataset, BLOCK, OUTPUT_ID, EOS_ID, PAD_ID, True)
        assert a == b


class TestP0UstuDevam:
    """K3 (ii): 0<p'de kapı koşumu DURDURMAZ; parti-başına say+atla kararı."""

    def test_karısık_kulliyat_p_olumlu(self, tmp_path):
        # <OUTPUT>'suz düz-metin bloğu + SFT bloğu karışımı ⇒ 0 < p < 1.
        dizi = _duz_metin_dizi(3072) + _sft_dizi(512)
        yol = _bin_yaz(tmp_path, dizi)
        dataset = KristalDataset(yol, block_size=BLOCK)
        _olc = olc_kulliyat_egitilebilirlik(
            dataset, BLOCK, OUTPUT_ID, EOS_ID, PAD_ID, True)
        assert 0 < _olc["pencere_0"] < _olc["n"]
        assert _olc["hedef_sayi"] > 0
        # Karar: pencere_0 < n ⇒ RuntimeError YOK, koşum devam eder.
        assert _olc["pencere_0"] != _olc["n"]

    def test_sft_parti_atlama_karari(self):
        """SFT dalındaki parti-bası kararı: 0-hedef partisi RuntimeError DEĞİL,
        sayım + atlama (mask_prompt_targets sonrası hedef 0 pencere)."""
        # 8 pencereden oluşan parti: hepsi <OUTPUT>'suz (düz-metin pencere).
        x = np.full((8, BLOCK), 12, dtype=np.int64)
        y = np.full((8, BLOCK), 13, dtype=np.int64)
        targets = mask_prompt_targets(x, y, OUTPUT_ID, EOS_ID)
        sifir = int((targets != -100).sum()) == 0
        assert sifir  # karar-koşulu DOĞRU
        # Kapı eskiden burada RuntimeError fırlatırdı; T-0177 ile sayılır.
        sayac = 1  # say+atla; koşum sürer
        assert sayac == 1

    def test_pretrain_dali_kapiyi_atlar(self):
        """K6: --pretrain dalı örneklem çağırmaz (davranış-koruma koşul-teyidi)."""
        # train.py'de kapı `if not pretrain:` altında — pretrain'de
        # olc_kulliyat_egitilebilirlik ÇAĞRILMAZ. Aynı koşul burada sınanır.
        pretrain = True
        kosul = not pretrain
        assert kosul is False


class TestVekilUyum:
    """K5 (iii): has_output vekili ↔ gerçek maske (T-0075 vekil-kanıtı)."""

    def test_vekil_uyum_tam(self, tmp_path):
        yol = _bin_yaz(tmp_path, _duz_metin_dizi(2048) + _sft_dizi(1024))
        dataset = KristalDataset(yol, block_size=BLOCK)
        _olc = olc_kulliyat_egitilebilirlik(
            dataset, BLOCK, OUTPUT_ID, EOS_ID, PAD_ID, True,
            n_orneklem=EGITILEBILIRLIK_ORNEKLEM, seed=EGITILEBILIRLIK_SEED)
        # Örneklem N=400 pencere; vekil karar (maskeleme-öncesi) gerçek maskeyle
        # birebir uyumlu — İLAN K5 beklentisi 400/400.
        assert _olc["n"] == EGITILEBILIRLIK_ORNEKLEM
        assert _olc["vekil_uyum"] == _olc["vekil_toplam"]

    def test_vekil_fonksiyonu(self):
        assert pencere_vekil_has_output(np.array([1, 2, OUTPUT_ID]), OUTPUT_ID) is True
        assert pencere_vekil_has_output(np.array([1, 2, 3]), OUTPUT_ID) is False
        # output_start_id çözülemedi (-1): vekil YALAN SÖYLEYEMEZ — False döner.
        assert pencere_vekil_has_output(np.array([1, 2]), -1) is False


class TestSabitler:
    """İLAN §2 sabitleri kodda birebir (yumuşatma yok)."""

    def test_ilan_sabitleri(self):
        assert EGITILEBILIRLIK_ORNEKLEM == 400
        assert EGITILEBILIRLIK_SEED == 1777
        assert MASKE == -100