"""WIKIPEDIA REPLAY DİLİMİ KAPILARI — her test İKİ DALLI (T-0075).

Bu kural r27/r28'in GEÇERLİLİĞİNİ taşıyor: replay bloğu bir değerlendirme penceresiyle aynı
makaleden gelirse, A CE'deki düşüş **koruma** değil **aynı makaleyi hatırlama** olur ve hüküm
sahte biçimde doğrulanır. Sınanmamış kural sessizce bozulur ⇒ kural KAPIYA bağlanır (T-0097/K15).

Tek dallı sınama kabul edilmez: "her şeye dur diyen sabit" de yeşil görünür. Her testte
(a) kapının ATEŞLEDİĞİ dal, (b) kapının GEÇTİĞİ dal birlikte ölçülür.

Sentetik sınırlar kullanılır (191 MB'lık donmuş `.bin`'e bağımlılık YOK) ⇒ testler hızlı ve
dosya durumundan bağımsız.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pytest

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from scripts.wiki_replay_disjoint import (  # noqa: E402
    B_PENCERE,
    KURAL_BLOK,
    KURAL_MAKALE,
    dilim_bloklari,
    izinli_blok_maskesi,
    kosular,
    makale_numaralari,
    secim_indeksleri,
)

# --- sentetik kurgu -------------------------------------------------------------
# 10 blok = 1280 jeton. Makaleler: [0,256) [256,640) [640,1280)
N_WIN = 10
SINIR_3 = np.array([0, 256, 640, N_WIN * B_PENCERE])
# blok 2 = [256,384) 320'de KESİLİR ⇒ iki makaleye birden değer
SINIR_KESEN = np.array([0, 320, 640, N_WIN * B_PENCERE])


def _izinli(baslar: List[int], kural: str, sinir: np.ndarray) -> List[int]:
    m = izinli_blok_maskesi(baslar, N_WIN, kural, sinir)
    return list(np.flatnonzero(m))


# ---------------------------------------------------------------- kosular
def test_kosular_bosluklari_dogru_verir() -> None:
    """POZİTİF: aradaki boşluk iki koşuya böler. NEGATİF: boşluksuz maske tek koşu."""
    assert kosular(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]
    assert kosular(np.array([True] * 5)) == [(0, 5)]


def test_kosular_bos_maskede_durmaz() -> None:
    """Sınır dalı: hiç izinli blok yoksa koşu listesi BOŞ olmalı (çökmemeli)."""
    assert kosular(np.array([False] * 5)) == []


# ---------------------------------------------------------------- makale numaraları
def test_makale_numaralari_pencerenin_makalesini_bulur() -> None:
    """POZİTİF: pencere 0 ve 1 → makale 0. NEGATİF: pencere 2 → makale 1."""
    assert makale_numaralari([0, 1], SINIR_3) == {0}
    assert makale_numaralari([2], SINIR_3) == {1}


def test_pencere_makale_sinirini_keserse_IKI_makale() -> None:
    """Sınır dalı: blok 2, 320'deki sınırı keser ⇒ HER İKİ makale de dışlanmalı."""
    assert makale_numaralari([2], SINIR_KESEN) == {0, 1}
    # NEGATİF dal: hizalı sınırda aynı pencere TEK makale verir
    assert makale_numaralari([2], SINIR_3) == {1}


# ---------------------------------------------------------------- kural: blok
def test_blok_kurali_yalniz_pencere_bloklarini_atar() -> None:
    """POZİTİF: pencere bloğu düşer. NEGATİF: kardeşi (aynı makalede) KALIR."""
    kalan = _izinli([0], KURAL_BLOK, SINIR_3)
    assert 0 not in kalan, "pencere bloğu düşmeliydi"
    assert 1 in kalan, "blok kuralı aynı makaledeki kardeşi ATMAZ"


# ---------------------------------------------------------------- kural: makale
def test_makale_kurali_ayni_makaleyi_TAMAMEN_atar() -> None:
    """POZİTİF: aynı makaledeki kardeş blok da düşer. NEGATİF: başka makale kalır.

    Bu, kuralın TAŞIDIĞI özelliktir — sınanmazsa kural sessizce 'blok' kuralına dönebilir.
    """
    kalan = _izinli([0], KURAL_MAKALE, SINIR_3)
    assert 0 not in kalan and 1 not in kalan, "makale 0'ın TÜM blokları düşmeliydi"
    assert 2 in kalan, "başka makale (2) korunmalıydı"


def test_makale_kurali_blok_kuralinin_ust_kumesidir() -> None:
    """İlişki iki yönlü: MAKALE'nin attığı küme ⊇ BLOK'un attığı küme, ve burada KESİN büyük."""
    atilan_blok = set(range(N_WIN)) - set(_izinli([0, 5], KURAL_BLOK, SINIR_3))
    atilan_makale = set(range(N_WIN)) - set(_izinli([0, 5], KURAL_MAKALE, SINIR_3))
    assert atilan_blok <= atilan_makale, "MAKALE kuralı BLOK'un attığını da atmalı"
    assert atilan_makale > atilan_blok, "bu kurguda KESİN olarak daha fazla atmalı"
    assert atilan_blok == {0, 5}, "blok kuralı tam olarak pencereyi atmalı"


def test_kesen_pencere_komsu_makaleyi_de_atar() -> None:
    """Sınır dalı: sınırı kesen pencere ⇒ makale kuralı 5 blok atar, blok kuralı 1."""
    assert len(_izinli([2], KURAL_BLOK, SINIR_KESEN)) == N_WIN - 1
    assert len(_izinli([2], KURAL_MAKALE, SINIR_KESEN)) == N_WIN - 5


# ---------------------------------------------------------------- fail-closed
def test_bilinmeyen_kural_DURUR() -> None:
    """POZİTİF: bilinmeyen kural rc=2 ile durur. NEGATİF: bilinen iki kural durmaz."""
    with pytest.raises(SystemExit) as e:
        izinli_blok_maskesi([0], N_WIN, "eval_her_sey_haric", SINIR_3)
    assert e.value.code == 2
    for k in (KURAL_BLOK, KURAL_MAKALE):
        izinli_blok_maskesi([0], N_WIN, k, SINIR_3)


def test_dilim_meta_kuralsiz_DURUR(tmp_path) -> None:
    """Kuralı kayıtlı olmayan dilim denetlenemez ⇒ sessizce varsaymak yerine DURUR."""
    with pytest.raises(SystemExit) as e:
        dilim_bloklari({"cikti": "x", "cakismasiz_blok": 10})
    assert e.value.code == 2
    # NEGATİF dal: kural varsa durmaz (burada `eval_blok_haric` gerçek wiki'yi okur)
    if not os.path.exists("data/anka_a1r_pretrain.bin"):
        pytest.skip("wiki .bin yok — gerçek dosyaya bağlı dal atlandı")
    bloklar = dilim_bloklari({"kural": KURAL_BLOK})
    assert len(bloklar) > 0 and (bloklar.size + 256) == 781250


# ---------------------------------------------------------------- seçim (saf)
def test_secim_indeksleri_karisim_kuranla_ayni() -> None:
    """POZİTİF: r27'nin replay ihtiyacı (2.044) tam çıkar. NEGATİF: farklı oran farklı sayı."""
    idx = secim_indeksleri(784797, 4, 780994)
    assert len(idx) == 2044 and idx[0] == 0 and idx[-1] == 780993
    assert np.all(np.diff(idx) >= 0), "seçim artan olmalı (karışım kuran böyle yapıyor)"
    assert len(secim_indeksleri(784797, 10, 780994)) == (6131 + 8) // 9


def test_secim_replay_every_metadan_okunur_DURUR() -> None:
    """Sabit varsayım yasağı: meta'da `replay_every` yoksa ya da bozuksa rc=2."""
    for bozuk in (None, 1, 0):
        with pytest.raises(SystemExit) as e:
            secim_indeksleri(784797, bozuk, 780994)
        assert e.value.code == 2
    # NEGATİF dal: yeterli blok yoksa da durur (sessiz kırpma yok)
    with pytest.raises(SystemExit) as e:
        secim_indeksleri(784797, 4, 100)
    assert e.value.code == 2
