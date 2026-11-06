#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 öğrenci çekirdek birimleri — scripts/train_v4_ogrenci.py (TDD).

Ağır model YÜKLENMEDEN sınanabilir birimler (karışım oranı, pencere-örnekleme
determinizmi, KL satır-seçimi, val-nöbet düzeni, LR, atomik kayıt) + iki duman:
mikro-parti kayıp denkliği (tiny model ile) ve okula paket-güveni probe'u
(gerçek tokenizer, tiny model). Qdrant YOK; commit YOK (controller işler).
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_v4_ogrenci import (
    BLOCK, KL_CHUNK, SUB_CHUNK,
    karisim_orani_hesapla, planla_pencere_baslar, kl_satirlari_sec,
    nobet_ateslenmeli, V4BatchSampler, V4OgrenciDataset,
    mikro_parti_kayip, paket_guveni_okula, get_lr, kaydet,
)


# --------------------------------------------------------------------------
# 1) Karışım oranı (token-bazlı İLAN beyanının matematiği)
# --------------------------------------------------------------------------

def test_blok_128_aile_tavani():
    """128 jeton = ailenin EĞİTİLMİŞ bağlam tavanı (P1 dersi) — sabit kalmalı."""
    assert BLOCK == 128
    assert (SUB_CHUNK, KL_CHUNK) == (8, 4)  # MPS OOM dersi sabitleri


def test_karisim_orani_hedef_bazli_olcumle_uyumlu():
    """Ölçülen ort 52,86 hedef/kayıt ile 10P+6S → hedef-bazlı ~%80/%20 (İLAN)."""
    k = karisim_orani_hesapla(pencere_satir=10, sft_satir=6, ort_sft_hedef=52.86)
    assert k["hedef_bazli"]["kulliyat"] == pytest.approx(1280 / (1280 + 6 * 52.86), abs=1e-9)
    assert 0.79 < k["hedef_bazli"]["kulliyat"] < 0.81  # ölçüm: %80,1
    assert k["hedef_bazli"]["kulliyat"] + k["hedef_bazli"]["sft"] == pytest.approx(1.0)
    # bağlam-jeton bazlı (satır=128 konum) İLAN'da AYRI beyan edilir — gizli kalmaz
    assert k["baglam_bazli"]["kulliyat"] == pytest.approx(10 / 16)
    assert k["baglam_bazli"]["sft"] == pytest.approx(6 / 16)


def test_karisim_orani_genel_formul():
    """Genel: pencere satırı block hedef, SFT satırı ort_sft_hedef hedef taşır."""
    k = karisim_orani_hesapla(pencere_satir=3, sft_satir=1, ort_sft_hedef=100.0, block=128)
    assert k["hedef_bazli"]["kulliyat"] == pytest.approx(384 / 484)
    k2 = karisim_orani_hesapla(pencere_satir=1, sft_satir=1, ort_sft_hedef=0.0)
    assert k2["hedef_bazli"]["kulliyat"] == 1.0  # sıfır-hedef SFT uç değeri


# --------------------------------------------------------------------------
# 2) Pencere-örnekleme determinizmi (memmap rastgele 128-pencereler)
# --------------------------------------------------------------------------

def test_pencere_ornekleme_determinizmi_ve_sinirlar():
    n_token = 128 * 50 + 77  # 50 tam blok + artık
    a = planla_pencere_baslar(n_token, adet=40, seed=42, epok=0)
    b = planla_pencere_baslar(n_token, adet=40, seed=42, epok=0)
    assert np.array_equal(a, b)                     # aynı (tohum, epok) → aynı akış
    c = planla_pencere_baslar(n_token, adet=40, seed=42, epok=1)
    assert not np.array_equal(a, c)                 # epok değişince akış değişir
    assert len(a) == 40 and len(set(a.tolist())) == 40  # TEKRARSIZ (permütasyon)
    assert all(0 <= int(s) and s % BLOCK == 0 and s + BLOCK + 1 <= n_token for s in a)
    # PencereDataset sınırı birebir: b ∈ [0, n_win-2]
    assert a.max() <= ((n_token // BLOCK) - 2) * BLOCK


def test_pencere_ornekleme_kapasite_kisiti():
    n_token = 128 * 30
    a = planla_pencere_baslar(n_token, adet=10_000, seed=7, epok=0)
    assert len(a) == (n_token // BLOCK) - 1  # istek > havuz → havuz tavanı


# --------------------------------------------------------------------------
# 3) KL satır-seçimi (kaynak tensorü; p26 B1 + v4 tüm-satır)
# --------------------------------------------------------------------------

def test_kl_satir_secimi_beta_kapali():
    kaynak = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert kl_satirlari_sec(kaynak, beta=0.0, tum_satirlar=True).numel() == 0
    assert kl_satirlari_sec(kaynak, beta=-1.0).numel() == 0  # beta<=0 → KL kapalı


def test_kl_satir_secimi_v4_tum_satirlar():
    kaynak = torch.tensor([0.0, 1.0, 1.0, 0.0])
    secim = kl_satirlari_sec(kaynak, beta=0.2, tum_satirlar=True)
    assert secim.tolist() == [0, 1, 2, 3]  # v4: TAM parti — SFT satırları DA çıpalı


def test_kl_satir_secimi_p26_kipi_yalniz_pencere():
    kaynak = torch.tensor([0.0, 1.0, 1.0, 0.0])
    secim = kl_satirlari_sec(kaynak, beta=0.2, tum_satirlar=False)
    assert secim.tolist() == [1, 2]  # p26 B1: kaynak==1 satırları (ölü-kod düzeltmesi)
    hepsi_sft = torch.zeros(6)
    assert kl_satirlari_sec(hepsi_sft, beta=0.2, tum_satirlar=False).numel() == 0


# --------------------------------------------------------------------------
# 3b) v4b KL KİPİ — --kl-kip {tum,sft,pencere}; satır-seçim TEK NOKTADA
# --------------------------------------------------------------------------

def test_kl_kip_sft_yalniz_kaynak_sifir_satirlarinda():
    """v4b SFT kipi: KL kaynak==0 (SFT) satırlarında ateşlenir; kaynak==1 (pencere) ÇIPASIZ."""
    kaynak = torch.tensor([0.0, 1.0, 1.0, 0.0])
    secim = kl_satirlari_sec(kaynak, beta=0.2, kip="sft")
    assert secim.tolist() == [0, 3]  # yalnız SFT satırları — pencere satırları (1, 2) YOK
    hepsi_pencere = torch.ones(6)
    assert kl_satirlari_sec(hepsi_pencere, beta=0.2, kip="sft").numel() == 0


def test_kl_kip_beta_sifir_hicbir_kipte_ateslenmez():
    """β=0 → ÜÇ kipte de satır-seçim BOŞ (KL kapalı; kip beyanı önemsiz)."""
    kaynak = torch.tensor([0.0, 0.0, 1.0, 1.0])
    for kip in ("tum", "sft", "pencere"):
        assert kl_satirlari_sec(kaynak, beta=0.0, kip=kip).numel() == 0


def test_kl_kip_tum_ve_pencere_tanimlari_aynen():
    """Kip uzayı tek noktada: tum (v4) TAM parti; pencere (p26) kaynak==1 — değerler aynen."""
    kaynak = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert kl_satirlari_sec(kaynak, beta=0.2, kip="tum").tolist() == [0, 1, 2, 3]
    assert kl_satirlari_sec(kaynak, beta=0.2, kip="pencere").tolist() == [1, 2]


def test_kl_kip_tum_satirlar_geriye_uyum_kisayolu():
    """Geriye uyum: tum_satirlar=True→tum, False→pencere; kip AÇIK verildiyse KAZANIR."""
    kaynak = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert kl_satirlari_sec(kaynak, 0.2, tum_satirlar=True).tolist() == [0, 1, 2, 3]
    assert kl_satirlari_sec(kaynak, 0.2, tum_satirlar=False).tolist() == [1, 2]
    assert kl_satirlari_sec(kaynak, 0.2, tum_satirlar=True, kip="sft").tolist() == [0, 3]


def test_kl_kip_bilinmeyen_deger_hata():
    """Yanlış kip adı sessiz DEĞİL — açık ValueError."""
    with pytest.raises(ValueError):
        kl_satirlari_sec(torch.tensor([0.0, 1.0]), beta=0.2, kip="yok")


def test_kl_kip_cozum_cli_geriye_uyum():
    """--kl-kip çözümü: varsayılan tum; --no-kl-tum-satirlar → pencere; çelişki → DUR."""
    from types import SimpleNamespace
    from scripts.train_v4_ogrenci import kl_kip_coz
    # hiç bayrak yok → varsayılan aynen tum (v4 geriye uyum)
    assert kl_kip_coz(SimpleNamespace(kl_kip=None, kl_tum_satirlar=None)) == "tum"
    # eski bayrak tek başına: --no-kl-tum-satirlar → pencere (p26), --kl-tum-satirlar → tum
    assert kl_kip_coz(SimpleNamespace(kl_kip=None, kl_tum_satirlar=False)) == "pencere"
    assert kl_kip_coz(SimpleNamespace(kl_kip=None, kl_tum_satirlar=True)) == "tum"
    # yeni bayrak tek başına: sft (v4b)
    assert kl_kip_coz(SimpleNamespace(kl_kip="sft", kl_tum_satirlar=None)) == "sft"
    # çelişkili ikili → sessiz kısayol zaferi YOK, açık RuntimeError
    with pytest.raises(RuntimeError):
        kl_kip_coz(SimpleNamespace(kl_kip="sft", kl_tum_satirlar=True))   # eski eşleme tum
    with pytest.raises(RuntimeError):
        kl_kip_coz(SimpleNamespace(kl_kip="tum", kl_tum_satirlar=False))  # eski eşleme pencere
    # TUTARLI ikili → geçer
    assert kl_kip_coz(SimpleNamespace(kl_kip="pencere", kl_tum_satirlar=False)) == "pencere"


# --------------------------------------------------------------------------
# 4) Mini-batch karışım düzeni (tam 10P+6S; SFT çapası tekrarlı döngü)
# --------------------------------------------------------------------------

def test_batch_sampler_tam_karisim_ve_kapsama():
    pencere_n, sft_n, batch, p_satir = 100, 13, 16, 10
    s = V4BatchSampler(pencere_n, sft_n, batch, p_satir, seed=42, epok=0)
    partiler = list(iter(s))
    assert len(partiler) == math.ceil(pencere_n / p_satir)  # 10 parti
    gorulen_p, sft_sayilar = set(), []
    for parti in partiler:
        p_idx = [i for i in parti if i < pencere_n]
        s_idx = [i for i in parti if i >= pencere_n]
        assert len(parti) == batch                       # her mini-batch 16 satır
        assert len(p_idx) == p_satir or len(p_idx) == pencere_n % p_satir
        assert len(s_idx) == batch - len(p_idx)
        gorulen_p.update(p_idx)
        assert all(pencere_n <= i < pencere_n + sft_n for i in s_idx)
        sft_sayilar.extend(i - pencere_n for i in s_idx)
    assert gorulen_p == set(range(pencere_n))            # epok kapsama: her pencere 1 kez
    # tekrarlı çopa: her TUR (13'lük bölüm) kendi içinde permütasyon — tekrar
    # yalnız tur SINIRLARININ ötesinde olabilir (tur başına yeniden karıştırma)
    for i in range(0, len(sft_sayilar), sft_n):
        tur = sft_sayilar[i:i + sft_n]
        assert len(set(tur)) == len(tur)


def test_batch_sampler_determinizmi():
    def akis(epok):
        return list(iter(V4BatchSampler(60, 9, 16, 10, seed=42, epok=epok)))
    assert akis(0) == akis(0)  # aynı (seed, epok) → birebir aynı partiler
    assert akis(0) != akis(1)  # epok akışı ayrışır (pencere permütasyonu + SFT tur tohumu)
    assert V4BatchSampler(60, 9, 16, 10, seed=7, epok=0).seed == 7


def test_batch_sampler_girdi_denetimi():
    with pytest.raises(ValueError):
        V4BatchSampler(100, 10, 16, 16, seed=1, epok=0)  # pencere_satir < batch
    with pytest.raises(ValueError):
        V4BatchSampler(5, 10, 16, 10, seed=1, epok=0)    # pencere_n < pencere_satir


# --------------------------------------------------------------------------
# 5) LR tasarımı (warmup + cosine; p26 formu birebir)
# --------------------------------------------------------------------------

def test_get_lr_warmup_ve_cosine():
    peak, min_lr, warmup, toplam = 5e-5, 1e-5, 200, 11380
    assert get_lr(0, toplam, peak, min_lr, warmup) == pytest.approx(peak * 1 / warmup)
    assert get_lr(warmup - 1, toplam, peak, min_lr, warmup) == pytest.approx(peak)
    assert get_lr(warmup, toplam, peak, min_lr, warmup) == pytest.approx(peak)  # cosine tepesi
    assert get_lr(toplam - 1, toplam, peak, min_lr, warmup) <= min_lr + 1e-12  # anneal sonu
    for s in (300, 2000, 8000):
        assert min_lr < get_lr(s, toplam, peak, min_lr, warmup) < peak
    # sıcak-kısıtlı cosinete monoton düşüş
    assert get_lr(8000, toplam, peak, min_lr, warmup) < get_lr(3000, toplam, peak, min_lr, warmup)


# --------------------------------------------------------------------------
# 6) Val-nöbet düzeni (save-every katları; final ayrı)
# --------------------------------------------------------------------------

def test_nobet_duzeni():
    toplam = 11380
    for adim in (500, 1000, 11000):
        assert nobet_ateslenmeli(adim, 500, toplam) is True
    for adim in (499, 300, 740):
        assert nobet_ateslenmeli(adim, 500, toplam) is False
    # toplam-adım KATI olsa bile ARA nöbet atlanır (final nöbet döngü sonrası)
    assert nobet_ateslenmeli(11380, 500, toplam) is False
    assert nobet_ateslenmeli(1000, 500, 1000) is False
    assert nobet_ateslenmeli(500, 0, toplam) is False  # save-every=0 → nöbet yok


# --------------------------------------------------------------------------
# 7) Atomik kayıt (.tmp + os.replace)
# --------------------------------------------------------------------------

def test_kaydet_atomik(tmp_path):
    hedef = str(tmp_path / "model.pt")
    m = torch.nn.Linear(4, 4)
    kaydet(m, hedef)
    assert os.path.exists(hedef) and not os.path.exists(hedef + ".tmp")
    yuklenen = torch.load(hedef, weights_only=True)
    assert torch.equal(yuklenen["weight"], m.weight)
    m.weight.data += 1.0
    kaydet(m, hedef)
    assert torch.equal(torch.load(hedef, weights_only=True)["weight"], m.weight)


# --------------------------------------------------------------------------
# 8) DUMAN — mikro-parti kayıp denkliği (tiny model; p26 not-6/7 matematiği)
# --------------------------------------------------------------------------

class MiniLM(torch.nn.Module):
    """KristalLM arayüzü (x → (logits, None)) — ağır model YOK."""

    def __init__(self, V, d=24, sayac=None):
        super().__init__()
        self.emb = torch.nn.Embedding(V, d)
        self.head = torch.nn.Linear(d, V)
        self.sayac = sayac  # öğretmen çağrı sayacı (KL-kanıt testi)

    def forward(self, x, targets=None, sign_mask=None):
        if self.sayac is not None:
            self.sayac["cagri"] += 1
            self.sayac["satir"] += int(x.size(0))
        return self.head(self.emb(x)), None


def _rastgele_parti(V, T=32, B=16, tohum=0):
    g = torch.Generator().manual_seed(tohum)
    x = torch.randint(0, V, (B, T), generator=g)
    y = torch.randint(0, V, (B, T), generator=g)
    y[:, :T // 2] = -100  # prompt konumları (SFT deseni)
    sm = torch.where(y == -100, torch.zeros(()), torch.ones(()))
    sm[3, 5] = -1.0  # işaret maskesi: ±1 (kayıp maskesi DEĞİL)
    kaynak = torch.tensor([0.0] * (B // 2) + [1.0] * (B - B // 2))
    return x, y, sm, kaynak


def _mini_cift(V, tohum=1):
    torch.manual_seed(tohum)
    ogrenci = MiniLM(V)
    torch.manual_seed(tohum + 100)
    hoca = MiniLM(V)
    for p_ in hoca.parameters():
        p_.requires_grad = False
    return ogrenci, hoca


def test_mikro_parti_denkligi_deger_ve_gradyan():
    """SUB_CHUNK/KL_CHUNK alt-yığınları TAM parti kaybına VE gradyanına eşit (not-7)."""
    V, beta = 40, 0.2
    x, y, sm, kaynak = _rastgele_parti(V)
    sonuclar, gradlar = [], []
    for sub, kl in ((SUB_CHUNK, KL_CHUNK), (9999, 9999)):
        ogrenci, hoca = _mini_cift(V)
        ogrenci.zero_grad()
        r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, beta,
                             tum_satirlar=True, accum=1, sub_chunk=sub, kl_chunk=kl)
        sonuclar.append(r)
        gradlar.append([p.grad.clone() for p in ogrenci.parameters()])
    assert sonuclar[0]["loss"] == pytest.approx(sonuclar[1]["loss"], rel=1e-5, abs=1e-6)
    assert sonuclar[0]["kl"] == pytest.approx(sonuclar[1]["kl"], rel=1e-4, abs=1e-7)
    for g1, g2 in zip(gradlar[0], gradlar[1]):
        assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-6)


def test_mikro_parti_kl_kanit_tum_satir_ve_beta_kapali():
    """v4: KL TÜM mikro-partilerde ateşlenir; beta=0 → öğretmen HİÇ çağrılmaz."""
    V, beta = 40, 0.2
    x, y, sm, kaynak = _rastgele_parti(V)
    kaynak = torch.zeros_like(kaynak)  # TAMAMI SFT satırı — v4'de yine de KL ateşlenir
    ogrenci, hoca = _mini_cift(V)
    sayac = {"cagri": 0, "satir": 0}
    hoca.sayac = sayac
    r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, beta, tum_satirlar=True,
                          accum=1, sub_chunk=8, kl_chunk=4)
    assert r["kl_aktif"] is True and r["kl_satir"] == 16
    assert sayac["satir"] == 16  # tüm-satır: öğretmen TAM 16 satırı gördü
    # beta=0: KL kapalı, öğretmen çağrılmaz
    ogrenci2, hoca2 = _mini_cift(V)
    sayac2 = {"cagri": 0, "satir": 0}
    hoca2.sayac = sayac2
    r2 = mikro_parti_kayip(ogrenci2, hoca2, x, y, sm, kaynak, 0.0, tum_satirlar=True, accum=1)
    assert r2["kl_aktif"] is False and r2["kl"] is None
    assert sayac2["cagri"] == 0


def test_mikro_parti_p26_kipi_yanliz_pencere():
    """p26 kipi (tum_satirlar=False): öğretmen YALNIZ kaynak==1 satırlarında."""
    V = 40
    x, y, sm, kaynak = _rastgele_parti(V)  # 8 SFT + 8 pencere satırı
    ogrenci, hoca = _mini_cift(V)
    sayac = {"cagri": 0, "satir": 0}
    hoca.sayac = sayac
    r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.2,
                          tum_satirlar=False, accum=1, sub_chunk=8, kl_chunk=4)
    assert r["kl_aktif"] is True and r["kl_satir"] == 8  # kaynak==1 satır sayısı
    assert sayac["satir"] == 8
    hepsi_sft = torch.zeros_like(kaynak)
    r2 = mikro_parti_kayip(ogrenci, hoca, x, y, sm, hepsi_sft, 0.2,
                           tum_satirlar=False, accum=1)
    assert r2["kl_aktif"] is False


def test_mikro_parti_sft_kipi_ogretmen_yalniz_sft_satirlarinda():
    """v4b duman: kip='sft' → öğretmen YALNIZ kaynak==0 satırlarında; TÜM-değil; denklik aynen."""
    V, beta = 40, 0.2
    x, y, sm, kaynak = _rastgele_parti(V)  # 8 SFT (kaynak=0) + 8 pencere (kaynak=1)
    ogrenci, hoca = _mini_cift(V)
    sayac = {"cagri": 0, "satir": 0}
    hoca.sayac = sayac
    r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, beta, kip="sft",
                          accum=1, sub_chunk=8, kl_chunk=4)
    assert r["kl_aktif"] is True and r["kl_satir"] == 8  # 8 SFT satırı — 16 DEĞİL (TÜM-değil)
    assert sayac["satir"] == 8  # öğretmen pencere satırlarını GÖRMEDİ
    # TAMAMI pencere satırı → SFT kipinde KL HİÇ ateşlenmez (çıpa yalnız öğrencinin bildiğinde)
    hepsi_pencere = torch.ones_like(kaynak)
    r2 = mikro_parti_kayip(ogrenci, hoca, x, y, sm, hepsi_pencere, beta, kip="sft", accum=1)
    assert r2["kl_aktif"] is False and r2["kl"] is None
    # sub-chunk denkliği kip='sft'te de TAM-seçimle eşit (değer VE gradyan)
    sonuclar, gradlar = [], []
    for sub, kl in ((SUB_CHUNK, KL_CHUNK), (9999, 9999)):
        o2, h2 = _mini_cift(V)
        o2.zero_grad()
        rr = mikro_parti_kayip(o2, h2, x, y, sm, kaynak, beta, kip="sft",
                              accum=1, sub_chunk=sub, kl_chunk=kl)
        sonuclar.append(rr)
        gradlar.append([p.grad.clone() for p in o2.parameters()])
    assert sonuclar[0]["loss"] == pytest.approx(sonuclar[1]["loss"], rel=1e-5, abs=1e-6)
    assert sonuclar[0]["kl"] == pytest.approx(sonuclar[1]["kl"], rel=1e-4, abs=1e-7)
    for g1, g2 in zip(gradlar[0], gradlar[1]):
        assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# 9) DUMAN — V4OgrenciDataset pencere okuma (sentez bin; lazy memmap)
# --------------------------------------------------------------------------

class _SapEmb:
    def compute_sign_mask(self, x):
        return torch.ones(*x.shape, 1)


class _SapModel:
    embedding = _SapEmb()


class _SapSFT:
    """FastSampleAlignedDataset ördek-tipi: (x, y, sm) üçlüsü."""

    def __init__(self, n, T):
        self.ornekler = [(torch.arange(n * T + i, n * T + i + T) % 97,
                          torch.arange(n * T + i + 1, n * T + i + T + 1) % 97,
                          torch.ones(T, 1)) for i in range(n)]

    def __len__(self):
        return len(self.ornekler)

    def __getitem__(self, idx):
        return self.ornekler[idx]


def test_v4_dataset_pencere_satiri(tmp_path):
    bin_yol = str(tmp_path / "mini.bin")
    tokenler = (np.arange(128 * 12) % 50).astype(np.uint16)  # 12 blok
    tokenler.tofile(bin_yol)
    baslar = np.array([0, 128, 256], dtype=np.int64)
    sft = _SapSFT(4, 32)
    ds = V4OgrenciDataset(sft, bin_yol, baslar, _SapModel())
    assert len(ds) == 3 + 4
    # pencere satırı: y-shift BİREBİR, kaynak=1.0, sm ±1 şekli
    x, y, sm, kaynak = ds[1]
    assert list(x.numpy()) == tokenler[128:256].tolist()
    assert list(y.numpy()) == tokenler[129:257].tolist()  # y[i] = token[i+1]
    assert kaynak.item() == 1.0 and sm.shape == (128, 1)
    # SFT satırı: ördek-tip geçiş, kaynak=0.0
    x2, y2, sm2, kaynak2 = ds[3]
    assert kaynak2.item() == 0.0
    assert torch.equal(x2, sft[0][0])


# --------------------------------------------------------------------------
# 10) DUMAN — okula paket güveni probe'u (gerçek tokenizer, tiny model)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gercek_tokenizer():
    kök = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, kök)
    os.chdir(kök)
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary
    lexicon = LexiconManager(); lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary(); vocab.load("data/rebuild/vocab_anka_r1_33114.json")
    return KristalTokenizer(compiler, vocab), vocab


def test_paket_guveni_okula_probe(gercek_tokenizer):
    tokenizer, vocab = gercek_tokenizer
    torch.manual_seed(5)
    mini = MiniLM(len(vocab.stoi), d=16)
    sonuc = paket_guveni_okula(mini, tokenizer, vocab, torch.device("cpu"))
    assert sonuc["beklenen"] == "case"
    assert isinstance(sonuc["top1"], str) and len(sonuc["top1"]) > 0
    assert 0.0 <= sonuc["guven"] <= 1.0
    assert isinstance(sonuc["dogru"], bool)
    assert mini.training is True  # ölçüm öncesi train'deydi → kapanışta GERİ DÖNER
    mini.eval()
    paket_guveni_okula(mini, tokenizer, vocab, torch.device("cpu"))
    assert mini.training is False  # eval'deydi → eval kalır (kip korumalı)


# --------------------------------------------------------------------------
# 3c) v4c PENCERE-KAYIP KİPİ — --pencere-kayip {sm,plain} (tek değişken:
#     yalnız pencere satırlarının KAYIP AĞIRLIĞI; mimari ve SFT formu aynen)
# --------------------------------------------------------------------------

def test_pencere_kayip_cli_varsayilan_sm_ve_gecersiz_hata(monkeypatch):
    """--pencere-kayip: varsayılan sm (v4b geriye uyum — eski komut birebir koşar),
    plain kabul; bilinmeyen değer argparse'dan açık SystemExit."""
    from scripts.train_v4_ogrenci import parse_args
    monkeypatch.setattr(sys, "argv", ["train_v4_ogrenci.py", "--save-path", "scratch/x.pt"])
    assert parse_args().pencere_kayip == "sm"
    monkeypatch.setattr(sys, "argv", ["train_v4_ogrenci.py", "--save-path", "scratch/x.pt",
                                      "--pencere-kayip", "plain"])
    assert parse_args().pencere_kayip == "plain"
    monkeypatch.setattr(sys, "argv", ["train_v4_ogrenci.py", "--save-path", "scratch/x.pt",
                                      "--pencere-kayip", "yok"])
    with pytest.raises(SystemExit):
        parse_args()


def test_pencere_kayip_plain_pencere_satiri_standart_ortalama():
    """v4c ÖZÜ: plain kipte pencere satırı kaybı = STANDART ORTALAMA CE (geçerli
    hedefler üzerinde); sm kipi aynı girdiyle FARKLI değer verir (dal değişti)."""
    V, B, T = 40, 8, 32
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, V, (B, T), generator=g)
    y = torch.randint(0, V, (B, T), generator=g)  # pencere: TÜM konumlar geçerli
    sm = torch.ones(B, T)
    sm[:, ::3] = -1.0  # ±1 işaret maskesi (pencerelerde negasyon bölgeleri)
    kaynak = torch.ones(B)  # TAMAMI pencere satırı
    ogrenci, hoca = _mini_cift(V)
    r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                          pencere_kayip="plain", accum=1, sub_chunk=9999)
    with torch.no_grad():
        logits, _ = ogrenci(x)
        beklenen = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    assert r["loss"] == pytest.approx(float(beklenen), rel=1e-6)
    assert r["pencere_ce"] == pytest.approx(float(beklenen), rel=1e-6)
    r_sm = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                             pencere_kayip="sm", accum=1, sub_chunk=9999)
    assert r_sm["loss"] != pytest.approx(float(beklenen), rel=1e-6)  # ±1 ağırlık farkı


def test_pencere_kayip_sft_satiri_formu_degismez():
    """TAMAMI SFT satırı: plain kipi sm kipine değer VE gradyan bakımından BİREBİR
    (iki değişken birden değişmez — SFT dalı dokunulmaz)."""
    V = 40
    x, y, sm, kaynak = _rastgele_parti(V)
    kaynak = torch.zeros_like(kaynak)  # TAMAMI SFT satırı
    sonuclar, gradlar = [], []
    for kip_k in ("sm", "plain"):
        ogrenci, hoca = _mini_cift(V)
        ogrenci.zero_grad()
        r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                              pencere_kayip=kip_k, accum=1, sub_chunk=9999)
        sonuclar.append(r)
        gradlar.append([p.grad.clone() for p in ogrenci.parameters()])
    assert sonuclar[0]["loss"] == pytest.approx(sonuclar[1]["loss"], rel=1e-9, abs=0.0)
    assert sonuclar[1]["pencere_ce"] is None  # pencere satırı yok → bileşen yok
    for g1, g2 in zip(gradlar[0], gradlar[1]):
        assert torch.equal(g1, g2)


def test_pencere_kayip_karisik_parti_el_formulu():
    """Karışık parti (8 SFT + 8 pencere): plain = [Σ_p CE×geçerli + Σ_s CE×sm] /
    [Σ_p geçerli + Σ_s sm] — pencere dalı geçerli-gösterge, SFT dalı sm AYNEN;
    sm kipi formülü de elle doğrulanır (v4b BİREBİR korunur)."""
    V = 40
    x, y, sm, kaynak = _rastgele_parti(V)  # 8 SFT + 8 pencere; y'nin ilk yarısı -100
    ogrenci, hoca = _mini_cift(V)
    with torch.no_grad():
        logits, _ = ogrenci(x)
        ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").view_as(sm)
        gecerli = (y != -100).to(sm.dtype).reshape(sm.shape)
        kose = (kaynak == 1.0).view(-1, 1)  # sm (B,T) → satır maskesi (B,1) broadcast
        w_plain = torch.where(kose, gecerli, sm)
        beklenen_plain = float((ce * w_plain).sum() / (w_plain.sum() + 1e-8))
        ag = kose.to(sm.dtype)
        beklenen_penc = float((ce * w_plain * ag).sum() / (w_plain * ag).sum())
        beklenen_sm = float((ce * sm).sum() / (sm.sum() + 1e-8))
    r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                          pencere_kayip="plain", accum=1, sub_chunk=9999)
    assert r["loss"] == pytest.approx(beklenen_plain, rel=1e-6)
    assert r["pencere_ce"] == pytest.approx(beklenen_penc, rel=1e-6)
    assert r["pencere_ce"] >= 0.0  # pencere dalı: geçerli hedeflerde standart ortalama
    r_sm = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                             pencere_kayip="sm", accum=1, sub_chunk=9999)
    assert r_sm["loss"] == pytest.approx(beklenen_sm, rel=1e-6)


def test_pencere_kayip_plain_dal_negatif_imkansiz():
    """ÖLÇEK: plain dalın pencere bileşeni ≥0 HER ZAMAN — adversarial −1-yoğun
    işaret maskesinde bile (negatif İMKÂNSIZ; patoloji testi)."""
    V = 40
    for tohum in range(6):
        x, y, sm, kaynak = _rastgele_parti(V, tohum=tohum)
        sm = sm.clone()
        sm[kaynak == 1.0] = -1.0       # pencere satırları: ağır −1 işaret maskesi
        sm[kaynak == 1.0, ::2] = 1.0
        ogrenci, hoca = _mini_cift(V, tohum=10 + tohum)
        r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                              pencere_kayip="plain", accum=1, sub_chunk=8)
        assert r["pencere_ce"] is not None
        assert r["pencere_ce"] >= 0.0


class _SabitLogitLM(torch.nn.Module):
    """Pencere-patolojisi testi için sabit-logit model: konum t'de hedef (t%V) ~0 nat,
    başka hedef ~10 nat — CE'nin konumlar arası yayılımını DETERMİNİSTİK kurar
    (parametreleri 0.0×katkıyla grafte tutar: backward geçerli)."""

    def __init__(self, V, d=8):
        super().__init__()
        self.emb = torch.nn.Embedding(V, d)
        self.head = torch.nn.Linear(d, V)
        self.V = V

    def forward(self, x, targets=None, sign_mask=None):
        B, T = x.shape
        sabit = torch.zeros(T, self.V, device=x.device)
        sabit[torch.arange(T), torch.arange(T) % self.V] = 10.0
        logits = sabit.unsqueeze(0) + 0.0 * self.head(self.emb(x))
        return logits, None


def test_pencere_kayip_sm_dali_patoloji_negatif_mumkun():
    """PATOLOJİ KANITI (v4b formu): sm dalında pencere bileşeni NEGATİF olabilir —
    −1 işaretli yüksek-CE konumlar kayıp payından ÇIKARIR. Aynı girdide plain dal
    muaftır (≥0). v4c'nin var olma sebebi budur."""
    V, B, T = 40, 8, 32
    x = torch.zeros(B, T, dtype=torch.long)
    y = (torch.arange(T) % V).unsqueeze(0).expand(B, T).clone()  # ~0 nat her konumda
    y[:, 3] = (3 + 1) % V   # 2B konum ~10 nat (imkânsız hedef)
    y[:, 17] = (17 + 1) % V
    kaynak = torch.ones(B)   # TAMAMI pencere satırı
    sm = torch.ones(B, T)
    sm[:, 3] = -1.0; sm[:, 17] = -1.0  # yüksek-CE konumlar −1 işaretli
    ogrenci, hoca = _SabitLogitLM(V), _SabitLogitLM(V)
    r_sm = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                             pencere_kayip="sm", accum=1, sub_chunk=9999)
    assert r_sm["pencere_ce"] < 0.0  # Σ(ce×sm) < 0 iken Σsm > 0 → negatif (form mümkünü)
    r_plain = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                                pencere_kayip="plain", accum=1, sub_chunk=9999)
    assert r_plain["pencere_ce"] > 0.0  # aynı girdide standart ortalama — patolojiden muaf


class _GommeIsaretKayitLM(torch.nn.Module):
    """KristalLM gömme davranışının kayıtlı taklidi: sign_mask verilmediyse iç hesap
    (KristalEmbedding.forward fallback BİREBİR) — ±1 işaret her forward'da uygulanır."""

    def __init__(self, V, d=24):
        super().__init__()
        self.emb = torch.nn.Embedding(V, d)
        self.head = torch.nn.Linear(d, V)
        self.isaret_kaydi = []

    def compute_sign_mask(self, x):
        return torch.where(x % 2 == 0, 1.0, -1.0).unsqueeze(-1)

    def forward(self, x, targets=None, sign_mask=None):
        if sign_mask is None:
            sign_mask = self.compute_sign_mask(x)  # KristalEmbedding fallback yolu
            kaynak = "ic-hesap"
        else:
            kaynak = "verildi"
        self.isaret_kaydi.append(kaynak)
        return self.head(self.emb(x) * sign_mask), None


def test_pencere_kayip_sm_gomme_girdisi_korunur():
    """v4c plain kipi MİMARİYİ DEĞİŞTİRMEZ: sm gömme girdisi olarak yaşamaya devam
    eder — model her forward'da ±1 işaretini UYGULAR (yalnız KAYIP AĞIRLIĞI değişti)."""
    V = 40
    x, y, sm, kaynak = _rastgele_parti(V)  # 16 satır, sub_chunk=8 → 2 ileri
    ogrenci = _GommeIsaretKayitLM(V)
    hoca = _GommeIsaretKayitLM(V)
    r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, 0.0, kip="sft",
                          pencere_kayip="plain", accum=1, sub_chunk=8)
    assert len(ogrenci.isaret_kaydi) == 2  # alt-yığın disiplini: 16 satır / SUB_CHUNK 8
    assert all(k == "ic-hesap" for k in ogrenci.isaret_kaydi)  # işaret UYGULANDI her ileride
    assert r["loss"] > 0.0  # kayıp hesaplandı (dal live)


def test_pencere_kayip_plain_denklik_deger_ve_gradyan():
    """v4c konfigürasyonu (kip=sft, β=0,2, plain): alt-yığınlar TAM partiye değer VE
    gradyan bakımından eşit; pencere bileşeni de birebir (payda TAM partiden)."""
    V, beta = 40, 0.2
    x, y, sm, kaynak = _rastgele_parti(V)
    sonuclar, gradlar, penc_degerler = [], [], []
    for sub, kl in ((SUB_CHUNK, KL_CHUNK), (9999, 9999)):
        ogrenci, hoca = _mini_cift(V)
        ogrenci.zero_grad()
        r = mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, beta, kip="sft",
                              pencere_kayip="plain", accum=1, sub_chunk=sub, kl_chunk=kl)
        sonuclar.append(r)
        gradlar.append([p.grad.clone() for p in ogrenci.parameters()])
        penc_degerler.append(r["pencere_ce"])
    assert sonuclar[0]["loss"] == pytest.approx(sonuclar[1]["loss"], rel=1e-5, abs=1e-6)
    assert sonuclar[0]["kl"] == pytest.approx(sonuclar[1]["kl"], rel=1e-4, abs=1e-7)
    assert penc_degerler[0] == pytest.approx(penc_degerler[1], rel=1e-5, abs=1e-6)
    for g1, g2 in zip(gradlar[0], gradlar[1]):
        assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# 6b) v4c K1 NÖBET KURALI — HER nöbet noktası (ilk %20 penceresi YOK; v4b dersi)
# --------------------------------------------------------------------------

def test_k1_nobet_kurali_tum_kosum():
    """ERKEN_DURUS_K1 her nöbet noktasında geçerli: adım 6000 (eski %20 sınırının
    çok ötesi) ve 11300'te bile SFT CE 2,45 üstü → DURUR; eşik kendisi geçer;
    eksik ölçüm kararsız bırakır."""
    from scripts.train_v4_ogrenci import k1_erken_durus_karar, K1_ESIK
    assert K1_ESIK == 2.45  # İLAN donmuş kapısı donuk
    assert k1_erken_durus_karar({"adim": 500, "sft_val_ce": 2.501}) is True
    assert k1_erken_durus_karar({"adim": 6000, "sft_val_ce": 2.501}) is True   # v4b'de DURMAZDI
    assert k1_erken_durus_karar({"adim": 11300, "sft_val_ce": 2.501}) is True  # koşum sonu yakınında bile
    assert k1_erken_durus_karar({"adim": 6000, "sft_val_ce": 2.45}) is False  # eşik aşımı SERT (>)
    assert k1_erken_durus_karar({"adim": 6000, "sft_val_ce": 2.30}) is False
    assert k1_erken_durus_karar({"adim": 6000, "sft_val_ce": None}) is False   # ölçüm yok → karar yok
