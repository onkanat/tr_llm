#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_v5_terfi.py — V5 ÖĞRENCİ TERFİSİ (2026-11-07, kullanıcı talimatı: "B")
=============================================================================
Terfi hedefi: canlı gateway TABANI `anka_taban_pure.pt` → **`anka_ogrenci_v5.pt`**
(v4/v4c öğrenci serisinin sonuncusu, serinin en iyisi).

AÇIK İSTİSNA (hüküm — bu uyarı KAPANMAZ): v5 üç donmuş kapıdan YALNIZ K1'i geçti.
- K1 okuryazarlık: 2,221915 ≤ 2,45 **GEÇTİ** (seri rekoru; pure 2,304304)
- K2 bilgi: 3,2636 > 3,23 **KALDI** (0,0336 nat eksik; pure 3,8205 → −0,5569)
- K3 nöbet: 4/4 top-1 ama PLURAL 0,9696 < 0,99 **KALDI** (0,0219 eksik)
Kanıt: data/eval/v5_ogrenci_hukum_2026-11-07.md · makine-okur istisna kaydı:
data/eval/v5_terfi_istisna_2026-11-07.json · terfi ölçümleri:
data/eval/v5_terfi_2026-11-07.md (router v3 + v5 taban: 8 sorgunun 7'sinde birincil
sınıf korundu, 1 pedagoji sorgusu pedagogy→gardener kaydı).

Kural: eşikler KAYDIRILMAZ; istisna AÇIKÇA yazılır ve bu test onu zorlar.
"""

import hashlib
import json
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5_MODEL = os.path.join(REPO_KOKU, "data", "anka_ogrenci_v5.pt")
PURE_MODEL = os.path.join(REPO_KOKU, "data", "anka_taban_pure.pt")
ISTISNA_JSON = os.path.join(REPO_KOKU, "data", "eval", "v5_terfi_istisna_2026-11-07.json")
VOCAB_JSON = os.path.join(REPO_KOKU, "data", "rebuild", "vocab_anka_r1_33114.json")

# Hüküm çıpası: v5 koşum çıktısının sha256'sı (data/eval/v5_ogrenci_hukum_2026-11-07.json)
BEKLENEN_V5_SHA = "1eded7fa2c4a5337496a890c01b683b74133191433f69b23853a70ee1f7e3904"
# Geri-dönüş çıpası: pure DEĞİŞMEDİ (terfi geri alınabilir olmalı)
BEKLENEN_PURE_SHA = "5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360"


def _sha256(yol):
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def _state_dict(yol):
    """v5/pure checkpoint'i DÜZ state_dict'tir (tensörler); weights_only=True yeterli
    ve keyfî pickle yüklemeyi kapatır (güvenlik)."""
    ck = torch.load(yol, map_location="cpu", weights_only=True)
    return ck.get("model_state_dict", ck)


class TestV5Terfi(unittest.TestCase):
    """Canlı yığın tabanı v5'e terfi etti; istisna kaydı ve çıpalar zorunlu."""

    def test_baslat_taban_varsayilani_v5(self):
        """baslat_canli_gateway --model varsayılanı v5 (pure değil)."""
        import scripts.baslat_canli_gateway as bcg
        self.assertEqual(bcg.DEFAULT_BASE_PATH, "data/anka_ogrenci_v5.pt")
        self.assertEqual(bcg.BEKLENEN_BASE_SHA, BEKLENEN_V5_SHA)

    def test_argparse_model_varsayilani_tabana_bagli(self):
        """--model argparse varsayılanı DEFAULT_BASE_PATH'e BAĞLI (sabit metin değil)."""
        yol = os.path.join(REPO_KOKU, "scripts", "baslat_canli_gateway.py")
        kaynak = open(yol, encoding="utf-8").read()
        self.assertIn("default=DEFAULT_BASE_PATH", kaynak,
                      "--model argparse varsayılanı DEFAULT_BASE_PATH'e bağlı olmalı")

    def test_taban_sha_cipasi_gercek_dosyayla_uyusuyor(self):
        """v5 çıpası gerçek dosyadan ölçülür (elle yazım yok); pure yerinde durur."""
        self.assertTrue(os.path.exists(V5_MODEL), "v5 checkpoint yok")
        self.assertEqual(_sha256(V5_MODEL), BEKLENEN_V5_SHA)
        self.assertEqual(_sha256(PURE_MODEL), BEKLENEN_PURE_SHA,
                         "pure DEĞİŞMEMELİ — geri-dönüş çıpası")

    def test_v5_yuklenebilir_ve_sozluk_uyumlu(self):
        """Artifact-guard: v5 state_dict yüklenir, embedding satırı == sözlük (33114)."""
        sd = _state_dict(V5_MODEL)
        satir = next((tuple(v.shape)[0] for k, v in sd.items()
                      if k.endswith("embedding.embedding.weight")), None)
        self.assertEqual(satir, 33114, f"v5 embedding satırı {satir} != 33114")
        self.assertIn("lm_head.weight", sd, "v5 state_dict beklenen şemada değil")
        sozluk = json.load(open(VOCAB_JSON, encoding="utf-8"))
        self.assertEqual(len(sozluk["stoi"]), 33114)
        # pure ile AYNI şema (terfi geri alınabilir; anahtar kümesi değişmemeli)
        self.assertEqual(set(sd.keys()), set(_state_dict(PURE_MODEL).keys()),
                         "v5 ve pure state_dict anahtar şemaları farklı")

    def test_istisna_kaydi_var_ve_kapilari_dogru_yaziyor(self):
        """Makine-okur istisna kaydı K2/K3'ün KALDI olduğunu açıkça taşır."""
        self.assertTrue(os.path.exists(ISTISNA_JSON), "istisna kaydı yok")
        kayit = json.load(open(ISTISNA_JSON, encoding="utf-8"))
        self.assertEqual(kayit["model"], "data/anka_ogrenci_v5.pt")
        self.assertEqual(kayit["model_sha256"], BEKLENEN_V5_SHA)
        kapilar = kayit["kapilar"]
        self.assertEqual(kapilar["K1"]["hukum"], "GECTI")
        self.assertEqual(kapilar["K2"]["hukum"], "KALDI")
        self.assertEqual(kapilar["K3"]["hukum"], "KALDI")
        self.assertAlmostEqual(kapilar["K2"]["olcum"], 3.2636, places=4)
        self.assertAlmostEqual(kapilar["K2"]["esik"], 3.23, places=4)
        self.assertAlmostEqual(kapilar["K3"]["olcum"], 0.9696, places=4)
        self.assertAlmostEqual(kapilar["K3"]["esik"], 0.99, places=4)
        self.assertEqual(kayit["istisna_turu"], "iki_kapi_istisnasi")
        self.assertTrue(kayit["onay"]["kullanici_talimati"])
        self.assertIn("v5_terfi_2026-11-07.md", kayit["kanit"]["terfi_raporu"])
        self.assertEqual(kayit["geri_donus"]["model"], "data/anka_taban_pure.pt")
        self.assertEqual(kayit["geri_donus"]["model_sha256"], BEKLENEN_PURE_SHA)

    def test_router_v3_ve_uzmanlar_degismedi(self):
        """Terfi YALNIZ tabanı değiştirir; router v3 ve uzman çözümü aynı kalır."""
        import scripts.baslat_canli_gateway as bcg
        self.assertEqual(bcg.DEFAULT_ROUTER_PATH, "data/anka_router_v3.pt")
        cozum = bcg.build_expert_model_paths()
        self.assertEqual(cozum.get("gardener"), "data/anka_bahcivan.pt")
        self.assertEqual(cozum.get("carpenter"), "data/anka_marangoz.pt")
        self.assertEqual(cozum.get("legal"), "data/anka_legal_replay.pt")


if __name__ == "__main__":
    unittest.main()
