#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-2b — Vektörel Gezgin v2: yetenek_ağı v2 TDD testleri
=======================================================
Kapsam (iş emri 2):
- GÖREV-1 TELEMETRİ: aktivasyon-olay kaydı (JSONL: zaman, sorgu-özeti,
  aktif yetenekler+skorlar, router kararı) + SONUÇ SİNYALİ (zirve yeteneğin
  alanı ↔ router uzmanı: uyum → +1; uyumsuz → −1; router yok / zirve uzmansız
  → kayıt-yalnız, sinyal None).
- GÖREV-2 AĞIRLIK EVRİMİ: deterministik bantlı kural (uyum ×(1+eps),
  uyumsuz ×(1−eps), [0,5; 2,0] bantları), idempotent payload geri-yazımı,
  geçmiş kaydı telemetri JSONL'e, yeniden-oynatma determinizmi.
- GÖREV-3 GATEWAY DANIŞMA (OP-İN, default KAPALI): açıkken zirve+uyum+sinyal
  META'ya yazılır ve telemetriye kaydedilir; uyumsuzluk yalnız LOGLANIR —
  router KARARI (active_expert/router_experts) DEĞİŞMEZ.
- GÖREV-4 KISIT KARTLARI: 'kisit' tipi kartlar (baskılayıcı — çarpan ≤ 1,
  asla yükseltmez), 5 GERÇEK kısıt (repo kuralları, kanıt dosyaları),
  aktivasyon motoru etki = ağırlık×benzerlik×kısıt-çarpanı uygular;
  kısıtsız koleksiyonda sonuç v1 ile BİREBİR.

Kurallar: Qdrant yazımı YALNIZ 'yetenek_agi'; yerel testler
VectorMemory(storage_path=TMPDIR) ile ağ-sız koşar; commit YOK.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.yetenek_agi import (  # noqa: E402
    AGIRLIK_MAX,
    AGIRLIK_MIN,
    DEFAULT_ESIK,
    EVRIM_EPSILON,
    KISIT_TIPI,
    TELEMETRI_YOLU,
    YetenekAgi,
    YetenekKarti,
    ad_kimligi,
    ad_uzmani,
    agirlik_evrimi,
    aktivasyon_seti,
    danisma_raporu,
    etki_hesapla,
    evrim_uygula,
    kart_envanteri,
    kisit_carpanlari,
    kisit_envanteri,
    sonuc_sinyali,
    telemetri_ekle,
    telemetri_oku,
)

REPO_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KARTLAR = {k.kimlik: k for k in kart_envanteri()}
AD_SOZLUGU = {k.kimlik: k.ad for k in kart_envanteri()}


class SahteTokenizer:
    """Birim-test tokenizer'ı: deterministik, sabit token akışı."""

    def encode(self, text):  # noqa: D102
        return [4, 5, 6, 7]

    def decode(self, ids):  # noqa: D102
        return "ahşap bilgi cart zımpara"


def sahte_bellek(eslesmeler, kisitlar=None):
    """dense_recall döndüren mock VectorMemory — Qdrant'a bağlanmaz.

    eslesmeler: [(ad, skor, agirlik, kimlik, etki_alani), ...]
    kisitlar:   [kısıt-metadata dict'leri (tip='kisit'), ...]
    """
    bellek = MagicMock()
    bellek.collection_name = "yetenek_agi"  # YetenekAgi'nin fail-closed ad-kapısı
    kayitlar = [
        {"score": skor, "text": f"yetenek:{ad}",
         "metadata": {"ad": ad, "agirlik": agirlik, "kimlik": kimlik,
                      "etki_alani": alan}}
        for (ad, skor, agirlik, kimlik, alan) in eslesmeler
    ]
    for k in (kisitlar or []):
        kayitlar.append({"score": 0.0, "text": f"kisit:{k.get('kimlik')}",
                         "metadata": dict({"ad": f"Kısıt: {k.get('kimlik')}"},
                                          **k, tip=KISIT_TIPI)})
    bellek.dense_recall.return_value = kayitlar
    return bellek


# ---------------------------------------------------------------------------
# 1) GÖREV-1 — Telemetri: olay kaydı (JSONL) + sonuç sinyali
# ---------------------------------------------------------------------------

class TestTelemetri(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.yol = os.path.join(self.tmp.name, "yetenek_telemetri.jsonl")

    def test_telemetri_ekle_jsonl_yazar_ve_geri_okunur(self):
        olay = {"olay": "aktivasyon", "zaman": "2026-11-05T00:00:00+00:00",
                "sorgu_ozeti": "zımpara nedir", "sinyal": 1}
        r1 = telemetri_ekle(olay, self.yol)
        r2 = telemetri_ekle(olay, self.yol)
        self.assertTrue(os.path.exists(self.yol))
        self.assertEqual(r1["toplam_satir"], 1)
        self.assertEqual(r2["toplam_satir"], 2)  # read-back sayısı doğrulanır
        kayitlar = telemetri_oku(self.yol)
        self.assertEqual(len(kayitlar), 2)
        self.assertEqual(kayitlar[0]["sorgu_ozeti"], "zımpara nedir")

    def test_telemetri_satiri_deterministik_serilestirme(self):
        olay = {"olay": "aktivasyon", "sinyal": -1, "uyum": False}
        telemetri_ekle(olay, self.yol)
        telemetri_ekle(olay, self.yol)
        with open(self.yol, encoding="utf-8") as f:
            satirlar = f.read().splitlines()
        self.assertEqual(satirlar[0], satirlar[1])  # sort_keys ile birebir

    def test_telemetri_oku_bos_dosya(self):
        self.assertEqual(telemetri_oku(self.yol), [])

    def test_sinyal_uyum_pozitif(self):
        aktif = [(AD_SOZLUGU["marangoz_ceketi"], 0.42, 0.525)]
        s = sonuc_sinyali(aktif, router_uzmani="carpenter")
        self.assertEqual(s["zirve_kimligi"], "marangoz_ceketi")
        self.assertEqual(s["zirve_uzmani"], "carpenter")
        self.assertIs(s["uyum"], True)
        self.assertEqual(s["sinyal"], 1)      # uyum → pozitif

    def test_sinyal_uyumsuz_negatif(self):
        aktif = [(AD_SOZLUGU["marangoz_ceketi"], 0.42, 0.525)]
        s = sonuc_sinyali(aktif, router_uzmani="legal")
        self.assertIs(s["uyum"], False)
        self.assertEqual(s["sinyal"], -1)     # uyumsuz → negatif

    def test_sinyal_router_yok_kayit_yalniz(self):
        aktif = [(AD_SOZLUGU["marangoz_ceketi"], 0.42, 0.525)]
        s = sonuc_sinyali(aktif, router_uzmani=None)
        self.assertIsNone(s["uyum"])
        self.assertIsNone(s["sinyal"])        # kayıt-yalnız: ağırlık DEĞİŞMEZ

    def test_sinyal_zirve_uzmansiz_kayit_yalniz(self):
        # zirve kart router uzman kümelerine karşılık gelmiyorsa karşılaştırma
        # zemini yok → kayıt-yalnız (ceza YOK — dürüst None)
        aktif = [(AD_SOZLUGU["vektor_bellek_yonetimi"], 0.26, 0.29)]
        s = sonuc_sinyali(aktif, router_uzmani="grammar_core")
        self.assertIsNone(s["zirve_uzmani"])
        self.assertIsNone(s["sinyal"])

    def test_sinyal_bos_aktivasyon(self):
        s = sonuc_sinyali([], router_uzmani="carpenter")
        self.assertIsNone(s["zirve_ad"])
        self.assertIsNone(s["sinyal"])

    def test_aktivasyon_olayi_formati(self):
        aktivasyon_sonuc = {
            "aktif_yetenekler": [(AD_SOZLUGU["marangoz_ceketi"], 0.42, 0.525)],
            "karar_notu": "test notu",
        }
        olay = danisma_raporu(
            aktivasyon_sonuc, sorgu="zımpara nedir", router_uzmani="carpenter",
            kaynak="test", telemetri_yolu=self.yol,
        )["olay"]
        for anahtar in ("olay", "zaman", "kaynak", "sorgu_ozeti",
                        "aktif_yetenekler", "zirve_kimligi", "zirve_uzmani",
                        "router_uzmani", "uyum", "sinyal", "karar_notu"):
            self.assertIn(anahtar, olay, f"telemetri olayında eksik: {anahtar}")
        self.assertEqual(olay["olay"], "aktivasyon")
        self.assertEqual(olay["aktif_yetenekler"][0]["kimlik"], "marangoz_ceketi")
        self.assertEqual(olay["aktif_yetenekler"][0]["skor"], 0.42)
        self.assertEqual(olay["aktif_yetenekler"][0]["etki"], 0.525)
        self.assertEqual(olay["kaynak"], "test")

    def test_sorgu_ozeti_kesilir(self):
        uzun = "x" * 500
        olay = danisma_raporu({"aktif_yetenekler": [], "karar_notu": "n"},
                              sorgu=uzun, telemetri_yolu=self.yol)["olay"]
        self.assertLessEqual(len(olay["sorgu_ozeti"]), 120)

    def test_danisma_raporu_telemetri_yazar(self):
        aktivasyon_sonuc = {
            "aktif_yetenekler": [(AD_SOZLUGU["legal_ceketi"], 0.16, 0.20)],
            "karar_notu": "n",
        }
        rapor = danisma_raporu(aktivasyon_sonuc, sorgu="iş kanunu maddesi",
                               router_uzmani="legal", telemetri_yolu=self.yol)
        self.assertEqual(rapor["meta"]["uyum"], True)
        self.assertEqual(rapor["meta"]["sinyal"], 1)
        self.assertEqual(rapor["telemetri"]["toplam_satir"], 1)
        kayitlar = telemetri_oku(self.yol)
        self.assertEqual(kayitlar[-1]["router_uzmani"], "legal")
        self.assertEqual(kayitlar[-1]["sinyal"], 1)


# ---------------------------------------------------------------------------
# 2) GÖREV-2 — Ağırlık evrimi: kural + bant + geçmiş + determinizm
# ---------------------------------------------------------------------------

class TestAgirlikEvrimiKurali(unittest.TestCase):
    def test_uyum_carpma(self):
        self.assertEqual(agirlik_evrimi(1.0, 1), 1.05)
        self.assertEqual(agirlik_evrimi(1.25, 1), round(1.25 * 1.05, 6))

    def test_uyumsuz_carpma(self):
        self.assertEqual(agirlik_evrimi(1.0, -1), 0.95)
        self.assertEqual(agirlik_evrimi(1.25, -1), round(1.25 * 0.95, 6))

    def test_sinyalsiz_degismez(self):
        self.assertEqual(agirlik_evrimi(1.23, None), 1.23)
        self.assertEqual(agirlik_evrimi(1.23, 0), 1.23)

    def test_ust_bant_kirpilir(self):
        # 1.95 × 1.05 = 2.0475 → bant tavanı 2.0
        self.assertEqual(agirlik_evrimi(1.95, 1), 2.0)
        self.assertEqual(agirlik_evrimi(2.0, 1), 2.0)

    def test_alt_bant_kirpilir(self):
        # 0.52 × 0.95 = 0.494 → bant tabanı 0.5
        self.assertEqual(agirlik_evrimi(0.52, -1), 0.5)
        self.assertEqual(agirlik_evrimi(0.5, -1), 0.5)

    def test_bant_sabitleri(self):
        self.assertEqual((AGIRLIK_MIN, AGIRLIK_MAX), (0.5, 2.0))
        self.assertEqual(EVRIM_EPSILON, 0.05)

    def test_evrim_uygula_saf_ve_yeniden_oynatma(self):
        a0 = {"marangoz_ceketi": 1.25, "legal_ceketi": 1.25}
        olaylar = [
            {"zirve_kimligi": "marangoz_ceketi", "sinyal": 1},
            {"zirve_kimligi": "marangoz_ceketi", "sinyal": -1},
            {"zirve_kimligi": "legal_ceketi", "sinyal": 1},
            {"zirve_kimligi": "legal_ceketi", "sinyal": None},  # kayıt-yalnız
        ]
        y1, g1 = evrim_uygula(a0, olaylar)
        y2, g2 = evrim_uygula(a0, olaylar)
        self.assertEqual(y1, y2, "yeniden-oynatma: aynı olaylar → aynı ağırlıklar")
        self.assertEqual(g1, g2, "yeniden-oynatma: aynı olaylar → aynı geçmiş")
        self.assertEqual(y1["marangoz_ceketi"], round(1.25 * 1.05 * 0.95, 6))
        self.assertEqual(y1["legal_ceketi"], round(1.25 * 1.05, 6))
        self.assertEqual(len(g1), 3)  # sinyal None olan olay geçmişe yazmaz
        self.assertEqual(a0["marangoz_ceketi"], 1.25)  # girdi sözlüğü DEĞİŞMEZ

    def test_evrim_uygula_gecmis_kaydi(self):
        a0 = {"pedagoji_uzmani": 1.15}
        _, gecmis = evrim_uygula(a0, [{"zirve_kimligi": "pedagoji_uzmani", "sinyal": -1}])
        self.assertEqual(len(gecmis), 1)
        kayit = gecmis[0]
        self.assertEqual(kayit["kimlik"], "pedagoji_uzmani")
        self.assertEqual(kayit["eski_agirlik"], 1.15)
        self.assertEqual(kayit["yeni_agirlik"], round(1.15 * 0.95, 6))
        self.assertEqual(kayit["sinyal"], -1)

    def test_evrim_uygula_bilinmeyen_kimlik_fail_closed(self):
        with self.assertRaises(ValueError):
            evrim_uygula({"a": 1.0}, [{"zirve_kimligi": "yok_boyle_kart", "sinyal": 1}])


class TestAgirlikYaz(unittest.TestCase):
    """Yerel (storage_path) Qdrant ile: idempotent payload geri-yazımı."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ag = YetenekAgi(storage_path=self.tmp.name, tokenizer=SahteTokenizer())
        self.ag.tohumla(kartlar=kart_envanteri())

    def agirlik_oku(self, kimlik):
        for p in self.ag.kartlari_oku(limit=50):
            if p.get("kimlik") == kimlik:
                return float(p.get("agirlik", 1.0))
        raise AssertionError(f"kart bulunamadı: {kimlik}")

    def test_yaz_ve_read_back(self):
        r = self.ag.agirlik_yaz("marangoz_ceketi", 1.40)
        self.assertEqual(r["eski_agirlik"], 1.25)
        self.assertEqual(r["yeni_agirlik"], 1.40)
        self.assertEqual(self.agirlik_oku("marangoz_ceketi"), 1.40)
        self.assertEqual(self.ag.kart_sayisi(), len(kart_envanteri()))  # çoğaltma yok

    def test_idempotent_ikinci_yaz(self):
        self.ag.agirlik_yaz("marangoz_ceketi", 1.40)
        self.ag.agirlik_yaz("marangoz_ceketi", 1.40)  # aynı nokta → güncelle
        self.assertEqual(self.ag.kart_sayisi(), len(kart_envanteri()))
        self.assertEqual(self.agirlik_oku("marangoz_ceketi"), 1.40)

    def test_bant_disi_fail_closed(self):
        with self.assertRaises(ValueError):
            self.ag.agirlik_yaz("marangoz_ceketi", 2.5)   # > AGIRLIK_MAX
        with self.assertRaises(ValueError):
            self.ag.agirlik_yaz("marangoz_ceketi", 0.4)   # < AGIRLIK_MIN

    def test_bilinmeyen_kimlik_fail_closed(self):
        with self.assertRaises(ValueError):
            self.ag.agirlik_yaz("yok_boyle_kart", 1.0)

    def test_etki_yeni_agirlikla_ucusur(self):
        once = self.ag.aktivasyon("zımpara nedir", top_k=5, esik=0.0)["aktif_yetenekler"]
        ad = once[0][0]
        kimlik = ad_kimligi(ad)
        eski_w = self.agirlik_oku(kimlik)
        yeni_w = round(eski_w * 1.2, 6)
        self.ag.agirlik_yaz(kimlik, yeni_w)
        sonra = self.ag.aktivasyon("zımpara nedir", top_k=5, esik=0.0)["aktif_yetenekler"]
        es_sozluk = {a: (s, e) for (a, s, e) in once}
        so_sozluk = {a: (s, e) for (a, s, e) in sonra}
        self.assertEqual(so_sozluk[ad][0], es_sozluk[ad][0])          # skor aynı
        self.assertAlmostEqual(so_sozluk[ad][1],
                               es_sozluk[ad][1] * (yeni_w / eski_w))  # etki ∝ ağırlık


class TestEvrimIsle(unittest.TestCase):
    """Canlı evrim döngüsü: olaylar → sinyal → ağırlık geri-yazımı + telemetri."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.telemetri = os.path.join(self.tmp.name, "telemetri.jsonl")

    def _yeni_ag(self):
        ag = YetenekAgi(storage_path=self.tmp.name, tokenizer=SahteTokenizer())
        ag.tohumla(kartlar=kart_envanteri())
        return ag

    def _zirve_bilgisi(self, ag):
        aktif = ag.aktivasyon("zımpara nedir", top_k=3)["aktif_yetenekler"]
        return sonuc_sinyali(aktif)["zirve_uzmani"]

    def test_evrim_isle_uyum_uyumsuz_kayityaliniz(self):
        ag = self._yeni_ag()
        zirve_uzmani = self._zirve_bilgisi(ag)
        self.assertIsNotNone(zirve_uzmani, "test zemini: zirvenin uzmanı bilinmeli")
        olaylar = [
            {"sorgu": "zımpara nedir", "router_uzmani": zirve_uzmani, "kaynak": "test"},   # uyum
            {"sorgu": "zımpara nedir", "router_uzmani": "legal", "kaynak": "test"},         # uyumsuz
            {"sorgu": "zımpara nedir", "router_uzmani": None, "kaynak": "test"},            # kayıt-yalnız
        ]
        r = ag.evrim_isle(olaylar, telemetri_yolu=self.telemetri)
        self.assertEqual(r["islenen"], 3)
        self.assertEqual(len(r["agirlik_gecmisi"]), 2)  # 3. olay sinyalsiz → değişiklik yok
        g = r["agirlik_gecmisi"]
        # uyum sonra uyumsuz → net ×(1+eps)×(1−eps)
        beklenen = round(round(1.25 * 1.05, 6) * 0.95, 6) if g[0]["eski_agirlik"] == 1.25 \
            else round(g[0]["eski_agirlik"] * 1.05 * 0.95, 6)
        self.assertAlmostEqual(g[-1]["yeni_agirlik"], beklenen, places=6)
        # telemetri: 3 aktivasyon olayı + 2 evrim geçmiş kaydı
        kayitlar = telemetri_oku(self.telemetri)
        tipler = [k.get("olay") for k in kayitlar]
        self.assertEqual(tipler.count("aktivasyon"), 3)
        self.assertEqual(tipler.count("agirlik_evrimi"), 2)
        evrim_kaydi = [k for k in kayitlar if k.get("olay") == "agirlik_evrimi"][0]
        for anahtar in ("kimlik", "eski_agirlik", "yeni_agirlik", "sinyal", "eps"):
            self.assertIn(anahtar, evrim_kaydi)

    def test_evrim_isle_koleksiyona_yazildi(self):
        ag = self._yeni_ag()
        zirve_uzmani = self._zirve_bilgisi(ag)
        olaylar = [{"sorgu": "zımpara nedir", "router_uzmani": zirve_uzmani, "kaynak": "t"}]
        r = ag.evrim_isle(olaylar, telemetri_yolu=self.telemetri)
        kimlik = r["agirlik_gecmisi"][0]["kimlik"]
        okunan = {p["kimlik"]: p["agirlik"] for p in ag.kartlari_oku(limit=50)}[kimlik]
        self.assertEqual(okunan, r["agirlik_gecmisi"][0]["yeni_agirlik"])
        self.assertEqual(ag.kart_sayisi(), len(kart_envanteri()))  # upsert → çoğaltma yok

    def test_evrim_isle_yeniden_oynatma_determinizmi(self):
        # AYNI başlangıç durumu + AYNI olaylar → birebir aynı geçmiş (2. koşum)
        g1 = self._yeni_ag().evrim_isle(
            [{"sorgu": "zımpara nedir", "router_uzmani": u, "kaynak": "t"}
             for u in ("carpenter", "legal", None, "gardener")],
            telemetri_yolu=os.path.join(self.tmp.name, "t1.jsonl"))["agirlik_gecmisi"]
        g2 = self._yeni_ag().evrim_isle(
            [{"sorgu": "zımpara nedir", "router_uzmani": u, "kaynak": "t"}
             for u in ("carpenter", "legal", None, "gardener")],
            telemetri_yolu=os.path.join(self.tmp.name, "t2.jsonl"))["agirlik_gecmisi"]
        temiz1 = [{k: v for k, v in g.items() if k != "zaman"} for g in g1]
        temiz2 = [{k: v for k, v in g.items() if k != "zaman"} for g in g2]
        self.assertEqual(temiz1, temiz2, "yeniden-oynatma determinizmi")


# ---------------------------------------------------------------------------
# 3) GÖREV-4 — Kısıt kartları: envanter + saf çekirdek + motor
# ---------------------------------------------------------------------------

class TestKisitEnvanteri(unittest.TestCase):
    def setUp(self):
        self.kisitlar = kisit_envanteri()

    def test_envanter_3_5_kart(self):
        self.assertGreaterEqual(len(self.kisitlar), 3, "görev-4: ≥3 kısıt kartı")
        self.assertLessEqual(len(self.kisitlar), 5, "görev-4: ≤5 kısıt kartı")

    def test_tip_ve_alan_kisit(self):
        for k in self.kisitlar:
            self.assertEqual(k.meta.get("tip"), KISIT_TIPI)
            self.assertEqual(k.etki_alani, "kisit")

    def test_carpan_baskilayici(self):
        for k in self.kisitlar:
            c = float(k.meta["carpan"])
            self.assertGreater(c, 0.0)
            self.assertLessEqual(c, 1.0, "kısıt çarpanı asla 1'den büyük olamaz")

    def test_kosul_gecerli_ve_hedef_tanimli(self):
        for k in self.kisitlar:
            self.assertIn(k.meta["kosul"], ("hedef_etkin", "gurultu_bandi"))
            hedef_var = bool(k.meta.get("hedef")) or bool(k.meta.get("hedef_alan"))
            self.assertTrue(hedef_var, f"hedef tanımsız: {k.kimlik}")
            if k.meta["kosul"] == "gurultu_bandi":
                self.assertIn("bant_esik", k.meta)

    def test_her_kisitin_kaniti_repo_gercek(self):
        for k in self.kisitlar:
            kanit = k.meta.get("kanit", "")
            self.assertTrue(kanit, f"kanit yok: {k.kimlik}")
            self.assertTrue(os.path.exists(os.path.join(REPO_KOKU, kanit)),
                            f"kanit repo'da yok: {kanit} ({k.kimlik})")

    def test_kimlikler_benzersiz_ve_yeteneklerden_ayrik(self):
        kimlikler = [k.kimlik for k in self.kisitlar]
        self.assertEqual(len(kimlikler), len(set(kimlikler)))
        self.assertFalse(set(kimlikler) & {k.kimlik for k in kart_envanteri()})

    def test_gercek_kurallar_hedefleriyle(self):
        """İş emrinin örnek verdigi GERÇEK kurallar envanterde ve doğru hedefli."""
        sozluk = {k.kimlik: k for k in self.kisitlar}
        # (1) donmuş sözlük 33.114 — yeni jeton yok → sözlük genişletme baskılanır
        self.assertIn("donmus_sozluk_33_114", sozluk)
        self.assertIn("sozluk_genisletme", sozluk["donmus_sozluk_33_114"].meta["hedef"])
        self.assertEqual(sozluk["donmus_sozluk_33_114"].meta["kanit"],
                         ".agent-bus/frozen.json")
        # (2) alan saflığı — ceket dışı sorguda alan kartı baskılanır
        self.assertIn("alan_safligi", sozluk)
        self.assertEqual(sozluk["alan_safligi"].meta.get("hedef_alan"), ["alan"])
        self.assertEqual(sozluk["alan_safligi"].meta["kanit"], "scripts/yetenek_probe.py")
        # (3) ölçülmüş UNK kuralı ('hükmü' UNK — kullanılmaz)
        self.assertIn("hukum_unk_yasagi", sozluk)
        self.assertEqual(sozluk["hukum_unk_yasagi"].meta["kanit"],
                         "scripts/prepare_legal_dataset_v3.py")


class TestKisitCekirdek(unittest.TestCase):
    def test_etki_carpanli_aritmetik(self):
        self.assertEqual(etki_hesapla(1.25, 0.04, 0.5), 0.025)   # ağırlık×skor×çarpan
        self.assertEqual(etki_hesapla(1.25, 0.04), 0.05)         # çarpansız = v1
        self.assertEqual(etki_hesapla(1.25, 0.04, 1.0), 0.05)    # çarpan 1 = v1

    def test_aktivasyon_seti_carpan_etkiyi_dusurur(self):
        adaylar = [("Ağır Hedef", 0.40, 1.5), ("Hafif Temiz", 0.35, 1.0)]
        carpanlar = {"Ağır Hedef": 0.5}
        sonuc = aktivasyon_seti(adaylar, esik=0.0, top_k=2, carpanlar=carpanlar)
        # Ağır Hedef etki 0.60 → 0.30; Hafif 0.35 → baskılama sırayı DEĞİŞTİRDİ
        self.assertEqual(sonuc[0][0], "Hafif Temiz")
        self.assertEqual(sonuc[1][2], 0.3)

    def test_kisit_carpanlari_hedef_etkin(self):
        adaylar = [("Sözlük Genişletme (expand_lexicon_and_vocab)",
                    "sozluk_genisletme", "ölçüm", 0.05),
                   ("RAG Boru Hattı (RagPipeline)", "rag_boru_hatti", "rag", 0.30)]
        kisit = {"kimlik": "donmus_sozluk_33_114", "carpan": 0.5,
                 "kosul": "hedef_etkin", "hedef": ["sozluk_genisletme"]}
        self.assertEqual(kisit_carpanlari(adaylar, [kisit], esik=0.02),
                         {"Sözlük Genişletme (expand_lexicon_and_vocab)": 0.5})

    def test_kisit_carpanlari_esik_alti_hedef_etkin_degil(self):
        adaylar = [("S", "sozluk_genisletme", "ölçüm", 0.01)]  # 0.01 < eşik 0.02
        kisit = {"kimlik": "k", "carpan": 0.5, "kosul": "hedef_etkin",
                 "hedef": ["sozluk_genisletme"]}
        self.assertEqual(kisit_carpanlari(adaylar, [kisit], esik=0.02), {})

    def test_kisit_carpanlari_gurultu_bandi(self):
        kisit = {"kimlik": "alan_safligi", "carpan": 0.5, "kosul": "gurultu_bandi",
                 "bant_esik": 0.03, "hedef_alan": ["alan"]}
        dusuk = [("Marangoz Alan Ceketi (carpenter)", "marangoz_ceketi", "alan", 0.025)]
        yuksek = [("Marangoz Alan Ceketi (carpenter)", "marangoz_ceketi", "alan", 0.033)]
        self.assertEqual(kisit_carpanlari(dusuk, [kisit]),
                         {"Marangoz Alan Ceketi (carpenter)": 0.5})   # bant içi → baskılanır
        self.assertEqual(kisit_carpanlari(yuksek, [kisit]), {})        # ilgili isabet → dokunulmaz

    def test_kisit_carpanlari_carpim_asla_yukseltmez(self):
        adaylar = [("S", "sozluk_genisletme", "ölçüm", 0.10)]
        kisitlar = [
            {"kimlik": "k1", "carpan": 0.5, "kosul": "hedef_etkin",
             "hedef": ["sozluk_genisletme"]},
            {"kimlik": "k2", "carpan": 0.9, "kosul": "hedef_etkin",
             "hedef": ["sozluk_genisletme"]},
        ]
        self.assertEqual(kisit_carpanlari(adaylar, kisitlar, esik=0.02)["S"], 0.45)
        # hedef olmayan kısıt hiç dokunmaz
        self.assertEqual(kisit_carpanlari(adaylar, kisitlar[:1] + [
            {"kimlik": "k3", "carpan": 0.1, "kosul": "hedef_etkin",
             "hedef": ["baska_kart"]}], esik=0.02)["S"], 0.5)


class TestMotorKisitli(unittest.TestCase):
    """Aktivasyon motoru + kısıt kartları (mock bellek)."""

    ESLESMELER = [
        (AD_SOZLUGU["marangoz_ceketi"], 0.033, 1.25, "marangoz_ceketi", "alan"),
        (AD_SOZLUGU["sozluk_genisletme"], 0.05, 0.85, "sozluk_genisletme", "ölçüm"),
        (AD_SOZLUGU["kristal_tokenizer"], 0.20, 1.15, "kristal_tokenizer", "tokenizer"),
    ]
    KISITLAR = [
        {"kimlik": "donmus_sozluk_33_114", "carpan": 0.5, "kosul": "hedef_etkin",
         "hedef": ["sozluk_genisletme"], "kanit": ".agent-bus/frozen.json"},
    ]

    def test_kisitli_aktivasyon_etkiyi_ve_raporu_uretir(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESMELER, self.KISITLAR),
                        tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("sözlüğe yeni kelime ekle", top_k=3, esik=0.02)
        etki = {ad: e for (ad, _, e) in sonuc["aktif_yetenekler"]}
        self.assertAlmostEqual(etki[AD_SOZLUGU["sozluk_genisletme"]],
                                0.85 * 0.05 * 0.5)  # kısıt çarpanı uygulandı
        self.assertIn("kisit_etkileri", sonuc)
        self.assertEqual(sonuc["kisit_etkileri"]["uygulanan"],
                         {AD_SOZLUGU["sozluk_genisletme"]: 0.5})
        # kısıt kartının KENDİSİ yetenek olarak aktifleşmez
        adlar = [ad for (ad, _, _) in sonuc["aktif_yetenekler"]]
        for k in self.KISITLAR:
            self.assertNotIn(k["kimlik"], adlar)

    def test_kisitsiz_bellekte_sonuc_v1_birebir(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESMELER), tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("zımpara nedir", top_k=3, esik=0.02)
        self.assertEqual(set(sonuc.keys()), {"aktif_yetenekler", "karar_notu"},
                         "kısıt yokken sonuç anahtar seti v1 ile birebir")
        etki = {ad: e for (ad, _, e) in sonuc["aktif_yetenekler"]}
        self.assertAlmostEqual(etki[AD_SOZLUGU["sozluk_genisletme"]], 0.85 * 0.05)  # çarpansız

    def test_bozuk_kisit_fail_closed(self):
        bozuk = [{"kimlik": "bozuk", "carpan": 1.5, "kosul": "hedef_etkin",
                  "hedef": ["sozluk_genisletme"]}]  # çarpan > 1 → yükseltici!
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESMELER, bozuk),
                        tokenizer=SahteTokenizer())
        with self.assertRaises(ValueError):
            ag.aktivasyon("q", top_k=3)  # baskılayıcı olmayan kısıt kurulumu durur


class TestKisitliTohumlama(unittest.TestCase):
    """Yerel Qdrant: yetenek + kısıt kartları birlikte tohumlanır (idempotent)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ag = YetenekAgi(storage_path=self.tmp.name, tokenizer=SahteTokenizer())

    def test_birlesik_tohum_idempotent(self):
        kartlar = list(kart_envanteri()) + list(kisit_envanteri())
        r1 = self.ag.tohumla(kartlar=kartlar)
        r2 = self.ag.tohumla(kartlar=kartlar)
        beklenen = len(kartlar)
        self.assertEqual(r1["kart_sayisi"], beklenen)
        self.assertEqual(r2["kart_sayisi"], beklenen, "ikinci koşum ÇOĞALTMAZ")

    def test_kisit_noktalari_aktivasyonda_yetenek_degil(self):
        kartlar = list(kart_envanteri()) + list(kisit_envanteri())
        self.ag.tohumla(kartlar=kartlar)
        sonuc = self.ag.aktivasyon("zımpara nedir", top_k=5)
        kisit_adlari = {k.ad for k in kisit_envanteri()}
        adlar = {ad for (ad, _, _) in sonuc["aktif_yetenekler"]}
        self.assertFalse(adlar & kisit_adlari, "kısıt kartları yetenek olarak aktifleşmez")

    def test_kisitli_tohumlamada_bozuk_kart_fail_closed(self):
        bozuk = YetenekKarti(
            kimlik="bozuk_kisit", ad="Bozuk", aciklama="çarpanı 1.5 olan sahte kısıt kartıdır.",
            etki_alani="kisit", agirlik=1.0,
            meta={"tip": "kisit", "carpan": 1.5, "kosul": "hedef_etkin", "hedef": ["x"]})
        with self.assertRaises(ValueError):
            self.ag.tohumla(kartlar=[bozuk])


# ---------------------------------------------------------------------------
# 4) GÖREV-3 — Gateway DANIŞMA kancası (OP-İN, default KAPALI)
# ---------------------------------------------------------------------------

class TestGatewayDanisma(unittest.TestCase):
    def _gateway_kur(self, yetenek_agi=None, danisma=False, telemetri_yolu=None):
        import torch
        from src.gateway.agent_gateway import AgentGateway
        from src.rag.vector_memory import VectorMemory

        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3, "ahşap": 4, "bilgi": 5}
            itos = {v: k for k, v in stoi.items()}

            def decode(self, idx):
                return self.itos.get(idx, "unk")

        class MockTokenizer:
            vocab = MockVocab()

            def encode(self, text):
                return [0, 4, 5, 2]

            def decode(self, ids):
                return "ahşap bilgi"

        class MockDecompiler:
            def decompile_sentence(self, tags_str, capitalize=False):
                return "ahşap bilgisi"

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 768)

            def forward(self, x, return_hidden_states=False):
                b, s = x.shape
                logits = torch.ones(b, s, 10)
                hidden = torch.randn(b, s, 768)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bellek = VectorMemory(collection_name="test_vg2b_danisma", vector_size=768,
                              storage_path=tmp.name, host=None)
        self.addCleanup(bellek.close)
        return AgentGateway(
            model=MockModel(), tokenizer=MockTokenizer(), decompiler=MockDecompiler(),
            memory=bellek,
            future_train_path=os.path.join(tmp.name, "ft.jsonl"),
            device="cpu", yetenek_agi=yetenek_agi,
            yetenek_danisma=danisma,
            yetenek_telemetri_yolu=telemetri_yolu or os.path.join(tmp.name, "tel.jsonl"),
        )

    @staticmethod
    def _kanca(zirve_kimlik):
        kanca = MagicMock()
        kanca.aktivasyon.return_value = {
            "aktif_yetenekler": [(AD_SOZLUGU[zirve_kimlik], 0.42, 0.525)],
            "karar_notu": "test notu",
        }
        return kanca

    def test_danisma_default_kapali_anahtar_yok(self):
        gw = self._gateway_kur(yetenek_agi=self._kanca("marangoz_ceketi"))
        res = gw.ask("zımpara nedir?", mode="RAG")
        self.assertIn("yetenek_aktivasyonu", res)      # v1 gözlem birebir
        self.assertNotIn("yetenek_danisma", res)      # danışma KAPALI: anahtar YOK

    def test_danisma_default_kapali_telemetri_yazmaz(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        yol = os.path.join(tmp.name, "tel.jsonl")
        gw = self._gateway_kur(yetenek_agi=self._kanca("marangoz_ceketi"),
                               telemetri_yolu=yol)
        gw.ask("zımpara nedir?", mode="RAG")
        self.assertFalse(os.path.exists(yol), "danışma kapalıyken telemetri YAZILMAZ")

    def test_danisma_acik_meta_ve_telemetri(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        yol = os.path.join(tmp.name, "tel.jsonl")
        kanca = self._kanca("marangoz_ceketi")
        gw = self._gateway_kur(yetenek_agi=kanca, danisma=True, telemetri_yolu=yol)
        res = gw.ask("zımpara nedir?", mode="RAG")
        self.assertIn("yetenek_aktivasyonu", res)      # v1 gözlem korunur
        self.assertIn("yetenek_danisma", res)         # OP-İN: META eklendi
        meta = res["yetenek_danisma"]
        for anahtar in ("zirve_ad", "zirve_kimligi", "zirve_uzmani",
                        "router_uzmani", "uyum", "sinyal"):
            self.assertIn(anahtar, meta)
        self.assertEqual(meta["zirve_kimligi"], "marangoz_ceketi")
        self.assertEqual(meta["zirve_uzmani"], "carpenter")
        beklenen_uyum = (meta["router_uzmani"] == "carpenter")
        self.assertEqual(meta["uyum"], beklenen_uyum)
        self.assertEqual(meta["sinyal"], 1 if beklenen_uyum else -1)
        kayitlar = telemetri_oku(yol)
        self.assertEqual(len(kayitlar), 1, "danışma açıkken olay telemetriye yazılır")
        self.assertEqual(kayitlar[0]["kaynak"], "gateway")
        self.assertEqual(kayitlar[0]["router_uzmani"], meta["router_uzmani"])

    def test_danisma_karari_degistirmez_ve_uyumsuzluk_loglanir(self):
        # AYNI sorgu, kapalı vs açık danışma → router KARARI birebir aynı
        kanca = self._kanca("marangoz_ceketi")
        res_kapali = self._gateway_kur(yetenek_agi=kanca).ask("zımpara nedir?", mode="RAG")
        res_acik = self._gateway_kur(yetenek_agi=kanca, danisma=True).ask(
            "zımpara nedir?", mode="RAG")
        self.assertEqual(res_acik["active_expert"], res_kapali["active_expert"],
                         "danışma router KARARINI DEĞİŞTİRMEZ")
        self.assertEqual(res_acik["router_experts"], res_kapali["router_experts"])
        self.assertEqual(res_acik["response_text"], res_kapali["response_text"])

        # uyumsuzluk senaryosu: zirve uzmanı ≠ router uzmanı → yalnız LOG
        router_uzmani = res_kapali["active_expert"]
        zit_kimlik = "legal_ceketi" if router_uzmani != "legal" else "marangoz_ceketi"
        kanca2 = self._kanca(zit_kimlik)
        gw2 = self._gateway_kur(yetenek_agi=kanca2, danisma=True)
        tampon = io.StringIO()
        with contextlib.redirect_stdout(tampon):
            res2 = gw2.ask("zımpara nedir?", mode="RAG")
        self.assertEqual(res2["yetenek_danisma"]["uyum"], False)
        self.assertEqual(res2["yetenek_danisma"]["sinyal"], -1)
        self.assertIn("VG2B_DANISMA", tampon.getvalue())  # uyumsuzluk LOGLANDI
        self.assertEqual(res2["active_expert"], router_uzmani,
                         "uyumsuzlukta bile karar DEĞİŞMEDİ")

    def test_danisma_hatasi_aski_bozmaz(self):
        kanca = MagicMock()
        kanca.aktivasyon.side_effect = RuntimeError("qdrant kapalı")
        gw = self._gateway_kur(yetenek_agi=kanca, danisma=True)
        res = gw.ask("zımpara nedir?", mode="RAG")
        self.assertIn("response_text", res)               # ask normale devam
        self.assertIsNone(res["yetenek_aktivasyonu"])      # dürüst None (v1 sözleşme)
        self.assertNotIn("yetenek_danisma", res)          # aktivasyon yoksa danışma da yok


if __name__ == "__main__":
    unittest.main()
