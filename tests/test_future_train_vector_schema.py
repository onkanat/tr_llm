#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""P2.3 — future_train_vector.jsonl ŞEMA SÖZLEŞMESİ (TDD).

Şemanın TEK yetkili kaynağı, kaydı YAZAN koddur:
`src/rag/epistemic_agent.py` — `EpistemicCuriosityAgent.process_query`
Step-5 "Model Anlamaz?!" (Epistemic Failure) değerlendirmesi, record sözlüğü
satır 518-533 (bölge 494-535). Yazar TAM 14 alan yazar:

    instruction, input, output, model_failed_output, decompiled_output,
    rag_document, retrieval_collection, similarity_score, similarity_threshold,
    entropy_pre, entropy_post, tau, reason, timestamp

Tip sözleşmesi:
  - similarity_score : float  (yazar `round(match_score, 4)`, match_score=float)
  - timestamp        : ISO-8601 saat-dilimli dize (yazar `datetime.now(timezone.utc).isoformat()`)
  - entropy_pre/post, tau, similarity_threshold : sayısal

`TRLLM_FUTURE_TRAIN_FILE` env'i test edilen dosya yolunu override eder:
şema geçişinin RED (eski-şema kopya) / GREEN (gerçek küçük) kanıtı ve
ilerideki göçler için tek test dosyası yeter.

`data/future_train_vector_legacy.bak.jsonl` = P2.3 sigorta kopyası: göç öncesi
küçük dosyanın bayt-özdeş yedeği (yıkıcı adım sigortası; retrain_pipeline
başarılı koşumda kütüğü arşive taşıyıp aktif dosyayı TEMİZLEDİĞİ için bu
kopya 5 tarihsel kaydın kanıtını saklar).
"""

import os
import sys
import json
import unittest
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Şema tanımı (kaynak: src/rag/epistemic_agent.py:518-533) -----------------
YAZICI_SEMASI = {
    "instruction",
    "input",
    "output",
    "model_failed_output",
    "decompiled_output",
    "rag_document",
    "retrieval_collection",
    "similarity_score",
    "similarity_threshold",
    "entropy_pre",
    "entropy_post",
    "tau",
    "reason",
    "timestamp",
}

AKTIF_YOL = os.environ.get(
    "TRLLM_FUTURE_TRAIN_FILE", "data/future_train_vector.jsonl")
YEDEK_YOL = "data/future_train_vector_legacy.bak.jsonl"


def kayitlari_oku(yol):
    """JSONL dosyasını kayıt listesi olarak döndürür (dosya yoksa [])"""
    if not os.path.exists(yol):
        return []
    with open(yol, "r", encoding="utf-8") as f:
        return [json.loads(satir) for satir in f if satir.strip()]


def sema_sorunlari(kayit):
    """Bir kaydın yazar şemasından tüm sapmalarını döndürür (boş liste = uyumlu)."""
    sorunlar = []
    eksik = sorted(YAZICI_SEMASI - set(kayit))
    fazla = sorted(set(kayit) - YAZICI_SEMASI)
    if eksik:
        sorunlar.append(f"eksik alanlar: {eksik}")
    if fazla:
        sorunlar.append(f"fazla/bilinmeyen alanlar: {fazla}")

    skor = kayit.get("similarity_score")
    if isinstance(skor, bool) or not isinstance(skor, float):
        sorunlar.append(
            f"similarity_score tip float değil: {type(skor).__name__} ({skor!r})")

    ts = kayit.get("timestamp")
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            sorunlar.append("timestamp saat dilimi (tz) taşımıyor")
    except (TypeError, ValueError):
        sorunlar.append(f"timestamp ISO-8601 ayrıştırılamadı: {ts!r}")

    for alan in ("entropy_pre", "entropy_post", "tau", "similarity_threshold"):
        deger = kayit.get(alan)
        if isinstance(deger, bool) or not isinstance(deger, (int, float)):
            sorunlar.append(f"{alan} sayısal değil: {type(deger).__name__}")

    return sorunlar


class TestAktifKutukSemasi(unittest.TestCase):
    """Aktif küçük (data/future_train_vector.jsonl): mevcut TÜM kayıtlar yeni şemada.

    Not: retrain_pipeline.run_training başarılı olursa kayıtları arşive taşıyıp
    aktif dosyayı boşaltır (tasarım budur) — 0 kayıt bu teste göre GEÇERLİ
    durumdur; sayı > 0 ise her kayıt birebir doğrulanır.
    """

    def setUp(self):
        self.kayitlar = kayitlari_oku(AKTIF_YOL)

    def test_aktif_dosya_yazar_semasina_uyuyor(self):
        dosya = os.path.abspath(AKTIF_YOL)
        self.assertTrue(
            os.path.exists(AKTIF_YOL),
            f"aktif küçük dosya yok: {dosya}")
        for i, kayit in enumerate(self.kayitlar, 1):
            sorunlar = sema_sorunlari(kayit)
            self.assertEqual(
                sorunlar, [],
                f"aktif dosya {dosya} kayıt #{i} şema dışı: {sorunlar}")

    def test_aktif_dosya_jsonl_gecerli(self):
        # Her satır ayrıştırılabilir olmalı (bozuk JSON sessizce yutulamaz)
        with open(AKTIF_YOL, "r", encoding="utf-8") as f:
            for i, satir in enumerate(f, 1):
                if satir.strip():
                    try:
                        json.loads(satir)
                    except json.JSONDecodeError as e:
                        self.fail(f"aktif dosya satır {i} bozuk JSON: {e}")


class TestLegacyYedek(unittest.TestCase):
    """P2.3 sigorta kopyası: göç öncesi 5 tarihsel kayıt, yeni şemada korunur."""

    def test_yedek_5_kayit_ve_yeni_semada(self):
        self.assertTrue(
            os.path.exists(YEDEK_YOL),
            f"sigorta kopyası yok: {os.path.abspath(YEDEK_YOL)}")
        kayitlar = kayitlari_oku(YEDEK_YOL)
        self.assertEqual(
            len(kayitlar), 5,
            f"sigorta kopyası tam olarak 5 göç-öncesi kayıt tutmalı, geldi: {len(kayitlar)}")
        for i, kayit in enumerate(kayitlar, 1):
            sorunlar = sema_sorunlari(kayit)
            self.assertEqual(
                sorunlar, [],
                f"yedek kayıt #{i} şema dışı: {sorunlar}")


class TestSemayaDisler(unittest.TestCase):
    """Doğrulayıcının DİŞLERİ: eski-şema tipi (timestamp/score yoksun) RED üretir.

    P2.3 kanıt zinciri: bu sınıf, şema doğrulayıcısının 'kayıt timestamp/score
    taşımasa da geçer' diye sessizce geçirmediğini kanıtlar — RED fazının
    tutarlılığı budur (mevcut dosyada eski-şema kayıt çıkmadığından dişler
    sentetik eski-şema kayıtlarıyla ölçülür).
    """

    def _ornek_kayit(self):
        return {
            "instruction": "Bilimsel ve net açıkla.",
            "input": "<BELGE> örnek belge </BELGE> örnek soru?",
            "output": "örnek POSS_3SG",
            "model_failed_output": "örnek",
            "decompiled_output": "Örnek.",
            "rag_document": "örnek belge",
            "retrieval_collection": "anka_bellek",
            "similarity_score": 0.91,
            "similarity_threshold": 0.85,
            "entropy_pre": 7.5,
            "entropy_post": 3.2,
            "tau": 2.5,
            "reason": "epistemic_gap_unresolved_high_similarity",
            "timestamp": "2026-11-04T12:00:00+00:00",
        }

    def test_uyumlu_kayit_sorun_yok(self):
        self.assertEqual(sema_sorunlari(self._ornek_kayit()), [])

    def test_timestamp_ve_score_eksik_eski_sema_reddedilir(self):
        eski = self._ornek_kayit()
        del eski["timestamp"]
        del eski["similarity_score"]
        sorunlar = sema_sorunlari(eski)
        self.assertTrue(sorunlar, "eski-şema kaydı (timestamp+score yoksun) RED üretmeli")
        self.assertTrue(any("eksik alanlar" in s for s in sorunlar))

    def test_timestamp_tip_hatasi_reddedilir(self):
        bozuk = self._ornek_kayit()
        bozuk["timestamp"] = "04-11-2026"  # ISO-8601 değil
        sorunlar = sema_sorunlari(bozuk)
        self.assertTrue(any("ISO-8601" in s for s in sorunlar))

    def test_score_tip_hatasi_reddedilir(self):
        bozuk = self._ornek_kayit()
        bozuk["similarity_score"] = "0.91"  # dize — float değil
        sorunlar = sema_sorunlari(bozuk)
        self.assertTrue(any("similarity_score" in s for s in sorunlar))


if __name__ == "__main__":
    unittest.main()