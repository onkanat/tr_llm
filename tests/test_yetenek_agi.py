#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-2 — Vektörel Gezgin v1: yetenek_ağı (yetenek_agi) TDD testleri
=================================================================
Kapsam (görev-3):
- aktivasyon determinizmi (aynı sorgu → aynı sonuç, tohum sabit)
- top-k / eşik davranışı
- ağırlıklı etki aritmetiği (etki = ağırlık × benzerlik)
- boş sonuç
- Qdrant'sız birim testleri (mock bellek)
- gerçek koleksiyonla duman (erişilemezse SKIP, sessiz-uydurma yok)
- gateway gözlem kancası (davranış DEĞİŞİMİ YOK: yetenek_agi verilmeden
  anahtar YOK; verilirse 'yetenek_aktivasyonu' gözlem alanı eklenir)

Kurallar: yetenek_agi koleksiyonu DIŞINDA hiçbir koleksiyona yazılmaz;
yerel testler VectorMemory(storage_path=TMPDIR) ile ağ-sız koşar.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.yetenek_agi import (  # noqa: E402
    DEFAULT_ESIK,
    KOLEKSIYON,
    VEKTOR_BOYUTU,
    YetenekAgi,
    YetenekKarti,
    aktivasyon_seti,
    etki_hesapla,
    karar_notu_olustur,
    kart_envanteri,
    nokta_kimligi,
)

REPO_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SahteTokenizer:
    """Birim-test tokenizer'ı: deterministik, sabit token akışı."""

    def encode(self, text):  # noqa: D102
        return [4, 5, 6, 7]

    def decode(self, ids):  # noqa: D102
        return "ahşap bilgi cart zımpara"


def sahte_bellek(eslesmeler):
    """dense_recall döndüren mock VectorMemory — Qdrant'a bağlanmaz.

    eslesmeler: [(ad, skor, agirlik), ...]
    """
    bellek = MagicMock()
    bellek.collection_name = "yetenek_agi"  # YetenekAgi'nin fail-closed ad-kapısı
    bellek.dense_recall.return_value = [
        {"score": skor, "text": f"yetenek:{ad}",
         "metadata": {"ad": ad, "agirlik": agirlik, "kimlik": ad}}
        for (ad, skor, agirlik) in eslesmeler
    ]
    return bellek


# ---------------------------------------------------------------------------
# 1) YetenekKarti + kayıt kimliği
# ---------------------------------------------------------------------------

class TestYetenekKarti(unittest.TestCase):
    def test_kart_alanlari_ve_doc_text(self):
        kart = YetenekKarti(
            kimlik="test_karti", ad="Test Kartı", aciklama="Test kapsamı.",
            etki_alani="rag", agirlik=1.25, meta={"kanit": "src/rag/embedding.py"},
        )
        self.assertEqual(kart.kimlik, "test_karti")
        self.assertEqual(kart.ad, "Test Kartı")
        self.assertEqual(kart.etki_alani, "rag")
        self.assertEqual(kart.agirlik, 1.25)
        doc = kart.doc_text()
        for parca in ("Test Kartı", "rag", "Test kapsamı."):
            self.assertIn(parca, doc)

    def test_kart_donmus_ve_deterministik(self):
        kart = YetenekKarti(kimlik="k", ad="A", aciklama="x", etki_alani="rag")
        with self.assertRaises(Exception):
            kart.ad = "B"  # frozen: alan atama kapalı
        self.assertEqual(kart.doc_text(), kart.doc_text())  # deterministik

    def test_nokta_kimligi_sabit_ve_ayristirici(self):
        a1, a2 = nokta_kimligi("marangoz_ceketi"), nokta_kimligi("marangoz_ceketi")
        self.assertEqual(a1, a2)                       # aynı kimlik → aynı nokta
        self.assertGreaterEqual(a1, 0)
        self.assertLess(a1, 2 ** 63)
        self.assertNotEqual(a1, nokta_kimligi("legal_ceketi"))  # ayrıştırıcı


# ---------------------------------------------------------------------------
# 2) Kart envanteri — GERÇEK yetenekler (uydurma yok)
# ---------------------------------------------------------------------------

class TestKartEnvanteri(unittest.TestCase):
    def setUp(self):
        self.kartlar = kart_envanteri()

    def test_envanter_boyutu_20_40(self):
        self.assertGreaterEqual(len(self.kartlar), 20, "görev-2: ≥20 kart")
        self.assertLessEqual(len(self.kartlar), 40, "görev-2: ≤40 kart")

    def test_kimlik_ve_ad_benzersiz(self):
        kimlikler = [k.kimlik for k in self.kartlar]
        adlar = [k.ad for k in self.kartlar]
        self.assertEqual(len(kimlikler), len(set(kimlikler)))
        self.assertEqual(len(adlar), len(set(adlar)))

    def test_alanlar_bilinen_kumede(self):
        gecerli = {"derleyici", "tokenizer", "rag", "alan", "router",
                   "gateway", "veri", "ölçüm"}
        for k in self.kartlar:
            self.assertIn(k.etki_alani, gecerli, f"geçersiz alan: {k.kimlik}")

    def test_agirlik_pozitif_ve_sinirli(self):
        for k in self.kartlar:
            self.assertGreater(k.agirlik, 0.0)
            self.assertLessEqual(k.agirlik, 2.0)

    def test_aciklama_tanimli_ve_zengin(self):
        for k in self.kartlar:
            self.assertGreaterEqual(len(k.aciklama), 30, f"zayıf açıklama: {k.kimlik}")

    def test_her_kartin_kaniti_repo_icinde_gercek(self):
        """Uydurma kart kapısı: her kartın meta['kanit'] yolu repo'da VAR olmalı."""
        for k in self.kartlar:
            kanit = k.meta.get("kanit", "")
            self.assertTrue(kanit, f"kanit yok: {k.kimlik}")
            tam = os.path.join(REPO_KOKU, kanit)
            self.assertTrue(os.path.exists(tam), f"kanit yok repo'da: {kanit} ({k.kimlik})")

    def test_alan_ceketleri_gercek_kapsam_kelimeleri(self):
        sozluk = {k.kimlik: k for k in self.kartlar}
        self.assertIn("zımpara", sozluk["marangoz_ceketi"].aciklama)
        self.assertIn("kanun", sozluk["legal_ceketi"].aciklama)
        self.assertIn("ağaç", sozluk["bahcivan_ceketi"].aciklama)
        self.assertIn("token", sozluk["kristal_tokenizer"].aciklama.lower())


# ---------------------------------------------------------------------------
# 3) Saf çekirdek (Qdrant'sız): eşik / top-k / etki aritmetiği
# ---------------------------------------------------------------------------

class TestSafCekirdek(unittest.TestCase):
    def test_etki_aritmetigi(self):
        self.assertEqual(etki_hesapla(1.5, 0.40), 0.6)        # ağırlık × skor
        self.assertEqual(etki_hesapla(1.0, 0.33), 0.33)
        self.assertAlmostEqual(etki_hesapla(0.85, 0.2), 0.17)

    def test_agirlik_etkiyi_golgeleme(self):
        """Vizyon çekirdeği: benzerlik DÜŞÜK ama ağırlığı yüksek kart,
        benzerliği yüksek hafif karttan ÖNCE gelir (ağırlıklı etki)."""
        adaylar = [("agirlikli_dusuk_benzer", 0.30, 2.0),
                   ("hafif_yuksek_benzer", 0.50, 1.0)]
        sonuc = aktivasyon_seti(adaylar, esik=0.0, top_k=2)
        self.assertEqual(sonuc[0][0], "agirlikli_dusuk_benzer")   # etki 0.6 > 0.5
        self.assertEqual(sonuc[0][2], 0.6)
        self.assertEqual(sonuc[1][0], "hafif_yuksek_benzer")

    def test_esik_alti_elensi(self):
        adaylar = [("a", 0.35, 1.0), ("b", 0.29, 1.0), ("c", 0.30, 1.0)]
        sonuc = aktivasyon_seti(adaylar, esik=0.30, top_k=5)
        self.assertEqual([s[0] for s in sonuc], ["a", "c"])  # b elendi; eşik dahil (>=)

    def test_top_k_kirpmasi(self):
        adaylar = [(f"kart_{i}", 0.9 - 0.1 * i, 1.0) for i in range(6)]
        sonuc = aktivasyon_seti(adaylar, esik=0.0, top_k=3)
        self.assertEqual(len(sonuc), 3)
        self.assertEqual([s[0] for s in sonuc], ["kart_0", "kart_1", "kart_2"])

    def test_beraberlikte_ad_kirici(self):
        adaylar = [("zeta", 0.50, 1.0), ("alfa", 0.50, 1.0)]
        sonuc = aktivasyon_seti(adaylar, esik=0.0, top_k=2)
        self.assertEqual([s[0] for s in sonuc], ["alfa", "zeta"])  # deterministik kirici

    def test_bos_aday_sonucu(self):
        self.assertEqual(aktivasyon_seti([], esik=0.2, top_k=3), [])

    def test_cekirdek_determinizmi(self):
        adaylar = [("a", 0.4, 1.2), ("b", 0.35, 1.0), ("c", 0.33, 0.85)]
        self.assertEqual(aktivasyon_seti(adaylar, esik=0.3, top_k=2),
                         aktivasyon_seti(adaylar, esik=0.3, top_k=2))

    def test_karar_notu_deterministik_ve_bilgilendirici(self):
        aktif = aktivasyon_seti([("a", 0.4, 1.2), ("b", 0.35, 1.0)], esik=0.3, top_k=2)
        n1 = karar_notu_olustur(aktif, aday_sayisi=2, esik=0.3, top_k=2)
        n2 = karar_notu_olustur(aktif, aday_sayisi=2, esik=0.3, top_k=2)
        self.assertEqual(n1, n2)
        self.assertIn("eşik=0.3", n1)
        self.assertIn("2/2", n1)
        self.assertIn("a", n1)  # zirve kart adı notta

    def test_karar_notu_bos_durumda(self):
        n = karar_notu_olustur([], aday_sayisi=0, esik=0.5, top_k=3)
        self.assertIn("0/0", n)
        self.assertEqual(n, karar_notu_olustur([], aday_sayisi=0, esik=0.5, top_k=3))

    def test_varsayilan_esik_makul(self):
        self.assertIsInstance(DEFAULT_ESIK, float)
        self.assertGreater(DEFAULT_ESIK, 0.0)
        self.assertLess(DEFAULT_ESIK, 1.0)


# ---------------------------------------------------------------------------
# 4) Aktivasyon motoru — mock bellek ile (Qdrant YOK)
# ---------------------------------------------------------------------------

class TestAktivasyonMotoru(unittest.TestCase):
    ESLESmeler = [
        ("marangoz_ceketi", 0.42, 1.25),
        ("legal_ceketi", 0.38, 1.25),
        ("kristal_tokenizer", 0.31, 1.10),
        ("olcum_kabi", 0.12, 0.85),
    ]

    def test_aktivasyon_yapisi(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("zımpara nedir", top_k=3)
        self.assertIn("aktif_yetenekler", sonuc)
        self.assertIn("karar_notu", sonuc)
        for (ad, skor, etki) in sonuc["aktif_yetenekler"]:
            self.assertIsInstance(ad, str)
            self.assertIsInstance(skor, float)
            self.assertIsInstance(etki, float)

    def test_aktivasyon_determinizmi(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=SahteTokenizer())
        s1 = ag.aktivasyon("iş kanunu maddesi", top_k=3)
        s2 = ag.aktivasyon("iş kanunu maddesi", top_k=3)
        self.assertEqual(s1, s2, "aynı sorgu → aynı sonuç (VG-2 determinizm)")

    def test_aktivasyon_topk_kirpar(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("q", top_k=2)
        self.assertEqual(len(sonuc["aktif_yetenekler"]), 2)

    def test_aktivasyon_esik_elendi(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("q", top_k=5, esik=0.30)
        adlar = [ad for (ad, _, _) in sonuc["aktif_yetenekler"]]
        self.assertNotIn("olcum_kabi", adlar)  # 0.12 < 0.30
        self.assertEqual(len(adlar), 3)

    def test_aktivasyon_bos_sonuc(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("q", top_k=3, esik=0.99)
        self.assertEqual(sonuc["aktif_yetenekler"], [])
        self.assertIn("0/4", sonuc["karar_notu"])

    def test_aktivasyon_etki_aritmetigi_ucusur(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=SahteTokenizer())
        sonuc = ag.aktivasyon("q", top_k=4, esik=0.0)
        etki_sozlugu = {ad: etki for (ad, _, etki) in sonuc["aktif_yetenekler"]}
        self.assertAlmostEqual(etki_sozlugu["marangoz_ceketi"], 1.25 * 0.42)

    def test_tokenizer_yoksa_fail_closed(self):
        ag = YetenekAgi(memory=sahte_bellek(self.ESLESmeler), tokenizer=None)
        with self.assertRaises(ValueError):
            ag.aktivasyon("q")  # sessiz boş-sonuç DÖNMEZ, kapı kapanır


# ---------------------------------------------------------------------------
# 5) Yerel tohumlama — ağ-sız gerçek Qdrant (storage_path), idempotent
# ---------------------------------------------------------------------------

class TestYerelTohumlama(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ag = YetenekAgi(storage_path=self.tmp.name, tokenizer=SahteTokenizer())

    def tearDown(self):
        self.tmp.cleanup()

    def test_tohumla_idempotent_ve_geri_okunur(self):
        kartlar = kart_envanteri()
        r1 = self.ag.tohumla(kartlar=kartlar)
        self.assertEqual(r1["kart_sayisi"], len(kartlar))
        r2 = self.ag.tohumla(kartlar=kartlar)  # ikinci koşum: ÇOĞALTMAZ
        self.assertEqual(r2["kart_sayisi"], len(kartlar), "idempotent upsert")
        adlar = {p["kimlik"] for p in self.ag.kartlari_oku(limit=len(kartlar))}
        self.assertIn("marangoz_ceketi", adlar)
        self.assertIn("legal_ceketi", adlar)

    def test_tohumlanmis_ag_aktivasyonu_tam_yol(self):
        kartlar = kart_envanteri()
        self.ag.tohumla(kartlar=kartlar)
        s1 = self.ag.aktivasyon("zımpara nedir", top_k=3)
        s2 = self.ag.aktivasyon("zımpara nedir", top_k=3)
        self.assertEqual(s1, s2)  # tam yol (tokenizer→vektör→arama→çekirdek) deterministik
        self.assertGreaterEqual(len(s1["aktif_yetenekler"]), 1)

    def test_koleksiyon_adi_yetenek_agi(self):
        self.assertEqual(KOLEKSIYON, "yetenek_agi")
        self.assertEqual(VEKTOR_BOYUTU, 768)


# ---------------------------------------------------------------------------
# 6) Gateway gözlem kancası (davranış DEĞİŞİMİ YOK)
# ---------------------------------------------------------------------------

class TestGatewayKanca(unittest.TestCase):
    def _gateway_kur(self, yetenek_agi=None):
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
            def decompile_sentence(self, tags_str):
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
        bellek = VectorMemory(collection_name="test_vg2_kanca", vector_size=768,
                              storage_path=tmp.name, host=None)
        self.addCleanup(bellek.close)
        return AgentGateway(
            model=MockModel(), tokenizer=MockTokenizer(), decompiler=MockDecompiler(),
            memory=bellek, future_train_path=os.path.join(tmp.name, "ft.jsonl"),
            device="cpu", yetenek_agi=yetenek_agi,
        )

    def test_yetenek_agi_yoksa_anahtar_yok(self):
        """Mevcut davranış birebir korunur: kanca kapalıyken yanıt şekli değişmez."""
        gw = self._gateway_kur(yetenek_agi=None)
        res = gw.ask("Kırlangıç kuyruğu nedir?", mode="RAG")
        self.assertNotIn("yetenek_aktivasyonu", res)

    def test_kanca_aktifse_gozlem_eklenir(self):
        kanca = MagicMock()
        kanca.aktivasyon.return_value = {
            "aktif_yetenekler": [("marangoz_ceketi", 0.42, 0.525)],
            "karar_notu": "test notu",
        }
        gw = self._gateway_kur(yetenek_agi=kanca)
        res = gw.ask("zımpara nedir?", mode="RAG")
        self.assertIn("yetenek_aktivasyonu", res)
        self.assertEqual(res["yetenek_aktivasyonu"]["aktif_yetenekler"],
                         [("marangoz_ceketi", 0.42, 0.525)])
        kanca.aktivasyon.assert_called_once()
        self.assertEqual(kanca.aktivasyon.call_args.kwargs.get("sorgu"), "zımpara nedir?")

    def test_kanca_hatasi_ask_i_bozmaz(self):
        kanca = MagicMock()
        kanca.aktivasyon.side_effect = RuntimeError("qdrant kapalı")
        gw = self._gateway_kur(yetenek_agi=kanca)
        res = gw.ask("zımpara nedir?", mode="RAG")  # hata YUTULMAZ-yoğ, None raporlanır
        self.assertIn("response_text", res)         # ask normale devam
        self.assertIsNone(res["yetenek_aktivasyonu"])  # gözlem başarısız → dürüst None


# ---------------------------------------------------------------------------
# 7) Gerçek koleksiyonla duman (erişilemezse SKIP)
# ---------------------------------------------------------------------------

def _gercek_tokenizer():
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary

    vocab = Vocabulary()
    vocab.load("data/rebuild/vocab_anka_r1_33114.json")
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    return KristalTokenizer(compiler, vocab)


class TestGercekKoleksiyonDumani(unittest.TestCase):
    """Seed sonrası gerçek 'yetenek_agi' koleksiyonuna karşı duman.

    Qdrant erişilemezse veya koleksiyon tohumlanmamışsa SKIP (uydurma yok).
    Koşum: QDRANT_URL=http://192.168.1.9:6333 venv/bin/python -m pytest
           tests/test_yetenek_agi.py -q -k Dumani
    """

    def test_gercek_koleksiyonda_aktivasyon_determinizmi(self):
        from qdrant_client import QdrantClient
        from src.rag.vector_memory import resolve_qdrant_url

        hedef = resolve_qdrant_url(None) or ("localhost", 6333)
        try:
            istemci = QdrantClient(host=hedef[0], port=hedef[1], timeout=5.0)
            adlar = [c.name for c in istemci.get_collections().collections]
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"Qdrant erişilemez ({hedef[0]}:{hedef[1]}): {e}")
        if KOLEKSIYON not in adlar:
            self.skipTest("yetenek_agi tohumlanmamış — önce scripts/seed_yetenek_agi.py")
        sayi = istemci.count(collection_name=KOLEKSIYON).count
        self.assertGreaterEqual(sayi, 20, f"koleksiyon çok küçük: {sayi}")

        ag = YetenekAgi(host=hedef[0], port=hedef[1], timeout=30.0,
                        tokenizer=_gercek_tokenizer())
        s1 = ag.aktivasyon("zımpara nedir", top_k=3)
        s2 = ag.aktivasyon("zımpara nedir", top_k=3)
        self.assertEqual(s1, s2, "gerçek koleksiyonda aynı sorgu aynı sonuç")
        self.assertGreaterEqual(len(s1["aktif_yetenekler"]), 1)
        for (ad, skor, etki) in s1["aktif_yetenekler"]:
            self.assertGreater(skor, 0.0)          # kosinüs yakınlık (0, 1]
            self.assertLessEqual(skor, 1.0)
            self.assertGreater(etki, 0.0)         # etki = ağırlık × skor > 0
            self.assertLessEqual(etki, 2.0)       # max ağırlık ≤ 2.0
        ag.memory.close()


if __name__ == "__main__":
    unittest.main()
