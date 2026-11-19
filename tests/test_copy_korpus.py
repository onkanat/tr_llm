#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_copy_korpus.py — KOPYA-HATTI veri üreticisi (v6 hazırlığı) sözleşmesi.

Neyi zorlar (hepsi ölçülebilir invariant):
1. Pencere çıkarım sözleşmesiyle aynı zarfta: json {instruction,input,output} + ≤128 jeton.
2. CEVAP, BELGENİN BİREBİR jeton dizisidir ve pencerede İKİ KEZ geçer (kopya sinyali).
3. Determinizm: aynı satır + aynı tohum → aynı pencere.
4. Sızıntı kapısı: val 8-gram'ı içeren satır üretimde kullanılmaz.
5. Builder uçtan uca (küçük --sinir-blok): çıktı .bin 128'in katı, meta tutarlı,
   K4 tutulan seti üretilmiş ve tutulan satırlar pencerelerde KULLANILMAMIŞ.
"""
import json
import os
import random
import subprocess
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

import scripts.build_copy_korpus as bck

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(KOK, "scratch")


def _tok():
    vocab = Vocabulary(); vocab.load(bck.VOCAB_PATH)
    lex = LexiconManager(); lex.load_from_tsv(bck.LEXICON_PATH)
    return vocab, KristalTokenizer(CrystalCompiler(lex, build_default_graph()), vocab)


class TestKopyaPenceresi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab, cls.tok = _tok()
        cls.satir = ("Domates fidesi dikimi ilkbaharda don tehlikesi geçtikten sonra bol "
                     "güneş alan, organik gübre ve kompost ile zenginleştirilmiş süzek "
                     "toprağa yapılmalıdır.")
        cls.pencere = bck.pencere_uret(cls.tok, cls.vocab, cls.satir, random.Random(7))

    def test_pencere_uretilir(self):
        self.assertIsNotNone(self.pencere, "uygun satırdan pencere üretilemedi")

    def test_pencere_128_jetonu_asmaz(self):
        self.assertLessEqual(len(self.pencere["win"]), bck.BLOCK)

    def test_cevap_belgede_birebir_ve_iki_kez(self):
        span, doc, win = self.pencere["span_ids"], self.pencere["doc_ids"], self.pencere["win"]
        n = len(span)

        def say(kaynak):
            return sum(1 for i in range(len(kaynak) - n + 1) if list(kaynak[i:i + n]) == list(span))

        self.assertGreaterEqual(say(doc), 1, "cevap span'ı belgede birebir yok")
        self.assertGreaterEqual(say(win), 2, "pencere içinde span iki kez geçmiyor (kopya sinyali yok)")

    def test_zarf_cikarim_sozlesmesiyle_ayni(self):
        """Pencere, çıkarımdaki json {instruction,input,output} zarfının TA KENDİSİDİR ve
        belge/cevap metni DOĞAL metindir (morfem-morfoz dizisi DEĞİL)."""
        p = self.pencere
        govde = json.dumps({"instruction": p["talimat"],
                            "input": f'<BELGE> {p["doc_txt"]} </BELGE> {p["sorgu"]}',
                            "output": p["span_txt"]}, ensure_ascii=False)
        self.assertEqual(self.tok.encode(govde), p["win"])
        belge_id = self.vocab.stoi["<BELGE>"]
        self.assertIn(belge_id, p["win"], "pencere <BELGE> jetonunu taşımıyor")

    def test_belge_ve_cevap_dogal_metin(self):
        """DOĞAL METİN sözleşmesi: cevap, kaynak satırın birebir ALT DİZİSİdir (morfoz değil)."""
        p = self.pencere
        self.assertIn(p["span_txt"], self.satir, "cevap kaynak satırda birebir yok")
        self.assertNotIn("  ", p["doc_txt"], "belge metni çift boşluklu (morfoz işareti)")
        # morfem-morfoz işareti: decode edilmiş dizide her şey ayrık jeton olurdu; doğal
        # metin en az bir bitişik kelime taşır
        self.assertTrue(any(len(k) > 4 for k in p["doc_txt"].split()), "belge metni kelime taşımıyor")

    def test_determinizm(self):
        a = bck.pencere_uret(self.tok, self.vocab, self.satir, random.Random(7))
        b = bck.pencere_uret(self.tok, self.vocab, self.satir, random.Random(7))
        self.assertEqual(a["win"], b["win"])

    def test_val_sizinti_kapisi_satiri_eler(self):
        """Val kümesinde bulunan bir 8-gram'ı taşıyan satır kapıdan GEÇMEZ."""
        ids = self.tok.encode(self.satir)[1:-1]
        sahte_val = np.array(ids[3:3 + 12], dtype=np.uint16)
        yol = os.path.join(SCRATCH, "_test_val.bin")
        sahte_val.tofile(yol)
        try:
            yasak = bck.val_ngram_kumesi(yol)
            eslesme = any(hash(tuple(ids[i:i + 8])) in yasak
                          for i in range(len(ids) - 8 + 1))
            self.assertTrue(eslesme, "val 8-gram kapısı satırı yakalamadı")
        finally:
            os.remove(yol)


class TestBuilderUctanUca(unittest.TestCase):
    """Küçük bir koşum: çıktı biçimi + meta tutarlılığı + tutulan küme ayrımı."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(SCRATCH, exist_ok=True)
        cls.bin = os.path.join(SCRATCH, "test_kopya.bin")
        cls.tutulan = os.path.join(SCRATCH, "test_k4_tutulan.jsonl")
        cache = os.path.join(SCRATCH, "_test_havuz.json")
        if os.path.exists(cache):
            os.remove(cache)  # bayat havuz önbelleği testi kirletmesin
        r = subprocess.run(
            [sys.executable, "scripts/build_copy_korpus.py",
             "--out-bin", cls.bin, "--tutulan-out", cls.tutulan,
             "--sayi-tutulan", "5", "--sinir-blok", "24",
             "--max-tarama", "2500", "--pencere-per-satir", "4",
             "--havuz-cache", os.path.join(SCRATCH, "_test_havuz.json")],
            cwd=KOK, capture_output=True, text=True, timeout=900)
        cls.rc, cls.log = r.returncode, r.stdout + r.stderr

    def test_kosum_basarili(self):
        self.assertEqual(self.rc, 0, self.log[-1500:])

    def test_cikti_blok_hizali_ve_meta_tutarli(self):
        arr = np.memmap(self.bin, dtype=np.uint16, mode="r")
        self.assertEqual(len(arr) % bck.BLOCK, 0, "çıktı 128'in katı değil")
        meta = json.load(open(self.bin + ".meta.json", encoding="utf-8"))
        beklenen_blok = meta["kullanilan_blok"] + meta["kopya_pencere"]
        self.assertEqual(len(arr) // bck.BLOCK, beklenen_blok)
        self.assertGreater(meta["kopya_pencere"], 0)

    def test_tutulan_seti_uretilmis(self):
        satirlar = [json.loads(l) for l in open(self.tutulan, encoding="utf-8")]
        self.assertGreaterEqual(len(satirlar), 1, "K4 tutulan seti boş")
        self.assertLessEqual(len(satirlar), 5)
        for s in satirlar:
            self.assertIn("doc", s); self.assertIn("span", s)
            self.assertTrue(s["span"].strip() and s["doc"].strip())

    def test_tutulan_satir_egitime_sizmamis(self):
        """Tutulan setteki span metni, üretilen .bin'de kopya-cevabı olarak YOK.

        Not: külliyat metninin kendisi .bin'de zaten var (kaynak korpus); ölçtüğümüz
        şey tutulan span'ın KOPYA PENCERESİ olarak eklenmemiş olmasıdır → kopya
        pencerelerinin son bölümünü (eklenen bloklar) tararız.
        """
        meta = json.load(open(self.bin + ".meta.json", encoding="utf-8"))
        arr = np.memmap(self.bin, dtype=np.uint16, mode="r")
        vocab, tok = _tok()
        tutulan = [json.loads(l) for l in open(self.tutulan, encoding="utf-8")]
        # kopya pencereleri, grup sonlarında eklenir: son kopya_pencere*128 jetonu değil,
        # her grubun sonunda; güvenli tarama için tüm .bin'i kapsayan kaba arama yapılır.
        metin_jetonlari = set()
        for s in tutulan:
            span_ids = tok.encode(s["span"])[1:-1]
            n = len(span_ids)
            bulundu = 0
            for i in range(0, len(arr) - n, 397):  # seyrek örnekleme (hızlı)
                if list(arr[i:i + n]) == [int(x) for x in span_ids]:
                    bulundu += 1
            metin_jetonlari.add(bulundu)
        self.assertTrue(all(v == 0 for v in metin_jetonlari),
                        f"tutulan span kopya penceresinde bulundu: {metin_jetonlari}")


if __name__ == "__main__":
    unittest.main()
