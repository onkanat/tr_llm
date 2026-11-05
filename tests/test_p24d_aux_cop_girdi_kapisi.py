#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P24d Kalem-3: aux yükseltme kapısında çöp-girdi ölçütü (TDD).

KANIT (p24c #91, ham kayıt data/eval/p24c_esik_dagilim_2026-11-04.json):
'?' sorgusu gts_bellek'te skor 1,0 ile döndü, aux yükseltmesi koşullamayı
açtı (cond False→True) ve "Model Anlamaz" A-yolu future-train kaydı üretti
(H_post 4,21 ≥ τ; skor 1,0 ≥ 0,85). Mekanizma (ölçüldü, p24d probe-1):
tek noktalama morfemi '?' extract_distinctive_roots'ta 'ayrıkcı kök' sayılır;
GTS maddesi ('ne söylüyorsun?: …') kök kümesinde '?' TAŞIR → mükemmel tek-kök
örtüşme (match_ratio=1,0 ≥ 0,5 → kök cezası YOK) + iki kolda rank-1 → ham RRF
tavanı 1,0. Yani kapı, sıfır sözcük-içerikli girdiye tam skor verebilir.

ÖLÇÜT (107'lik sette ölçüldü — 32 aux flip): aux yükseltmesi için sorguda
≥1 ALFANÜMERİK (harf/rakam) ayrıkcı kök zorunlu. Etki: 2 çöp flip ('?', '!!!')
düşer; 30/32 flip (VG-1 İHBAR/4857 dahil tüm sözcük-içerikli) KORUNUR; 0,50
aux-eşiği adayı ölçümde reddedildi ('?' 1,0 ile GEÇER, 15 meşru flip düşer).

ANKA BİRİNCİL ZİNCİRİ DOKUNULMAZ: ölçüt YALNIZ aux yükseltme bloğuna uygulanır;
birincil/yedek zincir (anka → simulasyon) ve aux-BOŞ davranış birebir.
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.rag.epistemic_agent import EpistemicCuriosityAgent
from src.rag.merak import CuriosityEngine
from src.llm.router import TriModalRouter


class _KuklaBellek:
    """hybrid_recall çağrılarını kaydeden sahte bellek (VectorMemory arayüzü)."""

    def __init__(self, ad, sonuclar):
        self.collection_name = ad
        self._sonuclar = sonuclar
        self.cagri_sayisi = 0

    def hybrid_recall(self, dense, sparse, top_k=1, query_tags=""):
        self.cagri_sayisi += 1
        return self._sonuclar


def _belge(metin, skor, kok_eslesme=True):
    return {
        "text": metin,
        "score": skor,
        "has_root_match": kok_eslesme,
        "metadata": {"crystal_tags": "test", "token_ids": [1]},
    }


def _ajan_kur(birincil, yedek=None, aux=None):
    class _KuklaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 64)

        def forward(self, x, return_hidden_states=False):
            logits = torch.zeros(x.shape[0], x.shape[1], 8)
            return (logits, None) if not return_hidden_states else (logits, None, torch.zeros(x.shape[0], x.shape[1], 64))

    return EpistemicCuriosityAgent(
        model=_KuklaModel(),
        tokenizer=None,
        memory=birincil,
        general_memory=yedek,
        auxiliary_memories=aux,
        curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5),
        router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
        tau=2.5,
        similarity_threshold=0.85,
        future_train_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_p24d_test_future.jsonl"),
        device="cpu",
    )


# p24c #91'in gerçek token dizimi: '?' → [BOS, ?, EOS]; tags: '<BOS> ? <EOS>'
COP_IDS = [2, 32139, 3]
COP_TAGS = "<BOS> ? <EOS>"
# VG-1'in meşru flip sorgusu (İHBAR/4857 profilinde)
LEX_IDS = [3, 14]
LEX_TAGS = "sözleş fes ihbar süre POSS_3SG"


class TestP24dAuxCopGirdiKapisi(unittest.TestCase):

    def test_noktalama_sorgusu_aux_yukseltilmez(self):
        """Çöp girdi ('?'): zincir kapı-altı + aux 1,0 skorlu kök-eşleşmeli aday
        döndürse BİLE yükseltme YAPILMAZ — aux hiç sorgulanmaz, zincir sonucu
        aynen döner (p24c #91 hatasına geri dönüş YOK)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        gts = _KuklaBellek("gts_bellek", [_belge(
            "ne söylüyorsun?: \"gerçek mi? doğru mu?\" anlamında kullanılan bir söz", 1.0)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[gts])
        doc, kaynak = ajan.search_memory(COP_IDS, COP_TAGS)
        self.assertEqual(gts.cagri_sayisi, 0, "çöp girdide aux sorgulanmamalı")
        self.assertNotEqual(kaynak, "gts_bellek")
        if doc is not None:
            self.assertLess(float(doc["score"]), 1.0)
        self.assertEqual(kaynak, "simulasyon_bellek")

    def test_cop_girdi_aux_yoksa_davranis_birebir(self):
        """Aux BOŞKEN çöp girdi bugünkü gibi: zincir sonucu aynen (ölçüt
        birincil zincire uygulanmaz — ANKA zinciri dokunulmaz)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[])
        doc, kaynak = ajan.search_memory(COP_IDS, COP_TAGS)
        self.assertEqual(kaynak, "simulasyon_bellek")
        self.assertAlmostEqual(doc["score"], 0.29, places=4)

    def test_cop_girdi_zincir_kullanilabilirsa_aux_dokunmaz(self):
        """Çöp girdi ama zincir 0,33 kapısını GEÇİYORSA: birincil belge aynen
        döner, aux hiç sorgulanmaz (VG-1 Kapı-2 invariant'ı çöp girdide de)."""
        anka = _KuklaBellek("anka", [_belge("ahşap...", 0.55)])
        gts = _KuklaBellek("gts_bellek", [_belge("ilgisi z:", 1.0)])
        ajan = _ajan_kur(anka, yedek=None, aux=[gts])
        doc, kaynak = ajan.search_memory(COP_IDS, COP_TAGS)
        self.assertEqual(kaynak, "anka")
        self.assertEqual(doc["text"], "ahşap...")
        self.assertEqual(gts.cagri_sayisi, 0)

    def test_lexikal_sorgu_aux_yukseltmesi_korunur(self):
        """Meşru flip (VG-1 İHBAR/4857 profili): sözcük-içerikli sorguda aux
        yükseltmesi ÖNCEKİ GİBİ çalışır (ölçüt meşru flip'i KAYBETMEZ)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        kanun = _KuklaBellek("kanun_madde", [_belge("Madde 17 - bildirim süreleri", 0.61)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[kanun])
        doc, kaynak = ajan.search_memory(LEX_IDS, LEX_TAGS)
        self.assertEqual(kaynak, "kanun_madde")
        self.assertAlmostEqual(doc["score"], 0.61, places=4)
        self.assertEqual(kanun.cagri_sayisi, 1)

    def test_rakam_iceren_sorgu_gecer(self):
        """Rakam kökü alfanümeriktir: '1+1' (tags: <BOS> 1 <NUMBER> + <EOS>
        profili) gibi sayı-içerikli sorgular kapıdan GEÇER (yalnız saf
        noktalama bloklanır)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        gts = _KuklaBellek("gts_bellek", [_belge("bir sayı tanımı...", 0.50)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[gts])
        doc, kaynak = ajan.search_memory([2, 30, 3], "<BOS> 1 <EOS>")
        self.assertEqual(kaynak, "gts_bellek")
        self.assertAlmostEqual(doc["score"], 0.50, places=4)

    def test_unlulerle_kalinan_sorgu_gecer(self):
        """'aaaaaa' gibi tekrarlı-ama-harf-içerikli kenar sorgular da kök
        ürettiği sürece kapıdan GEÇER (kara-liste DEĞİL, içerik ölçütü)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        gts = _KuklaBellek("gts_bellek", [_belge("aa ünlemi...", 0.44)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[gts])
        doc, kaynak = ajan.search_memory([2, 500, 3], "<BOS> aaaaaa <EOS>")
        self.assertEqual(kaynak, "gts_bellek")


if __name__ == "__main__":
    unittest.main()
