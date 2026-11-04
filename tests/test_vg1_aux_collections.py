#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-1 (GÖREV 4) — AUX KOLEKSİYON ZİNCİRİ TDD TESTLERİ
====================================================
Retrieval zincirine (anka_bellek → simulasyon_bellek) env/arg ile genişletilebilir
aux koleksiyonlar (gts_bellek, kanun_madde) eklenirken mevcut davranışın BİREBİR
korunmasını sınar. Ölçülmüş bugünkü taban (data/eval/vg1_retrieval_oncesi.json):
hukuk sorgularında anka kök-eşleşmesi vermiyor, zincir simulasyon'da 0,29/0,08
skorlu belgede duruyor (0,33 kapısı ALTINDA → koşullanma yok).

Kapılar:
1. Aux YOK → bugünkü davranış birebir (düşük-skorklu zincir sonucu aynen döner).
2. Zincir sonucu KULLANILABİLİR (≥0,33) → aux HİÇ sorgulanmaz.
3. Zincir sonucu kullanılamaz + aux kök-eşleşmeli ve ≥0,33 → aux belgesi yükseltilir.
4. Zincir sonucu kullanılamaz + aux kök-eşleşmeli ama <0,33 → zincir sonucu KORUNUR
   (aux koşullanma kapısını açmayan belgeyle telemetriyi değiştirmez).
5. parse_aux_collection_names: env/arg ayrıştırıcı sözleşmesi.
6. AgentGateway, auxiliary_memories'i kurduğu epistemic_agent'a taşır.
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
        future_train_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vg1_test_future.jsonl"),
        device="cpu",
    )


SORGU_IDS = [3, 14]
SORGU_TAGS = "sözleş fes ihbar süre POSS_3SG"


class TestVg1AuxKoleksiyonZinciri(unittest.TestCase):

    def test_aux_yok_bugunku_davranis_birebir(self):
        """Aux listesi boşsa zincir bugünkü gibi: anka kök-eşleşmesiz → yedek
        (simulasyon) kök-eşleşmeli 0,29 döner — ölçülmüş taban davranışı."""
        anka = _KuklaBellek("anka", [_belge("zıvana geçme...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik sözleşmesi...", 0.29)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[])
        doc, kaynak = ajan.search_memory(SORGU_IDS, SORGU_TAGS)
        self.assertEqual(kaynak, "simulasyon_bellek")
        self.assertAlmostEqual(doc["score"], 0.29, places=4)
        self.assertEqual(doc["text"], "Acentelik sözleşmesi...")

    def test_zincir_kullanilabilir_ise_aux_sorgulanmaz(self):
        """Zincir belgesi 0,33 kapısını GEÇİYORSA aux koleksiyonlar HİÇ
        sorgulanmaz (bugünkü koşullu üretim davranışı birebir korunur)."""
        anka = _KuklaBellek("anka", [_belge("ihbar süreleri... ", 0.55)])
        aux = _KuklaBellek("gts_bellek", [_belge("aux aday", 0.9)])
        ajan = _ajan_kur(anka, yedek=None, aux=[aux])
        doc, kaynak = ajan.search_memory(SORGU_IDS, SORGU_TAGS)
        self.assertEqual(kaynak, "anka")
        self.assertEqual(doc["text"], "ihbar süreleri... ")
        self.assertEqual(aux.cagri_sayisi, 0, "kullanılabilir zincir sonucunda aux sorgulanmamalı")

    def test_zincir_kullanilamaz_aux_yukseltir(self):
        """Zincir 0,29'da kalıyorsa (kapı altı) aux sorgulanır; kök-eşleşmeli
        ve 0,33+ skorlu aux belgesi zincirin yerine geçer (koşullanma açılır)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        gts = _KuklaBellek("gts_bellek", [_belge("ihbar: Bildirme...", 0.25)])
        kanun = _KuklaBellek("kanun_madde", [_belge("Madde 17 - bildirim süreleri", 0.61)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[gts, kanun])
        doc, kaynak = ajan.search_memory(SORGU_IDS, SORGU_TAGS)
        self.assertEqual(kaynak, "kanun_madde")
        self.assertAlmostEqual(doc["score"], 0.61, places=4)
        self.assertEqual(gts.cagri_sayisi, 1)
        self.assertEqual(kanun.cagri_sayisi, 1)

    def test_aux_kapi_altindaysa_zincir_sonucu_korunur(self):
        """Aux en iyi adayı 0,33 ALTINDA kalıyorsa zincir sonucu aynen döner
        (aux, koşullanmayı açmayan belgeyle telemetriyi değiştirmez)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        gts = _KuklaBellek("gts_bellek", [_belge("ihbar: Bildirme...", 0.20)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[gts])
        doc, kaynak = ajan.search_memory(SORGU_IDS, SORGU_TAGS)
        self.assertEqual(kaynak, "simulasyon_bellek")
        self.assertAlmostEqual(doc["score"], 0.29, places=4)

    def test_aux_kok_eslesmesiz_yukseltilmez(self):
        """Aux adayı 0,33+ skorlu olsa bile kök-eşleşmesi yoksa yükseltilmez
        (RRF kök cezası mekanizmasının zincir tarafı aux'a da işler)."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("Acentelik...", 0.29)])
        kanun = _KuklaBellek("kanun_madde", [_belge("ilgisiz belge", 0.60, kok_eslesme=False)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[kanun])
        doc, kaynak = ajan.search_memory(SORGU_IDS, SORGU_TAGS)
        self.assertEqual(kaynak, "simulasyon_bellek")

    def test_hepsi_miss_bugunku_none_davranisi(self):
        """Anka+yedek kök-eşleşmesiz, aux boş → (None, birincil adı) — bugünkü
        son-duruş sözleşmesi aynen."""
        anka = _KuklaBellek("anka", [_belge("zıvana...", 0.0306, kok_eslesme=False)])
        sim = _KuklaBellek("simulasyon_bellek", [_belge("balıklar...", 0.02, kok_eslesme=False)])
        ajan = _ajan_kur(anka, yedek=sim, aux=[])
        doc, kaynak = ajan.search_memory(SORGU_IDS, SORGU_TAGS)
        self.assertIsNone(doc)
        self.assertEqual(kaynak, "anka")


class TestVg1ParseAuxCollectionNames(unittest.TestCase):

    def test_parse_sozlesme(self):
        from src.gateway.agent_gateway import parse_aux_collection_names
        self.assertEqual(parse_aux_collection_names(None), [])
        self.assertEqual(parse_aux_collection_names(""), [])
        self.assertEqual(parse_aux_collection_names("   "), [])
        self.assertEqual(parse_aux_collection_names("gts_bellek,kanun_madde"), ["gts_bellek", "kanun_madde"])
        self.assertEqual(parse_aux_collection_names(" gts_bellek , kanun_madde "), ["gts_bellek", "kanun_madde"])
        self.assertEqual(parse_aux_collection_names("a,,b"), ["a", "b"])


class TestVg1GatewayAuxTasima(unittest.TestCase):

    def test_gateway_auxiliary_memories_epistemik_ajana_tasinir(self):
        from src.gateway.agent_gateway import AgentGateway

        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3}
            itos = {v: k for k, v in stoi.items()}

            def decode(self, idx):
                return self.itos.get(idx, "unk")

        class MockTokenizer:
            vocab = MockVocab()

            def encode(self, text):
                return [0, 2]

            def decode(self, ids):
                return "test"

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 768)

            def forward(self, x, return_hidden_states=False):
                b, s = x.shape
                logits = torch.ones(b, s, 10)
                hidden = torch.randn(b, s, 768)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        class MockMemory:
            collection_name = "anka_bellek"

        aux = _KuklaBellek("gts_bellek", [])
        gw = AgentGateway(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            memory=MockMemory(),
            auxiliary_memories=[aux],
            future_train_path="/tmp/_vg1_gw_future.jsonl",
            device="cpu",
        )
        self.assertEqual(gw.epistemic_agent.auxiliary_memories, [aux])


if __name__ == "__main__":
    unittest.main()
