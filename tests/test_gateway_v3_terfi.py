#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_gateway_v3_terfi.py — CANLI GATEWAY TERFİSİ (2026-11-04, kullanıcı talimatı)
=====================================================================================
Terfi hedefi: router v2 → v3 (data/anka_router_v3.pt) + legal uzmanı
(data/anka_legal_replay.pt) — taban (anka_taban_pure.pt) ve mevcut 2 uzman
(anka_marangoz.pt / anka_bahcivan.pt) DEĞİŞMEZ.

Kapsam (görev talimatı):
1. create_default varsayılan router'ı v3'ü çözmeli;
2. baslat_canli_gateway çözümü (arg varsayılanı + sd-sha çıpası + expert sözlüğü)
   v3 + legal uzmanını içermeli;
3. router v3 / legal ceketi artifact-guard: dosya VAR, yol DOĞRU, yüklenebilir,
   sözlük uyumlu (33114);
4. 'legal' birincil uzmanda MoE dağıtım ölçümü (uzmanı YOKSA tabana sessiz
   düşüş VAR — epistemic_agent.py:459 `.get(primary_expert, self.model)`;
   uzmanı VARSA legal ceketine gider).

AÇIK UYARI (hüküm, KALDI 0/3): legal ceketi (replay) P2.5 kapılarından GEÇMEDİ
(ΔCE −0,306/−0,50; kanonik-20 0/20; OOD 0,365/0,40 — bkz.
data/eval/p25_router_v3_hukum_2026-11-04.json ve CHANGELOG 1.17.0). Bu terfi
KULLANICI TALİMATIYLA yapılır; router v3'ün kendisi GEÇTİ (8/10 legal, diğer
4 sınıf v2-paritesi 1,0). Uyarı kodda ve raporda AÇIK kalır.
"""

import hashlib
import inspect
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.agent_gateway import AgentGateway
from src.rag.vector_memory import VectorMemory
from src.llm.router import TriModalRouter
from src.rag.merak import CuriosityEngine

REPO_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTER_V3 = os.path.join(REPO_KOKU, "data", "anka_router_v3.pt")
LEGAL_CEKET = os.path.join(REPO_KOKU, "data", "anka_legal_replay.pt")
VOCAB_JSON = os.path.join(REPO_KOKU, "data", "rebuild", "vocab_anka_r1_33114.json")

# P25 hüküm çıpası (K4 DETERMİNİZM): data/eval/p25_router_v3_hukum_2026-11-04.json
BEKLENEN_V3_SD_SHA = "4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5"


def _sd_sha(sd):
    """K8-tarifi — scripts/baslat_canli_gateway.py:_sd_sha ile BİREBİR:
    sıralı anahtar-adı + shape + baytlar üzerinden sha256."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class TestGatewayV3Terfi(unittest.TestCase):
    """Terfi çözümü: create_default + baslat betiği router_v3 + legal uzmanını çözer."""

    def test_create_default_router_varsayilani_v3(self):
        """create_default'un router_state_path varsayılanı v3'e terfi etti (v2 değil)."""
        sig = inspect.signature(AgentGateway.create_default)
        varsayilan = sig.parameters["router_state_path"].default
        self.assertEqual(
            varsayilan, "data/anka_router_v3.pt",
            f"create_default router varsayılanı v3 olmalı, şu an: {varsayilan!r}")

    def test_baslat_router_varsayilani_v3_ve_sd_sha_cipasi(self):
        """baslat betiği: --router varsayılanı v3 VE sd-sha çıpası v3'e güncellendi;
        çıpa gerçek dosyayla DOĞRULANIR (sessiz-yankı değil — yükleme)."""
        import scripts.baslat_canli_gateway as bcg
        self.assertEqual(bcg.DEFAULT_ROUTER_PATH, "data/anka_router_v3.pt")
        self.assertEqual(bcg.BEKLENEN_ROUTER_SD_SHA, BEKLENEN_V3_SD_SHA)
        # Çıpa ↔ gerçek dosya tutarlığı (dosya VAR → yükle ve ölç)
        self.assertTrue(os.path.exists(ROUTER_V3), f"router v3 yok: {ROUTER_V3}")
        sd = torch.load(ROUTER_V3, map_location="cpu")
        self.assertEqual(_sd_sha(sd), bcg.BEKLENEN_ROUTER_SD_SHA,
                         "v3 sd-sha çıpası gerçek dosyayla uyuşmuyor")

    def test_baslat_cozumu_legal_uzmani_icerir(self):
        """baslat expert çözümü: gardener+carpenter DEĞİŞMEDİ, legal EKLENDİ."""
        import scripts.baslat_canli_gateway as bcg
        cozum = bcg.build_expert_model_paths()
        self.assertEqual(cozum.get("gardener"), "data/anka_bahcivan.pt")
        self.assertEqual(cozum.get("carpenter"), "data/anka_marangoz.pt")
        self.assertIn("legal", cozum, "baslat çözümünde 'legal' uzmanı EKLENMELİ")
        self.assertEqual(cozum["legal"], "data/anka_legal_replay.pt")
        for ad, yol in cozum.items():
            self.assertTrue(os.path.exists(os.path.join(REPO_KOKU, yol)),
                            f"uzman checkpoint'i yok: {ad} -> {yol}")

    def test_legal_uzman_checkpoint_yuklenir_ve_sozluk_uyumlu(self):
        """Legal ceketi artifact-guard: yüklenebilir VE embedding satırı sözlükle
        AYNI (T-0087 sessiz-kırpma kapısının uzman tarafı)."""
        import json
        self.assertTrue(os.path.exists(LEGAL_CEKET), f"legal ceketi yok: {LEGAL_CEKET}")
        with open(VOCAB_JSON, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        sozluk_satir = len(vocab["stoi"])
        sd = torch.load(LEGAL_CEKET, map_location="cpu")
        emb = sd.get("embedding.embedding.weight")
        self.assertIsNotNone(emb, "legal ceketinde 'embedding.embedding.weight' yok")
        self.assertEqual(
            int(emb.shape[0]), sozluk_satir,
            f"sözlük/checkpoint uyumsuzluğu: ceket={int(emb.shape[0])} != sözlük={sozluk_satir}")

    def test_router_v3_5sinif_semasi(self):
        """Router v3 5-sınıf şeması: w_g 5×256, 9 anahtar (legal = sınıf 4)."""
        self.assertTrue(os.path.exists(ROUTER_V3))
        sd = torch.load(ROUTER_V3, map_location="cpu")
        self.assertEqual(tuple(sd["w_g.weight"].shape), (5, 256),
                         "router v3 5-sınıf (w_g 5×256) olmalı")
        self.assertEqual(len(sd), 9)

    def test_legal_birincilinde_dagilim_uzmanli_ve_uzmansiz(self):
        """ÖLÇÜM (mevcut MoE dağıtımı — epistemic_agent.py:459): router 'legal'i
        birincil seçtiğinde (a) expert_models'te 'legal' YOKSA üretim TABANA
        sessizce düşer (bugünkü tuzak), (b) 'legal' VARSA legal ceketi üretir.
        5-sınıf router legal'i zorla birincil yapar (w_g yalnız sınıf-4'a açık)."""
        class MockVocab:
            stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3,
                    "iş": 4, "sözleşmesi": 5, "bilgi": 6}
            itos = {v: k for k, v in stoi.items()}
            def decode(self, idx): return self.itos.get(idx, "bilgi")

        class MockTokenizer:
            vocab = MockVocab()
            def encode(self, text): return [0, 4, 5, 2]
            def decode(self, ids): return "iş sözleşmesi"

        class MockDecompiler:
            def decompile_sentence(self, tags_str, capitalize=True): return "iş sözleşmesi"

        class TrackingMockModel(torch.nn.Module):
            def __init__(self, ad):
                super().__init__()
                self.ad = ad
                self.generation_called = False
                self.embedding = torch.nn.Embedding(10, 64)

            def forward(self, x, return_hidden_states=False):
                if not return_hidden_states:
                    self.generation_called = True
                b, s = x.shape
                logits = torch.ones(b, s, 10)
                hidden = torch.randn(b, s, 64)
                return (logits, None, hidden) if return_hidden_states else (logits, None)

        def router_legal_birincil():
            r = TriModalRouter(
                prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32,
                num_experts=5, top_k=2,
                expert_names=["grammar_core", "pedagogy", "carpenter", "gardener", "legal"])
            with torch.no_grad():
                # Kanal-sabıtı: w_p.bias e_route'a sıfır-olmayan sabit enjekte eder
                # (LayerNorm sıfır-vektörü sıfırlar — w_g tek başına yetmez);
                # w_g yalnız sınıf-4 (legal) logitini bu sabite açar.
                r.w_p.weight.zero_(); r.w_p.bias.zero_()
                r.w_p.bias[0] = 5.0
                r.w_m.weight.zero_(); r.w_m.bias.zero_()
                r.w_r.weight.zero_(); r.w_r.bias.zero_()
                r.w_g.weight.zero_()
                r.w_g.weight[4, 0] = 5.0  # sınıf-4 (legal) tek açık logit
            return r

        def kur_gw(uzmanlar):
            taban = TrackingMockModel("taban")
            gw = AgentGateway(
                model=taban, tokenizer=MockTokenizer(), decompiler=MockDecompiler(),
                memory=VectorMemory(collection_name="v3_terfi_mock", vector_size=64,
                                   storage_path=None, host=None),
                expert_models=uzmanlar, device="cpu")
            gw.epistemic_agent.router = router_legal_birincil()
            gw.epistemic_agent.curiosity_engine = CuriosityEngine(
                hidden_dim=64, curiosity_dim=64, tau=2.5)
            return gw, taban

        # (a) BUGÜNKÜ DAVRANIŞ: 'legal' uzmanı YOK → üretim tabana düşer (sessiz)
        gw, taban = kur_gw({"carpenter": TrackingMockModel("marangoz")})
        res = gw.ask("İş sözleşmesi feshinde ihbar süresi nedir?", mode="DIRECT")
        self.assertEqual(res["active_expert"], "legal", "router legal'i birincil seçmeli")
        self.assertTrue(taban.generation_called,
                        "'legal' uzmanı yokken üretim TABANA sessizce düşer (ölçülen bugün-davranışı)")

        # (b) TERFİ SONRASI: 'legal' uzmanı VAR → üretim legal ceketinden
        legal = TrackingMockModel("legal_ceket")
        gw2, taban2 = kur_gw({"legal": legal})
        res2 = gw2.ask("İş sözleşmesi feshinde ihbar süresi nedir?", mode="DIRECT")
        self.assertEqual(res2["active_expert"], "legal")
        self.assertTrue(legal.generation_called, "legal uzmanı VARSA üretim legal ceketinden")
        self.assertFalse(taban2.generation_called)


if __name__ == "__main__":
    unittest.main()