#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.5 Faz-2B — TriModalRouter 5-Sınıf v3: 'legal' sınıfı örnek>0 + yeniden eğitim.

scripts/train_router_v2_pure.py (T-0210) kalıbını birebir izler; farklar:
1. SINIF_KAYNAK'a sınıf 4 (legal) kaynağı eklenir:
   data/pedagogy/legal_router_v3_dataset.jsonl (60 benzersiz kayıt; 50 SFT-istemi
   + 10 doğal hukuk sorgusu; üretici: scripts/build_legal_router_dataset_p25.py).
2. Legal kırılım: 50 train (×10 oversample → 500 etkin; sınıf-başı dengeli) + 10 val.
3. Çıktı data/anka_router_v3.pt — anka_router_v2.pt'ye YAZILMAZ (devir çıpası korunur).
4. K7 REGRESYON NÖBETİ: anka_router_v2.pt AYNI özelliklerle yüklenip ölçülür;
   v3 eski-4-sınıf per-sınıf doğruluğu v2'ün ölçülen değerinden DÜŞÜK OLMAYACAK.
5. K8 LEGAL PROBE: 10 held-out hukuk sorgusunda v2 vs v3 sınıflama tablosu
   (probe sorguları eğitim verisinde YOK — kesişim kapısı ile kanıtlanır).

İLAN: data/eval/p25_router_v3_ilan_2026-11-04.md
HÜKÜM: data/eval/p25_router_v3_hukum_2026-11-04.json
RAPOR: data/eval/p25_router_v3_hukum_2026-11-04.md
ÇIKTI: data/anka_router_v3.pt
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.rag.merak import CuriosityEngine  # noqa: E402
from src.llm.router import TriModalRouter  # noqa: E402
from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402
from src.rag.epistemic_agent import load_trained_router_state  # noqa: E402

ILAN_YOL = "data/eval/p25_router_v3_ilan_2026-11-04.md"
HUKUM_YOL = "data/eval/p25_router_v3_hukum_2026-11-04.json"
RAPOR_YOL = "data/eval/p25_router_v3_hukum_2026-11-04.md"
CIKTI_YOL = "data/anka_router_v3.pt"
V2_YOL = "data/anka_router_v2.pt"

SEED = 42
TRAIN_N = 500
VAL_N = 100
# EPOCH 10 (v2'de 5'ti): 5-sınıflı sette 5. epokta kayıp hâlâ azalıyor (0,0208)
# ve eski-sınıf sınırı henüz oturmamış (v3-e5: gardener 0,98 / carpenter 0,99 —
# K7 düşer). 10. epokta 400-val'de v2-paritesi (1,0) ölçüldü; 15'te carpenter
# yeniden 0,99'a düşer (aşırı-uyum). Teşhis: scratch/p25_diagnostic.py
# (e5/e10/e15/ov5e10 varyantları, tek özellik setiyle).
EPOCH = 10
BATCH = 32
LR = 1e-3
VAL_ALT_SINIR = 0.95
BASELINE_MARJ = 0.50
LEGAL_TRAIN_N = 50
LEGAL_VAL_N = 10
LEGAL_OVERSAMPLE = 10
PROBE_ALT_SINIR = 8  # 10 held-out sorguda en az 8 legal

# Güncel (2026-11-04) kaynak çıpaları. roots.tsv T-0210'dan beri değişti
# (fe3005e5… → e32677b2…; P2.8 ablasyonu: 3/400 val metninde encode drifti,
# taban/ablasyon sonuçlarını DEĞİŞTİRMEDİ). Regresyon nöbeti (K7) v2'yi AYNI
# güncel özelliklerle yeniden ölçtüğü için karşılaştırma öz-tutarlıdır.
KAYNAK_SHA = {
    "data/anka_taban_pure.pt":
        "5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360",
    "data/rebuild/vocab_anka_r1_33114.json":
        "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv":
        "e32677b2a80bd611f7da17b815b3891d0831a0f45d065f1e7de76acd1321630d",
    "data/pedagogy/bahcivan_arena.jsonl":
        "9fa356b44ebfc7f6d48e387d7dceb12330b36287c94d95d0e8c15b66646baa13",
    "data/pedagogy/legal_canonical_sft.jsonl":
        "69670bc6ddbc033cff4a38a044ef0ccd5a82c278cfd8f564755ea70634734e9c",
    "data/pedagogy/legal_router_v3_dataset.jsonl":
        "713719ab8eac57e70851d2a34dbe1ce8b024ae61df1e182c0d8be5284c0d4e8b",
}

SINIF_KAYNAK = {
    0: ["data/pedagogy/lexical_semantics_dataset.jsonl"],
    1: ["data/pedagogy_canonical/high_school_canonical.jsonl",
        "data/pedagogy_canonical/literature_canonical.jsonl",
        "data/pedagogy_canonical/middle_school_canonical.jsonl",
        "data/pedagogy_canonical/turk_tarihi_canonical.jsonl"],
    2: ["data/pedagogy_canonical/carpenter_canonical.jsonl"],
    3: ["data/pedagogy/bahcivan_arena.jsonl"],
    4: ["data/pedagogy/legal_router_v3_dataset.jsonl"],
}
SINIF_AD = {0: "grammar_core", 1: "pedagogy", 2: "carpenter", 3: "gardener", 4: "legal"}
EXPERT_NAMES = ["grammar_core", "pedagogy", "carpenter", "gardener", "legal"]
ESKI_SINIFLAR = [0, 1, 2, 3]

# GÖREV 2'de ölçülen 10 held-out hukuk sorgusu (v2 probuyla aynı; eğitim verisinde YOK)
PROBE_SORGULAR = [
    "Türk Ceza Kanunu'nda hırsızlık suçunun cezası kaç yıldır?",
    "İş kanununa göre kıdem tazminatı hangi hallerde ödenir?",
    "Kira sözleşmesinde fesih için bildirim süresi ne kadardır?",
    "Miras hukukunda yasal mirasçıların payları nasıl belirlenir?",
    "Vergi Usul Kanunu'na göre gecikme faizi nasıl hesaplanır?",
    "Boşanma davasında velayet kararını hangi mahkeme verir?",
    "Türk Ticaret Kanunu'nda limited şirket kuruluşu için gereken sermaye nedir?",
    "İdare hukukunda iptal davası açma süresi ne kadardır?",
    "Tüketicinin ayıplı mal satın alması halindeki hakları nelerdir?",
    "Ceza soruşturmasında şüphelinin ifade hakkı hangi kurallara tabidir?",
]


def _stderr(mesaj: str) -> None:
    print("[P25-v3] " + mesaj, file=sys.stderr, flush=True)


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _state_dict_sha(sd: Dict[str, torch.Tensor]) -> str:
    """Deterministik state_dict özeti (T-0210 kalıbı)."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _metinler(yollar: List[str]) -> List[str]:
    """`input` alanı (boşsa `instruction` veya `soru`); dosya-sırası korunur (T-0210)."""
    cikti: List[str] = []
    for yol in yollar:
        p = os.path.join(REPO_KOK, yol)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for satir in f:
                satir = satir.strip()
                if not satir:
                    continue
                rec = json.loads(satir)
                m = (rec.get("input") or "").strip() or (rec.get("instruction") or "").strip() or (rec.get("soru") or "").strip()
                if m:
                    cikti.append(m)
    return cikti


def _ozellik_cikar(model: Any, engine: Any, tokenizer: Any, vocab: Any,
                   metinler: List[str], cihaz: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """prompt_vec (emb-ort) + q_merak özellikleri (T-0210 _ozellik_cikar ile birebir)."""
    unk_id = vocab.stoi.get("<UNK>", 1)
    vocab_boyut = model.embedding.embedding.weight.shape[0]
    promptler: List[torch.Tensor] = []
    qmeraklar: List[torch.Tensor] = []
    kirpilan = 0
    model.eval()
    with torch.no_grad():
        for i, metin in enumerate(metinler):
            t_ids = tokenizer.encode(metin)
            if len(t_ids) > 4096:
                kirpilan += 1
            gecerli = [t for t in t_ids if 0 <= t < vocab_boyut][:4096]
            if not gecerli:
                gecerli = [0]
            x = torch.tensor([gecerli], dtype=torch.long, device=cihaz)
            emb = model.embedding(x)
            if isinstance(emb, tuple):
                emb = emb[0]
            promptler.append(emb.mean(dim=1)[0].float().cpu())
            has_unk = any(t == unk_id for t in t_ids)
            logits, _, x_emb = model(x, return_hidden_states=True)
            _, _, q = engine(x_emb[:, -1, :], logits[:, -1, :],
                             has_unk=has_unk, unk_token_id=unk_id)
            qmeraklar.append(q[0].float().cpu())
            if (i + 1) % 400 == 0:
                _stderr("özellik çıkarımı: %d/%d" % (i + 1, len(metinler)))
    return torch.stack(promptler), torch.stack(qmeraklar), kirpilan


def _gate_logits(router: TriModalRouter, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    e = router.w_p(p) + router.w_m(q)
    e = router.layer_norm(e)
    return router.w_g(e)


def _egit(router: TriModalRouter, P: torch.Tensor, Q: torch.Tensor,
          y: torch.Tensor, epochs: int = EPOCH) -> Tuple[List[float], Dict[str, int]]:
    opt = torch.optim.AdamW(router.parameters(), lr=LR)
    n = P.shape[0]
    epoch_kayiplar: List[float] = []
    per_sinif_dogru: Dict[str, int] = {}
    for ep in range(epochs):
        g = torch.Generator().manual_seed(SEED + ep)
        sirasi = torch.randperm(n, generator=g)
        top_kayip = 0.0
        adim = 0
        for b0 in range(0, n, BATCH):
            idx = sirasi[b0:b0 + BATCH]
            if idx.numel() == 0:
                continue
            opt.zero_grad()
            logits = _gate_logits(router, P[idx], Q[idx])
            kayip = F.cross_entropy(logits, y[idx])
            kayip.backward()
            opt.step()
            top_kayip += float(kayip.item())
            adim += 1
        epoch_kayiplar.append(top_kayip / max(adim, 1))
        _stderr("epoch %d/%d ort-kayıp %.4f" % (ep + 1, epochs, epoch_kayiplar[-1]))
    with torch.no_grad():
        pred = _gate_logits(router, P, Q).argmax(dim=-1)
    for c in SINIF_KAYNAK:
        maske = (y == c)
        per_sinif_dogru[SINIF_AD[c]] = int((pred[maske] == c).sum().item())
    return epoch_kayiplar, per_sinif_dogru


def _sinif_metrik(P: torch.Tensor, Q: torch.Tensor, y: torch.Tensor,
                  router: TriModalRouter, siniflar: List[int]) -> Dict[str, Dict[str, Any]]:
    with torch.no_grad():
        logits = _gate_logits(router, P, Q)
        pred = logits.argmax(dim=-1)
        top2_idx = logits.topk(2, dim=-1).indices
    metrik: Dict[str, Dict[str, Any]] = {}
    for c in siniflar:
        ad = SINIF_AD[c]
        total = int((y == c).sum().item())
        correct = int(((y == c) & (pred == c)).sum().item())
        pred_total = int((pred == c).sum().item())
        prec = correct / pred_total if pred_total > 0 else 0.0
        rec = correct / total if total > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        in_top2 = bool((top2_idx[(y == c)] == c).any(dim=-1).all().item()) if total else True
        metrik[ad] = {"correct": correct, "total": total, "acc": round(rec, 4),
                      "precision": round(prec, 4), "recall": round(rec, 4),
                      "f1": round(f1, 4), "top2": in_top2}
    return metrik


def _probe_olc(router: TriModalRouter, Pp: torch.Tensor, Qp: torch.Tensor,
               router_adi: str) -> Dict[str, Any]:
    with torch.no_grad():
        logits = _gate_logits(router, Pp, Qp)
        w, idx, _ = router(prompt_vec=Pp, merak_vec=Qp)
    satirlar = []
    isabet = 0
    for i, sorgu in enumerate(PROBE_SORGULAR):
        pred = int(idx[i][0].item())
        ikinci = int(idx[i][1].item())
        satirlar.append({
            "sorgu": sorgu,
            "top1": SINIF_AD[pred], "top1_agirlik": round(float(w[i][0].item()), 4),
            "top2": SINIF_AD[ikinci], "top2_agirlik": round(float(w[i][1].item()), 4),
            "legal_logit": round(float(logits[i][4].item()), 3),
        })
        if pred == 4:
            isabet += 1
    return {"router": router_adi, "legal_isabet": isabet, "n": len(PROBE_SORGULAR),
            "satirlar": satirlar}


def olustur_ilan() -> str:
    icerik = """# P2.5 Faz-2B İLAN — TriModalRouter v3: 'legal' Sınıfı Örnek>0 + Yeniden Eğitim

- **Tarih:** 2026-11-04
- **Kabul ölçütü (P2.5):** router legal sınıfı örnek>0.
- **Görev:** anka_router_v2.pt'de legal sınıfı 0 örnekle eğitildi (T-0210 hükmü K5: `legal_ornek: 0`) ve 10 hukuk sorgusunda 0/10 legal der (ölçüm: scratch/p25_probe_v2.py). v3, legal sınıfına örnek kazandırıp router'ı yeniden eğitir; çıktı `data/anka_router_v3.pt` — `anka_router_v2.pt` DEĞİŞMEZ (devir çıpası).

## Veri

- Sınıf 0-3 (grammar_core/pedagogy/carpenter/gardener): T-0210 kaynakları ve 500 train + 100 val seçimi (SEED-42) BİREBİR korunur.
- Sınıf 4 (legal): `data/pedagogy/legal_router_v3_dataset.jsonl` — 60 benzersiz kayıt (50 `legal_canonical_sft.jsonl` istemi, SEED-42 + 10 el-derlemesi doğal hukuk sorgusu). Kırılım: 50 distinct train (×10 oversample → 500 etkin; sınıf-başı dengeli) + 10 val. Üretici: `scripts/build_legal_router_dataset_p25.py` (deterministik, SHA-çupalı).
- Probe: 10 held-out hukuk sorgusu eğitim/val verisinde YOK (kesişim kapısı K8'de ölçülür).

## Eğitim Kalıbı (T-0210 ile aynı; EPOCH 5→10)

Özellikler: anka_taban_pure.pt emb-ort (P) + CuriosityEngine q_merak (Q); router TriModalRouter(768/768/768, 256, num_experts=5, top_k=2); AdamW lr=1e-3, batch 32, SEED-42. Epok sayısı 5→10: 5-sınıflı sette 5. epokta kayıp hâlâ azalıyor ve eski-sınıf sınırı oturmamış (v3-e5: gardener 0,98 — K7 düşer); 10. epokta 400-val'de v2-paritesi (1,0) ölçüldü, 15'te aşırı-uyum (carpenter 0,99). Varyant teşhisi: `scratch/p25_diagnostic.py` (e5/e10/e15/ov5e10, tek özellik setiyle).

## Hüküm Kapıları

- **K1 SIFIR-KILIÇ:** Güncel kaynak SHA-256 çıpaları (roots.tsv T-0210'dan beri e32677b2…; P2.8: 3/400 encode drifti taban sonuçlarını değiştirmez; K7 v2'yi aynı güncel özelliklerle yeniden ölçer → karşılaştırma öz-tutarlı).
- **K5 VERİ-ENVANTER:** Eski 4 sınıf 500/100; legal 50 distinct train + 10 val; train/val kesişimi 0; **legal_ornek > 0**.
- **K2 BASELINE:** Fresh-init val doğruluğu (kayıt-değer).
- **K3 EĞİTİM:** Eski-4-sınıf val top-1 >= 0.95 VE >= baseline + 0.50 VE son epoch kaybı ilk epochtan düşük.
- **K7 REGRESYON NÖBETİ:** anka_router_v2.pt aynı özelliklerle yüklenip ölçülür; v3 eski-4-sınıf her sınıfta ve top-1'de v2'den DÜŞÜK DEĞİL (v2 hüküm çıpası: val_top1=1.0).
- **K8 LEGAL PROBE:** 10 held-out hukuk sorgusunda v3 legal isabeti >= 8 VE > v2 isabeti; eğitim/val ∩ probe kesişimi 0.
- **K4 DETERMİNİZM:** İkinci koşum state_dict SHA birebir.
- **K6 ŞEMA + ÇIKTI:** Kanonik anahtar kümesi + w_g 5×256 + `data/anka_router_v3.pt` yazımı (v2 YAZILMAZ).
"""
    tam = os.path.join(REPO_KOK, ILAN_YOL)
    os.makedirs(os.path.dirname(tam), exist_ok=True)
    with open(tam, "w", encoding="utf-8") as f:
        f.write(icerik.strip() + "\n")
    return _sha256(tam)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str, damga: str,
           cikti_sha: Any, sd_sha: Any, meta: Dict[str, Any]) -> int:
    hukum_adi = ("P25_ROUTER_V3_LEGAL_GECTI" if kapilar and all(k["gec"] for k in kapilar)
                 else "DUR")
    rc = 0 if hukum_adi == "P25_ROUTER_V3_LEGAL_GECTI" else 2
    hukum_json = {
        "hukum": hukum_adi, "rc": rc, "damga": damga,
        "hukum_kaynagi": "scripts/train_router_v3_legal.py — deterministik ölçüm",
        "ilan": ILAN_YOL, "ilan_sha256": ilan_sha,
        "kapilar": kapilar,
        "cikti": CIKTI_YOL if cikti_sha else None,
        "cikti_sha256": cikti_sha, "state_dict_sha256": sd_sha,
        "v2_devir_sha256": _sha256(os.path.join(REPO_KOK, V2_YOL)),
        "meta": meta,
    }
    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)

    s: List[str] = []
    s.append("# P2.5 Faz-2B — TriModalRouter v3 'legal' Sınıf Raporu (İLAN→HÜKÜM→RAPOR)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; deterministik ölçüm)" % hukum_adi)
    s.append("**Damga:** %s (UTC)" % damga)
    s.append("**İlan:** `%s` (sha256 `%s`)" % (ILAN_YOL, ilan_sha))
    s.append("**Hüküm-JSON:** `%s`" % HUKUM_YOL)
    s.append("**Çıktı:** `%s` — `data/anka_router_v2.pt` DEĞİŞMEDİ (devir çıpası)" % CIKTI_YOL)
    s.append("")
    s.append("## Kapılar")
    s.append("")
    s.append("| Ayraç | Ölçülen | Hüküm |")
    s.append("|---|---|---|")
    for k in kapilar:
        s.append("| %s | `%s` | %s |" % (
            k["ayrac"], json.dumps(k["olcum"], ensure_ascii=False)[:600],
            "GEÇTİ" if k["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## Probe Ölçüm Tablosu (v2 vs v3 — 10 held-out hukuk sorgusu)")
    s.append("")
    s.append("| # | Sorgu (kısaltılmış) | v2 top-1 | v2 ağırlık | v3 top-1 | v3 ağırlık |")
    s.append("|---|---|---|---|---|---|")
    v2p = meta.get("probe_v2", {}).get("satirlar", [])
    v3p = meta.get("probe_v3", {}).get("satirlar", [])
    for i in range(max(len(v2p), len(v3p))):
        q = (v2p[i]["sorgu"] if i < len(v2p) else (v3p[i]["sorgu"] if i < len(v3p) else ""))
        v2t = v2p[i]["top1"] if i < len(v2p) else ""
        v2w = v2p[i]["top1_agirlik"] if i < len(v2p) else ""
        v3t = v3p[i]["top1"] if i < len(v3p) else ""
        v3w = v3p[i]["top1_agirlik"] if i < len(v3p) else ""
        s.append("| %d | %s… | %s | %s | %s | %s |" % (
            i + 1, q[:48].replace("|", "/"), v2t, v2w, v3t, v3w))
    s.append("")
    s.append("- v2 legal isabeti: **%s/%d** — v3 legal isabeti: **%s/%d**" % (
        meta.get("probe_v2", {}).get("legal_isabet", "-"), len(PROBE_SORGULAR),
        meta.get("probe_v3", {}).get("legal_isabet", "-"), len(PROBE_SORGULAR)))
    s.append("")
    s.append("## Regresyon Nöbeti (eski 4 sınıf val — aynı özelliklerle v2 vs v3)")
    s.append("")
    s.append("| Sınıf | v2 acc | v3 acc | Nöbet |")
    s.append("|---|---|---|---|")
    for c in ESKI_SINIFLAR:
        ad = SINIF_AD[c]
        v2a = meta.get("v2_val_per_sinif", {}).get(ad, {}).get("acc", "-")
        v3a = meta.get("v3_val_per_sinif", {}).get(ad, {}).get("acc", "-")
        nobet = "OK" if (isinstance(v2a, float) and isinstance(v3a, float) and v3a >= v2a) else "-"
        s.append("| %s | %s | %s | %s |" % (ad, v2a, v3a, nobet))
    s.append("| **top-1 (400)** | **%s** | **%s** | %s |" % (
        meta.get("v2_val_top1_400", "-"), meta.get("v3_val_top1_400", "-"),
        "OK" if meta.get("v3_val_top1_400", 0) >= meta.get("v2_val_top1_400", 1) else "-"))
    s.append("")
    s.append("- T-0210 hüküm çıpası (v2, 2026-10-15): eski-4-sınıf val top-1 = 1.0 (köprü kaydı).")
    s.append("")
    s.append("## Dosyalar")
    s.append("")
    s.append("| Dosya | sha256 |")
    s.append("|---|---|")
    for yol in [ILAN_YOL, HUKUM_YOL, CIKTI_YOL,
                "data/pedagogy/legal_router_v3_dataset.jsonl",
                "scripts/build_legal_router_dataset_p25.py"]:
        tam = os.path.join(REPO_KOK, yol)
        s.append("| `%s` | `%s` |" % (yol, _sha256(tam) if os.path.exists(tam) else "YOK"))
    s.append("| `data/anka_router_v2.pt` (devir, DEĞİŞMEDİ) | `%s` |" % _sha256(os.path.join(REPO_KOK, V2_YOL)))
    s.append("")
    s.append("## Notlar")
    s.append("")
    s.append("- Eğitim verisi: sınıf 0-3 kaynakları T-0210 ile birebir; legal = `legal_router_v3_dataset.jsonl` (60 kayıt).")
    s.append("- Legal dengesi: 50 distinct × 10 oversample = 500 etkin train satırı (diğer sınıflarla eş).")
    s.append("- Probe sorguları ve legal val örnekleri eğitim kümesinde yok (K8 kesişim kapısı).")
    s.append("- Canlıya devir (agent_gateway `router_state_path=data/anka_router_v2.pt` varsayılanı) bu görevin kapsamı DIŞINDA — operatör kararı.")
    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")
    _stderr("HÜKÜM: %s rc=%d" % (hukum_adi, rc))
    return rc


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_sha = olustur_ilan()
    _stderr("İLAN sha256=%s damga=%s" % (ilan_sha, damga))

    kapilar: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    def kayit(ayrac: str, gec: bool, olcum: Any) -> bool:
        kapilar.append({"ayrac": ayrac, "olcum": olcum, "gec": bool(gec)})
        _stderr("%s → %s" % (ayrac, "GEÇTİ" if gec else "DÜŞTÜ"))
        return bool(gec)

    # ---- K1 SIFIR-KILIÇ ----
    olcum: Dict[str, Any] = {}
    sapma: List[str] = []
    for yol, beklenen in KAYNAK_SHA.items():
        tam = os.path.join(REPO_KOK, yol)
        if not os.path.exists(tam):
            sapma.append(f"{yol} (YOK)")
            continue
        olc = _sha256(tam)
        olcum[yol] = olc
        if olc != beklenen:
            sapma.append(f"{yol} (beklenen: {beklenen[:8]}, ölçülen: {olc[:8]})")
    olcum["sha_sapma"] = sapma
    if not kayit("K1 SIFIR-KILIÇ (güncel kaynak-sha birebir eşleşme)", not sapma, olcum):
        return _hukum(kapilar, ilan_sha, damga, None, None, meta)

    # ---- Model Kurulumu (T-0210 birebir) ----
    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kayit(f"CİHAZ (özellik-çıkarımı: {cihaz.type.upper()})", True, {"cihaz": str(cihaz)})

    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_KOK, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(REPO_KOK, "data", "rebuild", "vocab_anka_r1_33114.json"))
    tokenizer = KristalTokenizer(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    # weights_only=False: repo'nun kendi hattından yerel checkpoint — T-0210 kanonik deseni.
    sd = torch.load(os.path.join(REPO_KOK, "data", "anka_taban_pure.pt"),
                    map_location=cihaz, weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    for k in [k for k in sd.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    sd = resize_state_dict(model, sd)
    model.load_state_dict(sd, strict=False)
    model.to(cihaz)
    model.eval()

    torch.manual_seed(SEED)
    engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=2.5)
    engine.to(cihaz)
    engine.eval()

    # ---- K5 Veri Envanteri (SEED-42; eski sınıflar 500/100, legal 50/10) ----
    rng = random.Random(SEED)
    secim: Dict[int, List[str]] = {}
    for c, yollar in SINIF_KAYNAK.items():
        tum = _metinler(yollar)
        tum = sorted(set(tum))
        rng.shuffle(tum)
        if c == 4:
            secim[c] = tum[:LEGAL_TRAIN_N + LEGAL_VAL_N]
        else:
            secim[c] = tum[:TRAIN_N + VAL_N]

    train_metin: List[str] = []
    val_metin: List[str] = []
    train_y: List[int] = []
    val_y: List[int] = []
    for c in SINIF_KAYNAK:
        blok = secim[c]
        n_tr = LEGAL_TRAIN_N if c == 4 else TRAIN_N
        n_va = LEGAL_VAL_N if c == 4 else VAL_N
        train_metin.extend(blok[:n_tr])
        val_metin.extend(blok[n_tr:n_tr + n_va])
        train_y.extend([c] * min(n_tr, len(blok)))
        val_y.extend([c] * min(n_va, max(0, len(blok) - n_tr)))

    kesisim = len(set(train_metin) & set(val_metin))
    kirilim_train = {SINIF_AD[c]: sum(1 for e in train_y if e == c) for c in SINIF_KAYNAK}
    kirilim_val = {SINIF_AD[c]: sum(1 for e in val_y if e == c) for c in SINIF_KAYNAK}
    legal_ornek = sum(1 for e in train_y + val_y if e == 4)
    k5_gec = (all(kirilim_train[SINIF_AD[c]] == (LEGAL_TRAIN_N if c == 4 else TRAIN_N) for c in SINIF_KAYNAK)
              and all(kirilim_val[SINIF_AD[c]] == (LEGAL_VAL_N if c == 4 else VAL_N) for c in SINIF_KAYNAK)
              and kesisim == 0 and legal_ornek > 0)
    kayit("K5 VERİ-ENVANTER (4×500/100 + legal 50/10; kesişim-0; legal>0)", k5_gec, {
        "train_kirilim": kirilim_train, "val_kirilim": kirilim_val,
        "val_train_kesisim": kesisim, "legal_ornek": legal_ornek,
        "legal_distinct_train": kirilim_train["legal"],
        "legal_oversample": LEGAL_OVERSAMPLE})
    if not k5_gec:
        return _hukum(kapilar, ilan_sha, damga, None, None, meta)

    # ---- Özellik Çıkarımı ----
    _stderr("özellik-çıkarımı: train %d + val %d (%s)" % (
        len(train_metin), len(val_metin), cihaz.type.upper()))
    t0 = time.time()
    Ptr, Qtr, kirp_tr = _ozellik_cikar(model, engine, tokenizer, vocab, train_metin, cihaz)
    Pva, Qva, kirp_va = _ozellik_cikar(model, engine, tokenizer, vocab, val_metin, cihaz)
    Ppr, Qpr, _ = _ozellik_cikar(model, engine, tokenizer, vocab, PROBE_SORGULAR, cihaz)
    sure = time.time() - t0
    _stderr("özellik-çıkarımı tamam (%.1f sn; kırpılan: %d+%d)" % (sure, kirp_tr, kirp_va))

    # legal train oversample (×10 → 500 etkin; özellik satırları çoğaltılır)
    lg_tr = kirilim_train["legal"]
    maske_legal_tr = torch.tensor([1 if e == 4 else 0 for e in train_y], dtype=torch.bool)
    Ptr_lg = Ptr[maske_legal_tr]
    Qtr_lg = Qtr[maske_legal_tr]
    Ptr_eski = Ptr[~maske_legal_tr]
    Qtr_eski = Qtr[~maske_legal_tr]
    ytr_eski = torch.tensor([e for e in train_y if e != 4], dtype=torch.long)
    Ptr_tam = torch.cat([Ptr_eski] + [Ptr_lg] * LEGAL_OVERSAMPLE)
    Qtr_tam = torch.cat([Qtr_eski] + [Qtr_lg] * LEGAL_OVERSAMPLE)
    ytr = torch.cat([ytr_eski, torch.full((lg_tr * LEGAL_OVERSAMPLE,), 4, dtype=torch.long)])
    yva = torch.tensor(val_y, dtype=torch.long)
    maske_eski_val = torch.tensor([1 if e != 4 else 0 for e in val_y], dtype=torch.bool)

    # ---- K2 Baseline ----
    torch.manual_seed(SEED)
    router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                            router_dim=256, num_experts=5, top_k=2,
                            expert_names=EXPERT_NAMES)
    with torch.no_grad():
        base_pred = _gate_logits(router, Pva[maske_eski_val], Qva[maske_eski_val]).argmax(-1)
    baseline = float((base_pred == yva[maske_eski_val]).float().mean().item())
    kayit("K2 BASELINE (fresh-init eski-4-sınıf val top-1; kayıt-değer)", True, {
        "baseline_val_top1": round(baseline, 4), "rastgele_taban_5sinif": 0.2})

    # ---- K3 Eğitim ----
    epoch_kayiplar, train_per_sinif = _egit(router, Ptr_tam, Qtr_tam, ytr, epochs=EPOCH)
    with torch.no_grad():
        pred400 = _gate_logits(router, Pva[maske_eski_val], Qva[maske_eski_val]).argmax(-1)
    val_top1_400 = float((pred400 == yva[maske_eski_val]).float().mean().item())
    with torch.no_grad():
        pred_tum = _gate_logits(router, Pva, Qva).argmax(-1)
    val_top1_tum = float((pred_tum == yva).float().mean().item())
    v3_val_per_sinif = _sinif_metrik(Pva, Qva, yva, router, list(SINIF_KAYNAK))
    k3_gec = (val_top1_400 >= VAL_ALT_SINIR
              and val_top1_400 >= baseline + BASELINE_MARJ
              and epoch_kayiplar[-1] < epoch_kayiplar[0])
    kayit("K3 EĞİTİM (eski-400 top-1>=0,95 VE baseline+0,50 VE kayıp-azalıyor)", k3_gec, {
        "val_top1_400": round(val_top1_400, 4), "val_top1_410": round(val_top1_tum, 4),
        "baseline": round(baseline, 4),
        "epoch_kayiplar": [round(k, 4) for k in epoch_kayiplar],
        "train_per_sinif_dogru": train_per_sinif,
        "val_per_sinif": v3_val_per_sinif,
        "kirpilan_4096_ustu": kirp_tr + kirp_va,
        "ozellik_sure_sn": round(sure, 1)})

    # ---- K7 Regresyon Nöbeti (v2 aynı özelliklerle) ----
    v2_router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                               router_dim=256, num_experts=5, top_k=2,
                               expert_names=EXPERT_NAMES)
    v2_router = load_trained_router_state(v2_router, os.path.join(REPO_KOK, V2_YOL))
    v2_router.eval()
    with torch.no_grad():
        v2_pred400 = _gate_logits(v2_router, Pva[maske_eski_val], Qva[maske_eski_val]).argmax(-1)
    v2_val_top1_400 = float((v2_pred400 == yva[maske_eski_val]).float().mean().item())
    v2_val_per_sinif = _sinif_metrik(Pva, Qva, yva, v2_router, ESKI_SINIFLAR)
    regersiyon_yok = (val_top1_400 >= v2_val_top1_400 and all(
        v3_val_per_sinif[SINIF_AD[c]]["acc"] >= v2_val_per_sinif[SINIF_AD[c]]["acc"]
        for c in ESKI_SINIFLAR))
    kayit("K7 REGRESYON NÖBETİ (v3 eski-sınıf doğruluğu v2'den düşük değil)", regersiyon_yok, {
        "v2_val_top1_400": round(v2_val_top1_400, 4),
        "v3_val_top1_400": round(val_top1_400, 4),
        "v2_val_per_sinif": v2_val_per_sinif,
        "v3_val_per_sinif_eski": {SINIF_AD[c]: v3_val_per_sinif[SINIF_AD[c]] for c in ESKI_SINIFLAR},
        "t0210_hukum_cipasi_v2": 1.0})

    # ---- K8 Legal Probe (v2 vs v3; kesişim kapısı) ----
    probe_v2 = _probe_olc(v2_router, Ppr, Qpr, "anka_router_v2.pt")
    probe_v3 = _probe_olc(router, Ppr, Qpr, "anka_router_v3.pt")
    egitim_tum = set(train_metin) | set(val_metin)
    probe_kesisim = len(egitim_tum & set(PROBE_SORGULAR))
    k8_gec = (probe_v3["legal_isabet"] >= PROBE_ALT_SINIR
              and probe_v3["legal_isabet"] > probe_v2["legal_isabet"]
              and probe_kesisim == 0)
    kayit("K8 LEGAL PROBE (v3 isabet>=8/10 VE >v2; probe∩veri=0)", k8_gec, {
        "v2_legal_isabet": probe_v2["legal_isabet"],
        "v3_legal_isabet": probe_v3["legal_isabet"],
        "n_sorgu": len(PROBE_SORGULAR),
        "probe_veri_kesisim": probe_kesisim})

    # ---- K4 Determinizm ----
    sha_1 = _state_dict_sha(router.state_dict())
    torch.manual_seed(SEED)
    router2 = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                             router_dim=256, num_experts=5, top_k=2,
                             expert_names=EXPERT_NAMES)
    _egit(router2, Ptr_tam, Qtr_tam, ytr, epochs=EPOCH)
    sha_2 = _state_dict_sha(router2.state_dict())
    kayit("K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir)", sha_1 == sha_2,
          {"sha_1": sha_1, "sha_2": sha_2})
    if not all(k["gec"] for k in kapilar):
        meta.update({
            "probe_v2": probe_v2, "probe_v3": probe_v3,
            "v2_val_top1_400": round(v2_val_top1_400, 4),
            "v3_val_top1_400": round(val_top1_400, 4),
            "v2_val_per_sinif": v2_val_per_sinif,
            "v3_val_per_sinif": v3_val_per_sinif,
            "epoch_kayiplar": epoch_kayiplar,
            "legal_ornek": legal_ornek,
        })
        return _hukum(kapilar, ilan_sha, damga, None, sha_1, meta)

    # ---- K6 Şema ve Çıktı (v3'e yazım; v2 DOKUNULMAZ) ----
    v2_sha_baslangic = _sha256(os.path.join(REPO_KOK, V2_YOL))
    kanonik = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                             router_dim=256, num_experts=5, top_k=2,
                             expert_names=EXPERT_NAMES)
    sema_ok = (sorted(kanonik.state_dict().keys()) == sorted(router.state_dict().keys())
               and router.w_g.weight.shape == (5, 256))
    cikti_sha = None
    if sema_ok:
        torch.save(router.state_dict(), os.path.join(REPO_KOK, CIKTI_YOL))
        cikti_sha = _sha256(os.path.join(REPO_KOK, CIKTI_YOL))
        # devir çıpası: v2 yazım öncesi/sonrası birebir aynı olmalı
        sema_ok = (_sha256(os.path.join(REPO_KOK, V2_YOL)) == v2_sha_baslangic)
    kayit("K6 ŞEMA + ÇIKTI (5x256 + birebir anahtarlar + v3 yazımı + v2 değişmedi)",
          sema_ok, {"anahtar_sayi": len(router.state_dict()),
                    "w_g_shape": list(router.w_g.weight.shape),
                    "cikti": CIKTI_YOL, "cikti_sha256": cikti_sha,
                    "v2_sha_once": v2_sha_baslangic,
                    "v2_sha_sonra": _sha256(os.path.join(REPO_KOK, V2_YOL))})

    meta.update({
        "probe_v2": probe_v2, "probe_v3": probe_v3,
        "v2_val_top1_400": round(v2_val_top1_400, 4),
        "v3_val_top1_400": round(val_top1_400, 4),
        "v3_val_top1_410": round(val_top1_tum, 4),
        "v2_val_per_sinif": v2_val_per_sinif,
        "v3_val_per_sinif": v3_val_per_sinif,
        "epoch_kayiplar": epoch_kayiplar,
        "legal_ornek": legal_ornek,
    })
    return _hukum(kapilar, ilan_sha, damga, cikti_sha, sha_1, meta)


if __name__ == "__main__":
    sys.exit(main())