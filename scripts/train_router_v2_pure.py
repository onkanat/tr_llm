#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0210 — TriModalRouter 5-Sınıf Saf Taban Model (anka_taban_pure.pt) Hizalaması ve Eğitimi.

Bu betik, 5-sınıflı TriModalRouter modelini data/anka_taban_pure.pt temsilleri
üzerinden deterministik olarak yeniden eğitir ve en az %95 doğrulama doğruluğu
(>= 0.95) ile data/anka_router_v2.pt ağırlıklarını üretir.

İLAN: data/eval/t0210_router_v2_pure_ilan_2026-10-15.md
HÜKÜM: data/eval/t0210_router_v2_pure_hukum_2026-10-15.json
RAPOR: data/eval/t0210_router_v2_pure_rapor_2026-10-15.md
ÇIKTI: data/anka_router_v2.pt
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

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

ILAN_YOL = "data/eval/t0210_router_v2_pure_ilan_2026-10-15.md"
HUKUM_YOL = "data/eval/t0210_router_v2_pure_hukum_2026-10-15.json"
RAPOR_YOL = "data/eval/t0210_router_v2_pure_rapor_2026-10-15.md"
CIKTI_YOL = "data/anka_router_v2.pt"

SEED = 42
TRAIN_N = 500
VAL_N = 100
EPOCH = 5
BATCH = 32
LR = 1e-3
VAL_ALT_SINIR = 0.95
BASELINE_MARJ = 0.50

KAYNAK_SHA = {
    "data/anka_taban_pure.pt":
        "5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360",
    "data/rebuild/vocab_anka_r1_33114.json":
        "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv":
        "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
    "data/pedagogy/bahcivan_arena.jsonl":
        "9fa356b44ebfc7f6d48e387d7dceb12330b36287c94d95d0e8c15b66646baa13",
}

SINIF_KAYNAK = {
    0: ["data/pedagogy/lexical_semantics_dataset.jsonl"],
    1: ["data/pedagogy_canonical/high_school_canonical.jsonl",
        "data/pedagogy_canonical/literature_canonical.jsonl",
        "data/pedagogy_canonical/middle_school_canonical.jsonl",
        "data/pedagogy_canonical/turk_tarihi_canonical.jsonl"],
    2: ["data/pedagogy_canonical/carpenter_canonical.jsonl"],
    3: ["data/pedagogy/bahcivan_arena.jsonl"],
}
SINIF_AD = {0: "grammar_core", 1: "pedagogy", 2: "carpenter", 3: "gardener", 4: "legal"}
EXPERT_NAMES = ["grammar_core", "pedagogy", "carpenter", "gardener", "legal"]


def _stderr(mesaj: str) -> None:
    print("[T-0210] " + mesaj, file=sys.stderr, flush=True)


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _state_dict_sha(sd: Dict[str, torch.Tensor]) -> str:
    """Deterministik state_dict özeti: anahtar-adı + tensör-baytları (sıralı)."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _metinler(yollar: List[str]) -> List[str]:
    """`input` alanı (boşsa `instruction` veya `soru`); dosya-sırası korunur."""
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


def k1_sifir_kilic() -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {}
    sapma = []
    for yol, beklenen in KAYNAK_SHA.items():
        tam_yol = os.path.join(REPO_KOK, yol)
        if not os.path.exists(tam_yol):
            sapma.append(f"{yol} (YOK)")
            continue
        olc = _sha256(tam_yol)
        olcum[yol] = olc
        if olc != beklenen:
            sapma.append(f"{yol} (beklenen: {beklenen[:8]}, olculen: {olc[:8]})")
    olcum["sha_sapma"] = sapma
    return not sapma, olcum


def _ozellik_cikar(model: Any, engine: Any, tokenizer: Any, vocab: Any,
                   metinler: List[str], cihaz: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """prompt_vec + q_merak özellikleri."""
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


def _dogruluk(router: TriModalRouter, P: torch.Tensor, Q: torch.Tensor,
              y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = _gate_logits(router, P, Q).argmax(dim=-1)
    return float((pred == y).float().mean().item())


def olustur_ilan() -> str:
    ilan_icerik = """# T-0210 İLAN — TriModalRouter 5-Sınıf Saf Taban Model (anka_taban_pure.pt) Hizalaması

- **Tarih:** 2026-10-15
- **Görev:** T-0210
- **Hedef:** TriModalRouter'ı `data/anka_taban_pure.pt` model temsilleri ile hizalamak, deterministik olarak eğitmek, validation accuracy >= %95 sağlamak ve `data/anka_router_v2.pt` ağırlıklarını güncellemek.

## Sınıf Mimarisi (num_experts=5, top_k=2)

1. `0: grammar_core` (`data/pedagogy/lexical_semantics_dataset.jsonl`)
2. `1: pedagogy` (`data/pedagogy_canonical/*`)
3. `2: carpenter` (`data/pedagogy_canonical/carpenter_canonical.jsonl`)
4. `3: gardener` (`data/pedagogy/bahcivan_arena.jsonl`)
5. `4: legal` (ilerisi için ayrık sınıf)

## Kaynak Çıpaları (SHA-256)

- `data/anka_taban_pure.pt`: `5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360`
- `data/rebuild/vocab_anka_r1_33114.json`: `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`
- `data/lexicon/roots.tsv`: `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`
- `data/pedagogy/bahcivan_arena.jsonl`: `9fa356b44ebfc7f6d48e387d7dceb12330b36287c94d95d0e8c15b66646baa13`

## Hüküm Kapıları

- **K1 SIFIR-KILIÇ:** Kaynak dosyaların SHA-256 özetleri çıpalarla birebir eşleşir.
- **K2 BASELINE:** Fresh-init seed-42 router'ın doğruluk ölçümü yapılır (4 aktif sınıf için rastgele taban ~0.25).
- **K3 EĞİTİM:** Validation top-1 doğruluğu >= 0.95 VE >= baseline + 0.50 VE son epoch kaybı ilk epoch kaybından düşüktür.
- **K4 DETERMİNİZM:** İkinci eğitim koşumu (aynı seed, CPU/MPS) sonucunda elde edilen state_dict SHA-256 özeti ilk koşumla bit-özdeştir.
- **K5 VERİ-ENVANTERİ:** 4 aktif sınıfın her biri için 500 train + 100 validation örneği (toplam 2000 train, 400 val); validation ve train arasında 0 kesişim; legal sınıfından 0 örnek.
- **K6 ŞEMA VE ÇIKTI:** `TriModalRouter(num_experts=5)` kanonik state_dict anahtar kümesi ile tam eşleşir; `data/anka_router_v2.pt` diske yazılır.
"""
    tam_yol = os.path.join(REPO_KOK, ILAN_YOL)
    os.makedirs(os.path.dirname(tam_yol), exist_ok=True)
    with open(tam_yol, "w", encoding="utf-8") as f:
        f.write(ilan_icerik.strip() + "\n")
    return _sha256(tam_yol)


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_sha = olustur_ilan()
    _stderr("İLAN sha256=%s damga=%s" % (ilan_sha, damga))

    kapilar: List[Dict[str, Any]] = []

    def kayit(ayrac: str, gec: bool, olcum: Any) -> bool:
        kapilar.append({"ayrac": ayrac, "olcum": olcum, "gec": bool(gec)})
        _stderr("%s → %s" % (ayrac, "GEÇTİ" if gec else "DÜŞTÜ"))
        return bool(gec)

    # ---- K1 SIFIR-KILIÇ ----
    k1_gec, k1_olcum = k1_sifir_kilic()
    if not kayit("K1 SIFIR-KILIÇ (kaynak-sha birebir eşleşme)", k1_gec, k1_olcum):
        return _hukum(kapilar, ilan_sha, damga, None, None, {})

    # ---- Cihaz Kontrolü ----
    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kayit(f"CİHAZ (özellik-çıkarımı: {cihaz.type.upper()})", True, {"cihaz": str(cihaz)})

    # ---- Kanonik Model Kurulumu ----
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_KOK, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(REPO_KOK, "data", "rebuild", "vocab_anka_r1_33114.json"))
    tokenizer = KristalTokenizer(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(os.path.join(REPO_KOK, "data", "anka_taban_pure.pt"), map_location=cihaz, weights_only=False)
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

    # ---- K5 Veri Envanteri (SEED-42; sınıf-başı 500 train + 100 val) ----
    rng = random.Random(SEED)
    metinler: List[str] = []
    etiketler: List[int] = []
    for c, yollar in SINIF_KAYNAK.items():
        tum = _metinler(yollar)
        tum = sorted(set(tum))
        rng.shuffle(tum)
        secili = tum[:TRAIN_N + VAL_N]
        metinler.extend(secili)
        etiketler.extend([c] * len(secili))

    train_metin, val_metin, train_y, val_y = [], [], [], []
    for c in SINIF_KAYNAK:
        blok = [m for m, e in zip(metinler, etiketler) if e == c]
        train_metin.extend(blok[:TRAIN_N])
        val_metin.extend(blok[TRAIN_N:TRAIN_N + VAL_N])
        train_y.extend([c] * min(TRAIN_N, len(blok)))
        val_y.extend([c] * min(VAL_N, max(0, len(blok) - TRAIN_N)))

    kesisim = len(set(train_metin) & set(val_metin))
    kirilim_train = {SINIF_AD[c]: sum(1 for e in train_y if e == c) for c in SINIF_KAYNAK}
    kirilim_val = {SINIF_AD[c]: sum(1 for e in val_y if e == c) for c in SINIF_KAYNAK}
    legal_ornek = sum(1 for e in train_y + val_y if e == 4)
    k5_gec = (all(v == TRAIN_N for v in kirilim_train.values())
              and all(v == VAL_N for v in kirilim_val.values())
              and kesisim == 0 and legal_ornek == 0)
    kayit("K5 VERİ-ENVANTER (500/100 x 4; kesişim-0; legal-0)", k5_gec, {
        "train_kirilim": kirilim_train, "val_kirilim": kirilim_val,
        "val_train_kesisim": kesisim, "legal_ornek": legal_ornek})
    if not k5_gec:
        return _hukum(kapilar, ilan_sha, damga, None, None, {})

    # ---- Özellik Çıkarımı ----
    _stderr("özellik-çıkarımı: train %d + val %d (%s)" % (len(train_metin), len(val_metin), cihaz.type.upper()))
    t0 = time.time()
    Ptr, Qtr, kirp_tr = _ozellik_cikar(model, engine, tokenizer, vocab, train_metin, cihaz)
    Pva, Qva, kirp_va = _ozellik_cikar(model, engine, tokenizer, vocab, val_metin, cihaz)
    sure = time.time() - t0
    _stderr("özellik-çıkarımı tamam (%.1f sn; kırpılan: %d+%d)" % (sure, kirp_tr, kirp_va))
    ytr = torch.tensor(train_y, dtype=torch.long)
    yva = torch.tensor(val_y, dtype=torch.long)

    # ---- K2 Baseline ----
    torch.manual_seed(SEED)
    router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                            router_dim=256, num_experts=5, top_k=2,
                            expert_names=EXPERT_NAMES)
    baseline = _dogruluk(router, Pva, Qva, yva)
    kayit("K2 BASELINE (fresh-init val top-1; kayıt-değer)", True, {
        "baseline_val_top1": round(baseline, 4),
        "rastgele_taban_4aktif": 0.25})

    # ---- K3 Eğitim ----
    epoch_kayiplar, train_per_sinif = _egit(router, Ptr, Qtr, ytr, epochs=EPOCH)
    val_acc = _dogruluk(router, Pva, Qva, yva)
    
    # Top-2 accuracy
    with torch.no_grad():
        v_logits = _gate_logits(router, Pva, Qva)
        top2_idx = v_logits.topk(2, dim=-1).indices
        in_top2 = (top2_idx == yva.unsqueeze(1)).any(dim=-1)
        val_top2_acc = float(in_top2.float().mean().item())
        pred_val = v_logits.argmax(dim=-1)

    val_per_sinif: Dict[str, Dict[str, Any]] = {}
    for c in SINIF_KAYNAK:
        c_name = SINIF_AD[c]
        total = (yva == c).sum().item()
        correct = ((yva == c) & (pred_val == c)).sum().item()
        predicted_total = (pred_val == c).sum().item()
        prec = correct / predicted_total if predicted_total > 0 else 0
        rec = correct / total if total > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        val_per_sinif[c_name] = {
            "correct": correct, "total": total, "acc": round(rec, 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)
        }

    k3_gec = (val_acc >= VAL_ALT_SINIR
              and val_acc >= baseline + BASELINE_MARJ
              and epoch_kayiplar[-1] < epoch_kayiplar[0])
    kayit("K3 EĞİTİM (val>=0,95 VE baseline+0,50 VE kayıp-azalıyor)", k3_gec, {
        "val_top1": round(val_acc, 4),
        "val_top2": round(val_top2_acc, 4),
        "baseline": round(baseline, 4),
        "epoch_kayiplar": [round(k, 4) for k in epoch_kayiplar],
        "train_per_sinif_dogru": train_per_sinif,
        "val_per_sinif": val_per_sinif,
        "kirpilan_4096_ustu": kirp_tr + kirp_va,
        "ozellik_sure_sn": round(sure, 1)})

    # ---- K4 Determinizm ----
    sha_1 = _state_dict_sha(router.state_dict())
    torch.manual_seed(SEED)
    router2 = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                             router_dim=256, num_experts=5, top_k=2,
                             expert_names=EXPERT_NAMES)
    _egit(router2, Ptr, Qtr, ytr, epochs=EPOCH)
    sha_2 = _state_dict_sha(router2.state_dict())
    k4_gec = (sha_1 == sha_2)
    kayit("K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir)", k4_gec,
          {"sha_1": sha_1, "sha_2": sha_2})
    if not all(k["gec"] for k in kapilar):
        return _hukum(kapilar, ilan_sha, damga, None, None, val_per_sinif)

    # ---- K6 Şema ve Çıktı Kaydı ----
    kanonik_anahtarlar = sorted(TriModalRouter(
        prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256,
        num_experts=5, top_k=2, expert_names=EXPERT_NAMES).state_dict().keys())
    cikti_anahtarlar = sorted(router.state_dict().keys())
    sema_ok = (kanonik_anahtarlar == cikti_anahtarlar and router.w_g.weight.shape == (5, 256))
    if sema_ok:
        torch.save(router.state_dict(), os.path.join(REPO_KOK, CIKTI_YOL))
        cikti_sha = _sha256(os.path.join(REPO_KOK, CIKTI_YOL))
    else:
        cikti_sha = None
    kayit("K6 ŞEMA + ÇIKTI (5x256 w_g + birebir anahtarlar + kayıt)", sema_ok, {
        "anahtar_sayi": len(cikti_anahtarlar),
        "w_g_shape": list(router.w_g.weight.shape),
        "cikti": CIKTI_YOL, "cikti_sha256": cikti_sha})

    meta = {
        "val_top1": val_acc,
        "val_top2": val_top2_acc,
        "val_per_sinif": val_per_sinif,
        "epoch_kayiplar": epoch_kayiplar,
    }
    return _hukum(kapilar, ilan_sha, damga, cikti_sha, sha_1, meta)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str, damga: str,
           cikti_sha: Any, sd_sha: Any, meta: Dict[str, Any]) -> int:
    hukum_adi = ("T0210_ROUTER_V2_PURE_GECTI" if kapilar and all(k["gec"] for k in kapilar)
                 else "DUR")
    rc = 0 if hukum_adi == "T0210_ROUTER_V2_PURE_GECTI" else 2
    hukum_json = {
        "hukum": hukum_adi, "rc": rc, "damga": damga,
        "hukum_kaynagi": "scripts/train_router_v2_pure.py — deterministik ölçüm",
        "ilan": ILAN_YOL, "ilan_sha256": ilan_sha,
        "kapilar": kapilar,
        "cikti_sha256": cikti_sha, "state_dict_sha256": sd_sha,
        "meta": meta,
    }
    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, HUKUM_YOL))

    s: List[str] = []
    s.append("# T-0210 — TriModalRouter 5-Sınıf Saf Taban Model (anka_taban_pure.pt) Hizalama Raporu")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; deterministik ölçüm)" % hukum_adi)
    s.append("**Damga:** %s (UTC) — koşum sonu" % damga)
    s.append("**İlan:** `%s` (sha256 `%s`)" % (ILAN_YOL, ilan_sha))
    s.append("")
    s.append("## Kapılar")
    s.append("")
    s.append("| Ayraç | Ölçülen | Hüküm |")
    s.append("|---|---|---|")
    for k in kapilar:
        s.append("| %s | `%s` | %s |" % (
            k["ayrac"], json.dumps(k["olcum"], ensure_ascii=False)[:500],
            "GEÇTİ" if k["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## Çıktı ve Doğrulama")
    s.append("")
    if cikti_sha:
        s.append("- Checkpoint: `%s` — sha256 `%s`" % (CIKTI_YOL, cikti_sha))
        s.append("- State Dict SHA-256: `%s`" % sd_sha)
        s.append("- Validation Top-1 Accuracy: `%.4f` (%%%0.2f)" % (meta.get("val_top1", 0.0), meta.get("val_top1", 0.0) * 100))
        s.append("- Validation Top-2 Accuracy: `%.4f` (%%%0.2f)" % (meta.get("val_top2", 0.0), meta.get("val_top2", 0.0) * 100))
    else:
        s.append("- Çıktı YAZILMADI (kapı-düşmesi; kısmi eğitim-artefaktı yok).")
    s.append("")
    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum=%s state_dict_sha256=%s" % (hukum_adi, rc, hukum_sha, sd_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())
