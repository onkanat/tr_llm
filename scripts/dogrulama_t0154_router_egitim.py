#!/usr/bin/env python
"""T-0154 — TriModalRouter eğitilmiş-ağırlık hüküm-betiği (3-sınıf).

İLAN: data/eval/t0154_router_egitim_ilan_2026-09-28.md
(damga 06:52:14Z + revizyon-1 tam-digest; koşum-ÖNCESİ; İLAN SABİT).

HÜKÜM (İLAN kapıları — koşum öncesi sabit):
  K1 SIFIR-KILIÇ: data/anka_router.pt koşum-öncesi YOK; kaynak SHA-256
     birebir (base_v2 / vocab_anka_r1_33114 / roots.tsv — İLAN tam-değerler)
  K2 BASELINE: fresh-init seed-42 router val top-1 doğruluk ölçülür (kayıt)
  K3 EĞİTİM: val top-1 >= 0,75 VE >= baseline + 0,15 VE
     train CE son-epoch < ilk-epoch
  K4 DETERMİNİZM: ikinci eğitim koşumu (aynı-seed, CPU) state_dict SHA
     bit-özdeş
  K5 VERİ-ENVANTER: sınıf-kırılım 500/100 x 3 birebir; val∩train metin
     kesişimi 0; legal etiketli örnek 0
  K6 ŞEMA + ÇIKTI: state_dict anahtar-kümesi kanonik TriModalRouter birebir;
     data/anka_router.pt yazılır + SHA-256 hüküm-JSON'da; canlıya 0 istek
  6/6 → T0154_ROUTER_GECTI (rc=0); aksi her dal → DUR (rc=2; çıktı YAZILMAZ).
Hüküm BETİKTEN; elle sayı/hüküm YOK. rc ∈ {0, 2}.
"""
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

ILAN_YOL = "data/eval/t0154_router_egitim_ilan_2026-09-28.md"
HUKUM_YOL = "data/eval/t0154_router_egitim_hukum_2026-09-28.json"
RAPOR_YOL = "data/eval/t0154_router_egitim_rapor_2026-09-28.md"
CIKTI_YOL = "data/anka_router.pt"

SEED = 42
TRAIN_N = 500
VAL_N = 100
EPOCH = 3
BATCH = 32
LR = 1e-3
VAL_ALT_SINIR = 0.75
BASELINE_MARJ = 0.15

KAYNAK_SHA = {
    "data/anka_base_v2.pt":
        "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50",
    "data/rebuild/vocab_anka_r1_33114.json":
        "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv":
        "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
}
# sınıf-0 grammar_core, 1 pedagogy, 2 carpenter; 3 legal — hedef YOK
SINIF_KAYNAK = {
    0: ["data/pedagogy/lexical_semantics_dataset.jsonl"],
    1: ["data/pedagogy_canonical/high_school_canonical.jsonl",
        "data/pedagogy_canonical/literature_canonical.jsonl",
        "data/pedagogy_canonical/middle_school_canonical.jsonl",
        "data/pedagogy_canonical/parenting_canonical.jsonl",
        "data/pedagogy_canonical/turk_tarihi_canonical.jsonl"],
    2: ["data/pedagogy_canonical/carpenter_canonical.jsonl"],
}
SINIF_AD = {0: "grammar_core", 1: "pedagogy", 2: "carpenter", 3: "legal"}


def _stderr(mesaj: str) -> None:
    print("[T-0154] " + mesaj, file=sys.stderr, flush=True)


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
    """`input` alanı (boşsa `instruction`); dosya-sırası korunur."""
    cikti: List[str] = []
    for yol in yollar:
        with open(os.path.join(REPO_KOK, yol), encoding="utf-8") as f:
            for satir in f:
                satir = satir.strip()
                if not satir:
                    continue
                rec = json.loads(satir)
                m = (rec.get("input") or "").strip() or (rec.get("instruction") or "").strip()
                if m:
                    cikti.append(m)
    return cikti


def k1_sifir_kilic() -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {"cikti_var": os.path.exists(os.path.join(REPO_KOK, CIKTI_YOL))}
    if olcum["cikti_var"]:
        return False, olcum
    sapma = []
    for yol, beklenen in KAYNAK_SHA.items():
        olc = _sha256(os.path.join(REPO_KOK, yol))
        olcum[yol] = olc
        if olc != beklenen:
            sapma.append(yol)
    olcum["sha_sapma"] = sapma
    return not sapma, olcum


def _ozellik_cikar(model: Any, engine: Any, tokenizer: Any, vocab: Any,
                   metinler: List[str], cihaz: torch.device) -> Tuple[Any, Any]:
    """prompt_vec + q_merak özellikleri (koşum-zamanı konfigürasyonu birebir).

    - prompt_vec = model.embedding(x).mean(dim=1)  (calculate_prompt_embedding)
    - q_merak = fresh-init CuriosityEngine(x_emb[:, -1, :], logits[:, -1, :])
    - rag_vec = None (İLAN)
    """
    unk_id = vocab.stoi.get("<UNK>", 1)
    vocab_boyut = model.embedding.embedding.weight.shape[0]
    promptler: List[torch.Tensor] = []
    qmeraklar: List[torch.Tensor] = []
    kirpilan = 0  # 4096-jeton üstü metin sayısı (block_size sınırı; GÖRÜNÜR ölçüm)
    model.eval()
    with torch.no_grad():
        for i, metin in enumerate(metinler):
            t_ids = tokenizer.encode(metin)
            if len(t_ids) > 4096:  # model block_size sınırı (causal-mask 4096²)
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
            if (i + 1) % 300 == 0:
                _stderr("özellik %d/%d" % (i + 1, len(metinler)))
    return torch.stack(promptler), torch.stack(qmeraklar), kirpilan


def _gate_logits(router: TriModalRouter, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """router.forward matematiği birebir (rag_vec=None): w_g(LN(w_p(p)+w_m(q)))."""
    e = router.w_p(p) + router.w_m(q)
    e = router.layer_norm(e)
    return router.w_g(e)


def _egit(router: TriModalRouter, P: torch.Tensor, Q: torch.Tensor,
          y: torch.Tensor) -> Tuple[List[float], Dict[str, int]]:
    """3-epoch AdamW CE; deterministik (CPU; generator seed-42 permütasyon)."""
    opt = torch.optim.AdamW(router.parameters(), lr=LR)
    n = P.shape[0]
    epoch_kayiplar: List[float] = []
    per_sinif_dogru: Dict[str, int] = {}
    for ep in range(EPOCH):
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
        _stderr("epoch %d ort-kayıp %.4f" % (ep + 1, epoch_kayiplar[-1]))
    # train doğruluk (sınıf-kırılım; kayıt)
    with torch.no_grad():
        pred = _gate_logits(router, P, Q).argmax(dim=-1)
    for c in range(3):
        maske = y == c
        per_sinif_dogru[SINIF_AD[c]] = int((pred[maske] == c).sum().item())
    return epoch_kayiplar, per_sinif_dogru


def _dogruluk(router: TriModalRouter, P: torch.Tensor, Q: torch.Tensor,
              y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = _gate_logits(router, P, Q).argmax(dim=-1)
    return float((pred == y).float().mean().item())


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_sha = _sha256(os.path.join(REPO_KOK, ILAN_YOL))
    _stderr("İLAN sha256=%s damga=%s" % (ilan_sha, damga))

    kapilar: List[Dict[str, Any]] = []

    def kayit(ayrac: str, gec: bool, olcum: Any) -> bool:
        kapilar.append({"ayrac": ayrac, "olcum": olcum, "gec": bool(gec)})
        _stderr("%s → %s" % (ayrac, "GEÇTİ" if gec else "DÜŞTÜ"))
        return bool(gec)

    # ---- K1 (çıktı-yok + kaynak-sha) ----
    k1_gec, k1_olcum = k1_sifir_kilic()
    if not kayit("K1 SIFIR-KILIÇ (çıktı-yok + kaynak-sha birebir)", k1_gec, k1_olcum):
        return _hukum(kapilar, ilan_sha, damga, None, None)

    # ---- cihaz (özellik-çıkarımı; İLAN: MPS hedef, yoksa DUR) ----
    if not torch.backends.mps.is_available():
        kayit("CİHAZ (özellik-çıkarımı: MPS)", False,
              {"mps": False, "hata": "MPS yok — sessiz-fallback YOK (İLAN)"})
        return _hukum(kapilar, ilan_sha, damga, None, None)
    cihaz = torch.device("mps")
    kayit("CİHAZ (özellik-çıkarımı: MPS)", True, {"mps": True})
    # (CİHAZ hüküm-kapısı DEĞİL — İLAN koşulu; K-kapılarına girmez)

    # ---- kanonik kurulum (P2/gateway kalıbı) ----
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_KOK, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(REPO_KOK, "data", "rebuild", "vocab_anka_r1_33114.json"))
    tokenizer = KristalTokenizer(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(os.path.join(REPO_KOK, "data", "anka_base_v2.pt"),
                    map_location=cihaz)
    for k in [k for k in sd.keys()
              if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    sd = resize_state_dict(model, sd)
    yukleme = model.load_state_dict(sd, strict=False)
    model.to(cihaz)
    model.eval()
    _stderr("taban-model yüklendi: eksik=%d fazla=%d"
            % (len(yukleme.missing_keys), len(yukleme.unexpected_keys)))

    # ---- fresh-init CuriosityEngine (seed-42; İLAN) ----
    torch.manual_seed(SEED)
    engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=2.5)
    engine.to(cihaz)
    engine.eval()

    # ---- veri-seçimi (SEED-42; sınıf-başı 500 train + 100 val) ----
    rng = random.Random(SEED)
    metinler: List[str] = []
    etiketler: List[int] = []
    for c, yollar in SINIF_KAYNAK.items():
        tum = _metinler(yollar)
        tum = sorted(set(tum))  # deterministik sıra
        rng.shuffle(tum)
        secili = tum[:TRAIN_N + VAL_N]
        metinler.extend(secili)
        etiketler.extend([c] * len(secili))
    # sınıf-kırılım: her sınıfın ilk 500'ü train, sonrası 100 val
    train_metin, val_metin, train_y, val_y = [], [], [], []
    for c in SINIF_KAYNAK:
        blok = [m for m, e in zip(metinler, etiketler) if e == c]
        train_metin.extend(blok[:TRAIN_N])
        val_metin.extend(blok[TRAIN_N:TRAIN_N + VAL_N])
        train_y.extend([c] * min(TRAIN_N, len(blok)))
        val_y.extend([c] * min(VAL_N, max(0, len(blok) - TRAIN_N)))
    # K5 envanter ölçümü (kapı-aşağıda)
    kesisim = len(set(train_metin) & set(val_metin))
    kirilim_train = {SINIF_AD[c]: sum(1 for e in train_y if e == c) for c in range(3)}
    kirilim_val = {SINIF_AD[c]: sum(1 for e in val_y if e == c) for c in range(3)}
    legal_ornek = sum(1 for e in train_y + val_y if e == 3)
    k5_gec = (all(v == TRAIN_N for v in kirilim_train.values())
              and all(v == VAL_N for v in kirilim_val.values())
              and kesisim == 0 and legal_ornek == 0)
    kayit("K5 VERİ-ENVANTER (500/100 x 3; kesişim-0; legal-0)", k5_gec, {
        "train_kirilim": kirilim_train, "val_kirilim": kirilim_val,
        "val_train_kesisim": kesisim, "legal_ornek": legal_ornek})
    if not k5_gec:
        return _hukum(kapilar, ilan_sha, damga, None, None)

    # ---- özellik-çıkarımı (tek-sefer; iki eğitim koşumu paylaşır) ----
    _stderr("özellik-çıkarımı: train %d + val %d (MPS)" % (len(train_metin), len(val_metin)))
    t0 = time.time()
    Ptr, Qtr, kirp_tr = _ozellik_cikar(model, engine, tokenizer, vocab, train_metin, cihaz)
    Pva, Qva, kirp_va = _ozellik_cikar(model, engine, tokenizer, vocab, val_metin, cihaz)
    sure = time.time() - t0
    _stderr("özellik-çıkarımı tamam (%.1f sn; 4096-üstü kırpılan: %d+%d)"
            % (sure, kirp_tr, kirp_va))
    ytr = torch.tensor(train_y, dtype=torch.long)
    yva = torch.tensor(val_y, dtype=torch.long)

    # ---- K2 baseline (fresh-init seed-42) + K3 eğitim-1 ----
    torch.manual_seed(SEED)
    router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                            router_dim=256, num_experts=4, top_k=2,
                            expert_names=["grammar_core", "pedagogy",
                                          "carpenter", "legal"])
    baseline = _dogruluk(router, Pva, Qva, yva)
    kayit("K2 BASELINE (fresh-init val top-1; kayıt-değer)", True,
          {"baseline_val_top1": round(baseline, 4),
           "rastgele_taban_3sinif": 0.3333})

    epoch_kayiplar, train_per_sinif = _egit(router, Ptr, Qtr, ytr)
    val_acc = _dogruluk(router, Pva, Qva, yva)
    k3_gec = (val_acc >= VAL_ALT_SINIR
              and val_acc >= baseline + BASELINE_MARJ
              and epoch_kayiplar[-1] < epoch_kayiplar[0])
    kayit("K3 EĞİTİM (val>=0,75 VE baseline+0,15 VE kayıp-azalıyor)", k3_gec, {
        "val_top1": round(val_acc, 4), "baseline": round(baseline, 4),
        "epoch_kayiplar": [round(k, 4) for k in epoch_kayiplar],
        "train_per_sinif_dogru": train_per_sinif,
        "kirpilan_4096_ustu": kirp_tr + kirp_va,
        "ozellik_sure_sn": round(sure, 1)})

    # ---- K4 determinizm (ikinci koşum; CPU-eğitim zaten; state_dict sha) ----
    sha_1 = _state_dict_sha(router.state_dict())
    torch.manual_seed(SEED)
    router2 = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                             router_dim=256, num_experts=4, top_k=2,
                             expert_names=["grammar_core", "pedagogy",
                                           "carpenter", "legal"])
    _egit(router2, Ptr, Qtr, ytr)
    sha_2 = _state_dict_sha(router2.state_dict())
    kayit("K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir)", sha_1 == sha_2,
          {"sha_1": sha_1, "sha_2": sha_2})
    if not all(k["gec"] for k in kapilar):
        return _hukum(kapilar, ilan_sha, damga, None, None)

    # ---- K6 şema + çıktı-yazımı ----
    kanonik_anahtarlar = sorted(TriModalRouter(
        prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256,
        num_experts=4, top_k=2).state_dict().keys())
    cikti_anahtarlar = sorted(router.state_dict().keys())
    sema_ok = kanonik_anahtarlar == cikti_anahtarlar
    if sema_ok:
        torch.save(router.state_dict(), os.path.join(REPO_KOK, CIKTI_YOL))
        cikti_sha = _sha256(os.path.join(REPO_KOK, CIKTI_YOL))
    else:
        cikti_sha = None
    kayit("K6 ŞEMA + ÇIKTI (anahtar-küme birebir + yazım)", sema_ok, {
        "anahtar_sayi": len(cikti_anahtarlar),
        "cikti": CIKTI_YOL, "cikti_sha256": cikti_sha,
        "canli_istek": 0})

    return _hukum(kapilar, ilan_sha, damga, cikti_sha, sha_1)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str, damga: str,
           cikti_sha: Any, sd_sha: Any) -> int:
    hukum_adi = ("T0154_ROUTER_GECTI" if kapilar and all(k["gec"] for k in kapilar)
                else "DUR")
    rc = 0 if hukum_adi == "T0154_ROUTER_GECTI" else 2
    hukum_json = {
        "hukum": hukum_adi, "rc": rc, "damga": damga,
        "hukum_kaynagi": "BU BETİK — elle sayı/hüküm YOK",
        "ilan": ILAN_YOL, "ilan_sha256": ilan_sha,
        "kapilar": kapilar,
        "cikti_sha256": cikti_sha, "state_dict_sha256": sd_sha,
    }
    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, HUKUM_YOL))

    s: List[str] = []
    s.append("# T-0154 — TriModalRouter eğitilmiş-ağırlık (sonuç)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
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
    s.append("## Çıktı")
    s.append("")
    if cikti_sha:
        s.append("- `data/anka_router.pt` (donmuş-desen yazım; operatör-onayı "
                 "T-0154) — sha256 `%s`" % cikti_sha)
        s.append("- state_dict sha256 `%s` (determinizm-kanıtı)" % sd_sha)
    else:
        s.append("- Çıktı YAZILMADI (kapı-düşmesi; kısmi eğitim-artefaktı yok).")
    s.append("")
    s.append("## Entegrasyon (AYRI onarım maddesi — bu turda YAPILMADI)")
    s.append("")
    s.append("Gateway `create_default` şu an router'ı fresh-init kurar; eğitilmiş")
    s.append("state_dict'in gateway'de yüklenmesi ayrı İLAN'lı kod-onarımıdır.")
    s.append("")
    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())