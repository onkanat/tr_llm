#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0200 — mutasyon-kaniti + paylaşımlı bileşenler (BETİKTEN; MPS; koşum-ÖNCESİ).

Şartname MADDE-5: küçük-koşum (12 opt-adım, tam-reçete) fail-closed kanıtlar:
- K1 KL pozitif+sonlu, adım-ort >0 (çıpa-sessiz-yutma kapısı; KL≡0 ⇒ DUR)
- K2 KL gradyan-akışı: 12 adımda kl_ort düşüş (yön-beyanı BETİKTEN kayıt)
- K3 karışım-ds uzunluk çıpası: len == 5491 (SFT) + 5491 (pencere) == 10982
- K4 pencere_ce_olc determinizmi: iki-koşum birebir
- K5 init pencere-CE == 3,9268 birebir (halef ce_ort, t0199_m1_b1_5_best.json —
  BETİKTEN-okunur; uyuşmaz ⇒ ölçüm-kabı bozuk DUR)
Hüküm: T0200_MUTASYON_GECTI rc=0 | T0200_MUTASYON_DUR rc=2.
Paylaşımlı yüzeyler (train_t0200_optimum_zarar.py import eder):
KarısımDataset (SFT ağırlıksız + PencereDataset, kaynak-bayrağı), pencere_ce_olc,
kl_pencere_kayip. NOTE: cache-pickle YOK (T-0198 __main__-modül dersi) — her betik
deterministik yeniden-kurulum yapar ve len-çıpasını BETİKTEN doğrular."""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict
from scripts.train_step_b1_5_rigorous import (
    FastSampleAlignedDataset,
    compute_sha256,
)
from scripts.train_t0198_hiza import HALEF_SHA  # halef-çıpa — yalnız-OKU

# ---- Şartname BÖLÜM-A çıpaları (BETİKTEN sabit; 30 Eyl shasum ölçümü) ----
CIPI_SHA = {
    "data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50",
    "data/anka_b1_5_best.pt": HALEF_SHA,
    "data/anka_b1_5_arz.pt": "80b0fd9a5a60de90536ad4d6d3a3f8eb1de16d1474501a0eac669d2458f106fa",
    "data/anka_b1_5_usaf.pt": "8f846023f2fb2374dd411f674b199f373198c108cd961c836c44e094948b2c10",
    "data/anka_b1_5_hiza.pt": "05d8e4cf55a2e27e46eadab6b93f50b2cd90202b6f17e2f57958d4ac4d5727df",
}
HOCA = "data/anka_base_v2.pt"          # KL-öğretmen = taban (akıcılık-referansı)
HALEF = "data/anka_b1_5_best.pt"       # eğitim-başlangıcı — YALNIZ-OKU
SFT_PATH = os.path.join(KÖK, "data/eval/t0197_uretim_saf.jsonl")
BIN_PATH = os.path.join(KÖK, "data/anka_a1r_pretrain.bin")
BIN_META = os.path.join(KÖK, "data/anka_a1r_pretrain.bin.meta.json")
T0199_HUKUM = os.path.join(KÖK, "data/eval/t0199_tani_hukum_2026-09-30.json")
T0199_M1_HALEF = os.path.join(KÖK, "data/eval/t0199_m1_b1_5_best.json")
BLOCK = 128
BETA = 1.0            # MADDE-9: BETİKTEN sabit; değişimi ayrı-görev
PEAK_LR = 2e-5        # MADDE-8
MIN_LR = 5e-6
WARMUP = 20
BATCH = 16
ACCUM = 2
PENCERE_N = 5491
DS_N_BEKLENEN = 10982
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0200_mutasyon_hukum_2026-09-30.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError("T0200_DUR: MPS yok — koşum sandbox DIŞI olmalı")


class PencereDataset:
    """Taban-natural pencere-dataset (KristalDataset kalıbı; sabit-adet BETİKTEN).
    koşum-2 DUR-onarımı: SFT-pad [128] ile collate-stack için pencere 129-uzun
    okunur (x/y 128/128; y-shift pretrain-hedefi birebir); havuz n_win-1=781.249
    (son-pencere [b+129] taşma-guard). pencere_ce_olc (K4/K5) rig-birebir [127] kalır —
    eğitim-penceresi ile ölçüm-penceresi rolleri AYRI."""

    def __init__(self, bin_path: str, model: KristalLM, device: torch.device,
                 adet: int = PENCERE_N, seed: int = 42):
        mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n_win = int(mm.size // BLOCK)
        rng = random.Random(seed)
        self.baslar = sorted(rng.sample(range(n_win - 1), min(adet, n_win - 1)))
        mm_i = np.asarray(mm, dtype=np.uint16)  # tek-okuma materializasyonu
        self.samples = []
        for b in self.baslar:
            seq = mm_i[b * BLOCK:b * BLOCK + BLOCK + 1].astype(np.int64)
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            sm = model.embedding.compute_sign_mask(x.unsqueeze(0))[0]
            # koşum-4 DUR-onarımı: sm CPU'da SAKLANIR (compute_sign_mask saf-CPU;
            # cihaz-taşıma eğitim-döngüsünde). MPS-resident sm'lerin default-collate
            # torch.stack'i SIGSEGV çöküyor (koşum-4 çökme-raporu 061257:
            # structured_cat_out_mps → MPSStream::copy → blitCDMBuffer).
            self.samples.append((x, y, sm))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, sm = self.samples[idx]
        return (x, y, sm, torch.tensor(1.0))


class KarısımDataset:
    """SFT (kaynak 0) + Pencere (kaynak 1) birleşik-dataset; her örnek
    (x, y, sm, kaynak) 4'lüsü — default collate stack-uyumlu."""

    def __init__(self, sft_records, tokenizer, vocab, model, device: torch.device):
        self.sft = FastSampleAlignedDataset(sft_records, tokenizer, vocab, model,
                                            block_size=BLOCK)
        self.pencere = PencereDataset(BIN_PATH, model, device)

    def __len__(self) -> int:
        return len(self.sft) + len(self.pencere)

    def __getitem__(self, idx: int):
        if idx < len(self.sft):
            x, y, sm = self.sft[idx]
            return (x, y, sm, torch.tensor(0.0))
        return self.pencere[idx - len(self.sft)]


def pencere_ce_olc(model: KristalLM, device: torch.device) -> float:
    """taban_akicilik_tanisi.py kap-rig yeniden-söylem (BETİKTEN; kaynak-dokunuş YOK):
    rng=Random(7), sorted(rng.sample(range(mm.size//128, 25))); pencere-128 shift;
    tam-CE sign-mask; ort-CE. K4/K5 çıpasının ölçüm-kabı.
    koşum-1 DUR-onarımı (T-0199 rig-teyidi): dropout ölçümü bozar (no_grad dropout'u
    kapatmaz) → ölçüm eval-modu (evaluate_val_loss kalıbı: eval→ölç→train)."""
    was_training = model.training
    model.eval()
    mm = np.memmap(BIN_PATH, dtype=np.uint16, mode="r")
    n_win = int(mm.size // BLOCK)
    rng = random.Random(7)
    baslar = sorted(rng.sample(range(n_win), 25))
    kayiplar = []
    mm_i = np.asarray(mm, dtype=np.uint16)
    for b in baslar:
        seq = mm_i[b * BLOCK:(b + 1) * BLOCK].astype(np.int64)
        x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            sm = model.embedding.compute_sign_mask(x).to(device)
            _, loss = model(x, targets=y, sign_mask=sm)
        kayiplar.append(float(loss.item()))
        torch.mps.empty_cache()  # R4: pencere-döngü-cache-boşaltma
    if was_training:
        model.train()
    return round(float(np.mean(kayiplar)), 4)


def kl_pencere_kayip(logits_student: torch.Tensor, hoca_model: KristalLM,
                     x: torch.Tensor, y: torch.Tensor, sm: torch.Tensor,
                     kaynak_cpu: torch.Tensor) -> torch.Tensor:
    """Forward-KL T=1 tam-vocab YALNIZ pencere-konumlarında (MADDE-9):
    KL = Σ_v p_t(v)·[log p_t(v) − log p_s(v)] (öğrenci-satır-seçimi; öğretmen no_grad;
    logits_t del → MPS-baskı R4).
    koşum-3 DUR-onarımı: satır-seçimi CPU'da nonzero + MPS'te index_select —
    MPS'te nonzero+fancy-indexing SIGSEGV çökmesi (koşum-3 çökme-raporu 060754:
    MPSStream::copy, fault 0x3f80000020) — ilk-kez-KL-MPS yolu."""
    satirlar = (kaynak_cpu == 1.0).nonzero(as_tuple=True)[0]
    if satirlar.numel() == 0:
        return torch.zeros((), device=logits_student.device)
    idx = satirlar.to(logits_student.device)
    xs = torch.index_select(x, 0, idx)
    ys = torch.index_select(y, 0, idx)
    sms = torch.index_select(sm, 0, idx)
    with torch.no_grad():
        logits_t, _ = hoca_model(xs, ys, sms)
        log_p_t = F.log_softmax(logits_t, dim=-1)
        del logits_t  # R4: öğretmen-logit referansı hemen bırak
    sel_s = torch.index_select(logits_student, 0, idx)
    log_p_s = F.log_softmax(sel_s, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1).mean()
    del log_p_t
    return kl


def yukle_checkpoint(yol: str, model: KristalLM) -> None:
    sd = torch.load(yol, map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)


def cipe_dogrula() -> None:
    """Şartname BÖLÜM-A fail-closed ön-kontrol (5-sha + sha-triplet + t0199 hüküm)."""
    for yol, hiz in CIPI_SHA.items():
        if compute_sha256(os.path.join(KÖK, yol)) != hiz:
            raise RuntimeError(f"T0200_DUR: çıpa-uyuşmazlığı {yol}")
    meta = json.load(open(os.path.join(KÖK, "data/eval/t0197_onarim_meta.json"), encoding="utf-8"))
    if compute_sha256(SFT_PATH) != meta["cikti_sha256"]:
        raise RuntimeError("T0200_DUR: üretim-saf sha uymaz")
    bm = json.load(open(BIN_META, encoding="utf-8"))
    if compute_sha256(BIN_PATH) != bm["bin_sha256"]:
        raise RuntimeError("T0200_DUR: pretrain-bin sha uymaz")
    h199 = json.load(open(T0199_HUKUM, encoding="utf-8"))
    if h199.get("hukum") != "T0199_TANI_KAYDI" or h199.get("rc") != 0:
        raise RuntimeError(f"T0200_DUR: T-0199 hüküm {h199.get('hukum')} rc={h199.get('rc')}")


def ortak_kur():
    """vocab/tokenizer/model(halef)/öğretmen(hoca) + datasets BETİKTEN kurar."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    if len(vocab.stoi) != 33114:
        raise RuntimeError(f"T0200_DUR: vocab boyutu {len(vocab.stoi)}")

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    yukle_checkpoint(os.path.join(KÖK, HALEF), model)
    model.to(DEVICE)
    model.train()

    hoca = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    yukle_checkpoint(os.path.join(KÖK, HOCA), hoca)
    hoca.to(DEVICE)
    hoca.eval()
    for p in hoca.parameters():
        p.requires_grad_(False)

    sft_records = [json.loads(l) for l in open(SFT_PATH, encoding="utf-8")]
    ds = KarısımDataset(sft_records, tokenizer, vocab, model, DEVICE)
    return vocab, tokenizer, model, hoca, ds


def get_lr(step: int, total_steps: int) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, total_steps - WARMUP)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    cipe_dogrula()
    print("[CIPI] 5-sha + sha-triplet + T-0199 hüküm BETİKTEN — PASS", flush=True)

    _, _, model, hoca, ds = ortak_kur()
    if len(ds) != DS_N_BEKLENEN:
        print(f"T0200_MUTASYON_DUR: ds-uzunluk {len(ds)} != {DS_N_BEKLENEN}", flush=True)
        return 2
    print(f"[K3] karışım-ds n={len(ds)} (SFT+Pencere) — PASS", flush=True)

    # ---- K5: init pencere-CE birebir-çıpası + K4 determinizmi ----
    ce1 = pencere_ce_olc(model, DEVICE)
    cipa = json.load(open(T0199_M1_HALEF, encoding="utf-8"))["ce_ort"]
    ce2 = pencere_ce_olc(model, DEVICE)
    print(f"[K4] iki-koşum pencere-CE: {ce1} / {ce2} birebir: {ce1 == ce2}", flush=True)
    if ce1 != cipa or ce1 != ce2:
        print(f"T0200_MUTASYON_DUR: init pencere-CE {ce1} != çıpa {cipa} "
              f"(t0199_m1_b1_5_best BETİKTEN) veya determinizmi-yok", flush=True)
        return 2
    print(f"[K5] init pencere-CE {ce1} == halef-çıpa {cipa} birebir — PASS", flush=True)

    # ---- K1/K2: 12 opt-adım (tam-reçete; kl_ort yön-beyanı) ----
    torch.manual_seed(42)
    opt_steps = 12
    total_steps = 344  # tam-koşumla aynı schedule (10.982/16/2)
    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    kl_seri, adim = [], 0
    accum_kl, accum_say = 0.0, 0
    b_iter = iter(loader)
    for _ in range(opt_steps):
        for _ac in range(ACCUM):
            x, y, sm, kz = next(b_iter)
            x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
            # kz CPU'da KALIR — koşum-3 onarımı: nonzero CPU'da (kl_pencere_kayip)
            logits, loss_std = model(x, y, sm)
            kl = kl_pencere_kayip(logits, hoca, x, y, sm, kz)
            kayip = loss_std + BETA * kl
            del logits  # R4
            kayip.backward()
            accum_kl += float(kl.item())
            accum_say += 1
        adim += 1
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(adim, total_steps)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.mps.empty_cache()  # R4: koşum-5 OOM-onarımı — KL-grağı 271MB-sınıf
        # geçici-tensor'lar iterasyon-başına allocator-parçalanması biriktiriyor
        # (koşum-5: adım-4 MPS-OOM, other=1,90GiB = canlı-gateway payı sabit)
        kl_ort = accum_kl / max(1, accum_say)
        kl_seri.append(round(kl_ort, 6))
        accum_kl, accum_say = 0.0, 0
        print(f"  adım {adim:>2}/{opt_steps} kl_ort={kl_ort:.6f}", flush=True)
    torch.mps.empty_cache()

    k1 = all(math.isfinite(v) and v > 0 for v in kl_seri)
    k2 = kl_seri[-1] < kl_seri[0]
    kapilar = {"k1_kl_pozitif_sonlu": k1, "k2_kl_dusus_yon": k2,
               "k3_ds_uzunluk": True, "k4_determinizm": ce1 == ce2,
               "k5_init_ce_birebir": ce1 == cipa}
    gecti = all(kapilar.values())
    hukum = "T0200_MUTASYON_GECTI" if gecti else "T0200_MUTASYON_DUR"
    donem = {"task": "T-0200", "hukum": hukum, "rc": 0 if gecti else 2,
             "kapilar": kapilar, "kl_seri": kl_seri, "kl_ilk": kl_seri[0],
             "kl_son": kl_seri[-1], "beta": BETA, "peak_lr": PEAK_LR,
             "pencere_ce_init": ce1, "pencere_ce_cipa": cipa, "ds_n": len(ds),
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())