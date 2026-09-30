#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0198 Faz-4 — hiza-ağırlıklı üretim kaybı devam-eğitimi (BETİKTEN; MPS).

T-0197 `train_t0197_uretim_saf.py` mimarisi birebir; TEK-değişken ağırlık:
- loss: hiza-ağırlıklı CE (ALPHA=2,0; scripts/t0198_mutasyon_kanit.HizaliDataset)
- eval-time kayıp AĞIRLIKSIZ (MADDE-3; T-0053 ölçek-dersi) — standard `model(x,y,sm)`
- load: data/anka_b1_5_best.pt (halef — YALNIZ-OKU); train: t0197 üretim-saf
- save: data/anka_b1_5_hiza_epoch{1,2}.pt + best → data/anka_b1_5_hiza.pt
- Faz-3 AYRI betikle koşulur (scripts/t0198_faz3_kabul.py)."""
import json
import math
import os
import random
import shutil
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
    evaluate_val_loss,
    load_jsonl_records,
    compute_sha256,
)
from scripts.t0198_mutasyon_kanit import (  # T-0198 — K3 GECTI sonrası yüzey
    ALPHA,
    HizaliDataset,
)

HALEF_SHA = "e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7"
MUTASYON_PATH = os.path.join(KÖK, "data/eval/t0198_mutasyon.json")
TRAIN_PATH = os.path.join(KÖK, "data/eval/t0197_uretim_saf.jsonl")
TRAIN_META = os.path.join(KÖK, "data/eval/t0197_onarim_meta.json")
VAL_CACHE = os.path.join(KÖK, "data/b1_5_splits/anka_fast_ds_val.pt")  # T-0191/T-0197 val-cache (SADECE-OKU)
HISTORY_PATH = os.path.join(KÖK, "data/eval/t0198_training_history.json")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0198_egitim_hukum_2026-09-29.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError("T0198_DUR: MPS yok — koşum sandbox DIŞI olmalı")

TR_100 = 200


def agirlikli_kayip(logits: torch.Tensor, ys: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """Hiza-ağırlıklı CE (MADDE-1): pad/-100 ağırlık 0; ALPHA-jetonlar 2,0.
    Kayıp-ölçeği İLAN-değişir — eval-time ağırlıksız (MADDE-3) birebir."""
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1),
                         reduction="none", ignore_index=-100)
    w = ws.reshape(-1)
    toplam = w.sum()
    if float(toplam) <= 0.0:
        raise RuntimeError("T0198_DUR: parti-ağırlık-toplamı 0 — sessiz geçiş yok")
    return (ce * w).sum() / toplam


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    import scripts.train_t0197_uretim_saf as ts  # halef — yalnız-OKU (CIPE)
    for yol, hiz in ts.CIPE.items():
        if compute_sha256(os.path.join(KÖK, yol)) != hiz:
            print(f"T0198_DUR: çıpa-uyuşmazlığı {yol}", flush=True)
            return 2
    mut = json.load(open(MUTASYON_PATH, encoding="utf-8"))
    if mut.get("hukum") != "T0198_MUTASYON_GECTI" or mut.get("rc") != 0:
        print(f"T0198_DUR: mutasyon hükmü {mut.get('hukum')} rc={mut.get('rc')}", flush=True)
        return 2
    onarim_sha = compute_sha256(TRAIN_PATH)
    meta197 = json.load(open(os.path.join(KÖK, "data/eval/t0197_onarim_meta.json"), encoding="utf-8"))
    if onarim_sha != meta197["cikti_sha256"]:
        print(f"T0198_DUR: üretim-saf sha uymaz {onarim_sha}", flush=True)
        return 2
    print(f"[CIPI] çıpa + mutasyon-hüküm BETİKTEN teyit — PASS | train n={sum(1 for _ in open(TRAIN_PATH, encoding='utf-8')):,}".replace(", ", ""), flush=True)

    import scripts.train_step_b1_5_rigorous as eski
    if eski.DEVICE.type != DEVICE.type:
        raise RuntimeError("T0198_DUR: modül-DEVICE farklı")

    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    if len(vocab.stoi) != 33114:
        print(f"T0198_DUR: vocab boyutu {len(vocab.stoi)}", flush=True)
        return 2

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    sd = torch.load(os.path.join(KÖK, "data/anka_b1_5_best.pt"), map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print("[MODELI] halef=anka_b1_5_best (e5eb114e…) YALNIZ-OKU", flush=True)

    train_records = load_jsonl_records([TRAIN_PATH])
    val_records = load_jsonl_records([os.path.join(KÖK, "data/b1_5_splits/val.jsonl")])
    print(f"train(üretim-saf)={len(train_records):,} val={len(val_records):,}", flush=True)

    # HizaliDataset: deterministik yeniden-kurulum (K3 kopya-bütünlüğü bit-birebir PASS;
    # __main__.HizaliDataset-pickle'lı cache yüklenemez — BETİKTEN bilinen-kalıp) +
    # uzunluk-çıpası mutasyon-hüküm ds_n'e (BETİKTEN):
    train_ds = HizaliDataset(train_records, tokenizer, vocab, model, block_size=128)
    if len(train_ds) != mut["ds_n"]:
        print(f"T0198_DUR: hiza-ds uzunluk {len(train_ds)} != mutasyon ds_n {mut['ds_n']}", flush=True)
        return 2
    val_ds = torch.load(VAL_CACHE, weights_only=False)
    print(f"ds: train={len(train_ds)} (ağırlıklı; çıpa {mut['ds_n']}) | "
          f"val={len(val_ds)} (ağırlıksız-eval)", flush=True)

    batch_size, grad_accum_steps = 16, 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    epochs = 2
    total_opt_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps, peak_lr, min_lr = 50, 1e-4, 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    history = {"damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ilan": "data/eval/t0198_egitim_ilan_2026-09-29.md",
               "mutasyon": {"hukum": mut["hukum"], "alpha": ALPHA},
               "recepte": {"epochs": epochs, "batch": batch_size, "accum": grad_accum_steps,
                            "peak_lr": peak_lr, "min_lr": min_lr, "warmup": warmup_steps,
                            "block_size": 128, "val_max_batches": 50, "seed": 42,
                            "load": "data/anka_b1_5_best.pt", "alpha": ALPHA,
                            "agirlik": {"kesisim": ALPHA, "predicate": ALPHA, "diger": 1.0,
                                         "pad_eos_disi": 0.0, "eval_agirliksiz": True}},
               "init": None, "probe": None, "history": []}

    def history_yaz() -> None:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    init_loss, init_ppl = evaluate_val_loss(model, val_loader)  # ağırlıksız — MADDE-3
    history["init"] = {"val_loss": init_loss, "val_ppl": init_ppl}
    history_yaz()
    print(f"[INIT] Val Loss: {init_loss:.4f} | PPL: {init_ppl:.2f} (ağırlıksız; 50-batch)", flush=True)

    global_step = 0
    probe_t0 = time.time()
    ckpt_sha = {}
    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses, accum_loss = [], 0.0
        ep_t0 = time.time()
        for b_idx, (x, y, sm, w) in enumerate(train_loader):
            x, y, sm, w = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE), w.to(DEVICE)
            logits, _ = model(x, y, sm)
            loss = agirlikli_kayip(logits, y, w)
            del logits  # MPS-baskısı: ara-logit referansını hemen bırak (OOM-onarım)
            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(train_loader):
                global_step += 1
                cur = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses.append(accum_loss / grad_accum_steps)
                accum_loss = 0.0
                if global_step == TR_100:
                    dt = time.time() - probe_t0
                    adim = dt / TR_100
                    kalan = total_opt_steps - TR_100
                    history["probe"] = {"adim": TR_100, "sn": round(dt, 1),
                                         "sn_adim": round(adim, 4),
                                         "tam_kosum_tamini_min": round(kalan * adim / 60.0, 1)}
                    history_yaz()
                    print(f"PROBE: {TR_100} adım sn={dt:.1f} sn/adım={adim:.3f} "
                          f"kalan-adım={kalan} tam-koşum-tamini≈{kalan * adim / 60.0:.1f} dk "
                          f"(beyan; koşum devam)", flush=True)
                if global_step % 50 == 0 or global_step == 1:
                    torch.mps.empty_cache()  # OOM-onarım: 50-adımda-bir MPS-cache-boşaltma
                    recent = float(np.mean(epoch_losses[-20:])) or 0.0
                    print(f"  Epoch {ep}/{epochs} | adım {global_step}/{total_opt_steps} | "
                          f"agirlikli: {recent:.4f} | lr {cur:.6f}", flush=True)
        ep_dt = time.time() - ep_t0
        ep_train = float(np.mean(epoch_losses))
        ep_val, ep_ppl = evaluate_val_loss(model, val_loader)  # ağırlıksız — MADDE-3
        torch.mps.empty_cache()  # OOM-onarım: epoch-val-sonu MPS-cache-boşaltma

        ckpt = os.path.join(KÖK, "data", f"anka_b1_5_hiza_epoch{ep}.pt")
        torch.save(model.state_dict(), ckpt)
        ckpt_sha[os.path.basename(ckpt)] = compute_sha256(ckpt)
        history["history"].append({"epoch": ep, "train_loss_agirlikli": ep_train,
                                   "val_loss_agirliksiz": ep_val, "val_ppl": ep_ppl,
                                   "path": ckpt, "duration": ep_dt,
                                   "sha256": ckpt_sha[os.path.basename(ckpt)]})
        history_yaz()
        print(f">>> [Epoch {ep}] train(agirlikli) {ep_train:.4f} | val(agirliksiz) {ep_val:.4f} "
              f"(PPL {ep_ppl:.2f}) | {ep_dt:.1f} sn | ckpt sha BETİKTEN "
              f"({ckpt_sha[os.path.basename(ckpt)][:16]}…)", flush=True)

    best = min(history["history"], key=lambda c: c["val_loss_agirliksiz"])
    best_path = os.path.join(KÖK, "data", "anka_b1_5_hiza.pt")
    shutil.copyfile(best["path"], best_path)
    best_sha = compute_sha256(best_path)
    history["best"] = {"epoch": best["epoch"], "val_loss_agirliksiz": best["val_loss_agirliksiz"],
                       "path": best_path, "sha256": best_sha}
    history_yaz()

    if best["val_loss_agirliksiz"] >= init_loss:
        print(f"T0198_DUR: EK3 — best_val(ağırlıksız) {best['val_loss_agirliksiz']:.4f} "
              f">= init {init_loss:.4f}", flush=True)
        donem = {"task": "T-0198", "hukum": "T0198_EGITIM_DUR", "rc": 2,
                 "kapilar": {"EK3_pozitif_etki": False, "init_val_loss_agirliksiz": init_loss,
                              "best_val_loss_agirliksiz": best["val_loss_agirliksiz"],
                              "best_epoch": best["epoch"]},
                 "best_sha256": best_sha, "history_path": HISTORY_PATH,
                 "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(HUKUM_PATH, "w", encoding="utf-8") as f:
            json.dump(donem, f, ensure_ascii=False, indent=2)
        print("T0198_EGITIM_DUR", flush=True)
        return 2

    donem = {"task": "T-0198", "hukum": "T0198_EGITIM_GECTI", "rc": 0,
             "kapilar": {"EK1_cipe": "PASS", "EK2_onarim_sha": "PASS",
                          "EK3_pozitif_etki": True, "EK4_mutasyon": "PASS",
                          "init_val_loss_agirliksiz": init_loss,
                          "best_val_loss_agirliksiz": best["val_loss_agirliksiz"],
                          "val_dusus": init_loss - best["val_loss_agirliksiz"],
                          "best_epoch": best["epoch"]},
             "probe": history.get("probe"),
             "best": {"path": best_path, "sha256": best_sha},
             "history_path": HISTORY_PATH,
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    print(f"[BEST] epoch {best['epoch']} val(ağırlıksız) {best['val_loss_agirliksiz']:.4f} "
          f"< init {init_loss:.4f} -> {best_path} sha={best_sha}", flush=True)
    print("T0198_EGITIM_GECTI", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())