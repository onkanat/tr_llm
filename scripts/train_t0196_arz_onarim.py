#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0196 Faz-2 — arz-dengeli onarım devam-eğitimi (BETİKTEN; MPS).

T-0191 `train_anka_b1_5.py` reçete birebir; yalnız yüzeyler değişik:
- train: data/eval/t0196_kulliyat_arz_dengeli.jsonl (T-0196 curation; shası
  curation-meta'dan okunur ve BETİKTEN teyit edilir)
- load: data/anka_b1_5_best.pt (halef — YALNIZ-OKU)
- save: data/anka_b1_5_arz_epoch{1,2}.pt + best → data/anka_b1_5_arz.pt
- epochs=2, peak_lr=1e-4; init_val < best_val — EK3 kapısı aksi DUR rc=2
- 200-adım PROBE satırı (sn/adım + tam-koşum-tamini — beyan; koşum devam)
- history data/eval/t0196_training_history.json artımlı. Eğitim sonrası
  Faz-3 kabul-ölçümü AYRI betikle koşulur."""
import json
import math
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
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
    evaluate_val_loss,
    load_jsonl_records,
    compute_sha256,
)
from scripts.t0196_k1_onkontrol import HALEF_SHA

CIPE = {
    "data/anka_b1_5_best.pt": HALEF_SHA,
    "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
    "data/b1_5_splits/val.jsonl": "eb96241534e71ae12329de84998cfdc04aaaa50fb8cbcd6236a922d32812f6a4",
}
CURATED = os.path.join(KÖK, "data/eval/t0196_kulliyat_arz_dengeli.jsonl")
CURATED_META = os.path.join(KÖK, "data/eval/t0196_curation_meta.json")
VAL_CACHE = os.path.join(KÖK, "data/b1_5_splits/anka_fast_ds_val.pt")
TRAIN_CACHE = os.path.join(KÖK, "data/eval/t0196_fast_ds_train.pt")
VAL_CACHE_YENI = os.path.join(KÖK, "data/eval/t0196_fast_ds_val.pt")
HISTORY_PATH = os.path.join(KÖK, "data/eval/t0196_training_history.json")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0196_egitim_hukum_2026-09-29.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError(f"T0196_DUR: MPS yok (DEVICE={DEVICE}) — koşum sandbox DIŞI olmalı")

TR_100 = 200


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for yol, bek in CIPE.items():
        if compute_sha256(os.path.join(KÖK, yol)) != bek:
            print(f"T0196_DUR: çıpa-uyuşmazlığı {yol}", flush=True)
            return 2
    meta = json.load(open(CURATED_META, encoding="utf-8"))
    curated_sha = compute_sha256(CURATED)
    if curated_sha != meta["cikti_sha256"]:
        print(f"T0196_DUR: curated sha uymaz {curated_sha}", flush=True)
        return 2
    print(f"[CIPI] 4+1 çıpa BETİKTEN teyit — PASS | curated n={meta['adim_ab']['n']}", flush=True)

    import scripts.train_step_b1_5_rigorous as eski
    if eski.DEVICE.type != DEVICE.type:
        raise RuntimeError("T0196_DUR: modül-DEVICE farklı")

    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    if len(vocab.stoi) != 33114:
        print(f"T0196_DUR: vocab boyutu {len(vocab.stoi)}", flush=True)
        return 2

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    sd = torch.load(os.path.join(KÖK, "data/anka_b1_5_best.pt"), map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print("[MODELI] halef=anka_b1_5_best (e5eb114e…) YALNIZ-OKU", flush=True)

    train_records = load_jsonl_records([CURATED])
    val_records = load_jsonl_records([os.path.join(KÖK, "data/b1_5_splits/val.jsonl")])
    print(f"train(curated)={len(train_records):,} val={len(val_records):,} (test 0 okuma)", flush=True)

    block_size = 128
    train_ds = (torch.load(TRAIN_CACHE, weights_only=False)
                if os.path.exists(TRAIN_CACHE)
                else FastSampleAlignedDataset(train_records, tokenizer, vocab, model, block_size=block_size))
    if not os.path.exists(TRAIN_CACHE):
        torch.save(train_ds, TRAIN_CACHE)
    if os.path.exists(VAL_CACHE):
        val_ds = torch.load(VAL_CACHE, weights_only=False)
    elif os.path.exists(VAL_CACHE_YENI):
        val_ds = torch.load(VAL_CACHE_YENI, weights_only=False)
    else:
        val_ds = FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=block_size)
        torch.save(val_ds, VAL_CACHE_YENI)
    print(f"cache: train={len(train_ds)} val={len(val_ds)}", flush=True)

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
               "ilan": "data/eval/t0196_onarim_ilan_2026-09-29.md",
               "cipe": CIPE, "curated_sha256": curated_sha,
               "recepte": {"epochs": epochs, "batch": batch_size, "accum": grad_accum_steps,
                            "peak_lr": peak_lr, "min_lr": min_lr, "warmup": warmup_steps,
                            "block_size": block_size, "val_max_batches": 50, "seed": 42,
                            "load": "data/anka_b1_5_best.pt", "kaynak": "T-0191 reçete birebir"},
               "init": None, "probe": None, "history": []}

    def history_yaz() -> None:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    init_loss, init_ppl = evaluate_val_loss(model, val_loader)  # 50-batch kısmi-örneklem
    history["init"] = {"val_loss": init_loss, "val_ppl": init_ppl}
    history_yaz()
    print(f"[INIT] Val Loss: {init_loss:.4f} | PPL: {init_ppl:.2f} (max_batches=50)", flush=True)

    global_step = 0
    probe_t0 = time.time()
    ckpt_sha = {}
    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses, accum_loss = [], 0.0
        ep_t0 = time.time()
        for b_idx, (x, y, sm) in enumerate(train_loader):
            x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
            _, loss = model(x, y, sm)
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
                    recent = float(np.mean(epoch_losses[-20:])) or 0.0
                    print(f"  Epoch {ep}/{epochs} | adım {global_step}/{total_opt_steps} | "
                          f"train: {recent:.4f} | lr {cur:.6f}", flush=True)
        ep_dt = time.time() - ep_t0
        ep_train = float(np.mean(epoch_losses))
        ep_val, ep_ppl = evaluate_val_loss(model, val_loader)

        ckpt = os.path.join(KÖK, "data", f"anka_b1_5_arz_epoch{ep}.pt")
        torch.save(model.state_dict(), ckpt)
        ckpt_sha[os.path.basename(ckpt)] = compute_sha256(ckpt)
        history["history"].append({"epoch": ep, "train_loss": ep_train, "val_loss": ep_val,
                                   "val_ppl": ep_ppl, "path": ckpt, "duration": ep_dt,
                                   "sha256": ckpt_sha[os.path.basename(ckpt)]})
        history_yaz()
        print(f">>> [Epoch {ep}] train {ep_train:.4f} | val {ep_val:.4f} (PPL {ep_ppl:.2f}) "
              f"| {ep_dt:.1f} sn | ckpt sha BETİKTEN ({ckpt_sha[os.path.basename(ckpt)][:16]}…)", flush=True)

    best = min(history["history"], key=lambda c: c["val_loss"])
    best_path = os.path.join(KÖK, "data", "anka_b1_5_arz.pt")
    shutil.copyfile(best["path"], best_path)
    best_sha = compute_sha256(best_path)
    history["best"] = {"epoch": best["epoch"], "val_loss": best["val_loss"],
                       "path": best_path, "sha256": best_sha}
    history_yaz()

    # EK3 pozitif-etki kapısı
    if best["val_loss"] >= init_loss:
        print(f"T0196_DUR: EK3 — best_val {best['val_loss']:.4f} >= init {init_loss:.4f}", flush=True)
        donem = {"task": "T-0196", "hukum": "T0196_EGITIM_DUR", "rc": 2,
                 "kapilar": {"EK3_pozitif_etki": False, "init_val_loss": init_loss,
                              "best_val_loss": best["val_loss"], "best_epoch": best["epoch"]},
                 "best_sha256": best_sha, "history_path": HISTORY_PATH,
                 "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(HUKUM_PATH, "w", encoding="utf-8") as f:
            json.dump(donem, f, ensure_ascii=False, indent=2)
        print("T0196_EGITIM_DUR", flush=True)
        return 2

    donem = {"task": "T-0196", "hukum": "T0196_EGITIM_GECTI", "rc": 0,
             "kapilar": {"EK1_cipe": "PASS", "EK2_curated_sha": "PASS",
                          "EK3_pozitif_etki": True,
                          "init_val_loss": init_loss, "best_val_loss": best["val_loss"],
                          "val_dusus": init_loss - best["val_loss"], "best_epoch": best["epoch"]},
             "probe": history.get("probe"),
             "best": {"path": best_path, "sha256": best_sha},
             "history_path": HISTORY_PATH,
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    print(f"[BEST] epoch {best['epoch']} val {best['val_loss']:.4f} < init {init_loss:.4f} "
          f"-> {best_path} sha={best_sha}", flush=True)
    print("T0196_EGITIM_GECTI", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())