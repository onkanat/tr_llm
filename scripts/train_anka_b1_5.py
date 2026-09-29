#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0191 — B1.5 eğitim-koşumu (Anka-çıpa, T-0190 regen-split).

Reçete `train_step_b1_5_rigorous.py` ile birebir (3 epoch / batch 16 /
accum 2 / peak_lr 3e-4 cosine / block 128 / seed 42); o betik DOKUNULMAZ,
sınıf ve yardımcılar import'la yeniden-kullanılır. Değişen yüzey:
- çıpa: data/anka_base_v2.pt (tarihsel kristal-zincir silindi; T-0162)
- vocab: data/rebuild/vocab_anka_r1_33114.json (33.114 — data/vocab.json
  silinmiş durumda)
- tokenizer: literal_entity_mode=True (T-0186/T-0189 kanonik-çıpa)
- çıkış-adları: data/anka_b1_5_*.pt — kristal_model.pt YAZILMAZ
- tokenize-cache: data/b1_5_splits/anka_fast_ds_{train,val}.pt (yeni ad;
  eski 32.816-vocab cache dokunulmaz)
- history: data/eval/t0191_training_history.json her-epoch artımlı yazılır."""
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
from scripts.train_step_b1_5_rigorous import (  # tarihsel betik dokunulmaz
    FastSampleAlignedDataset,
    evaluate_val_loss,
    load_jsonl_records,
    compute_sha256,
)

# --- İLAN-çıpası (koşum-başı BETİKTEN) ---
CIPE = {
    "data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50",
    "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
    "data/b1_5_splits/test.jsonl": "f106e7d2c7854ea653a6039ac6fc2578ee7b8e2d36926b2763bc7ae2e2926260",
    "data/b1_5_splits/val.jsonl": "eb96241534e71ae12329de84998cfdc04aaaa50fb8cbcd6236a922d32812f6a4",
    "data/b1_5_splits/train.jsonl": "c2d8480b866f9fd0b57d7e671eb091c91e5c80cca5a4f133b62e61cdafaae39e",
}

DATA_DIR = os.path.join(KÖK, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")
TRAIN_PATH = os.path.join(SPLITS_DIR, "train.jsonl")
VAL_PATH = os.path.join(SPLITS_DIR, "val.jsonl")
HISTORY_PATH = os.path.join(KÖK, "data/eval", "t0191_training_history.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError(f"T0191_DUR: MPS yok (DEVICE={DEVICE}) — koşum sandbox DIŞI olmalı")


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # EK1-EK2 çıpa-kapısı
    for yol, beklenen in CIPE.items():
        ölç = compute_sha256(os.path.join(KÖK, yol))
        if ölç != beklenen:
            print(f"T0191_DUR: çıpa-uyuşmazlığı {yol}: {ölç}")
            return 2
    print("[CIPI] 6/6 çıpa BETİKTEN teyit — PASS", flush=True)

    # Modül-cihaz ile betik-cihaz uyuşmalı (evaluate_val_loss modül-DEVICE kullanır)
    import scripts.train_step_b1_5_rigorous as eski
    if eski.DEVICE.type != DEVICE.type:
        raise RuntimeError("T0191_DUR: modül-DEVICE ile koşum-DEVICE farklı")

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "rebuild", "vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    print(f"vocab: {len(vocab.stoi):,} jeton (33.114 beklenir)", flush=True)
    if len(vocab.stoi) != 33114:
        print("T0191_DUR: vocab boyutu 33.114 değil")
        return 2

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    ata = os.path.join(DATA_DIR, "anka_base_v2.pt")
    sd = torch.load(ata, map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    print(f"[SOYAGACI] ata=anka_base_v2 sha={compute_sha256(ata)}", flush=True)
    model.to(DEVICE)

    train_records = load_jsonl_records([TRAIN_PATH])
    val_records = load_jsonl_records([VAL_PATH])
    print(f"train={len(train_records):,} val={len(val_records):,} (test 0 okuma)", flush=True)

    block_size = 128
    train_cache = os.path.join(SPLITS_DIR, "anka_fast_ds_train.pt")
    val_cache = os.path.join(SPLITS_DIR, "anka_fast_ds_val.pt")
    train_ds = (torch.load(train_cache, weights_only=False)
                if os.path.exists(train_cache)
                else FastSampleAlignedDataset(train_records, tokenizer, vocab, model, block_size=block_size))
    if not os.path.exists(train_cache):
        torch.save(train_ds, train_cache)
    val_ds = (torch.load(val_cache, weights_only=False)
              if os.path.exists(val_cache)
              else FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=block_size))
    if not os.path.exists(val_cache):
        torch.save(val_ds, val_cache)
    print(f"cache: train={len(train_ds)} val={len(val_ds)} örnekleme", flush=True)

    batch_size, grad_accum_steps = 16, 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    epochs = 3
    total_opt_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps, peak_lr, min_lr = 50, 3e-4, 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    history = {"damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ilan": "data/eval/t0191_b15_egitim_ilan_2026-09-29.md",
               "cipe": CIPE,
               "recepte": {"epochs": epochs, "batch": batch_size, "accum": grad_accum_steps,
                            "peak_lr": peak_lr, "min_lr": min_lr, "warmup": warmup_steps,
                            "block_size": block_size, "val_max_batches": 50, "seed": 42},
               "init": None, "history": []}

    def history_yaz() -> None:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    init_loss, init_ppl = evaluate_val_loss(model, val_loader)  # 50-batch kısmi-örneklem (İLAN-3)
    history["init"] = {"val_loss": init_loss, "val_ppl": init_ppl}
    history_yaz()  # artımlı yazım — koşum-erken çocuğu kaybı önler
    print(f"[INIT] Val Loss: {init_loss:.4f} | PPL: {init_ppl:.2f} (max_batches=50)", flush=True)

    global_step = 0
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
                if global_step % 50 == 0 or global_step == 1:
                    recent = float(np.mean(epoch_losses[-20:])) or 0.0
                    print(f"  Epoch {ep}/{epochs} | adım {global_step}/{total_opt_steps} | "
                          f"train: {recent:.4f} | lr {cur:.6f}", flush=True)
        ep_dt = time.time() - ep_t0
        ep_train = float(np.mean(epoch_losses))
        ep_val, ep_ppl = evaluate_val_loss(model, val_loader)

        ckpt = os.path.join(DATA_DIR, f"anka_b1_5_epoch{ep}.pt")
        torch.save(model.state_dict(), ckpt)
        ckpt_sha[os.path.basename(ckpt)] = compute_sha256(ckpt)
        history["history"].append({"epoch": ep, "train_loss": ep_train, "val_loss": ep_val,
                                   "val_ppl": ep_ppl, "path": ckpt, "duration": ep_dt,
                                   "sha256": ckpt_sha[os.path.basename(ckpt)]})
        history_yaz()  # her-epoch artımlı
        print(f">>> [Epoch {ep}] train {ep_train:.4f} | val {ep_val:.4f} (PPL {ep_ppl:.2f}) "
              f"| {ep_dt:.1f} sn | ckpt sha BETİKTEN ({ckpt_sha[os.path.basename(ckpt)][:16]}…)", flush=True)

    best = min(history["history"], key=lambda c: c["val_loss"])
    best_path = os.path.join(DATA_DIR, "anka_b1_5_best.pt")
    shutil.copyfile(best["path"], best_path)
    best_sha = compute_sha256(best_path)

    history["best"] = {"epoch": best["epoch"], "val_loss": best["val_loss"],
                       "path": best_path, "sha256": best_sha}
    history_yaz()

    # EK3 pozitif-etki kapısı (İLAN-6 birebir)
    if best["val_loss"] >= init_loss:
        print(f"T0191_DUR: EK3 kapısı — best_val_loss {best['val_loss']:.4f} >= init {init_loss:.4f}")
        print("T0191_EGITIM_DUR")
        return 2
    print(f"[BEST] epoch {best['epoch']} val {best['val_loss']:.4f} < init {init_loss:.4f} "
          f"-> {best_path} sha={best_sha}", flush=True)
    print("T0191_EGITIM_GECTI")
    return 0


if __name__ == "__main__":
    sys.exit(main())