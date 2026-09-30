#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0200 Faz-2 — optimum-zarar onarım devam-eğitimi (BETİKTEN; MPS).

Üç-bileşen tek-betik (operatör-onaylı BÜTÜNÜ reçete; şartname BÖLÜM-C):
- Bileşen-1 KARIŞIM 1:1: SFT t0197-üretim-saf (ağırlıksız FastSampleAlignedDataset)
  + PencereDataset (pretrain-bin natural 128-pencere, seed-42 grid-örneklem)
  → KarısımDataset (scripts/t0200_mutasyon_kanit import; ds çıpası 10.982)
- Bileşen-2 TEK-EPOCH DÜŞÜK-LR: epochs=1, batch 16, accum 2, warmup 20,
  peak_lr 2e-5, min_lr 5e-6 (T-0198 get_lr cos-kalıbı; 344-adım bütçesi birebir)
- Bileşen-3 KL-ÇIPA: öğretmen=anka_base_v2 (BETİKTEN-çıpalı); forward-KL T=1
  tam-vocab yalnız pencere-konumlarında; β=1,0; L = ağırlıksız-CE + β·KL
R4: öğretmen no_grad + del logits + torch.mps.empty_cache.
Doz-beyanı: veri-½ × LR-¼ = T-0198 reçetesinin ⅛'ü.
Hüküm BETİKTEN: T0200_EGITIM_GECTI (EK1-EK6) | T0200_EGITIM_DUR rc=2."""
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
sys.path.insert(0, KÖK)  # standalone-koşum: scripts.* importı için (koşum-küçük-DUR)

import scripts.train_step_b1_5_rigorous as eski  # val-cache/DEVICE hizası (yalnız-OKU)
from scripts.train_step_b1_5_rigorous import evaluate_val_loss
from scripts.t0200_mutasyon_kanit import (  # Şartname BÖLÜM-B kanıtlı yüzeyler
    ACCUM, BATCH, BETA, CIPI_SHA, DS_N_BEKLENEN, HALEF, HOCA, HUKUM_PATH,
    MIN_LR, PEAK_LR, WARMUP, DEVICE, BIN_PATH, SFT_PATH, T0199_M1_HALEF,
    KarısımDataset, cipe_dogrula, get_lr, kl_pencere_kayip, ortak_kur,
    pencere_ce_olc, yukle_checkpoint,
)

HISTORY_PATH = os.path.join(KÖK, "data/eval/t0200_training_history.json")
EGITIM_HUKUM_PATH = os.path.join(KÖK, "data/eval/t0200_egitim_hukum_2026-09-30.json")
TR_100 = 200
EPOCHS = 1


def main() -> int:
    # ---- EK1/EK2 çıpa-seti (fail-closed) ----
    cipe_dogrula()
    print("[EK1/EK2] 5-sha + sha-triplet + T-0199 hüküm BETİKTEN — PASS", flush=True)
    mut = json.load(open(HUKUM_PATH, encoding="utf-8"))
    if mut.get("hukum") != "T0200_MUTASYON_GECTI" or mut.get("rc") != 0:
        print(f"T0200_DUR: mutasyon hükmü {mut.get('hukum')} rc={mut.get('rc')}", flush=True)
        return 2
    print(f"[EK4] mutasyon BETİKTEN-çıpa K1..K5 PASS (kl_ilk={mut['kl_ilk']})", flush=True)

    _, _, model, hoca, ds = ortak_kur()
    if len(ds) != DS_N_BEKLENEN:
        print(f"T0200_DUR: karışım-ds uzunluk {len(ds)} != {DS_N_BEKLENEN}", flush=True)
        return 2
    print(f"[DS] karışım-ds n={len(ds)} — PASS", flush=True)

    # val (ağırlıksız — MADDE-3 ölçek-dersi; val-cache SADECE-OKU)
    val_ds = torch.load(os.path.join(KÖK, "data/b1_5_splits/anka_fast_ds_val.pt"),
                        weights_only=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    train_loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    print(f"ds: train={len(ds)} | val={len(val_ds)}", flush=True)

    total_opt_steps = 344  # Şartname çıpası: 10.982/16/2 birebir
    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)

    history = {"damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ilan": "data/eval/t0200_egitim_ilan_2026-09-30.md",
               "mutasyon": {"hukum": mut["hukum"], "kl_ilk": mut["kl_ilk"],
                            "kl_son": mut["kl_son"]},
               "recete": {"epochs": EPOCHS, "batch": BATCH, "accum": ACCUM,
                          "peak_lr": PEAK_LR, "min_lr": MIN_LR, "warmup": WARMUP,
                          "beta": BETA, "kaynak": {"sft": SFT_PATH, "bin": BIN_PATH,
                                                    "hoca": HOCA, "load": HALEF},
                          "ds_n": len(ds), "opt_steps_cipa": total_opt_steps, "seed": 42},
               "init": None, "probe": None, "history": []}

    def history_yaz() -> None:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    init_val, init_ppl = evaluate_val_loss(model, val_loader)
    init_ce = pencere_ce_olc(model, DEVICE)
    cipa_ce = json.load(open(T0199_M1_HALEF, encoding="utf-8"))["ce_ort"]
    history["init"] = {"val_loss": init_val, "val_ppl": init_ppl, "pencere_ce": init_ce}
    history_yaz()
    print(f"[INIT] val {init_val:.4f} (PPL {init_ppl:.2f}) | pencere-CE {init_ce:.4f} "
          f"(çıpa {cipa_ce:.4f}: birebir={init_ce == cipa_ce})", flush=True)
    if init_ce != cipa_ce:
        print("T0200_DUR: init pencere-CE ölçüm-kabı çıpası uymaz", flush=True)
        return 2

    global_step = 0
    probe_t0 = time.time()
    kl_adim_seri, kl_adim_top, kl_adim_say = [], 0.0, 0
    model.train()
    for b_idx, (x, y, sm, kz) in enumerate(train_loader):
        x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
        # kz CPU'da KALIR — koşum-3 onarımı: nonzero CPU'da (kl_pencere_kayip)
        logits, loss_std = model(x, y, sm)
        kl = kl_pencere_kayip(logits, hoca, x, y, sm, kz)
        kayip = loss_std + BETA * kl
        del logits  # R4
        (kayip / ACCUM).backward()
        kl_adim_top += float(kl.item())
        kl_adim_say += 1
        if (b_idx + 1) % ACCUM == 0 or (b_idx + 1) == len(train_loader):
            global_step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(global_step, total_opt_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.mps.empty_cache()  # R4: koşum-5 OOM-dersi — iterasyon-başına
            kl_adim_seri.append(kl_adim_top / max(1, kl_adim_say))
            kl_adim_top, kl_adim_say = 0.0, 0
            if global_step == TR_100:
                dt = time.time() - probe_t0
                adim = dt / TR_100
                history["probe"] = {"adim": TR_100, "sn": round(dt, 1),
                                     "sn_adim": round(adim, 4),
                                     "tam_kosum_tamini_min": round((total_opt_steps - TR_100) * adim / 60.0, 1)}
                history_yaz()
                print(f"PROBE: {TR_100} adım sn={dt:.1f} sn/adım={adim:.3f} "
                      f"kalan-tamini≈{history['probe']['tam_kosum_tamini_min']:.1f} dk "
                      f"(beyan; koşum devam)", flush=True)
            if global_step % 50 == 0 or global_step == 1:
                torch.mps.empty_cache()  # R4: 50-adımda-bir cache-boşaltma
                recent_kl = float(np.mean(kl_adim_seri[-20:])) or 0.0
                print(f"  adım {global_step}/{total_opt_steps} | kl_ort(son20) "
                      f"{recent_kl:.4f} | lr {pg['lr']:.7f}", flush=True)

    ep_val, ep_ppl = evaluate_val_loss(model, val_loader)
    ep_ce = pencere_ce_olc(model, DEVICE)
    kl_ort_epok = float(np.mean(kl_adim_seri))
    torch.mps.empty_cache()

    ckpt = os.path.join(KÖK, "data", "anka_t0200_epoch1.pt")
    torch.save(model.state_dict(), ckpt)
    best_path = os.path.join(KÖK, "data", "anka_t0200.pt")
    shutil.copyfile(ckpt, best_path)
    from scripts.train_step_b1_5_rigorous import compute_sha256 as _csha
    ckpt_sha, best_sha = _csha(ckpt), _csha(best_path)
    history["history"].append({"epoch": 1, "val_loss_agirliksiz": ep_val,
                               "val_ppl": ep_ppl, "pencere_ce": ep_ce,
                               "kl_ort": kl_ort_epok, "path": ckpt,
                               "duration": time.time() - probe_t0, "sha256": ckpt_sha})
    history["best"] = {"epoch": 1, "val_loss_agirliksiz": ep_val, "path": best_path,
                       "sha256": best_sha}
    history_yaz()
    print(f"[BEST] epoch 1 val {ep_val:.4f} | pencere-CE {ep_ce:.4f} | kl_ort {kl_ort_epok:.4f} "
          f"-> {best_path} sha={best_sha[:16]}…", flush=True)

    # ---- HÜKÜM: EK1..EK6 BETİKTEN ----
    kapilar = {"EK1_cipe": True, "EK2_sha_triplet": True,
               "EK3_pozitif_etki": ep_val < init_val,
               "EK4_mutasyon": mut.get("hukum") == "T0200_MUTASYON_GECTI",
               "EK5_pencere_akis": ep_ce < init_ce and ep_ce < cipa_ce,
               "EK6_kl_faaliyet": math.isfinite(kl_ort_epok) and kl_ort_epok > 0}
    gecti = all(kapilar.values())
    hukum = "T0200_EGITIM_GECTI" if gecti else "T0200_EGITIM_DUR"
    donem = {"task": "T-0200", "hukum": hukum, "rc": 0 if gecti else 2,
             "kapilar": kapilar,
             "init": {"val_loss_agirliksiz": init_val, "pencere_ce": init_ce},
             "best": {"path": best_path, "sha256": best_sha,
                      "val_loss_agirliksiz": ep_val, "pencere_ce": ep_ce,
                      "kl_ort_epok": kl_ort_epok},
             "kl_adim_seri_ilk_12": kl_adim_seri[:12],
             "history_path": HISTORY_PATH,
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(EGITIM_HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    for k, v in kapilar.items():
        print(f"  ESIK {k}: {v}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())