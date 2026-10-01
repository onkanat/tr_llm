#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0201 — β-gerileri eğrisi (BETİMTEN; MPS).

İki-mod:
- `--kosum 0.5|2.0`: T-0200 reçetesi birebir (karışım 1:1 + tek-epoch düşükLR + KL-çıpa,
  öğretmen base_v2), farklar yalnız β + çıktı-adları. Koşum-kapılar EK-seti DEĞİL —
  yalnız ölçüm-kabı bütünlüğü: init pencere-CE==3,9268 · init val==T-0200-init ·
  KL finiter>0. EK3/EK5 EĞRİ-NOKTASI olarak ölçülür (hüküm-kapısı değil — şartname BÖLÜM-D).
- `--egri-analiz`: 3-nokta eğri (β=0,5 koşumu · β=1,0 T-0200'den devral · β=2,0 koşumu)
  → t0201_geri_egri.json (noktalar + duyarlılık + β*-aralığı BEYAN kayıt) +
  t0201_egitim_hukum_2026-09-30.json (T0201_BETA_EGRI_KAYDI/DUR BETİMTEN).

R4 (devralınır): öğretmen no_grad + del logits + torch.mps.empty_cache per-iterasyon.
Kanıt-tabanı BETİMTEN (t0200_egitim_hukum / t0200_mutasyon_hukum / t0199_m1_b1_5_best)."""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)  # standalone-koşum: scripts.* import'ı

import scripts.train_step_b1_5_rigorous as eski  # val-cache/DEVICE hizası (yalnız-OKU)
from scripts.train_step_b1_5_rigorous import evaluate_val_loss, compute_sha256
from scripts.t0200_mutasyon_kanit import (  # Şartname kanıt-yüzeyleri BİRSATAN
    ACCUM, BATCH, CIPI_SHA, DS_N_BEKLENEN, HALEF, HOCA, MIN_LR, PEAK_LR, WARMUP,
    DEVICE, BIN_PATH, SFT_PATH, T0199_M1_HALEF, cipe_dogrula, get_lr, kl_pencere_kayip,
    ortak_kur, pencere_ce_olc,
)

T0200_HUKUM_PATH = os.path.join(KÖK, "data/eval/t0200_egitim_hukum_2026-09-30.json")
EGRI_PATH = os.path.join(KÖK, "data/eval/t0201_geri_egri.json")
EGITIM_HUKUM_PATH = os.path.join(KÖK, "data/eval/t0201_egitim_hukum_2026-09-30.json")
TR_100 = 200
EPOCHS = 1
BETALAR = (0.5, 2.0)


def kosum(beta: float) -> int:
    cipe_dogrula()
    print("[EK1/EK2] 5-sha + sha-triplet + T-0199 hüküm BETİMTEN — PASS", flush=True)
    mut = json.load(open(os.path.join(KÖK, "data/eval/t0200_mutasyon_hukum_2026-09-30.json"),
                         encoding="utf-8"))
    if mut.get("hukum") != "T0200_MUTASYON_GECTI" or mut.get("rc") != 0:
        print(f"T0201_DUR: mutasyon hükmü {mut.get('hukum')} rc={mut.get('rc')}", flush=True)
        return 2
    t0200 = json.load(open(T0200_HUKUM_PATH, encoding="utf-8"))
    if t0200.get("hukum") != "T0200_EGITIM_DUR":
        print(f"T0201_DUR: T-0200 hükmü {t0200.get('hukum')} — eğri-devir-zamanı değil", flush=True)
        return 2
    cipa_ce = json.load(open(T0199_M1_HALEF, encoding="utf-8"))["ce_ort"]
    init_cipa_val = t0200["init"]["val_loss_agirliksiz"]
    print(f"[ÇIPA] pencere-CE çıpası {cipa_ce} · init-val çıpası {init_cipa_val} "
          f"(T-0200 hüküm-JSON'da BETİMTEN)", flush=True)

    _, _, model, hoca, ds = ortak_kur()
    if len(ds) != DS_N_BEKLENEN:
        print(f"T0201_DUR: karışım-ds uzunluk {len(ds)} != {DS_N_BEKLENEN}", flush=True)
        return 2
    val_ds = torch.load(os.path.join(KÖK, "data/b1_5_splits/anka_fast_ds_val.pt"),
                        weights_only=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    train_loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    print(f"ds: train={len(ds)} | val={len(val_ds)} | beta={beta}", flush=True)

    total_opt_steps = 344  # Şartname çıpası: 10.982/16/2 birebir
    ek_etiket = "b05" if beta == 0.5 else "b20"
    history_path = os.path.join(KÖK, f"data/eval/t0201_{ek_etiket}_history.json")
    ckpt = os.path.join(KÖK, "data", f"anka_t0201_{ek_etiket}_.pt_temp")
    best_path = os.path.join(KÖK, "data", f"anka_t0201_{ek_etiket}.pt")

    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)
    history = {"damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ilan": "data/eval/t0201_egitim_ilan_2026-09-30.md",
               "beta": beta,
               "recete": {"epochs": EPOCHS, "batch": BATCH, "accum": ACCUM,
                          "peak_lr": PEAK_LR, "min_lr": MIN_LR, "warmup": WARMUP,
                          "kaynak": {"sft": SFT_PATH, "bin": BIN_PATH, "hoca": HOCA,
                                     "load": HALEF},
                          "ds_n": len(ds), "opt_steps_cipa": total_opt_steps, "seed": 42},
               "init": None, "probe": None, "history": [], "kosum_hukum": None}

    init_val, init_ppl = evaluate_val_loss(model, val_loader)
    init_ce = pencere_ce_olc(model, DEVICE)
    history["init"] = {"val_loss": init_val, "val_ppl": init_ppl, "pencere_ce": init_ce}
    _history_yaz(history, history_path)
    print(f"[INIT] val {init_val:.4f} (PPL {init_ppl:.2f}) | pencere-CE {init_ce:.4f}"
          f" (çıpa {cipa_ce}: birebir={init_ce == cipa_ce})"
          f" | val-çıpa birebir={init_val == init_cipa_val}", flush=True)
    if init_ce != cipa_ce or init_val != init_cipa_val:
        print("T0201_DUR: init ölçüm-kabı çıpası uymaz (pencere-CE veya val)", flush=True)
        return 2

    global_step = 0
    probe_t0 = time.time()
    kl_adim_seri, kl_top, kl_say = [], 0.0, 0
    model.train()
    for b_idx, (x, y, sm, kz) in enumerate(train_loader):
        x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
        # kz CPU'da KALIR — T-0200 koşum-3 SIGSEGV-dersi
        logits, loss_std = model(x, y, sm)
        kl = kl_pencere_kayip(logits, hoca, x, y, sm, kz)
        kayip = loss_std + beta * kl
        del logits
        (kayip / ACCUM).backward()
        kl_top += float(kl.item())
        kl_say += 1
        if (b_idx + 1) % ACCUM == 0 or (b_idx + 1) == len(train_loader):
            global_step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(global_step, total_opt_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.mps.empty_cache()  # R4: T-0200 koşum-5 OOM-dersi
            kl_adim_seri.append(kl_top / max(1, kl_say))
            kl_top, kl_say = 0.0, 0
            if global_step == TR_100:
                dt = time.time() - probe_t0
                adim = dt / TR_100
                history["probe"] = {"adim": TR_100, "sn": round(dt, 1),
                                    "sn_adim": round(adim, 4),
                                    "tam_kosum_tamini_min":
                                        round((total_opt_steps - TR_100) * adim / 60.0, 1)}
                _history_yaz(history, history_path)
                print(f"PROBE: {TR_100} adım sn={dt:.1f} sn/adım={adim:.3f} "
                      f"kalan-tamini≈{history['probe']['tam_kosum_tamini_min']:.1f} dk "
                      f"(beyan; koşum devam)", flush=True)
            if global_step % 50 == 0 or global_step == 1:
                torch.mps.empty_cache()
                recent_kl = float(np.mean(kl_adim_seri[-20:])) or 0.0
                print(f"  adım {global_step}/{total_opt_steps} | kl_ort(son20) "
                      f"{recent_kl:.4f} | lr {pg['lr']:.7f}", flush=True)

    ep_val, ep_ppl = evaluate_val_loss(model, val_loader)
    ep_ce = pencere_ce_olc(model, DEVICE)
    kl_ort_epok = float(np.mean(kl_adim_seri))
    torch.mps.empty_cache()

    torch.save(model.state_dict(), ckpt)
    os.replace(ckpt, best_path)  # atomic-rename; yarım-yazım diske best-için düşmez
    best_sha = compute_sha256(best_path)
    history["history"].append({"epoch": 1, "val_loss_agirliksiz": ep_val,
                               "val_ppl": ep_ppl, "pencere_ce": ep_ce,
                               "kl_ort": kl_ort_epok, "path": best_path,
                               "duration": time.time() - probe_t0, "sha256": best_sha})
    history["best"] = {"epoch": 1, "val_loss_agirliksiz": ep_val, "path": best_path,
                       "sha256": best_sha, "pencere_ce": ep_ce, "kl_ort": kl_ort_epok}
    print(f"[BEST] beta={beta} val {ep_val:.4f} | pencere-CE {ep_ce:.4f} | kl_ort "
          f"{kl_ort_epok:.4f} -> {best_path} sha={best_sha[:16]}…", flush=True)

    # ---- koşum-kapılar (ölçüm-kabı bütünlüğü; EK3/EK5 KAPI DEĞİL) ----
    kl_fin = bool(kl_adim_seri) and math.isfinite(kl_ort_epok) and kl_ort_epok > 0
    kapilar = {"init_cek_birebir": init_ce == cipa_ce and init_val == init_cipa_val,
               "opt_adim_344": total_opt_steps == 344,
               "kl_finit_pozitif": kl_fin,
               "ds_10982": len(ds) == DS_N_BEKLENEN}
    hukum = "T0201_KOSUM_GECTI" if all(kapilar.values()) else "T0201_KOSUM_DUR"
    history["kosum_hukum"] = {"hukum": hukum, "rc": 0 if hukum == "T0201_KOSUM_GECTI" else 2,
                              "kapilar": kapilar,
                              "egri_nokta": {"beta": beta,
                                             "d_val": ep_val - init_val,
                                             "d_pencere_ce": ep_ce - init_ce,
                                             "kl_ort": kl_ort_epok, "sha256": best_sha},
                              "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _history_yaz(history, history_path)
    for k, v in kapilar.items():
        print(f"  KAPI {k}: {v}", flush=True)
    print(f"KOSUM-HUKUM(beta={beta}): {hukum}", flush=True)
    if hukum != "T0201_KOSUM_GECTI":
        print("T0201_DUR: eğri-analiz yine de ayrı çalışmalı", flush=True)
        return 2
    return 0


def egri_analiz() -> int:
    # T-0200 devralınan merkez-noktası BETİMTEN
    t0200 = json.load(open(T0200_HUKUM_PATH, encoding="utf-8"))
    if t0200.get("hukum") != "T0200_EGITIM_DUR":
        print(f"T0201_DUR: T-0200 hükmü {t0200.get('hukum')}", flush=True)
        return 2
    noktalar = [{"beta": 1.0,
                 "val": t0200["best"]["val_loss_agirliksiz"],
                 "pencere_ce": t0200["best"]["pencere_ce"],
                 "kl_ort": t0200["best"]["kl_ort_epok"],
                 "kaynak": "T-0200-devral", "sha256": t0200["best"]["sha256"]}]
    init_val = t0200["init"]["val_loss_agirliksiz"]
    init_ce = t0200["init"]["pencere_ce"]

    uyarilar = []
    for ek, beta in (("b05", 0.5), ("b20", 2.0)):
        path = os.path.join(KÖK, f"data/eval/t0201_{ek}_history.json")
        hist = json.load(open(path, encoding="utf-8"))
        kh = hist.get("kosum_hukum")
        if not kh or kh["hukum"] != "T0201_KOSUM_GECTI":
            uyarilar.append(f"{ek}: koşum-hüküm {kh and kh['hukum']}")
            continue
        noktalar.append({"beta": beta,
                         "val": hist["best"]["val_loss_agirliksiz"],
                         "pencere_ce": hist["best"]["pencere_ce"],
                         "kl_ort": hist["best"]["kl_ort"],
                         "kaynak": f"T-0201-{ek}", "sha256": hist["best"]["sha256"]})
        # çift-bağımsızlık: her koşum kendi init'inden Δ alır (çıpa-birebir kuralı yüzünden eşittir)
        init_val = hist["init"]["val_loss"]
        init_ce = hist["init"]["pencere_ce"]

    for n in noktalar:
        n["d_val"] = n["val"] - init_val
        n["d_pencere_ce"] = n["pencere_ce"] - init_ce
        n["ek3_ok"] = n["d_val"] < 0
        n["ek5_ok"] = n["d_pencere_ce"] < 0
    noktalar.sort(key=lambda n: n["beta"] if isinstance(n.get("beta"), float) else -9)

    # duyarlılık BETİMTEN (birinci-fark; noktalar β-sıralı)
    duyar = {"d_dval": [], "d_dpencere": []}
    s_noktalar = [n for n in noktalar]
    for i in range(1, len(s_noktalar)):
        dbeta = s_noktalar[i]["beta"] - s_noktalar[i - 1]["beta"]
        if dbeta <= 0:
            uyarilar.append(f"duyarlılık: nokta-β sırası bozuk {i}")
            break
        duyar["d_dval"].append(round((s_noktalar[i]["d_val"] - s_noktalar[i - 1]["d_val"]) / dbeta, 6))
        duyar["d_dpencere"].append(round((s_noktalar[i]["d_pencere_ce"]
                                          - s_noktalar[i - 1]["d_pencere_ce"]) / dbeta, 6))

    # β*-aralığı BEYAN (KARAR DEĞİL): EK3+EK5 ikisi-de-ok noktaları / komşu-interpolasyon
    ikisi_ok = [n for n in noktalar if n["ek3_ok"] and n["ek5_ok"]]
    interpolasyon_beyan = "ADET-YOK"
    if ikisi_ok:
        betalar_ok = sorted(n["beta"] for n in ikisi_ok)
        interpolasyon_beyan = f"koşum-noktalarında-çift-ok-betaları={betalar_ok} " \
                              f"→ β*-adayı BEYAN-KAYDI (uygulama yeni-görevde)"
    egri = {"damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "init_cipa": {"val_loss": init_val, "pencere_ce": init_ce},
            "noktalar": noktalar, "duyarliilik_birinci_fark": duyar,
            "ikili_beyan": interpolasyon_beyan, "uyarilar": uyarilar}
    with open(EGRI_PATH, "w", encoding="utf-8") as f:
        json.dump(egri, f, ensure_ascii=False, indent=2)

    ok = len(noktalar) == 3 and not uyarilar
    hukum = "T0201_BETA_EGRI_KAYDI" if ok else "T0201_BETA_EGRI_DUR"
    donem = {"task": "T-0201", "hukum": hukum, "rc": 0 if ok else 2,
             "noktalar": noktalar, "duyarliilik": duyar, "ikili_beyan": interpolasyon_beyan,
             "t0200_sha": t0200["best"]["sha256"], "egri_path": EGRI_PATH,
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(EGITIM_HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    for n in noktalar:
        print(f"  EGRI β={n['beta']} Δval={n['d_val']:.4f} Δpce={n['d_pencere_ce']:.4f} "
              f"kl_ort={n['kl_ort']:.4f} (EK3={n['ek3_ok']} EK5={n['ek5_ok']})", flush=True)
    print(f"IKILI-BEYAN: {interpolasyon_beyan}", flush=True)
    if uyarilar:
        print(f"UYARI: {uyarilar}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if ok else 2


def _history_yaz(history: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = sys.argv[1:]
    if "--egri-analiz" in args:
        return egri_analiz()
    if len(args) >= 2 and args[0] == "--kosum":
        beta = float(args[1])
        if beta not in BETALAR:
            print(f"T0201_DUR: β={beta} şartname-seti dışında {BETALAR}", flush=True)
            return 2
        return kosum(beta)
    print("kullanim: --kosum 0.5|2.0 · --egri-analiz", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())