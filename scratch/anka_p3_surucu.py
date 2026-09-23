#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 / Aşama 1+2 — ÜÇ KAYNAKLI YETENEK KOŞUMU: wiki %20 + ceket %40 + SFT %40.

İLAN: data/eval/anka_p3_ilani_2026-09-23.md (koşumdan ÖNCE yazıldı).
Plan: `~/.claude/plans/enchanted-wiggling-moon.md` (P3 — onaylı).
TEŞHİS: P2'nin "ceket eğitimde OLMAZ" kuralı, T-0094/T-0097 carve politikasının
held-out 539 = eval / 4.861 = yasal eğitim beyanından KATIYDI (V7: cevap ekseni 0/0,
kayıt kesişimi 0) ⇒ tek yetenek-uyumlu külliyat eğitimden gitmişti. Operatör kararı
(23 Eyl 2026): V7 carve ile geri dön — ceket r18 büyütmesi
(`data/train_carpenter_specialization_anka_r18.bin`, 1.568.123 jeton / 22.283 kayıt,
held-out carve AYNI).

KANONİK KOD İMPORT EDİLİR (kopya YASAK): pencere_oku/yaz_json/ekle_jsonl/
kapi_karari/wikipedia_ce/durdur ← scratch.anka_p2_temel2_surucu;
MASKE/get_lr/maske_pad_hedefleri/optimizer_sidecar_path/sha256_file ← train.py;
a_ekseni ECA'da sürücü wikipedia_ce içinde zaten import'lu; mask_prompt_targets ←
scripts.train_step_demo.

KAPILAR (fail-closed):
  H1    tek_egitici_onkontrol — iki eğitici MPS'te kilitlenir (T-0097/K5)
  L1    ilk adım kaybı 0,0 VEYA ≈ ln V ⇒ DUR (devam koşumunda ln V BOZUKLUK imzası)
  CARRY taban sidecar digest uyuşmazsa DUR (T-0092)
  CE    her segment sonunda Wikipedia CE <= CE_TAVAN (kanonik; assert'li)
  TEPE  mini yetenek sondası (kanonik kap, n=100) — ROUGE önceki segmentten > 0,05
        DÜŞERSE (≈ >2 SE) koşum DURUR, tepe segment nihaidir (T-0097 24K gerileme
        dersi). ezber_orani >= 10 ise de DURUR (alan ZATEN YÜZDEDİR, 5,0 = %5;
        ECA:513 — ilk koşumda çift çarpım yalancı TEPE üretti; 23 Eyl onarıldı).
HELD-OUT ÇIPASI: held-out 539 digest'i başta/sonda hesaplanır, koşum boyunca
birebir kalmak zorunda (T-0096 deseni; V7 carve sızıntı kanıtı).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.join(KOK, "scratch"))

import anka_p2_temel2_surucu as P2  # noqa: E402  (yeniden kullanılabilir parçalar)
from scripts.kosum_kapisi import tek_egitici_onkontrol  # noqa: E402
from scripts.train_step_demo import KristalLM, mask_prompt_targets  # noqa: E402
from src.llm.frozen_guard import is_frozen_path  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402
from src.llm.tokenizer import Vocabulary  # noqa: E402
from train import (MASKE, get_lr, maske_pad_hedefleri,  # noqa: E402
                   optimizer_sidecar_path, sha256_file)

# --- İLAN EDİLMİŞ TARİF (anka_p3_ilani_2026-09-23.md §tarif) ---
TABAN = "scratch/anka_p2_ince_ayr_kos/seg_2.pt"      # Aşama 3.1 nihai (carry VAR)
SOZLUK = "data/rebuild/vocab_anka_r1_33114.json"
WIKI_BIN = "data/anka_a1r_pretrain.bin"
SFT_BIN = "data/rebuild/anka_p2_sft_mix.bin"
CEKET_BIN = "data/train_carpenter_specialization_anka_r18.bin"
HELDOUT = "data/eval/anka_r17_heldout_2026-09-20.jsonl"
KOS = "scratch/anka_p3_kos"
SEGMENTLER: List[int] = [2000, 2000, 2000]           # 3 × 2.000 = 6.000 adım
BLOK, BATCH, LR = 128, 8, 1e-4
WARMUP, TOPLAM_ADIM, MIN_LR = 50, 6000, 1e-6
CLIP = 1.0
PATERN = 5   # pencere idx % 5: 0 ⇒ wiki (%20) · 1,2 ⇒ ceket (%40) · 3,4 ⇒ SFT (%40)
TEPE_ESIK = 0.05   # ROUGE-L düşüşü > 0,05 (≈ >2 SE; SE ≈ std/√100) ⇒ tepe, DUR

assert abs(P2.CE_TAVAN - 3.5351739511825144 * (1 + 10.0 / 100.0)) < 1e-3, \
    "CE_TAVAN kanonik sabitle uyuşmuyor — P2 modülünden devral"


def sonda_yap(ckpt: str, taban: str, cihaz: str, sonda_n: int) -> Dict[str, Any]:
    """Mini yetenek sondası — KANONİK kap `scripts/evaluate_carpenter_anka.py`
    alt süreçte (import ederek koşmak model yükleyicisini iki kez kurar; süreç
    izolasyonu MPS belleğini de boşaltır). baseline = KOŞUM BAŞLANGICI (taban):
    unutma A/B koşumun birikimli LM bedelini ölçer."""
    cikti = f"{KOS}/sonda_{os.path.basename(ckpt).replace('.pt', '')}.json"
    cmd = [sys.executable, "scripts/evaluate_carpenter_anka.py",
           "--model", ckpt, "--baseline", taban,
           "--n", str(sonda_n), "--seed", "42",
           "--output", cikti, "--device", cihaz, "--ceket-ekseni"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=KOK)
    if r.returncode != 0:
        P2.durdur(f"SONDA arka planı başarısız (rc={r.returncode}): "
                  f"{r.stderr[-600:]}")
    with open(cikti, encoding="utf-8") as f:
        return json.load(f)


def tepe_karari(onceki: Dict[str, Any] | None, sonda: Dict[str, Any]) -> Tuple[bool, str]:
    """TEPE kapısı: ROUGE >0,05 düşüş VEYA ezber ≥ %10 ⇒ DUR (tepe önceki segment).
    ⚠ ezber_orani JSON'da ZATEN YÜZDEDİR (5,0 = %5) — çift çarpım YAPILMAZ."""
    ck = sonda["ceket_ekseni"]
    ezber = float(ck["ezber_orani"])
    if ezber >= 10.0:
        return True, f"ezber {ezber:.2f} ≥ %10 — ezber kapısı; tepe önceki segment"
    if onceki is None:
        return False, "ilk sonda — referans yok"
    d = float(ck["rouge_l_ort"]) - float(onceki["ceket_ekseni"]["rouge_l_ort"])
    if d < -TEPE_ESIK:
        return True, (f"ROUGE-L {d:+.4f} (< -{TEPE_ESIK}) — tepe yapıp geriliyor "
                      f"(T-0097 dersi); tepe önceki segment")
    return False, f"ROUGE-L {d:+.4f} (eşik -{TEPE_ESIK}) · ezber {ezber:.2f}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="P3 üç kaynaklı yetenek koşumu")
    ap.add_argument("--kapi-sinamasi", action="store_true",
                    help="TEPE + CE kapılarının iki dalını sına ve çık")
    ap.add_argument("--taban", default=TABAN)
    ap.add_argument("--kos-adi", default=None)
    ap.add_argument("--segmentler", default=None, help="virgüllü, ör. '2000,2000,2000'")
    ap.add_argument("--cihaz", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--sonda-n", type=int, default=100)
    ap.add_argument("--sonda-yok", action="store_true", help="smoke: sonda atla")
    ap.add_argument("--referans-sonda", default=None,
                    help="TEPE kapısının ilk segment referansı (önceki koşumun "
                         "son sonda JSON'u); verilmezse ilk sonda referanssız")
    ap.add_argument("--ilan", default="data/eval/anka_p3_ilani_2026-09-23.md")
    a = ap.parse_args()

    global SEGMENTLER, TOPLAM_ADIM, WARMUP, KOS
    if a.kapi_sinamasi:
        # TEPE: yukarı ok +0,01 ⇒ DUR yok · aşağı ok -0,06 ⇒ DUR var ·
        # ezber_orani YÜZDE ölçeğinde: 9,9 ⇒ geç · 10,0 ⇒ DUR (ölçek kanıtı ECA:513)
        r1, g1 = tepe_karari({"ceket_ekseni": {"rouge_l_ort": 0.20, "ezber_orani": 0.0}},
                             {"ceket_ekseni": {"rouge_l_ort": 0.21, "ezber_orani": 0.0}})
        r2, g2 = tepe_karari({"ceket_ekseni": {"rouge_l_ort": 0.20, "ezber_orani": 0.0}},
                             {"ceket_ekseni": {"rouge_l_ort": 0.14, "ezber_orani": 0.0}})
        r3, g3 = tepe_karari(None, {"ceket_ekseni": {"rouge_l_ort": 0.0, "ezber_orani": 10.0}})
        r3b, g3b = tepe_karari(None, {"ceket_ekseni": {"rouge_l_ort": 0.0, "ezber_orani": 9.9}})
        ok = (not r1) and r2 and r3 and (not r3b)
        print(f"[kapı sınaması] +0,01 ⇒ DUR={r1} · -0,06 ⇒ DUR={r2} · "
              f"ezber 10,0 ⇒ DUR={r3} · ezber 9,9 ⇒ DUR={r3b}")
        return 0 if ok else 2
    if a.segmentler:
        SEGMENTLER = [int(s) for s in a.segmentler.split(",") if s.strip()]
        TOPLAM_ADIM = sum(SEGMENTLER)
    if a.kos_adi:
        KOS = f"scratch/{a.kos_adi}"
    taban = a.taban
    cihaz = a.cihaz
    onceki_sonda: Dict[str, Any] | None = None
    if a.referans_sonda:
        if not os.path.exists(a.referans_sonda):
            P2.durdur(f"referans sonda yok: '{a.referans_sonda}'")
        with open(a.referans_sonda, encoding="utf-8") as f:
            onceki_sonda = json.load(f)
        rk = onceki_sonda["ceket_ekseni"]
        print(f"[referans] TEPE referansı {a.referans_sonda}: "
              f"ROUGE {rk.get('rouge_l_ort')} · ezber {rk.get('ezber_orani')}", flush=True)

    for yol in (taban, SOZLUK, WIKI_BIN, SFT_BIN, CEKET_BIN, HELDOUT):
        if not os.path.exists(yol):
            P2.durdur(f"girdi yok: '{yol}'")
    if is_frozen_path(f"{KOS}/seg_1.pt"):
        P2.durdur("çıktı yolu donmuş desene düşüyor — scratch/ altında kalmalı")

    heldout_sha_bas = sha256_file(HELDOUT)          # HELD-OUT ÇIPASI (V7/T-0096)

    h1 = tek_egitici_onkontrol(WIKI_BIN)
    print(f"[H1] tek eğitici: {h1['tutucu']} tutucu {h1['pidler']}", flush=True)

    os.makedirs(KOS, exist_ok=True)
    vocab = Vocabulary()
    vocab.load(SOZLUK)
    device = torch.device(cihaz)
    if cihaz == "mps" and not torch.backends.mps.is_available():
        P2.durdur("MPS yok — koşum sandbox DIŞINDA koşulmalı")

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    d = torch.load(taban, map_location="cpu")
    for k in [k for k in list(d.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del d[k]
    model.load_state_dict(resize_state_dict(model, d), strict=False)
    model.to(device)
    taban_sha = sha256_file(taban)
    print(f"[model] taban {taban} · sha256 {taban_sha[:16]}…", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    opt_yolu = optimizer_sidecar_path(taban)
    if os.path.exists(opt_yolu):
        payload = torch.load(opt_yolu, map_location="cpu")
        if payload.get("model_sha256") != taban_sha:
            P2.durdur(f"CARRY KAPISI: {opt_yolu} digest'i tabanla ESLESMIYOR")
        optimizer.load_state_dict(payload["optimizer"])
        carry_beyan = (f"momentler yuklendi ({len(optimizer.state)} parametre, "
                       f"adim={payload.get('adim')}; T-0092)")
        print(f"[carry] optimizer {carry_beyan}", flush=True)
    else:
        carry_beyan = "sidecar YOK ⇒ sıfırdan (açık uyarı, T-0092/G4a)"
        print(f"[carry] {opt_yolu} YOK — optimizer SIFIRDAN (T-0092/G4a)", flush=True)

    wiki_mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    sft_mm = np.memmap(SFT_BIN, dtype=np.uint16, mode="r")
    ceket_mm = np.memmap(CEKET_BIN, dtype=np.uint16, mode="r")
    wiki_blok = len(wiki_mm) // BLOK
    sft_blok = len(sft_mm) // BLOK
    ceket_blok = len(ceket_mm) // BLOK
    print(f"[veri] wiki {wiki_blok:,} · ceket {ceket_blok:,} ({len(ceket_mm):,} jeton) · "
          f"sft {sft_blok:,} pencere", flush=True)

    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    pad_id = vocab.stoi.get("<PAD>", 1)
    rng = np.random.default_rng(42)

    sonuc: Dict[str, Any] = {
        "gorev": "anka_p3 · üç kaynaklı yetenek koşumu (wiki %20/ceket %40/SFT %40)",
        "ilan": a.ilan,
        "baslangic_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "taban": taban, "taban_sha256": taban_sha,
        "wiki_bin": WIKI_BIN, "sft_bin": SFT_BIN,
        "ceket_bin": CEKET_BIN, "ceket_bin_sha256": sha256_file(CEKET_BIN),
        "ceket_beyan": ("V7 carve: held-out 539 hariç, 4.861+r18 büyütme yasal eğitim "
                        "verisi (operatör kararı 23 Eyl; r17_build V7 cevap ekseni 0/0)"),
        "heldout": HELDOUT, "heldout_sha256_bas": heldout_sha_bas,
        "segmentler": SEGMENTLER, "blok": BLOK, "batch": BATCH, "lr": LR,
        "scheduler": {"warmup": WARMUP, "toplam_adim": TOPLAM_ADIM, "min_lr": MIN_LR},
        "clip": CLIP,
        "patern": "pencere idx % 5: 0⇒wiki · 1,2⇒ceket · 3,4⇒sft",
        "tepe_esik": TEPE_ESIK,
        "optimizer_carry": carry_beyan,
        "ce_tavan": P2.CE_TAVAN,
        "maske": "PAD→−100 + ignore_index=−100 (P2/Aşama 1)",
        "dallar": [], "durum": "SURUYOR",
    }
    P2.yaz_json(f"{KOS}/sonuc.json", sonuc)

    seg_jsonl = f"{KOS}/segments.jsonl"
    sonuc_yolu = f"{KOS}/sonuc.json"
    sentinel = f"{KOS}/SONUC_BITTI"
    t0 = time.time()
    global_adim = 0
    pencere_no = 0
    bos_hedef_atlanan = 0
    kaynak_sayac = {"wiki": 0, "ceket": 0, "sft": 0}
    tepe_ckpt: str | None = None

    for seg_no, seg_uzunluk in enumerate(SEGMENTLER, 1):
        ckpt = f"{KOS}/seg_{seg_no}.pt"
        seg_t0 = time.time()
        kayip_gecmis: List[float] = []
        for _ in range(seg_uzunluk):
            xs, hedefler = [], []
            deneme = 0
            while len(xs) < BATCH:
                m = pencere_no % PATERN
                if m == 0:
                    x_cpu, y_cpu = P2.pencere_oku(wiki_mm, int(rng.integers(0, wiki_blok)))
                    h_np = y_cpu.numpy().copy()
                    h_np = maske_pad_hedefleri(h_np, pad_id, True)
                    kaynak_sayac["wiki"] += 1
                elif m in (1, 2):
                    x_cpu, y_cpu = P2.pencere_oku(ceket_mm, int(rng.integers(0, ceket_blok)))
                    h2 = mask_prompt_targets(x_cpu.unsqueeze(0), y_cpu.unsqueeze(0),
                                             output_start_id, eos_id)
                    h_np = np.asarray(h2)[0]
                    h_np = maske_pad_hedefleri(h_np, pad_id, True)
                    kaynak_sayac["ceket"] += 1
                else:
                    x_cpu, y_cpu = P2.pencere_oku(sft_mm, int(rng.integers(0, sft_blok)))
                    h2 = mask_prompt_targets(x_cpu.unsqueeze(0), y_cpu.unsqueeze(0),
                                             output_start_id, eos_id)
                    h_np = np.asarray(h2)[0]
                    h_np = maske_pad_hedefleri(h_np, pad_id, True)
                    kaynak_sayac["sft"] += 1
                pencere_no += 1
                deneme += 1
                if int((h_np != MASKE).sum()) == 0:
                    bos_hedef_atlanan += 1
                    if deneme >= 4 * BATCH:
                        break
                    continue
                xs.append(x_cpu)
                hedefler.append(torch.from_numpy(h_np))
            if not xs:
                continue
            x_cpu_bat = torch.stack(xs)
            x = x_cpu_bat.to(device)
            hedef = torch.stack(hedefler).to(device)
            sign_mask = model.embedding.compute_sign_mask(x_cpu_bat).to(device)

            optimizer.zero_grad()
            _, loss = model(x, targets=hedef, sign_mask=sign_mask, ignore_index=MASKE)
            kayip = float(loss.item())
            if not math.isfinite(kayip):
                P2.durdur(f"L1: adım {global_adim + 1} kayıp SONLU DEĞİL ({kayip})")
            if global_adim == 0:
                ln_v = math.log(len(vocab.stoi))
                if kayip == 0.0:
                    P2.durdur("L1: ilk kayıp 0,0 — MPS sessiz no-op (T-0073 sınıfı)")
                if kayip > ln_v - 1.0:
                    P2.durdur(f"L1: ilk kayıp {kayip:.4f} ≈ ln V ({ln_v:.4f}) — "
                              f"model SIFIRDAKİNE yakın (devam koşumunda BOZUKLUK imzası)")
            loss.backward()
            global_adim += 1
            cur_lr = get_lr(global_adim, LR, WARMUP, TOPLAM_ADIM, MIN_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            kayip_gecmis.append(kayip)
            if global_adim % 500 == 0:
                son100 = kayip_gecmis[-100:]
                print(f"  adım {global_adim}/{TOPLAM_ADIM} | kayıp "
                      f"{sum(son100) / len(son100):.4f} | LR {cur_lr:.6f} | "
                      f"{time.time() - seg_t0:.0f}s", flush=True)
                if device.type == "mps":
                    torch.mps.empty_cache()

        # segment kaydı + CE kapısı + TEPE sondası
        torch.save(model.state_dict(), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        opt_gecici = f"{ckpt}.opt.pt.tmp"
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": {"warmup_steps": WARMUP, "toplam_adim": TOPLAM_ADIM,
                                  "min_lr": MIN_LR, "peak_lr": LR},
                    "adim": global_adim,
                    "model_sha256": sha256_file(ckpt)}, opt_gecici)
        os.replace(opt_gecici, f"{ckpt}.opt.pt")
        # Yalnız SON segmentin taşıyıcısı korunur (P2 planı Aşama 2 tasarımı);
        # ara taşıyıcılar DAL-1 kaldıracı değildir ve disk tavanını şişirir.
        if seg_no > 1:
            onceki_opt = f"{KOS}/seg_{seg_no - 1}.pt.opt.pt"
            if os.path.exists(onceki_opt):
                os.remove(onceki_opt)

        ce = P2.wikipedia_ce(ckpt, vocab, device)
        dustu, gerekce = P2.kapi_karari(ce["CE_ort"])
        sonda: Dict[str, Any] = {"olculdu": False}
        tepe, tepe_gerekce = False, "sonda atlandı (--sonda-yok)"
        if not a.sonda_yok:
            sonda = sonda_yap(ckpt, taban, cihaz, a.sonda_n)
            tepe, tepe_gerekce = tepe_karari(onceki_sonda, sonda)
        ck = sonda.get("ceket_ekseni", {}) if isinstance(sonda, dict) else {}
        kayit = {
            "segment": seg_no, "adim": seg_uzunluk, "global_adim": global_adim,
            "sure_sn": round(time.time() - seg_t0, 1),
            "ckpt": ckpt, "ckpt_sha256": sha256_file(ckpt),
            "ce_ort": round(ce["CE_ort"], 4), "ce_std": round(ce["CE_std"], 4),
            "ppl": round(float(math.exp(ce["CE_ort"])), 2),
            "kayip_son": round(kayip_gecmis[-1], 4) if kayip_gecmis else None,
            "lr_son": round(cur_lr, 8) if kayip_gecmis else None,
            "bos_hedef_pencere": bos_hedef_atlanan,
            "kaynak_sayac": dict(kaynak_sayac),
            "ce_kapi_dustu": dustu, "ce_kapi_gerekce": gerekce,
            "sonda": {"n": ck.get("n"), "rouge_l_ort": ck.get("rouge_l_ort"),
                      "tutarsizlik_orani": ck.get("tutarsizlik_orani"),
                      "ezber_orani": ck.get("ezber_orani"),
                      "kesisim_orani": ck.get("kesisim_orani"),
                      "kesisim_f1_ort": ck.get("kesisim_f1_ort"),
                      "A_artis_yuzde": sonda.get("hukum", {}).get("A_artis_yuzde")
                      if isinstance(sonda, dict) else None},
            "tepe_dustu": tepe, "tepe_gerekce": tepe_gerekce,
        }
        sonuc["dallar"].append(kayit)
        P2.yaz_json(sonuc_yolu, sonuc)
        P2.ekle_jsonl(seg_jsonl, kayit)
        if isinstance(sonda, dict) and ck:
            onceki_sonda = sonda
            tepe_ckpt = ckpt
        print(f"[seg {seg_no}] CE {ce['CE_ort']:.4f} (ppl {kayit['ppl']}) · "
              f"ROUGE {ck.get('rouge_l_ort')} · ezber {ck.get('ezber_orani')} · "
              f"{kayit['sure_sn']:.0f} sn · CE kapısı {'DÜŞTÜ' if dustu else 'geçti'} · "
              f"TEPE {'DÜŞTÜ' if tepe else 'geçti'}", flush=True)
        if dustu:
            sonuc.update({"durum": "DURDU", "duran_dal": f"seg_{seg_no} CE KAPISI",
                          "aciklama": gerekce})
            P2.yaz_json(sonuc_yolu, sonuc)
            P2.durdur(f"CE KAPISI seg_{seg_no}: {gerekce}")
        if tepe:
            sonuc.update({"durum": "DURDU", "duran_dal": f"seg_{seg_no} TEPE KAPISI",
                          "aciklama": tepe_gerekce, "tepe_ckpt": tepe_ckpt})
            P2.yaz_json(sonuc_yolu, sonuc)
            P2.durdur(f"TEPE KAPISI seg_{seg_no}: {tepe_gerekce}")

    sonuc.update({"durum": "BITTI",
                  "bitis_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "toplam_sn": round(time.time() - t0, 1),
                  "nihai_ckpt": tepe_ckpt or f"{KOS}/seg_{len(SEGMENTLER)}.pt",
                  "nihai_sha256": sha256_file(tepe_ckpt or f"{KOS}/seg_{len(SEGMENTLER)}.pt"),
                  "heldout_sha256_son": sha256_file(HELDOUT),
                  "kaynak_sayac": dict(kaynak_sayac)})
    if sonuc["heldout_sha256_son"] != heldout_sha_bas:
        sonuc["durum"] = "HELDOUT_CIPASI_BOZULDU"
        P2.yaz_json(sonuc_yolu, sonuc)
        P2.durdur("HELD-OUT ÇIPASI: koşum boyunca digest DEĞİŞTİ — sızıntı/yan etki")
    P2.yaz_json(sonuc_yolu, sonuc)
    with open(sentinel, "w", encoding="utf-8") as f:
        f.write(f"BITTI {sonuc['bitis_utc']} nihai={sonuc['nihai_ckpt']} "
                f"sha256={sonuc['nihai_sha256']}\n")
    print(f"\n[BİTTİ] nihai={sonuc['nihai_ckpt']} · {sonuc['toplam_sn']:.0f} sn · "
          f"kaynak dağılımı {kaynak_sayac}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)