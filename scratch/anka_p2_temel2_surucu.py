#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 / Aşama 2b — TEMEL 2.0 KOŞUM SÜRÜCÜSÜ: wiki+talimat KARIŞIMLI koşum (b8).

İLAN: data/eval/anka_p2_temel2_ilani_2026-09-22.md (koşumdan ÖNCE yazıldı).
Plan: `~/.claude/plans/enchanted-wiggling-moon.md` (onaylı).

NEDEN AYRI SÜRÜCÜ + KENDİ DÖNGÜSÜ: `train.py` tek veri kümesi + tek hedef modu
koşar; bu koşum PENCERE başına İKİ mod karıştırır (wiki = ön-eğitim hedefi,
talimat = SFT maskeleme). Global --pretrain modu talimat pencerelerini unmaskeli
öğretir, SFT modu wiki pencerelerinde T-0073 kapısını ateşler — ikisi de yanlış.
Desenler: "i1 segment zinciri" (`scratch/anka_i1_surucu.py` — CE kapısı, artımlı
yazım) + "r5 üç mod" (başlatma disiplini) + B1 scheduler (`get_lr`, train.py'den
import). Kapanmış kayıtlar DÜZENLENMEZ; kanonik kod parçaları İMPORT edilir
(kopya YASAK kuralı): get_lr/maske_pad_hedefleri/optimizer_sidecar_path/sha256_file
← train.py, a_ekseni ← ECA, mask_prompt_targets/KristalLM ← train_step_demo.

KAPILAR (fail-closed; hepsi ölçülmüş kusur sınıfından):
  H1  tek_egitici_onkontrol — iki eğitici MPS'te kilitlendi (T-0097/K5)
  L1  ilk adım kaybı: 0,0 VEYA ≈ ln V (10,4) ⇒ DUR — MPS sessiz no-op + soğuk
      başlangıç (canlılık-imzasi-moda-bagli: devam koşumunda ln V imzası BOZUKLUKTUR)
  CARRY  taban sidecar'ı VARSA digest uyuşmazsa DUR (G5 sınıfı, T-0092); YOKSA
      AÇIK uyarıyla sıfırdan optimizer (G4a deseni — anka_a1r moment taşımaz)
  CE  her segment sonunda Wikipedia CE <= CE_TAVAN (ECA.a_ekseni, kanonik)
YAZIM: segments.jsonl append · sonuc.json atomik · SONUC_BITTI sentinel yalnız rc==0.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

import scripts.evaluate_carpenter_anka as ECA  # noqa: E402  (kanonik kap)
from scripts.kosum_kapisi import tek_egitici_onkontrol  # noqa: E402
from scripts.train_step_demo import KristalLM, mask_prompt_targets  # noqa: E402
from src.llm.frozen_guard import is_frozen_path  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402
from src.llm.tokenizer import Vocabulary  # noqa: E402
from train import (MASKE, get_lr, maske_pad_hedefleri,  # noqa: E402
                   optimizer_sidecar_path, sha256_file)

# --- İLAN EDİLMİŞ TARİF (anka_p2_temel2_ilani_2026-09-22.md §tarif) ---
TABAN = "data/anka_a1r.pt"
SOZLUK = "data/rebuild/vocab_anka_r1_33114.json"
WIKI_BIN = "data/anka_a1r_pretrain.bin"
SFT_BIN = "data/rebuild/anka_p2_sft_mix.bin"
KOS = "scratch/anka_p2_temel2_kos"
SEGMENTLER: List[int] = [5000, 5000, 5000, 5000]   # 4 × 5000 = 20.000 adım
BLOK, BATCH, LR = 128, 8, 1e-4
WARMUP, TOPLAM_ADIM, MIN_LR = 100, 20000, 1e-6
CLIP = 1.0
WIKI_PATERN = 4          # pencere idx % 4 == 0 ⇒ wiki (%25), aksi talimat (%75)
CE_TAVAN = 3.8887        # kanonik: taban CE 3,5352 × (1 + ESIK_A_ARTIS/100)
assert abs(CE_TAVAN - 3.5351739511825144 * (1 + ECA.ESIK_A_ARTIS / 100.0)) < 1e-3, \
    "CE_TAVAN kanonik sabitle uyuşmuyor — elle yazılmış olabilir"


def durdur(mesaj: str) -> None:
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def yaz_json(yol: str, nesne: Dict[str, Any]) -> None:
    gecici = yol + ".tmp"
    with open(gecici, "w", encoding="utf-8") as f:
        json.dump(nesne, f, ensure_ascii=False, indent=2)
    os.replace(gecici, yol)


def ekle_jsonl(yol: str, nesne: Dict[str, Any]) -> None:
    with open(yol, "a", encoding="utf-8") as f:
        f.write(json.dumps(nesne, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def wikipedia_ce(ckpt: str, vocab: Vocabulary, device: torch.device) -> Dict[str, Any]:
    """Wikipedia CE — KANONİK `ECA.a_ekseni` (import; kopya değil)."""
    m = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    d = torch.load(ckpt, map_location="cpu")
    for k in [k for k in list(d.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del d[k]
    m.load_state_dict(resize_state_dict(m, d), strict=False)
    m.to(device)
    m.eval()
    r = ECA.a_ekseni(m, vocab, device)
    del m
    if device.type == "mps":
        torch.mps.empty_cache()
    return r


def kapi_karari(ce: float, tavan: float = CE_TAVAN) -> Tuple[bool, str]:
    return (ce > tavan, f"CE {ce:.4f} {'>' if ce > tavan else '<='} tavan {tavan:.4f}")


def pencere_oku(mm, blok_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kanonik hedef konvansiyonu (`KristalDataset.get_batch`): y = data[i+1 : i+1+BLOK].
    roll DEĞİL — roll sarmal bağlantı öğretir, train.py'nin yolu öğretmez."""
    off = blok_idx * BLOK
    x = torch.from_numpy(np.asarray(mm[off:off + BLOK], dtype=np.int64))
    y = torch.from_numpy(np.asarray(mm[off + 1:off + 1 + BLOK], dtype=np.int64))
    return x, y


def main() -> int:
    global SEGMENTLER, TOPLAM_ADIM, WIKI_PATERN, LR, KOS, WARMUP
    import argparse
    ap = argparse.ArgumentParser(description="P2 Temel 2.0 karışımlı koşum sürücüsü")
    ap.add_argument("--kapi-sinamasi", action="store_true",
                    help="yalnız CE kapısının İKİ DALINI sına ve çık")
    # Aşama 3 (ince ayar) için parametrik çalışma: sabitler CLI'dan override edilebilir.
    # Her override ilanlı tarifte beyan edilir; ilansız koşum YASAK.
    ap.add_argument("--taban", default=TABAN, help="başlangıç checkpoint'i (donmuş yoldan okuma)")
    ap.add_argument("--kos-adi", default=None, help="çıktı dizini (KOS override; ilanla beyanlı)")
    ap.add_argument("--segmentler", default=None, help="virgüllü adım listesi, ör. '1000,1000'")
    ap.add_argument("--lr", type=float, default=LR, help="peak lr")
    ap.add_argument("--wiki-patern", type=int, default=WIKI_PATERN, help="idx %% N == 0 ⇒ wiki")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="warmup adım (0=kapalı)")
    ap.add_argument("--ilan", default="data/eval/anka_p2_temel2_ilani_2026-09-22.md",
                    help="bu koşumu tarif eden ön-kayıtlı ilan (sonuc.json'a yazılır)")
    a = ap.parse_args()

    if a.kapi_sinamasi:
        dus, s1 = kapi_karari(CE_TAVAN + 0.01)
        gec, s2 = kapi_karari(CE_TAVAN - 0.01)
        print(f"[kapı sınaması] tavan {CE_TAVAN}: üstü ⇒ DUR={dus} ({s1}) · "
              f"altı ⇒ DUR={gec} ({s2})")
        return 0 if (dus and not gec) else 2

    if a.segmentler:
        SEGMENTLER = [int(s) for s in a.segmentler.split(",") if s.strip()]
        TOPLAM_ADIM = sum(SEGMENTLER)
    if a.wiki_patern:
        WIKI_PATERN = a.wiki_patern
    LR = a.lr
    if a.warmup >= 0:
        WARMUP = a.warmup
    if a.kos_adi:
        KOS = f"scratch/{a.kos_adi}"
    taban = a.taban

    for yol in (taban, SOZLUK, WIKI_BIN, SFT_BIN):
        if not os.path.exists(yol):
            durdur(f"girdi yok: '{yol}'")
    if is_frozen_path(f"{KOS}/seg_1.pt"):
        durdur("çıktı yolu donmuş desene düşüyor — scratch/ altında kalmalı")

    # H1 — tek eğitici (koşum ÖNCESİ; T-0097/K5)
    h1 = tek_egitici_onkontrol(WIKI_BIN)
    print(f"[H1] tek eğitici: {h1['tutucu']} tutucu {h1['pidler']}", flush=True)

    os.makedirs(KOS, exist_ok=True)
    vocab = Vocabulary()
    vocab.load(SOZLUK)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        durdur("MPS yok — koşum sandbox DIŞINDA koşulmalı (MPS sandbox'ta gizli)")

    # MODEL — taban (soyağacı: taban-anka-kristal-yok; base_sha256 anka_a1r.pt)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    d = torch.load(taban, map_location="cpu")
    for k in [k for k in list(d.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del d[k]
    model.load_state_dict(resize_state_dict(model, d), strict=False)
    model.to(device)
    taban_sha = sha256_file(taban)
    print(f"[model] taban {taban} · sha256 {taban_sha[:16]}…", flush=True)

    # CARRY (T-0092): sidecar VARSA digest kapısı, YOKSA AÇIK uyarı.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    opt_yolu = optimizer_sidecar_path(taban)   # <taban>.pt.opt.pt
    if os.path.exists(opt_yolu):
        payload = torch.load(opt_yolu, map_location="cpu")
        if payload.get("model_sha256") != taban_sha:
            durdur(f"CARRY KAPISI: {opt_yolu} digest'i tabanla ESLESMIYOR "
                   f"(payload {str(payload.get('model_sha256'))[:16]}… ≠ taban {taban_sha[:16]}…)")
        optimizer.load_state_dict(payload["optimizer"])
        carry_beyan = (f"momentler yuklendi ({len(optimizer.state)} parametre, "
                       f"adim={payload.get('adim')}; T-0092)")
        print(f"[carry] optimizer {carry_beyan}", flush=True)
    else:
        carry_beyan = f"sidecar YOK ⇒ sıfırdan (açık uyarı, T-0092/G4a)"
        print(f"[carry] {opt_yolu} YOK — AdamW momenti bulunamadi, optimizer SIFIRDAN "
              f"(T-0092/G4a)", flush=True)

    # VERİ — memmap; pencere seçimi deterministik (seed 42)
    wiki_mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    sft_mm = np.memmap(SFT_BIN, dtype=np.uint16, mode="r")
    wiki_blok_sayisi = len(wiki_mm) // BLOK
    sft_blok_sayisi = len(sft_mm) // BLOK
    print(f"[veri] wiki {wiki_blok_sayisi:,} pencere · sft {sft_blok_sayisi:,} pencere", flush=True)

    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    pad_id = vocab.stoi.get("<PAD>", 1)
    rng = np.random.default_rng(42)

    seg_jsonl = f"{KOS}/segments.jsonl"
    sonuc_yolu = f"{KOS}/sonuc.json"
    sentinel = f"{KOS}/SONUC_BITTI"
    sonuc: Dict[str, Any] = {
        "gorev": "anka_p2 · karışımlı koşum (parametrik sürücü)",
        "ilan": a.ilan,
        "baslangic_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "taban": taban, "taban_sha256": taban_sha,
        "wiki_bin": WIKI_BIN, "sft_bin": SFT_BIN, "sft_bin_sha256": sha256_file(SFT_BIN),
        "segmentler": SEGMENTLER, "blok": BLOK, "batch": BATCH, "lr": LR,
        "scheduler": {"warmup": WARMUP, "toplam_adim": TOPLAM_ADIM, "min_lr": MIN_LR},
        "clip": CLIP, "wiki_patern": f"pencere idx %{WIKI_PATERN} == 0 ⇒ wiki",
        "optimizer_carry": carry_beyan,
        "ce_tavan": CE_TAVAN, "maske": "PAD→−100 + ignore_index=−100 (P2/Aşama 1)",
        "dallar": [], "durum": "SURUYOR",
    }
    yaz_json(sonuc_yolu, sonuc)

    t0 = time.time()
    global_adim = 0
    pencere_no = 0          # pencere sayaç — karışım paterni BUNA bağlı (adım değil)
    bos_hedef_atlanan = 0
    for seg_no, seg_uzunluk in enumerate(SEGMENTLER, 1):
        ckpt = f"{KOS}/seg_{seg_no}.pt"
        seg_t0 = time.time()
        kayip_gecmis: List[float] = []
        for _ in range(seg_uzunluk):
            # b8 PARTİ: 8 pencere; pencere idx % 4 == 0 ⇒ wiki, aksi talimat (%25/%75)
            xs, hedefler = [], []
            deneme = 0
            while len(xs) < BATCH:
                if pencere_no % WIKI_PATERN == 0:
                    x_cpu, y_cpu = pencere_oku(wiki_mm, int(rng.integers(0, wiki_blok_sayisi)))
                    h_np = y_cpu.numpy().copy()          # duz sonraki-jeton hedefi (kanonik +1 pencere)
                    h_np = maske_pad_hedefleri(h_np, pad_id, True)
                else:
                    x_cpu, y_cpu = pencere_oku(sft_mm, int(rng.integers(0, sft_blok_sayisi)))
                    # mask_prompt_targets 2-B bekler (train.py [B,L] yolu): [1,128] sar → [128]'e geri
                    h2 = mask_prompt_targets(x_cpu.unsqueeze(0), y_cpu.unsqueeze(0),
                                             output_start_id, eos_id)
                    h_np = np.asarray(h2)[0]
                    h_np = maske_pad_hedefleri(h_np, pad_id, True)
                pencere_no += 1
                deneme += 1
                if int((h_np != MASKE).sum()) == 0:
                    bos_hedef_atlanan += 1               # sıfır-hedef pencere: AÇIKÇA sayılır, sessiz DEĞİL
                    if deneme >= 4 * BATCH:
                        break                            # aşırı boş partide az pencereyle devam
                    continue
                xs.append(x_cpu)
                hedefler.append(torch.from_numpy(h_np))
            if not xs:
                continue                                 # partide tek geçerli pencere bile yok → adım sayılmaz
            x_cpu_bat = torch.stack(xs)                  # [8, 128] CPU
            x = x_cpu_bat.to(device)
            hedef = torch.stack(hedefler).to(device)
            sign_mask = model.embedding.compute_sign_mask(x_cpu_bat).to(device)

            optimizer.zero_grad()
            _, loss = model(x, targets=hedef, sign_mask=sign_mask, ignore_index=MASKE)
            kayip = float(loss.item())
            if not math.isfinite(kayip):
                durdur(f"L1: adım {global_adim + 1} kayıp SONLU DEĞİL ({kayip}) — MPS garbaşı")
            if global_adim == 0:
                ln_v = math.log(len(vocab.stoi))
                if kayip == 0.0:
                    durdur(f"L1: ilk kayıp 0,0 — MPS sessiz no-op (T-0073 sınıfı); "
                           f"devam koşumu bozulmuş modelle başladı")
                if kayip > ln_v - 1.0:
                    durdur(f"L1: ilk kayıp {kayip:.4f} ≈ ln V ({ln_v:.4f}) — model SIFIRDAKİNE "
                           f"yakın: taban yüklenememiş/bozulmuş (canlılık imzası moda bağlı; "
                           f"devam koşumunda ln V BOZUKLUK imzasıdır)")
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
                print(f"  adım {global_adim}/{TOPLAM_ADIM} | kayıp {sum(son100) / len(son100):.4f} "
                      f"| LR {cur_lr:.6f} | {time.time() - seg_t0:.0f}s", flush=True)
                if device.type == "mps":
                    torch.mps.empty_cache()

        # segment kaydı + CE kapısı
        torch.save(model.state_dict(), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        # OPTIMIZER + SCHEDULER CARRY (T-0092 + P2/Aşama 1): model önce, yan dosya sonra.
        opt_gecici = f"{ckpt}.opt.pt.tmp"
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": {"warmup_steps": WARMUP, "toplam_adim": TOPLAM_ADIM,
                                  "min_lr": MIN_LR, "peak_lr": LR},
                    "adim": global_adim,
                    "model_sha256": sha256_file(ckpt)}, opt_gecici)
        os.replace(opt_gecici, f"{ckpt}.opt.pt")

        ce = wikipedia_ce(ckpt, vocab, device)
        dustu, gerekce = kapi_karari(ce["CE_ort"])
        kayit = {
            "segment": seg_no, "adim": seg_uzunluk, "global_adim": global_adim,
            "sure_sn": round(time.time() - seg_t0, 1),
            "ckpt": ckpt, "ckpt_sha256": sha256_file(ckpt),
            "ce_ort": round(ce["CE_ort"], 4), "ce_std": round(ce["CE_std"], 4),
            "ppl": round(float(math.exp(ce["CE_ort"])), 2),
            "kayip_son": round(kayip_gecmis[-1], 4) if kayip_gecmis else None,
            "lr_son": round(cur_lr, 8) if kayip_gecmis else None,
            "bos_hedef_pencere": bos_hedef_atlanan,
            "kapi_dustu": dustu, "kapi_gerekce": gerekce,
        }
        sonuc["dallar"].append(kayit)
        yaz_json(sonuc_yolu, sonuc)
        ekle_jsonl(seg_jsonl, kayit)
        print(f"[seg {seg_no}] CE {ce['CE_ort']:.4f} (ppl {kayit['ppl']}) · "
              f"{kayit['sure_sn']:.0f} sn · kapı {'DÜŞTÜ' if dustu else 'geçti'}", flush=True)
        if dustu:
            sonuc.update({"durum": "DURDU", "duran_dal": f"seg_{seg_no} CE KAPISI",
                          "aciklama": gerekce})
            yaz_json(sonuc_yolu, sonuc)
            durdur(f"CE KAPISI seg_{seg_no}: {gerekce}")

    sonuc.update({"durum": "BITTI",
                  "bitis_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "toplam_sn": round(time.time() - t0, 1),
                  "nihai_ckpt": f"{KOS}/seg_{len(SEGMENTLER)}.pt",
                  "nihai_sha256": sha256_file(f"{KOS}/seg_{len(SEGMENTLER)}.pt")})
    yaz_json(sonuc_yolu, sonuc)
    with open(sentinel, "w", encoding="utf-8") as f:
        f.write(f"BITTI {sonuc['bitis_utc']} nihai={sonuc['nihai_ckpt']} "
                f"sha256={sonuc['nihai_sha256']}\n")
    print(f"\n[BİTTİ] nihai={sonuc['nihai_ckpt']} · {sonuc['toplam_sn']:.0f} sn", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)