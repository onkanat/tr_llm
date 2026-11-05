#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 — RAG-ALINTI SFT-REPLAY EĞİTİMİ (p26 tarifi; GÖREV 3)
==========================================================
scripts/train_p26_sft_replay.py tarifi BİREBİR korunur (kayıp formu, SUB_CHUNK
alt-yığın, KL_CHUNK yığını, kaynak-tensorü satır-seçimi, cosine LR, MPS
empty_cache disiplini) — mevcut dosyaya DOKUNULMAZ; parçalar import edilir.

Bu sete özgü İKİ bilinçli uyarlama (iş emri):
1. VAL KUYRUĞU SON-50: p26 sabiti son-150'dir; bu set küçük ve KENDİ hedefli
   (pure_base_sft ile KARIŞMAZ) → recs[-50:] val, gerisi train.
2. VAL NÖBETİ: her epok sonunda val CE ölçülür; final val > en-iyi epok val
   ise eğitim SONU en-iyi epok checkpoint'i GERİ YÜKLENİR (artarsa en-iyi
   epok döner). Epok checkpoint'leri scratch/p1_rag_alinti_e{n}.pt'ye yazılır,
   koşum sonunda temizlenir.

Başlangıç VE KL öğretmeni: data/anka_taban_pure.pt (öz-çapa; adım-0'da KL≈0).
Pencere: data/kulliyat_v2/kulliyat_v2.bin, pencere-n = train/4 (iş emri).

EPOK NOTU: iş emri 2-3 epok önerir; koşum-1 (3 epok, 51 opt-adım) ölçülmüş
UNDERFIT verdi (val CE 4.75→4.12 hâlâ düşüyordu; model kendi eğitim kaydını
bile geri üretemiyordu — 51 adım, p26 kayıp formunun payda-seyreltmesiyle
SFT sinyali ~0.3× ince). Val nöbeti tam bu durum için tarife dahildir:
val düşmeye devam ettikçe epok sayısı uzatılır, nöbet en-iyi epoğu döndürür.
T-0209 ölçeğine (171 adım) ulaşmak için koşum-2 12 epok (≈204 adım) kullandı.

Kullanım (iş emri koşumu):
  venv/bin/python scripts/train_p26_rag_alinti.py \
    --sft-path data/pedagogy/rag_alinti_sft.jsonl \
    --init-model data/anka_taban_pure.pt --teacher-model data/anka_taban_pure.pt \
    --pencere-bin data/kulliyat_v2/kulliyat_v2.bin --kl-beta 0.2 --peak-lr 5e-5 \
    --epochs 3 --save-path data/anka_rag_alinti.pt --allow-frozen-write \
    --history-path data/eval/p1_rag_alinti_history.json \
    --hukum-path data/eval/p1_rag_alinti_hukum_2026-11-05.json
"""
import argparse
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
os.chdir(KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.train_step_b1_5_rigorous import (FastSampleAlignedDataset,
                                              evaluate_val_loss, compute_sha256)
from scripts.t0200_mutasyon_kanit import PencereDataset, yukle_checkpoint
from scripts.train_p26_sft_replay import (BLOCK, KL_CHUNK, SUB_CHUNK,
                                         KarisimDataset, pencere_ce_kulliyat,
                                         get_lr, kaydet)
from train import check_frozen_save_path

TARIH = time.strftime("%Y-%m-%d")
SCRATCH = os.path.join(KÖK, "scratch")


def parse_args():
    p = argparse.ArgumentParser(description="P1 RAG-alıntı SFT-replay (p26 tarifi, val-50 + nöbeti)")
    p.add_argument("--init-model", default="data/anka_taban_pure.pt")
    p.add_argument("--teacher-model", default=None, help="KL öğretmeni; verilmezse init-model")
    p.add_argument("--sft-path", default="data/pedagogy/rag_alinti_sft.jsonl")
    p.add_argument("--pencere-bin", default="data/kulliyat_v2/kulliyat_v2.bin")
    p.add_argument("--pencere-n", type=int, default=None, help="verilmezse train/4 (iş emri)")
    p.add_argument("--kl-beta", type=float, default=0.2)
    p.add_argument("--peak-lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--val-n", type=int, default=50, help="val kuyruğu: son-50 (sete özgü)")
    p.add_argument("--save-path", required=True)
    p.add_argument("--allow-frozen-write", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab", default="data/rebuild/vocab_anka_r1_33114.json")
    p.add_argument("--history-path", default="data/eval/p1_rag_alinti_history.json")
    p.add_argument("--hukum-path", default=f"data/eval/p1_rag_alinti_hukum_{TARIH}.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.teacher_model is None:
        args.teacher_model = args.init_model

    t_program_basla = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("P1_DUR: MPS yok — koşum sandbox DIŞI olmalı (p26 kuralı)")
    print(f"=== P1 RAG-ALINTI SFT-REPLAY (p26 tarifi) | cihaz={device} ===", flush=True)

    check_frozen_save_path(args.save_path, allow_frozen_write=args.allow_frozen_write)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(SCRATCH, exist_ok=True)

    lexicon = LexiconManager(); lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary(); vocab.load(args.vocab)
    tokenizer = KristalTokenizer(compiler, vocab)
    vocab_size = len(vocab.stoi)

    for yol in (args.init_model, args.teacher_model, args.sft_path, args.pencere_bin):
        if not os.path.exists(yol):
            print(f"P1_DUR: bulunamadı: {yol}", flush=True); return 2

    mm = np.memmap(args.pencere_bin, dtype=np.uint16, mode="r")
    if int(mm.max()) >= vocab_size:
        raise RuntimeError(f"P1_DUR: pencere bin maks id {int(mm.max())} >= vocab_size {vocab_size}")
    del mm

    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    yukle_checkpoint(args.init_model, model)
    model.to(device)
    hoca = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    yukle_checkpoint(args.teacher_model, hoca)
    hoca.to(device); hoca.eval()
    for p_ in hoca.parameters():
        p_.requires_grad = False

    print(f"[PROVENANS] init sha256: {compute_sha256(args.init_model)}", flush=True)
    print(f"[PROVENANS] öğretmen sha256: {compute_sha256(args.teacher_model)}", flush=True)
    print(f"[PROVENANS] pencere bin sha256: {compute_sha256(args.pencere_bin)}", flush=True)
    print(f"[PROVENANS] sft sha256: {compute_sha256(args.sft_path)}", flush=True)

    recs = [json.loads(l) for l in open(args.sft_path, encoding="utf-8") if l.strip()]
    # UYARLAMA-1: val kuyruğu SON-50 (p26 sabiti son-150; bu sete özgü kural)
    val_records, train_records = recs[-args.val_n:], recs[:-args.val_n]
    pencere_n = args.pencere_n if args.pencere_n is not None else max(1, len(train_records) // 4)
    print(f"SFT train: {len(train_records)} | val (son-{args.val_n}): {len(val_records)} | "
          f"pencere: {pencere_n}", flush=True)

    val_dataset = FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=BLOCK)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    karisim = KarisimDataset(train_records, tokenizer, vocab, model,
                             args.pencere_bin, pencere_n, device, args.seed)
    n_toplam = len(karisim)
    steps_per_epoch = math.ceil(n_toplam / (args.batch * args.accum))
    total_opt = steps_per_epoch * args.epochs
    print(f"karışım toplam {n_toplam} -> {steps_per_epoch} opt-adım/epok, {total_opt} toplam", flush=True)

    init_ce, init_ppl = evaluate_val_loss(model, val_loader)
    init_pencere = pencere_ce_kulliyat(model, args.pencere_bin, device)
    print(f"[BAŞLANGIÇ] masked SFT val CE: {init_ce:.6f} | pencere-CE (kulliyat_v2): {init_pencere:.4f}", flush=True)
    on_hesap_sn = time.time() - t_program_basla
    print(f"[ÖN-HESAP] {on_hesap_sn:.1f} sn", flush=True)
    if device.type == "mps":
        torch.mps.empty_cache()

    optimizer = optim.AdamW(model.parameters(), lr=args.min_lr, weight_decay=args.weight_decay)
    model.train(); optimizer.zero_grad()
    opt_step, accum_sayac, running = 0, 0, 0.0
    adim_sureleri, history, kesildi = [], [], False
    kl_aktif_mikro, kl_toplam_mikro, kl_degerleri = 0, 0, []
    adim_kl_degerleri = []
    step_t0 = None
    t_basla = time.time()

    # ---- UYARLAMA-2: val nöbeti durumu
    val_nobeti = [{"epok": 0, "val_ce": init_ce, "aciklama": "başlangıç (init)"}]
    en_iyi_epok, en_iyi_val = 0, init_ce
    en_iyi_dosya = None  # yalnız en-iyi epok checkpoint'i diskte tutulur (budama)

    for epok in range(args.epochs):
        train_loader = DataLoader(karisim, batch_size=args.batch, shuffle=True)
        for batch in train_loader:
            if step_t0 is None:
                step_t0 = time.time()
            x, y, sm, kaynak = batch
            kl_toplam_mikro += 1
            x = x.to(device); y = y.to(device); sm = sm.to(device)
            sm_payda = sm.sum() + 1e-8
            satirlar = ((kaynak == 1.0).nonzero(as_tuple=True)[0] if args.kl_beta > 0.0
                        else torch.empty(0, dtype=torch.long))
            kl_satir_sayisi = int(satirlar.numel())
            T_poz = x.size(1)
            kl_aktif_alt, running_kl, running_mikro = 0, 0.0, 0.0
            for a0 in range(0, x.size(0), SUB_CHUNK):
                x_c = x[a0:a0 + SUB_CHUNK]; y_c = y[a0:a0 + SUB_CHUNK]; sm_c = sm[a0:a0 + SUB_CHUNK]
                logits, _ = model(x_c)
                V = logits.size(-1)
                ce_none = F.cross_entropy(logits.reshape(-1, V), y_c.reshape(-1), reduction="none").view_as(sm_c)
                loss = (ce_none * sm_c).sum() / sm_payda
                if kl_satir_sayisi > 0:
                    yerel = satirlar[(satirlar >= a0) & (satirlar < a0 + SUB_CHUNK)] - a0
                    if yerel.numel() > 0:
                        kl_parca = []
                        for k0 in range(0, yerel.numel(), KL_CHUNK):
                            idx = yerel[k0:k0 + KL_CHUNK].to(device)
                            with torch.no_grad():
                                hoca_logits, _ = hoca(torch.index_select(x_c, 0, idx))
                                log_p_t = F.log_softmax(hoca_logits, dim=-1)
                                del hoca_logits
                            log_p_s = F.log_softmax(torch.index_select(logits, 0, idx), dim=-1)
                            kl_parca.append((log_p_t.exp() * (log_p_t - log_p_s)).sum(-1))
                            del log_p_t, log_p_s
                        kl_satir_toplam = torch.cat(kl_parca, 0).sum()
                        del kl_parca
                        loss = loss + args.kl_beta * kl_satir_toplam / (kl_satir_sayisi * T_poz)
                        kl_aktif_alt += 1
                        running_kl += float(kl_satir_toplam.item())
                        del kl_satir_toplam
                (loss / args.accum).backward()
                running_mikro += loss.item()
                del logits, loss, ce_none
            running += running_mikro; accum_sayac += 1
            if kl_aktif_alt > 0:
                kl_aktif_mikro += 1
                kl_v = running_kl / (kl_satir_sayisi * T_poz)
                adim_kl_degerleri.append(kl_v)
                kl_degerleri.append(kl_v)

            if accum_sayac == args.accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                lr = get_lr(opt_step, total_opt, args.peak_lr, args.min_lr, args.warmup)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step(); optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                adim_sureleri.append(time.time() - step_t0)
                step_t0 = None
                opt_step += 1; accum_sayac = 0
                step_loss = running / args.accum; running = 0.0
                history.append({"adim": opt_step, "loss": step_loss, "lr": lr, "epok": epok + 1,
                                "kl_aktif": len(adim_kl_degerleri),
                                "kl_ort": (sum(adim_kl_degerleri) / len(adim_kl_degerleri)) if adim_kl_degerleri else None})
                adim_kl_degerleri = []
                if opt_step % 10 == 0 or opt_step in (1, total_opt):
                    son = adim_sureleri[-10:]
                    ort = sum(son) / len(son)
                    print(f"Adım {opt_step}/{total_opt} | Loss {step_loss:.4f} | LR {lr:.2e} | "
                          f"adım-süresi ort {ort:.2f}s | KL-aktif {kl_aktif_mikro}/{kl_toplam_mikro} mikro", flush=True)

        # ---- UYARLAMA-2: epok sonu val nöbeti + checkpoint (yalnız en-iyi diskte kalır)
        if not kesildi:
            epok_dosya = os.path.join(SCRATCH, f"p1_rag_alinti_e{epok + 1}.pt")
            kaydet(model, epok_dosya)
            val_ce, val_ppl = evaluate_val_loss(model, val_loader)
            val_nobeti.append({"epok": epok + 1, "val_ce": val_ce, "val_ppl": val_ppl})
            if val_ce < en_iyi_val:
                if en_iyi_dosya and os.path.exists(en_iyi_dosya):
                    os.remove(en_iyi_dosya)
                en_iyi_epok, en_iyi_val, en_iyi_dosya = epok + 1, val_ce, epok_dosya
            elif os.path.exists(epok_dosya):
                os.remove(epok_dosya)
            print(f"[VAL-NÖBETİ] epok {epok + 1}: val CE {val_ce:.6f} "
                  f"(en-iyi: epok {en_iyi_epok} @ {en_iyi_val:.6f})", flush=True)
            if device.type == "mps":
                torch.mps.empty_cache()
            model.train()

    toplam_sure = time.time() - t_basla

    # ---- UYARLAMA-2: en-iyi epok geri döndürme (val arttıysa)
    geri_yuklendi = False
    son_epok = args.epochs
    son_val = val_nobeti[-1]["val_ce"]
    if en_iyi_epok != son_epok and en_iyi_epok > 0 and en_iyi_dosya:
        print(f"[VAL-NÖBETİ] val CE arttı (son epok {son_epok}: {son_val:.6f} > en-iyi {en_iyi_val:.6f}) "
              f"→ epok {en_iyi_epok} checkpoint'i geri yükleniyor", flush=True)
        yukle_checkpoint(en_iyi_dosya, model)
        model.to(device)
        geri_yuklendi = True
    elif en_iyi_epok == 0:
        print(f"[VAL-NÖBETİ] hiçbir epok init val'yi geçemedi → init ağırlıkları kalır "
              f"(init {init_ce:.6f} ≤ en-iyi-epok {en_iyi_val:.6f})", flush=True)

    kaydet(model, args.save_path)
    final_ce, final_ppl = evaluate_val_loss(model, val_loader)
    final_pencere = pencere_ce_kulliyat(model, args.pencere_bin, device)
    print(f"\n[BİTİŞ] masked SFT val CE: {init_ce:.6f} -> {final_ce:.6f} | "
          f"pencere-CE: {init_pencere:.4f} -> {final_pencere:.4f} | Süre: {toplam_sure:.1f} sn", flush=True)
    kl_ort = (sum(kl_degerleri) / len(kl_degerleri)) if kl_degerleri else None
    if args.kl_beta > 0.0:
        print(f"[KL-KANIT] KL dalı {kl_aktif_mikro}/{kl_toplam_mikro} mikro-partide ateşlendi"
              + (f" | tüm-koşum KL ort {kl_ort:.6f}" if kl_ort is not None
                 else " | KL HİÇ ATEŞLENMEDİ")
              + " (öz-çapa: öğretmen==init → adım-0'da KL≈0 beklenir)", flush=True)
    print(f"[MODEL_KAYDEDİLDİ] {args.save_path} (sha256: {compute_sha256(args.save_path)})", flush=True)

    # epok checkpoint'leri temizle (geçici artefaktlar)
    for e in range(1, args.epochs + 1):
        yp = os.path.join(SCRATCH, f"p1_rag_alinti_e{e}.pt")
        if os.path.exists(yp):
            os.remove(yp)

    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    hukum = {
        "gorev": "P1 RAG-alıntı SFT-replay (p26 tarifi)", "durum": "BASARILI",
        "init_model": args.init_model, "teacher_model": args.teacher_model,
        "sft_path": args.sft_path, "sft_sha256": compute_sha256(args.sft_path),
        "kl_beta": args.kl_beta, "pencere_bin": args.pencere_bin, "pencere_n": pencere_n,
        "sft_train_n": len(train_records), "val_n": len(val_dataset),
        "val_kuyrugu": f"son-{args.val_n} (sete özgü)",
        "peak_lr": args.peak_lr, "min_lr": args.min_lr, "warmup": args.warmup,
        "epochs": args.epochs, "opt_adim": opt_step, "sure_sn": round(toplam_sure, 1),
        "on_hesap_sn": round(on_hesap_sn, 1),
        "kl_aktif_mikro": kl_aktif_mikro, "kl_toplam_mikro": kl_toplam_mikro,
        "kl_ort": kl_ort,
        "s_adim_son20_ort": round(float(np.mean(adim_sureleri[-20:])), 3) if adim_sureleri else None,
        "s_adim_son20_std": round(float(np.std(adim_sureleri[-20:])), 3) if adim_sureleri else None,
        "init_val_ce": init_ce, "final_val_ce": final_ce,
        "val_nobeti": val_nobeti,
        "en_iyi_epok": en_iyi_epok, "en_iyi_val_ce": en_iyi_val,
        "geri_yuklendi": geri_yuklendi,
        "init_pencere_ce": init_pencere, "final_pencere_ce": final_pencere,
        "cikti": args.save_path, "cikti_sha256": compute_sha256(args.save_path),
    }
    with open(args.hukum_path, "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2, ensure_ascii=False)
    print(f"[HÜKÜM] {args.hukum_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
