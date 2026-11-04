#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.6 — SFT-REPLAY: anka_taban_v3 üzerine SFT + külliyat pencere karışımı (T-0209 kalıbı).

T-0209 (scripts/train_t0209_taban_pure.py) kalıbının v3 uyarlaması. Üç bilinçli fark:
1. KL KAYNAK-DÜZELTMESİ: T-0209'da 'len(batch)==4' her iki dalda da doğruydu →
   is_pencere hep False → KL dali ÖLÜ KODDU (ölçüldü; marangoz varyantında da aynısı).
   Burada kaynak tensorü (4. eleman) ile satır-seçim yapılır → KL yalnız PENCERE
   satırlarında gerçekten ateşlenir (öğretmen no_grad; t0200 kl_pencere_kayip deseni).
   T-0209'un ölçülmüş (KL'siz) davranışı için: --kl-beta 0.
2. Pencere kaynağı: data/kulliyat_v2/kulliyat_v2.bin (T-0209: anka_a1r_pretrain.bin).
3. Başlangıç VE KL-öğretmeni: data/anka_taban_v3.pt (öz-çapa; adım-0'da KL≈0).

Kayıp formu T-0209'dan BİREBİR korunmuştur (bilinçli): ce_loss =
(CE_none * sm).sum()/(sm.sum()+1e-8); sm = ±1 İŞARET maskesidir (kayıp maskesi
DEĞİL), -100 hedefler CE'de 0 verir, prompt konumları PAYDADADIR (SFT kaybını
~0.5× seyreltir — ölçülen tarife sadakat için değiştirilmedi).

Yazım-sırası doldurmaları (taslak yer tutucularının gerçek hali):
4. ADIM-SÜRESİ GERÇEK ÖLÇÜM: her opt-adımın BAŞINDA (accum penceresinin ilk
   mikro-partisinde) step_t0=time.time() alınır; optimizer.step() hemen ardından
   fark adim_sureleri'ne kaydedilir (baskı/checkpoint süresi ölçüme girmez).
   Pilot sonunda son-20 adım ort±std basılır.
5. KL-KANIT sayaçları: kl_aktif_mikro/kl_toplam_mikro ve KL değerleri history +
   hüküm JSON'a yazılır — B1 düzeltmesinin canlı kanıtı koşum çıktısından okunur.
6. MPS-BELLEK ONARIMI (pilot çökmesi 04.11.2026: komut-tamponu 'Insufficient
   Memory', adım-2, 8GB M2): KL satırları KL_CHUNK'lık yığınlarla işlenir
   (öğretmen no_grad transientleri ~4× küçülür; formül t0200 kl_pencere_kayip
   ile BİREBİR — yığın satır-toplamlarının cat().mean()'i tam-seçimin
   .sum(-1).mean()'ine eşittir) + her opt-adım sonunda torch.mps.empty_cache()
   (t0200 R4 deseni) + döngü öncesi temizlik.
7. MPS-BELLEK ONARIMI-2 (ikinci çökme: adım-2 sessiz SIGKILL, GPU wired-limit):
   mikro-parti SUB_CHUNK satırlık alt-yığınlarla ileri+geri yayılır. Kayıp
   formu BİREBİR korunur: payda (sm.sum()+1e-8) TAM partiden bir kez alınır;
   KL satır-seçimi global kaynak tensorüyle yapılır ve KL payı mikro-partinin
   TOPLAM pencere-satır sayısına bölünür → alt-yığın kayıplarının toplamı
   tam-parti (T-0209 ce_loss + beta*kl) DEĞERİNE VE GRADYANINA eşittir.
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
from train import check_frozen_save_path  # donmuş yol kapısı (data/*.pt)

BLOCK = 128
KL_CHUNK = 4  # MPS-bellek: öğretmen/KL satırları bu yığın boyuyla işlenir (docstring not-6)
SUB_CHUNK = 8  # MPS-bellek: mikro-parti bu satır sayısıyla alt-yığınlanır (docstring not-7)
TARIH = time.strftime("%Y-%m-%d")


def parse_args():
    p = argparse.ArgumentParser(description="P2.6 SFT-replay (T-0209 kalıbı, v3 üzerine)")
    p.add_argument("--init-model", default="data/anka_taban_v3.pt")
    p.add_argument("--teacher-model", default=None, help="KL öğretmeni; verilmezse init-model")
    p.add_argument("--sft-path", default="data/pedagogy/pure_base_sft.jsonl")
    p.add_argument("--pencere-bin", default="data/kulliyat_v2/kulliyat_v2.bin")
    p.add_argument("--pencere-n", type=int, default=1087, help="4:1 (4348/4) = ~%%20 örnek")
    p.add_argument("--kl-beta", type=float, default=0.2)
    p.add_argument("--peak-lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--save-path", required=True, help="koşum sonunda yazılır; data/*.pt için --allow-frozen-write şart")
    p.add_argument("--allow-frozen-write", action="store_true")
    p.add_argument("--max-opt-steps", type=int, default=None, help="PİLOT: N opt-adımda kes, değerlendirmeleri atla")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab", default="data/rebuild/vocab_anka_r1_33114.json")
    p.add_argument("--history-path", default="data/eval/p26_replay_history.json")
    p.add_argument("--hukum-path", default=f"data/eval/p26_replay_hukum_{TARIH}.json")
    return p.parse_args()


def get_lr(step, total_steps, peak, min_lr, warmup):
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def pencere_ce_kulliyat(model, bin_path, device, n=25, seed=7):
    """t0200 pencere_ce_olc rig'i, parametreleştirilmiş (kulliyat_v2 için)."""
    was_training = model.training
    model.eval()
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_win = int(mm.size // BLOCK)
    rng = random.Random(seed)
    baslar = sorted(rng.sample(range(n_win), min(n, n_win)))
    mm_i = np.asarray(mm, dtype=np.uint16)
    kayiplar = []
    for b in baslar:
        seq = mm_i[b * BLOCK:(b + 1) * BLOCK].astype(np.int64)
        x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            sm = model.embedding.compute_sign_mask(x.cpu()).to(device)
            _, loss = model(x, targets=y, sign_mask=sm)
        kayiplar.append(float(loss.item()))
        if device.type == "mps":
            torch.mps.empty_cache()
    if was_training:
        model.train()
    return round(float(np.mean(kayiplar)), 4)


class KarisimDataset(torch.utils.data.Dataset):
    """T-0209 PureTabanKarısımDataset kalıbı: SFT jsonl + pencere .bin, kaynak etiketiyle."""

    def __init__(self, sft_records, tokenizer, vocab, model, pencere_bin, pencere_n, device, seed):
        self.sft = FastSampleAlignedDataset(sft_records, tokenizer, vocab, model, block_size=BLOCK)
        self.pencere = PencereDataset(pencere_bin, model, device, adet=pencere_n, seed=seed)

    def __len__(self):
        return len(self.sft) + len(self.pencere)

    def __getitem__(self, idx):
        if idx < len(self.sft):
            x, y, sm = self.sft[idx]
            return (x, y, sm, torch.tensor(0.0))
        return self.pencere[idx - len(self.sft)]


def kaydet(model, hedef):
    gecici = hedef + ".tmp"
    torch.save(model.state_dict(), gecici)
    os.replace(gecici, hedef)  # atomik (train.py:646-648 deseni)


def main():
    args = parse_args()
    if args.teacher_model is None:
        args.teacher_model = args.init_model

    t_program_basla = time.time()  # ön-hesap (sözlük+yükleme+tokenizasyon+val) ölçümü

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("P26_DUR: MPS yok — koşum sandbox DIŞI olmalı (T-0209 kuralı)")
    pilot = args.max_opt_steps is not None
    print(f"=== P2.6 SFT-REPLAY (T-0209 kalıbı) | cihaz={device} | pilot={pilot} ===", flush=True)
    if device.type == "mps":
        print(f"[MPS] önerilen çalışma-seti sınırı: {torch.mps.recommended_max_memory() / 2**30:.2f} GB", flush=True)

    check_frozen_save_path(args.save_path, allow_frozen_write=args.allow_frozen_write)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    lexicon = LexiconManager(); lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary(); vocab.load(args.vocab)
    tokenizer = KristalTokenizer(compiler, vocab)
    vocab_size = len(vocab.stoi)

    for yol in (args.init_model, args.teacher_model, args.sft_path, args.pencere_bin):
        if not os.path.exists(yol):
            print(f"P26_DUR: bulunamadı: {yol}", flush=True); return 2

    # Sınır kapısı (train.py T-0080 deseni): pencere bin'inin maks id'si sözlüğün altında olmalı
    mm = np.memmap(args.pencere_bin, dtype=np.uint16, mode="r")
    if int(mm.max()) >= vocab_size:
        raise RuntimeError(f"P26_DUR: pencere bin maks id {int(mm.max())} >= vocab_size {vocab_size}")
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

    recs = [json.loads(l) for l in open(args.sft_path, encoding="utf-8") if l.strip()]
    if len(recs) != 4498:
        print(f"UYARI: SFT kayıt sayısı {len(recs)} != 4498 (T-0209 sabiti)", flush=True)
    val_records, train_records = recs[-150:], recs[:-150]

    val_dataset = FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=BLOCK)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    karisim = KarisimDataset(train_records, tokenizer, vocab, model,
                             args.pencere_bin, args.pencere_n, device, args.seed)
    n_toplam = len(karisim)
    steps_per_epoch = math.ceil(n_toplam / (args.batch * args.accum))
    total_opt = steps_per_epoch * args.epochs
    print(f"SFT train: {len(karisim.sft)} | pencere: {len(karisim.pencere)} | toplam {n_toplam}"
          f" -> {steps_per_epoch} opt-adım/epok, {total_opt} toplam", flush=True)

    init_ce, init_ppl = evaluate_val_loss(model, val_loader)
    init_pencere = pencere_ce_kulliyat(model, args.pencere_bin, device)
    print(f"[BAŞLANGIÇ] masked SFT val CE: {init_ce:.6f} | pencere-CE (kulliyat_v2): {init_pencere:.4f}", flush=True)
    on_hesap_sn = time.time() - t_program_basla
    print(f"[ÖN-HESAP] sözlük+model/öğretmen yükleme+tokenizasyon+başlangıç değerlendirmeleri: "
          f"{on_hesap_sn:.1f} sn", flush=True)
    if device.type == "mps":
        torch.mps.empty_cache()  # değerlendirme transient'lerini sürücüye bırak (t0200 R4; docstring not-6)

    optimizer = optim.AdamW(model.parameters(), lr=args.min_lr, weight_decay=args.weight_decay)
    model.train(); optimizer.zero_grad()
    opt_step, accum_sayac, running = 0, 0, 0.0
    adim_sureleri, history, kesildi = [], [], False
    # KL-kanıt sayaçları (B1 düzeltmesinin ateşlendiğinin çıktıdan okunabilir kanıtı)
    kl_aktif_mikro, kl_toplam_mikro, kl_degerleri = 0, 0, []
    adim_kl_degerleri = []
    step_t0 = None  # gerçek adım-süresi ölçümü: opt-adımın ilk mikro-partisinde alınır
    t_basla = time.time()

    for epok in range(args.epochs):
        train_loader = DataLoader(karisim, batch_size=args.batch, shuffle=True)
        for batch in train_loader:
            if step_t0 is None:
                step_t0 = time.time()  # opt-adımın BAŞI (accum penceresinin ilk mikro-partisi)
            x, y, sm, kaynak = batch   # kaynak: 0=SFT, 1=pencere (T-0209 DÜZELTMESİ)
            kl_toplam_mikro += 1
            x = x.to(device); y = y.to(device); sm = sm.to(device)
            # MPS-BELLEK (docstring not-7): mikro-parti SUB_CHUNK satırlık alt-yığınlarla
            # ileri+geri yayılır. Kayıp formu BİREBİR korunur — payda (sm.sum()+1e-8) TAM
            # partiden bir kez alınır; alt-yığın kayıplarının toplamı tam-parti
            # (T-0209 ce_loss + beta*kl) DEĞERİNE VE GRADYANINA eşittir (toplam lineerdir).
            sm_payda = sm.sum() + 1e-8
            # P2.6-B1 DÜZELTMESİ (beyan): T-0209'da satır-seçim 'len(batch)==4' ile yapılıyordu;
            # default_collate 4'lü örnekleri HER ZAMAN 4 elemanlı listeye yığdığından koşul her
            # partide doğru çıktı → is_pencere hep False → BETA=0.2'lik KL dalı ÖLÜ KODDU
            # (ölçüldü; marangoz varyantında da aynısı). Burada seçim KAYNAK tensorüyle
            # (4. eleman: 0=SFT, 1=pencere) yapılır → KL yalnız PENCERE satırlarında gerçekten
            # ateşlenir. t0200 kl_pencere_kayip deseni: forward-KL T=1, öğretmen no_grad,
            # CPU'da nonzero + cihazda index_select (MPS SIGSEGV önlemi), del öğretmen-logit.
            satirlar = ((kaynak == 1.0).nonzero(as_tuple=True)[0] if args.kl_beta > 0.0
                        else torch.empty(0, dtype=torch.long))  # CPU'da nonzero (t0200 koşum-3 onarımı)
            kl_satir_sayisi = int(satirlar.numel())      # mikro-parti GENELİ pencere satır sayısı
            T_poz = x.size(1)
            kl_aktif_alt, running_kl, running_mikro = 0, 0.0, 0.0
            for a0 in range(0, x.size(0), SUB_CHUNK):
                x_c = x[a0:a0 + SUB_CHUNK]; y_c = y[a0:a0 + SUB_CHUNK]; sm_c = sm[a0:a0 + SUB_CHUNK]
                logits, _ = model(x_c)
                V = logits.size(-1)
                ce_none = F.cross_entropy(logits.reshape(-1, V), y_c.reshape(-1), reduction="none").view_as(sm_c)
                loss = (ce_none * sm_c).sum() / sm_payda  # T-0209 BİREBİR; -100 hedef CE'de 0
                if kl_satir_sayisi > 0:
                    # Bu alt-yığının pencere satırları (global seçim → yerel indeks):
                    yerel = satirlar[(satirlar >= a0) & (satirlar < a0 + SUB_CHUNK)] - a0
                    if yerel.numel() > 0:
                        # KL satırları KL_CHUNK'lık yığınlarla işlenir (MPS-bellek, docstring
                        # not-6): yığın satır-toplamları tam-seçimin satır-toplamlarıyla AYNI
                        # küme; pay = mikro-partinin TOPLAM pencere satır sayısı × T_poz
                        # → mikro-parti geneli kl, t0200 .sum(-1).mean() ile BİREBİR.
                        kl_parca = []
                        for k0 in range(0, yerel.numel(), KL_CHUNK):
                            idx = yerel[k0:k0 + KL_CHUNK].to(device)
                            with torch.no_grad():
                                hoca_logits, _ = hoca(torch.index_select(x_c, 0, idx))
                                log_p_t = F.log_softmax(hoca_logits, dim=-1)
                                del hoca_logits  # MPS baskısı: öğretmen referansını hemen bırak (t0200 R4)
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
                kl_v = running_kl / (kl_satir_sayisi * T_poz)  # mikro-parti geneli KL (BİREBİR)
                adim_kl_degerleri.append(kl_v)
                kl_degerleri.append(kl_v)

            if accum_sayac == args.accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                lr = get_lr(opt_step, total_opt, args.peak_lr, args.min_lr, args.warmup)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step(); optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()  # MPS-bellek: adım-arası fragmantasyon önlemi (docstring not-6)
                # GERÇEK ADIM-SÜRESİ (taslak yer tutucusunun doldurulmuş hali): opt-adım başında
                # alınan step_t0 ile fark; baskı/checkpoint süresi ölçüme girmez.
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
                    mps_gb = (f" | MPS {torch.mps.current_allocated_memory() / 2**30:.2f}GB"
                              if device.type == "mps" else "")
                    print(f"Adım {opt_step}/{total_opt} | Loss {step_loss:.4f} | LR {lr:.2e} | "
                          f"adım-süresi ort {ort:.2f}s (son {len(son)}){mps_gb} | "
                          f"KL-aktif {kl_aktif_mikro}/{kl_toplam_mikro} mikro-parti", flush=True)
                if args.save_every and opt_step % args.save_every == 0 and opt_step != total_opt:
                    kaydet(model, args.save_path)
                    print(f"  [ckpt] adım {opt_step}: '{args.save_path}' güncellendi (ara, atomik)", flush=True)
                if pilot and opt_step >= args.max_opt_steps:
                    kesildi = True
                    break
        if not pilot and args.epochs > 1 and epok + 1 < args.epochs:
            kaydet(model, args.save_path + f"_e{epok + 1}.pt")
            print(f"  [ckpt] epok {epok + 1} sonu: '{args.save_path}_e{epok + 1}.pt' "
                  f"(NOT: cosine ortası — tam anneal edilmedi)", flush=True)
        if kesildi:
            break

    toplam_sure = time.time() - t_basla

    kaydet(model, args.save_path)
    final_ce, final_ppl = evaluate_val_loss(model, val_loader)
    final_pencere = pencere_ce_kulliyat(model, args.pencere_bin, device)
    print(f"\n[BİTİŞ] masked SFT val CE: {init_ce:.6f} -> {final_ce:.6f} | "
          f"pencere-CE: {init_pencere:.4f} -> {final_pencere:.4f} | Süre: {toplam_sure:.1f} sn", flush=True)
    kl_ort = (sum(kl_degerleri) / len(kl_degerleri)) if kl_degerleri else None
    if args.kl_beta > 0.0:
        kanit = (f"[KL-KANIT] KL dalı {kl_aktif_mikro}/{kl_toplam_mikro} mikro-partide ateşlendi"
                 + (f" | tüm-koşum KL ort {kl_ort:.6f}" if kl_ort is not None
                    else " | KL HİÇ ATEŞLENMEDİ (pencere satırı yok mu?)")
                 + " (öz-çapa: öğretmen==init → adım-0'da KL≈0 beklenir)")
    else:
        kanit = "[KL-KANIT] kl-beta=0 → KL kapalı (T-0209'un ölçülmüş KL'siz kalıbı)"
    print(kanit, flush=True)
    if pilot:
        son20 = adim_sureleri[-20:]
        ort20 = float(np.mean(son20)); std20 = float(np.std(son20))
        print(f"[PİLOT] {opt_step} opt-adım | son-20 s/adım: {ort20:.2f}±{std20:.2f} | "
              f"ETA kuralı: sürekli koşum için ~×3 ile beyan et "
              f"(170 adımlık tam koşum ≈ {ort20 * 3 * 170 / 60:.0f} dk)", flush=True)
    print(f"[MODEL_KAYDEDİLDİ] {args.save_path} (sha256: {compute_sha256(args.save_path)})", flush=True)

    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    hukum = {
        "gorev": "P2.6 SFT-replay", "pilot": pilot, "durum": "BASARILI",
        "init_model": args.init_model, "teacher_model": args.teacher_model,
        "kl_beta": args.kl_beta, "pencere_bin": args.pencere_bin, "pencere_n": args.pencere_n,
        "sft_train_n": len(karisim.sft), "val_n": len(val_dataset),
        "peak_lr": args.peak_lr, "min_lr": args.min_lr, "warmup": args.warmup,
        "epochs": args.epochs, "opt_adim": opt_step, "sure_sn": round(toplam_sure, 1),
        "on_hesap_sn": round(on_hesap_sn, 1),
        "kl_aktif_mikro": kl_aktif_mikro, "kl_toplam_mikro": kl_toplam_mikro,
        "kl_ort": kl_ort,
        "s_adim_son20_ort": round(float(np.mean(adim_sureleri[-20:])), 3) if adim_sureleri else None,
        "s_adim_son20_std": round(float(np.std(adim_sureleri[-20:])), 3) if adim_sureleri else None,
        "init_val_ce": init_ce, "final_val_ce": final_ce,
        "init_pencere_ce": init_pencere, "final_pencere_ce": final_pencere,
        "cikti": args.save_path, "cikti_sha256": compute_sha256(args.save_path),
    }
    with open(args.hukum_path, "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())