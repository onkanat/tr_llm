#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.5 FAZ-2A — LEGAL CEKETİ EĞİTİMİ (T-0209 marangoz kalıbı; P2.6 iskeletinden).

Kaynak iskelet: scripts/train_p26_sft_replay.py (T-0209 kayıp formu BİREBİR, MPS
mikro-parti, atomik kayıt, donmuş yol kapısı — hepsi orada hazır). P2.6 iskeletinden
bilinçli adaptasyonlar (görev P2.5 FAZ-2A şartnamesi):

1. BAŞLANGIÇ = data/anka_taban_pure.pt (CEKET TABANI pure'dur — P2.6 hükmü; v3
   ailesi DEĞİL). Pencere/KL dalı YOKTUR: ceket eğitimi saf alan SFT'idir.
   T-0209'un ÖLÇÜLMÜŞ davranışı zaten saf SFT'tir (KL dalı ölü koddu — P2.6-B1
   bulgusu); ölçülmemiş bir KL'yi burada uydurmak yerine dal tümüyle KALDIRILDI
   (ölü kod kopyalanmadı; satır-seçim düzeltmesi yalnız KL'li desen için gerekli).
2. VAL DİLİMİ: 463 kayıttan son-50 (T-0209 son-150 deseninin ölçekli karşılığı;
   kayıt başına ~174 token). train = ilk 413.
3. LR (T-0209 ceket tarifinden ÇIKARILDI, scripts/train_t0209_marangoz.py:56-64):
   peak_lr=1e-5, min_lr=2e-6, batch=16, accum=2, weight_decay=0.01, clip=1.0 —
   BİREBİR. WARMUP: marangoz 15/234 adım (%6,4) → ceil(413/32)=13 adım/epok × 3
   epok = 39 adımlık koşuma ORANSAL ölçekleme: 3.
4. EPOK: 3 (küçük set) + AŞIRI-EĞİTİM NÖBETİ: her epok sonunda masked SFT val CE
   ölçülür; önceki epoktan ARTTIYSA koşum durur ve en iyi epoğun ağırlıkları geri
   yüklenir (final ölçümleri GERÇEKTEN KAYDEDİLEN model üzerinde yapılır).
5. TOKENIZER: literal_entity_mode=True (FAZ-1 veri sözleşmesi —
   legal_canonical_sft.jsonl.meta.json 'literal_entity_mode': true; UNK %1,62 bu
   modda ölçüldü; mode=False apostrofları sessizce SİLER, hukuk metni apostrof-
   yoğundur). T-0209/p26 rig'i mode'suz kuruyordu — bilinçli fark, gerekçeli.
6. Kayıp formu T-0209'dan BİREBİR: ce = (CE_none*sm).sum()/(sm.sum()+1e-8);
   sm ±1 İŞARET maskesidir (kayıp maskesi DEĞİL), -100 hedefler CE'de 0 verir,
   prompt konumları PAYDADADIR (~0.5× seyreltme — ölçülen tarife sadakat).

MODLAR:
  --mod egitim       (varsayılan) taban çıpası + eğitim + değerlendirme + hüküm JSON
  --mod taban-cipasi yalnız taban çıpası (val CE + kanonik-20 + OOD-3) — kapı
                     dondurma öncesi çıpa ölçümü; eğitim YOK, model KAYDI YOK

DONMUŞ KAPILAR (koşumdan ÖNCE donduruldu; bkz. data/eval/p25_legal_ceket_kriterler_<tarih>.md):
  KAPI-1 (val CE)     : final_val_ce <= init_val_ce - 0.25 nat
  KAPI-2 (kanonik)    : ilk-16-jeton birebir (kart türü+kânun/kavram etiketi) oranı
                        >= 0.50 VE taban oranından katı yüksek
  KAPI-3 (OOD nöbeti) : 3 alan-dışı sorguda ort (CE_ceket - CE_taban) <= +0.15 nat
                        VE ilk-32 argmax üretim taban-uyumu >= 0.70
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
from src.llm.prompt_contract import render_prompt
from scripts.train_step_demo import KristalLM
from scripts.train_step_b1_5_rigorous import (FastSampleAlignedDataset,
                                              evaluate_val_loss, compute_sha256)
from scripts.t0200_mutasyon_kanit import yukle_checkpoint
from train import check_frozen_save_path  # donmuş yol kapısı (data/*.pt)

BLOCK = 128
SUB_CHUNK = 8  # MPS mikro-parti (P2.6 not-7): 16'lık parti 8'lik alt-yığınlarla
TARIH = time.strftime("%Y-%m-%d")

# --- DONMUŞ KAPI SABİTLERİ (koşum ÖNCESİ; betik yazımında donduruldu) ---
KAPI1_VAL_CE_DELTA = 0.25   # nat — "belirgin düşük" barı
KAPI2_ETIKET16_MIN = 0.50   # 20 örneklemde ilk-16 birebir (kart+etiket) oranı
KAPI3_CE_DELTA_MAX = 0.15   # nat — OOD kayıt CE'sinde kabul edilen artış tavanı
KAPI3_UYUM32_MIN = 0.70     # OOD ilk-32 argmax jeton taban-uyumu (ort)
KANONIK_N = 20
KANONIK_TOHUM = 42
URETIM_MAKS = 48            # argmax üretim tavanı (metrikler ilk-16/32'ye bakar)
BEKLENEN_KAYIT = 463

# Alan-sızıntısı nöbeti sorguları (DONMUŞ: dosya + kayıt indeksi):
# 2 marangoz + 1 bahçıvan — ceket legal'dir; alan dışı sorguda taban gibi
# davranmalı (üretim uyum + CE sapması ölçülür).
OOD_KAYNAKLAR = [
    ("data/pedagogy/marangoz_canonical_sft.jsonl", 0),
    ("data/pedagogy/marangoz_canonical_sft.jsonl", 1),
    ("data/pedagogy/bahcivan_canonical_sft.jsonl", 0),
]


def parse_args():
    p = argparse.ArgumentParser(description="P2.5 FAZ-2A legal ceketi (T-0209 marangoz tarifi)")
    p.add_argument("--mod", choices=["egitim", "taban-cipasi"], default="egitim")
    p.add_argument("--init-model", default="data/anka_taban_pure.pt")
    p.add_argument("--sft-path", default="data/pedagogy/legal_canonical_sft.jsonl")
    p.add_argument("--vocab", default="data/rebuild/vocab_anka_r1_33114.json")
    p.add_argument("--val-n", type=int, default=50, help="son-N val (T-0209 son-150'nin ölçekli karşılığı)")
    p.add_argument("--peak-lr", type=float, default=1e-5)  # marangoz: PEAK_LR=1e-5
    p.add_argument("--min-lr", type=float, default=2e-6)   # marangoz: MIN_LR=2e-6
    p.add_argument("--warmup", type=int, default=3)        # marangoz 15/234 (%6,4) → 39 adımda 3
    p.add_argument("--batch", type=int, default=16)       # marangoz: BATCH=16
    p.add_argument("--accum", type=int, default=2)         # marangoz: ACCUM=2
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--save-path", default="data/anka_legal.pt",
                   help="data/*.pt donmuş desen → --allow-frozen-write şart")
    p.add_argument("--allow-frozen-write", action="store_true")
    p.add_argument("--max-opt-steps", type=int, default=None, help="PİLOT: N opt-adımda kes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-path", default="data/eval/p25_legal_history.json")
    p.add_argument("--hukum-path", default=f"data/eval/p25_legal_ceket_hukum_{TARIH}.json")
    p.add_argument("--cipa-path", default=f"data/eval/p25_legal_taban_cipasi_{TARIH}.json")
    return p.parse_args()


def get_lr(step, total_steps, peak, min_lr, warmup):
    """Marangoz/p26 cosine çizelgesi (BİREBİR)."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def kaydet(model, hedef, allow_frozen):
    """Atomik kayıt (train.py:646-648 deseni) + donmuş yol kapısı her yazım için."""
    check_frozen_save_path(hedef, allow_frozen_write=allow_frozen)
    gecici = hedef + ".tmp"
    torch.save(model.state_dict(), gecici)
    os.replace(gecici, hedef)


def kur_tokenizer(vocab_yolu):
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(vocab_yolu)
    # FAZ-1 veri sözleşmesi: literal_entity_mode=True (meta.json beyanı; UNK %1,62
    # bu modda ölçüldü; mode=False apostrofları sessizce siler — hukuk metni
    # apostrof-yoğundur: "Kanunu'nda", "Sayıştay'a").
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True, verbose=False)
    return tokenizer, vocab


def olc_val_ce(model, val_kayitlari, tokenizer, vocab, batch=16):
    """Masked SFT val CE — FastSampleAlignedDataset rig'i (metrik kimliği:
    T-0209 2,3049 ile AYNI ölçüm: masked SFT val CE, pencere-CE DEĞİL)."""
    ds = FastSampleAlignedDataset(val_kayitlari, tokenizer, vocab, model, block_size=BLOCK)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    ce, ppl = evaluate_val_loss(model, loader)
    return ce, ppl, len(ds)


def kayit_ce(model, rec, tokenizer, vocab, device):
    """Tek kayıt masked CE (FastSampleAlignedDataset rig'i, birebir)."""
    ds = FastSampleAlignedDataset([rec], tokenizer, vocab, model, block_size=BLOCK)
    x, y, sm = ds[0]
    model.eval()
    with torch.no_grad():
        _, loss = model(x.unsqueeze(0).to(device),
                        targets=y.unsqueeze(0).to(device),
                        sign_mask=sm.unsqueeze(0).to(device))
    return float(loss.item())


def argmax_uret(model, tokenizer, vocab, inst, inp, device, maks=URETIM_MAKS):
    """Kanonik argmax üretimi (evaluate_carpenter_generation_100.py deseni)."""
    model.eval()
    eos = vocab.stoi.get("<EOS>", 3)
    out_end = vocab.stoi.get("</OUTPUT>", 9)
    ids = tokenizer.encode(render_prompt(inst, inp))
    if ids and ids[-1] == eos:
        ids = ids[:-1]
    curr = torch.tensor([ids], dtype=torch.long, device=device)
    gen = []
    with torch.no_grad():
        for _ in range(maks):
            sm = model.embedding.compute_sign_mask(curr.cpu()).to(device)
            logits, _ = model(curr, sign_mask=sm)
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            if nxt in (eos, out_end):
                break
            gen.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            if curr.size(1) >= 256:
                break
    return gen


def gold_govde_ids(tokenizer, vocab, out):
    """Altın yanıt gövde id'leri (BOS/EOS soyulmuş)."""
    ids = tokenizer.encode(out)
    bos = vocab.stoi.get("<BOS>", 2)
    eos = vocab.stoi.get("<EOS>", 3)
    if ids and ids[0] == bos:
        ids = ids[1:]
    if ids and ids[-1] == eos:
        ids = ids[:-1]
    return ids


def kanonik_dogruluk(model, tokenizer, vocab, recs, val_n, device,
                     tohum=KANONIK_TOHUM, n=KANONIK_N):
    """20 örneklemde argmax üretim vs altın gövde (jeton-id düzeyi).

    Ölçümlü metrik sabitleri (463 kayıt üzerinde tokenizasyon ölçümü):
      kart türü ilk-4 jetonda ayrışır ('kanun tanım POSS_3SG :' 4; 'mevzuat
      <ENT> <CAP> h ü' Hükmü-kartında 4; 'mevzuat yanıt POSS_3SG :' 4;
      'kavram ayrım POSS_3SG :' 4; 'prosedür …' 2);
      ilk-16 jeton kart türü + kânun/kavram kimliğini kapsar ('( kanun No' dahil).
    """
    idxs = sorted(random.Random(tohum).sample(range(len(recs)), min(n, len(recs))))
    satirlar = []
    for i in idxs:
        r = recs[i]
        gen = argmax_uret(model, tokenizer, vocab, r.get("instruction", ""), r.get("input", ""), device)
        gold = gold_govde_ids(tokenizer, vocab, r["output"])
        ortak = 0
        for a, b in zip(gen, gold):
            if a != b:
                break
            ortak += 1
        satirlar.append({
            "idx": i,
            "val_mi": i >= len(recs) - val_n,
            "kart4": bool(gold) and gen[:4] == gold[:4],
            "etiket16": bool(gold) and gen[:16] == gold[:16],
            "ortak_on": ortak,
            "gen_uzunluk": len(gen),
            "gold_uzunluk": len(gold),
        })
    ozet = {
        "tohum": tohum,
        "orneklem_idx": idxs,
        "n": len(satirlar),
        "valdeki": sum(1 for s in satirlar if s["val_mi"]),
        "kart4_oran": round(sum(s["kart4"] for s in satirlar) / len(satirlar), 4),
        "etiket16_oran": round(sum(s["etiket16"] for s in satirlar) / len(satirlar), 4),
        "ortak_on_ort": round(float(np.mean([s["ortak_on"] for s in satirlar])), 3),
    }
    return ozet, satirlar


def ood_sorgulari_yukle():
    sorgular = []
    for yol, idx in OOD_KAYNAKLAR:
        recs = [json.loads(l) for l in open(yol, encoding="utf-8") if l.strip()]
        sorgular.append({"kaynak": f"{yol}#{idx}", "rec": recs[idx]})
    return sorgular


def ood_olc(model, tokenizer, vocab, device, sorgular):
    """Her OOD sorgu: tek-kayıt masked CE + ilk-48 argmax üretim."""
    satirlar = []
    for s in sorgular:
        r = s["rec"]
        ce = kayit_ce(model, r, tokenizer, vocab, device)
        gen = argmax_uret(model, tokenizer, vocab, r.get("instruction", ""), r.get("input", ""), device)
        satirlar.append({"kaynak": s["kaynak"], "ce": round(ce, 6), "gen_ids": gen})
    return satirlar


def uyum32(g1, g2):
    """İlk-32 pozisyonda jeton-uyumu (kısa üretim kalan konumları uyumsuz sayar)."""
    m = sum(1 for a, b in zip(g1[:32], g2[:32]) if a == b)
    return m / 32


def main():
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("P25_DUR: MPS yok — koşum sandbox DIŞI olmalı (T-0209 kuralı)")
    pilot = args.max_opt_steps is not None
    print(f"=== P2.5 FAZ-2A LEGAL CEKETİ (T-0209 marangoz kalıbı) | cihaz={device} | "
          f"mod={args.mod} | pilot={pilot} ===", flush=True)
    print(f"[MPS] önerilen çalışma-seti sınırı: {torch.mps.recommended_max_memory() / 2**30:.2f} GB", flush=True)

    if args.mod == "egitim":
        check_frozen_save_path(args.save_path, allow_frozen_write=args.allow_frozen_write)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tokenizer, vocab = kur_tokenizer(args.vocab)
    vocab_size = len(vocab.stoi)

    for yol in (args.init_model, args.sft_path, args.vocab):
        if not os.path.exists(yol):
            print(f"P25_DUR: bulunamadı: {yol}", flush=True)
            return 2

    print(f"[PROVENANS] init: {args.init_model} sha256: {compute_sha256(args.init_model)}", flush=True)
    print(f"[PROVENANS] sft : {args.sft_path} sha256: {compute_sha256(args.sft_path)}", flush=True)
    print(f"[PROVENANS] vocab sha256: {compute_sha256(args.vocab)}", flush=True)

    recs = [json.loads(l) for l in open(args.sft_path, encoding="utf-8") if l.strip()]
    if len(recs) != BEKLENEN_KAYIT:
        print(f"UYARI: SFT kayıt sayısı {len(recs)} != {BEKLENEN_KAYIT} (FAZ-1 sabiti)", flush=True)
    val_records = recs[-args.val_n:]
    train_records = recs[:-args.val_n]
    print(f"SFT: {len(recs)} kayıt | train {len(train_records)} | val (son-{args.val_n}) {len(val_records)}", flush=True)

    # Tek model nesnesi: taban çıpası EĞİTİM ÖNCESİ aynı nesneyle ölçülür,
    # sonra aynı nesne eğitilir (başlangıç ağırlıkları = taban_pure).
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    yukle_checkpoint(args.init_model, model)
    model.to(device)

    t_cipa = time.time()
    # --- TABAN ÇIPASI (kapılar bu çıpaya göre donduruldu) ---
    init_ce, init_ppl, val_ds_n = olc_val_ce(model, val_records, tokenizer, vocab, batch=args.batch)
    kanon_taban, kanon_satir_taban = kanonik_dogruluk(model, tokenizer, vocab, recs, args.val_n, device)
    sorgular = ood_sorgulari_yukle()
    ood_taban = ood_olc(model, tokenizer, vocab, device, sorgular)
    print(f"[ÇIPA-TABAN] legal val CE: {init_ce:.6f} | PPL {init_ppl:.2f} | val_ds {val_ds_n} "
          f"| çıpa süresi {time.time() - t_cipa:.1f} sn", flush=True)
    print(f"[ÇIPA-TABAN] kanonik-20: kart4 {kanon_taban['kart4_oran']:.2f} | "
          f"etiket16 {kanon_taban['etiket16_oran']:.2f} | ortak-önek {kanon_taban['ortak_on_ort']:.1f} | "
          f"val'deki {kanon_taban['valdeki']}/{kanon_taban['n']}", flush=True)
    for s in ood_taban:
        print(f"[ÇIPA-TABAN] OOD {s['kaynak']} | CE {s['ce']:.4f} | gen-uzunluk {len(s['gen_ids'])}", flush=True)
    if device.type == "mps":
        torch.mps.empty_cache()

    if args.mod == "taban-cipasi":
        cipa = {
            "gorev": "P2.5 FAZ-2A taban çıpası (kapı dondurma ölçümü)",
            "init_model": args.init_model,
            "init_model_sha256": compute_sha256(args.init_model),
            "sft_path": args.sft_path,
            "tokenizer_modu": "literal_entity_mode=True",
            "val_ce": init_ce, "val_ppl": init_ppl, "val_ds_n": val_ds_n,
            "kanonik": kanon_taban, "kanonik_satirlar": kanon_satir_taban,
            "ood": [{"kaynak": s["kaynak"], "ce": s["ce"], "gen_ids": s["gen_ids"]} for s in ood_taban],
        }
        with open(args.cipa_path, "w", encoding="utf-8") as f:
            json.dump(cipa, f, indent=2, ensure_ascii=False)
        print(f"[ÇIPA YAZILDI] {args.cipa_path}", flush=True)
        return 0

    # ================= EĞİTİM =================
    t_on_hesap = time.time()
    train_dataset = FastSampleAlignedDataset(train_records, tokenizer, vocab, model, block_size=BLOCK)
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch * args.accum))
    total_opt = steps_per_epoch * args.epochs
    print(f"train_ds: {len(train_dataset)} örnek -> {steps_per_epoch} opt-adım/epok, "
          f"{total_opt} toplam (batch {args.batch} × accum {args.accum})", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.min_lr, weight_decay=args.weight_decay)
    model.train(); optimizer.zero_grad()
    opt_step, accum_sayac, running = 0, 0, 0.0
    adim_sureleri, history = [], []
    step_t0 = None
    epoch_val_ces, best_val_ce, best_epok, best_state_cpu = [], None, None, None
    kesildi, dur_nedeni = False, None
    t_basla = time.time()

    for epok in range(args.epochs):
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        for batch in train_loader:
            if step_t0 is None:
                step_t0 = time.time()
            x, y, sm = batch
            x = x.to(device); y = y.to(device); sm = sm.to(device)
            # P2.6 not-7 (BİREBİR): payda TAM partiden bir kez; alt-yığın kayıpları
            # toplamı tam-parti T-0209 ce_loss değerine VE gradyanına eşittir.
            sm_payda = sm.sum() + 1e-8
            running_mikro = 0.0
            for a0 in range(0, x.size(0), SUB_CHUNK):
                x_c = x[a0:a0 + SUB_CHUNK]; y_c = y[a0:a0 + SUB_CHUNK]; sm_c = sm[a0:a0 + SUB_CHUNK]
                logits, _ = model(x_c)  # p26 çağrı deseni (sign_mask fallback — BİREBİR)
                V = logits.size(-1)
                ce_none = F.cross_entropy(logits.reshape(-1, V), y_c.reshape(-1),
                                          reduction="none").view_as(sm_c)
                loss = (ce_none * sm_c).sum() / sm_payda  # T-0209 BİREBİR; -100 CE'de 0
                (loss / args.accum).backward()
                running_mikro += loss.item()
                del logits, loss, ce_none
            running += running_mikro; accum_sayac += 1
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
                history.append({"adim": opt_step, "loss": step_loss, "lr": lr, "epok": epok + 1})
                if opt_step % 5 == 0 or opt_step in (1, total_opt):
                    ort = float(np.mean(adim_sureleri[-5:]))
                    print(f"Adım {opt_step}/{total_opt} | Loss {step_loss:.4f} | LR {lr:.2e} | "
                          f"adım-süresi ort {ort:.2f}s | epok {epok + 1}", flush=True)
                if args.save_every and opt_step % args.save_every == 0 and opt_step != total_opt:
                    kaydet(model, args.save_path, args.allow_frozen_write)
                    print(f"  [ckpt] adım {opt_step}: '{args.save_path}' güncellendi (ara, atomik)", flush=True)
                if pilot and opt_step >= args.max_opt_steps:
                    kesildi = True
                    break
        if kesildi:
            break

        if pilot:
            continue

        # --- EPOK SONU: val CE + aşırı-eğitim nöbeti + ara kayıt ---
        ep_ce, ep_ppl, _ = olc_val_ce(model, val_records, tokenizer, vocab, batch=args.batch)
        epoch_val_ces.append(round(ep_ce, 6))
        history.append({"epok_sonu": epok + 1, "val_ce": ep_ce, "val_ppl": ep_ppl})
        print(f"[EPOK {epok + 1}] val CE {ep_ce:.6f} | PPL {ep_ppl:.2f} | "
              f"epok-önceki: {epoch_val_ces[-2] if len(epoch_val_ces) > 1 else '—'}", flush=True)
        if best_val_ce is None or ep_ce < best_val_ce:
            best_val_ce, best_epok = ep_ce, epok + 1
            best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if len(epoch_val_ces) >= 2 and epoch_val_ces[-1] > epoch_val_ces[-2]:
            dur_nedeni = (f"val CE artışı: epok {epok + 1} {epoch_val_ces[-1]:.6f} > "
                          f"epok {epok} {epoch_val_ces[-2]:.6f}")
            print(f"[NÖBET] {dur_nedeni} — koşum durduruldu", flush=True)
            kesildi = True
        kaydet(model, args.save_path + f"_e{epok + 1}.pt", args.allow_frozen_write)
        print(f"  [ckpt] epok {epok + 1}: '{args.save_path}_e{epok + 1}.pt'", flush=True)
        if kesildi:
            break

    toplam_sure = time.time() - t_basla

    # En iyi epoğa dönüş (nöbet ya da son-epok-best-değilse)
    restore_edildi = False
    if best_state_cpu is not None and best_epok is not None and best_epok != len(epoch_val_ces):
        model.load_state_dict(best_state_cpu)
        model.to(device)
        restore_edildi = True
        print(f"[GERİ-YÜKLE] en iyi epok {best_epok} ağırlıkları geri yüklendi "
              f"(val CE {best_val_ce:.6f})", flush=True)
    del best_state_cpu
    if device.type == "mps":
        torch.mps.empty_cache()

    kaydet(model, args.save_path, args.allow_frozen_write)
    print(f"[MODEL_KAYDEDİLDİ] {args.save_path} (sha256: {compute_sha256(args.save_path)})", flush=True)

    # ================= DEĞERLENDİRME (donmuş kapılar) =================
    final_ce, final_ppl, _ = olc_val_ce(model, val_records, tokenizer, vocab, batch=args.batch)
    kanon_ceket, kanon_satir_ceket = kanonik_dogruluk(model, tokenizer, vocab, recs, args.val_n, device)
    ood_ceket = ood_olc(model, tokenizer, vocab, device, sorgular)
    print(f"[CEKET] legal val CE: {final_ce:.6f} | PPL {final_ppl:.2f}", flush=True)
    print(f"[CEKET] kanonik-20: kart4 {kanon_ceket['kart4_oran']:.2f} | "
          f"etiket16 {kanon_ceket['etiket16_oran']:.2f} | ortak-önek {kanon_ceket['ortak_on_ort']:.1f}", flush=True)

    ood_karsilastirma, uyumlar, ce_deltalari = [], [], []
    for st, sc in zip(ood_taban, ood_ceket):
        u = uyum32(st["gen_ids"], sc["gen_ids"])
        d = sc["ce"] - st["ce"]
        uyumlar.append(u); ce_deltalari.append(d)
        ood_karsilastirma.append({
            "kaynak": st["kaynak"], "ce_taban": st["ce"], "ce_ceket": sc["ce"],
            "ce_delta": round(d, 6), "uyum32": round(u, 4),
            "gen_taban_uzunluk": len(st["gen_ids"]), "gen_ceket_uzunluk": len(sc["gen_ids"]),
        })
        print(f"[OOD] {st['kaynak']} | CE {st['ce']:.4f} -> {sc['ce']:.4f} (Δ {d:+.4f}) | "
              f"uyum32 {u:.2f}", flush=True)

    # --- KAPI HESAPLARI (donmuş sabitlerle) ---
    kapi1 = {
        "kural": f"final_val_ce ({final_ce:.6f}) <= init_val_ce ({init_ce:.6f}) - {KAPI1_VAL_CE_DELTA}",
        "delta": round(init_ce - final_ce, 6),
        "gecti": bool(final_ce <= init_ce - KAPI1_VAL_CE_DELTA),
    }
    kapi2 = {
        "kural": f"etiket16_oran ({kanon_ceket['etiket16_oran']:.2f}) >= {KAPI2_ETIKET16_MIN} "
                 f"VE > taban ({kanon_taban['etiket16_oran']:.2f})",
        "gecti": bool(kanon_ceket["etiket16_oran"] >= KAPI2_ETIKET16_MIN and
                      kanon_ceket["etiket16_oran"] > kanon_taban["etiket16_oran"]),
    }
    uyum_ort = float(np.mean(uyumlar)) if uyumlar else 0.0
    ce_delta_ort = float(np.mean(ce_deltalari)) if ce_deltalari else 0.0
    kapi3 = {
        "kural": f"ort CE delta ({ce_delta_ort:+.4f}) <= +{KAPI3_CE_DELTA_MAX} VE "
                 f"ort uyum32 ({uyum_ort:.2f}) >= {KAPI3_UYUM32_MIN}",
        "ce_delta_ort": round(ce_delta_ort, 6), "uyum32_ort": round(uyum_ort, 4),
        "gecti": bool(ce_delta_ort <= KAPI3_CE_DELTA_MAX and uyum_ort >= KAPI3_UYUM32_MIN),
    }
    hep_si = all(k["gecti"] for k in (kapi1, kapi2, kapi3))
    print(f"\n[KAPI-1 val CE] {'GEÇTİ' if kapi1['gecti'] else 'KALDI'} — Δ {kapi1['delta']:+.4f} nat "
          f"(bar ≥ {KAPI1_VAL_CE_DELTA})", flush=True)
    print(f"[KAPI-2 kanonik] {'GEÇTİ' if kapi2['gecti'] else 'KALDI'} — {kapi2['kural']}", flush=True)
    print(f"[KAPI-3 OOD nöbeti] {'GEÇTİ' if kapi3['gecti'] else 'KALDI'} — {kapi3['kural']}", flush=True)
    print(f"[HÜKÜM] {'GEÇTİ' if hep_si else 'KALDI'}", flush=True)
    print(f"[SÜRE] eğitim {toplam_sure:.1f} sn | {opt_step} opt-adım | "
          f"s/adım son-20: {float(np.mean(adim_sureleri[-20:])):.2f}±"
          f"{float(np.std(adim_sureleri[-20:])):.2f}", flush=True)

    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    hukum = {
        "gorev": "P2.5 FAZ-2A legal ceketi (T-0209 marangoz kalıbı)",
        "mod": args.mod, "pilot": pilot,
        "durum": "BASARILI" if hep_si else "KAPILAR_KALDI",
        "hukum": "GECTI" if hep_si else "KALDI",
        "tarif": {
            "init_model": args.init_model,
            "pencere_karisim": "YOK (saf alan SFT — T-0209'un ölçülmüş davranışı; KL dalı kaldırıldı)",
            "peak_lr": args.peak_lr, "min_lr": args.min_lr, "warmup": args.warmup,
            "warmup_gerekce": "marangoz 15/234 adım (%6,4) → 39 adımda oransal 3",
            "batch": args.batch, "accum": args.accum, "weight_decay": args.weight_decay,
            "clip": args.clip, "epochs": args.epochs, "block": BLOCK, "sub_chunk": SUB_CHUNK,
            "tokenizer_modu": "literal_entity_mode=True (FAZ-1 sözleşmesi)",
        },
        "veri": {"sft_path": args.sft_path, "kayit": len(recs), "val_n": args.val_n,
                 "train_n": len(train_records), "train_ds_n": len(train_dataset)},
        "provenans": {
            "init_sha256": compute_sha256(args.init_model),
            "sft_sha256": compute_sha256(args.sft_path),
            "vocab_sha256": compute_sha256(args.vocab),
            "cikti": args.save_path,
            "cikti_sha256": compute_sha256(args.save_path),
        },
        "cipa_taban": {"val_ce": init_ce, "val_ppl": init_ppl,
                       "kanonik": kanon_taban,
                       "ood": [{"kaynak": s["kaynak"], "ce": s["ce"]} for s in ood_taban]},
        "egitim": {
            "sure_sn": round(toplam_sure, 1), "opt_adim": opt_step,
            "s_adim_son20_ort": round(float(np.mean(adim_sureleri[-20:])), 3) if adim_sureleri else None,
            "s_adim_son20_std": round(float(np.std(adim_sureleri[-20:])), 3) if adim_sureleri else None,
            "epoch_val_ces": epoch_val_ces, "dur_nedeni": dur_nedeni,
            "en_iyi_epok": best_epok, "geri_yuklendi": restore_edildi,
        },
        "degerlendirme": {
            "final_val_ce": final_ce, "final_val_ppl": final_ppl,
            "val_ce_delta": round(init_ce - final_ce, 6),
            "kanonik_ceket": kanon_ceket, "kanonik_satirlar_ceket": kanon_satir_ceket,
            "kanonik_satirlar_taban": kanon_satir_taban,
            "ood_karsilastirma": ood_karsilastirma,
        },
        "kapilar": {"kapi1_val_ce": kapi1, "kapi2_kanonik": kapi2, "kapi3_ood": kapi3},
        "kapi_sabitleri": {
            "kapi1_val_ce_delta": KAPI1_VAL_CE_DELTA, "kapi2_etiket16_min": KAPI2_ETIKET16_MIN,
            "kapi3_ce_delta_max": KAPI3_CE_DELTA_MAX, "kapi3_uyum32_min": KAPI3_UYUM32_MIN,
        },
        "history_path": args.history_path,
    }
    with open(args.hukum_path, "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2, ensure_ascii=False)
    print(f"[HÜKÜM JSON] {args.hukum_path}", flush=True)
    return 0 if hep_si else 1


if __name__ == "__main__":
    sys.exit(main())