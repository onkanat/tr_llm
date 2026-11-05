#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 ÖĞRENCİ — külliyat bilgisi + SFT okuryazarlık çapası TEK koşumda (T-0209 felsefesi).

FELSEFE ZEMİNİ (skill dersi 'TEMEL MODEL CEKETLERE GÖRE BÜKÜLMEZ', 05.11.2026):
T3.4'ün saf-korpus koşumu İHLALDİ (okuryazarlık 0,9998→0,0726 çöktü; pencere-CE
3,8205→3,1936 kazancı gerçekti ama okuryazarlık feda edildi). P2.6'nın 5-tur
SFT-replay merdiveni ihlalin başarısız geri-almasıydı (2,30 CE'ye asla dönmedi).
v4 öğrenci tarifi ikisini BAŞTAN işlemez: bilgi ile okuryazarlık AYNI ANDA,
TEK karışımda; öğrenci sınıf atlar ama okuryazarlığını ASLA kaybetmez.

TARİF (ölçülmüş zemin; kanıt: data/eval/p26_kapı_olcum_2026-11-04.md):
- Başlangıç VE KL öğretmeni: data/anka_taban_pure.pt (öz-çapa; adım-0'da KL≈0).
  KL satır-seçimi kipe göre: tum → tüm model pure'a çıpalanır; sft (v4b) →
  yalnız SFT satırları; pencere (p26) → yalnız kaynak==1 satırları.
- Veri karışımı (token-bazlı, ölçüm: scratch/v4_zemin_probe.py 05.11.2026):
  pure_base_sft_v4 kayıt başına GERÇEK hedef konumu ort 52,86 → mikro-parti
  (16 satır) 10 pencere + 6 SFT satırı = hedef-konum bazlı %80,1 külliyat /
  %19,9 SFT. Pencere satırı 128 hedef taşır; 1280/(1280+317,2)=0,801.
  Pencereler kulliyat_v2.bin'den RASTGELE (memmap; epok başına permütasyon,
  tekrarsız → 1 epok = külliyatın ~tümü bir kez). SFT çapası her turda yeniden
  karıştırılarak SONSUZ döngü (tekrarlı çapa; ~5.585 kayıt / 12 SFT-satır-adım
  ≈ 465 adımda tam tur).
- Kayıp formu p26'dan BİREBİR (bilinçli): ce = (CE_none×sm).sum()/(sm.sum()+1e-8);
  sm ±1 İŞARET maskesidir (kayıp maskesi değil), -100 hedefler CE'de 0, prompt
  konumları paydada (seyreltme ölçümü: hedef/sm ≈ 0,417 — v4'te çapa ~25 tur
  tekrarlandığından underfit riski p26 dar-set dersiyle AŞILIR).
- KL: forward-KL T=1, öğretmen no_grad, β=0,2; mikro-parti SUB_CHUNK=8 (MPS OOM
  dersi), öğretmen satırları KL_CHUNK=4 yığınlarla; satır-seçim KAYNAK tensorüyle
  (p26 B1 düzeltmesi; T-0209 'len(batch)==4' ölü-kod dersinin tekrarı YOK).
- KL KİPİ (v4b): --kl-kip {tum,sft,pencere}, satır-seçim TEK NOKTA kl_satirlari_sec.
  "tum" = v4 (TÜM satır → bilgi ekseni bloklandı: pencere-CE 3,96→4,15, resmi VAL
  3,905). "sft" = v4b (KL YALNIZ kaynak==0 SFT satırlarında; pencere satırları
  ÇIPASIZ → bilgi serbest, okuryazarlık çıpalı). "pencere" = p26. Geriye uyum:
  --no-kl-tum-satirlar → pencere; varsayılan aynen tum.
- lr: v4b peak 1e-4 ORTA NOKTA (v4 kanıtı: 5e-5 bilgiyi blokladı — KALDI; T3.4
  kanıtı: 2e-4 okuryazarlığı öldürdü — İHLAL; 1e-4 kanıtsız, K1 erken-duruş
  sigortasıyla), warmup 200, cosine → min 1e-5.
- VAL-NÖBETİ (her --save-every'de): SFT val-CE (val-150, FastSampleAlignedDataset
  — TABAN 2,3043 rig'i) + külliyat pencere-CE (TRAIN bin, 25 pencere, tohum 7 —
  p26 avlusu; resmi VAL avlusu eval_ce_val.py'de AYRI) + paket güveni (tek görev
  'okula'→'case'; TABAN 0,9998 referansı) + kayıp eğrisi (history).
- K1 ERKEN-DURUŞ İZNİ: toplam adımın ilk %20'sindeki nöbetlerde SFT val CE 2,45'i
  AŞARSA koşum DURUR (İLAN kapısı K1'in canlı tripwire'ı).
- Pencere tavanı 128 (aile dersi); KristalLM block_size=4096 (p26 deseni).

PİLOT modu: --max-opt-steps N → N opt-adımda kes, yine de nöbet/val/final-ölçüm
koşar; ETA kuralı pilot s/adım × 3 (sürekli-koşum çarpanı; T3.4: pilot 1,18 →
gerçek 3,27 sn/adım) × --toplam-adim.

Kullanım (PİLOT — scratch'e kaydeder, donmuş yol YOK):
  venv/bin/python scripts/train_v4_ogrenci.py \
    --init-model data/anka_taban_pure.pt \
    --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
    --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
    --kl-kip sft --kl-beta 0.2 --peak-lr 1e-4 --min-lr 1e-5 --warmup 200 \
    --save-every 25 --max-opt-steps 50 --toplam-adim 11380 \
    --save-path scratch/v4b_ogrenci_pilot.pt \
    --history-path scratch/v4b_ogrenci_pilot_history.json \
    --hukum-path scratch/v4b_ogrenci_pilot_hukum.json

TAM koşum (İLAN onayı SONRASI; data/ çıktısı için --allow-frozen-write):
  venv/bin/python scripts/train_v4_ogrenci.py \
    --init-model data/anka_taban_pure.pt \
    --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
    --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
    --kl-kip sft --kl-beta 0.2 --peak-lr 1e-4 --min-lr 1e-5 --warmup 200 \
    --batch 16 --accum 2 --pencere-satir 10 \
    --toplam-adim 11380 --save-every 500 \
    --save-path data/anka_ogrenci_v4b.pt --allow-frozen-write \
    --history-path data/eval/v4b_ogrenci_history.json \
    --hukum-path data/eval/v4b_ogrenci_hukum_2026-11-05.json
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
from scripts.t0200_mutasyon_kanit import yukle_checkpoint
from scripts.train_p26_sft_replay import (BLOCK, KL_CHUNK, SUB_CHUNK,
                                          pencere_ce_kulliyat, get_lr, kaydet)
from train import check_frozen_save_path

TARIH = time.strftime("%Y-%m-%d")
K1_ESIK = 2.45          # İLAN donmuş kapısı: SFT val CE üst-sınırı (pure 2,3043 bandı)
K1_IZLEME_ORANI = 0.20  # erken-duruş izni: toplam adımın ilk %20'si


# ----------------------------------------------------------------------------
# Çekirdek birimler (tests/test_v4_ogrenci.py buradan sınar; ağır model GEREKTİRMEZ)
# ----------------------------------------------------------------------------

def planla_pencere_baslar(n_token, adet, seed, epok, block=BLOCK):
    """Külliyat pencere başlangıçları: blok-hizalı, TEKRARSIZ (permütasyon), deterministik.

    PencereDataset sınırı birebir: b ∈ [0, n_win-2] (b*block+block+1 ≤ n_token).
    Tohum-dizisi [seed, epok]: epok başına bağımsız ama yeniden üretilebilir akış.
    Dönüş: int64 ndarray (blok-hizalı başlangıçlar), len ≤ min(adet, kullanılabilir).
    """
    n_win = int(n_token) // block
    kullanilabilir = max(1, n_win - 1)
    rng = np.random.default_rng([int(seed), int(epok)])
    secim = rng.permutation(kullanilabilir)[:min(int(adet), kullanilabilir)]
    return (secim * block).astype(np.int64)


def kl_satirlari_sec(kaynak, beta, tum_satirlar=True, kip=None):
    """KL satır-seçimi — TEK NOKTA (p26 B1 + v4 tüm-satır + v4b SFT kipi).

    Kipler: "tum" (v4, varsayılan) → TAM parti: öğrenci tümüyle pure'a çıpalanır.
    "sft" (v4b) → yalnız kaynak==0 (SFT) satırları: okuryazarlık çapası çıpalı,
    pencere (bilgi) satırları ÇIPASIZ — bilgi gradyanı KL ile bloklanamaz.
    "pencere" (p26) → yalnız kaynak==1 satırları. tum_satirlar geriye-uyum
    kısayolu (True→tum, False→pencere); kip AÇIK verildiyse O kazanır.
    beta<=0 → boş (KL kapalı; kip ne olursa olsun). CPU tensoründe nonzero
    (t0200 koşum-3 MPS SIGSEGV önlemi).
    """
    if beta <= 0.0:
        return torch.empty(0, dtype=torch.long)
    if kip is None:
        kip = "tum" if tum_satirlar else "pencere"
    if kip == "tum":
        return torch.arange(int(kaynak.numel()), dtype=torch.long)
    if kip == "sft":
        return (kaynak == 0.0).nonzero(as_tuple=True)[0].to(torch.long)
    if kip == "pencere":
        return (kaynak == 1.0).nonzero(as_tuple=True)[0].to(torch.long)
    raise ValueError(f"bilinmeyen KL kipi: {kip!r} (tum|sft|pencere)")


def kl_kip_coz(args):
    """CLI KL-kipi çözümü: --kl-kip açık beyanı; --kl-tum-satirlar geriye-uyum kısayolu.

    Eski --kl-tum-satirlar bayrağı EŞLENİR: bayrak → "tum",
    --no-kl-tum-satirlar → "pencere" (p26 kipi). --kl-kip verilmediyse eşleme
    yürür; hiçbiri verilmediyse varsayılan "tum" (v4 aynen). İkisi de açık
    verilip ÇELİŞİYORSA RuntimeError — sessiz kısayol zaferi YOK (çıpa
    daima açık beyanlı: hangi satırların çıpalandığı kayıttan okunabilir).
    """
    if args.kl_kip is None:
        if args.kl_tum_satirlar is None:
            return "tum"
        return "tum" if args.kl_tum_satirlar else "pencere"
    if args.kl_tum_satirlar is not None:
        eski = "tum" if args.kl_tum_satirlar else "pencere"
        if eski != args.kl_kip:
            raise RuntimeError(
                f"V4_DUR: --kl-kip {args.kl_kip!r} ile --kl-tum-satirlar="
                f"{args.kl_tum_satirlar} (eşlemesi {eski!r}) çelişiyor — tek kip beyanı ver")
    return args.kl_kip


KL_KIP_AD = {"tum": "TÜM satırlar (v4)",
             "sft": "yalnız SFT satırları (v4b — pencere satırları ÇIPASIZ)",
             "pencere": "yalnız pencere satırları (p26)"}


def karisim_orani_hesapla(pencere_satir, sft_satir, ort_sft_hedef, block=BLOCK):
    """Token-bazlı karışım oranı — İLAN'ın NET beyanı.

    'Token-bazlı' = GERÇEK HEDEF KONUMU (kayıp sinyalinin birimi; y != -100).
    Ölçülen ort_sft_hedef ile: pencere satırı `block` hedef, SFT satırı ort_sft_hedef
    hedef taşır. Bağlam-jeton bazlı (satır=128 konum) oran da beyan edilir.
    """
    p_tok = float(block) * pencere_satir
    s_tok = float(ort_sft_hedef) * sft_satir
    toplam = p_tok + s_tok
    satir_toplam = pencere_satir + sft_satir
    return {
        "pencere_satir": int(pencere_satir), "sft_satir": int(sft_satir),
        "ort_sft_hedef": float(ort_sft_hedef),
        "hedef_bazli": {"kulliyat": p_tok / toplam, "sft": s_tok / toplam},
        "baglam_bazli": {"kulliyat": pencere_satir / satir_toplam,
                         "sft": sft_satir / satir_toplam},
    }


def nobet_ateslenmeli(opt_step, save_every, toplam_adim):
    """Ara val-nöbet düzeni: save-every katları, toplam-adımda ara-atlama yok.

    Son nöbet döngü sonrasında koşulsuz atılır (final); burada yalnız ARA nöbetler.
    """
    return save_every > 0 and opt_step % save_every == 0 and opt_step < toplam_adim


def _sft_akisi_uret(sft_n, seed, epok):
    """SFT çapası sonsuz döngü akışı: her tur ayrı tohumla yeniden karıştırma."""
    tur = 0
    while True:
        rng = np.random.default_rng([int(seed), 13, int(epok), tur])
        for j in rng.permutation(int(sft_n)):
            yield int(j)
        tur += 1


class V4BatchSampler:
    """Her mini-batch TAM `pencere_satir` pencere + (batch-pencere_satir) SFT satırı.

    Pencere tarafı: epok permütasyonu (tekrarsız, tam kapsam). SFT tarafı: tur
    başına yeniden karıştırılan sonsuz döngü (tekrarlı çapa — 'epoch başına yeniden
    döngü' iş emrinin tur-granül versiyonu). Deterministik: (seed, epok) → aynı akış.
    """

    def __init__(self, pencere_n, sft_n, batch, pencere_satir, seed, epok):
        if not (0 < pencere_satir < batch):
            raise ValueError(f"pencere_satir ({pencere_satir}) 0 < x < batch ({batch}) olmalı")
        if pencere_n < pencere_satir:
            raise ValueError(f"pencere_n ({pencere_n}) < pencere_satir ({pencere_satir})")
        self.pencere_n, self.sft_n = int(pencere_n), int(sft_n)
        self.batch, self.pencere_satir = int(batch), int(pencere_satir)
        self.seed, self.epok = int(seed), int(epok)

    def __len__(self):
        return math.ceil(self.pencere_n / self.pencere_satir)

    def __iter__(self):
        rng_p = np.random.default_rng([self.seed, 7, self.epok])
        p_sira = rng_p.permutation(self.pencere_n)
        sft_akis = _sft_akisi_uret(self.sft_n, self.seed, self.epok)
        for i in range(0, self.pencere_n, self.pencere_satir):
            parca = [int(v) for v in p_sira[i:i + self.pencere_satir]]
            sft_idx = [self.pencere_n + next(sft_akis)
                       for _ in range(self.batch - len(parca))]
            yield parca + sft_idx


class V4OgrenciDataset(torch.utils.data.Dataset):
    """v4 karışım veri kümesi: külliyat penceresi (lazy memmap) + SFT çapası.

    idx < pencere_n → rastgele 128-pencere (kaynak 1.0); idx >= pencere_n → SFT
    örneği (kaynak 0.0). Pencere satırı anında okunur (uint16 memmap dilimi → int64
    x/y; y-shift pretrain-hedefi BİREBİR); sign mask CPU'da (PencereDataset
    MPS-SIGSEGV dersi). SFT tarafı FastSampleAlignedDataset (ördek-tip: len + [i]).
    """

    def __init__(self, sft_dataset, bin_path, baslar, model, block=BLOCK):
        self.sft = sft_dataset
        self.mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.baslar = np.asarray(baslar, dtype=np.int64)
        self.model = model
        self.block = int(block)
        if int(self.mm.max()) >= 1 << 16:
            raise RuntimeError("uint16 sınırı aşıldı")

    def __len__(self):
        return len(self.baslar) + len(self.sft)

    def __getitem__(self, idx):
        if idx < len(self.baslar):
            b = int(self.baslar[idx])
            seq = np.asarray(self.mm[b:b + self.block + 1], dtype=np.int64)
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            sm = self.model.embedding.compute_sign_mask(x.unsqueeze(0))[0]
            return (x, y, sm, torch.tensor(1.0))
        x, y, sm = self.sft[idx - len(self.baslar)]
        return (x, y, sm, torch.tensor(0.0))


def mikro_parti_kayip(ogrenci, hoca, x, y, sm, kaynak, beta, tum_satirlar=True,
                      kip=None, accum=1, sub_chunk=SUB_CHUNK, kl_chunk=KL_CHUNK):
    """p26 mikro-parti kayıp matematiği BİREBİR (docstring not-6/7); kip kl_satirlari_sec'te.

    CE payı: alt-yığınların (CE_none×sm).sum()/sm_payda toplamı TAM parti
    (T-0209 ce_loss) değerine VE gradyanına eşit (toplam lineer; payda TAM partiden).
    KL payı: satır-seçim TEK NOKTA (kl_satirlari_sec; kip tum/sft/pencere) global
    kaynak/beta ile; KL_CHUNK yığın satır-toplamlarının cat().sum()'ı tam-seçimin
    .sum(-1).sum()'ına eşit; pay = mikro-parti GENELİ seçili-satır × T_poz
    (tum kipinde 16×128 konum ortalaması; sft/pencere kiplerinde seçilen
    satır sayısının T_poz'u). Yan etki: (loss/accum).backward() her alt-yığında
    çağrılır (p26 akışı).
    """
    sm_payda = sm.sum() + 1e-8
    satirlar = kl_satirlari_sec(kaynak, beta, tum_satirlar=tum_satirlar, kip=kip)
    kl_satir_sayisi = int(satirlar.numel())
    T_poz = x.size(1)
    kl_toplam_deger, kl_aktif = 0.0, False
    toplam_loss = 0.0
    for a0 in range(0, x.size(0), sub_chunk):
        x_c = x[a0:a0 + sub_chunk]; y_c = y[a0:a0 + sub_chunk]; sm_c = sm[a0:a0 + sub_chunk]
        logits, _ = ogrenci(x_c)
        V = logits.size(-1)
        ce_none = F.cross_entropy(logits.reshape(-1, V), y_c.reshape(-1), reduction="none").view_as(sm_c)
        loss = (ce_none * sm_c).sum() / sm_payda  # T-0209 BİREBİR; -100 hedef CE'de 0
        if kl_satir_sayisi > 0:
            yerel = satirlar[(satirlar >= a0) & (satirlar < a0 + sub_chunk)] - a0
            if yerel.numel() > 0:
                kl_parca = []
                for k0 in range(0, yerel.numel(), kl_chunk):
                    idx = yerel[k0:k0 + kl_chunk].to(x.device)
                    with torch.no_grad():
                        hoca_logits, _ = hoca(torch.index_select(x_c, 0, idx))
                        log_p_t = F.log_softmax(hoca_logits, dim=-1)
                        del hoca_logits  # MPS baskısı: öğretmen referansını hemen bırak
                    log_p_s = F.log_softmax(torch.index_select(logits, 0, idx), dim=-1)
                    kl_parca.append((log_p_t.exp() * (log_p_t - log_p_s)).sum(-1))
                    del log_p_t, log_p_s
                kl_satir_toplam = torch.cat(kl_parca, 0).sum()
                del kl_parca
                loss = loss + beta * kl_satir_toplam / (kl_satir_sayisi * T_poz)
                kl_aktif = True
                kl_toplam_deger += float(kl_satir_toplam.item())
                del kl_satir_toplam
        (loss / accum).backward()
        toplam_loss += float(loss.item())
        del logits, loss, ce_none
    kl_deger = (kl_toplam_deger / (kl_satir_sayisi * T_poz)) if kl_aktif else None
    return {"loss": toplam_loss, "kl_aktif": kl_aktif, "kl": kl_deger,
            "kl_satir": kl_satir_sayisi}


def paket_guveni_okula(model, tokenizer, vocab, device):
    """K3 nöbet tripwire'ı: tek-görev paket güveni ('okula' → 'case').

    test_model.py SFT-görev rig'inin birebir tek-görev uyarlaması (KAPI-3 aleti;
    TABAN referansı: durum-ekleri görevinde 'case' @ 0,9998 — p26 kapı ölçümü).
    """
    case = {"instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
            "input": "okula", "output": ""}
    cikti_id = vocab.stoi.get("<OUTPUT>", -1)
    ids = tokenizer.encode(json.dumps(case))
    if cikti_id not in ids:
        return {"top1": None, "guven": None, "beklenen": "case", "dogru": False,
                "hata": "<OUTPUT> etiketi akışta bulunamadı"}
    kesit = ids[:ids.index(cikti_id) + 1]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.tensor([kesit], dtype=torch.long, device=device)
        logits, _ = model(x)
        olasilik = torch.softmax(logits[0, -1, :], dim=-1)
        guven, en_iyi = torch.max(olasilik, dim=-1)
        top1 = vocab.decode(int(en_iyi.item()))
    if was_training:
        model.train()
    return {"top1": top1, "guven": float(guven.item()),
            "beklenen": "case", "dogru": top1 == "case"}


def val_nobeti_olc(model, val_loader, pencere_bin, device, tokenizer, vocab,
                  n_pencere=25, seed=7):
    """Üç hızlı nöbet göstergesi: SFT val-CE + külliyat pencere-CE (TRAIN bin,
    25 pencere, tohum 7 — p26 avlusu) + okula paket güveni. ~30 sn bandı."""
    sft_ce, sft_ppl = evaluate_val_loss(model, val_loader)
    pencere_ce = pencere_ce_kulliyat(model, pencere_bin, device, n=n_pencere, seed=seed)
    guven = paket_guveni_okula(model, tokenizer, vocab, device)
    model.train()
    return {"sft_val_ce": sft_ce, "sft_val_ppl": sft_ppl,
            "pencere_ce_train": pencere_ce,
            "okula_top1": guven["top1"], "okula_guven": guven["guven"],
            "okula_dogru": guven["dogru"]}


def parse_args():
    p = argparse.ArgumentParser(description="V4 öğrenci: külliyat %80 + SFT çapası + tüm-satır KL (pure'a)")
    p.add_argument("--init-model", default="data/anka_taban_pure.pt")
    p.add_argument("--teacher-model", default=None, help="KL öğretmeni; verilmezse init-model (öz-çapa)")
    p.add_argument("--sft-path", default="data/pedagogy/pure_base_sft_v4.jsonl")
    p.add_argument("--pencere-bin", default="data/kulliyat_v2/kulliyat_v2.bin")
    p.add_argument("--vocab", default="data/rebuild/vocab_anka_r1_33114.json")
    p.add_argument("--kl-beta", type=float, default=0.2)
    p.add_argument("--kl-kip", choices=["tum", "sft", "pencere"], default=None,
                   help="KL satır-seçim kipi: tum (v4: TÜM satır) | sft (v4b: YALNIZ SFT "
                        "satırları, pencere satırları ÇIPASIZ) | pencere (p26: yalnız pencere)")
    p.add_argument("--kl-tum-satirlar", action=argparse.BooleanOptionalAction, default=None,
                   help="GERİYE UYUM kısayolu: --no-kl-tum-satirlar → --kl-kip pencere; "
                        "--kl-kip ile çelişkide koşum DURUR; hiçbiri yoksa kip tum")
    p.add_argument("--peak-lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch", type=int, default=16, help="mikro-parti satır sayısı")
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--pencere-satir", type=int, default=10,
                   help="mikro-parti başına pencere satırı (SFT = batch - bu; ölçülen hedef-bazlı ~%%80/%%20)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--pencere-n", type=int, default=None,
                   help="epok pencere sayısı; verilmezse TAM külliyat epoğu (tekrarsız kaplama)")
    p.add_argument("--toplam-adim", type=int, default=None,
                   help="koşum planı (cosine + erken-duruş %%20 + kapı); verilmezse epok×adım/epok")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--save-path", required=True)
    p.add_argument("--allow-frozen-write", action="store_true")
    p.add_argument("--max-opt-steps", type=int, default=None, help="PİLOT: N opt-adımda kes")
    p.add_argument("--val-n", type=int, default=150, help="SFT val kuyruğu (son-N)")
    p.add_argument("--nobet-pencere-n", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-path", default="data/eval/v4_ogrenci_history.json")
    p.add_argument("--hukum-path", default=f"data/eval/v4_ogrenci_hukum_{TARIH}.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.teacher_model is None:
        args.teacher_model = args.init_model
    try:
        kl_kip = kl_kip_coz(args)
    except RuntimeError as e:
        print(str(e), flush=True)
        return 2

    t_program_basla = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("V4_DUR: MPS yok — koşum sandbox DIŞI olmalı (p26 kuralı)")
    pilot = args.max_opt_steps is not None
    print(f"=== V4 ÖĞRENCİ (külliyat+SFT çapası, KL kipi={kl_kip} β={args.kl_beta}) | "
          f"cihaz={device} | pilot={pilot} ===", flush=True)
    print(f"[MPS] önerilen çalışma-seti sınırı: {torch.mps.recommended_max_memory() / 2**30:.2f} GB", flush=True)

    check_frozen_save_path(args.save_path, allow_frozen_write=args.allow_frozen_write)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    for yol in (args.init_model, args.teacher_model, args.sft_path, args.pencere_bin):
        if not os.path.exists(yol):
            print(f"V4_DUR: bulunamadı: {yol}", flush=True); return 2

    lexicon = LexiconManager(); lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary(); vocab.load(args.vocab)
    tokenizer = KristalTokenizer(compiler, vocab)
    vocab_size = len(vocab.stoi)

    # Sınır kapısı (train.py T-0080 / p26 deseni): pencere bin maks id sözlük altında
    mm = np.memmap(args.pencere_bin, dtype=np.uint16, mode="r")
    if int(mm.max()) >= vocab_size:
        raise RuntimeError(f"V4_DUR: pencere bin maks id {int(mm.max())} >= vocab_size {vocab_size}")
    n_token = int(mm.size)
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
    if len(recs) != 5735:
        print(f"UYARI: SFT kayıt sayısı {len(recs)} != 5735 (v4 sabiti)", flush=True)
    val_records, train_records = recs[-args.val_n:], recs[:-args.val_n]

    # ---- Karışım geometrisi (İLAN'ın ölçülmüş beyanı) ----
    sft_dataset = FastSampleAlignedDataset(train_records, tokenizer, vocab, model, block_size=BLOCK)
    ort_sft_hedef = float(np.mean([int((y != -100).sum()) for _, y, _ in sft_dataset.samples]))
    sft_satir = args.batch - args.pencere_satir
    karisim = karisim_orani_hesapla(args.pencere_satir, sft_satir, ort_sft_hedef)
    pencere_satir_adim = args.pencere_satir * args.accum
    n_win = n_token // BLOCK
    kullanilabilir = n_win - 1
    if args.pencere_n is None:
        steps_per_epoch = kullanilabilir // pencere_satir_adim
        pencere_n = steps_per_epoch * pencere_satir_adim
    else:
        pencere_n = min(args.pencere_n, kullanilabilir)
        steps_per_epoch = math.ceil(pencere_n / pencere_satir_adim)
    toplam_adim = args.toplam_adim if args.toplam_adim is not None else steps_per_epoch * args.epochs
    sft_tur_adim = math.ceil(len(sft_dataset) / (sft_satir * args.accum)) if sft_satir > 0 else None
    print(f"[KARIŞIM] SFT train: {len(sft_dataset)} | val: {args.val_n} | "
          f"pencere: {pencere_n}/{kullanilabilir} kullanılabilir blok", flush=True)
    print(f"[KARIŞIM] mikro-parti {args.pencere_satir}P+{sft_satir}S satır | "
          f"hedef-bazlı: külliyat %{100 * karisim['hedef_bazli']['kulliyat']:.1f} / "
          f"SFT %{100 * karisim['hedef_bazli']['sft']:.1f} | bağlam-bazlı: "
          f"%{100 * karisim['baglam_bazli']['kulliyat']:.1f}/%{100 * karisim['baglam_bazli']['sft']:.1f} "
          f"(ölçülen ort SFT hedefi {ort_sft_hedef:.2f}/128)", flush=True)
    print(f"[KARIŞIM] adım/epok {steps_per_epoch} | toplam plan {toplam_adim} | "
          f"SFT çapası tam tur ≈ {sft_tur_adim} adımda (tekrarlı)", flush=True)

    val_dataset = FastSampleAlignedDataset(val_records, tokenizer, vocab, model, block_size=BLOCK)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    baslar = planla_pencere_baslar(n_token, pencere_n, args.seed, epok=0)
    veri = V4OgrenciDataset(sft_dataset, args.pencere_bin, baslar, model)

    # ---- Başlangıç nöbeti (t=0 eğri noktası) ----
    init_nobet = val_nobeti_olc(model, val_loader, args.pencere_bin, device,
                                tokenizer, vocab, n_pencere=args.nobet_pencere_n)
    print(f"[BAŞLANGIÇ] SFT val CE: {init_nobet['sft_val_ce']:.6f} (PPL {init_nobet['sft_val_ppl']:.4f}) | "
          f"pencere-CE (TRAIN, {args.nobet_pencere_n}): {init_nobet['pencere_ce_train']:.4f} | "
          f"okula: {init_nobet['okula_top1']} @ {init_nobet['okula_guven']:.4f}", flush=True)
    on_hesap_sn = time.time() - t_program_basla
    print(f"[ÖN-HESAP] sözlük+model/öğretmen+tokenizasyon+başlangıç nöbeti: {on_hesap_sn:.1f} sn", flush=True)
    torch.mps.empty_cache()

    optimizer = optim.AdamW(model.parameters(), lr=args.min_lr, weight_decay=args.weight_decay)
    model.train(); optimizer.zero_grad()
    opt_step, accum_sayac, running = 0, 0, 0.0
    adim_sureleri, history = [], []
    kl_aktif_mikro, kl_toplam_mikro, kl_degerleri = 0, 0, []
    kl_satir_toplami, satir_toplami = 0, 0  # v4b KL-deseni: satır-bazlı ateşlenme kanıtı
    adim_kl_degerleri = []
    val_nobeti = [{"adim": 0, **init_nobet, "aciklama": "başlangıç (init)"}]
    mps_tez_gb, mps_surucu_gb = 0.0, 0.0
    step_t0 = None
    erken_durus = None
    kesildi = False
    k1_siniri = math.ceil(K1_IZLEME_ORANI * toplam_adim)
    # nöbet düzeni planın ETKİN sonuna bakar (pilot kesmesi = son; final nöbet zaten
    # döngü sonrası atılır → aynı noktada çifte kayıt OLMAZ)
    etkin_son = min(toplam_adim, args.max_opt_steps) if pilot else toplam_adim
    t_basla = time.time()

    for epok in range(args.epochs):
        sampler = V4BatchSampler(pencere_n, len(sft_dataset), args.batch,
                                 args.pencere_satir, args.seed, epok)
        train_loader = DataLoader(veri, batch_sampler=sampler)
        for batch in train_loader:
            if step_t0 is None:
                step_t0 = time.time()  # opt-adımın BAŞI (accum penceresinin ilk mikro-partisi)
            x, y, sm, kaynak = batch  # kaynak: 0=SFT, 1=pencere (p26 B1)
            kl_toplam_mikro += 1
            satir_toplami += int(kaynak.numel())
            x = x.to(device); y = y.to(device); sm = sm.to(device)
            sonuc = mikro_parti_kayip(model, hoca, x, y, sm, kaynak, args.kl_beta,
                                      kip=kl_kip, accum=args.accum)
            kl_satir_toplami += int(sonuc["kl_satir"])
            running += sonuc["loss"]; accum_sayac += 1
            if sonuc["kl_aktif"]:
                kl_aktif_mikro += 1
                adim_kl_degerleri.append(sonuc["kl"])
                kl_degerleri.append(sonuc["kl"])

            if accum_sayac == args.accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                lr = get_lr(opt_step, toplam_adim, args.peak_lr, args.min_lr, args.warmup)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step(); optimizer.zero_grad()
                torch.mps.empty_cache()
                mps_tez_gb = max(mps_tez_gb, torch.mps.current_allocated_memory() / 2**30)
                try:
                    mps_surucu_gb = max(mps_surucu_gb, torch.mps.driver_allocated_memory() / 2**30)
                except (AttributeError, RuntimeError):
                    pass
                adim_sureleri.append(time.time() - step_t0)
                step_t0 = None
                opt_step += 1; accum_sayac = 0
                step_loss = running / args.accum; running = 0.0
                history.append({"adim": opt_step, "loss": step_loss, "lr": lr, "epok": epok + 1,
                               "kl_aktif": len(adim_kl_degerleri),
                               "kl_ort": (sum(adim_kl_degerleri) / len(adim_kl_degerleri))
                                         if adim_kl_degerleri else None})
                adim_kl_degerleri = []
                if opt_step % 10 == 0 or opt_step in (1, toplam_adim):
                    son = adim_sureleri[-10:]
                    ort = sum(son) / len(son)
                    print(f"Adım {opt_step}/{toplam_adim} | Loss {step_loss:.4f} | LR {lr:.2e} | "
                          f"adım-süresi ort {ort:.2f}s (son {len(son)}) | "
                          f"MPS {torch.mps.current_allocated_memory() / 2**30:.2f}GB tepe {mps_tez_gb:.2f} | "
                          f"KL-aktif {kl_aktif_mikro}/{kl_toplam_mikro} mikro-parti", flush=True)

                # ---- VAL-NÖBETİ (her save-every; ara checkpoint atomik) ----
                if nobet_ateslenmeli(opt_step, args.save_every, etkin_son):
                    kaydet(model, args.save_path)
                    nobet = val_nobeti_olc(model, val_loader, args.pencere_bin, device,
                                           tokenizer, vocab, n_pencere=args.nobet_pencere_n)
                    nobet["adim"] = opt_step
                    val_nobeti.append(nobet)
                    print(f"[VAL-NÖBETİ] adım {opt_step}/{toplam_adim}: SFT CE {nobet['sft_val_ce']:.6f} | "
                          f"pencere-CE {nobet['pencere_ce_train']:.4f} | "
                          f"okula {nobet['okula_top1']} @ {nobet['okula_guven']:.4f} | "
                          f"ckpt '{args.save_path}' (ara, atomik)", flush=True)
                    if opt_step <= k1_siniri and nobet["sft_val_ce"] > K1_ESIK:
                        erken_durus = {"adim": opt_step, "sft_val_ce": nobet["sft_val_ce"],
                                       "esik": K1_ESIK, "sinir_adim": k1_siniri}
                        print(f"[K1-ERKEN-DURUŞ] adım {opt_step}: SFT val CE {nobet['sft_val_ce']:.6f} > "
                              f"{K1_ESIK} (ilk %{int(K1_IZLEME_ORANI * 100)} penceresi) → koşum DURUR; "
                              f"ara checkpoint nöbette kaydedildi", flush=True)
                        kesildi = True
                if erken_durus:
                    break
                if opt_step >= toplam_adim:
                    kesildi = True
                    break
                if pilot and opt_step >= args.max_opt_steps:
                    kesildi = True
                    break
        if kesildi or erken_durus:
            break

    toplam_sure = time.time() - t_basla

    # ---- Final: kayıt + final nöbet ----
    kaydet(model, args.save_path)
    final_nobet = val_nobeti_olc(model, val_loader, args.pencere_bin, device,
                                tokenizer, vocab, n_pencere=args.nobet_pencere_n)
    final_nobet["adim"] = opt_step
    val_nobeti.append({**final_nobet, "aciklama": "final"})
    print(f"\n[BİTİŞ] adım {opt_step}/{toplam_adim} | SFT val CE: "
          f"{init_nobet['sft_val_ce']:.6f} -> {final_nobet['sft_val_ce']:.6f} | "
          f"pencere-CE (TRAIN): {init_nobet['pencere_ce_train']:.4f} -> "
          f"{final_nobet['pencere_ce_train']:.4f} | okula: "
          f"{init_nobet['okula_top1']}@{init_nobet['okula_guven']:.4f} -> "
          f"{final_nobet['okula_top1']}@{final_nobet['okula_guven']:.4f} | "
          f"Süre: {toplam_sure:.1f} sn", flush=True)
    kl_ort = (sum(kl_degerleri) / len(kl_degerleri)) if kl_degerleri else None
    if args.kl_beta > 0.0:
        kanit = (f"[KL-KANIT] KL dalı {kl_aktif_mikro}/{kl_toplam_mikro} mikro-partide ateşlendi"
                 + (f" | tüm-koşum KL ort {kl_ort:.6f}" if kl_ort is not None
                    else " | KL HİÇ ATEŞLENMEDİ")
                 + f" | kip: {KL_KIP_AD[kl_kip]}"
                 + (f" | satır-deseni: {kl_satir_toplami}/{satir_toplami} satır"
                    f" (%{100 * kl_satir_toplami / satir_toplami:.1f})"
                    if satir_toplami else "")
                 + " (öz-çapa öğretmen==init; ölçülen başlangıç KL ~0,125 = train/eval"
                   " dropout asimetrisi, ağırlık özdeşliğiyle çelişkili DEĞİL)")
    else:
        kanit = "[KL-KANIT] kl-beta=0 → KL kapalı"
    print(kanit, flush=True)

    ort20 = float(np.mean(adim_sureleri[-20:])) if adim_sureleri else None
    std20 = float(np.std(adim_sureleri[-20:])) if adim_sureleri else None
    eta_bant = None
    if ort20 is not None and std20 is not None:
        # ETA kuralı (İLAN): pilot s/adım × 3 (sürekli-koşum çarpanı; T3.4 kanıtı:
        # pilot 1,18 → gerçek 3,27 sn/adım) × toplam-adım. Dürüst bant: ±1 std.
        eta_bant = {"carpan": 3.0, "toplam_adim": toplam_adim,
                    "alt_sa": round((ort20 - std20) * 3 * toplam_adim / 3600, 1),
                    "ust_sa": round((ort20 + std20) * 3 * toplam_adim / 3600, 1),
                    "nokta_sa": round(ort20 * 3 * toplam_adim / 3600, 1)}
        print(f"[ETA] tam koşum ({toplam_adim} adım) dürüst bandı: "
              f"{eta_bant['alt_sa']}-{eta_bant['ust_sa']} sa "
              f"(nokta: {eta_bant['nokta_sa']} sa; pilot {ort20:.2f}±{std20:.2f} s/adım × 3)", flush=True)
    print(f"[MODEL_KAYDEDİLDİ] {args.save_path} (sha256: {compute_sha256(args.save_path)})", flush=True)

    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    hukum = {
        "gorev": f"V4 öğrenci (külliyat %80 + SFT çapası %20, KL kipi={kl_kip} pure'a)",
        "pilot": pilot, "durum": ("ERKEN_DURUS_K1" if erken_durus else "BASARILI"),
        "init_model": args.init_model, "teacher_model": args.teacher_model,
        "init_sha256": compute_sha256(args.init_model),
        "sft_path": args.sft_path, "sft_sha256": compute_sha256(args.sft_path),
        "pencere_bin": args.pencere_bin, "pencere_bin_sha256": compute_sha256(args.pencere_bin),
        "vocab_sha256": compute_sha256(args.vocab),
        "kulliyat_token": n_token, "pencere_n": pencere_n,
        "kulliyat_epok_kapsama": round(pencere_n / kullanilabilir, 6),
        "karisim": karisim, "pencere_satir": args.pencere_satir, "sft_satir": sft_satir,
        "sft_train_n": len(sft_dataset), "val_n": len(val_dataset),
        "sft_tur_adim": sft_tur_adim,
        "kl_beta": args.kl_beta, "kl_kip": kl_kip,
        "kl_tum_satirlar_bayrak": args.kl_tum_satirlar,
        "kl_aktif_orani": round(kl_aktif_mikro / kl_toplam_mikro, 4) if kl_toplam_mikro else None,
        "kl_satir_orani": round(kl_satir_toplami / satir_toplami, 4) if satir_toplami else None,
        "peak_lr": args.peak_lr, "min_lr": args.min_lr, "warmup": args.warmup,
        "batch": args.batch, "accum": args.accum, "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch, "toplam_adim": toplam_adim,
        "opt_adim": opt_step, "sure_sn": round(toplam_sure, 1),
        "on_hesap_sn": round(on_hesap_sn, 1),
        "s_adim_son20_ort": round(ort20, 3) if ort20 is not None else None,
        "s_adim_son20_std": round(std20, 3) if std20 is not None else None,
        "eta_bant": eta_bant,
        "kl_aktif_mikro": kl_aktif_mikro, "kl_toplam_mikro": kl_toplam_mikro,
        "kl_satir_toplami": kl_satir_toplami, "satir_toplami": satir_toplami,
        "kl_ort": kl_ort,
        "mps_tez_gb": round(mps_tez_gb, 3), "mps_surucu_gb": round(mps_surucu_gb, 3),
        "val_nobeti": val_nobeti,
        "k1_esik": K1_ESIK, "k1_siniri": k1_siniri, "erken_durus": erken_durus,
        "cikti": args.save_path, "cikti_sha256": compute_sha256(args.save_path),
    }
    with open(args.hukum_path, "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2, ensure_ascii=False)
    print(f"[HÜKÜM] {args.hukum_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
