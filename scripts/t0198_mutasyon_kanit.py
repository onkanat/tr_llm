#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0198 K3 — hiza-ağırlık mutasyon-kaniti (BETİKTEN; MPS; koşum-ÖNCESİ).

MADDE-6 birebir: dal-i KAPAT (ağırlık=1,0 geçerli-jeton) → halef-standard-CE
ile BETİKTEN-iki-yol birebir (≤1e-9) + HizaliDataset kopya-bütünlüğü
(halef-cache ilk-64 örnek (x,y,sm) bit-birebir) + 3-jeton-argmax-id
BEKLENEN_K BETİKTEN-kayıt; dal-ii AÇIK (ALPHA=2,0) → aynı-parti kayıp-farkı
>0. DUR rc=2 ⇒ eğitim BAŞLAMAZ. Eval-time kayıp ağırlıksız (MADDE-3)."""
import json
import os
import random
import sys
import time
import torch

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
    compute_sha256,
    render_prompt,  # halef — birebir (src.llm.prompt_contract re-export)
)

ALPHA = 2.0
BLOCK = 128
TRAIN_PATH = os.path.join(KÖK, "data/eval/t0197_uretim_saf.jsonl")
HALEF_CACHE = os.path.join(KÖK, "data/eval/t0197_fast_ds_train.pt")
HIZA_CACHE = os.path.join(KÖK, "data/eval/t0198_fast_ds_train.pt")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0198_mutasyon.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type != "mps":
    raise RuntimeError("T0198_DUR: MPS yok — koşum sandbox DIŞI olmalı")


def kelime(x: str) -> list:  # halef t0194 — birebir normalizasyon (yalnız-OKU)
    import re
    return re.findall(r"[\w']+", x.lower())


class HizaliDataset(FastSampleAlignedDataset):
    """halef __init__ yapısal-kopyası (T-0197 BETİKTEN; 4. tensor w eklendi).
    Kopya-bütünlüğü koşumda halef-cache ile bit-birebir denetlenir."""

    def __init__(self, records, tokenizer, vocab, model, block_size: int = 128, alpha: float = ALPHA):
        self.alpha = alpha
        self.block_size = block_size
        self.pad_id = vocab.stoi.get("<PAD>", 1)
        self.bos_id = vocab.stoi.get("<BOS>", 2)
        self.eos_id = vocab.stoi.get("<EOS>", 3)
        self.samples: list = []
        for rec in records:
            inst = rec.get("instruction", "").strip()
            inp = rec.get("input", "").strip()
            out = rec.get("output", "").strip()
            prompt_str = render_prompt(inst, inp)
            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == self.eos_id:
                prompt_ids = prompt_ids[:-1]
            out_ids = tokenizer.encode(out)
            if out_ids and out_ids[0] == self.bos_id:
                out_ids = out_ids[1:]
            if not out_ids or out_ids[-1] != self.eos_id:
                out_ids.append(self.eos_id)
            full_seq = prompt_ids + out_ids
            P = len(prompt_ids)
            if len(full_seq) < 3:
                continue
            if len(full_seq) > self.block_size + 1:
                excess = len(full_seq) - (self.block_size + 1)
                if P - 4 > excess:
                    prompt_ids = prompt_ids[:2] + prompt_ids[2 + excess:]
                    P = len(prompt_ids)
                    full_seq = prompt_ids + out_ids
                else:
                    full_seq = full_seq[-(self.block_size + 1):]
                    P = max(1, P - excess)
            x_ids = full_seq[:-1]
            y_ids = full_seq[1:]
            y_masked, w = [], []
            in_words = set(kelime(inp)) if inp else set()
            for i, tok in enumerate(y_ids):
                if i < P - 1:
                    y_masked.append(-100)
                    w.append(0.0)
                else:
                    y_masked.append(tok)
                    d = vocab.decode(tok).strip().lower()
                    agir = 1.0
                    if tok != self.eos_id:
                        if d.startswith("TENSE_") or d.startswith("COPULA_"):
                            agir = self.alpha
                        elif any(p in in_words for p in kelime(d)):
                            agir = self.alpha
                    w.append(agir)
            curr_len = len(x_ids)
            if curr_len < self.block_size:
                pad_len = self.block_size - curr_len
                x_ids = x_ids + [self.pad_id] * pad_len
                y_masked = y_masked + [-100] * pad_len
                w = w + [0.0] * pad_len
            else:
                x_ids = x_ids[:self.block_size]
                y_masked = y_masked[:self.block_size]
                w = w[:self.block_size]
            x_tensor = torch.tensor(x_ids, dtype=torch.long)
            y_tensor = torch.tensor(y_masked, dtype=torch.long)
            w_tensor = torch.tensor(w, dtype=torch.float32)
            sign_mask = model.embedding.compute_sign_mask(x_tensor.unsqueeze(0))[0]
            self.samples.append((x_tensor, y_tensor, sign_mask, w_tensor))


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    import scripts.train_t0197_uretim_saf as ts  # halef — yalnız-OKU (CIPE/HALEF_SHA)
    for yol, hiz in ts.CIPE.items():
        if compute_sha256(os.path.join(KÖK, yol)) != hiz:
            print(f"T0198_DUR: çıpa-uyuşmazlığı {yol}", flush=True)
            return 2

    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)

    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096)
    sd = torch.load(os.path.join(KÖK, "data/anka_b1_5_best.pt"), map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    model.eval()
    print("[MODELI] halef=anka_b1_5_best (e5eb114e…) YALNIZ-OKU — mutasyon", flush=True)

    train_records = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    hiza_ds = HizaliDataset(train_records, tokenizer, vocab, model, block_size=BLOCK)
    torch.save(hiza_ds, HIZA_CACHE)
    print(f"hiza-ds n={len(hiza_ds)} (cache yazıldı)", flush=True)

    # ---- kopya-bütünlüğü: halef-cache (x,y,sm) bit-birebir ----
    halef_ds = torch.load(HALEF_CACHE, weights_only=False)
    if len(halef_ds) != len(hiza_ds):
        print(f"T0198_DUR: ds-uzunluk {len(halef_ds)} != {len(hiza_ds)}", flush=True)
        return 2
    n_kontrol = min(64, len(halef_ds))
    for i in range(n_kontrol):
        hx, hy, hs = halef_ds.samples[i][:3]
        zx, zy, zs, _ = hiza_ds.samples[i]
        if not (torch.equal(hx, zx) and torch.equal(hy, zy) and torch.equal(hs.float(), zs.float())):
            print(f"T0198_DUR: kopya-bütünlüğü i={i}", flush=True)
            return 2
    print(f"[KOPYA] halef-cache birebir PASS (n={n_kontrol})", flush=True)

    # ---- dal-i: KAPAT (ağırlık=1,0 geçerli) — BETİKTEN-iki-yol birebir ----
    batch = [hiza_ds.samples[i] for i in range(16)]
    xs = torch.stack([b[0] for b in batch]).to(DEVICE)
    ys = torch.stack([b[1] for b in batch]).to(DEVICE)
    ss = torch.stack([b[2] for b in batch]).to(DEVICE)
    with torch.no_grad():
        logits, loss_std = model(xs, ys, ss)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), ys.view(-1), reduction="none", ignore_index=-100)
        w_kapat = (ys != -100).float()
        loss_kapat = (ce * w_kapat.view(-1)).sum() / w_kapat.view(-1).sum()
    fark1 = abs(float(loss_std) - float(loss_kapat))
    argmaxs = logits.argmax(-1)[0, [0, 1, 10]].tolist()
    print(f"[DAL-i] std={float(loss_std):.10f} kapat={float(loss_kapat):.10f} fark={fark1:.2e} "
          f"(≤1e-9: {fark1 <= 1e-9})", flush=True)
    print(f"[BEKLENEN_K] argmax-3-id [0,1,10]={argmaxs}", flush=True)

    # ---- dal-ii: AÇIK (ALPHA=2,0) — aynı-parti fark >0 ----
    ws = torch.stack([b[3] for b in batch]).to(DEVICE)
    with torch.no_grad():
        w_acik = torch.where(ys != -100, ws, torch.zeros_like(ws))
        loss_acik = (ce * w_acik.view(-1)).sum() / w_acik.view(-1).sum()
    fark2 = float(loss_acik) - float(loss_kapat)
    n_alpha = int(((ys != -100) & (w_acik > 1.0)).sum())
    print(f"[DAL-ii] agirlikli={float(loss_acik):.10f} fark={fark2:.4e} >0: {fark2 > 0} "
          f"| alpha-jeton={n_alpha}/parti", flush=True)

    kapilar = {"kopya_birebir": True, "kapat_le1e9": fark1 <= 1e-9, "acik_fark_pozitif": fark2 > 0}
    gecti = all(kapilar.values())
    hukum = "T0198_MUTASYON_GECTI" if gecti else "T0198_MUTASYON_DUR"
    donem = {"task": "T-0198", "hukum": hukum, "rc": 0 if gecti else 2, "alpha": ALPHA,
             "kapilar": kapilar, "kayip_std": float(loss_std), "kayip_kapat": float(loss_kapat),
             "kayip_acik": float(loss_acik), "fark_acik": fark2,
             "beklenen_k_argmax_3id": argmaxs, "ds_n": len(hiza_ds), "kopya_kontrol_n": n_kontrol,
             "hiza_cache": HIZA_CACHE,
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())