"""Val dilimi üzerinde ortalama cross-entropy (CE) ve PPL ölçümü (T3.4).

KristalLM kurar (n_embd 768, n_layer 6, n_head 6, block_size 128),
checkpoint'i test_model.py deseniyle yükler (torch.load → cos/sin/mask
anahtarlarını at → resize_state_dict → strict=False), val bin'den sabit
tohumla N adet 128-token pencere seçer ve ortalama cross-entropy +
PPL yazdırır. Hedef maskeleme YOKTUR (düz sonraki-jeton CE; --pretrain
kayıp tanımıyla aynı, PAD hedefi val diliminde ölçülüp raporlanır).

Kullanım (taban ölçümü):
  venv/bin/python scripts/eval_ce_val.py \
      --model data/anka_taban_pure.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json \
      --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500
"""
import argparse
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.llm.tokenizer import Vocabulary
from src.llm.prompt_contract import resize_state_dict
from scripts.train_step_demo import KristalLM


def parse_args():
    ap = argparse.ArgumentParser(description="Val dilimi CE/PPL ölçümü (T3.4)")
    ap.add_argument("--model", required=True, help="Değerlendirilecek .pt checkpoint")
    ap.add_argument("--vocab", required=True, help="Sözlük JSON yolu")
    ap.add_argument("--val", required=True, help="Val .bin yolu (uint16)")
    ap.add_argument("--samples", type=int, default=500, help="Ölçüm pencere sayısı (varsayılan 500)")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337, help="Pencere seçim tohumu (sabit)")
    ap.add_argument("--device", default=None, help="cpu/mps/cuda; verilmezse otomatik (mps→cuda→cpu)")
    return ap.parse_args()


def secil_cihaz(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = secil_cihaz(args.device)

    # 1. Sözlük
    vocab = Vocabulary()
    vocab.load(args.vocab)
    vocab_size = len(vocab.stoi)

    # 2. Val verisi + sınır kapısı (train.py T-0080 deseni)
    val = np.memmap(args.val, dtype=np.uint16, mode="r")
    val_uzunluk = int(len(val))
    maks_id = int(val.max())
    if maks_id >= vocab_size:
        raise RuntimeError(
            f"DURDURULDU: val maks jeton id {maks_id} >= vocab_size {vocab_size}. "
            f"--vocab ile bu külliyata uygun sözlüğü verin.")

    # 3. Sabit tohumlu pencere seçimi
    bs = args.block_size
    max_start = val_uzunluk - bs - 1
    if max_start < 1:
        raise RuntimeError(f"DURDURULDU: val çok küçük ({val_uzunluk} token) — {bs + 1} tokenlik pencere sığmıyor.")
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, max_start + 1, size=args.samples)

    # 4. Model kur + yükle (test_model.py deseni: cos/sin/mask at + resize + strict=False)
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab,
                     n_layer=6, n_head=6, block_size=bs)
    # weights_only=True güvenli: bu repo'nun kendi ürettiği checkpoint'leri düz
    # tensor state_dict'tir (torch.save(model.state_dict())); pickle kodu içermez.
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    atilan = [k for k in state_dict if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in atilan:
        del state_dict[k]
    state_dict = resize_state_dict(model, state_dict)
    eksik, beklenmedik = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 5. Ölçüm: her pencere x=128 token, y=sonraki 128 token (düz sonraki-jeton CE)
    toplam_ce = 0.0
    toplam_konum = 0
    pad_id = vocab.stoi.get("<PAD>", 1)
    pad_hedef = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, args.samples, args.batch_size):
            parca = starts[i:i + args.batch_size]
            x_np = np.stack([np.asarray(val[s:s + bs], dtype=np.int64) for s in parca])
            y_np = np.stack([np.asarray(val[s + 1:s + 1 + bs], dtype=np.int64) for s in parca])
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            # Sign mask train.py deseni: CPU'da hesapla, sonra cihaza taşı
            sign_mask = model.embedding.compute_sign_mask(x.detach().cpu()).to(device)
            logits, _ = model(x, None, sign_mask)
            pad_hedef += int((y_np == pad_id).sum())
            ce_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="sum",
            )
            toplam_ce += float(ce_sum.item())
            toplam_konum += int(y.numel())
    sure = time.time() - t0

    ce = toplam_ce / toplam_konum
    ppl = math.exp(ce)

    print("[CE-VAL] model: " + args.model)
    print(f"[CE-VAL] vocab: {args.vocab} ({vocab_size} giriş) | val: {args.val} ({val_uzunluk:,} token)")
    print(f"[CE-VAL] block_size={bs} | samples={args.samples} | batch={args.batch_size} | seed={args.seed} | device={device}")
    print(f"[CE-VAL] yükleme: atılan cos/sin/mask={len(atilan)} | eksik={len(eksik)} | beklenmedik={len(beklenmedik)}")
    if eksik:
        print(f"[CE-VAL] UYARI eksik anahtarlar: {eksik}")
    if beklenmedik:
        print(f"[CE-VAL] UYARI beklenmedik anahtarlar: {beklenmedik}")
    if pad_hedef:
        print(f"[CE-VAL] UYARI: {pad_hedef} PAD hedefi CE'ye DAHİL edildi (train kaybı bunları -100'ler)")
    print(f"[CE-VAL] ORTALAMA CROSS-ENTROPY: {ce:.4f}  ({args.samples} pencere × {bs} hedef = {toplam_konum:,} konum)")
    print(f"[CE-VAL] PERPLEXITY (PPL): {ppl:.4f}")
    print(f"[CE-VAL] süre: {sure:.1f} sn")


if __name__ == "__main__":
    main()