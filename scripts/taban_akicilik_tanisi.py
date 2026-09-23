#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TABAN AKICILIK TANISI — "temeli doğru eğittik mi?" sorusunu ÖLÇER (eğitim yok, saf çıkarım).

NEDEN VAR (22 Eyl 2026): r33'te bir LoRA modülü, tam ince ayar checkpoint'inin Wikipedia
hasarının %68'ini onardı. Bu, "checkpoint'lerde eğitim kaynaklı bir bozukluk mu var?"
sorusunu doğurdu. Ayırt etmenin en ucuz yolu: modelin **KENDİ dağılımında** akıcı üretip
üretmediğine bakmak.

  · Taban kendi dağılımında (Wikipedia) AKICI üretiyor, ama talimat zarfında çöküyorsa
    ⇒ taban bir **dil modeli olarak SAĞLAM**; eksik olan **talimat aşamasıdır** (bozukluk değil).
  · Taban kendi dağılımında da döngüye giriyorsa
    ⇒ taban eğitimi **BOZUK**tur ve üstüne kurulan her şey boşa gider.

⚠ ÖLÇÜLERE GÜVENMEYİN — BU SONDA BİR KEZ YANLIŞ CEVAP VERDİ (22 Eyl 2026, r34).
  `benzersiz_4gram` ve `dongu_orani` ilan edilmiş eşiklerle DÖRT modelin DÖRDÜNE de "BOZUK"
  dedi — ppl'i **33,4** olan sağlam taban dâhil. Sebep: sonda **greedy** üretiyor ve greedy
  kod çözme **her** dil modelinde tekrara kaçar (bilinen kod çözme artefaktı, eğitim kusuru
  DEĞİL). Kanonik kap aynı modeller için tersini söylüyor (seg_6: kanonik tutarsızlık %5,
  bu sonda döngü %96). ⇒ **Ayrım yapmayan ölçüt, ölçüt değildir.**

AYIRT EDEN ÖLÇÜT: **`ppl`** (öğretmen zorlamalı, kod çözmeye bağlı değil).
  Ölçüldü: taban **33,38** (en iyi) · seg_1 76,85 · seg_6 **599,72** · seg_6+modül 86,41
  ⇒ taban dil modeli olarak SAĞLAM; bozukluk **tam ince ayarda** (ppl 18× bozuluyor).

Kaydedilen ölçütler (yorum için, kapı DEĞİL):
  · `benzersiz_4gram` · `dongu_orani` : GREEDY kod çözme artefaktı — modelleri AYIRT ETMEZ
  · `ppl`             : A ekseni pencerelerinde perplexity — **ayırt eden budur**

Kullanım:
  venv/bin/python scripts/taban_akicilik_tanisi.py --model data/anka_a1r.pt --device mps
  venv/bin/python scripts/taban_akicilik_tanisi.py --model scratch/t0096_kos/seg_6.pt \
      --module modules/seg6_marangoz_1000.mod.pt --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.modules import modul_katmanlari, modul_yukle  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402
from src.llm.tokenizer import Vocabulary  # noqa: E402

WIKI_BIN = "data/anka_a1r_pretrain.bin"
SOZLUK = "data/rebuild/vocab_anka_r1_33114.json"
B_PENCERE, SEED = 128, 7


def durdur(mesaj: str) -> None:
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def yukle(yol: str, vocab: Vocabulary, device: torch.device) -> KristalLM:
    if not os.path.exists(yol):
        durdur(f"checkpoint yok: '{yol}'")
    m = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    d = torch.load(yol, map_location="cpu")
    for k in [k for k in list(d.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del d[k]
    m.load_state_dict(resize_state_dict(m, d), strict=False)
    m.to(device)
    m.eval()
    return m


def dortgram_olcut(tok: List[int]) -> Dict[str, Any]:
    """Akıcılık ölçütleri — saf fonksiyon (test edilebilir)."""
    if len(tok) < 8:
        return {"benzersiz_4gram": 1.0, "en_cok_tekrar": 0, "uzunluk": len(tok), "dongu": False}
    g = [tuple(tok[i:i + 4]) for i in range(len(tok) - 3)]
    say = {}
    for x in g:
        say[x] = say.get(x, 0) + 1
    en = max(say.values())
    return {"benzersiz_4gram": len(say) / len(g), "en_cok_tekrar": int(en),
            "uzunluk": len(tok), "dongu": bool(en >= 3)}


def uret(model: KristalLM, x: torch.Tensor, max_new: int) -> List[int]:
    """Greedy devam — kanonik rig'in `uret`'i gibi imza maskesini HER adımda yeniden hesaplar."""
    out: List[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            sm = model.embedding.compute_sign_mask(x).to(x.device)
            logits, _ = model(x, sign_mask=sm)
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=x.device)], dim=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Taban akıcılık tanısı (saf çıkarım)")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--module", type=str, default=None)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=str, default=None)
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available())
                          else "cpu")
    if a.device == "mps" and device.type != "mps":
        durdur("--device mps istendi ama MPS yok (sandbox MPS'i gizler; sandbox DIŞINDA koşun)")

    vocab = Vocabulary()
    vocab.load(SOZLUK)
    model = yukle(a.model, vocab, device)
    modul_meta = None
    if a.module:
        modul_meta = modul_yukle(model, a.module, a.model)
        model.eval()

    mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    n_win = int(mm.size // B_PENCERE)
    rng = random.Random(SEED)
    baslar = sorted(rng.sample(range(n_win), min(a.n, n_win)))

    olcumler: List[Dict[str, Any]] = []
    kayiplar: List[float] = []
    for b in baslar:
        seq = np.asarray(mm[b * B_PENCERE:(b + 1) * B_PENCERE], dtype=np.int64)
        x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            sm = model.embedding.compute_sign_mask(x).to(device)
            _, loss = model(x, targets=y, sign_mask=sm)
        kayiplar.append(float(loss.item()))
        olcumler.append(dortgram_olcut(uret(model, x[:, :64].clone(), a.max_new)))

    n = len(olcumler)
    benz = float(np.mean([o["benzersiz_4gram"] for o in olcumler]))
    dongu = 100.0 * sum(1 for o in olcumler if o["dongu"]) / n
    ce = float(np.mean(kayiplar))
    sonuc = {
        "model": a.model, "module": a.module,
        "ce_ort": round(ce, 4), "ppl": round(math.exp(ce), 2),
        "benzersiz_4gram": round(benz, 4), "dongu_orani": round(dongu, 2),
        "n": n, "max_new": a.max_new, "device": str(device),
        "modul_katman": len(modul_katmanlari(model)) if a.module else 0,
        "ornek_uretim": [o for o in olcumler[:3]],
    }
    print(json.dumps({k: v for k, v in sonuc.items() if k != "ornek_uretim"},
                     ensure_ascii=False, indent=2))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(sonuc, f, ensure_ascii=False, indent=2)
        print(f"[out] {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
