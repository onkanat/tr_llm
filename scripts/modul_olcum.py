#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MODÜL ÖLÇÜM KABI — bir yetenek modülü tabanı BOZUYOR mu, yeteneği TAŞIYOR mu?

NEDEN AYRI BETİK: `scripts/evaluate_carpenter_anka.py` tam donanımlıdır (ceket ekseni,
oracle katmanı, lexicon, üretim) ama modül YÜKLEYEMEZ. Ona `--module` eklemek yasak değil
ama o betiğin satır numaraları kapalı kayıtlarda REFERANS olarak geçiyor ve T-0098'in
`test_esik_referanslari` kapısı `:62-69`'un `ESIK_*` taşıdığını doğruluyor ⇒ satır
kaydıran bir düzenleme kapanmış bir kapıyı düşürür. Bu yüzden ölçüm YENİ bir dosyada,
kanonik yardımcı **kopyalanmadan import edilerek** yapılır.

İKİ EKSEN, İKİ FARKLI SORU:
  A (unutma) : Wikipedia diliminde MASKESİZ CE — kanonik `a_ekseni`, 256 pencere, seed 7.
               Kapalı kayıtlarla KIYASLANABİLİR olması için parametreler AYNEN kanonikten.
               Eşik ilan edilmiş: artış <= **+%10** (ESIK_A_ARTIS).
  F (uyum)   : marangoz külliyatında düz sonraki-jeton CE. **GENELLEME ÖLÇÜSÜ DEĞİLDİR** —
               aynı külliyatın pencereleri üzerinde ölçülür, yani "model bu metne ne kadar
               alıştı" der. Yeteneğin kendisi (ezber/ROUGE/kesişim) ancak tam kapıyla ölçülür.
               Buraya konmasının tek nedeni: modülün HİÇ öğrenip öğrenmediğini ayırt etmek.

POZİTİF KONTROL (kabın kendi kendini sınaması, zorunlu): taban modele AYNI tarifle ama
**sıfır** bir modül takılır (`lora_B=0`) ⇒ A ölçümü tabanla **BİREBİR** aynı çıkmalıdır.
Çıkmazsa ölçüm kabı gürültülü demektir ve "modül fark yaratmadı" hükmü KURULAMAZ.

Fail-closed: `--base`/`--module` zorunlu · modülün `base_sha256` çapası uyuşmazsa DUR
(`modul_yukle`) · istenen cihaz yoksa DUR (sessiz CPU düşmesi yok).
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.evaluate_carpenter_anka import ESIK_A_ARTIS, a_ekseni  # kanonik, kopya YOK
from scripts.train_step_demo import KristalLM
from src.llm.modules import (
    ModulSpec,
    modul_ekle,
    modul_ozeti,
    modul_yukle,
    sha256_dosya,
)
from src.llm.prompt_contract import resize_state_dict
from src.llm.tokenizer import Vocabulary

VOCAB_VARSAYILAN = "data/rebuild/vocab_anka_r1_33114.json"
MARANGOZ_BIN = "data/train_carpenter_specialization_anka_r18.bin"
F_PENCERE = 128
F_ADET = 128


def durdur(mesaj: str) -> None:
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def _arg(adlar: Tuple[str, ...], varsayilan: Optional[str] = None) -> Optional[str]:
    for i, a in enumerate(sys.argv):
        if a in adlar and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return varsayilan


def taban_yukle(base_yol: str, vocab: Vocabulary, device: torch.device) -> KristalLM:
    """Çıplak taban model — HER ölçüm için SIFIRDAN kurulur (durum sızmasın)."""
    m = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    d = torch.load(base_yol, map_location="cpu")
    for k in [k for k in list(d) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del d[k]
    m.load_state_dict(resize_state_dict(m, d), strict=False)
    m.to(device).eval()
    return m


def uyum_ekseni(model: KristalLM, device: torch.device, yol: str) -> Dict[str, Any]:
    """F ekseni: marangoz külliyatında MASKESİZ CE. GENELLEME ÖLÇÜSÜ DEĞİL (bkz. başlık)."""
    if not os.path.exists(yol):
        durdur(f"F ekseni külliyatı YOK: {yol}")
    mm = np.memmap(yol, dtype=np.uint16, mode="r")
    n_win = mm.size // F_PENCERE
    if n_win < F_ADET:
        durdur(f"F ekseni: yalnız {n_win} pencere var, {F_ADET} gerekli")
    baslar = sorted(random.Random(7).sample(range(n_win), F_ADET))
    kayiplar: List[float] = []
    with torch.no_grad():
        for b in baslar:
            seq = np.asarray(mm[b * F_PENCERE:(b + 1) * F_PENCERE], dtype=np.int64)
            x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
            # `compute_sign_mask` CPU döndürür (kanonik `a_ekseni` bu yüzden `.to(device)`
            # yapar); unutulursa MPS'te "found at least two devices" ile çöker — ölçüldü.
            _, loss = model(x, targets=y,
                            sign_mask=model.embedding.compute_sign_mask(x).to(device))
            kayiplar.append(float(loss.item()))
    return {"CE_ort": float(np.mean(kayiplar)), "CE_std": float(np.std(kayiplar)),
            "pencere": F_ADET, "blok": F_PENCERE, "seed": 7, "dilim": yol,
            "maskesiz": True, "genelleme_degil": True}


def main() -> None:
    base_yol = _arg(("--base",))
    modul_yol = _arg(("--module",))
    cikti = _arg(("--cikti",))
    if not base_yol or not os.path.exists(base_yol):
        durdur(f"--base yok/eksik: '{base_yol}'")
    if not modul_yol or not os.path.exists(modul_yol):
        durdur(f"--module yok/eksik: '{modul_yol}'")

    istenen = _arg(("--device",), "cpu")
    if istenen == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif istenen == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif istenen in ("cpu", "", None):
        device = torch.device("cpu")
    else:
        durdur(f"--device {istenen} kullanılamıyor (mps={torch.backends.mps.is_available()})")
    print(f"Cihaz: {device} | taban: {base_yol} ({sha256_dosya(base_yol)[:16]}…)")

    vocab = Vocabulary()
    vocab.load(str(_arg(("--vocab",), VOCAB_VARSAYILAN)))
    meta = modul_ozeti(str(modul_yol))
    spec = ModulSpec.sozlukten(meta["spec"])
    print(f"Modül: {modul_yol} | '{spec.ad}' r={spec.r} alpha={spec.alpha} "
          f"| {meta['katman_sayisi']} katman | {meta['modul_parametre']:,} parametre "
          f"| eğitim adımı={meta.get('adim')}")

    # ---------------- 1) TABAN ----------------
    m = taban_yukle(str(base_yol), vocab, device)
    A_taban = a_ekseni(m, vocab, device)
    F_taban = uyum_ekseni(m, device, MARANGOZ_BIN)
    del m

    # ---------------- 2) POZİTİF KONTROL: sıfır modül ----------------
    # AYNI tarif, ama lora_B sıfır ⇒ delta = 0 ⇒ A BİREBİR aynı olmalı.
    m = taban_yukle(str(base_yol), vocab, device)
    modul_ekle(m, spec)
    m.eval()
    A_sifir = a_ekseni(m, vocab, device)
    del m
    esit = abs(A_sifir["CE_ort"] - A_taban["CE_ort"]) == 0.0
    print(f"[kontrol] sıfır modül A CE = {A_sifir['CE_ort']:.6f} | taban {A_taban['CE_ort']:.6f} "
          f"| {'BİREBİR EŞİT (kap sağlam)' if esit else 'FARKLI — KAP GÜRÜLTÜLÜ'}")
    if not esit:
        durdur("pozitif kontrol düştü: takma işlemi ya da ölçüm yolu çıktıyı değiştiriyor.")

    # ---------------- 3) EĞİTİLMİŞ MODÜL ----------------
    m = taban_yukle(str(base_yol), vocab, device)
    modul_yukle(m, str(modul_yol), str(base_yol))  # digest + vocab kapıları içinde
    m.eval()
    A_modul = a_ekseni(m, vocab, device)
    F_modul = uyum_ekseni(m, device, MARANGOZ_BIN)
    del m

    # ---------------- 4) HÜKÜM ----------------
    artis = 100.0 * (A_modul["CE_ort"] / A_taban["CE_ort"] - 1.0)
    a_gec = artis <= ESIK_A_ARTIS
    # Ayrım gücü: A ekseni Wilson yarı-genişliği esik payının altında mı (T-0078 dersi)
    import math
    yari = 1.96 * A_modul["CE_std"] / math.sqrt(A_modul["pencere"])
    pay = (ESIK_A_ARTIS / 100.0) * A_taban["CE_ort"]
    ayirt = yari < pay
    f_dusus = 100.0 * (1.0 - F_modul["CE_ort"] / F_taban["CE_ort"])

    print("\n" + "=" * 68)
    print(f"A ekseni (unutma, Wikipedia, maskesiz)  taban {A_taban['CE_ort']:.4f} "
          f"± {A_taban['CE_std']:.4f}  ->  modül {A_modul['CE_ort']:.4f} "
          f"± {A_modul['CE_std']:.4f}")
    print(f"   artış: {artis:+.2f}%  | ilan edilen eşik <= +{ESIK_A_ARTIS:.1f}%  =>  "
          f"{'GEÇTİ' if a_gec else 'DÜŞTÜ'}")
    print(f"   ayrım gücü: ±{yari:.4f} < eşik payı {pay:.4f}  =>  "
          f"{'AYIRT EDİCİ' if ayirt else 'AYIRT EDEMEZ (hüküm kurulamaz)'}")
    print(f"F ekseni (uyum, marangoz, GENELLEME DEĞİL)  taban {F_taban['CE_ort']:.4f} "
          f"->  modül {F_modul['CE_ort']:.4f}   düşüş {f_dusus:+.2f}%")
    print("=" * 68)

    if cikti:
        os.makedirs(os.path.dirname(os.path.abspath(cikti)), exist_ok=True)
        with open(cikti, "w", encoding="utf-8") as f:
            json.dump({
                "taban": base_yol, "taban_sha256": sha256_dosya(str(base_yol)),
                "modul": modul_yol, "modul_sha256": sha256_dosya(str(modul_yol)),
                "modul_meta": {k: meta[k] for k in ("surum", "spec", "katman_sayisi",
                                                    "modul_parametre") if k in meta},
                "A_taban": A_taban, "A_sifir_kontrol": A_sifir, "A_modul": A_modul,
                "A_artis_yuzde": artis, "A_esik": ESIK_A_ARTIS, "A_gec": a_gec,
                "A_ayirt_edicilik": {"wilson_yari": yari, "esik_payi": pay, "ayirt": ayirt},
                "F_taban": F_taban, "F_modul": F_modul, "F_dusus_yuzde": f_dusus,
                "F_uyari": "uyum (fit) ekseni; GENELLEME ya da yetenek ölçüsü DEĞİL",
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON yazıldı: {cikti}")


if __name__ == "__main__":
    main()
