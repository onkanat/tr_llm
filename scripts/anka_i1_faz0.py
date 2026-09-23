#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ANKA_i1 · FAZ 0 ÖN-ÖLÇÜM — talimat aşaması koşumundan ÖNCE, kodsuz.

Plan: `~/.claude/plans/enchanted-wiggling-moon.md` §Faz 0.
Dört ölçüm üretir; hepsi `data/eval/anka_i1_faz0_*.json`'a yazılır:

  1. **PAD sıklığı** — `--no-pad-mask` kararının BEDELİNİ sayısallaştırır. Karar ölçülmüş:
     `train.py:115` CE'ye `ignore_index=<PAD>` verirken `:372` maskeye `-100` yazıyor ⇒ MPS'te
     maske ETKİSİZ. `--no-pad-mask` ile `ignore_index=None` ⇒ CE varsayılanı `-100` ⇒ maske
     DOĞRU; ama PAD konumları kayba girer. Bu ölçüm o bedeli sayıya çevirir.
  2. **Zarf sözleşmesi** — `<OUTPUT>`/`<EOS>` oranları ve her iki rejimde korunan hedef jetonu.
  3. **Sözlük hizası** — külliyat sözlüğü ile hedef sözlük önek-uyumlu mu; ortak jetonlarda
     id kayması var mı; külliyatın `max_id`'si hedef sözlüğün içinde mi.
  4. **Taban çizgisi** — kabul ölçütünün referansı (kanonik kaptan ve tanı betiğinden OKUNUR,
     yeniden ölçülmez; "kayıtlı" olarak işaretlenir).

Kullanım:  venv/bin/python scripts/anka_i1_faz0.py --data data/train_chat_balanced.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_step_demo import mask_prompt_targets  # noqa: E402
from src.llm.tokenizer import Vocabulary  # noqa: E402

HEDEF_SOZLUK = "data/rebuild/vocab_anka_r1_33114.json"
TABAN_CKPT = "data/anka_a1r.pt"
KANONIK_JSON = "data/eval/anka_r23_taban_yetenek_2026-09-21.json"
TANI_JSON = "data/eval/anka_r34_akicilik_taban_2026-09-22.json"
ESIK_A_ARTIS = 10.0            # `evaluate_carpenter_anka.py:68` — elle yazılmaz, sabitten gelir
BLOK = 128
N_ORNEK = 4000
TOHUM = 7


def durdur(mesaj: str) -> None:
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def sha256_dosya(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for par in iter(lambda: f.read(1 << 20), b""):
            h.update(par)
    return h.hexdigest()


def pad_ve_zarf(data: str, vocab: Vocabulary) -> Dict[str, Any]:
    """(1) PAD sıklığı + (2) zarf sözleşmesi ve korunan hedef oranları."""
    PAD = vocab.stoi["<PAD>"]
    OUT = vocab.stoi["<OUTPUT>"]
    EOS = vocab.stoi["<EOS>"]
    mm = np.memmap(data, dtype=np.uint16, mode="r")
    n = int(mm.size)
    if n == 0:
        durdur(f"veri BOŞ: '{data}'")
    pad_say = int((np.asarray(mm) == PAD).sum())
    jeton_max = int(mm[:].max())

    rng = np.random.default_rng(TOHUM)
    n_blok = n // BLOK
    idx = rng.choice(n_blok, size=min(N_ORNEK, n_blok), replace=False)
    w = np.stack([np.asarray(mm[i * BLOK:(i + 1) * BLOK], dtype=np.int64) for i in idx])
    x, y = w[:, :-1], w[:, 1:]
    var = np.array([OUT in row for row in w])

    # İKİ REJİM — farkı ölçmek kararın bedelini verir
    korunan = {}
    t = mask_prompt_targets(x, y, OUT, EOS)
    korunan["pad_maskesi_KAPALI"] = float((t != -100).mean())          # --no-pad-mask (seçilen)
    t2 = t.copy()
    t2[t2 == PAD] = -100
    korunan["pad_maskesi_ACIK"] = float((t2 != -100).mean())

    del mm
    return {
        "data": data, "jeton": n, "blok": n // BLOK, "ornek_blok": int(len(idx)),
        "pad_id": PAD, "output_id": OUT, "eos_id": EOS,
        "pad_jeton": pad_say, "pad_orani_pct": round(100.0 * pad_say / n, 6),
        "max_id": jeton_max,
        "output_icerenn_blok_pct": round(100.0 * float(var.mean()), 4),
        "korunan_hedef_orani": {k: round(100.0 * v, 4) for k, v in korunan.items()},
        "output_suz_blokta_korunan_pct": round(
            100.0 * float((t[~var] != -100).mean()) if (~var).any() else -1.0, 4),
    }


def sozluk_hizasi(data: str, vocab: Vocabulary) -> Dict[str, Any]:
    """(3) Külliyat sözlüğü ↔ hedef sözlük hizası — id kayması ÖLÇÜLÜR."""
    meta_yolu = data + ".meta.json"
    if not os.path.exists(meta_yolu):
        return {"meta": None, "not": "meta yok ⇒ külliyat sözlüğü BEYAN EDİLMEMİŞ"}
    meta = json.load(open(meta_yolu, encoding="utf-8"))
    kaynak = meta.get("sozluk") or meta.get("vocab_path")
    if not kaynak or not os.path.exists(kaynak):
        return {"meta": meta_yolu, "kaynak": kaynak, "not": "külliyat sözlüğü bulunamadı"}
    a = json.load(open(kaynak, encoding="utf-8"))["stoi"]
    b = vocab.stoi
    ortak = set(a) & set(b)
    kayan = [t for t in ortak if a[t] != b[t]]
    return {
        "kaynak_sozluk": kaynak, "kaynak_giris": len(a), "hedef_giris": len(b),
        "ortak_jeton": len(ortak), "id_kayan_jeton": len(kayan),
        "ornek_kayanlar": sorted(kayan)[:5],
        "onek_uyumlu": len(kayan) == 0,
    }


def taban_cizgisi(vocab: Vocabulary) -> Dict[str, Any]:
    """(4) Kabul ölçütünün referansı — KAYITLI sayılardan okunur, yeniden ölçülmez."""
    out: Dict[str, Any] = {"kaynaklar": []}
    if os.path.exists(KANONIK_JSON):
        d = json.load(open(KANONIK_JSON, encoding="utf-8"))
        c = d["ceket_ekseni"]
        out["zarf_tutarsizlik_pct"] = c["tutarsizlik_orani"]
        out["ezber_pct"] = c["ezber_orani"]
        out["rouge_l"] = c["rouge_l_ort"]
        out["kesisim_pct"] = c["kesisim_orani"]
        out["a_ce"] = d["A_ekseni"]["CE_ort"]
        out["kaynaklar"].append(KANONIK_JSON)
    if os.path.exists(TANI_JSON):
        out["ppl"] = json.load(open(TANI_JSON, encoding="utf-8"))["ppl"]
        out["kaynaklar"].append(TANI_JSON)
    if "a_ce" in out:
        out["ppl_kanonik"] = round(float(np.exp(out["a_ce"])), 2)
        out["ce_tavan"] = round(out["a_ce"] * (1.0 + ESIK_A_ARTIS / 100.0), 4)
        out["ce_tavan_kaynagi"] = f"ESIK_A_ARTIS={ESIK_A_ARTIS} (evaluate_carpenter_anka.py:68)"
    out["taban_sha256"] = sha256_dosya(TABAN_CKPT) if os.path.exists(TABAN_CKPT) else None
    out["hedef_sozluk_sha256"] = sha256_dosya(HEDEF_SOZLUK)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Anka_i1 Faz 0 ön-ölçüm")
    ap.add_argument("--data", type=str, default="data/train_chat_balanced.bin")
    ap.add_argument("--output", type=str,
                    default="data/eval/anka_i1_faz0_2026-09-22.json")
    a = ap.parse_args()
    if not os.path.exists(a.data):
        durdur(f"veri yok: '{a.data}'")

    vocab = Vocabulary()
    vocab.load(HEDEF_SOZLUK)
    sonuc: Dict[str, Any] = {
        "gorev": "anka_i1 · Faz 0", "data": a.data, "data_sha256": sha256_dosya(a.data),
        "hedef_sozluk": HEDEF_SOZLUK,
        "pad_zarf": pad_ve_zarf(a.data, vocab),
        "sozluk_hizasi": sozluk_hizasi(a.data, vocab),
        "taban_cizgisi": taban_cizgisi(vocab),
    }

    # KAPILAR (fail-closed) — ölçüm bitti, hüküm burada
    hata: List[str] = []
    if sonuc["pad_zarf"]["max_id"] >= len(vocab.stoi):
        hata.append(f"max_id {sonuc['pad_zarf']['max_id']} >= sözlük {len(vocab.stoi)}")
    if sonuc["sozluk_hizasi"].get("onek_uyumlu") is False:
        hata.append(f"id kayması: {sonuc['sozluk_hizasi']['id_kayan_jeton']} jeton")
    if sonuc["pad_zarf"]["output_icerenn_blok_pct"] < 50.0:
        hata.append(f"<OUTPUT>'lu blok yalnız %{sonuc['pad_zarf']['output_icerenn_blok_pct']}")
    if hata:
        durdur("Faz 0 kapıları: " + " · ".join(hata))
    sonuc["kapilar"] = "GECTI"

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))
    print(f"\n[out] {a.output} sha256={sha256_dosya(a.output)[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
