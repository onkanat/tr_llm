#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0190 regen — B1.5 split'lerin cevap-düzeyi sızıntısını %0'a indirme.

Pool = mevcut donmuş üç dosyanın birebir birliği (yeniden-derleme YOK);
kümeleme = normalize(input) ∪ normalize(output) union-find (T-0012 BETİKTEN
yargısı); bölme küme-atomik seed=42. Eski dosyalar .bak_t0190_ önekine
taşınır (bit-birebir); .pt türevlerine ve yardımcı json'lara dokunulmaz."""
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any

DIZIN = os.path.join("data", "b1_5_splits")
DOSYALAR = ["test.jsonl", "val.jsonl", "train.jsonl"]
ONCESI_SHA = {
    "test.jsonl": "3530de8eaec10829c7f800f3c1266df3dbf3a9466e2ff37ae1f79e11ec263d18",
    "val.jsonl": "9013b0a878ad5643d7bd70cea99032cd7c11e6984e17921f824b75547a5eb2c6",
    "train.jsonl": "7c08c762d27af7daf45d3948c812ebc1068a26049dc098f25614d7b5841eb056",
}  # İLAN-1 BETİKTEN


def sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def normalize_question(q: str) -> str:
    """prepare_b1_5_datasets.py:217-220 birebir."""
    clean = re.sub(r'[\d\'".,!?:;()—–-]+', '', q.lower())
    return " ".join(clean.split())


def ans_norm(o: str) -> str:
    return " ".join(o.strip().lower().split())


def main() -> int:
    pool: List[Dict[str, str]] = []
    for ad in DOSYALAR:
        yol = os.path.join(DIZIN, ad)
        ölç = sha256(yol).replace(" ", "")
        if ölç != ONCESI_SHA[ad]:
            print(f"DUR: {ad} sha İLAN-çıpasıyla uyuşmuyor: {ölç}")
            return 2
        with open(yol, encoding="utf-8") as f:
            satır = [json.loads(s) for s in f if s.strip()]
        print(f"okuma: {ad} {len(satır)} satır")
        pool.extend(satır)
    toplam = len(pool)
    print(f"pool_toplam: {toplam}")

    # --- union-find (normalize soru ∪ cevap) ---
    class UF:
        def __init__(self) -> None:
            self.parent: Dict[Any, Any] = {}

        def find(self, i):
            if self.parent.setdefault(i, i) != i:
                self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j) -> None:
            ri, rj = self.find(i), self.find(j)
            if ri != rj:
                self.parent[ri] = rj

    uf = UF()
    for idx, r in enumerate(pool):
        rek = ("r", idx)
        uf.union(rek, ("q", normalize_question(r.get("input", ""))))
        uf.union(rek, ("a", ans_norm(r.get("output", ""))))
    kumeler: Dict[Any, List[int]] = {}
    for idx in range(toplam):
        kumeler.setdefault(uf.find(("r", idx)), []).append(idx)
    boyutlar = sorted((len(g) for g in kumeler.values()), reverse=True)
    print(f"kume_sayisi: {len(kumeler)} en-buyuk: {boyutlar[:5]} tekli: "
          f"{sum(1 for b in boyutlar if b == 1)}")

    # --- küme-atomik bölme (val 10%, test 10%, seed=42) ---
    rng = random.Random(42)
    anahtarlar = list(kumeler.keys())
    rng.shuffle(anahtarlar)
    val_hedef = int(toplam * 0.10)
    test_hedef = int(toplam * 0.10)
    val_i: List[int] = []
    test_i: List[int] = []
    train_i: List[int] = []
    for k in anahtarlar:
        g = kumeler[k]
        if len(val_i) + len(g) <= val_hedef or not val_i:
            val_i.extend(g)
        elif len(test_i) + len(g) <= test_hedef or not test_i:
            test_i.extend(g)
        else:
            train_i.extend(g)
    eklemler = {"val": val_i, "test": test_i, "train": train_i}
    print("bölme:", {k: len(v) for k, v in eklemler.items()})

    # --- yazım: önce bit-birebir backup, sonra yeni dosya ---
    for ad in DOSYALAR:
        src = os.path.join(DIZIN, ad)
        hedef = os.path.join(DIZIN, "bak_t0190_" + ad)
        shutil.move(src, hedef)  # rename = bit-birebir
        print(f"backup: {ad} -> {hedef} (rename, sha öncesi BETİKTEN-ölçüldü)")

    yeni_icerik = {}
    for ad, indeksler in eklemler.items():
        kayıtlar = [pool[i] for i in indeksler]
        yol = os.path.join(DIZIN, ad + ".jsonl")
        with open(yol, "w", encoding="utf-8") as f:
            for r in kayıtlar:
                f.write(json.dumps(
                    {"instruction": r.get("instruction", ""),
                     "input": r.get("input", ""),
                     "output": r.get("output", "")},
                    ensure_ascii=False) + "\n")
        yeni_icerik[ad] = sha256(yol)

    # --- hüküm-sayımı (İLAN-5 birebir) ---
    tr = [pool[i] for i in train_i]
    te = [pool[i] for i in eklemler["test"]]
    va = [pool[i] for i in eklemler["val"]]
    train_ans_bit = {r["output"] for r in tr}
    train_ans_norm = {ans_norm(r["output"]) for r in tr}
    train_qs = {normalize_question(r["input"]) for r in tr}

    def sayim(hedef: List[dict]) -> dict:
        return {
            "bit": sum(1 for r in hedef if r["output"] in train_ans_bit),
            "normalize": sum(1 for r in hedef
                             if ans_norm(r["output"]) in train_ans_norm),
            "soru": sum(1 for r in hedef
                        if normalize_question(r["input"]) in train_qs),
        }

    sonuc = {
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pool_toplam": toplam,
        "kume_sayisi": len(kumeler),
        "bölme": {k: len(v) for k, v in eklemler.items()},
        "yeni_sha256": yeni_icerik,
        "test_in_train": sayim(te),
        "val_in_train": sayim(va),
    }
    print("HUKUM_JSON:")
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))
    gecti = (sonuc["test_in_train"]["bit"] == 0
             and sonuc["test_in_train"]["normalize"] == 0
             and sonuc["test_in_train"]["soru"] == 0
             and sonuc["val_in_train"]["bit"] == 0
             and sonuc["val_in_train"]["normalize"] == 0
             and sonuc["val_in_train"]["soru"] == 0)
    print("T0190_REGEN_GECTI" if gecti else "T0190_REGEN_DUR")
    return 0 if gecti else 2


if __name__ == "__main__":
    sys.exit(main())