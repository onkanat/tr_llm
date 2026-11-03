"""Külliyat v2 katmanlı val dilimi üretici (T3.4-pilot-hazırlık, 04.11.2026).

Kaynak: data/kulliyat_v2/kulliyat_v2.bin — 29.135.254 uint16 token,
text++wiki düzenli (text bölgesi dosya başında, wiki devamında).

Yöntem (T3.4-prep): dosya boyunca eşit aralıklı ~20 bitişik dilim.
Dosya K=20 eşit katmana bölünür; her katmanın BAŞINDAN bir bitişik
S_i token'lık dilim alınır ve uç uca eklenir. Böylece val kümesi hem
text hem wiki bölgelerini orantılı temsil eder (katmanlı/stratified
%2 örnekleme). Toplam = int(kaynak_token * 0.02) = 582.705 token.

SIZINTI BEYANI: dilim train bin'den ÇIKARILMADI — %2 sızıntı bilinçli
ve burada belgelenmiştir (1-epokluk devam-ön-eğitim için pragmatik
kabul; T3.4-prep raporu).

Kullanım:
  venv/bin/python scripts/make_val_slice.py \
      --bin data/kulliyat_v2/kulliyat_v2.bin \
      --out data/kulliyat_v2/kulliyat_v2_val.bin
"""
import argparse
import hashlib
import json
import os

import numpy as np

# vocab_anka_r1_33114.json kontrol ID'leri (Vocabulary ile ölçüldü: PAD=1, BOS=2, EOS=3)
BILINEN_ID_ETIKET = {0: "id0", 1: "<PAD>", 2: "<BOS>", 3: "<EOS>"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Külliyat v2 katmanlı val dilimi üretici (T3.4)")
    ap.add_argument("--bin", default="data/kulliyat_v2/kulliyat_v2.bin")
    ap.add_argument("--out", default="data/kulliyat_v2/kulliyat_v2_val.bin")
    ap.add_argument("--katman", type=int, default=20, help="Katman (bitişik dilim) sayısı")
    ap.add_argument("--oran", type=float, default=0.02, help="Val oranı (varsayılan %%2)")
    ap.add_argument("--vocab-size", type=int, default=33114, help="Sınır kapısı: maks jeton id bunun altında olmalı")
    args = ap.parse_args()

    data = np.memmap(args.bin, dtype=np.uint16, mode="r")
    toplam = int(len(data))
    hedef = int(toplam * args.oran)
    katman_sayisi = int(args.katman)
    taban = hedef // katman_sayisi
    artan = hedef - taban * katman_sayisi

    # Katmanlı plan: her katmanın başından bitişik dilim.
    # Tam sayı aritmetiği (i * toplam // katman) — kayan nokta sürprizi yok.
    plan = []
    parcalar = []
    for i in range(katman_sayisi):
        katman_bas = (i * toplam) // katman_sayisi
        uzunluk = taban + (1 if i < artan else 0)
        parcalar.append(np.array(data[katman_bas:katman_bas + uzunluk], dtype=np.uint16))
        plan.append({"dilim": i, "baslangic": katman_bas, "uzunluk": uzunluk})

    val = np.concatenate(parcalar)
    if len(val) != hedef:
        raise RuntimeError(f"Dilim toplamı {len(val)} != hedef {hedef}")
    val.tofile(args.out)
    del val
    del parcalar

    # --- DOĞRULAMA ---
    beklenen_boyut = hedef * 2
    gercek_boyut = os.path.getsize(args.out)
    if gercek_boyut != beklenen_boyut:
        raise RuntimeError(f"Boyut uyumsuz: {gercek_boyut} != token×2 = {beklenen_boyut}")

    kontrol = np.memmap(args.out, dtype=np.uint16, mode="r")
    if int(len(kontrol)) != hedef:
        raise RuntimeError(f"Geri okunan token sayısı {len(kontrol)} != {hedef}")

    # Birebir geri okuma: her dilim kaynak konumuyla karşılaştırılır.
    ofset = 0
    geri_okuma_birebir = True
    for dilim in plan:
        kaynak = np.array(data[dilim["baslangic"]:dilim["baslangic"] + dilim["uzunluk"]])
        yazilan = np.array(kontrol[ofset:ofset + dilim["uzunluk"]])
        if not np.array_equal(kaynak, yazilan):
            geri_okuma_birebir = False
            print(f"[HATA] dilim {dilim['dilim']} geri okumada kaynakla uyuşmuyor")
        ofset += dilim["uzunluk"]

    maks_id_val = int(kontrol.max())
    maks_id_kaynak = int(data.max())

    ozet = {
        "kaynak": args.bin,
        "kaynak_token": toplam,
        "kaynak_maks_jet": maks_id_kaynak,
        "val_oran": args.oran,
        "katman_sayisi": katman_sayisi,
        "val_token": hedef,
        "val_dosya": args.out,
        "val_boyut_bayt": gercek_boyut,
        "boyut_tokenx2_dogrulandi": gercek_boyut == hedef * 2,
        "geri_okuma_birebir": geri_okuma_birebir,
        "val_maks_jet": maks_id_val,
        "val_sha256": sha256_file(args.out),
        "ozel_jet": {BILINEN_ID_ETIKET.get(i, f"id{i}"): int((kontrol == i).sum()) for i in range(4)},
        "sizinti_notu": "val dilimi train bin'den ÇIKARILMADI; %2 sızıntı bilinçli (1-epokluk devam-ön-eğitim için pragmatik kabul, T3.4-prep)",
        "plan": plan,
    }

    if maks_id_val >= args.vocab_size:
        raise RuntimeError(f"DURDURULDU: val maks jeton id {maks_id_val} >= vocab_size {args.vocab_size}")
    if not geri_okuma_birebir:
        raise RuntimeError("DURDURULDU: geri okuma doğrulaması başarısız")

    print(json.dumps(ozet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()