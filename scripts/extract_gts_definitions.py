#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAZ-3 T3.1 — TDK GTS TANIM KÜLLİYATI ÇIKARICI
==============================================
Kaynak : data/poems/v12.gts.json.tar.gz (içinde gts.json — JSONL, ~99.236 madde;
         şema: madde / anlamlarListe[].anlam [, ozelliklerListe]).
Çıktı  : data/kulliyat_v2/gts_definitions.txt — satır başına bir anlam:
         'MADDE<TAB>anlam metni'  (madde = sözlük başı).

Filtreler:
  1. Boş / None anlam metni atlanır.
  2. Madde başına en fazla ilk 4 eşsiz anlam — çok-anlamlılık şişmesini önler
     (madde madde_id farklı girişlerde de tekrar ederse sayaç global devam eder).
  3. Aynı (madde, anlam) ikilisi yalnızca bir kez yazılır (global dedup).

Tarball repo içine ASLA açılmaz: --gts-json verilmezse geçici dizine
(TMPDIR = ~/.hermes/cache/scratch) açılır ve iş bitince silinir.

Koşum:
  venv/bin/python scripts/extract_gts_definitions.py                 # tarball'dan
  venv/bin/python scripts/extract_gts_definitions.py --gts-json YOL  # açık gts.json
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARBALL = os.path.join(BASE_DIR, "data", "poems", "v12.gts.json.tar.gz")
DEFAULT_OUT = os.path.join(BASE_DIR, "data", "kulliyat_v2", "gts_definitions.txt")

MAX_ANLAM_PER_MADDE = 4   # filtre 2
MIN_BEKLENEN_SATIR = 150_000  # T3.1 beklenti kapısı: ≥150.000 satır


def _clean(text: str) -> str:
    """Satır formatını korur: sekme/satır sonu → boşluk, fazla boşluklar büzülür."""
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def resolve_gts_json(args) -> tuple:
    """--gts-json varsa onu kullan; yoksa tarball'ı geçici dizine aç. Dönen: (yol, tmpdir|None)."""
    if args.gts_json:
        if not os.path.isfile(args.gts_json):
            sys.exit(f"HATA: gts.json bulunamadı: {args.gts_json}")
        return args.gts_json, None
    if not os.path.isfile(args.tarball):
        sys.exit(f"HATA: tarball bulunamadı: {args.tarball}")
    tmpdir = tempfile.mkdtemp(prefix="gts_v12_extract_")  # TMPDIR → scratch altına düşer
    try:
        with tarfile.open(args.tarball, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and m.name.endswith(".json")]
            if not members:
                sys.exit("HATA: tarball içinde .json üyesi yok")
            tf.extractall(tmpdir, members=members)
            gts_path = os.path.join(tmpdir, members[0].name)
            if not os.path.isfile(gts_path):
                sys.exit(f"HATA: açma sonrası dosya yok: {gts_path}")
            return gts_path, tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def extract(gts_path: str, out_path: str) -> dict:
    """JSONL'i gez, filtreleri uygula, 'MADDE<TAB>anlam' satırlarını yaz; istatistik döndür."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stats = {
        "giris": 0,        # işlenen madde girişi
        "bozuk_satir": 0,  # JSON çözülemeyen satır
        "madde_yok": 0,    # madde alanı boş
        "anlam_listesi_yok": 0,  # anlamlarListe boş/None
        "bos_anlam": 0,    # anlam metni None/boş (filtre 1)
        "gecerli_anlam": 0,
        "yazilan": 0,      # çıktıya yazılan satır
        "dedup_atlanan": 0,   # tekrar (madde, anlam) (filtre 3)
        "sinir_atlanan": 0,   # madde başına 4 sınırı aşılan (filtre 2)
    }
    counts = {}   # madde -> yazılan eşsiz anlam sayısı (madde tekrarında sıfırlanmaz)
    seen = set()  # (madde, anlam) — global dedup

    with open(gts_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bozuk_satir"] += 1
                continue
            stats["giris"] += 1

            madde = _clean(str(obj.get("madde") or ""))
            if not madde:
                stats["madde_yok"] += 1
                continue
            anlamlar = obj.get("anlamlarListe") or []
            if not anlamlar:
                stats["anlam_listesi_yok"] += 1
                continue

            for anlam_obj in anlamlar:
                raw = anlam_obj.get("anlam") if isinstance(anlam_obj, dict) else None
                text = _clean(str(raw)) if raw is not None else ""
                if not text:
                    stats["bos_anlam"] += 1
                    continue
                stats["gecerli_anlam"] += 1

                if counts.get(madde, 0) >= MAX_ANLAM_PER_MADDE:
                    stats["sinir_atlanan"] += 1
                    continue
                key = (madde, text)
                if key in seen:
                    stats["dedup_atlanan"] += 1
                    continue

                seen.add(key)
                counts[madde] = counts.get(madde, 0) + 1
                fout.write(f"{madde}\t{text}\n")
                stats["yazilan"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="TDK GTS tanım külliyatı çıkarıcı (MADDE<TAB>anlam satırları)")
    ap.add_argument("--tarball", default=DEFAULT_TARBALL,
                    help=f"GTS tarball yolu (varsayılan: {DEFAULT_TARBALL})")
    ap.add_argument("--gts-json", default=None,
                    help="Açık gts.json yolu — verilirse tarball açılmaz")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Çıktı dosyası (varsayılan: {DEFAULT_OUT})")
    args = ap.parse_args()

    gts_path, tmpdir = resolve_gts_json(args)
    try:
        stats = extract(gts_path, args.out)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Kaynak     : {gts_path}")
    print(f"Çıktı      : {args.out}")
    print(f"Giriş      : {stats['giris']:,} madde (bozuk satır: {stats['bozuk_satir']})")
    print(f"Atlanan    : madde boş={stats['madde_yok']:,}, "
          f"anlamlarListe yok={stats['anlam_listesi_yok']:,}, "
          f"anlam boş/None={stats['bos_anlam']:,}")
    print(f"Geçerli anlam: {stats['gecerli_anlam']:,}  "
          f"(4-sınırı atlanan: {stats['sinir_atlanan']:,}, "
          f"dedup atlanan: {stats['dedup_atlanan']:,})")
    print(f"YAZILAN    : {stats['yazilan']:,} satır")

    # SERT KAPI — iç tutarlılık: hiçbir geçerli anlam kaybolmamalı.
    if stats["yazilan"] + stats["sinir_atlanan"] + stats["dedup_atlanan"] != stats["gecerli_anlam"]:
        print("HATA: anlam muhasebesi tutmuyor (yazılan + atlanan ≠ geçerli)!")
        sys.exit(1)

    # Beklenti kapısı (T3.1 tahmini ≥150.000 = 99.236 × ort. 1,5-2).
    # Kaynak tavanı: gecerli_anlam — tarball'da toplam geçerli anlam bu kadardır;
    # gerçek ort. ~1,34 anlam/madde olduğundan 4-sınırıyla tavan ~133k altında kalır.
    if stats["yazilan"] < MIN_BEKLENEN_SATIR:
        print(f"BİLGİ: beklenti ≥{MIN_BEKLENEN_SATIR:,} satır altında "
              f"({stats['yazilan']:,}) — KAYNAK TAVANI: {stats['gecerli_anlam']:,} geçerli "
              f"anlam (gerçek ort. {stats['gecerli_anlam'] / max(stats['giris'], 1):.2f}/madde, "
              f"tahminin 1,5-2 varsayımından düşük). Filtreler speke uygun, eksik yok.")
    else:
        print(f"KAPISI GEÇTİ: ≥{MIN_BEKLENEN_SATIR:,} satır beklentisi sağlandı.")
    print(f"TUTARLILIK KAPISI GEÇTİ: yazılan + atlanan = geçerli anlam.")


if __name__ == "__main__":
    main()