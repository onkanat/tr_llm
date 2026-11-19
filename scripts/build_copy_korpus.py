#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kopya-formatlı ön-eğitim pencereleri üretici (KOPYA-HATTI, v6 hazırlığı).

AMAÇ (ölçülmüş gerekçe — P1 + 07.11.2026 tanısı): ANKA ailesinde endüksiyon/kopya
devresi YOK; RAG koşullandırması belgeyi kopyalamıyor (4-gram örtüşme 0/34). Kopya,
SFT ile değil ÖN-EĞİTİM pencereleriyle kurulur: pencere içinde cevap, belgenin
BİREBİR jeton dizisidir → model ileriye bakarak kopyalamayı öğrenmek zorunda kalır.

PENCERE BİÇİMİ — çıkarım sözleşmesiyle BİREBİR aynı:
  json.dumps({"instruction": ..., "input": "<BELGE> doc </BELGE> query", "output": span})
  + <EOS>   ve TOPLAM ≤ BLOCK (128) jeton.  (build_rag_prompt_tokens ile aynı zarf;
  `input` = prompt_contract.build_rag_input). Böylece ön-eğitimde öğrenilen biçim,
  çıkarımdaki gerçek RAG prompt'unun ta kendisidir.

KAYNAK: data/raw_corpus.txt + data/kulliyat_v2/gts_definitions.txt (külliyat_v2'nin
metin bölgesini üreten iki kaynak; wiki bölgesi satır-bazlı değil).

SIZINTI KORUMASI (iki katman):
  1. SATIR-TUTARLILIK: val .bin'den 8-gram özet kümesi çıkarılır; bu kümede 8-gramı
     bulunan satırlar pencere ÜRETİMİNDE kullanılmaz.
  2. TUTULAN KÜME (K4 kapısı): satır hash'i ile %2 satır ayrılır, bunlar HİÇ
     kullanılmaz; K4 ölçüm seti onlardan üretilir (data/eval/k4_kopya_tutulan.jsonl).

KARIŞIM: her 8 özgün 128-blokta 2 kopya penceresi (%20 blok payı). Blok-hizalı
(çıktı .bin 128'in katıdır), deterministik (tohum sabit).

Koşum:
  venv/bin/python scripts/build_copy_korpus.py \
      --src-bin data/kulliyat_v2/kulliyat_v2.bin \
      --val-bin data/kulliyat_v2/kulliyat_v2_val.bin \
      --out-bin data/kulliyat_v3_copy.bin \
      --tutulan-out data/eval/k4_kopya_tutulan.jsonl \
      --sayi-tutulan 200 [--sinir-blok 2000]   # --sinir-blok = pilot için kısa çıktı
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import build_rag_input

VOCAB_PATH = "data/rebuild/vocab_anka_r1_33114.json"
LEXICON_PATH = "data/lexicon/roots.tsv"
RAW_PATH = "data/raw_corpus.txt"
GTS_PATH = "data/kulliyat_v2/gts_definitions.txt"
BLOCK = 128
TOHUM = 20261107

# Çıkarımda kullanılan talimat; sorgu kalıpları çeşitlendirilir (tek-şablon sızmasını
# önlemek için 4 kalıp — P1 dersi: tek-şablon K3 sonda bandına sızar).
TALIMATLAR = [
    "Belgeye göre cevapla.",
    "Belgedeki bilgiyi kullan.",
    "Yalnız belgeye dayanarak yanıtla.",
]
SORGULAR = [
    "bu belgede ne anlatılır ?",
    "belgedeki tanım nedir ?",
    "bu metne göre hangi bilgi verilir ?",
    "belge hangi konuyu açıklar ?",
]


def _sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def val_ngram_kumesi(val_bin: str, n: int = 8) -> set:
    """Val .bin'in tüm n-gram özetleri — satır-tutarlılık kapısı."""
    mm = np.memmap(val_bin, dtype=np.uint16, mode="r")
    arr = np.asarray(mm, dtype=np.uint16)
    kume = set()
    for i in range(len(arr) - n + 1):
        kume.add(hash(tuple(arr[i:i + n].tolist())))
    return kume


def satirlari_oku() -> list:
    """(metin, kaynak) — külliyat_v2 metin bölgesini üreten kaynaklar."""
    satirlar = []
    if os.path.exists(RAW_PATH):
        for s in open(RAW_PATH, encoding="utf-8"):
            s = s.strip()
            if s:
                satirlar.append((s, "raw"))
    if os.path.exists(GTS_PATH):
        for s in open(GTS_PATH, encoding="utf-8"):
            if "\t" in s:
                anlam = s.split("\t", 1)[1].strip()
                if anlam:
                    satirlar.append((anlam, "gts"))
    return satirlar


def _kirp(ids, span_bas, span_uzun, tavan_doc):
    """Belgeyi span'ı KORUYARAK tavan_doc jetonuna indir (span çevresinden pencere)."""
    if len(ids) <= tavan_doc:
        return ids
    yarim = max(0, (tavan_doc - span_uzun) // 2)
    bas = max(0, span_bas - yarim)
    son = min(len(ids), bas + tavan_doc)
    bas = max(0, son - tavan_doc)
    return ids[bas:son]


def _kelime_spani(satir, rng, kelime_say):
    """Doğal metinde KELİME-sınırlı bitişik bir dilim seç: (alt_metin, bas_krkt, son_krkt)."""
    kelimeler = satir.split()
    if len(kelimeler) < kelime_say:
        return None
    bas = rng.randrange(0, len(kelimeler) - kelime_say + 1)
    alt = " ".join(kelimeler[bas:bas + kelime_say])
    # karakter konumunu bul (kelime sınırlı)
    i = 0
    for _ in range(bas):
        i = satir.index(kelimeler[_], i) + len(kelimeler[_])
    b0 = satir.index(kelimeler[bas], i)
    b1 = b0 + len(alt)
    return alt, b0, b1


def _dogal_doc(satir, b0, b1, hedef_krkt):
    """Satırı cevap dilimini KORUYARAK hedef karakter uzunluğuna kırp (kelime sınırlı)."""
    if len(satir) <= hedef_krkt:
        return satir
    yarim = max(0, (hedef_krkt - (b1 - b0)) // 2)
    a = max(0, b0 - yarim)
    son = min(len(satir), a + hedef_krkt)
    a = max(0, son - hedef_krkt)
    # kelime sınırlarına çek
    while a < b0 and a > 0 and satir[a - 1] != " ":
        a += 1
    while son > b1 and son < len(satir) and satir[son] != " ":
        son -= 1
    return satir[a:son].strip()


def pencere_uret(tok, vocab, satir, rng, sayac=None):
    """Bir satırdan kopya penceresi — DOĞAL METİN (v6 düzeltmesi 07.11.2026).

    ÖNEMLİ: belge ve cevap metni `vocab.decode` ile kurulmaz; kaynak satırın KENDİSİNDEN
    kelime-sınırlı dilim alınır. Morfem-morfoz dizisi ("domates fide si dikim i") çıkarımdaki
    doğal metinle uyuşmuyordu — hem okuryazarlığı bozuyor hem kopya hizalamasını gizliyordu
    (pilot-3: K1 149 adımda eşiği aştı).

    Sözleşme aynen: json{instruction,input:"<BELGE> doc </BELGE> sorgu",output: span} + <EOS>,
    ≤128 jeton; cevap span'ı BELGEDE ve pencerede BİREBİR (jeton düzeyinde) bulunur.
    """
    def _ret(neden):
        if sayac is not None:
            sayac[neden] = sayac.get(neden, 0) + 1
        return None

    ids = tok.encode(satir)[1:-1]  # <BOS>/<EOS> at
    if len(ids) < 16:
        return _ret("satir_kisa")
    if len(satir.split()) < 6:
        return _ret("kelime_az")

    kelime_say = rng.choice([4, 5, 6, 7, 8])
    span = _kelime_spani(satir, rng, kelime_say)
    if span is None:
        return _ret("kelime_az")
    span_txt, b0, b1 = span
    span_ids = tok.encode(span_txt)[1:-1]
    if not (6 <= len(span_ids) <= 30):
        return _ret("span_jeton_bandi")

    for hedef_krkt in (420, 330, 250, 190, 140):
        doc_txt = _dogal_doc(satir, b0, b1, hedef_krkt)
        doc_ids = tok.encode(doc_txt)[1:-1]
        if len(doc_ids) < len(span_ids):
            continue
        talimat = TALIMATLAR[rng.randrange(len(TALIMATLAR))]
        sorgu = SORGULAR[rng.randrange(len(SORGULAR))]
        govde = json.dumps({"instruction": talimat,
                            "input": build_rag_input(doc_txt, sorgu),
                            "output": span_txt}, ensure_ascii=False)
        win = tok.encode(govde)
        if len(win) > BLOCK:
            continue

        def _say(hedef, kaynak):
            n, c = len(hedef), 0
            for i in range(len(kaynak) - n + 1):
                if list(kaynak[i:i + n]) == list(hedef):
                    c += 1
            return c
        if _say(span_ids, doc_ids) < 1:
            continue          # belge kırpması span'ı böldü → daha uzun belge dene
        if _say(span_ids, win) < 2:
            continue          # cevap bölgesi birebir değil → dene
        return {"win": win, "doc_ids": doc_ids, "span_ids": span_ids,
                "doc_txt": doc_txt, "span_txt": span_txt,
                "talimat": talimat, "sorgu": sorgu}
    return _ret("birebir_zarf_kurulamadi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-bin", default="data/kulliyat_v2/kulliyat_v2.bin")
    ap.add_argument("--val-bin", default="data/kulliyat_v2/kulliyat_v2_val.bin")
    ap.add_argument("--out-bin", default="data/kulliyat_v3_copy.bin")
    ap.add_argument("--tutulan-out", default="data/eval/k4_kopya_tutulan.jsonl")
    ap.add_argument("--sayi-tutulan", type=int, default=200)
    ap.add_argument("--grup", type=int, default=8, help="her N özgün blokta")
    ap.add_argument("--kopya", type=int, default=2, help="N blok başına kopya penceresi")
    ap.add_argument("--pencere-per-satir", type=int, default=2,
                    help="bir kaynak satırdan en çok kaç pencere (havuz yeterliği)")
    ap.add_argument("--max-satir", type=int, default=None, help="pilot: havuzu kısalt")
    ap.add_argument("--max-tarama", type=int, default=None,
                    help="pilot: kaynak taramasını ilk N satırla sınırla (val kapısı dahil)")
    ap.add_argument("--havuz-cache", default=None,
                    help="satır havuzunu JSON önbelleğe al / önbellekten yükle "
                         "(tam koşumda tarama maliyetini tekrarlamamak için)")
    ap.add_argument("--yalniz-kopya", type=int, default=None,
                    help="CURRICULUM: yalnız kopya pencerelerinden oluşan N pencerelik .bin "
                         "yaz (kaynak külliyat blokları EKLENMEZ; tekrar öğrenme testi)")
    ap.add_argument("--sinir-blok", type=int, default=None, help="pilot: çıktıyı kısalt")
    ap.add_argument("--seed", type=int, default=TOHUM)
    args = ap.parse_args()

    vocab = Vocabulary(); vocab.load(VOCAB_PATH)
    lex = LexiconManager(); lex.load_from_tsv(LEXICON_PATH)
    tok = KristalTokenizer(CrystalCompiler(lex, build_default_graph()), vocab)
    rng = random.Random(args.seed)

    print("[1/5] val 8-gram kümesi çıkarılıyor…", flush=True)
    yasak = val_ngram_kumesi(args.val_bin)

    print("[2/5] kaynak satırlar okunuyor…", flush=True)
    onbellek = None
    if args.havuz_cache and os.path.exists(args.havuz_cache):
        onbellek = json.load(open(args.havuz_cache, encoding="utf-8"))
        egitim_satir = [tuple(x) for x in onbellek["egitim"]]
        tutulan_satir = [tuple(x) for x in onbellek["tutulan"]]
        val_sizinti = onbellek["val_sizinti"]
        print(f"      ÖNBELLEKTEN: eğitim {len(egitim_satir)} · tutulan {len(tutulan_satir)} · "
              f"val-sızıntısı {val_sizinti}", flush=True)
    else:
        satirlar = satirlari_oku()
        if args.max_tarama:
            satirlar = satirlar[:args.max_tarama]
        print(f"      toplam satır: {len(satirlar)}", flush=True)

        # Sızıntı kapısı + tutulan küme ayrımı (hash ile deterministik)
        egitim_satir, tutulan_satir, val_sizinti = [], [], 0
        for s, kaynak in satirlar:
            ids = tok.encode(s)
            n = 8
            if any(hash(tuple(ids[i:i + n])) in yasak for i in range(max(0, len(ids) - n + 1))):
                val_sizinti += 1
                continue
            (tutulan_satir if int(_sha8(s), 16) % 50 == 0 else egitim_satir).append((s, kaynak))
        if args.havuz_cache:
            json.dump({"egitim": egitim_satir, "tutulan": tutulan_satir,
                       "val_sizinti": val_sizinti, "max_tarama": args.max_tarama},
                      open(args.havuz_cache, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"      havuz önbelleğe yazıldı: {args.havuz_cache}", flush=True)
    print(f"      val-sızıntısı nedeniyle atılan satır: {val_sizinti} · "
          f"eğitim: {len(egitim_satir)} · tutulan(K4): {len(tutulan_satir)}", flush=True)

    print("[3/5] K4 tutulan ölçüm seti yazılıyor…", flush=True)
    os.makedirs(os.path.dirname(args.tutulan_out), exist_ok=True)
    trng = random.Random(args.seed + 1)
    yazilan = 0
    with open(args.tutulan_out, "w", encoding="utf-8") as f:
        for s, kaynak in tutulan_satir:
            if yazilan >= args.sayi_tutulan:
                break
            p = pencere_uret(tok, vocab, s, trng)
            if p is None:
                continue
            f.write(json.dumps({"doc": p["doc_txt"], "span": p["span_txt"],
                                "talimat": p["talimat"], "sorgu": p["sorgu"],
                                "kaynak": kaynak}, ensure_ascii=False) + "\n")
            yazilan += 1
    print(f"      {yazilan} tutulan örnek → {args.tutulan_out}", flush=True)

    print("[4/5] kopya pencereleri üretiliyor + yeni .bin yazılıyor…", flush=True)
    kaynak_bin = np.memmap(args.src_bin, dtype=np.uint16, mode="r")
    toplam_blok = len(kaynak_bin) // BLOCK
    hedef_blok = toplam_blok if args.sinir_blok is None else min(toplam_blok, args.sinir_blok)
    if args.yalniz_kopya:
        hedef_blok = 0
        gerekli = args.yalniz_kopya
    else:
        gerekli = (hedef_blok // args.grup) * args.kopya

    havuz = egitim_satir[:]
    if args.max_satir:
        havuz = havuz[:args.max_satir]
    rng.shuffle(havuz)
    sayac = {}
    pencereler, i = [], 0
    t0 = time.time()
    while len(pencereler) < gerekli and i < len(havuz):
        satir = havuz[i][0]
        for _ in range(max(1, args.pencere_per_satir)):
            p = pencere_uret(tok, vocab, satir, rng, sayac)
            if p is not None:
                w = p["win"]
                pad = BLOCK - len(w)
                pencereler.append(np.array(w + [vocab.stoi["<PAD>"]] * pad, dtype=np.uint16))
                if len(pencereler) >= gerekli:
                    break
        i += 1
        if i % 200 == 0:
            hiz = i / max(1e-9, time.time() - t0)
            print(f"      {i}/{len(havuz)} satır · {len(pencereler)}/{gerekli} pencere "
                  f"({hiz:.0f} satır/sn)", flush=True)
    ret = sum(sayac.values())
    if len(pencereler) < gerekli:
        raise RuntimeError(f"yetersiz kopya penceresi: {len(pencereler)}/{gerekli} "
                           f"(kaynak satır {len(havuz)}, ret {ret}) — bu yol kapalı")

    parcalar, kopya_idx = [], 0
    for b in range(0, hedef_blok, args.grup):
        parcalar.append(np.asarray(kaynak_bin[b * BLOCK:(b + args.grup) * BLOCK]))
        for _ in range(args.kopya):
            if kopya_idx < len(pencereler):
                parcalar.append(pencereler[kopya_idx]); kopya_idx += 1
    # YALNIZ-KOPYA (curriculum) ya da döngüde artan pencereler: kalanları sona ekle
    while kopya_idx < min(len(pencereler), gerekli):
        parcalar.append(pencereler[kopya_idx]); kopya_idx += 1
    if not parcalar:
        raise RuntimeError("çıktı boş: hiç blok üretilmedi (parametreleri denetle)")
    cikti = np.concatenate(parcalar)
    cikti.tofile(args.out_bin)

    meta = {
        "gorev": "kopya_formatli_pencere_karisimi",
        "kaynak_bin": args.src_bin, "kaynak_blok": toplam_blok,
        "kullanilan_blok": hedef_blok,
        "grup": args.grup, "kopya_per_grup": args.kopya,
        "kopya_pencere": kopya_idx,
        "blok_payi_kopya": round(kopya_idx / max(1, kopya_idx + hedef_blok), 5),
        "blok": BLOCK, "seed": args.seed,
        "sozluk": VOCAB_PATH, "sozluk_giris": len(vocab.stoi),
        "sizinti_kapisi": "val 8-gram kumesi (satir duzeyi)",
        "val_sizinti_atilan_satir": val_sizinti,
        "egitim_satir": len(egitim_satir), "tutulan_satir": len(tutulan_satir),
        "k4_tutulan": args.tutulan_out,
        "format": 'json.dumps({instruction, input: "<BELGE> doc </BELGE> query", output: span}) + <EOS>',
        "ret_pencere": ret, "ret_nedenleri": sayac,
        "pencere_per_satir": args.pencere_per_satir,
        "max_tarama": args.max_tarama,
    }
    with open(args.out_bin + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[5/5] yazıldı: {args.out_bin} | {len(cikti):,} jeton | "
          f"kopya pencere {kopya_idx} (%{100*meta['blok_payi_kopya']:.1f} blok payı) | ret {ret} "
          f"| nedenler {sayac}", flush=True)


if __name__ == "__main__":
    main()
