#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5-A — A-KOLU DERİN ANALİZ (ön-kayıt: data/eval/anka_p5akol_derin_analiz_
ilani_2026-09-23.md; görev T-0100). Koşum YOK — CPU ölçüm.

Soru: P5 A-kolu JETON tavanı (0,4164) ne kadarı ham decode kaybıdır? Aynı
orneklem (heldout 539, n=100, seed 42 — P5 sonda betiğiyle birebir) üç
temsilde ölçülür: RAW / DECOMP (kanonik decompiler) / YUZEY (=1,0).
Kayıp ayrışması: UNK · lexicon kök vuruşu · affix çözümleme vs fallback.
TANISAL — eşik önerisi YAPMAZ (ilan §2.5, §4)."""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

from src.compiler.core import CrystalCompiler  # noqa: E402
from src.compiler.decompiler import COMMON_FALLBACKS, MorphemeDecompiler  # noqa: E402
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from scripts.evaluate_b1_5_rigorous import rouge_l_score  # noqa: E402
from scripts.evaluate_carpenter_anka import kelimeler  # noqa: E402  (KANONİK)

VOCAB = os.path.join(KOK, "data/rebuild/vocab_anka_r1_33114.json")
LEXICON = os.path.join(KOK, "data/lexicon/roots_anka_r1.tsv")
HELDOUT = os.path.join(KOK, "data/eval/anka_r17_heldout_2026-09-20.jsonl")
N = 100
SEED = 42
META = {"<BOS>", "<EOS>", "<PAD>"}


def main() -> int:
    vocab = Vocabulary()
    vocab.load(VOCAB, freeze=True)
    lex = LexiconManager()
    lex.load_from_tsv(LEXICON)
    compiler = CrystalCompiler(lex, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=False)
    decompiler = MorphemeDecompiler(compiler, vocab)

    held = [json.loads(l) for l in open(HELDOUT, encoding="utf-8") if l.strip()]
    ornek = random.Random(SEED).sample(held, min(N, len(held)))
    print(f"[kap] sozluk {len(vocab.stoi):,} · orneklem {len(ornek)}", flush=True)

    kayitlar: List[Dict[str, Any]] = []
    kok_vurus, kok_vurus_yok = 0, 0
    kok_yok_sinif: Dict[str, int] = {}
    ek_toplam, ek_graph, ek_fallback = 0, 0, 0
    istisna_sayi = 0
    rouge_raw_l: List[float] = []
    rouge_decomp_l: List[float] = []

    for i, r in enumerate(ornek):
        ref = r["output"]
        ref_k = kelimeler(ref)
        ids = [int(t) for t in tokenizer.encode(ref)]
        gen_tok = [vocab.decode(t) for t in ids]
        gm = " ".join(gen_tok)
        try:
            decomp = decompiler.decompile_sentence(gm)
        except Exception:
            decomp = gm
            istisna_sayi += 1
        rg_raw = rouge_l_score(kelimeler(gm), ref_k)
        rg_dc = rouge_l_score(kelimeler(decomp), ref_k)
        rouge_raw_l.append(rg_raw)
        rouge_decomp_l.append(rg_dc)

        # jeton kalitesi: UNK · kök vuruşu · ek çözümleme
        unk = sum(1 for t in gen_tok if t == "<UNK>")
        rootlar = [t for t in gen_tok
                   if not decompiler.is_suffix(t)
                   and t not in META and t not in {".", ",", "?", "!", ":", ";", "(", ")", "-"}
                   and not t.startswith("<") and not t.endswith(">")]
        hit = miss = 0
        for kok in rootlar:
            bul = compiler.lexicon.find_stems(kok)
            if any(e["lemma"] == kok for _p, e in bul):
                hit += 1
            else:
                miss += 1
        kok_vurus += hit
        kok_vurus_yok += miss
        if miss and len(kayitlar) < 12:
            # örnekleme: vuruşsuz kök sınıfları (etiket adı büyük harf dersi: küçük harfe bak)
            pass
        ekler = [t for t in gen_tok if decompiler.is_suffix(t)]
        ek_toplam += len(ekler)
        for e in ekler:
            if e in decompiler.affix_info:
                ek_graph += 1
            else:
                ek_fallback += 1

        kayitlar.append({
            "idx": i + 1, "instruction": r.get("instruction", ""), "input": r.get("input", ""),
            "ref": ref, "gm": gm, "decomp": decomp,
            "rouge_raw": round(rg_raw, 4), "rouge_decomp": round(rg_dc, 4),
            "delta": round(rg_dc - rg_raw, 4),
            "unk": unk, "ek_sayi": len(ekler), "kok_hit": hit, "kok_miss": miss,
        })
        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{len(ornek)}", flush=True)

    def ort(x: List[float]) -> float:
        return round(sum(x) / len(x), 4)

    n = len(ornek)
    unklu = [k for k in kayitlar if k["unk"] > 0]
    unklu_ort = ort([k["rouge_decomp"] for k in unklu]) if unklu else None
    unksiz_ort = ort([k["rouge_decomp"] for k in kayitlar if k["unk"] == 0]) \
        if len(unklu) < n else None

    ozet = {
        "kap": {"vocab": VOCAB, "lexicon": LEXICON, "heldout": HELDOUT,
                "n": n, "seed": SEED,
                "vocab_boyut": len(vocab.stoi),
                "lexicon_satir": 52582,
                "decompile": "MorphemeDecompiler.decompile_sentence (ECA:482-484 deseni, import)"},
        "temsil_tavani": {
            "YUZEY": 1.0,
            "RAW": {"rouge_ort": ort(rouge_raw_l), "not": "P5 tavanı 0,4164 birebir (bağlaç)"},
            "DECOMP": {"rouge_ort": ort(rouge_decomp_l),
                       "delta_raw": round(ort(rouge_decomp_l) - ort(rouge_raw_l), 4)},
            "raw_geciyor": sum(1 for a, b in zip(rouge_raw_l, rouge_decomp_l) if b < a),
        },
        "jeton_kalitesi": {
            "unk_sayi": sum(k["unk"] for k in kayitlar),
            "unklu_kayit": len(unklu),
            "unklu_ort_decomp_rouge": unklu_ort,
            "unksiz_ort_decomp_rouge": unksiz_ort,
            "kok_vurus": kok_vurus, "kok_vurus_yok": kok_vurus_yok,
            "kok_vurus_orani": round(kok_vurus / max(1, kok_vurus + kok_vurus_yok), 4),
            "ek_toplam": ek_toplam, "ek_graph": ek_graph, "ek_fallback": ek_fallback,
            "ek_graph_orani": round(ek_graph / max(1, ek_toplam), 4),
            "decompile_istisna": istisna_sayi,
        },
        "acik_konu": "kapı ROUGE'unu RAW yerine DECOMP'tan saymak (ECA:481) "
                     "ayrı operatör kararı — bu analizde YAPILMAZ",
    }

    print("\n=== ÜÇ TEMSİL TAVANI (aynı orneklem) ===")
    print(f"  YUZEY : 1.0000 (referansın kendisi)")
    print(f"  RAW   : {ozet['temsil_tavani']['RAW']['rouge_ort']:.4f} (P5 tavanı)")
    d = ozet["temsil_tavani"]["DECOMP"]
    print(f"  DECOMP: {d['rouge_ort']:.4f} (Δ {d['delta_raw']:+.4f}) "
          f"· RAW'ı geçmeyen kayıt: {ozet['temsil_tavani']['raw_geciyor']}/{n}")
    print("=== JETON KALİTESİ ===")
    print(f"  UNK {ozet['jeton_kalitesi']['unk_sayi']} token · "
          f"UNK'lu kayıt {ozet['jeton_kalitesi']['unklu_kayit']}/{n} "
          f"(decomp ROUGE {unklu_ort} vs {unksiz_ort})")
    print(f"  kök vuruş {kok_vurus}/{kok_vurus + kok_vurus_yok} "
          f"({ozet['jeton_kalitesi']['kok_vurus_orani']:.4f}) · "
          f"ek graph {ek_graph}/{ek_toplam} · fallback {ek_fallback} · "
          f"istisna {istisna_sayi}")

    hedef = os.path.join(KOK, "scratch/anka_p5akol_derin_analiz.json")
    with open(hedef, "w", encoding="utf-8") as f:
        json.dump({"ozet": ozet, "kayitlar": kayitlar}, f, ensure_ascii=False, indent=2)
    print(f"[kayit] {hedef}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())