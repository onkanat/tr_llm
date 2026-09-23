#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 — ÖLÇÜT TAZELEME TAVAN SONDASI (ön-kayıt: data/eval/anka_p5_olcut_
tazeleme_ilani_2026-09-23.md; görev T-0099). Koşum YOK — CPU ölçüm.

İki kol:
  A-kolu: heldout 539 reproduce (kapanmış kayıt scratch/t0096_kos/
          kesisim_sondasi.py'nin JETON tavanları) — FAIL-CLOSED: beklenen
          değerler ilanlı bantların dışına çıkarsa rc=2.
  B-kolu: arena_base temiz kayıtlardan n=100 seed 42 — ROUGE/tutarsızlık
          borusu-tavanları yenilenir; kesişim TANISAL (domain-uyuşmaz).

NEDEN KOPYA: kaynak sonda scratch/t0096_kos/kesisim_sondasi.py KAPANMIŞ
kayıttır (scratch/t0096_* SALT OKUNUR; "kapalı kayıt düzenlenmez, desen
kopyalanır"). `kelimeler` + `lcs_f1` ECA'dan, `rouge_l_score`
evaluate_b1_5_rigorous'tan IMPORT edilir (kopya YASAK).
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

from src.compiler.core import CrystalCompiler  # noqa: E402
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from scripts.evaluate_b1_5_rigorous import rouge_l_score  # noqa: E402
from scripts.evaluate_carpenter_anka import kelimeler, lcs_f1  # noqa: E402  (KANONİK)

VOCAB = os.path.join(KOK, "data/rebuild/vocab_anka_r1_33114.json")
LEXICON = os.path.join(KOK, "data/lexicon/roots_anka_r1.tsv")
HELDOUT = os.path.join(KOK, "data/eval/anka_r17_heldout_2026-09-20.jsonl")
ARENA_BASE = os.path.join(KOK, "data/pedagogy/arena_base_accumulated.jsonl")
ARENA_CARP = os.path.join(KOK, "data/pedagogy/arena_carpenter_accumulated.jsonl")
KAYNAK = os.path.join(KOK, "data/pedagogy/carpenter_specialization_dataset.jsonl")
N = 100
SEED = 42

# --- ilanlı eşik kaynakları (kapanmış kayıt; yalnız A-kolu kabul bandı) ---
BANK_ROUGE = 0.4164   # kapanmış kayıt JETON tavanı
BANK_TUTARSIZ = 3.0
BANK_KESISIM = 13.0
BANK_EZBER = 0.0

# P2/0c katsayıları — DAL-K girdisi; dal kararı raporda
K_ROUGE = 0.84
K_TUTARSIZ = 1.67
PREDICATE_TAGS = ("TENSE_", "COPULA_")


def wilson(k: int, n: int, z: float = 1.96) -> float:
    """Wilson %95 yarı-genişlik (ilan §3: oran ölçütleri için)."""
    if n == 0:
        return 0.0
    p = k / n
    pay = p * (1.0 - p) / n + z * z / (4.0 * n * n)
    return z * math.sqrt(pay) / (1.0 + z * z / n)


def dortgram_kumesi(kayitlar: List[dict]) -> set:
    s = set()
    for r in kayitlar:
        w = kelimeler(r.get("output", ""))
        for i in range(len(w) - 3):
            s.add(tuple(w[i:i + 4]))
    return s


def oku_jsonl(yol: str) -> List[dict]:
    with open(yol, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def carve_denetimi() -> Dict[str, Any]:
    """Çıktı-düzeyi carve: arena kaydının `output`'u havuzda geçiyorsa KİRLİ.
    Havuz beyanı ilan §2'nin 5 üyesi; kanıt JSON'a yazılır."""
    havuz = [os.path.join(KOK, "data/pedagogy/carpenter_specialization_dataset.jsonl"),
             os.path.join(KOK, "data/pedagogy/high_school_foundation_dataset.jsonl"),
             os.path.join(KOK, "data/pedagogy_canonical/carpenter_canonical.jsonl"),
             os.path.join(KOK, "data/pedagogy_canonical/high_school_canonical.jsonl"),
             HELDOUT]
    havuz_n, havuz_cikti = 0, set()
    for y in havuz:
        kayitlar = oku_jsonl(y)
        havuz_n += len(kayitlar)
        havuz_cikti |= {r["output"] for r in kayitlar if r.get("output")}
    rapor: Dict[str, Any] = {"havuz_kayit": havuz_n, "havuz_ozgun_cikti": len(havuz_cikti),
                             "havuz_uyeleri": [os.path.relpath(y, KOK) for y in havuz]}
    for ad, yol in (("arena_base", ARENA_BASE), ("arena_carpenter", ARENA_CARP)):
        kayitlar = oku_jsonl(yol)
        temiz = [r for r in kayitlar if r.get("output") and r["output"] not in havuz_cikti]
        domainler = sorted({r.get("domain", "?") for r in temiz})
        rapor[ad] = {"toplam": len(kayitlar), "kirli": len(kayitlar) - len(temiz),
                     "temiz": len(temiz), "temiz_domain": domainler}
        if ad == "arena_base":
            rapor["arena_base_temiz"] = temiz
    return rapor


def olc(tokenizer, vocab, ornek: List[dict], train_4g: set, etiket: str) -> Dict[str, Any]:
    """P2/kesisim_sondasi deseninin birebir mekanigi: YUZEY + JETON."""
    sonuc = {"YUZEY": {"ezber": 0, "tutarsiz": 0, "kesisim": 0, "rouges": [],
                       "kes_f1ler": []},
             "JETON": {"ezber": 0, "tutarsiz": 0, "kesisim": 0, "rouges": [],
                       "kes_f1ler": []}}
    for i, r in enumerate(ornek):
        ref_k = kelimeler(r["output"])
        inp_k = set(kelimeler(r.get("input", "")))
        ids = [int(t) for t in tokenizer.encode(r["output"])]
        gen_tok = [vocab.decode(t) for t in ids]

        for ad, gw in (("YUZEY", list(ref_k)),
                       ("JETON", kelimeler(" ".join(gen_tok)))):
            k = sonuc[ad]
            cg = [tuple(gw[x:x + 4]) for x in range(len(gw) - 3)]
            if cg and (sum(1 for g in cg if g in train_4g) / len(cg)) >= 0.90:
                k["ezber"] += 1
            son = gen_tok[-5:] if ad == "JETON" else ref_k[-5:]
            yuklem = any(t.startswith(PREDICATE_TAGS) for t in son)
            kaynaklar = gen_tok if ad == "JETON" else ref_k
            dongu = any(kaynaklar[x:x + 2] == kaynaklar[x + 2:x + 4] == kaynaklar[x + 4:x + 6]
                        for x in range(max(0, len(kaynaklar) - 5)))
            if (not yuklem) or dongu:
                k["tutarsiz"] += 1
            if len(inp_k & set(gw)) >= 2:
                k["kesisim"] += 1
            k["rouges"].append(rouge_l_score(gw, ref_k))
            k["kes_f1ler"].append(lcs_f1(list(kelimeler(r.get("input", ""))), gw))
        if (i + 1) % 25 == 0:
            print(f"  [{etiket}] … {i + 1}/{len(ornek)}", flush=True)

    n = len(ornek)
    tablo: Dict[str, Dict[str, Any]] = {}
    for ad, k in sonuc.items():
        rg = k["rouges"]
        f1 = k["kes_f1ler"]
        tablo[ad] = {
            "rouge_l_ort": round(sum(rg) / n, 4),
            "rouge_l_std": round((sum((x - sum(rg) / n) ** 2 for x in rg) / n) ** 0.5, 4),
            "tutarsizlik_orani": round(100.0 * k["tutarsiz"] / n, 4),
            "tutarsizlik_sayi": k["tutarsiz"],
            "kesisim_orani": round(100.0 * k["kesisim"] / n, 4),
            "kesisim_sayi": k["kesisim"],
            "ezber_orani": round(100.0 * k["ezber"] / n, 4),
            "kesisim_f1_ort": round(sum(f1) / n, 4),  # TANISAL (P2/0b)
        }
    return tablo


def main() -> int:
    import numpy as np  # noqa: E402

    vocab = Vocabulary()
    vocab.load(VOCAB, freeze=True)
    lex = LexiconManager()
    lex.load_from_tsv(LEXICON)
    compiler = CrystalCompiler(lex, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=False)

    carve = carve_denetimi()
    temiz = carve.pop("arena_base_temiz")
    egitim = oku_jsonl(KAYNAK)
    train_4g = dortgram_kumesi(egitim)
    print(f"[kap] sozluk {len(vocab.stoi):,} · carve {carve['arena_base']['temiz']}/"
          f"{carve['arena_base']['toplam']} temiz · egitim 4-gram {len(train_4g):,}", flush=True)

    held = oku_jsonl(HELDOUT)
    rng_a = random.Random(SEED)
    ornek_a = rng_a.sample(held, min(N, len(held)))
    tablo_a = olc(tokenizer, vocab, ornek_a, train_4g, "A")

    rng_b = random.Random(SEED)
    ornek_b = rng_b.sample(temiz, min(N, len(temiz)))
    tablo_b = olc(tokenizer, vocab, ornek_b, train_4g, "B")

    # --- A-kolu FAIL-CLOSED: ilanlı kabul bantları ---
    j = tablo_a["JETON"]
    rouge_se_yarim = 1.96 * j["rouge_l_std"] / math.sqrt(N)
    bantlar = {
        "rouge_l_ort": {"banka": BANK_ROUGE, "olculen": j["rouge_l_ort"],
                        "yarim_genislik": round(rouge_se_yarim, 4),
                        "tanim": "±1.96·std/√n"},
        "tutarsizlik_orani": {"banka": BANK_TUTARSIZ, "olculen": j["tutarsizlik_orani"],
                              "yarim_genislik": round(wilson(j["tutarsizlik_sayi"], N) * 100.0, 4),
                              "tanim": "Wilson %95"},
        "kesisim_orani": {"banka": BANK_KESISIM, "olculen": j["kesisim_orani"],
                          "yarim_genislik": round(wilson(j["kesisim_sayi"], N) * 100.0, 4),
                          "tanim": "Wilson %95"},
        "ezber_orani": {"banka": BANK_EZBER, "olculen": j["ezber_orani"],
                        "yarim_genislik": 0.0, "tanim": "birebir"},
    }
    hatalar = []
    for ad_olcut, b in bantlar.items():
        fark = abs(b["olculen"] - b["banka"])
        b["mutlak_fark"] = round(fark, 4)
        ok = fark <= max(b["yarim_genislik"], 1e-9)
        b["kabul"] = ok
        if not ok:
            hatalar.append(ad_olcut)

    # --- Dal girdileri (karar raporda) ---
    dal_k_esik_adayi = {
        "rouge": round(tablo_b["JETON"]["rouge_l_ort"] * K_ROUGE, 4),
        "tutarsizlik": round(tablo_b["JETON"]["tutarsizlik_orani"] * K_TUTARSIZ, 4),
    }

    cikti = {
        "kap": {"vocab": VOCAB, "lexicon": LEXICON, "heldout": HELDOUT,
                "arena_base": ARENA_BASE, "arena_carpenter": ARENA_CARP,
                "n": N, "seed": SEED,
                "olcut": "kelimeler+lcs_f1 (ECA import) · rouge_l_score (b1_5 import) · JETON kapı"},
        "carve": carve,
        "a_kolu": {"kaynak": "heldout 539", "tavan": tablo_a,
                   "kabul_banti": bantlar, "fail_closed_hatalar": hatalar},
        "b_kolu": {"kaynak": "arena_base temiz (highschool_genz)", "tavan": tablo_b,
                   "kesisim": "TANISAL — domain-uyuşmaz, eşik kaynağı OLAMAZ (ilan §2)"},
        "dal_k_esik_adayi": dal_k_esik_adayi,
        "katsayilar": {"rouge": K_ROUGE, "tutarsizlik": K_TUTARSIZ},
    }

    print("\n=== A-KOLU (heldout reproduce — kapanmış kayıt karşılaştırması) ===")
    for ad_olcut, b in bantlar.items():
        print(f"  {ad_olcut:18s} banka {b['banka']:>8.4f} · ölçülen {b['olculen']:>8.4f} "
              f"· fark {b['mutlak_fark']:.4f} · bant ±{b['yarim_genislik']:.4f} "
              f"({'OK' if b['kabul'] else '*** BANT DIŞI ***'})")
    print("\n=== B-KOLU (arena_base temiz — yeni tavan adayı) ===")
    for ad, t in tablo_b.items():
        print(f"  {ad}: ROUGE {t['rouge_l_ort']:.4f} · tutarsızlık %{t['tutarsizlik_orani']:.2f} "
              f"· kesişim %{t['kesisim_orani']:.2f} (tanısal) · ezber %{t['ezber_orani']:.2f}")
    print(f"\n[HUKUM] A-kolu bant dışı: {hatalar or 'YOK'}")
    print(f"[DAL] {'DAL-T (eşik değişmez)' if not hatalar else 'DAL-K adayı: ' + str(dal_k_esik_adayi)}")

    hedef = os.path.join(KOK, "scratch/anka_p5_tavan_sondasi.json")
    with open(hedef, "w", encoding="utf-8") as f:
        json.dump(cikti, f, ensure_ascii=False, indent=2)
    print(f"[kayit] {hedef}")
    return 2 if hatalar else 0


if __name__ == "__main__":
    raise SystemExit(main())