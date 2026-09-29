#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0196 Faz-2 — arz-dengeli külliyat-curation (BETİKTEN; CPU-only).

Kaynak data/b1_5_splits/train.jsonl (donmuş, YALNIZ-OKU). Adım-A:
birebir-output tekilleştir (keep-first); Adım-B: token-düzeyi hiç-
predicate-siz hedeflerin (son-5 TENSE_/COPULA_ yok VE dizide-hiç yok —
T-0195 M2 ölçütü) %50'si seed=42 atla. Çıktı data/eval/
t0196_kulliyat_arz_dengeli.jsonl + BETİKTEN oran-tablosu + sha."""
import json
import os
import random
import sys
import time

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

TRAIN_SHA = "c2d8480b866f9fd0b57d7e671eb091c91e5c80cca5a4f133b62e61cdafaae39e"
TRAIN_PATH = os.path.join(KÖK, "data/b1_5_splits/train.jsonl")
VOCAB_PATH = os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json")
OUT_PATH = os.path.join(KÖK, "data/eval/t0196_kulliyat_arz_dengeli.jsonl")
META_PATH = os.path.join(KÖK, "data/eval/t0196_curation_meta.json")
PREDICATE = ("TENSE_", "COPULA_")


def main() -> int:
    import hashlib
    sha = hashlib.sha256(open(TRAIN_PATH, "rb").read()).hexdigest()
    if sha != TRAIN_SHA:
        print(f"T0196_CURATION_DUR: train sha uymaz: {sha}", flush=True)
        return 2

    recs = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    n0 = len(recs)

    # ---- Adım-A: birebir-output tekilleştir (keep-first) ----
    gör = set()
    adim_a = []
    for r in recs:
        out = r.get("output", "")
        if out in gör:
            continue
        gör.add(out)
        adim_a.append(r)
    n_a = len(adim_a)

    # ---- Adım-B: token-düzeyi predicate-siz (hiç-yok sınıfı) %50 atla ----
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)

    pre_son5 = pre_orta = pre_hic = 0
    predicate_siz = []
    for i, r in enumerate(adim_a):
        tok = [vocab.decode(t) for t in tokenizer.encode(r.get("output", ""))]
        tok = [t for t in tok if t not in ("<BOS>", "<EOS>")]
        son5 = any(t.startswith(PREDICATE) for t in (tok[-5:] if len(tok) >= 5 else tok))
        hic = not any(t.startswith(PREDICATE) for t in tok)
        if son5:
            pre_son5 += 1
        elif not hic:
            pre_orta += 1
        else:
            pre_hic += 1
            predicate_siz.append(i)
        if (i + 1) % 2000 == 0:
            print(f"  B tarayış {i + 1}/{n_a}…", flush=True)

    rng = random.Random(42)
    atla = set(rng.sample(predicate_siz, len(predicate_siz) // 2)) if predicate_siz else set()
    adim_ab = [r for i, r in enumerate(adim_a) if i not in atla]
    n_ab = len(adim_ab)

    post_son5 = post_orta = post_hic = 0
    for r in adim_ab:
        tok = [vocab.decode(t) for t in tokenizer.encode(r.get("output", ""))]
        tok = [t for t in tok if t not in ("<BOS>", "<EOS>")]
        son5 = any(t.startswith(PREDICATE) for t in (tok[-5:] if len(tok) >= 5 else tok))
        if son5:
            post_son5 += 1
        elif any(t.startswith(PREDICATE) for t in tok):
            post_orta += 1
        else:
            post_hic += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in adim_ab:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_sha = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()

    tablo = {
        "task": "T-0196", "faz": "Faz-2", "betik": "curation",
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_sha256": TRAIN_SHA, "n_onesi": n0,
        "adim_a_dedup": {"n": n_a, "silin": n0 - n_a, "baki_pay": n_a / n0 * 100.0},
        "adim_ab": {"predicate_siz": len(predicate_siz), "atlanan": len(atla), "n": n_ab},
        "predicate_onesi": {"son5": pre_son5, "orta": pre_orta, "hic": pre_hic,
                             "son5_pay": (pre_son5 + pre_orta) / n_a * 100.0},
        "predicate_sonrasi": {"son5": post_son5, "orta": post_orta, "hic": post_hic,
                               "hic_pay": post_hic / n_ab * 100.0,
                               "predicate_pay": (post_son5 + post_orta) / n_ab * 100.0},
        "cikti": OUT_PATH, "cikti_sha256": out_sha,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(tablo, f, ensure_ascii=False, indent=2)
    print("[CURATION]", json.dumps(tablo, ensure_ascii=False)[:400], flush=True)
    print(f"T0196_CURATION_GECTI n={n0}->{n_a}->{n_ab} | hic_pay %34.43->{post_hic / n_ab * 100.0:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())