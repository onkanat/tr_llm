#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0195 Faz-1 — yüklemsizlik/arz tanısı (M1-M3, BETİKTEN; CPU-only).

M1: erken-kesme-13 soru→üretim→sınıf tablosu + "root" şablon-sızma ölçütü.
M2: 13.003 train-target encode → predicate-tag payı (son-5) + birebir-
output kalıp-homojenite. M3: 9 uzun-yüklemsiz yüklemler-konumu.
Donmuş yollar yalnız-okuma; hüküm BETİKTEN."""
import json
import os
import sys
import time

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

VOCAB_SHA = "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984"
VOCAB = os.path.join(KÖK, "data/rebuild/vocab_anka_r1_33114.json")
SPLITS = os.path.join(KÖK, "data/b1_5_splits")
KAYIT = os.path.join(KÖK, "data/eval/t0194_giyim_kayitlar.json")
PREDICATE = ("TENSE_", "COPULA_")


def sha(p: str) -> str:
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sinifi(ins: str) -> str:
    if "Kelimedeki kök" in ins:
        return "leksik"
    if "Ahşap" in ins:
        return "marangoz"
    return "tarih"


def main() -> int:
    print("CIPI voc", sha(VOCAB)[:16], flush=True)

    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary
    vocab = Vocabulary()
    vocab.load(VOCAB)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)

    recs = [json.loads(l) for l in open(os.path.join(SPLITS, "test.jsonl"), encoding="utf-8")]
    train = [json.loads(l) for l in open(os.path.join(SPLITS, "train.jsonl"), encoding="utf-8")]
    kay = json.load(open(KAYIT, encoding="utf-8"))["kayitlar"]
    A = [k for k in kay if k["kol"] == "A"]

    # ---- M1 ----
    erken = [k for k in A if k["yuklem_yok"] and k["n_jeton"] <= 12]
    uzun_yuklemsiz = [k for k in A if k["yuklem_yok"] and k["n_jeton"] > 12]
    m1_tablo = []
    root_sizma = 0
    for k in erken:
        r = recs[k["idx"]]
        m1_tablo.append({"idx": k["idx"], "sınıf": sinifi(r["instruction"]),
                         "soru": r["input"][:100], "gen": k["gen_text"][:80],
                         "jeton": k["n_jeton"]})
        if k["gen_text"].split()[:1] == ["root"]:
            root_sizma += 1
    test_root = 0
    for k in A:
        if k["gen_text"].split()[:1] == ["root"]:
            test_root += 1
    print(f"[M1] erken={len(erken)} uzun_yuklemsiz={len(uzun_yuklemsiz)} "
          f"root_sizma={root_sizma}/{len(erken)} | test-100'de root-arzı toplam {test_root}/100", flush=True)
    for t in m1_tablo:
        print("  M1_TABLO", json.dumps(t, ensure_ascii=False), flush=True)

    # ---- M2: 13.003 train-target ----
    print(f"[M2] {len(train)} target encode…", flush=True)
    son5 = 0; hic = 0; orta = 0
    duplicate = {}
    for d in train:
        out = d.get("output", "")
        duplicate[out] = duplicate.get(out, 0) + 1
        tok = [vocab.decode(t) for t in tokenizer.encode(out)]
        tok = [t for t in tok if t not in ("<BOS>", "<EOS>")]
        p5 = any(t.startswith(PREDICATE) for t in (tok[-5:] if len(tok) >= 5 else tok))
        if p5:
            son5 += 1
        elif any(t.startswith(PREDICATE) for t in tok):
            orta += 1
        else:
            hic += 1
        if (son5 + orta + hic) % 2000 == 0:
            print(f"  M2 tarama {(son5+orta+hic)}…", flush=True)
    n = son5 + orta + hic
    print(f"[M2] n={n} | son-5 predicate-tag payı={son5/n*100:.2f}% "
          f"| dizide-orta={orta} | hiç-yok={hic} ({hic/n*100:.2f}%)", flush=True)
    top_dup = sorted(duplicate.items(), key=lambda kv: -kv[1])[:10]
    dups = sum(1 for out, c in duplicate.items() if c > 1)
    print(f"[M2] kalıp-homojenite: birebir-aynı-output {dups} farklı-çıkış / "
          f"{n} satır; top-10:", flush=True)
    for out, c in top_dup:
        print(f"  ×{c} {out[:90]!r}", flush=True)

    # ---- M3: uzun-yüklemsiz yüklemler-konumu ----
    m3 = []
    for k in uzun_yuklemsiz:
        tok = k["gen_text"].split()
        poz = [i for i, t in enumerate(tok) if t.startswith(PREDICATE)]
        m3.append({"idx": k["idx"], "n_jeton": k["n_jeton"], "predicate_sayım": len(poz),
                   "predicate_konum": poz})
        print(f"[M3] idx={k['idx']} jeton={k['n_jeton']} predicate={len(poz)} konum={poz}", flush=True)

    hukum = {
        "task": "T-0195", "faz": "Faz-1", "hukum": "T0195_TANI_KAYDI", "rc": 0,
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "M1": {"erken_kesme": len(erken), "root_sazlon": f"{root_sizma}/{len(erken)}",
               "root_test100": test_root, "_tablo": m1_tablo},
        "M2": {"n": n, "son5_predicate_pay": son5 / n * 100.0,
               "orta": orta, "hic_yok": hic, "hic_yok_pay": hic / n * 100.0,
               "birebir_ayni_cikis_sayim": dups, "top10": [{"x": c, "out": out[:120]} for out, c in top_dup]},
        "M3": m3,
        "cipa": {"vocab": sha(VOCAB), "test": sha(os.path.join(SPLITS, "test.jsonl")),
                 "train": sha(os.path.join(SPLITS, "train.jsonl")), "kayitlar": sha(KAYIT)},
        "ilan": "data/eval/t0195_arz_ilan_2026-09-29.md",
    }
    out_json = "data/eval/t0195_arz_hukum_2026-09-29.json"
    with open(os.path.join(KÖK, out_json), "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2, ensure_ascii=False)
    print(f"[HUKUM] {out_json} yazıldı | damga {hukum['damga']}", flush=True)
    print("HUKUM: T0195_TANI_KAYDI RC=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())