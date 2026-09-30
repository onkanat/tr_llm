#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0200 Faz-3 — kabul-ölçümü (BETİKTEN; MPS; optimum-zarar onarım modeli).

t0198_faz3_kabul kalıbı birebir: arm-A `calis_arm` YENİDE-KULLANIM, N=100/seed-42
eşli; train_4grams ORİJİNAL regen-train'den. Eşik-set T-0198 birebir-devralınan:
Ezber ≤5 VE Tutarsızlık <28 VE decomp ≥0,35 VE yüklemsiz-pir <28 VE erken-kes-pir
<17 → T0200_ONARIM_GECTI rc=0; aksi DUR rc=2. BETİKTEN-ekler (şartname MADDE-12):
- tutarsızlık ≥30 ⇒ OTOMATİK-DUR (T-0198 zincir-sonu; lt28 kapısının fail-closed-tamamı)
- başarı-beyanı: tutarsızlık ≤22 (T-0192 çıpası) ayrı-rapor alanı (kapı DEĞİL)
- YENİ-ÇIPA (kayıt, kapı DEĞİL): M1-ÇEVRİM — taban_akicilik_tanisi subprocess
  --model data/anka_t0200.pt → data/eval/t0200_m1_t0200.json; beklenen-kayıt
  ppl < 50,75 (halef-beyanı; hüküm-formülüne GİRMEZ)."""
import json
import os
import random
import subprocess
import sys
import time
import torch

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict
from scripts.evaluate_anka_b1_5 import DEVICE, DATA_DIR, SPLITS_DIR, compute_sha256
from scripts.t0194_giyim_testi import calis_arm, kelime  # halef — yalnız-OKU
from scripts.train_t0198_hiza import HALEF_SHA  # halef-çıpa — yalnız-OKU

ESIK = {"ezber_max": 5.0, "tutarsizlik_max": 28.0, "decomp_rouge_min": 0.35,
         "yuklemsiz_pir_max": 28.0, "erken_kes_pir_max": 17.0}
CIPA = {"tutarsizlik_t0192": 22.0, "tutarsizlik_t0197_eqigi": 28.0, "zincir_son_t0198": 30.0,
         "halef_ppl_beyan": 50.75, "decomp_rouge": 0.1165,
         "dil_hizali": {"ing_hedef": {"n": 16, "raw_ort": 0.0467},
                         "turkce_ref": {"n": 84, "raw_ort": 0.1078}}}
ONARIM_PATH = os.path.join(KÖK, "data/anka_t0200.pt")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0200_faz3_hukum_2026-09-30.json")
KAYIT_PATH = os.path.join(KÖK, "data/eval/t0200_faz3_kayitlar.json")
EGITIM_HUKUM = os.path.join(KÖK, "data/eval/t0200_egitim_hukum_2026-09-30.json")
AKICILIK_BETIK = os.path.join(KÖK, "scripts", "taban_akicilik_tanisi.py")
M1_OUTPUT = os.path.join(KÖK, "data/eval/t0200_m1_t0200.json")


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    eg = json.load(open(EGITIM_HUKUM, encoding="utf-8"))
    if eg.get("hukum") != "T0200_EGITIM_GECTI":
        print(f"T0200_FAZ3_DUR: eğitim hükmü {eg.get('hukum')}", flush=True)
        return 2
    if DEVICE.type != "mps":
        print(f"T0200_FAZ3_DUR: MPS yok ({DEVICE})", flush=True)
        return 2
    onarim_sha = compute_sha256(ONARIM_PATH)
    if onarim_sha != eg["best"]["sha256"]:
        print(f"T0200_FAZ3_DUR: onarım sha uymaz {onarim_sha}", flush=True)
        return 2
    if compute_sha256(os.path.join(KÖK, "data/anka_b1_5_best.pt")) != HALEF_SHA:
        print("T0200_FAZ3_DUR: halef sha değişti", flush=True)
        return 2

    train_rec = [json.loads(l) for l in open(os.path.join(SPLITS_DIR, "train.jsonl"), encoding="utf-8")]
    test_records = [json.loads(l) for l in open(os.path.join(SPLITS_DIR, "test.jsonl"), encoding="utf-8")]
    eval_records = random.Random(42).sample(test_records, min(100, len(test_records)))

    train_4grams = set()
    for d in train_rec:
        w = kelime(d.get("output", ""))
        for i in range(len(w) - 3):
            train_4grams.add(tuple(w[i:i + 4]))

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "rebuild", "vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data/lexicon/roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    decompiler = MorphemeDecompiler(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(ONARIM_PATH, map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print("[MODELI] onarım=anka_t0200 yüklendi", flush=True)

    kayit, ozet = calis_arm("A", eval_records, tokenizer, vocab, model, train_4grams,
                            decompiler, vocab.stoi.get("<EOS>", 3), vocab.stoi.get("</OUTPUT>", 9), [0])
    print("OZET:", json.dumps(ozet, ensure_ascii=False), flush=True)

    yuklemsiz = sum(r["yuklem_yok"] for r in kayit) * 100.0 / len(kayit)
    dongu = sum(r["dongu"] for r in kayit) * 100.0 / len(kayit)
    erken_kes = sum(1 for r in kayit
                    if not r["decomp_text"].rstrip().endswith((".", "!", "?", ":"))) * 100.0 / len(kayit)

    ezber = ozet["memorization_rate"]
    tutarsiz = ozet["incoherence_rate"]
    kapilar = {"ezber_le5": ezber <= ESIK["ezber_max"],
               "tutarsizlik_lt28": tutarsiz < ESIK["tutarsizlik_max"],
               "tutarsizlik_ge30_otomatik_dur": tutarsiz < CIPA["zincir_son_t0198"],
               "decomp_ge0_35": ozet["mean_rouge_l_decomp"] >= ESIK["decomp_rouge_min"],
               "yuklemsiz_pir_lt28": yuklemsiz < ESIK["yuklemsiz_pir_max"],
               "erken_kes_pir_lt17": erken_kes < ESIK["erken_kes_pir_max"]}
    basari_beyani = {"tutarsizlik_le22_t0192": tutarsiz <= CIPA["tutarsizlik_t0192"],
                      "beyan": "kapı DEĞİL — zincir-çıpa %22'ye dönüş-beyanı"}

    # ---- YENİ-ÇIPA: M1-ÇEVRİM (kayıt; hüküm-formülüne GİRMEZ) ----
    m1 = None
    olcum = subprocess.run([sys.executable, AKICILIK_BETIK, "--model", "data/anka_t0200.pt",
                            "--device", "mps", "--output", M1_OUTPUT],
                           capture_output=True, text=True, cwd=KÖK)
    if olcum.returncode == 0:
        m1 = json.load(open(M1_OUTPUT, encoding="utf-8"))
        print(f"[M1-ÇEVRİM] t0200 ppl={m1['ppl']} ce={m1['ce_ort']} "
              f"(beklenen-kayıt ppl<50,75: {m1['ppl'] < 50.75})", flush=True)
    else:
        print(f"[M1-ÇEVRİM-ISTISNA] rc={olcum.returncode} "
              f"{olcum.stderr.strip()[-200:]}", flush=True)

    gecti = all(kapilar.values())
    hukum = "T0200_ONARIM_GECTI" if gecti else "T0200_ONARIM_DUR"
    donem = {
        "task": "T-0200", "hukum": hukum, "rc": 0 if gecti else 2,
        "esikler": ESIK,
        "olculen": {"ezber_rate": ezber, "tutarsizlik_rate": tutarsiz,
                     "decomp_rouge": ozet["mean_rouge_l_decomp"],
                     "raw_rouge": ozet["mean_rouge_l"],
                     "yuklemsiz_pir": yuklemsiz, "erken_kes_pir": erken_kes,
                     "dongu_pir": dongu},
        "kapi_dersleri": kapilar,
        "basari_beyani": basari_beyani,
        "m1_cevrim": {"kayit": M1_OUTPUT,
                       "ppl": m1["ppl"] if m1 else None,
                       "ce_ort": m1["ce_ort"] if m1 else None,
                       "halef_ppl_beyan": CIPA["halef_ppl_beyan"],
                       "ppl_lt_halef": (m1["ppl"] < 50.75) if m1 else None,
                       "kayit_turu": "kapı DEĞİL — taban-akıcılık ÇEVRİM-kanıtı"},
        "cipa_tablo": CIPA,
        "egitim_hukum": eg.get("hukum"), "egitim_sha256": onarim_sha,
        "history_path": "data/eval/t0200_training_history.json",
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ilan": "data/eval/t0200_egitim_ilan_2026-09-30.md",
    }
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    with open(KAYIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"ozet": ozet, "kayitlar": kayit,
                    "m1_cevrim": donem["m1_cevrim"]}, f, indent=2, ensure_ascii=False)
    print(f"ESIK: ezber={ezber:.2f}<=5 {kapilar['ezber_le5']} | "
          f"tutarsiz={tutarsiz:.2f}<28 {kapilar['tutarsizlik_lt28']} | "
          f"decomp={ozet['mean_rouge_l_decomp']:.4f}>=0.35 {kapilar['decomp_ge0_35']} | "
          f"yuklemsiz={yuklemsiz:.2f}<28 {kapilar['yuklemsiz_pir_lt28']} | "
          f"erken-kes={erken_kes:.2f}<17 {kapilar['erken_kes_pir_lt17']} | "
          f"basari-beyani<=22 {basari_beyani['tutarsizlik_le22_t0192']}", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())