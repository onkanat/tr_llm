#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0194 — Giyim+system_message şartlanma testi (3-kol × 2-yüzey, B1.5).

Kol-A birebir T-0192 (pozitif-kontrol: sayım T-0192 hüküm-değerlerine
eşit); Kol-B1 kod-canonical ROL_ZARFI salt-eklenti (src/llm/
prompt_contract.py — yalnız-OKU); Kol-B2 alan-canonical sabit
system-message (BETİKTEN ölçülen mod-dominan komut birebir-string).
Yüzeyler: RAW (elle-parça jeton-dizim) + DECOMP ( MorphemeDecompiler
insan-yüzeyi). src/compiler/** ve prompt_contract yalnız-okuma."""
import json
import os
import random
import re
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
from src.llm.prompt_contract import ROL_ZARFI
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict
from scripts.evaluate_b1_5_rigorous import rouge_l_score
from scripts.evaluate_anka_b1_5 import CIPE, DATA_DIR, SPLITS_DIR, DEVICE, compute_sha256

BEKLENEN_A = {  # T-0192 hüküm-çıpası — K1 pozitif-kontrol
    "n": 100, "memorization_rate": 0.0, "incoherence_rate": 22.0,
    "conditioning_rate": 6.0, "mean_rouge_l": 0.105808408705547,
}
SYS_MESAJ = "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla."  # BETİKTEN mod-dominan (train 4.849/13.003, test 856/1.625)
GIYIM_ESIK = 0.35
PREDICATE_TAGS = ("TENSE_", "COPULA_")


def kelime(x: str) -> list:
    return re.findall(r"[\w']+", x.lower())


def lcs_s(a: str, b: str) -> int:
    dp_prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = dp_prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp_prev[j], cur[j - 1])
        dp_prev = cur
    return dp_prev[-1]


def kes_f1(uret_w: list, girdi_w: list) -> float:
    if not uret_w or not girdi_w:
        return 0.0
    l = lcs_s(uret_w, girdi_w)
    p = l / len(uret_w)
    r = l / len(girdi_w)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def calis_arm(arm_anahtari, eval_records, tokenizer, vocab, model,
              train_4grams, decompiler, eos_id, out_end_id, decomp_istisna):
    kayit = []
    mem = inc = cond = r_toplam = 0.0
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for idx, item in enumerate(eval_records):
            inst, inp, ref_out = item.get("instruction", ""), item["input"], item["output"]
            if arm_anahtari == "A":
                slot = inst
            elif arm_anahtari == "B1":
                slot = f"{ROL_ZARFI}\n\n{inst}" if inst else ROL_ZARFI
            else:
                slot = f"{SYS_MESAJ}\n\n{inst}" if inst else SYS_MESAJ
            parts = []
            if slot:
                parts.extend(["<INSTRUCTION>", slot, "</INSTRUCTION>"])
            parts.extend(["<INPUT>", inp, "</INPUT>", "<OUTPUT>"])
            prompt_ids = tokenizer.encode(" ".join(parts))
            if prompt_ids and prompt_ids[-1] == eos_id:
                prompt_ids = prompt_ids[:-1]
            curr_x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
            gen_ids = []
            for _ in range(128):
                sign_mask = model.embedding.compute_sign_mask(curr_x.cpu()).to(DEVICE)
                logits, _ = model(curr_x, sign_mask=sign_mask)
                nt = int(torch.argmax(logits[0, -1, :]).item())
                if nt in (eos_id, out_end_id):
                    break
                gen_ids.append(nt)
                curr_x = torch.cat([curr_x, torch.tensor([[nt]], dtype=torch.long, device=DEVICE)], dim=1)
                if curr_x.size(1) >= 256:
                    break
            gen_tokens = [vocab.decode(t) for t in gen_ids]
            gen_text = " ".join(gen_tokens)
            try:
                decomp_text = decompiler.decompile_sentence(gen_text, capitalize=True)
            except Exception:
                decomp_text = ""
                decomp_istisna[0] += 1

            gw = kelime(gen_text)
            cand = [tuple(gw[i:i + 4]) for i in range(len(gw) - 3)]
            mem_i = bool(cand) and sum(c in train_4grams for c in cand) / len(cand) >= 0.90
            mem += int(mem_i)
            son = gen_tokens[-5:] if len(gen_tokens) >= 5 else gen_tokens
            yuklem = any(t.startswith(PREDICATE_TAGS) for t in son)
            dongu = any(gen_tokens[i:i + 2] == gen_tokens[i + 2:i + 4] == gen_tokens[i + 4:i + 6]
                        for i in range(len(gen_tokens) - 5)) if len(gen_tokens) >= 6 else False
            inc_i = (not yuklem) or dongu
            inc += int(inc_i)
            if len(set(kelime(inp)) & set(gw)) >= 2:
                cond += 1
            rl = rouge_l_score(gw, kelime(ref_out))
            r_toplam += rl
            dw = kelime(decomp_text)
            rl_d = rouge_l_score(dw, kelime(ref_out))
            kf = kes_f1(gw, kelime(inp))
            kayit.append({
                "kol": arm_anahtari, "idx": idx, "rouge_l_raw": rl, "rouge_l_decomp": rl_d,
                "kesisim_f1": kf, "memorized": bool(mem_i), "incoherent": bool(not yuklem or dongu),
                "yuklem_yok": not yuklem, "dongu": bool(dongu), "soru_unk": decomp_text.count("[?]"),
                "n_jeton": len(gen_tokens), "decomp_text": decomp_text, "gen_text": gen_text,
            })
    n = len(kayit)
    return kayit, {"n": n, "memorization_rate": mem / n * 100.0, "incoherence_rate": inc / n * 100.0,
                   "mean_rouge_l": r_toplam / n, "conditioning_rate": cond / n * 100.0,
                   "mean_rouge_l_decomp": sum(k["rouge_l_decomp"] for k in kayit) / n,
                   "mean_kesisim_f1": sum(k["kesisim_f1"] for k in kayit) / n,
                   "bos_decompile": sum(1 for k in kayit if not k["decomp_text"].strip()),
                   "sure_sn": round(time.time() - t0, 1)}


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    for yol, bek in CIPE.items():
        if compute_sha256(os.path.join(KÖK, yol)) != bek:
            print("T0194_GIYIM_DUR: cipa uymaz", yol, flush=True)
            return 2
    print("[CIPI] 6/6 çıpa BETİKTEN teyit — PASS", flush=True)

    with open(os.path.join(SPLITS_DIR, "train.jsonl"), encoding="utf-8") as f:
        train_rec = [json.loads(l) for l in f]
    sayım = sum(1 for r in train_rec if r.get("instruction", "") == SYS_MESAJ)
    if sayım < 1:
        print("T0194_GIYIM_DUR: SYS_MESAJ train'de verbatim yok", flush=True)
        return 2
    print(f"[SYS_MESAJ_CANONICAL] train verbatim sayım: {sayım}", flush=True)

    train_4grams = set()
    for d in train_rec:
        w = kelime(d.get("output", ""))
        for i in range(len(w) - 3):
            train_4grams.add(tuple(w[i:i + 4]))
    with open(os.path.join(SPLITS_DIR, "test.jsonl"), encoding="utf-8") as f:
        test_records = [json.loads(l) for l in f]
    eval_records = random.Random(42).sample(test_records, min(100, len(test_records)))

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "rebuild", "vocab_anka_r1_33114.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    decompiler = MorphemeDecompiler(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(os.path.join(KÖK, "data/anka_b1_5_best.pt"), map_location="cpu")
    for k in [k for k in list(sd.keys()) if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(DEVICE)
    print("[MODELI] anka_b1_5_best yüklendi", flush=True)

    kayit = []
    eos_id = vocab.stoi.get("<EOS>", 3)
    out_end_id = vocab.stoi.get("</OUTPUT>", 9)
    özet = {}
    for anahtar in ("A", "B1", "B2"):
        decomp_istisna = [0]
        kol_kayit, öz = calis_arm(anahtar, eval_records, tokenizer, vocab, model,
                                  train_4grams, decompiler, eos_id, out_end_id, decomp_istisna)
        öz["decomp_istisna"] = decomp_istisna[0]
        kayit.extend(kol_kayit)
        özet[anahtar] = öz
        print(f"ARM_{anahtar}:", json.dumps(öz, ensure_ascii=False), flush=True)

    # K1 pozitif-kontrol (Kol-A raw)
    for k, bek in BEKLENEN_A.items():
        fark = abs(özet["A"][k] - bek)
        if fark > (1e-6 if k == "mean_rouge_l" else 1e-9):
            print(f"T0194_GIYIM_DUR K1: {k} fark {fark}", flush=True)
            return 2
    print("[K1] Kol-A birebir T-0192 — PASS", flush=True)

    b1_d, b2_d, a_d = özet["B1"]["mean_rouge_l_decomp"], özet["B2"]["mean_rouge_l_decomp"], özet["A"]["mean_rouge_l_decomp"]
    delta_b1, delta_b2 = b1_d - a_d, b2_d - a_d
    yineleme_b2 = sum(1 for r in eval_records if r.get("instruction", "") == SYS_MESAJ)
    gecti = max(b1_d, b2_d) >= GIYIM_ESIK
    hukum = "T0194_GIYIM_GECTİ" if gecti else "T0194_GIYIM_DUR"
    donem = {
        "task": "T-0194", "hukum": hukum, "rc": 0 if gecti else 2,
        "esik": {"decomp_mean_min": GIYIM_ESIK, "k1": "Kol-A birebir"},
        "matris": özet,
        "sartlanma_delta": {"B1_decomp_delta": delta_b1, "B2_decomp_delta": delta_b2},
        "rol_zarf_metni": ROL_ZARFI, "sys_mesaj": SYS_MESAJ,
        "b2_birebir_yineleme_sayım": yineleme_b2,
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ilan": "data/eval/t0194_giyim_ilan_2026-09-29.md",
    }
    with open(os.path.join(KÖK, "data/eval/t0194_giyim_hukum_2026-09-29.json"), "w", encoding="utf-8") as f:
        json.dump(donem, f, indent=2, ensure_ascii=False)
    with open(os.path.join(KÖK, "data/eval/t0194_giyim_kayitlar.json"), "w", encoding="utf-8") as f:
        json.dump({"matris": özet, "kayitlar": kayit}, f, indent=2, ensure_ascii=False)
    print(f"DECOMP_MEANS: A={a_d:.4f} B1={b1_d:.4f} B2={b2_d:.4f}", flush=True)
    print(f"DELTA: B1={delta_b1:+.4f} B2={delta_b2:+.4f} | B2-yineleme={yineleme_b2}/100", flush=True)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())