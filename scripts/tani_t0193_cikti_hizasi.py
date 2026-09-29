#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0193 — Çıktı-biçim hizası tanısı (ROUGE gösterim-payı ayrışımı).

T-0192 koşum-kalıbı BİREBİR yeniden-koşulur (pozitif-kontrol K1: sayım
T-0192 hüküm değerleridir); her üretim iki gövdeye ayrılır: RAW
(jeton-dizi T-0192 yüzeyi) ve DECOMP (kanonik decompiler insan-yüzeyi,
MorphemeDecompiler.decompile_sentence — gateway'in çağrı-yüzeyi).
ROUGE-L tarihsel eval betiğinin tek-metrik-tanımıyla import'la ölçülür.
src/compiler/** yalnız-okuma."""
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
from scripts.train_step_demo import KristalLM
from scripts.evaluate_mcq_conditioning import resize_state_dict
from scripts.evaluate_b1_5_rigorous import rouge_l_score  # tek-metrik-tanım
from scripts.evaluate_anka_b1_5 import CIPE, DATA_DIR, SPLITS_DIR, DEVICE, compute_sha256

BEKLENEN = {  # T-0192 hüküm-çıpası — K1 pozitif-kontrol hedefi
    "n": 100, "memorization_rate": 0.0, "incoherence_rate": 22.0,
    "conditioning_rate": 6.0, "mean_rouge_l": 0.105808408705547,
}
DECOMP_ROUGE_TAM, DELTA_TAM, DELTA_MIN, DELTA_YOK = 0.35, 0.10, 0.05, 0.05


def main() -> int:
    random.seed(42)
    torch.manual_seed(42)
    for yol, bek in CIPE.items():
        ölç = compute_sha256(os.path.join(KÖK, yol))
        if ölç != bek:
            print(f"T0193_TANI_DUR: çıpa-uyuşmazlığı {yol}: {ölç}", flush=True)
            return 2
    print("[CIPI] 6/6 çıpa BETİKTEN teyit — PASS", flush=True)

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

    # --- T-0192 koşum-kalıbı birebir (tarihsel eval satır-satır kopya) ---
    train_4grams = set()
    with open(os.path.join(SPLITS_DIR, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            words = re.findall(r"[\w']+", d.get("output", "").lower())
            for i in range(len(words) - 3):
                train_4grams.add(tuple(words[i:i + 4]))
    with open(os.path.join(SPLITS_DIR, "test.jsonl"), encoding="utf-8") as f:
        test_records = [json.loads(l) for l in f]
    rng = random.Random(42)
    eval_records = rng.sample(test_records, min(100, len(test_records)))

    eos_id = vocab.stoi.get("<EOS>", 3)
    out_end_id = vocab.stoi.get("</OUTPUT>", 9)
    PREDICATE_TAGS = ("TENSE_", "COPULA_")

    kayıt = []
    mem_count = incoh_count = cond_count = rouge_toplam = 0.0
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for idx, item in enumerate(eval_records):
            inst, inp, ref_out = item.get("instruction", ""), item["input"], item["output"]
            parts = []
            if inst:
                parts.extend(["<INSTRUCTION>", inst, "</INSTRUCTION>"])
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
            decomp_text = decompiler.decompile_sentence(gen_text, capitalize=True)

            gen_words = re.findall(r"[\w']+", gen_text.lower())
            cand = [tuple(gen_words[i:i + 4]) for i in range(len(gen_words) - 3)]
            mem = cand and sum(c in train_4grams for c in cand) / len(cand) >= 0.90
            mem_count += int(bool(mem))
            last = gen_tokens[-5:] if len(gen_tokens) >= 5 else gen_tokens
            has_pred = any(t.startswith(PREDICATE_TAGS) for t in last)
            loop = any(gen_tokens[i:i + 2] == gen_tokens[i + 2:i + 4] == gen_tokens[i + 4:i + 6]
                       for i in range(len(gen_tokens) - 5)) if len(gen_tokens) >= 6 else False
            incoh = (not has_pred) or loop
            incoh_count += int(incoh)
            ref_words = re.findall(r"[\w']+", ref_out.lower())
            r_l = rouge_l_score(gen_words, ref_words)
            rouge_toplam += r_l
            if len(set(re.findall(r"[\w']+", inp.lower())) & set(gen_words)) >= 2:
                cond_count += 1
            tag_pay = (sum(1 for t in gen_tokens if re.match(r"^[A-Z][A-Z_0-9]|^GERUND|^INF|^<ENT>$|^</ENT>$|^<CAP>$|^</CAP>$|^<ALL_CAPS>$", t))
                       / len(gen_tokens)) if gen_tokens else 0.0
            kayıt.append({
                "idx": idx, "gen_tokens": gen_tokens, "gen_text": gen_text,
                "decomp_text": decomp_text, "rouge_l_raw": r_l,
                "memorized": bool(mem), "incoherent": incoh, "predicate_yok": not has_pred,
                "tekrar_dongu": loop, "tag_orani": tag_pay,
                "ent_blok": gen_tokens.count("<ENT>"), "n_jeton": len(gen_tokens),
            })
            print(f"[URETIM {idx+1}/100] r_l={r_l:.4f} decomp_uzunluk={len(decomp_text)}", flush=True)
    n = len(kayıt)
    özet_raw = {"n": n, "memorization_rate": mem_count / n * 100.0,
                "incoherence_rate": incoh_count / n * 100.0,
                "mean_rouge_l": rouge_toplam / n, "conditioning_rate": cond_count / n * 100.0}

    # --- K1 pozitif-kontrol ---
    farklar = {}
    for k, bek in BEKLENEN.items():
        fark = abs(özet_raw[k] - bek)
        farklar[k] = {"ölçüm": özet_raw[k], "beklenen": bek, "fark": fark}
        if k == "mean_rouge_l":
            if fark > 1e-6:
                print(f"T0193_TANI_DUR K1: {k} fark {fark}", flush=True)
                return 2
        elif fark > 1e-9:
            print(f"T0193_TANI_DUR K1: {k} fark {fark}", flush=True)
            return 2
    print("[K1] pozitif-kontrol: T-0192 hüküm-sayım birebir — PASS", flush=True)

    # --- DECOMP yüzeyi ROUGE-L ---
    rouge_decomp = [rouge_l_score(re.findall(r"[\w']+", k["decomp_text"].lower()),
                                  re.findall(r"[\w']+", eval_records[k["idx"]]["output"].lower()))
                    for k in kayıt]
    decomp_mean = float(sum(rouge_decomp) / n)
    raw_mean = özet_raw["mean_rouge_l"]
    delta = decomp_mean - raw_mean
    bos_decomp = sum(1 for k in kayıt if not k["decomp_text"].strip())

    if decomp_mean >= DECOMP_ROUGE_TAM and delta > DELTA_TAM:
        sınıf = "HIZA_TAM"
    elif delta >= DELTA_MIN:
        sınıf = "HIZA_KISMİ"
    else:
        sınıf = "HIZA_YOK"

    tutarsizlar = [{"idx": k["idx"], "predicate_yok": k["predicate_yok"],
                    "tekrar_dongu": k["tekrar_dongu"]} for k in kayıt if k["incoherent"]]
    envanter = {"ort_tag_orani": sum(k["tag_orani"] for k in kayıt) / n,
                "ort_ent_blok": sum(k["ent_blok"] for k in kayıt) / n,
                "ort_jeton": sum(k["n_jeton"] for k in kayıt) / n,
                "bos_decompile": bos_decomp}
    hukum = {
        "task": "T-0193", "hukum": "T0193_TANI_GECTİ", "hiza_sinifi": sınıf,
        "raw_mean_rouge_l": raw_mean, "decomp_mean_rouge_l": decomp_mean,
        "delta_rouge": delta,
        "esikler": {"HIZA_TAM": "decomp>=0.35 ve delta>0.10", "HIZA_KISMİ": "0.05<=delta<=0.10", "HIZA_YOK": "delta<0.05"},
        "k1_pozitif_kontrol": {"sonuc": "PASS", "farklar": farklar},
        "k2_decompiler": {"bos_decompile": bos_decomp, "istisna": 0},
        "bileşen": {"tutarsiz_sayım": len(tutarsizlar),
                    "predicate_yok_sayım": sum(1 for t in tutarsizlar if t["predicate_yok"]),
                    "dongu_sayım": sum(1 for t in tutarsizlar if t["tekrar_dongu"]),
                    "idx_list": [t["idx"] for t in tutarsizlar]},
        "format_envanter": envanter,
        "cikti_rapor": "data/eval/t0193_cikti_hiza_ornekler.json",
        "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sure_sn": round(time.time() - t0, 1),
        "ilan": "data/eval/t0193_cikti_hiza_ilan_2026-09-29.md",
    }
    with open(os.path.join(KÖK, "data/eval/t0193_cikti_hiza_hukum_2026-09-29.json"), "w", encoding="utf-8") as f:
        json.dump(hukum, f, indent=2, ensure_ascii=False)
    with open(os.path.join(KÖK, "data/eval/t0193_cikti_hiza_ornekler.json"), "w", encoding="utf-8") as f:
        json.dump({"ozet_raw": özet_raw, "kayitlar": kayıt, "decomp_rouge_l": rouge_decomp},
                  f, indent=2, ensure_ascii=False)
    print("ÖZET_RAW:", json.dumps(özet_raw, ensure_ascii=False), flush=True)
    print(f"DECOMP_MEAN_ROUGE: {decomp_mean:.4f} | RAW: {raw_mean:.4f} | DELTA: {delta:.4f}", flush=True)
    print("BILESEN:", json.dumps(hukum["bileşen"], ensure_ascii=False), flush=True)
    print("ENVANTER:", json.dumps(envanter, ensure_ascii=False), flush=True)
    print(f"HUKUM: T0193_TANI_GECTİ | HIZA_SINIFI: {sınıf}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    print(f"RC={rc}", flush=True)
    raise SystemExit(rc)