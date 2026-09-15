#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: FAZ B2 KARŞILAŞTIRMALI BAZ HATLAR (BASELINES)
=======================================================================
Held-out 100 test sorusu (`data/rag_pilot/test_100.jsonl`) üzerinde:
1. Zero-Shot Baseline (kristal_b1_5_best.pt)
2. Belge-İlk-Cümle Baseline (Kural tabanlı)
3. Rastgele Cümle Baseline
Her baz hat için Stratified Bootstrap %95 GA ile ROUGE-L, Sadakat, Kısalık ve İlgili ölçer.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Any

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")
TEST_PATH = os.path.join(PILOT_DIR, "test_100.jsonl")


def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> float:
    """Computes ROUGE-L F1 score based on token LCS."""
    m, n = len(candidate_tokens), len(reference_tokens)
    if m == 0 or n == 0:
        return 0.0
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    lcs_len = lcs[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def stratified_bootstrap_ci(scores_wood: List[float], scores_hist: List[float], n_boot=1000, alpha=0.05):
    """Computes stratified bootstrap confidence interval across wood and history."""
    combined_means = []
    n_w = len(scores_wood)
    n_h = len(scores_hist)
    rng = np.random.RandomState(42)

    for _ in range(n_boot):
        sample_w = rng.choice(scores_wood, size=n_w, replace=True)
        sample_h = rng.choice(scores_hist, size=n_h, replace=True)
        combined_mean = (np.mean(sample_w) + np.mean(sample_h)) / 2.0
        combined_means.append(combined_mean)

    lower = float(np.percentile(combined_means, 100 * (alpha / 2)))
    upper = float(np.percentile(combined_means, 100 * (1 - alpha / 2)))
    mean_val = float(np.mean(combined_means))
    return mean_val, [lower, upper]


def extract_content_roots(tokens: List[str], vocab: Vocabulary) -> set:
    """Extracts non-grammatical, content-bearing morpheme tokens."""
    roots = set()
    stop_tags = {"<BOS>", "<EOS>", "<PAD>", "<UNK>", "<PROPER_NOUN>", "<INSTRUCTION>", "</INSTRUCTION>",
                 "<INPUT>", "</INPUT>", "<BELGE>", "</BELGE>", "<OUTPUT>", "</OUTPUT>"}
    for t in tokens:
        if t in stop_tags:
            continue
        if any(t.startswith(p) for p in ["CASE_", "POSS_", "TENSE_", "PERSON_", "COPULA_", "PART_", "DERIV_", "GERUND_", "INF_"]):
            continue
        if t in {".", ",", ":", ";", "-", "!", "?"}:
            continue
        roots.add(t.lower())
    return roots


def generate_greedy(model, prompt_ids: List[int], max_new_tokens: int = 50, eos_id: int = 2) -> List[int]:
    """Generates autoregressively with greedy argmax."""
    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            x_cond = x if x.size(1) <= 256 else x[:, -256:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :]
            next_token = int(torch.argmax(next_token_logits).item())

            if next_token == eos_id:
                break
            generated.append(next_token)
            x = torch.cat((x, torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)), dim=1)

    return generated


def main():
    print("=" * 75)
    print(" FAZ B2: HELD-OUT 100 TESTİ KARŞILAŞTIRMALI BAZ HATLAR (BASELINES)")
    print("=" * 75)

    # 1. Sözlük ve Model Yükleme
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)

    model = KristalLM(
        vocab_size=len(vocab.stoi),
        n_embd=768,
        vocab=vocab,
        block_size=4096,
        n_layer=6,
        n_head=6
    )
    ckpt_path = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"Model Yüklendi: {ckpt_path} (Cihaz: {DEVICE})")

    # 2. Test Kümelerini Yükle
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]
    print(f"Held-Out Test Kayıtları: {len(test_records)} örnek")

    eos_id = vocab.stoi.get("<EOS>", 2)

    # Değerlendirme Veri Yapıları
    baselines = {
        "zero_shot": {"wood_rouge": [], "hist_rouge": [], "fidelity": [], "length_ratio": [], "query_roots": []},
        "first_sentence": {"wood_rouge": [], "hist_rouge": [], "fidelity": [], "length_ratio": [], "query_roots": []},
        "random_sentence": {"wood_rouge": [], "hist_rouge": [], "fidelity": [], "length_ratio": [], "query_roots": []}
    }

    # Rastgele cümleler havuzu (külliyattan bağımsız)
    random_pool = [
        "Bu konuda arşivlerde yer alan belgeler dönemin mali yapısını açıklayan genel çerçeveyi çizer.",
        "Ahşap yüzeyin düzgün şekilde zımparalanması mobilyanın uzun ömürlü kullanımını temin eder.",
        "Kervan yolları üzerinde kurulan hanlar tüccarların emniyetini sağlamak üzere planlanmıştı.",
        "Masif tablaların kenarlarında açılan pahlar liflerin kopmasını engelleyen bir korumadır."
    ]

    print("\n--- TEST KÜMESİ ÜZERİNDE BAZ HATLAR DEĞERLENDİRİLİYOR ---")

    for idx, rec in enumerate(test_records):
        inp = rec["input"]
        doc = rec["belge"]
        ref = rec["output"]
        dom = rec["domain"]

        ref_tokens = tokenizer.encode(ref)
        doc_tokens = tokenizer.encode(doc)
        inp_tokens = tokenizer.encode(inp)

        ref_roots = extract_content_roots([vocab.decode(t) for t in ref_tokens], vocab)
        doc_roots = extract_content_roots([vocab.decode(t) for t in doc_tokens], vocab)
        inp_roots = extract_content_roots([vocab.decode(t) for t in inp_tokens], vocab)

        # -------------------------------------------------------------
        # BAZ HAT 1: ZERO-SHOT (Eğitilmemiş kristal_b1_5_best.pt)
        # -------------------------------------------------------------
        prompt_str = f"<INSTRUCTION> Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla. </INSTRUCTION> <INPUT> {inp} </INPUT> <BELGE> {doc} </BELGE> <OUTPUT>"
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]

        zs_gen_ids = generate_greedy(model, prompt_ids, max_new_tokens=40, eos_id=eos_id)
        zs_gen_tokens = [vocab.decode(t) for t in zs_gen_ids]
        zs_roots = extract_content_roots(zs_gen_tokens, vocab)

        zs_rouge = compute_rouge_l(zs_gen_tokens, [vocab.decode(t) for t in ref_tokens])
        zs_fidelity = len(zs_roots & doc_roots) / max(1, len(zs_roots))
        zs_len_ratio = len(zs_gen_tokens) / max(1, len(doc_tokens))
        zs_q_overlap = len(zs_roots & inp_roots)

        if dom == "carpenter":
            baselines["zero_shot"]["wood_rouge"].append(zs_rouge)
        else:
            baselines["zero_shot"]["hist_rouge"].append(zs_rouge)
        baselines["zero_shot"]["fidelity"].append(zs_fidelity)
        baselines["zero_shot"]["length_ratio"].append(zs_len_ratio)
        baselines["zero_shot"]["query_roots"].append(zs_q_overlap)

        # -------------------------------------------------------------
        # BAZ HAT 2: BELGE-İLK-CÜMLE KOPYALAMA
        # -------------------------------------------------------------
        first_sentence = doc.split(".")[0] + "."
        fs_tokens = [vocab.decode(t) for t in tokenizer.encode(first_sentence)]
        fs_roots = extract_content_roots(fs_tokens, vocab)

        fs_rouge = compute_rouge_l(fs_tokens, [vocab.decode(t) for t in ref_tokens])
        fs_fidelity = 1.0  # Doğrudan belgeden kopyalandığı için %100
        fs_len_ratio = len(fs_tokens) / max(1, len(doc_tokens))
        fs_q_overlap = len(fs_roots & inp_roots)

        if dom == "carpenter":
            baselines["first_sentence"]["wood_rouge"].append(fs_rouge)
        else:
            baselines["first_sentence"]["hist_rouge"].append(fs_rouge)
        baselines["first_sentence"]["fidelity"].append(fs_fidelity)
        baselines["first_sentence"]["length_ratio"].append(fs_len_ratio)
        baselines["first_sentence"]["query_roots"].append(fs_q_overlap)

        # -------------------------------------------------------------
        # BAZ HAT 3: RASTGELE CÜMLE
        # -------------------------------------------------------------
        rand_sentence = random_pool[idx % len(random_pool)]
        rnd_tokens = [vocab.decode(t) for t in tokenizer.encode(rand_sentence)]
        rnd_roots = extract_content_roots(rnd_tokens, vocab)

        rnd_rouge = compute_rouge_l(rnd_tokens, [vocab.decode(t) for t in ref_tokens])
        rnd_fidelity = len(rnd_roots & doc_roots) / max(1, len(rnd_roots))
        rnd_len_ratio = len(rnd_tokens) / max(1, len(doc_tokens))
        rnd_q_overlap = len(rnd_roots & inp_roots)

        if dom == "carpenter":
            baselines["random_sentence"]["wood_rouge"].append(rnd_rouge)
        else:
            baselines["random_sentence"]["hist_rouge"].append(rnd_rouge)
        baselines["random_sentence"]["fidelity"].append(rnd_fidelity)
        baselines["random_sentence"]["length_ratio"].append(rnd_len_ratio)
        baselines["random_sentence"]["query_roots"].append(rnd_q_overlap)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_records):
            print(f"  İlerleme: {idx + 1:3d}/100 test kaydı işlendi...", flush=True)

    # 3. İstatistiksel Raporlama (Stratified Bootstrap %95 GA)
    results = {}
    print("\n" + "=" * 75)
    print(f"{'BAZ HAT (BASELINE)':<25} | {'ROUGE-L (Bootstrap %95 GA)':<26} | {'SADAKAT':<8} | {'UZUNLUK':<8} | {'SORU KÖK':<8}")
    print("-" * 75)

    for name, data in baselines.items():
        mean_rouge, ci_rouge = stratified_bootstrap_ci(data["wood_rouge"], data["hist_rouge"])
        mean_fid = float(np.mean(data["fidelity"])) * 100
        mean_len = float(np.mean(data["length_ratio"]))
        mean_q = float(np.mean(data["query_roots"]))

        ci_str = f"{mean_rouge:.4f} [{ci_rouge[0]:.4f}, {ci_rouge[1]:.4f}]"
        print(f"{name:<25} | {ci_str:<26} | %{mean_fid:<7.1f} | {mean_len:<8.2f} | {mean_q:<8.2f}")

        results[name] = {
            "mean_rouge_l": mean_rouge,
            "bootstrap_95_ci": ci_rouge,
            "fidelity_rate": mean_fid,
            "length_ratio": mean_len,
            "query_root_overlap": mean_q
        }

    print("=" * 75)

    # 4. Sonuçları Kaydet
    out_path = os.path.join(PILOT_DIR, "baseline_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Baz hat sonuçları kaydedildi: {out_path}\n")


if __name__ == "__main__":
    main()
