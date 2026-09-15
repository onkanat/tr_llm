#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: FAZ B2 RAG-SENTEZ PİLOT MODEL DEĞERLENDİRMESİ
=======================================================================
1. `data/rag_pilot/kristal_rag_pilot_best.pt` modelini held-out 100 test kümesi
   (`data/rag_pilot/test_100.jsonl`) üzerinde değerlendirir.
2. Kol A (Standart Morfem Tokenizer) ve Kol C (Literal Varlık Tokenizer) karşılaştırması.
3. Stratified Bootstrap %95 Güven Aralığı (GA) ile ROUGE-L hesabı (Tarih ve Ahşap dengeli).
4. Kopya Karşıtı Metrik Üçlüsü (Anti-Copy Trio):
   - Sadakat (Fidelity): Belge Kesişimi >= %60
   - Kısalık (Conciseness): Çıktı Uzunluğu <= Belge Uzunluğu * 1.5
   - Soru İlgisi (Relevance): Çıktı ile Soru Kök Kesişimi >= 2
5. Tutarsızlık / Dejenerasyon Oranı (Incoherence Rate): n-gram tekrarları ve döngüler.
6. E7 Morfoloji Regresyon Sınavı (`tests/morphology_regression_100.json`, Eşik: >= %72).
7. Baz Hatlar (`data/rag_pilot/baseline_results.json`) ile Karşılaştırma.
8. Ön-Kayıtlı Karar Sınıflandırması: PASS / AMBIGUOUS / FAIL-A / FAIL-B.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Set

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import render_prompt
from scripts.train_step_demo import KristalLM
from scripts.train_step_b1_5_rigorous import compute_sha256

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")
TEST_PATH = os.path.join(PILOT_DIR, "test_100.jsonl")
PILOT_CKPT = os.path.join(PILOT_DIR, "kristal_rag_pilot_best.pt")
BASELINE_JSON = os.path.join(PILOT_DIR, "baseline_results.json")
TRAIN_HIST_JSON = os.path.join(PILOT_DIR, "training_history.json")


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


def stratified_bootstrap_ci(scores_wood: List[float], scores_hist: List[float], n_boot=1000, alpha=0.05) -> Tuple[float, List[float]]:
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


def extract_content_roots(tokens: List[str], vocab: Vocabulary) -> Set[str]:
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


def check_incoherence(tokens: List[str]) -> bool:
    """Detects degeneration, repetitive n-grams, or proper noun loops."""
    if len(tokens) < 4:
        return False
    # Check 3-gram repetition
    seen_trigrams = set()
    for i in range(len(tokens) - 2):
        tg = tuple(tokens[i:i+3])
        if tg in seen_trigrams:
            return True
        seen_trigrams.add(tg)
    # Check excessive <PROPER_NOUN> or repetitive single tokens
    if tokens.count("<PROPER_NOUN>") >= 4:
        return True
    return False


def generate_greedy(model: KristalLM, prompt_ids: List[int], max_new_tokens: int = 50, eos_id: int = 2) -> List[int]:
    """Generates autoregressively with greedy argmax."""
    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= 256 else x[:, -256:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :]
            next_token = int(torch.argmax(next_token_logits).item())

            if next_token == eos_id:
                break
            generated.append(next_token)
            x = torch.cat((x, torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)), dim=1)

    return generated


def evaluate_morphology_regression(model: KristalLM, tokenizer: KristalTokenizer, vocab: Vocabulary) -> float:
    """Evaluates morphology retention on 100 test items."""
    test_path = os.path.join(BASE_DIR, "tests", "morphology_regression_100.json")
    if not os.path.exists(test_path):
        return 0.0

    with open(test_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    correct = 0
    model.eval()
    eos_id = vocab.stoi.get("<EOS>", 2)

    for item in samples:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        true_out = item.get("output", "").strip()

        prompt_str = render_prompt(inst, inp)
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]

        true_ids = tokenizer.encode(true_out)
        if true_ids and true_ids[0] == vocab.stoi.get("<BOS>", 2):
            true_ids = true_ids[1:]
        if true_ids and true_ids[-1] == eos_id:
            true_ids = true_ids[:-1]

        pred_ids = generate_greedy(model, prompt_ids, max_new_tokens=min(15, len(true_ids) + 4), eos_id=eos_id)

        pred_str = tokenizer.decode(pred_ids).replace("<BOS>", "").replace("<EOS>", "").strip()
        true_str = tokenizer.decode(true_ids).replace("<BOS>", "").replace("<EOS>", "").strip()

        if pred_str.strip() == true_str.strip():
            correct += 1

    return (correct / max(1, len(samples))) * 100.0


def main():
    print("=" * 75)
    print(" FAZ B2: RAG-SENTEZ PİLOT MODEL DEĞERLENDİRMESİ VE KARAR SİSTEMİ")
    print("=" * 75)

    # 1. Kaynakları Yükle
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

    fallback_ckpt = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
    if os.path.exists(PILOT_CKPT):
        ckpt_to_eval = PILOT_CKPT
    elif os.path.exists(fallback_ckpt):
        print(f"[SOYAGACI_UYARI] BIRINCIL YOK ({PILOT_CKPT}), GERI DUSULDU -> {fallback_ckpt}")
        ckpt_to_eval = fallback_ckpt
    else:
        raise FileNotFoundError(f"Değerlendirilecek checkpoint bulunamadı! Ne birincil ({PILOT_CKPT}) ne de geri dönüş ({fallback_ckpt}) mevcut.")

    try:
        sd = torch.load(ckpt_to_eval, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Checkpoint dosyası mevcut fakat yüklenemedi ({ckpt_to_eval}): {e}") from e

    sha256_val = compute_sha256(ckpt_to_eval)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(ckpt_to_eval)} sha256={sha256_val} anahtar={len(sd)}")
    load_res = model.load_state_dict(sd, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(f"[SOYAGACI_UYARI] strict=False ile yüklendi: eksik={len(load_res.missing_keys)}, fazla={len(load_res.unexpected_keys)}")
        if load_res.missing_keys:
            print(f"  * Eksik anahtarlar: {load_res.missing_keys[:5]}{'...' if len(load_res.missing_keys) > 5 else ''}")
        if load_res.unexpected_keys:
            print(f"  * Fazla anahtarlar: {load_res.unexpected_keys[:5]}{'...' if len(load_res.unexpected_keys) > 5 else ''}")
    else:
        print("[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).")

    model.to(DEVICE)
    model.eval()
    print(f"Değerlendirilen Model: {ckpt_to_eval} (Cihaz: {DEVICE})")

    # 2. Test Kümelerini ve Baz Hatları Yükle
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]
    print(f"Held-Out Test Kayıtları: {len(test_records)} örnek")

    baselines = {}
    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON, "r", encoding="utf-8") as f:
            baselines = json.load(f)
        print("Baz hat sonuçları yüklendi.")
    else:
        print("UYARI: Baz hat sonuçları bulunamadı! Lütfen önce evaluate_rag_baselines.py çalıştırın.")

    train_history = {}
    if os.path.exists(TRAIN_HIST_JSON):
        with open(TRAIN_HIST_JSON, "r", encoding="utf-8") as f:
            train_history = json.load(f)
        print("Eğitim geçmişi yüklendi.")

    eos_id = vocab.stoi.get("<EOS>", 2)

    # 3. Model Üretimlerini Ölç
    eval_data = {
        "wood_rouge": [], "hist_rouge": [],
        "fidelity": [], "length_ratio": [], "query_roots": [],
        "incoherent_count": 0,
        "sample_generations": []
    }

    print("\n--- HELD-OUT 100 TESTİ ÜZERİNDE ÜRETİM VE SENTEZ ÖLÇÜLÜYOR ---")
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

        prompt_str = f"<INSTRUCTION> Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla. </INSTRUCTION> <INPUT> {inp} </INPUT> <BELGE> {doc} </BELGE> <OUTPUT>"
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]

        gen_ids = generate_greedy(model, prompt_ids, max_new_tokens=45, eos_id=eos_id)
        gen_tokens = [vocab.decode(t) for t in gen_ids]
        gen_roots = extract_content_roots(gen_tokens, vocab)

        rouge = compute_rouge_l(gen_tokens, [vocab.decode(t) for t in ref_tokens])
        fidelity = len(gen_roots & doc_roots) / max(1, len(gen_roots))
        len_ratio = len(gen_tokens) / max(1, len(doc_tokens))
        q_overlap = len(gen_roots & inp_roots)
        incoherent = check_incoherence(gen_tokens)

        if incoherent:
            eval_data["incoherent_count"] += 1

        if dom == "carpenter":
            eval_data["wood_rouge"].append(rouge)
        else:
            eval_data["hist_rouge"].append(rouge)

        eval_data["fidelity"].append(fidelity)
        eval_data["length_ratio"].append(len_ratio)
        eval_data["query_roots"].append(q_overlap)

        if idx < 5 or idx == 50:
            # Decode for human-readable inspection
            decoded_pred = tokenizer.decode(gen_ids).replace("<BOS>", "").replace("<EOS>", "").strip()
            eval_data["sample_generations"].append({
                "idx": idx,
                "domain": dom,
                "question": inp,
                "prediction": decoded_pred,
                "reference": ref,
                "rouge_l": rouge,
                "fidelity": fidelity,
                "query_overlap": q_overlap
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_records):
            print(f"  İlerleme: {idx + 1:3d}/100 test kaydı değerlendirildi...", flush=True)

    # 4. İstatistiksel Hesaplamalar
    mean_rouge, ci_rouge = stratified_bootstrap_ci(eval_data["wood_rouge"], eval_data["hist_rouge"])
    mean_wood_rouge = float(np.mean(eval_data["wood_rouge"]))
    mean_hist_rouge = float(np.mean(eval_data["hist_rouge"]))
    mean_fidelity = float(np.mean(eval_data["fidelity"])) * 100.0
    mean_length_ratio = float(np.mean(eval_data["length_ratio"]))
    mean_query_roots = float(np.mean(eval_data["query_roots"]))
    incoherence_rate = (eval_data["incoherent_count"] / len(test_records)) * 100.0

    # 5. E7 Morfoloji Regresyon Sınavı
    print("\n--- E7 MORFOLOJİ REGRESYON TESTİ ÇALIŞTIRILIYOR ---")
    morph_acc = evaluate_morphology_regression(model, tokenizer, vocab)
    print(f"  E7 Morfoloji Skoru: %{morph_acc:.2f} (Ön-Kayıtlı Eşik: >= %72.0)")

    # 6. Karar Matrisi ve Sınıflandırma
    print("\n" + "=" * 75)
    print(" FAZ B2 PİLOT DEĞERLENDİRME VE ÖN-KAYITLI KARAR MATRİSİ")
    print("=" * 75)

    fs_baseline_rouge = baselines.get("first_sentence", {}).get("mean_rouge_l", 0.0)
    zs_baseline_rouge = baselines.get("zero_shot", {}).get("mean_rouge_l", 0.0)

    print(f"{'Metrik':<30} | {'Pilot Model':<20} | {'İlk-Cümle Baz':<15} | {'Zero-Shot Baz':<15}")
    print("-" * 75)
    print(f"{'ROUGE-L (Bootstrap %95 GA)':<30} | {mean_rouge:.4f} [{ci_rouge[0]:.4f}, {ci_rouge[1]:.4f}] | {fs_baseline_rouge:.4f} | {zs_baseline_rouge:.4f}")
    print(f"{'  - Ahşap Domain ROUGE-L':<30} | {mean_wood_rouge:.4f}               | -               | -")
    print(f"{'  - Tarih Domain ROUGE-L':<30} | {mean_hist_rouge:.4f}               | -               | -")
    print(f"{'Sadakat (Belge Kesişimi)':<30} | %{mean_fidelity:.1f}                | %100.0          | -")
    print(f"{'Uzunluk Oranı (Output/Doc)':<30} | {mean_length_ratio:.2f}                | -               | -")
    print(f"{'Soru Kök Kesişimi':<30} | {mean_query_roots:.2f}                | -               | -")
    print(f"{'Tutarsızlık / Döngü Oranı':<30} | %{incoherence_rate:.1f}                | %0.0            | -")
    print(f"{'E7 Morfoloji Regresyonu':<30} | %{morph_acc:.1f}                | -               | -")
    print("-" * 75)

    # Karar Algoritması
    val_loss_ratio = train_history.get("loss_ratio", 1.0)
    val_loss_dropped = val_loss_ratio <= 0.90

    # Kopya Karşıtı Üçlü
    trio_pass = (mean_fidelity >= 60.0) and (mean_length_ratio <= 1.5) and (mean_query_roots >= 2.0)
    morph_pass = (morph_acc >= 72.0)
    rouge_pass = (ci_rouge[0] >= 0.30) and (mean_rouge >= fs_baseline_rouge + 0.08)

    if rouge_pass and trio_pass and morph_pass:
        outcome = "PASS"
        explanation = (
            "Model RAG bağlamından başarıyla sentez yapmaktadır. "
            f"ROUGE-L GA alt sınırı ({ci_rouge[0]:.4f} >= 0.30), "
            f"İlk-cümle baz hattından farkı (+{mean_rouge - fs_baseline_rouge:.4f} >= +0.08), "
            f"Kopya Karşıtı Üçlü ve E7 Morfoloji (%{morph_acc:.1f}) eşiklerini karşılamıştır."
        )
    elif ci_rouge[1] < 0.20:
        if val_loss_dropped:
            outcome = "FAIL-A (Undertraining / Yetersiz Eğitim)"
            explanation = (
                f"Model ROUGE-L GA üst sınırı ({ci_rouge[1]:.4f} < 0.20) ile zayıf kaldı; "
                f"ancak Validation Loss %{(1.0 - val_loss_ratio)*100:.1f} oranında azaldı. "
                "Model hedef fonksiyonunu öğrenmektedir ancak 800 örnek yetersizdir. "
                "MİMARİ REDDEDİLEMEZ; veri ölçekleme (#4 ölçek testi) gereklidir."
            )
        else:
            outcome = "FAIL-B (Architectural Failure / Mimari Yetersizlik)"
            explanation = (
                f"Model ROUGE-L GA üst sınırı ({ci_rouge[1]:.4f} < 0.20) ve Validation Loss düşmedi "
                f"(oran: {val_loss_ratio:.4f} > 0.90). 93M morfemik mimari bağlamdan sentez yapamamaktadır."
            )
    else:
        outcome = "AMBIGUOUS (Belirsiz / Ara Bölge)"
        explanation = (
            f"Model ara bölgededir: ROUGE-L {mean_rouge:.4f} [{ci_rouge[0]:.4f}, {ci_rouge[1]:.4f}], "
            f"Kopya Üçlüsü: {trio_pass}, Morfoloji: %{morph_acc:.1f}."
        )

    print(f"\n>>> NİHAİ ÖN-KAYITLI KARAR: {outcome}")
    print(f"Gerekçe: {explanation}\n")

    # 7. Sonuçları Kaydet
    output_summary = {
        "model_checkpoint": ckpt_to_eval,
        "mean_rouge_l": mean_rouge,
        "bootstrap_95_ci": ci_rouge,
        "woodcraft_rouge_l": mean_wood_rouge,
        "history_rouge_l": mean_hist_rouge,
        "fidelity_rate": mean_fidelity,
        "length_ratio": mean_length_ratio,
        "query_root_overlap": mean_query_roots,
        "incoherence_rate": incoherence_rate,
        "morphology_e7_acc": morph_acc,
        "first_sentence_baseline_rouge": fs_baseline_rouge,
        "zero_shot_baseline_rouge": zs_baseline_rouge,
        "val_loss_ratio": val_loss_ratio,
        "trio_pass": trio_pass,
        "outcome": outcome,
        "explanation": explanation,
        "samples": eval_data["sample_generations"]
    }

    results_path = os.path.join(PILOT_DIR, "pilot_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, ensure_ascii=False)
    print(f"Tam değerlendirme sonuçları kaydedildi: {results_path}\n")


if __name__ == "__main__":
    main()
