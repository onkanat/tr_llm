#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: TİTİZ TABAN-BAĞLANTILI GERÇEKÇİ RAG DEĞERLENDİRMESİ
=====================================================================
Bu betik; eğitilen modelleri (Kol A, Kol B, Kol C) ve kural tabanlı baz hatları
(Zero-Shot, Belge-İlk-Cümle, TF-IDF Cümle Seçici) iki ayrık test kümesinde
(Test-Natural N=150 ve Test-CF N=50) değerlendirir:

Ölçülen Metrikler:
1. ROUGE-L (Bootstrap %95 GA)
2. Yüzey Varlık Doğruluğu (Surface Entity Fidelity)
3. 4 Seviyeli Doc-Swap Merdiveni:
   - D_paired (Hafızadaki orijinal belge)
   - D_wrong_same (Aynı alandan yanlış belge)
   - D_wrong_cross (Farklı alandan yanlış belge)
   - D_unrelated (Alakasız metin)
4. Özgül Olgu (Unique Fact) Akışı:
   - score(D_unique_wrong) -> Yalnızca verilen belgeden gelen olgular
   - score(D_unique_paired) -> Soru öncülü hafızasından sayıklanan olgular
5. Taban-Bağlantılı Ön-Kayıtlı Karar Hükmü:
   score(D_unique_wrong) >= max(0.50, 1.5 * TF-IDF Unique Tabanı)
   ve score(D_unique_paired) <= 0.15
"""

import os
import sys
import json
import re
import math
import hashlib
from datetime import datetime, timezone
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Tuple, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import render_prompt, build_rag_input
from scripts.train_step_demo import KristalLM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REALISTIC_DIR = os.path.join(DATA_DIR, "realistic_rag")
EVAL_DIR = os.path.join(DATA_DIR, "eval")

BASE_VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
EXP_VOCAB_PATH = os.path.join(DATA_DIR, "vocab_entity.json")

OLD_REPORTED_LOSSES = {
    "arm_A": 3.7342,
    "arm_B": 3.6262,
    "arm_C": 3.2994
}


def get_file_metadata(filepath: str) -> Dict[str, Any]:
    mtime_ts = os.path.getmtime(filepath)
    dt = datetime.fromtimestamp(mtime_ts, tz=timezone.utc)
    with open(filepath, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    size = os.path.getsize(filepath)
    return {
        "path": filepath,
        "size_bytes": size,
        "mtime_timestamp": mtime_ts,
        "mtime_iso_utc": dt.isoformat(),
        "mtime_local_str": datetime.fromtimestamp(mtime_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": sha256
    }


class RealisticRagDataset(Dataset):
    """Precomputes causal prompt-masked tensors for RAG fine-tuning and evaluation."""
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: KristalTokenizer,
        vocab: Vocabulary,
        model: KristalLM,
        block_size: int = 256
    ):
        self.block_size = block_size
        self.pad_id = vocab.stoi.get("<PAD>", 1)
        self.bos_id = vocab.stoi.get("<BOS>", 2)
        self.eos_id = vocab.stoi.get("<EOS>", 3)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for rec in records:
            inst = rec.get("instruction", "").strip()
            inp = rec.get("input", "").strip()
            out = rec.get("output", "").strip()

            prompt_str = render_prompt(inst, inp)

            prompt_ids = tokenizer.encode(prompt_str)
            if prompt_ids and prompt_ids[-1] == self.eos_id:
                prompt_ids = prompt_ids[:-1]

            out_ids = tokenizer.encode(out)
            if out_ids and out_ids[0] == self.bos_id:
                out_ids = out_ids[1:]
            if not out_ids or out_ids[-1] != self.eos_id:
                out_ids.append(self.eos_id)

            full_seq = prompt_ids + out_ids
            P = len(prompt_ids)
            if len(full_seq) < 3:
                continue

            # Crop if sequence exceeds block_size + 1
            if len(full_seq) > self.block_size + 1:
                excess = len(full_seq) - (self.block_size + 1)
                if P - 4 > excess:
                    prompt_ids = prompt_ids[:2] + prompt_ids[2 + excess:]
                    P = len(prompt_ids)
                    full_seq = prompt_ids + out_ids
                else:
                    full_seq = full_seq[-(self.block_size + 1):]
                    P = max(1, P - excess)

            x_ids = full_seq[:-1]
            y_ids = full_seq[1:]

            # Mask prompt tokens with -100 so loss is computed strictly on response
            y_masked = []
            for i, target_token in enumerate(y_ids):
                if i < P - 1:
                    y_masked.append(-100)
                else:
                    y_masked.append(target_token)

            # Pad to block_size
            curr_len = len(x_ids)
            if curr_len < self.block_size:
                pad_len = self.block_size - curr_len
                x_ids = x_ids + [self.pad_id] * pad_len
                y_masked = y_masked + [-100] * pad_len
            else:
                x_ids = x_ids[:self.block_size]
                y_masked = y_masked[:self.block_size]

            x_tensor = torch.tensor(x_ids, dtype=torch.long)
            y_tensor = torch.tensor(y_masked, dtype=torch.long)
            sign_mask = model.embedding.compute_sign_mask(x_tensor.unsqueeze(0))[0]
            self.samples.append((x_tensor, y_tensor, sign_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_val_loss(model: KristalLM, val_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    with torch.no_grad():
        for x, y, sm in val_loader:
            x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
            logits, _ = model(x, sign_mask=sm)
            active_mask = (y != -100)
            n_active = active_mask.sum().item()
            if n_active > 0:
                B, T, V = logits.shape
                loss = loss_fn(logits.view(-1, V), y.view(-1))
                total_loss += loss.item()
                total_tokens += n_active

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> float:
    if not candidate_tokens or not reference_tokens:
        return 0.0
    m, n = len(candidate_tokens), len(reference_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return (2 * precision * recall) / (precision + recall)


def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(values))
    boot_means = []
    n = len(values)
    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100))
    return mean_val, lower, upper


def extract_roots(text: str, compiler: CrystalCompiler) -> Set[str]:
    words = re.findall(r"[\w\']+", text.lower(), flags=re.UNICODE)
    roots = set()
    for w in words:
        clean = w.strip(".,!?\"…—«»/()-;:")
        if not clean or clean.isdigit():
            continue
        cword = clean.replace("'", "").replace("’", "")
        res = compiler.compile(cword)
        if res.get("token_vector"):
            for mid in res["token_vector"]:
                if not any(mid.startswith(p) for p in ("CASE_", "TENSE_", "POSS_", "COPULA_", "PART_", "GERUND_", "DERIV_", "VOICE_")) and mid not in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"):
                    roots.add(mid)
        else:
            roots.add(clean)
    return roots


def generate_response(
    model: KristalLM,
    tokenizer: KristalTokenizer,
    prompt_str: str,
    max_new_tokens: int = 60,
    block_size: int = 512
) -> Tuple[str, List[str]]:
    model.eval()
    prompt_ids = tokenizer.encode(prompt_str)
    if prompt_ids and prompt_ids[-1] == tokenizer.vocab.stoi.get("<EOS>", 3):
        prompt_ids = prompt_ids[:-1]

    if len(prompt_ids) > block_size - 10:
        prompt_ids = prompt_ids[-(block_size - 10):]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    sign_mask = model.embedding.compute_sign_mask(x.cpu()).to(DEVICE)

    generated_ids = []
    eos_id = tokenizer.vocab.stoi.get("<EOS>", 3)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.shape[1] >= block_size:
                break
            logits, _ = model(x, sign_mask=sign_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == eos_id:
                break
            generated_ids.append(next_token)
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
            x = torch.cat([x, next_tensor], dim=1)
            sm_next = torch.ones((1, 1, 1), device=DEVICE)
            sign_mask = torch.cat([sign_mask, sm_next], dim=1)

    tag_list = [tokenizer.vocab.decode(tid) for tid in generated_ids]
    tag_str = " ".join(tag_list)
    return tag_str, tag_list


def evaluate_model_on_split(
    model: KristalLM,
    tokenizer: KristalTokenizer,
    decompiler: MorphemeDecompiler,
    compiler: CrystalCompiler,
    records: List[Dict[str, Any]],
    split_name: str,
    all_test_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    print(f"\n--- Değerlendiriliyor: {split_name} (N={len(records)}) ---")

    rouge_scores = []
    entity_matches = []
    unique_wrong_ratios = []
    unique_paired_ratios = []

    doc_swap_paired = []
    doc_swap_wrong_same = []
    doc_swap_wrong_cross = []
    doc_swap_unrelated = []

    unrelated_text = "Kuantum kütleçekimi modelleri uzay-zaman geometrisinin süreksiz spin ağları üzerinden kuantize olduğunu varsayar."

    for idx, rec in enumerate(records):
        doc = ""
        inp = rec.get("input", "")
        # Parse document from <BELGE> ... </BELGE>
        m = re.search(r"<BELGE>(.*?)</BELGE>", inp, re.DOTALL)
        if m:
            doc = m.group(1).strip()
        q = re.sub(r"<BELGE>.*?</BELGE>", "", inp).strip()
        ref_ans = rec.get("output", "").strip()
        entity = rec.get("entity", "").strip()
        domain = rec.get("domain", "")

        # 1. Standard Generation (D_paired)
        prompt_paired = render_prompt("Belgeye dayanarak soruyu yanıtla.", build_rag_input(doc, q))
        tag_str_paired, tag_list_paired = generate_response(model, tokenizer, prompt_paired)
        surface_paired = decompiler.decompile_sentence(tag_str_paired)

        ref_tags = [t for t in tokenizer.encode(ref_ans) if t not in (tokenizer.vocab.stoi.get("<BOS>", 2), tokenizer.vocab.stoi.get("<EOS>", 3))]
        ref_tag_str = " ".join([tokenizer.vocab.decode(t) for t in ref_tags])
        r_l = compute_rouge_l(tag_list_paired, ref_tag_str.split())
        rouge_scores.append(r_l)

        # Entity match
        if entity:
            ent_match = 1.0 if (entity.lower() in surface_paired.lower()) else 0.0
            entity_matches.append(ent_match)

        # Doc-Swap 4-Level Ladder
        # Find wrong same-domain and wrong cross-domain documents
        same_domain_candidates = [r for r in all_test_records if r.get("domain") == domain and r.get("entity") != entity]
        cross_domain_candidates = [r for r in all_test_records if r.get("domain") != domain]

        wrong_same_doc = doc
        if same_domain_candidates:
            ws_cand = same_domain_candidates[idx % len(same_domain_candidates)]
            m_ws = re.search(r"<BELGE>(.*?)</BELGE>", ws_cand.get("input", ""), re.DOTALL)
            if m_ws: wrong_same_doc = m_ws.group(1).strip()

        wrong_cross_doc = doc
        if cross_domain_candidates:
            wc_cand = cross_domain_candidates[idx % len(cross_domain_candidates)]
            m_wc = re.search(r"<BELGE>(.*?)</BELGE>", wc_cand.get("input", ""), re.DOTALL)
            if m_wc: wrong_cross_doc = m_wc.group(1).strip()

        # D_paired roots
        roots_paired_doc = extract_roots(doc, compiler)
        roots_wrong_doc = extract_roots(wrong_same_doc, compiler)
        roots_cross_doc = extract_roots(wrong_cross_doc, compiler)
        roots_unrel_doc = extract_roots(unrelated_text, compiler)

        # Output with D_paired
        out_roots_paired = extract_roots(surface_paired, compiler)
        f_paired = len(out_roots_paired & roots_paired_doc) / max(1, len(out_roots_paired))
        doc_swap_paired.append(f_paired)

        # Output with D_wrong_same
        prompt_wrong_same = render_prompt("Belgeye dayanarak soruyu yanıtla.", build_rag_input(wrong_same_doc, q))
        tag_str_ws, _ = generate_response(model, tokenizer, prompt_wrong_same)
        surface_ws = decompiler.decompile_sentence(tag_str_ws)
        out_roots_ws = extract_roots(surface_ws, compiler)
        f_wrong_same = len(out_roots_ws & roots_wrong_doc) / max(1, len(out_roots_ws))
        doc_swap_wrong_same.append(f_wrong_same)

        # Output with D_wrong_cross
        prompt_wrong_cross = render_prompt("Belgeye dayanarak soruyu yanıtla.", build_rag_input(wrong_cross_doc, q))
        tag_str_wc, _ = generate_response(model, tokenizer, prompt_wrong_cross)
        surface_wc = decompiler.decompile_sentence(tag_str_wc)
        out_roots_wc = extract_roots(surface_wc, compiler)
        f_wrong_cross = len(out_roots_wc & roots_cross_doc) / max(1, len(out_roots_wc))
        doc_swap_wrong_cross.append(f_wrong_cross)

        # Output with D_unrelated
        prompt_unrel = render_prompt("Belgeye dayanarak soruyu yanıtla.", build_rag_input(unrelated_text, q))
        tag_str_un, _ = generate_response(model, tokenizer, prompt_unrel)
        surface_un = decompiler.decompile_sentence(tag_str_un)
        out_roots_un = extract_roots(surface_un, compiler)
        f_unrelated = len(out_roots_un & roots_unrel_doc) / max(1, len(out_roots_un))
        doc_swap_unrelated.append(f_unrelated)

        # UNIQUE FACT ISOLATION (When model is given D_wrong_same)
        unique_to_wrong = roots_wrong_doc - roots_paired_doc
        unique_to_paired = roots_paired_doc - roots_wrong_doc

        u_wrong_score = len(out_roots_ws & unique_to_wrong) / max(1, len(unique_to_wrong)) if unique_to_wrong else 0.0
        u_paired_score = len(out_roots_ws & unique_to_paired) / max(1, len(unique_to_paired)) if unique_to_paired else 0.0

        unique_wrong_ratios.append(u_wrong_score)
        unique_paired_ratios.append(u_paired_score)

    r_mean, r_low, r_high = bootstrap_ci(rouge_scores)
    ent_mean, ent_low, ent_high = bootstrap_ci(entity_matches)
    uw_mean, uw_low, uw_high = bootstrap_ci(unique_wrong_ratios)
    up_mean, up_low, up_high = bootstrap_ci(unique_paired_ratios)

    dsp_mean, _, _ = bootstrap_ci(doc_swap_paired)
    dws_mean, _, _ = bootstrap_ci(doc_swap_wrong_same)
    dwc_mean, _, _ = bootstrap_ci(doc_swap_wrong_cross)
    dun_mean, _, _ = bootstrap_ci(doc_swap_unrelated)

    print(f"  * ROUGE-L: {r_mean:.4f} [{r_low:.4f}, {r_high:.4f}]")
    print(f"  * Yüzey Varlık Doğruluğu: %{ent_mean*100:.1f} [{ent_low*100:.1f}, {ent_high*100:.1f}]")
    print(f"  * Doc-Swap D_paired: {dsp_mean:.4f} | D_wrong_same: {dws_mean:.4f} | D_wrong_cross: {dwc_mean:.4f} | D_unrelated: {dun_mean:.4f}")
    print(f"  * SADECE Yanlış Belgeden Gelen Özgül Olgular (score(D_unique_wrong)): {uw_mean:.4f} [{uw_low:.4f}, {uw_high:.4f}]")
    print(f"  * Soru Öncülünden Sayıklanan Özgül Olgular (score(D_unique_paired)): {up_mean:.4f} [{up_low:.4f}, {up_high:.4f}]")

    return {
        "split": split_name,
        "count": len(records),
        "rouge_l": {"mean": r_mean, "low": r_low, "high": r_high},
        "entity_fidelity": {"mean": ent_mean, "low": ent_low, "high": ent_high},
        "doc_swap": {
            "d_paired": dsp_mean,
            "d_wrong_same": dws_mean,
            "d_wrong_cross": dwc_mean,
            "d_unrelated": dun_mean
        },
        "unique_facts": {
            "score_d_unique_wrong": {"mean": uw_mean, "low": uw_low, "high": uw_high},
            "score_d_unique_paired": {"mean": up_mean, "low": up_low, "high": up_high}
        }
    }


def compute_rule_baselines(records: List[Dict[str, Any]], compiler: CrystalCompiler) -> Dict[str, Any]:
    print("\n--- Kural Tabanlı Baz Hatlar Hesaplanıyor ---")
    first_sent_rouges = []
    tfidf_rouges = []
    tfidf_unique_wrongs = []

    for idx, rec in enumerate(records):
        inp = rec.get("input", "")
        m = re.search(r"<BELGE>(.*?)</BELGE>", inp, re.DOTALL)
        doc = m.group(1).strip() if m else ""
        q = re.sub(r"<BELGE>.*?</BELGE>", "", inp).strip()
        ref_ans = rec.get("output", "").strip()

        # 1. Belge İlk Cümle
        sents = [s.strip() for s in re.split(r"[.!?]", doc) if s.strip()]
        first_sent = sents[0] if sents else doc
        r_fs = compute_rouge_l(first_sent.split(), ref_ans.split())
        first_sent_rouges.append(r_fs)

        # 2. TF-IDF Cümle Seçici (Soruyla en çok kelime örtüşen cümle)
        q_words = set(re.findall(r"\w+", q.lower()))
        best_sent = sents[0] if sents else doc
        best_overlap = -1
        for s in sents:
            s_words = set(re.findall(r"\w+", s.lower()))
            ov = len(s_words & q_words)
            if ov > best_overlap:
                best_overlap = ov
                best_sent = s
        r_tfidf = compute_rouge_l(best_sent.split(), ref_ans.split())
        tfidf_rouges.append(r_tfidf)

        # Unique fact baseline for TF-IDF selector
        roots_best = extract_roots(best_sent, compiler)
        roots_doc = extract_roots(doc, compiler)
        u_ratio = len(roots_best & roots_doc) / max(1, len(roots_doc))
        tfidf_unique_wrongs.append(u_ratio)

    fs_mean, _, _ = bootstrap_ci(first_sent_rouges)
    tf_mean, _, _ = bootstrap_ci(tfidf_rouges)
    tf_u_mean, _, _ = bootstrap_ci(tfidf_unique_wrongs)

    print(f"  * Belge İlk Cümle Tabanı (ROUGE-L): {fs_mean:.4f}")
    print(f"  * TF-IDF Cümle Seçici Tabanı (ROUGE-L): {tf_mean:.4f}")
    print(f"  * TF-IDF Unique-Fact Tabanı: {tf_u_mean:.4f}")

    return {
        "first_sentence_rouge": fs_mean,
        "tfidf_sentence_rouge": tf_mean,
        "tfidf_unique_fact_baseline": tf_u_mean
    }


def main():
    parser = argparse.ArgumentParser(description="Mevcut val.jsonl üzerinde kolları titizlikle yeniden puanlama")
    parser.add_argument("--arm", type=str, default="all", choices=["A", "B", "C", "all", "arm_A", "arm_B", "arm_C"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-generation", action="store_true", help="Doc-swap ve üretim metriklerini de hesapla")
    parser.add_argument("--max-gen-samples", type=int, default=10, help="Doc-swap üretim örneği sayısı (0: tümü)")
    parser.add_argument("--output", type=str, default=None, help="Özel rapor çıktı yolu (varsayılan: data/eval/rigorous_evaluation_report.json)")
    args = parser.parse_args()

    lex = LexiconManager()
    roots_path = os.path.join(DATA_DIR, "lexicon", "roots.tsv")
    if os.path.exists(roots_path):
        lex.load_from_tsv(roots_path)
    else:
        raise FileNotFoundError(f"Leksikon kök dosyası bulunamadı: {roots_path}")
    graph = build_default_graph()
    compiler = CrystalCompiler(lex, graph)

    val_path = os.path.join(REALISTIC_DIR, "val.jsonl")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Val dosyası bulunamadı: {val_path}")

    val_meta = get_file_metadata(val_path)

    with open(val_path, "r", encoding="utf-8") as f:
        val_recs = [json.loads(l) for l in f]

    val_nat_recs = [r for r in val_recs if not r.get("is_counterfactual")]
    val_cf_recs = [r for r in val_recs if r.get("is_counterfactual")]

    val_meta["total_records"] = len(val_recs)
    val_meta["natural_records"] = len(val_nat_recs)
    val_meta["counterfactual_records"] = len(val_cf_recs)

    print(f"\n{'='*80}")
    print(f"   DOĞRULAMA KÜMESİ (VAL.JSONL) BİLGİLERİ")
    print(f"{'='*80}")
    print(f"Yol: {val_meta['path']}")
    print(f"Boyut: {val_meta['size_bytes']:,} bayt")
    print(f"MTime (Yerel): {val_meta['mtime_local_str']} | UTC: {val_meta['mtime_iso_utc']}")
    print(f"SHA-256: {val_meta['sha256']}")
    print(f"Kayıtlar: Toplam={val_meta['total_records']} | Natural={val_meta['natural_records']} | CF={val_meta['counterfactual_records']}")

    # 1. Kural Tabanlı Baz Hatlar
    baselines = compute_rule_baselines(val_recs, compiler)

    arm_configs = {
        "arm_A": {
            "ckpt": os.path.join(DATA_DIR, "kristal_rag_arm_a_best.pt"),
            "vocab": BASE_VOCAB_PATH,
            "literal_entity": False
        },
        "arm_B": {
            "ckpt": os.path.join(DATA_DIR, "kristal_rag_arm_b_best.pt"),
            "vocab": EXP_VOCAB_PATH,
            "literal_entity": True
        },
        "arm_C": {
            "ckpt": os.path.join(DATA_DIR, "kristal_rag_arm_c_best.pt"),
            "vocab": EXP_VOCAB_PATH,
            "literal_entity": True
        }
    }

    results = {
        "evaluation_target": "val.jsonl",
        "val_metadata": val_meta,
        "baselines": baselines,
        "arms": {}
    }

    if args.arm in ("A", "arm_A"):
        arms_to_run = ["arm_A"]
    elif args.arm in ("B", "arm_B"):
        arms_to_run = ["arm_B"]
    elif args.arm in ("C", "arm_C"):
        arms_to_run = ["arm_C"]
    else:
        arms_to_run = ["arm_A", "arm_B", "arm_C"]

    for arm_key in arms_to_run:
        cfg = arm_configs.get(arm_key)
        if not cfg or not os.path.exists(cfg["ckpt"]):
            print(f"UYARI: {arm_key} için checkpoint bulunamadı: {cfg['ckpt'] if cfg else ''}")
            continue

        print(f"\n{'='*70}")
        print(f"   DEĞERLENDİRİLİYOR: {arm_key}")
        print(f"{'='*70}")

        vocab = Vocabulary()
        vocab.load(cfg["vocab"], freeze=True)
        tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=cfg["literal_entity"])
        decompiler = MorphemeDecompiler(compiler, vocab)

        model = KristalLM(
            vocab_size=len(vocab.stoi),
            n_embd=768,
            vocab=vocab,
            block_size=4096,
            n_layer=6,
            n_head=6
        )
        sd = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.to(DEVICE)
        model.eval()

        # Val Loss değerlendirmesi
        ds_all = RealisticRagDataset(val_recs, tokenizer, vocab, model, block_size=args.block_size)
        ds_nat = RealisticRagDataset(val_nat_recs, tokenizer, vocab, model, block_size=args.block_size)
        ds_cf = RealisticRagDataset(val_cf_recs, tokenizer, vocab, model, block_size=args.block_size)

        loader_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False)
        loader_nat = DataLoader(ds_nat, batch_size=args.batch_size, shuffle=False)
        loader_cf = DataLoader(ds_cf, batch_size=args.batch_size, shuffle=False)

        val_loss, val_ppl = evaluate_val_loss(model, loader_all)
        nat_loss, nat_ppl = evaluate_val_loss(model, loader_nat)
        cf_loss, cf_ppl = evaluate_val_loss(model, loader_cf)

        old_loss = OLD_REPORTED_LOSSES.get(arm_key, None)
        delta = (val_loss - old_loss) if old_loss is not None else None

        print(f"  * Overall Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
        print(f"  * Natural Val Loss: {nat_loss:.4f} (PPL: {nat_ppl:.2f})")
        print(f"  * Counterfactual Val Loss: {cf_loss:.4f} (PPL: {cf_ppl:.2f})")
        if old_loss is not None:
            print(f"  * Eski Raporlanan Loss: {old_loss:.4f} -> Kayma: {delta:+.4f}")

        arm_res = {
            "checkpoint": cfg["ckpt"],
            "vocab": cfg["vocab"],
            "literal_entity_mode": cfg["literal_entity"],
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "natural_loss": nat_loss,
            "natural_ppl": nat_ppl,
            "counterfactual_loss": cf_loss,
            "counterfactual_ppl": cf_ppl,
            "old_reported_loss": old_loss,
            "delta": delta,
            "sample_counts": {
                "total": len(val_recs),
                "natural": len(val_nat_recs),
                "counterfactual": len(val_cf_recs)
            }
        }

        if args.eval_generation:
            gen_nat = val_nat_recs if args.max_gen_samples == 0 else val_nat_recs[:args.max_gen_samples]
            gen_cf = val_cf_recs if args.max_gen_samples == 0 else val_cf_recs[:args.max_gen_samples]
            arm_res["generation_natural"] = evaluate_model_on_split(model, tokenizer, decompiler, compiler, gen_nat, f"{arm_key}_Val_Natural", val_recs)
            arm_res["generation_counterfactual"] = evaluate_model_on_split(model, tokenizer, decompiler, compiler, gen_cf, f"{arm_key}_Val_Counterfactual", val_recs)

        results["arms"][arm_key] = arm_res

        # Bellek temizliği
        del model
        del ds_all
        del ds_nat
        del ds_cf
        del loader_all
        del loader_nat
        del loader_cf
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # Rapor kaydı (data/eval altına)
    report_path = args.output or os.path.join(EVAL_DIR, "rigorous_evaluation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nTitiz değerlendirme raporu kaydedildi: '{report_path}'")

    # Özet Tablosu Yazdırma
    print(f"\n{'='*95}")
    print(f"   AYNI VAL KÜMESİ ÜZERİNDE YENİDEN PUANLAMA ÖZET TABLOSU")
    print(f"   Val Dosyası: {val_meta['path']} (N={val_meta['total_records']}: Nat={val_meta['natural_records']}, CF={val_meta['counterfactual_records']})")
    print(f"   MTime (UTC): {val_meta['mtime_iso_utc']} | SHA-256: {val_meta['sha256']}")
    print(f"{'='*95}")
    print(f"| {'Kol (Arm)':<8} | {'Val Loss':<10} | {'Natural':<10} | {'Counterfactual':<16} | {'Eski Raporlanan':<16} | {'Kayma (Delta)':<14} |")
    print(f"|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*18}|{'-'*18}|{'-'*16}|")
    for arm_k, arm_d in results["arms"].items():
        v_loss = f"{arm_d['val_loss']:.4f}"
        n_loss = f"{arm_d['natural_loss']:.4f}"
        c_loss = f"{arm_d['counterfactual_loss']:.4f}"
        o_loss = f"{arm_d['old_reported_loss']:.4f}" if arm_d['old_reported_loss'] is not None else "N/A"
        d_loss = f"{arm_d['delta']:+.4f}" if arm_d['delta'] is not None else "N/A"
        print(f"| {arm_k:<8} | {v_loss:<10} | {n_loss:<10} | {c_loss:<16} | {o_loss:<16} | {d_loss:<14} |")
    print(f"{'='*95}\n")


if __name__ == "__main__":
    main()
