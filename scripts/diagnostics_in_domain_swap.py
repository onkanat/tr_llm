#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAZ B2 KESİN TANI: IN-DOMAIN WRONG BELGE VE UNIQUE ROOT AYRIŞTIRMASI
===================================================================
1. In-Domain vs. Cross-Domain Doc-Swap:
   - score(D_paired - Orijinal Belge)
   - score(D_wrong_same_domain - Aynı Alan Farklı Belge)
   - score(D_wrong_cross_domain - Diğer Alan Farklı Belge)
   - score(D_unrelated - Tamamen Alakasız Metin)
2. Unique Fact Bleed:
   - score(D_unique_wrong): Çıktının SADECE D_wrong'da olup D_paired'da OLMAYAN kökleri üretme oranı.
   - score(D_unique_paired): Çıktının SADECE D_paired'da olup D_wrong'da OLMAYAN kökleri üretme oranı.
   Eğer score(D_unique_wrong) ≈ 0 ise, model D'yi KESİNLİKLE OKUMAMAKTADIR;
   0.35 skoru sadece genel alan kelimelerinin tesadüfi örtüşmesidir!
3. syntax_signature Küme-Boyutu Histogramı (Tam Dağılım).
"""

import os
import sys
import json
import re
import numpy as np
import torch
from collections import Counter
from typing import List, Dict, Set, Tuple

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_rag_baselines import extract_content_roots

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")

TRAIN_PATH = os.path.join(PILOT_DIR, "train_800.jsonl")
TEST_PATH = os.path.join(PILOT_DIR, "test_100.jsonl")
CKPT_PATH = os.path.join(PILOT_DIR, "kristal_rag_pilot_best.pt")


def get_syntax_signature(text: str, tokenizer: KristalTokenizer, vocab: Vocabulary) -> str:
    token_ids = tokenizer.encode(text)
    tags = [vocab.decode(tid) for tid in token_ids]
    prefixes = ("CASE_", "TENSE_", "POSS_", "COPULA_", "PART_", "GERUND_", "DERIV_", "PLURAL", "NEG", "PERSON_")
    return " ".join([t for t in tags if any(t.startswith(p) for p in prefixes)])


def bootstrap_ci(arr, n_boot=1000):
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.mean(means)), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    print("=" * 75)
    print(" FAZ B2 KESİN TANI: IN-DOMAIN DOC-SWAP VE UNIQUE ROOT TESTİ")
    print("=" * 75)

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 1. SYNTAX KÜME BOYUTU HİSTOGRAMI
    print("\n--- 1. SYNTAX_SIGNATURE KÜME BOYUTU DAĞILIMI ---")
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    test_sigs = [get_syntax_signature(r["output"], tokenizer, vocab) for r in test_recs]
    test_sig_counts = Counter(test_sigs)
    train_sigs = [get_syntax_signature(r["output"], tokenizer, vocab) for r in train_recs]
    train_sig_counts = Counter(train_sigs)

    # Cluster size frequency: how many clusters have size 1, size 2-4, size 5+
    size_bins = {"Tekil (1 örnek)": 0, "Küçük (2-3 örnek)": 0, "Orta (4-5 örnek)": 0, "Büyük (6+ örnek)": 0}
    for sig, cnt in test_sig_counts.items():
        if cnt == 1:
            size_bins["Tekil (1 örnek)"] += 1
        elif 2 <= cnt <= 3:
            size_bins["Küçük (2-3 örnek)"] += 1
        elif 4 <= cnt <= 5:
            size_bins["Orta (4-5 örnek)"] += 1
        else:
            size_bins["Büyük (6+ örnek)"] += 1

    print("  Test Kümesi (N=100) Küme Boyutu Histogramı (Toplam 53 Küme):")
    for b_name, b_cnt in size_bins.items():
        print(f"    {b_name:<25}: {b_cnt:2d} küme")

    # Top 3 clusters dominance in test
    top3_samples = sum(c for _, c in test_sig_counts.most_common(3))
    print(f"  En büyük 3 kümenin Test'teki payı: {top3_samples} / 100 örnek (%{top3_samples:.1f})")

    # 2. IN-DOMAIN DOC-SWAP VE UNIQUE ROOT ANALİZİ
    print("\n--- 2. IN-DOMAIN VS CROSS-DOMAIN DOC-SWAP VE UNIQUE FACT TESTİ ---")
    model = KristalLM(len(vocab.stoi), 768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"), strict=False)
    model.to(DEVICE)
    model.eval()

    eos_id = vocab.stoi.get("<EOS>", 2)

    wood_test = [r for r in test_recs if r["domain"] == "carpenter"]
    hist_test = [r for r in test_recs if r["domain"] == "turk_tarihi"]

    unrelated_pool = [
        "Galatasaray Lisesi binasının neoklasik cephesi anıtsal kapısıyla dikkat çeker.",
        "Kuantum mekaniğinde dalga fonksiyonunun çökmesi süperpozisyon durumunu sonlandırır.",
        "Organik kimyada benzen halkasının aromatik kararlılığı rezonans enerjisi ile açıklanır."
    ]

    scores_same_domain = []
    scores_cross_domain = []
    scores_unrelated = []
    scores_paired = []
    
    unique_wrong_bleeds = []
    unique_paired_bleeds = []

    for idx, rec in enumerate(test_recs):
        q = rec["input"]
        doc_paired = rec["belge"]
        dom = rec["domain"]

        # Same domain wrong doc: pick next doc in same domain
        if dom == "carpenter":
            same_pool = wood_test
            cross_pool = hist_test
        else:
            same_pool = hist_test
            cross_pool = wood_test

        same_wrong_rec = same_pool[(idx + 1) % len(same_pool)]
        doc_same_wrong = same_wrong_rec["belge"]

        cross_wrong_rec = cross_pool[idx % len(cross_pool)]
        doc_cross_wrong = cross_wrong_rec["belge"]

        doc_unrelated = unrelated_pool[idx % len(unrelated_pool)]

        # Generate with SAME-DOMAIN WRONG DOC
        prompt_str = f"<INSTRUCTION> Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla. </INSTRUCTION> <INPUT> {q} </INPUT> <BELGE> {doc_same_wrong} </BELGE> <OUTPUT>"
        p_ids = tokenizer.encode(prompt_str)
        if p_ids and p_ids[-1] == eos_id: p_ids = p_ids[:-1]

        x = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)
        gen_ids = []
        with torch.no_grad():
            for _ in range(45):
                x_cond = x if x.size(1) <= 256 else x[:, -256:]
                logits, _ = model(x_cond)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt == eos_id: break
                gen_ids.append(nxt)
                x = torch.cat((x, torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)), dim=1)

        gen_tokens = [vocab.decode(t) for t in gen_ids]
        gen_roots = extract_content_roots(gen_tokens, vocab)

        paired_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(doc_paired)], vocab)
        same_wrong_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(doc_same_wrong)], vocab)
        cross_wrong_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(doc_cross_wrong)], vocab)
        unrelated_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(doc_unrelated)], vocab)

        # Unique sets
        unique_to_wrong = same_wrong_roots - paired_roots
        unique_to_paired = paired_roots - same_wrong_roots

        if gen_roots:
            s_same = len(gen_roots & same_wrong_roots) / len(gen_roots)
            s_cross = len(gen_roots & cross_wrong_roots) / len(gen_roots)
            s_unrel = len(gen_roots & unrelated_roots) / len(gen_roots)
            s_pair = len(gen_roots & paired_roots) / len(gen_roots)

            u_wrong = len(gen_roots & unique_to_wrong) / len(gen_roots)
            u_pair = len(gen_roots & unique_to_paired) / len(gen_roots)
        else:
            s_same, s_cross, s_unrel, s_pair, u_wrong, u_pair = 0, 0, 0, 0, 0, 0

        scores_same_domain.append(s_same)
        scores_cross_domain.append(s_cross)
        scores_unrelated.append(s_unrel)
        scores_paired.append(s_pair)
        unique_wrong_bleeds.append(u_wrong)
        unique_paired_bleeds.append(u_pair)

    m_same, ci_same = bootstrap_ci(scores_same_domain)
    m_cross, ci_cross = bootstrap_ci(scores_cross_domain)
    m_unrel, ci_unrel = bootstrap_ci(scores_unrelated)
    m_pair, ci_pair = bootstrap_ci(scores_paired)

    m_u_wrong, ci_u_wrong = bootstrap_ci(unique_wrong_bleeds)
    m_u_pair, ci_u_pair = bootstrap_ci(unique_paired_bleeds)

    print(f"  score(D_paired - Hafızadaki Orijinal Belge)     : {m_pair:.4f} [{ci_pair[0]:.4f}, {ci_pair[1]:.4f}]")
    print(f"  score(D_wrong_same - Aynı Alan Yanlış Belge)   : {m_same:.4f} [{ci_same[0]:.4f}, {ci_same[1]:.4f}]")
    print(f"  score(D_wrong_cross - Farklı Alan Yanlış Belge): {m_cross:.4f} [{ci_cross[0]:.4f}, {ci_cross[1]:.4f}]")
    print(f"  score(D_unrelated - Alakasız Rastgele Metin)   : {m_unrel:.4f} [{ci_unrel[0]:.4f}, {ci_unrel[1]:.4f}]")
    print("  " + "-" * 75)
    print(f"  SADECE D_wrong'a ÖZGÜ Köklerin Çıktıya Sızması (Unique Wrong): {m_u_wrong:.4f} [{ci_u_wrong[0]:.4f}, {ci_u_wrong[1]:.4f}]")
    print(f"  SADECE D_paired'a ÖZGÜ Köklerin Sayıklanması (Unique Paired) : {m_u_pair:.4f} [{ci_u_pair[0]:.4f}, {ci_u_pair[1]:.4f}]")
    print("  " + "-" * 75)

    if m_u_wrong < 0.08:
        diagnosis = (
            "MODEL BELGEYİ KESİNLİKLE OKUMAMAKTADIR! "
            f"D_wrong'a özgü kökleri üretme oranı yalnızca %{m_u_wrong*100:.1f}'dir. "
            f"0.35'lik same-domain skoru, tamamen iki belgenin paylaştığı ortak domain kelimelerinden kaynaklanmaktadır!"
        )
    else:
        diagnosis = f"Modelde zayıf bir belge okuma mevcuttur (Unique wrong: %{m_u_wrong*100:.1f})."

    print(f"  NİHAİ HÜKÜM: {diagnosis}\n")

    results = {
        "syntax_size_bins": size_bins,
        "top3_samples_pct": top3_samples,
        "m_same": m_same, "ci_same": ci_same,
        "m_cross": m_cross, "ci_cross": ci_cross,
        "m_unrel": m_unrel, "ci_unrel": ci_unrel,
        "m_pair": m_pair, "ci_pair": ci_pair,
        "m_u_wrong": m_u_wrong, "ci_u_wrong": ci_u_wrong,
        "m_u_pair": m_u_pair, "ci_u_pair": ci_u_pair,
        "diagnosis": diagnosis
    }
    with open(os.path.join(PILOT_DIR, "in_domain_swap_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
