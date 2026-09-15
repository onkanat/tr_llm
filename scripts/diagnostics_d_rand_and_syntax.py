#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAZ B2 EK TANI: D_RAND KOLU, F-STRING KAYNAK DOĞRULAMASI VE SYNTAX_SIGNATURE
============================================================================
1. Doc-Swap D_rand Kolu: score(D_rand) vs score(D') vs score(D) ile H1 vs H2 ayrımı.
2. f-string Kaynak Analizi: Şablon kuyruklarının Train vs Test frekansları ve tekillik oranı.
3. syntax_signature Çalıştırılabilir Fonksiyonu ve Çıktı Küme Histogramı.
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


# ==============================================================================
# GÖREV 2: F-STRING KAYNAK DOĞRULAMASI (TRAIN VS TEST DAĞILIMI)
# ==============================================================================
def run_fstring_source_audit():
    print("\n" + "=" * 75)
    print(" GÖREV 2: F-STRING KAYNAK DOĞRULAMASI (TRAIN VS TEST DAĞILIMI)")
    print("=" * 75)

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    # Target template tails
    tails = [
        "sağlayarak parçaların güvenle birleşmesini temin eder.",
        "tatbik edilerek yüzeyin neme karşı korunmasını ve estetik görünmesini sağlar.",
        "anlayışını vurgular ve"
    ]

    print("  Şablon Kuyruklarının Frekans Karşılaştırması:")
    print(f"  {'Şablon Kuyruğu (Tail)':<60} | {'Train (N=800)':<14} | {'Test (N=100)':<14}")
    print("  " + "-" * 90)

    for tail in tails:
        tr_cnt = sum(1 for r in train_recs if tail in r.get("output", ""))
        te_cnt = sum(1 for r in test_recs if tail in r.get("output", ""))
        print(f"  {tail[:58]:<60} | {tr_cnt:<14} | {te_cnt:<14}")

    # Count how many total samples in train and test contain ANY of these template tails
    tr_template_total = sum(1 for r in train_recs if any(t in r.get("output", "") for t in tails))
    te_template_total = sum(1 for r in test_recs if any(t in r.get("output", "") for t in tails))

    print("  " + "-" * 90)
    print(f"  {'TOPLAM ŞABLONLU ÖRNEK SAYISI':<60} | {tr_template_total:<14} | {te_template_total:<14}")
    print(f"  {'ŞABLON ORANI (%)':<60} | %{tr_template_total/len(train_recs)*100:<13.1f} | %{te_template_total/len(test_recs)*100:<13.1f}")
    
    return {
        "train_template_count": tr_template_total,
        "test_template_count": te_template_total,
        "train_template_pct": tr_template_total / len(train_recs) * 100,
        "test_template_pct": te_template_total / len(test_recs) * 100
    }


# ==============================================================================
# GÖREV 3: SYNTAX_SIGNATURE VE KÜME HİSTOGRAMI
# ==============================================================================
def get_syntax_signature(text: str, tokenizer: KristalTokenizer, vocab: Vocabulary) -> str:
    """Extracts the grammatical affix and closed-class tag skeleton of a text."""
    token_ids = tokenizer.encode(text)
    tags = [vocab.decode(tid) for tid in token_ids]
    
    skeleton_prefixes = (
        "CASE_", "TENSE_", "POSS_", "COPULA_", "PART_", 
        "GERUND_", "DERIV_", "PLURAL", "NEG", "PERSON_"
    )
    skeleton_tags = [t for t in tags if any(t.startswith(p) for p in skeleton_prefixes)]
    return " ".join(skeleton_tags)


def run_syntax_signature_audit(tokenizer, vocab):
    print("\n" + "=" * 75)
    print(" GÖREV 3: SYNTAX_SIGNATURE ANALİZİ VE KÜME HİSTOGRAMI")
    print("=" * 75)

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    train_signatures = [get_syntax_signature(r["output"], tokenizer, vocab) for r in train_recs]
    test_signatures = [get_syntax_signature(r["output"], tokenizer, vocab) for r in test_recs]

    train_sig_counter = Counter(train_signatures)
    test_sig_counter = Counter(test_signatures)

    unique_train_sigs = len(train_sig_counter)
    unique_test_sigs = len(test_sig_counter)

    # Measure Train ∩ Test signature overlap
    shared_sigs = set(train_sig_counter.keys()) & set(test_sig_counter.keys())
    test_samples_with_shared_sig = sum(test_sig_counter[sig] for sig in shared_sigs)
    pct_test_syntax_leak = (test_samples_with_shared_sig / len(test_recs)) * 100.0

    print(f"  Eğitim Kümesi (Train N=800) Tekil Syntax İskeleti Sayısı: {unique_train_sigs}")
    print(f"  Test Kümesi (Test N=100) Tekil Syntax İskeleti Sayısı    : {unique_test_sigs}")
    print(f"  Ortak Syntax İskeleti Sayısı                             : {len(shared_sigs)}")
    print(f"  Ortak İskelete Sahip Test Örnekleri Sayısı               : {test_samples_with_shared_sig} / {len(test_recs)}")
    print(f"  Syntax-Held-Out İhlal Oranı (Sızıntı)                    : %{pct_test_syntax_leak:.1f}")

    print("\n  En Büyük 5 Syntax İskeletinin Testteki Dağılımı:")
    for i, (sig, count) in enumerate(test_sig_counter.most_common(5), 1):
        tr_count = train_sig_counter.get(sig, 0)
        sig_disp = (sig[:70] + "...") if len(sig) > 70 else sig
        print(f"    Küme {i}: Test'te {count:2d} kez | Train'de {tr_count:3d} kez | İskelet: [{sig_disp}]")

    return {
        "unique_train_sigs": unique_train_sigs,
        "unique_test_sigs": unique_test_sigs,
        "shared_sigs": len(shared_sigs),
        "test_syntax_leak_pct": pct_test_syntax_leak
    }


# ==============================================================================
# GÖREV 1: D_RANDOM KOLU İLE H1 VS H2 AYRIŞTIRMASI
# ==============================================================================
def run_d_rand_experiment(model, tokenizer, vocab):
    print("\n" + "=" * 75)
    print(" GÖREV 1: DOC-SWAP D_RAND KOLU VE H1 VS H2 AYRIŞTIRMASI")
    print("=" * 75)
    print("  Hipotez 1 (H1): Model harici belge okumuyor, soru-şablon ezberini oynatıyor.")
    print("  Hipotez 2 (H2): Model herhangi bir belgeyi zayıf okuyor; D'nin skoru sözcüksel örtüşmeden geliyor.")
    print("  Kriter: score(D_rand) < score(D') < score(D) ise H2 (zayıf okuma); score(D_rand) ≈ score(D') < score(D) ise H1 (sıfır okuma).")

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    eos_id = vocab.stoi.get("<EOS>", 2)
    n = len(test_recs)

    # Independent random documents (unrelated academic / domain texts)
    unrelated_corpus = [
        "Galatasaray Lisesi binasının neoklasik cephesi, Beyoğlu caddesindeki tarihi dokuyu tamamlayan anıtsal giriş kapısıyla dikkat çeker.",
        "Kuantum mekaniğinde dalga fonksiyonunun çökmesi, ölçüm anında süperpozisyon durumunun belirli bir özdeğere indirgenmesini açıklar.",
        "Organik kimyada benzen halkasının aromatik kararlılığı, altı pi elektronunun delokalize olması ve rezonans enerjisi ile açıklanır.",
        "Merkez bankasının politika faizini artırması, piyasadaki toplam talebi kısıtlayarak enflasyonist baskıları frenlemeyi hedefler.",
        "Derin deniz termal bacalarında yaşayan kemosentetik bakteriler, güneş ışığı olmaksızın kükürt bileşiklerini oksitleyerek enerji üretir."
    ]

    scores_d_prime = []
    scores_d_orig = []
    scores_d_rand = []

    for i in range(n):
        orig_rec = test_recs[i]
        swap_rec = test_recs[(i + 1) % n]
        rand_doc = unrelated_corpus[i % len(unrelated_corpus)]

        q = orig_rec["input"]
        doc_orig = orig_rec["belge"]
        doc_swap = swap_rec["belge"]

        # 1. Generate with D_swap
        p_swap = f"<INSTRUCTION> Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla. </INSTRUCTION> <INPUT> {q} </INPUT> <BELGE> {doc_swap} </BELGE> <OUTPUT>"
        p_ids = tokenizer.encode(p_swap)
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

        swap_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(doc_swap)], vocab)
        orig_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(doc_orig)], vocab)
        rand_roots = extract_content_roots([vocab.decode(t) for t in tokenizer.encode(rand_doc)], vocab)

        if gen_roots:
            s_prime = len(gen_roots & swap_roots) / len(gen_roots)
            s_orig = len(gen_roots & orig_roots) / len(gen_roots)
            s_rand = len(gen_roots & rand_roots) / len(gen_roots)
        else:
            s_prime, s_orig, s_rand = 0.0, 0.0, 0.0

        scores_d_prime.append(s_prime)
        scores_d_orig.append(s_orig)
        scores_d_rand.append(s_rand)

    def bootstrap_ci(arr, n_boot=1000):
        rng = np.random.RandomState(42)
        means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
        return float(np.mean(means)), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    m_prime, ci_prime = bootstrap_ci(scores_d_prime)
    m_orig, ci_orig = bootstrap_ci(scores_d_orig)
    m_rand, ci_rand = bootstrap_ci(scores_d_rand)

    print("\n  AMPİRİK SONUÇLAR (Bootstrap %95 GA):")
    print(f"  score(D - Orijinal/Eski Belge)    : {m_orig:.4f} [{ci_orig[0]:.4f}, {ci_orig[1]:.4f}]")
    print(f"  score(D' - Verilen Yeni Belge)    : {m_prime:.4f} [{ci_prime[0]:.4f}, {ci_prime[1]:.4f}]")
    print(f"  score(D_rand - Rastgele Belge)    : {m_rand:.4f} [{ci_rand[0]:.4f}, {ci_rand[1]:.4f}]")
    print("  " + "-" * 75)
    print(f"  Fark: score(D') - score(D_rand)   : {m_prime - m_rand:+.4f}")
    print(f"  Fark: score(D) - score(D')        : {m_orig - m_prime:+.4f}")

    if m_prime > m_rand + 0.15 and ci_prime[0] > ci_rand[1]:
        diagnosis = "H2 DESTEKLENİYOR (Modelde zayıf ama gerçek bir belge-okuma sinyali mevcut: D_rand < D')"
    else:
        diagnosis = "H1 DESTEKLENİYOR (Model harici belgeyi okumuyor, soru-şablon ezberini oynatıyor: D_rand ≈ D')"

    print(f"\n  NİHAİ TEŞHİS: {diagnosis}")

    return {
        "mean_score_orig": m_orig,
        "ci_orig": ci_orig,
        "mean_score_prime": m_prime,
        "ci_prime": ci_prime,
        "mean_score_rand": m_rand,
        "ci_rand": ci_rand,
        "diagnosis": diagnosis
    }


def main():
    print("=" * 75)
    print(" FAZ B2 EK TANI PROTOKOLÜ (D_RAND, F-STRING VE SYNTAX_SIGNATURE)")
    print("=" * 75)

    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 1. f-string Source Audit
    fstring_res = run_fstring_source_audit()

    # 2. Syntax-Held-Out Audit
    syntax_res = run_syntax_signature_audit(tokenizer, vocab)

    # 3. Model Load & D_rand Experiment
    model = KristalLM(
        vocab_size=len(vocab.stoi),
        n_embd=768,
        vocab=vocab,
        block_size=4096,
        n_layer=6,
        n_head=6
    )
    sd = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()

    drand_res = run_d_rand_experiment(model, tokenizer, vocab)

    # Save summary
    out_summary = {
        "fstring_source_audit": fstring_res,
        "syntax_signature_audit": syntax_res,
        "drand_experiment": drand_res
    }
    out_file = os.path.join(PILOT_DIR, "drand_and_syntax_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSonuçlar kaydedildi: {out_file}\n")


if __name__ == "__main__":
    main()
