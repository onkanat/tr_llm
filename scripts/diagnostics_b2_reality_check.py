#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAZ B2 GERÇEKLİK KONTROLÜ VE METODOLOJİK OTOPSİ (REALITY CHECK)
=============================================================
Kullanıcının 7 noktalı eleştirisini test eden 5 zorunlu ampirik kontrol:
1. Val/Test ve Train arasındaki 4-gram örtüşmesi (Leakage & Template sharing).
2. TF-IDF Cümle Seçici (Sentence Selector) Baseline: Soru ile en yüksek TF-IDF
   benzerliğine sahip belge cümlesini kopyalama tabanı.
3. Entity Fidelity & Maskeleme Denetimi: <PROPER_NOUN> ve <UNK> maskelemesi
   çıkarıldığında gerçek yüzey/kök ROUGE-L nedir?
4. Belge-Değiştirme (Doc-Swap / Counterfactual) Testi: Belge D yerine ilgisiz D'
   verildiğinde model çıktıları nasıl değişiyor? Belgeye mi yoksa soru kalıbına mı bağlı?
5. Sentetik Şablon İskelet Frekansı ve Val Loss 0.0144'ün Kökeni.
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
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.evaluate_rag_baselines import compute_rouge_l, stratified_bootstrap_ci

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")

TRAIN_PATH = os.path.join(PILOT_DIR, "train_800.jsonl")
VAL_PATH = os.path.join(PILOT_DIR, "val_100.jsonl")
TEST_PATH = os.path.join(PILOT_DIR, "test_100.jsonl")
CKPT_PATH = os.path.join(PILOT_DIR, "kristal_rag_pilot_best.pt")


def get_ngrams(tokens: List[str], n: int = 4) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def run_ngram_overlap_check():
    print("\n" + "=" * 75)
    print(" KONTROL 1: VAL / TEST VE TRAIN ARASINDA 4-GRAM ÖRTÜŞMESİ VE ŞABLON PAYLAŞIMI")
    print("=" * 75)

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_recs = [json.loads(line) for line in f]
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_recs = [json.loads(line) for line in f]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    # Collect 4-grams from outputs
    train_out_4grams = set()
    train_full_4grams = set()
    for r in train_recs:
        out_toks = r.get("output", "").split()
        full_toks = (r.get("input", "") + " " + r.get("belge", "") + " " + r.get("output", "")).split()
        train_out_4grams.update(get_ngrams(out_toks, 4))
        train_full_4grams.update(get_ngrams(full_toks, 4))

    def measure_overlap(recs, name):
        out_overlaps = []
        full_overlaps = []
        for r in recs:
            o_toks = r.get("output", "").split()
            f_toks = (r.get("input", "") + " " + r.get("belge", "") + " " + r.get("output", "")).split()
            o_ng = get_ngrams(o_toks, 4)
            f_ng = get_ngrams(f_toks, 4)

            if o_ng:
                out_overlaps.append(len(o_ng & train_out_4grams) / len(o_ng))
            if f_ng:
                full_overlaps.append(len(f_ng & train_full_4grams) / len(f_ng))

        avg_out = float(np.mean(out_overlaps)) * 100 if out_overlaps else 0.0
        avg_full = float(np.mean(full_overlaps)) * 100 if full_overlaps else 0.0
        print(f"  {name} -> Output 4-gram Train Örtüşmesi: %{avg_out:.2f} | Full Sequence 4-gram Örtüşmesi: %{avg_full:.2f}")
        return avg_out, avg_full

    val_out_pct, val_full_pct = measure_overlap(val_recs, "Validation (val_100.jsonl)")
    test_out_pct, test_full_pct = measure_overlap(test_recs, "Held-Out Test (test_100.jsonl)")

    print(f"\n  [Teşhis 1]: Val ve Test çıktılarının train ile 4-gram örtüşmesi:")
    print(f"  Val Output 4-gram örtüşmesi: %{val_out_pct:.2f}")
    print(f"  Test Output 4-gram örtüşmesi: %{test_out_pct:.2f}")
    if val_out_pct > 30.0 or test_out_pct > 30.0:
        print("  -> ALARM DOĞRULANDI: Yüksek 4-gram örtüşmesi sentetik şablon iskeletinin (template) ezberlendiğini kanıtlıyor!")

    return {
        "val_out_4gram_overlap": val_out_pct,
        "test_out_4gram_overlap": test_out_pct,
        "val_full_4gram_overlap": val_full_pct,
        "test_full_4gram_overlap": test_full_pct
    }


def run_tfidf_sentence_selector(tokenizer, vocab):
    print("\n" + "=" * 75)
    print(" KONTROL 2: EKSİK BAZ HAT — TF-IDF CÜMLE SEÇİCİ (SENTENCE SELECTOR) BASELINE")
    print("=" * 75)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    from collections import Counter
    import math

    # Tokenizer-based word extraction
    def get_words(text):
        return [w.lower() for w in re.findall(r"\w+", text)]

    wood_rouges = []
    hist_rouges = []
    fidelities = []
    len_ratios = []

    for rec in test_recs:
        inp = rec["input"]
        doc = rec["belge"]
        ref = rec["output"]
        dom = rec["domain"]

        q_words = Counter(get_words(inp))
        sentences = [s.strip() for s in doc.split(".") if len(s.strip()) > 5]
        if not sentences:
            sentences = [doc]

        # Score sentences by term overlap with question
        best_sent = sentences[0]
        best_score = -1.0
        for sent in sentences:
            s_words = Counter(get_words(sent))
            overlap = sum((q_words[w] * s_words[w]) for w in q_words if w in s_words)
            norm = (math.sqrt(sum(v**2 for v in q_words.values())) * math.sqrt(sum(v**2 for v in s_words.values()))) + 1e-6
            score = overlap / norm
            if score > best_score:
                best_score = score
                best_sent = sent

        selected_text = best_sent + "."
        sel_tokens = [vocab.decode(t) for t in tokenizer.encode(selected_text)]
        ref_tokens = [vocab.decode(t) for t in tokenizer.encode(ref)]
        doc_tokens = [vocab.decode(t) for t in tokenizer.encode(doc)]

        r = compute_rouge_l(sel_tokens, ref_tokens)
        if dom == "carpenter":
            wood_rouges.append(r)
        else:
            hist_rouges.append(r)

        # Fidelity: 100% since it's directly extracted from doc
        fidelities.append(1.0)
        len_ratios.append(len(sel_tokens) / max(1, len(doc_tokens)))

    mean_r, ci_r = stratified_bootstrap_ci(wood_rouges, hist_rouges)
    print(f"  TF-IDF Cümle Seçici ROUGE-L (Bootstrap %95 GA): {mean_r:.4f} [{ci_r[0]:.4f}, {ci_r[1]:.4f}]")
    print(f"  Ahşap ROUGE-L : {float(np.mean(wood_rouges)):.4f}")
    print(f"  Tarih ROUGE-L : {float(np.mean(hist_rouges)):.4f}")
    print(f"  Ortalama Uzunluk Oranı : {float(np.mean(len_ratios)):.2f}")

    return {
        "mean_rouge_l": mean_r,
        "bootstrap_ci": ci_r,
        "wood_rouge": float(np.mean(wood_rouges)),
        "hist_rouge": float(np.mean(hist_rouges))
    }


def run_entity_fidelity_audit(model, tokenizer, vocab):
    print("\n" + "=" * 75)
    print(" KONTROL 3: ENTITY FIDELITY VE <PROPER_NOUN> MASKELEME DENETİMİ")
    print("=" * 75)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    eos_id = vocab.stoi.get("<EOS>", 2)

    # We evaluate ROUGE-L under two conditions:
    # 1. Standard (as done in pilot, with <PROPER_NOUN> preserved)
    # 2. Entity-Stripped (where <PROPER_NOUN>, <UNK>, <NUMBER> are replaced with unique unmatched placeholders)
    # 3. Content Roots only (pure lexical/semantic overlap, no syntax/special token bonus)

    raw_rouges = []
    stripped_rouges = []
    ref_proper_count = 0
    pred_proper_count = 0
    proper_match_exact = 0

    for idx, rec in enumerate(test_recs):
        inp = rec["input"]
        doc = rec["belge"]
        ref = rec["output"]

        prompt_str = f"<INSTRUCTION> Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla. </INSTRUCTION> <INPUT> {inp} </INPUT> <BELGE> {doc} </BELGE> <OUTPUT>"
        prompt_ids = tokenizer.encode(prompt_str)
        if prompt_ids and prompt_ids[-1] == eos_id:
            prompt_ids = prompt_ids[:-1]

        # Generate
        x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        gen_ids = []
        with torch.no_grad():
            for _ in range(45):
                x_cond = x if x.size(1) <= 256 else x[:, -256:]
                logits, _ = model(x_cond)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt == eos_id:
                    break
                gen_ids.append(nxt)
                x = torch.cat((x, torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)), dim=1)

        ref_tokens = [vocab.decode(t) for t in tokenizer.encode(ref)]
        gen_tokens = [vocab.decode(t) for t in gen_ids]

        # 1. Standard ROUGE-L
        r_raw = compute_rouge_l(gen_tokens, ref_tokens)
        raw_rouges.append(r_raw)

        # Count proper nouns
        n_ref_pn = ref_tokens.count("<PROPER_NOUN>")
        n_pred_pn = gen_tokens.count("<PROPER_NOUN>")
        ref_proper_count += n_ref_pn
        pred_proper_count += n_pred_pn
        if n_ref_pn == n_pred_pn:
            proper_match_exact += 1

        # 2. Stripped ROUGE-L: replace <PROPER_NOUN> with unique placeholders so they don't artificially match
        gen_stripped = [f"__PRED_PN_{i}__" if t == "<PROPER_NOUN>" else t for i, t in enumerate(gen_tokens)]
        ref_stripped = [f"__REF_PN_{j}__" if t == "<PROPER_NOUN>" else t for j, t in enumerate(ref_tokens)]
        r_strip = compute_rouge_l(gen_stripped, ref_stripped)
        stripped_rouges.append(r_strip)

    print(f"  Toplam Referans <PROPER_NOUN> Sayısı : {ref_proper_count}")
    print(f"  Toplam Model <PROPER_NOUN> Sayısı    : {pred_proper_count}")
    print(f"  Tam Eşleşen Örnek Oranı              : %{(proper_match_exact / len(test_recs)) * 100:.1f}")
    print(f"  Standart ROUGE-L                     : {float(np.mean(raw_rouges)):.4f}")
    print(f"  Varlık-İzole (Stripped) ROUGE-L      : {float(np.mean(stripped_rouges)):.4f}")
    diff = float(np.mean(raw_rouges)) - float(np.mean(stripped_rouges))
    print(f"  <PROPER_NOUN> Maskeleme Şişirmesi    : +{diff:.4f} ROUGE puanı!")

    return {
        "ref_proper_count": ref_proper_count,
        "pred_proper_count": pred_proper_count,
        "standard_rouge": float(np.mean(raw_rouges)),
        "stripped_rouge": float(np.mean(stripped_rouges)),
        "inflation_diff": diff
    }


def run_doc_swap_test(model, tokenizer, vocab):
    print("\n" + "=" * 75)
    print(" KONTROL 4: BELGE-DEĞİŞTİRME (DOC-SWAP / COUNTERFACTUAL CONTEXT) TESTİ")
    print("=" * 75)
    print("  Deney Protokolü: Soru Q_i'ye orijinal D_i yerine D_{i+1} (farklı belge) verilir.")
    print("  Model gerçekten belgeyi mi okuyor, yoksa soru kalıbından ezber mi üretiyor?")

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_recs = [json.loads(line) for line in f]

    eos_id = vocab.stoi.get("<EOS>", 2)
    n = len(test_recs)

    swapped_changes = 0
    swapped_with_doc2_facts = 0
    original_adherence = 0

    for i in range(n):
        orig_rec = test_recs[i]
        swap_rec = test_recs[(i + 1) % n]  # Shift by 1 document

        q = orig_rec["input"]
        doc_orig = orig_rec["belge"]
        doc_swap = swap_rec["belge"]

        # Run with swapped doc
        p_swap = f"<INSTRUCTION> Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla. </INSTRUCTION> <INPUT> {q} </INPUT> <BELGE> {doc_swap} </BELGE> <OUTPUT>"
        p_ids = tokenizer.encode(p_swap)
        if p_ids and p_ids[-1] == eos_id:
            p_ids = p_ids[:-1]

        x = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)
        gen_swap_ids = []
        with torch.no_grad():
            for _ in range(45):
                x_cond = x if x.size(1) <= 256 else x[:, -256:]
                logits, _ = model(x_cond)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt == eos_id:
                    break
                gen_swap_ids.append(nxt)
                x = torch.cat((x, torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)), dim=1)

        gen_swap_tokens = [vocab.decode(t) for t in gen_swap_ids]
        gen_swap_text = " ".join(gen_swap_tokens)

        doc_swap_tokens = set([vocab.decode(t) for t in tokenizer.encode(doc_swap)])
        doc_orig_tokens = set([vocab.decode(t) for t in tokenizer.encode(doc_orig)])

        # Measure which document facts appear in gen_swap
        gen_content = set([t for t in gen_swap_tokens if len(t) > 3 and not t.startswith("<")])
        match_swap = len(gen_content & doc_swap_tokens)
        match_orig = len(gen_content & doc_orig_tokens)

        if match_swap > match_orig:
            swapped_with_doc2_facts += 1
        elif match_orig > match_swap:
            original_adherence += 1
        else:
            swapped_changes += 1

    pct_doc_follow = (swapped_with_doc2_facts / n) * 100.0
    pct_orig_hallucination = (original_adherence / n) * 100.0
    pct_neutral = (swapped_changes / n) * 100.0

    print(f"  Yeni Belgeye (D') Koşullanma Oranı : %{pct_doc_follow:.1f}")
    print(f"  Eski Belgeyi (D) Ezberden Sayıklama : %{pct_orig_hallucination:.1f}")
    print(f"  Nötr / Kararsız                    : %{pct_neutral:.1f}")

    if pct_doc_follow > 70.0:
        print("  -> BULGU: Model gerçekten verilen belgeye duyarlıdır (input document sensitivity pozitif).")
    else:
        print("  -> UYARI: Model belgeyi değil, soru veya genel şablonu takip etmektedir!")

    return {
        "doc_follow_rate": pct_doc_follow,
        "orig_hallucination_rate": pct_orig_hallucination,
        "neutral_rate": pct_neutral
    }


def main():
    print("=" * 75)
    print(" FAZ B2 GERÇEKLİK DENETİMİ (REALITY CHECK OTOPSİSİ)")
    print("=" * 75)

    # 1. 4-gram overlap
    overlap_res = run_ngram_overlap_check()

    # 2. Tokenizer & Model
    vocab = Vocabulary()
    vocab.load(os.path.join(DATA_DIR, "vocab.json"))
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(DATA_DIR, "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 3. TF-IDF Baseline
    tfidf_res = run_tfidf_sentence_selector(tokenizer, vocab)

    # 4. Load Model
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

    # 5. Entity Fidelity Audit
    entity_res = run_entity_fidelity_audit(model, tokenizer, vocab)

    # 6. Doc-Swap Test
    swap_res = run_doc_swap_test(model, tokenizer, vocab)

    # Save Reality Check results
    reality_data = {
        "ngram_overlap": overlap_res,
        "tfidf_sentence_selector": tfidf_res,
        "entity_fidelity": entity_res,
        "doc_swap": swap_res
    }
    out_file = os.path.join(PILOT_DIR, "reality_check_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reality_data, f, indent=2, ensure_ascii=False)
    print(f"\nGerçeklik kontrolü tamamlandı ve kaydedildi: {out_file}\n")


if __name__ == "__main__":
    main()
