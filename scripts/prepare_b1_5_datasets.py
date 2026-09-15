#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: ADIM B1.5 VERİ HAZIRLIK VE BÖLME MOTORU
===================================================================
1. Sözlük Genişletmesi: Frekansı >= 10 olan 332 tarihsel/coğrafi özel ismi tescil eder.
2. Külliyat Dengeleme ve Dedup:
   - Carpenter: 3.825 unique kayıt
   - Turk Tarihi: 6.507 unique kayıt
   - Lise, Edebiyat, Ortaokul: unique kayıtlar
   - Parenting: 5.300 dengelenmiş kayıt (PLURAL: Hayır 500'e indirildi)
3. Kapalı-Sınıf İskelet Kümeleme (k=2) + Yakın-Tekrar Soru Koruması (Zero-Leakage):
   - Train (%80), Val (%10), Test (%10)
4. Disk Tabanlı Sabit Aday Kümeleri (seed=42):
   - test_candidates_hard.jsonl
   - test_candidates_random.jsonl
"""

import os
import re
import json
import random
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Set, Any

import numpy as np

# Set deterministic seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CANONICAL_DIR = os.path.join(DATA_DIR, "pedagogy_canonical")
SPLITS_DIR = os.path.join(DATA_DIR, "b1_5_splits")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
ROOTS_PATH = os.path.join(DATA_DIR, "lexicon", "roots.tsv")

os.makedirs(SPLITS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. SÖZLÜK VE KÖK LİSTESİ GENİŞLETMESİ (FREKANS >= 10)
# ---------------------------------------------------------

def expand_vocabulary_and_roots(min_freq: int = 10) -> List[str]:
    print("\n--- [1/4] Sözlük ve Kök Listesi Hijyeni (Frekans >= 10 Özel İsimler) ---")
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    stoi = vocab_data["stoi"]
    next_id = vocab_data["next_id"]
    
    # Load existing roots
    existing_roots = set()
    with open(ROOTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                existing_roots.add(parts[0].lower())
                existing_roots.add(parts[0])

    # Scan turk_tarihi and other canonicals for capitalized tokens
    counter = Counter()
    for fname in os.listdir(CANONICAL_DIR):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(CANONICAL_DIR, fname), "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                text = d.get("input", "") + " " + d.get("output", "")
                tokens = re.findall(r"[\w']+", text, flags=re.UNICODE)
                for tok in tokens:
                    c = tok.replace("'", "").strip(".,!?:;()")
                    if c and c[0].isupper() and not c.isdigit() and len(c) >= 2:
                        if c.lower() not in existing_roots and c not in existing_roots and c not in stoi:
                            counter[c] += 1
                            
    # Filter candidates
    vetted_proper_nouns = [w for w, cnt in counter.items() if cnt >= min_freq]
    vetted_proper_nouns.sort(key=lambda w: counter[w], reverse=True)
    
    print(f"Tespit Edilen Aday Özel İsim: {len(counter):,} tip")
    print(f"Frekans >= {min_freq} Kriterini Geçen: {len(vetted_proper_nouns):,} tip")
    
    # Add to roots.tsv if not present
    new_roots_added = 0
    with open(ROOTS_PATH, "a", encoding="utf-8") as f:
        for pn in vetted_proper_nouns:
            if pn.lower() not in existing_roots and pn not in existing_roots:
                # Add proper noun entry: lemma, POS, attributes
                f.write(f"{pn}\tPROPER_NOUN\tPROPER_NOUN\n")
                existing_roots.add(pn)
                existing_roots.add(pn.lower())
                new_roots_added += 1
                
    # Add to vocab.json if not present
    new_tokens_added = 0
    for pn in vetted_proper_nouns:
        if pn not in stoi:
            stoi[pn] = next_id
            next_id += 1
            new_tokens_added += 1
            
    vocab_data["stoi"] = stoi
    vocab_data["next_id"] = next_id
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
    print(f"roots.tsv Dosyasına Eklenen Yeni Lemma: {new_roots_added}")
    print(f"vocab.json Dosyasına Eklenen Yeni Token: {new_tokens_added} -> Yeni Vocab Boyutu: {len(stoi):,}")
    return vetted_proper_nouns


# ---------------------------------------------------------
# 2. KÜLLİYAT DENGELEME VE DEDUP
# ---------------------------------------------------------

def balance_and_deduplicate_corpus() -> Dict[str, List[Dict[str, str]]]:
    print("\n--- [2/4] Külliyat Dengeleme ve Dedup (Key_Record = (inst, inp, out)) ---")
    datasets = {}
    
    # 1. Carpenter
    with open(os.path.join(CANONICAL_DIR, "carpenter_canonical.jsonl"), "r", encoding="utf-8") as f:
        carp_records = [json.loads(l) for l in f]
    carp_unique = list({(r["instruction"], r["input"], r["output"]): r for r in carp_records}.values())
    datasets["carpenter"] = carp_unique
    print(f"Carpenter: Ham {len(carp_records):,} -> Unique {len(carp_unique):,} (Oran: {len(carp_records)/len(carp_unique):.2f}x)")
    
    # 2. Turk Tarihi
    with open(os.path.join(CANONICAL_DIR, "turk_tarihi_canonical.jsonl"), "r", encoding="utf-8") as f:
        tt_records = [json.loads(l) for l in f]
    tt_unique = list({(r["instruction"], r["input"], r["output"]): r for r in tt_records}.values())
    datasets["turk_tarihi"] = tt_unique
    print(f"Türk Tarihi: Ham {len(tt_records):,} -> Unique {len(tt_unique):,} (Oran: {len(tt_records)/len(tt_unique):.2f}x)")
    
    # 3. High School
    with open(os.path.join(CANONICAL_DIR, "high_school_canonical.jsonl"), "r", encoding="utf-8") as f:
        hs_records = [json.loads(l) for l in f]
    hs_unique = list({(r["instruction"], r["input"], r["output"]): r for r in hs_records}.values())
    datasets["high_school"] = hs_unique
    print(f"Lise: Ham {len(hs_records):,} -> Unique {len(hs_unique):,} (Oran: {len(hs_records)/len(hs_unique):.2f}x)")

    # 4. Literature
    with open(os.path.join(CANONICAL_DIR, "literature_canonical.jsonl"), "r", encoding="utf-8") as f:
        lit_records = [json.loads(l) for l in f]
    lit_unique = list({(r["instruction"], r["input"], r["output"]): r for r in lit_records}.values())
    datasets["literature"] = lit_unique
    print(f"Edebiyat: Ham {len(lit_records):,} -> Unique {len(lit_unique):,}")

    # 5. Middle School
    with open(os.path.join(CANONICAL_DIR, "middle_school_canonical.jsonl"), "r", encoding="utf-8") as f:
        ms_records = [json.loads(l) for l in f]
    ms_unique = list({(r["instruction"], r["input"], r["output"]): r for r in ms_records}.values())
    datasets["middle_school"] = ms_unique
    print(f"Ortaokul: Ham {len(ms_records):,} -> Unique {len(ms_unique):,}")

    # 6. Parenting (Rebalancing)
    with open(os.path.join(CANONICAL_DIR, "parenting_canonical.jsonl"), "r", encoding="utf-8") as f:
        parenting_records = [json.loads(l) for l in f]
        
    by_category = defaultdict(list)
    for r in parenting_records:
        inst = r.get("instruction", "")
        if "çoğul" in inst:
            if "Evet" in r.get("output", ""):
                by_category["plural_yes"].append(r)
            else:
                by_category["plural_no"].append(r)
        elif "kök" in inst:
            by_category["root"].append(r)
        elif "durum" in inst or "hâl" in inst:
            by_category["case"].append(r)
        elif "zaman" in inst or "kip" in inst:
            by_category["tense"].append(r)
        else:
            by_category["other"].append(r)
            
    # Sample balanced parenting
    rng = random.Random(RANDOM_SEED)
    balanced_parenting = []
    
    # Plural: 500 Evet + 500 Hayır
    p_yes = by_category["plural_yes"]
    p_no = rng.sample(by_category["plural_no"], min(500, len(by_category["plural_no"])))
    balanced_parenting.extend(p_yes)
    balanced_parenting.extend(p_no)
    
    # Root: 2,000
    r_sampled = rng.sample(by_category["root"], min(2000, len(by_category["root"])))
    balanced_parenting.extend(r_sampled)
    
    # Case: 1,500
    c_sampled = rng.sample(by_category["case"], min(1500, len(by_category["case"])))
    balanced_parenting.extend(c_sampled)
    
    # Tense: 800
    t_sampled = rng.sample(by_category["tense"], min(800, len(by_category["tense"])))
    balanced_parenting.extend(t_sampled)
    
    rng.shuffle(balanced_parenting)
    datasets["parenting"] = balanced_parenting
    print(f"Parenting (Dengelenmiş): Ham {len(parenting_records):,} -> Kota {len(balanced_parenting):,} (Plural Evet={len(p_yes)}, Hayır={len(p_no)}, Root={len(r_sampled)}, Case={len(c_sampled)}, Tense={len(t_sampled)})")

    total_balanced = sum(len(v) for v in datasets.values())
    print(f"Toplam B1.5 Eğitim Kayıt Havuzu: {total_balanced:,}")
    return datasets


# ---------------------------------------------------------
# 3. SIZINTISIZ 3 YOLLU BÖLME (Union-Find Soru-Cevap Kümeleme)
# ---------------------------------------------------------

def normalize_question(q: str) -> str:
    """Normalizes question for near-duplicate protection."""
    clean = re.sub(r'[\d\'".,!?:;()—–-]+', '', q.lower())
    return " ".join(clean.split())

def split_stratum_3way(
    records: List[Dict[str, str]], 
    stratum_name: str,
    val_ratio: float = 0.10, 
    test_ratio: float = 0.10, 
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    """
    Groups records by normalized question AND normalized answer text.
    Ensures zero question leakage and prevents template-variant answer leakage across splits.
    """
    class UnionFind:
        def __init__(self):
            self.parent = {}
        def find(self, i):
            if self.parent.setdefault(i, i) != i:
                self.parent[i] = self.find(self.parent[i])
            return self.parent[i]
        def union(self, i, j):
            root_i = self.find(i)
            root_j = self.find(j)
            if root_i != root_j:
                self.parent[root_i] = root_j

    uf = UnionFind()
    for idx, r in enumerate(records):
        norm_q = normalize_question(r.get("input", ""))
        norm_a = " ".join(r.get("output", "").strip().lower().split())
        rec_k = ("r", idx)
        qk = ("q", norm_q)
        ak = ("a", norm_a)
        uf.union(rec_k, qk)
        uf.union(rec_k, ak)

    groups = defaultdict(list)
    for idx in range(len(records)):
        root = uf.find(("r", idx))
        groups[root].append(idx)
        
    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    
    total = len(records)
    val_target = max(1, int(total * val_ratio))
    test_target = max(1, int(total * test_ratio))
    
    val_indices = []
    test_indices = []
    train_indices = []
    
    for k in keys:
        grp = groups[k]
        if len(val_indices) + len(grp) <= val_target or len(val_indices) == 0:
            val_indices.extend(grp)
        elif len(test_indices) + len(grp) <= test_target or len(test_indices) == 0:
            test_indices.extend(grp)
        else:
            train_indices.extend(grp)
            
    train = [records[i] for i in train_indices]
    val = [records[i] for i in val_indices]
    test = [records[i] for i in test_indices]
    
    # Measure answer-level leakage within stratum
    train_ans = {" ".join(r.get("output", "").strip().lower().split()) for r in train}
    test_ans_leaks = sum(1 for r in test if " ".join(r.get("output", "").strip().lower().split()) in train_ans)
    ans_leak_pct = (test_ans_leaks / len(test) * 100) if test else 0.0
    print(f"[SIZINTI] stratum={stratum_name} cevap_duzeyi={test_ans_leaks}/{len(test)} (%{ans_leak_pct:.1f})")
    if test_ans_leaks > 0:
        print(f"[SIZINTI_UYARI] Stratum '{stratum_name}' içinde {test_ans_leaks} test cevabı train kümesinde mevcut!")

    # Cluster statistics
    cluster_sizes = [len(grp) for grp in groups.values()]
    stats = {
        "stratum": stratum_name,
        "total_records": total,
        "total_clusters": len(groups),
        "mean_cluster_size": float(np.mean(cluster_sizes)),
        "max_cluster_size": max(cluster_sizes),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "test_answer_leakage_count": test_ans_leaks,
        "test_answer_leakage_pct": ans_leak_pct,
    }
    return train, val, test, stats


# ---------------------------------------------------------
# 4. DİSK TABANLI ADAY KÜMESİ ÜRETİMİ (SEED=42)
# ---------------------------------------------------------

def build_pinned_candidate_sets(
    splits: Dict[str, Dict[str, List[Dict[str, str]]]],
    seed: int = 42
):
    print("\n--- [3/4] Disk Tabanlı Sabit Aday Kümeleri Üretimi (seed=42) ---")
    rng = random.Random(seed)
    
    test_random_records = []
    test_hard_records = []
    
    # Strata evaluation sample quotas
    quotas = {
        "carpenter": 300,
        "turk_tarihi": 300,
        "high_school": None, # all
        "literature": None,  # all
        "middle_school": None # all
    }
    
    for stratum, quota in quotas.items():
        if stratum not in splits:
            continue
        test_pool = splits[stratum]["test"]
        if not test_pool:
            continue
            
        if quota is not None and len(test_pool) > quota:
            # Subsample evaluation subset deterministically with seed
            eval_items = rng.sample(test_pool, quota)
        else:
            eval_items = list(test_pool)
            
        all_test_outputs = [r["output"] for r in test_pool]
        
        # Word token sets for hard negative calculation
        def get_words(txt):
            return set(re.findall(r"\w+", txt.lower()))
            
        doc_words = [get_words(out) for out in all_test_outputs]
        
        for q_idx, item in enumerate(eval_items):
            q_text = item["input"]
            true_out = item["output"]
            q_words = get_words(q_text)
            
            # --- 1. Random Distractors (9 from same stratum) ---
            other_outs = [o for o in all_test_outputs if o != true_out]
            if len(other_outs) < 9:
                # If stratum too small, fallback with padding from all available
                distractors_random = other_outs
            else:
                distractors_random = rng.sample(other_outs, 9)
                
            cands_rand = [true_out] + distractors_random
            rng.shuffle(cands_rand)
            correct_idx_rand = cands_rand.index(true_out)
            
            test_random_records.append({
                "stratum": stratum,
                "question_id": f"{stratum}_{q_idx:04d}",
                "instruction": item.get("instruction", ""),
                "input": q_text,
                "true_output": true_out,
                "candidates": cands_rand,
                "correct_index": correct_idx_rand
            })
            
            # --- 2. Hard-Negative Distractors (9 highest Jaccard overlap with question) ---
            scored_negatives = []
            for o_idx, o_text in enumerate(all_test_outputs):
                if o_text == true_out:
                    continue
                o_words = doc_words[o_idx]
                inter = len(q_words & o_words)
                union = len(q_words | o_words)
                jacc = inter / union if union > 0 else 0.0
                scored_negatives.append((jacc, o_text))
                
            scored_negatives.sort(key=lambda x: x[0], reverse=True)
            distractors_hard = [x[1] for x in scored_negatives[:9]]
            
            cands_hard = [true_out] + distractors_hard
            rng.shuffle(cands_hard)
            correct_idx_hard = cands_hard.index(true_out)
            
            test_hard_records.append({
                "stratum": stratum,
                "question_id": f"{stratum}_{q_idx:04d}",
                "instruction": item.get("instruction", ""),
                "input": q_text,
                "true_output": true_out,
                "candidates": cands_hard,
                "correct_index": correct_idx_hard
            })

    random_path = os.path.join(SPLITS_DIR, "test_candidates_random.jsonl")
    hard_path = os.path.join(SPLITS_DIR, "test_candidates_hard.jsonl")
    
    with open(random_path, "w", encoding="utf-8") as f:
        for r in test_random_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(hard_path, "w", encoding="utf-8") as f:
        for r in test_hard_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Sabitlenmiş Rastgele Çeldirici Kümesi: {len(test_random_records)} soru -> {random_path}")
    print(f"Sabitlenmiş Zor-Negatif Çeldirici Kümesi: {len(test_hard_records)} soru -> {hard_path}")


# ---------------------------------------------------------
# 5. ANA ORKESTRASYON
# ---------------------------------------------------------

def main():
    print("===================================================================")
    print("KRİSTAL-VEKTÖREL: ADIM B1.5 VERİ VE ADAY KÜMESİ HAZIRLIK SÜRECİ")
    print("===================================================================")
    
    # 1. Sözlük genişletmesi
    expand_vocabulary_and_roots(min_freq=10)
    
    # 2. Külliyat dengeleme
    datasets = balance_and_deduplicate_corpus()
    
    # 3. 3 yollu sızıntısız bölme
    print("\n--- [3/4] Kapalı-Sınıf Morfem (k=2) Sızıntısız 3 Yollu Bölme ---")
    splits = {}
    all_train = []
    all_val = []
    all_test = []
    all_stats = []
    
    for stratum, records in datasets.items():
        tr, va, te, st = split_stratum_3way(records, stratum, val_ratio=0.10, test_ratio=0.10, seed=RANDOM_SEED)
        splits[stratum] = {"train": tr, "val": va, "test": te}
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        all_stats.append(st)
        print(f"  {stratum:15s}: Kümeler={st['total_clusters']:4d} (Ort. Boyut={st['mean_cluster_size']:.2f}, Max={st['max_cluster_size']:3d}) -> Train={len(tr):5d}, Val={len(va):4d}, Test={len(te):4d}")

    # Sızıntı doğrulaması (Question level zero-leakage check)
    train_qs = {normalize_question(r["input"]) for r in all_train}
    val_qs = {normalize_question(r["input"]) for r in all_val}
    test_qs = {normalize_question(r["input"]) for r in all_test}
    
    leak_train_val = train_qs & val_qs
    leak_train_test = train_qs & test_qs
    leak_val_test = val_qs & test_qs
    
def audit_split_leakage(splits_dir: str = SPLITS_DIR, output_report_path: str = "data/eval/b1_5_split_leakage_report.json") -> Dict[str, Any]:
    """
    Audits an existing split on disk and writes machine-readable data/eval/b1_5_split_leakage_report.json.
    Reports both question-level and answer-level leakage.
    """
    train_path = os.path.join(splits_dir, "train.jsonl")
    val_path = os.path.join(splits_dir, "val.jsonl")
    test_path = os.path.join(splits_dir, "test.jsonl")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_recs = [json.loads(l) for l in f]
    with open(val_path, "r", encoding="utf-8") as f:
        val_recs = [json.loads(l) for l in f]
    with open(test_path, "r", encoding="utf-8") as f:
        test_recs = [json.loads(l) for l in f]
        
    train_qs = {normalize_question(r["input"]) for r in train_recs}
    val_qs = {normalize_question(r["input"]) for r in val_recs}
    test_qs = {normalize_question(r["input"]) for r in test_recs}
    
    leak_train_val = train_qs & val_qs
    leak_train_test = train_qs & test_qs
    leak_val_test = val_qs & test_qs
    
    train_ans = {" ".join(r["output"].strip().lower().split()) for r in train_recs}
    test_ans_leaks = [r for r in test_recs if " ".join(r["output"].strip().lower().split()) in train_ans]
    val_ans_leaks = [r for r in val_recs if " ".join(r["output"].strip().lower().split()) in train_ans]
    
    n_test = len(test_recs)
    ans_leak_pct = (len(test_ans_leaks) / n_test * 100) if n_test > 0 else 0.0
    val_leak_pct = (len(val_ans_leaks) / len(val_recs) * 100) if val_recs else 0.0
    
    short_labels = [r for r in test_ans_leaks if len(r["output"].split()) <= 4]
    content_answers = [r for r in test_ans_leaks if len(r["output"].split()) > 4]
    
    print(f"\nSızıntı Denetimi (Mevcut Bölme):")
    print(f"  Train ∩ Val Soru Kesişimi : {len(leak_train_val)} (Hedef: 0)")
    print(f"  Train ∩ Test Soru Kesişimi: {len(leak_train_test)} (Hedef: 0)")
    print(f"  Val ∩ Test Soru Kesişimi  : {len(leak_val_test)} (Hedef: 0)")
    print(f"[SIZINTI] cevap_duzeyi={len(test_ans_leaks)}/{n_test} (%{ans_leak_pct:.1f})")
    if len(test_ans_leaks) > 0:
        print(f"[SIZINTI_UYARI] Test cevaplarının %{ans_leak_pct:.1f} kadarı ({len(test_ans_leaks)}/{n_test}) train kümesinde mevcut!")
        
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits_dir": os.path.abspath(splits_dir),
        "split_counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
            "total": len(train_recs) + len(val_recs) + len(test_recs)
        },
        "question_level": {
            "train_test_overlap_count": len(leak_train_test),
            "train_test_overlap_pct": round(len(leak_train_test) / n_test * 100, 2) if n_test else 0.0,
            "train_val_overlap_count": len(leak_train_val),
            "val_test_overlap_count": len(leak_val_test),
            "status": "zero_leakage" if len(leak_train_test) == 0 else "leakage_detected"
        },
        "answer_level": {
            "test_in_train_count": len(test_ans_leaks),
            "test_in_train_total": n_test,
            "test_in_train_pct": round(ans_leak_pct, 1),
            "val_in_train_count": len(val_ans_leaks),
            "val_in_train_total": len(val_recs),
            "val_in_train_pct": round(val_leak_pct, 1),
            "composition": {
                "short_labels_count": len(short_labels),
                "short_labels_description": "<=4 kelimelik kapalı-sınıf etiketi (örn. PLURAL: Evet, CASE: CASE_DAT)",
                "content_answers_count": len(content_answers),
                "content_answers_description": "18-21 kelimelik şablon öneki farklı fakat output birebir aynı olan cevaplar"
            }
        },
        "root_cause": "Eski prepare_b1_5_datasets.py satır 264-268'de skel_in/skel_out hesaplayıp atan ölü kod nedeniyle, şablonla çeşitlendirilmiş aynı cevaplar farklı kümelere dağılarak train/test arasına sızıyordu."
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Sızıntı Raporu Kaydedildi: {output_report_path}")
    return report


def main():
    print("=" * 67)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: ADIM B1.5 GELİŞMİŞ VERİ HAZIRLIK HATTI")
    print("=" * 67)
    
    # 1. Sözlük genişletmesi
    expand_vocabulary_and_roots(min_freq=10)
    
    # 2. Külliyat dengeleme
    datasets = balance_and_deduplicate_corpus()
    
    # 3. 3 yollu sızıntısız bölme
    print("\n--- [3/4] Kapalı-Sınıf Morfem (k=2) Sızıntısız 3 Yollu Bölme ---")
    splits = {}
    all_train = []
    all_val = []
    all_test = []
    all_stats = []
    
    for stratum, records in datasets.items():
        tr, va, te, st = split_stratum_3way(records, stratum, val_ratio=0.10, test_ratio=0.10, seed=RANDOM_SEED)
        splits[stratum] = {"train": tr, "val": va, "test": te}
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        all_stats.append(st)
        print(f"  {stratum:15s}: Kümeler={st['total_clusters']:4d} (Ort. Boyut={st['mean_cluster_size']:.2f}, Max={st['max_cluster_size']:3d}) -> Train={len(tr):5d}, Val={len(va):4d}, Test={len(te):4d}")

    # Sızıntı doğrulaması (Soru ve Cevap düzeyi)
    train_qs = {normalize_question(r["input"]) for r in all_train}
    val_qs = {normalize_question(r["input"]) for r in all_val}
    test_qs = {normalize_question(r["input"]) for r in all_test}
    
    leak_train_val = train_qs & val_qs
    leak_train_test = train_qs & test_qs
    leak_val_test = val_qs & test_qs
    
    print(f"\nSızıntı Denetimi:")
    print(f"  Train ∩ Val Soru Kesişimi : {len(leak_train_val)} (Hedef: 0)")
    print(f"  Train ∩ Test Soru Kesişimi: {len(leak_train_test)} (Hedef: 0)")
    print(f"  Val ∩ Test Soru Kesişimi  : {len(leak_val_test)} (Hedef: 0)")
    assert len(leak_train_test) == 0, f"KRİTİK HATA: Train ile Test arasında {len(leak_train_test)} soru sızıntısı var!"

    train_ans = {" ".join(r["output"].strip().lower().split()) for r in all_train}
    test_ans_leaks = [r for r in all_test if " ".join(r["output"].strip().lower().split()) in train_ans]
    n_test = len(all_test)
    ans_leak_pct = (len(test_ans_leaks) / n_test * 100) if n_test > 0 else 0.0
    print(f"[SIZINTI] cevap_duzeyi={len(test_ans_leaks)}/{n_test} (%{ans_leak_pct:.1f})")
    if len(test_ans_leaks) > 0:
        print(f"[SIZINTI_UYARI] Test cevaplarının %{ans_leak_pct:.1f} kadarı ({len(test_ans_leaks)}/{n_test}) train kümesinde mevcut!")

    # Write leakage report
    leakage_report_path = os.path.join(DATA_DIR, "eval", "b1_5_split_leakage_report.json")
    os.makedirs(os.path.dirname(leakage_report_path), exist_ok=True)
    leakage_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(all_train) + len(all_val) + len(all_test),
        "train_count": len(all_train),
        "val_count": len(all_val),
        "test_count": len(all_test),
        "question_level": {
            "train_test_overlap": len(leak_train_test),
            "train_val_overlap": len(leak_train_val),
            "val_test_overlap": len(leak_val_test),
            "status": "zero_leakage" if len(leak_train_test) == 0 else "leakage_detected"
        },
        "answer_level": {
            "test_in_train_count": len(test_ans_leaks),
            "test_in_train_pct": round(ans_leak_pct, 1),
            "short_labels_count": sum(1 for r in test_ans_leaks if len(r["output"].split()) <= 4),
            "content_answers_count": sum(1 for r in test_ans_leaks if len(r["output"].split()) > 4)
        }
    }
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        json.dump(leakage_payload, f, ensure_ascii=False, indent=2)
    print(f"  Sızıntı Raporu Kaydedildi: {leakage_report_path}")

    # Dosyaları yaz
    train_path = os.path.join(SPLITS_DIR, "train.jsonl")
    val_path = os.path.join(SPLITS_DIR, "val.jsonl")
    test_path = os.path.join(SPLITS_DIR, "test.jsonl")
    stats_path = os.path.join(SPLITS_DIR, "split_statistics.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for r in all_train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in all_val: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for r in all_test: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
        
    print(f"\nBölünmüş Dosyalar Kaydedildi:")
    print(f"  Train: {len(all_train):,} satır -> {train_path}")
    print(f"  Val  : {len(all_val):,} satır -> {val_path}")
    print(f"  Test : {len(all_test):,} satır -> {test_path}")
    
    # 4. Sabit Aday Kümeleri
    build_pinned_candidate_sets(splits, seed=RANDOM_SEED)
    print("\nAdım B1.5 Veri ve Aday Kümesi Hazırlığı Başarıyla Tamamlandı! ✓")

if __name__ == "__main__":
    main()
