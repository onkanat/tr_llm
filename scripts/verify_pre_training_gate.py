#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: 5 AŞAMALI ÖN-EĞİTİM KAPI DENETİMİ (PRE-TRAINING GATE)
======================================================================
Bu betik; eğitim başlamadan önce veri setinin sızıntı, sentaks ve ezber
izolasyonunu 5 bağımsız kriterde denetler:

1. Kriter: Train ∩ Test 4-Gram Örtüşmesi <= %10.0
2. Kriter: Train ∩ Test Syntax Signature İskelet Örtüşmesi <= %10.0
3. Kriter: Pilot Belgeleri ∩ B1.5 Eğitim Külliyatı Metni = 0 (4-gram <= %5.0)
4. Kriter: Pilot Belgeleri ∩ Simülasyon Belleği = 0 (4-gram <= %5.0)
5. Kriter: Test Çekirdek Varlıkları ∩ B1.5 Çekirdek Varlıkları = 0

Tüm kriterler sağlanırsa exit code 0 döner; aksi halde eğitimi kilitler (exit code 1).
"""

import os
import sys
import json
import re
from collections import Counter
from typing import Set, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import Vocabulary, KristalTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REALISTIC_DIR = os.path.join(DATA_DIR, "realistic_rag")
B1_5_TRAIN = os.path.join(DATA_DIR, "b1_5_splits", "train.jsonl")
SIM_MEMORY = os.path.join(DATA_DIR, "simulasyon_bellek_export.jsonl")


def extract_ngrams(text: str, n: int = 4) -> Set[str]:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def get_syntax_signature(text: str, compiler: CrystalCompiler) -> str:
    """Extracts the grammatical affix skeleton of the text."""
    PUNCT_SET = {".", ",", "?", "!", "-", ":", ";", "(", ")"}
    tokens = re.findall(r"[\w\']+|[.,!?;:()\"—–-]", text, flags=re.UNICODE)
    tags = []
    for token in tokens:
        clean_word = token.strip(".,!?\"…—«»/()-;:")
        if not clean_word or clean_word.isdigit():
            continue
        compile_word = clean_word.replace("'", "").replace("’", "")
        res = compiler.compile(compile_word)
        if res.get("token_vector"):
            for mid in res["token_vector"]:
                if any(mid.startswith(p) for p in ("CASE_", "TENSE_", "POSS_", "COPULA_", "PART_", "GERUND_", "DERIV_", "VOICE_")) or mid in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"):
                    tags.append(mid)
    return " ".join(tags)


import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="5 Aşamalı Ön-Eğitim Kapı Denetimi")
    parser.add_argument("--json", type=str, default=os.path.join(REALISTIC_DIR, "pre_training_gate_result.json"), help="JSON çıktı dosya yolu")
    args = parser.parse_args()

    print("=================================================================")
    print("   KRİSTAL-VEKTÖREL: 5 AŞAMALI ÖN-EĞİTİM KAPI DENETİMİ (GATE)   ")
    print("=================================================================\n")

    lex = LexiconManager()
    roots_path = os.path.join(DATA_DIR, "lexicon", "roots.tsv")
    if os.path.exists(roots_path):
        lex.load_from_tsv(roots_path)
    else:
        raise FileNotFoundError(f"Leksikon kök dosyası bulunamadı: {roots_path}")
    graph = build_default_graph()
    compiler = CrystalCompiler(lex, graph)

    # 1. Veri Kümelerini Yükle
    train_file = os.path.join(REALISTIC_DIR, "train_natural_3000.jsonl")
    test_file = os.path.join(REALISTIC_DIR, "test_natural_150.jsonl")
    test_cf_file = os.path.join(REALISTIC_DIR, "test_cf_50.jsonl")

    with open(train_file, "r", encoding="utf-8") as f:
        train_records = [json.loads(l) for l in f]
    with open(test_file, "r", encoding="utf-8") as f:
        test_records = [json.loads(l) for l in f]
    with open(test_cf_file, "r", encoding="utf-8") as f:
        test_cf_records = [json.loads(l) for l in f]

    all_test_records = test_records + test_cf_records

    print(f"Eğitim Kayıtları (Train): {len(train_records):,} adet")
    print(f"Test Kayıtları (Test): {len(all_test_records):,} adet (150 Natural + 50 CF)")

    # -------------------------------------------------------------
    # KRİTER 1: Train ∩ Test 4-Gram Örtüşmesi (Hedef <= %10.0)
    # -------------------------------------------------------------
    print("\n--- [KRİTER 1] 4-Gram Sızıntı Denetimi ---")
    train_4grams = set()
    for r in train_records:
        train_4grams.update(extract_ngrams(r["output"]))

    test_total_4grams = 0
    test_shared_4grams = 0
    for r in all_test_records:
        ng = extract_ngrams(r["output"])
        test_total_4grams += len(ng)
        test_shared_4grams += len(ng & train_4grams)

    overlap_4gram = (test_shared_4grams / max(1, test_total_4grams)) * 100
    print(f"  Toplam Test 4-Gramı: {test_total_4grams:,}")
    print(f"  Train ile Paylaşılan 4-Gram: {test_shared_4grams:,}")
    print(f"  4-Gram Örtüşme Oranı: %{overlap_4gram:.2f} (Eşik: <= %10.0)")
    passed_1 = (overlap_4gram <= 10.0)
    print(f"  -> Durum: {'GEÇTİ [PASSED]' if passed_1 else 'KALDI [FAILED]'}")

    # -------------------------------------------------------------
    # KRİTER 2: Train ∩ Test Syntax Signature Örtüşmesi (Hedef <= %10.0)
    # -------------------------------------------------------------
    print("\n--- [KRİTER 2] Syntax Signature (İskelet) Örtüşme Denetimi ---")
    train_skeletons = set()
    for r in train_records:
        sk = get_syntax_signature(r["output"], compiler)
        if sk:
            train_skeletons.add(sk)

    test_shared_skeletons = 0
    for r in all_test_records:
        sk = get_syntax_signature(r["output"], compiler)
        if sk and sk in train_skeletons:
            test_shared_skeletons += 1

    overlap_syntax = (test_shared_skeletons / len(all_test_records)) * 100
    print(f"  Train Tekil Sentaktik İskelet Sayısı: {len(train_skeletons):,}")
    print(f"  Test Örneklerinde Paylaşılan İskelet Sayısı: {test_shared_skeletons} / {len(all_test_records)}")
    print(f"  Sentaktik Örtüşme Oranı: %{overlap_syntax:.2f} (Eşik: <= %10.0)")
    passed_2 = (overlap_syntax <= 10.0)
    print(f"  -> Durum: {'GEÇTİ [PASSED]' if passed_2 else 'KALDI [FAILED]'}")

    # -------------------------------------------------------------
    # KRİTER 3: Pilot Belgeleri ∩ B1.5 Eğitim Külliyatı Metni = 0
    # -------------------------------------------------------------
    print("\n--- [KRİTER 3] B1.5 Eğitim Külliyatı İzolasyon Denetimi ---")
    b1_5_4grams = set()
    b1_5_count = 0
    if os.path.exists(B1_5_TRAIN):
        with open(B1_5_TRAIN, "r", encoding="utf-8") as f:
            for l in f:
                b1_5_count += 1
                rec = json.loads(l)
                text = (rec.get("instruction", "") + " " + rec.get("input", "") + " " + rec.get("output", ""))
                b1_5_4grams.update(extract_ngrams(text))

    print(f"  B1.5 Külliyatı Satır Sayısı: {b1_5_count:,}")
    print(f"  B1.5 Tekil 4-Gram Sayısı: {len(b1_5_4grams):,}")

    pilot_docs_total_4grams = 0
    pilot_docs_b1_5_overlap = 0
    for r in train_records + all_test_records:
        ng = extract_ngrams(r.get("input", ""))
        pilot_docs_total_4grams += len(ng)
        pilot_docs_b1_5_overlap += len(ng & b1_5_4grams)

    overlap_b1_5 = (pilot_docs_b1_5_overlap / max(1, pilot_docs_total_4grams)) * 100
    print(f"  Pilot Belgeleri Toplam 4-Gram: {pilot_docs_total_4grams:,}")
    print(f"  B1.5 ile Ortak 4-Gram: {pilot_docs_b1_5_overlap:,}")
    print(f"  B1.5 Örtüşme Oranı: %{overlap_b1_5:.2f} (Eşik: <= %5.0)")
    passed_3 = (overlap_b1_5 <= 5.0)
    print(f"  -> Durum: {'GEÇTİ [PASSED]' if passed_3 else 'KALDI [FAILED]'}")

    # -------------------------------------------------------------
    # KRİTER 4: Pilot Belgeleri ∩ Simülasyon Belleği = 0
    # -------------------------------------------------------------
    print("\n--- [KRİTER 4] Simülasyon Bellek İzolasyon Denetimi ---")
    sim_4grams = set()
    sim_count = 0
    if os.path.exists(SIM_MEMORY):
        with open(SIM_MEMORY, "r", encoding="utf-8") as f:
            for l in f:
                sim_count += 1
                d = json.loads(l)
                payload = d.get("payload", {})
                txt = payload.get("crystal_tags", "") + " " + payload.get("yazar", "")
                sim_4grams.update(extract_ngrams(txt))

    print(f"  Simülasyon Bellek Kart Sayısı: {sim_count:,}")
    sim_overlap_count = 0
    for r in train_records + all_test_records:
        ng = extract_ngrams(r.get("input", ""))
        sim_overlap_count += len(ng & sim_4grams)

    overlap_sim = (sim_overlap_count / max(1, pilot_docs_total_4grams)) * 100
    print(f"  Simülasyon Belleği ile Ortak 4-Gram: {sim_overlap_count:,}")
    print(f"  Simülasyon Bellek Örtüşme Oranı: %{overlap_sim:.2f} (Eşik: <= %5.0)")
    passed_4 = (overlap_sim <= 5.0)
    print(f"  -> Durum: {'GEÇTİ [PASSED]' if passed_4 else 'KALDI [FAILED]'}")

    # -------------------------------------------------------------
    # KRİTER 5: Test Çekirdek Varlıkları ∩ B1.5 Çekirdek Varlıkları = 0
    # -------------------------------------------------------------
    print("\n--- [KRİTER 5] Test Çekirdek Varlık İzolasyon Denetimi ---")
    test_entities = set(r.get("entity", "") for r in all_test_records if r.get("entity"))
    print(f"  Test Kümesindeki Çekirdek Varlıklar ({len(test_entities)} adet):")
    print(f"  {sorted(list(test_entities))[:15]}...")

    # B1.5 külliyatında bu çekirdek varlıkların geçip geçmediğini denetle
    b1_5_entity_leaks = set()
    if os.path.exists(B1_5_TRAIN):
        with open(B1_5_TRAIN, "r", encoding="utf-8") as f:
            b1_5_raw_text = f.read().lower()
            for ent in test_entities:
                if ent.lower() in b1_5_raw_text:
                    b1_5_entity_leaks.add(ent)

    print(f"  B1.5 Eğitim Külliyatına Sızan Test Varlığı Sayısı: {len(b1_5_entity_leaks)}")
    if b1_5_entity_leaks:
        print(f"  Sızan Varlıklar: {b1_5_entity_leaks}")
    passed_5 = (len(b1_5_entity_leaks) == 0)
    print(f"  -> Durum: {'GEÇTİ [PASSED]' if passed_5 else 'KALDI [FAILED]'}")

    # -------------------------------------------------------------
    # NİHAİ KAPI HÜKMÜ
    # -------------------------------------------------------------
    print("\n=================================================================")
    all_passed = passed_1 and passed_2 and passed_3 and passed_4 and passed_5

    gate_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kriter_1_4gram_overlap": {
            "test_total_4grams": test_total_4grams,
            "test_shared_4grams": test_shared_4grams,
            "overlap_pct": round(overlap_4gram, 4),
            "threshold_pct": 10.0,
            "passed": passed_1
        },
        "kriter_2_syntax_signature_overlap": {
            "train_unique_skeletons": len(train_skeletons),
            "test_shared_skeletons": test_shared_skeletons,
            "test_total_records": len(all_test_records),
            "overlap_pct": round(overlap_syntax, 4),
            "threshold_pct": 10.0,
            "passed": passed_2
        },
        "kriter_3_b1_5_corpus_isolation": {
            "pilot_docs_total_4grams": pilot_docs_total_4grams,
            "pilot_docs_b1_5_overlap": pilot_docs_b1_5_overlap,
            "overlap_pct": round(overlap_b1_5, 4),
            "threshold_pct": 5.0,
            "passed": passed_3
        },
        "kriter_4_sim_memory_isolation": {
            "pilot_docs_total_4grams": pilot_docs_total_4grams,
            "sim_overlap_count": sim_overlap_count,
            "overlap_pct": round(overlap_sim, 4),
            "threshold_pct": 5.0,
            "passed": passed_4
        },
        "kriter_5_test_core_entity_isolation": {
            "test_core_entities_count": len(test_entities),
            "leaked_entities_count": len(b1_5_entity_leaks),
            "leaked_entities": sorted(list(b1_5_entity_leaks)),
            "passed": passed_5
        },
        "all_passed": all_passed
    }

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(gate_data, fp, ensure_ascii=False, indent=2)
        print(f"  [Kayıt] Kapı denetim sonuçları kaydedildi: '{args.json}'\n")

    if all_passed:
        print("  >>> NİHAİ ÖN-EĞİTİM KAPISI: TÜM 5 KRİTER BAŞARIYLA GEÇİLDİ! <<<")
        print("  >>> EĞİTİM SÜRECİNE BAŞLANMASINA ONAY VERİLMİŞTİR. <<<")
        print("=================================================================\n")
        sys.exit(0)
    else:
        print("  >>> NİHAİ ÖN-EĞİTİM KAPISI: BAZI KRİTERLER SAĞLANAMADI! <<<")
        print("  >>> EĞİTİM GÜVENLİK GEREKÇESİYLE KİLİTLENMİŞTİR. <<<")
        print("=================================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
