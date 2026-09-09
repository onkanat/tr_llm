#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from typing import List, Dict, Set

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

def extract_corpus_words() -> Set[str]:
    """Extracts unique clean Turkish words from raw_corpus and gts examples."""
    words = set()
    
    # 1. Read from raw_corpus.txt
    raw_path = 'data/raw_corpus.txt'
    if os.path.exists(raw_path):
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                for w in line.strip().split():
                    clean_w = w.strip(".,!?\"'…—«»/()-;:*1234567890")
                    if len(clean_w) >= 3 and not clean_w.isupper():
                        words.add(clean_w.lower())

    # 2. Read sample words from gts.json
    gts_path = 'data/poems/gts.json'
    if os.path.exists(gts_path):
        count = 0
        with open(gts_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    madde = entry.get("madde", "")
                    if madde and " " not in madde and len(madde) >= 3:
                        words.add(madde.lower())
                    
                    # Extract from literary examples
                    for a in entry.get("anlamlarListe", []):
                        for o in a.get("orneklerListe", []):
                            for w in o.get("ornek", "").split():
                                clean_w = w.strip(".,!?\"'…—«»/()-;:*1234567890")
                                if len(clean_w) >= 3 and not clean_w.isupper():
                                    words.add(clean_w.lower())
                except Exception:
                    continue
                count += 1
                if count >= 15000: # Sample sufficient variety
                    break
                    
    print(f"Toplam {len(words)} benzersiz kelime toplandı.")
    return words

def generate_deep_parenting_dataset(target_count: int = 40000):
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DERİNLEŞTİRİLMİŞ EBEVEYNLİK (SFT) VERİ ÜRETİCİ")
    print("=" * 60)

    # 1. Initialize Compiler
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)

    words = extract_corpus_words()
    print("Kelimeler Kristal Derleyici ile ayrıştırılıyor...")

    dataset: List[Dict[str, str]] = []
    task_counts = {
        "root": 0,
        "case": 0,
        "tense": 0,
        "plural": 0,
        "possessive": 0,
        "negation": 0,
        "potential": 0,
        "derivation": 0,
        "segmentation": 0,
        "synthesis": 0
    }

    words_list = list(words)
    random.seed(42)
    random.shuffle(words_list)

    for word in words_list:
        if len(dataset) >= target_count:
            break
            
        res = compiler.compile(word)
        if not res.get("analyses"):
            continue
            
        best = res["analyses"][0]
        morphemes = best.get("morphemes", [])
        if not morphemes:
            continue
            
        root_id = morphemes[0]["id"]
        suffix_ids = [m["id"] for m in morphemes[1:]]
        full_tags = " ".join([m["id"] for m in morphemes])

        # 1. Kök Bulma
        dataset.append({
            "instruction": "Kelimedeki kök morfemini bul.",
            "input": word,
            "output": f"ROOT: {root_id}"
        })
        task_counts["root"] += 1

        # 2. Hâl (Durum) Eki Tespiti
        cases = [s for s in suffix_ids if s.startswith("CASE_")]
        if cases:
            dataset.append({
                "instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
                "input": word,
                "output": f"CASE: {', '.join(cases)}"
            })
            task_counts["case"] += 1

        # 3. Zaman / Kip Eki Tespiti
        tenses = [s for s in suffix_ids if s.startswith("TENSE_")]
        if tenses:
            dataset.append({
                "instruction": "Kelimedeki eylemin zamanını veya kipini tespit et.",
                "input": word,
                "output": f"TENSE: {', '.join(tenses)}"
            })
            task_counts["tense"] += 1

        # 4. Çoğul Eki Tespiti
        has_plural = "PLURAL" in suffix_ids
        dataset.append({
            "instruction": "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.",
            "input": word,
            "output": "PLURAL: Evet" if has_plural else "PLURAL: Hayır"
        })
        task_counts["plural"] += 1

        # 5. İYELİK EKİ TESPİTİ (YENİ)
        possessives = [s for s in suffix_ids if s.startswith("POSS_")]
        if possessives:
            dataset.append({
                "instruction": "Kelimenin aldığı iyelik (aitlik) ekini tespit et.",
                "input": word,
                "output": f"POSS: {', '.join(possessives)}"
            })
            task_counts["possessive"] += 1

        # 6. OLUMSUZLUK EKİ TESPİTİ (YENİ)
        has_neg = any(s in suffix_ids for s in ["NEG", "IMPOTENTIAL_NEG"])
        if has_neg or (random.random() < 0.2 and morphemes[0].get("pos") == "VERB"):
            dataset.append({
                "instruction": "Kelimede olumsuzluk eki olup olmadığını tespit et.",
                "input": word,
                "output": "NEG: Evet" if has_neg else "NEG: Hayır"
            })
            task_counts["negation"] += 1

        # 7. YETERLİLİK EKİ TESPİTİ (YENİ)
        has_pot = any(s in suffix_ids for s in ["POTENTIAL", "IMPOTENTIAL_NEG"])
        if has_pot or (random.random() < 0.15 and morphemes[0].get("pos") == "VERB"):
            dataset.append({
                "instruction": "Kelimede yeterlilik eki olup olmadığını tespit et.",
                "input": word,
                "output": "POTENTIAL: Evet" if has_pot else "POTENTIAL: Hayır"
            })
            task_counts["potential"] += 1

        # 8. YAPIM/TÜRETİM EKİ TESPİTİ (YENİ)
        derivs = [s for s in suffix_ids if s.startswith("DERIV_")]
        if derivs:
            dataset.append({
                "instruction": "Kelimenin aldığı yapım eklerini tespit et.",
                "input": word,
                "output": f"DERIV: {', '.join(derivs)}"
            })
            task_counts["derivation"] += 1

        # 9. TAM MORFOLOJİK AYRIŞTIRMA / SEGMENTASYON (YENİ)
        if len(morphemes) >= 2 and random.random() < 0.5:
            dataset.append({
                "instruction": "Kelimeyi kök ve ek morfemlerine ayrıştır.",
                "input": word,
                "output": full_tags
            })
            task_counts["segmentation"] += 1

        # 10. TERSİNE SENTEZLEME / DECOMPILATION GÖREVİ (YENİ)
        if len(morphemes) >= 2 and random.random() < 0.4:
            dataset.append({
                "instruction": "Verilen morfem dizisinden Türkçe kelimeyi sentezle.",
                "input": full_tags,
                "output": word
            })
            task_counts["synthesis"] += 1

    random.shuffle(dataset)
    
    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'parenting_deep_dataset.jsonl')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print(" DERİNLEŞTİRİLMİŞ EBEVEYNLİK VERİ SETİ TAMAMLANDI!")
    print("=" * 60)
    print(f"Toplam Örnek Sayısı: {len(dataset)}")
    print("Görev Dağılımı:")
    for task_name, count in task_counts.items():
        print(f"  * {task_name:<14}: {count:,}")
    print(f"Kayıt Yolu: {out_path}")

if __name__ == '__main__':
    generate_deep_parenting_dataset()
