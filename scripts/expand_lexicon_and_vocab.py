#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: ÖZEL İSİM VE TARİHİ KÖKLERİ SÖZLÜĞE VE KELİME DAĞARCIĞINA EKLEME
================================================================================
Bu betik:
1. Tarihsel ve coğrafi özel isim köklerini (Atatürk, Anadolu, Çin, Pers, İskit, Orhun vb.)
   `data/lexicon/roots.tsv` içine ekler.
2. Tüm pedagoji ve SFT veri setlerini tarar.
3. Keşfedilen yeni morfem ve özel isim köklerini `data/vocab.json` sözlüğüne kaydeder.
"""

import os
import sys
import csv
import json
import re
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import Vocabulary

ROOTS_FILE = "data/lexicon/roots.tsv"
VOCAB_FILE = "data/vocab.json"

HISTORICAL_PROPER_NOUNS = [
    # Türk Tarihi & Devletleri
    ("Atatürk", "NOUN", "-"),
    ("atatürk", "NOUN", "-"),
    ("Anadolu", "NOUN", "-"),
    ("anadolu", "NOUN", "-"),
    ("Çin", "NOUN", "-"),
    ("çin", "NOUN", "-"),
    ("Pers", "NOUN", "-"),
    ("pers", "NOUN", "-"),
    ("İskit", "NOUN", "-"),
    ("iskit", "NOUN", "-"),
    ("Orhun", "NOUN", "-"),
    ("orhun", "NOUN", "-"),
    ("Yenisey", "NOUN", "-"),
    ("yenisey", "NOUN", "-"),
    ("Cengiz", "NOUN", "-"),
    ("cengiz", "NOUN", "-"),
    ("Timur", "NOUN", "-"),
    ("timur", "NOUN", "-"),
    ("Atina", "NOUN", "-"),
    ("atina", "NOUN", "-"),
    ("Sparta", "NOUN", "-"),
    ("sparta", "NOUN", "-"),
    ("İran", "NOUN", "-"),
    ("iran", "NOUN", "-"),
    ("Arap", "NOUN", "-"),
    ("arap", "NOUN", "-"),
    ("Moğol", "NOUN", "-"),
    ("moğol", "NOUN", "-"),
    ("Hazar", "NOUN", "-"),
    ("hazar", "NOUN", "-"),
    ("Avar", "NOUN", "-"),
    ("avar", "NOUN", "-"),
    ("Kıpçak", "NOUN", "-"),
    ("kıpçak", "NOUN", "-"),
    ("Peçenek", "NOUN", "-"),
    ("peçenek", "NOUN", "-"),
    ("Tonyukuk", "NOUN", "-"),
    ("tonyukuk", "NOUN", "-"),
    ("Kültigin", "NOUN", "-"),
    ("kültigin", "NOUN", "-"),
    ("Babür", "NOUN", "-"),
    ("babür", "NOUN", "-"),
    ("Gazneli", "NOUN", "-"),
    ("gazneli", "NOUN", "-"),
    ("Harzemşah", "NOUN", "-"),
    ("harzemşah", "NOUN", "-"),
    ("Bizans", "NOUN", "-"),
    ("bizans", "NOUN", "-"),
    ("Frig", "NOUN", "-"),
    ("frig", "NOUN", "-"),
    ("Lidya", "NOUN", "-"),
    ("lidya", "NOUN", "-"),
    ("Urartu", "NOUN", "-"),
    ("urartu", "NOUN", "-"),
    ("Sümer", "NOUN", "-"),
    ("sümer", "NOUN", "-"),
    ("Hitit", "NOUN", "-"),
    ("hitit", "NOUN", "-"),
    ("Selçuk", "NOUN", "-"),
    ("selçuk", "NOUN", "-"),
    ("Selçuklu", "NOUN", "-"),
    ("selçuklu", "NOUN", "-"),
    ("Osmanlı", "NOUN", "-"),
    ("osmanlı", "NOUN", "-"),
    ("Göktürk", "NOUN", "-"),
    ("göktürk", "NOUN", "-"),
    ("Köktürk", "NOUN", "-"),
    ("köktürk", "NOUN", "-"),
    ("Uygur", "NOUN", "-"),
    ("uygur", "NOUN", "-"),
    ("Hun", "NOUN", "-"),
    ("hun", "NOUN", "-"),
    ("Oğuz", "NOUN", "-"),
    ("oğuz", "NOUN", "-"),
    ("Avrasya", "NOUN", "-"),
    ("avrasya", "NOUN", "-"),
    ("Asya", "NOUN", "-"),
    ("asya", "NOUN", "-"),
    ("Avrupa", "NOUN", "-"),
    ("avrupa", "NOUN", "-"),
    ("Mezopotamya", "NOUN", "-"),
    ("mezopotamya", "NOUN", "-"),
    ("Kafkas", "NOUN", "-"),
    ("kafkas", "NOUN", "-"),
    ("Balkan", "NOUN", "-"),
    ("balkan", "NOUN", "-"),
    ("Ege", "NOUN", "-"),
    ("ege", "NOUN", "-"),
    ("Akdeniz", "NOUN", "-"),
    ("akdeniz", "NOUN", "-"),
    ("Karadeniz", "NOUN", "-"),
    ("karadeniz", "NOUN", "-"),
    ("Mısır", "NOUN", "-"),
    ("mısır", "NOUN", "-"),
    ("Yunan", "NOUN", "-"),
    ("yunan", "NOUN", "-"),
]


def update_roots_tsv():
    print("[1] roots.tsv taranıyor ve eksik özel isimler ekleniyor...")
    existing = set()
    with open(ROOTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            existing.add(row["lemma"])

    added = 0
    with open(ROOTS_FILE, "a", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for lemma, pos, attrs in HISTORICAL_PROPER_NOUNS:
            if lemma not in existing:
                writer.writerow([lemma, pos, attrs])
                existing.add(lemma)
                added += 1

    print(f"  -> roots.tsv dosyasına {added} yeni özel isim kökü eklendi.")


def update_vocabulary():
    print("\n[2] Compiler yeniden yükleniyor ve veri setleri taranıyor...")
    lexicon = LexiconManager()
    lexicon.load_from_tsv(ROOTS_FILE)
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)

    vocab = Vocabulary()
    vocab.load(VOCAB_FILE, freeze=False)
    initial_size = len(vocab.stoi)
    print(f"  -> Başlangıç Sözlük Boyutu: {initial_size} token.")

    # 1. Add all HISTORICAL_PROPER_NOUNS directly
    for lemma, _, _ in HISTORICAL_PROPER_NOUNS:
        res = compiler.compile(lemma)
        for m in res.get("token_vector", []):
            if m not in vocab.stoi:
                vocab.add_token(m)
        if lemma not in vocab.stoi:
            vocab.add_token(lemma)

    # 2. Scan datasets for common words and compile them
    dataset_files = glob.glob("data/pedagogy/*.jsonl") + glob.glob("data/*.jsonl")
    print(f"  -> {len(dataset_files)} veri seti taranıyor...")

    candidate_words = set()
    for filepath in dataset_files:
        if "vector" in filepath or "archive" in filepath:
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        text = d.get("instruction", "") + " " + d.get("input", "") + " " + d.get("output", "")
                        words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text)
                        for w in words:
                            if len(w) >= 2:
                                candidate_words.add(w)
                    except Exception:
                        pass
        except Exception:
            pass

    print(f"  -> Toplam {len(candidate_words):,} aday kelime derlendi. Compiler ile çözümleniyor...")

    new_tokens = set()
    for w in candidate_words:
        res = compiler.compile(w)
        for m in res.get("token_vector", []):
            if m not in vocab.stoi:
                new_tokens.add(m)

    sorted_tokens = sorted(list(new_tokens))
    print(f"  -> Keşfedilen yeni morfem/kök sayısı: {len(sorted_tokens)}")
    for t in sorted_tokens:
        vocab.add_token(t)

    vocab.save(VOCAB_FILE)
    print(f"  -> Sözlük güncellendi ve kaydedildi: {initial_size} -> {len(vocab.stoi)} token.")


if __name__ == "__main__":
    update_roots_tsv()
    update_vocabulary()
