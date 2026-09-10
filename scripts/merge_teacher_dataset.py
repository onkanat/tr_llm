#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: ÖĞRETMEN VERİSİ ENTEGRASYONU VE SÖZLÜK GENİŞLETME
==========================================================================
Bu betik:
1. Arena oturumlarında öğretmen model (gpt-oss:20b / Ollama / Gemini) tarafından
   üretilen bilgi kartlarını ve epistemik kütük kayıtlarını (future_train) toplar.
2. Bu verileri temiz instruction formatına dönüştürerek temel eğitim veri setine
   (`data/pedagogy/high_school_foundation_dataset.jsonl`) tekilleştirerek ekler.
3. Yeni eklenen metinlerdeki kelimeleri analiz ederek sözlükte bulunmayan yeni
   morfemleri tespit eder ve model ağırlık cerrahisi (weight surgery) ile ekler.
4. Temel eğitim setini yeniden derler (`data/train_pedagogy_highschool.bin`) ve
   modeli genişletilmiş dağarcıkla eğitir.
"""

import os
import sys
import re
import json
import argparse
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Set, Any, Tuple

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.gateway.retrain_pipeline import expand_model_vocabulary
from scripts.train_step_demo import KristalLM


def extract_teacher_records(
    archive_path: str = "data/future_train_archive.jsonl",
    backlog_path: str = "data/future_train_vector.jsonl"
) -> List[Dict[str, str]]:
    """
    Extracts high-quality instruction-input-output pairs from archive and active backlog.
    """
    records = []
    seen_keys = set()

    paths = [archive_path, backlog_path]
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rag_doc = data.get("rag_document", "").strip()
                    raw_input = data.get("input", "").strip()
                    instruction = data.get("instruction", "Belgeye göre cevapla.")
                    decompiled_out = data.get("decompiled_output", "").strip()

                    # Extract query from raw_input if present
                    query = ""
                    if "sorgu:" in raw_input:
                        query = raw_input.split("sorgu:")[-1].strip()
                    elif "Soru:" in raw_input:
                        query = raw_input.split("Soru:")[-1].strip()

                    # 1. Clean Direct QA Record (if we have a clear query and teacher doc)
                    if query and rag_doc:
                        key = (query.lower(), rag_doc.lower())
                        if key not in seen_keys:
                            seen_keys.add(key)
                            records.append({
                                "instruction": "Konuyu lise müfredatına ve bilimsel ilkelere uygun olarak açıkla.",
                                "input": f"Soru: {query}",
                                "output": f"Cevap: {rag_doc}"
                            })

                    # 2. Document-grounded RAG Record
                    if rag_doc and decompiled_out and query:
                        key2 = (raw_input.lower(), decompiled_out.lower())
                        if key2 not in seen_keys:
                            seen_keys.add(key2)
                            records.append({
                                "instruction": instruction,
                                "input": f"belge: {rag_doc} sorgu: {query}",
                                "output": decompiled_out
                            })
                except Exception:
                    continue

    return records


def merge_into_foundation_dataset(
    teacher_records: List[Dict[str, str]],
    foundation_path: str = "data/pedagogy/high_school_foundation_dataset.jsonl",
    dry_run: bool = False
) -> int:
    """
    Merges teacher records into foundation dataset without duplicates.
    Returns number of newly added records.
    """
    existing_fingerprints = set()
    if os.path.exists(foundation_path):
        with open(foundation_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        fp = (d.get("input", "").strip().lower(), d.get("output", "").strip().lower())
                        existing_fingerprints.add(fp)
                    except Exception:
                        pass

    added = 0
    new_lines = []
    for rec in teacher_records:
        fp = (rec.get("input", "").strip().lower(), rec.get("output", "").strip().lower())
        if fp not in existing_fingerprints:
            existing_fingerprints.add(fp)
            new_lines.append(json.dumps(rec, ensure_ascii=False))
            added += 1

    if added > 0 and not dry_run:
        with open(foundation_path, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

    return added


def discover_new_tokens_and_expand(
    new_records: List[Dict[str, str]],
    vocab_path: str = "data/vocab.json",
    lexicon_path: str = "data/lexicon/roots.tsv",
    model_path: str = "data/kristal_model.pt",
    device: str = "cpu",
    dry_run: bool = False
) -> List[str]:
    """
    Analyzes all text in new records using CrystalCompiler.
    Finds morphemes/lemmas not currently in Vocabulary, registers them,
    and performs model weight surgery if model exists.
    """
    vocab = Vocabulary()
    vocab.load(vocab_path, freeze=False)
    initial_size = len(vocab.stoi)

    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())

    candidate_new_tokens = set()
    for rec in new_records:
        text = f"{rec.get('input', '')} {rec.get('output', '')}"
        words = re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]+', text)
        for w in words:
            pack = compiler.compile(w)
            if pack and pack.get("token_vector"):
                for morph in pack["token_vector"]:
                    if morph not in vocab.stoi:
                        candidate_new_tokens.add(morph)

    sorted_new_tokens = sorted(list(candidate_new_tokens))
    if not sorted_new_tokens:
        print("  * Keşfedilen yeni morfem/kök yok (Sözlük güncel).")
        return []

    print(f"  * {len(sorted_new_tokens)} adet yeni morfem/kök keşfedildi: {sorted_new_tokens[:10]}...")

    if dry_run:
        return sorted_new_tokens

    # 1. Register in Vocabulary
    new_ids = vocab.register_new_tokens(sorted_new_tokens)
    vocab.save(vocab_path)
    print(f"  * Sözlük güncellendi: {initial_size} -> {len(vocab.stoi)} morfem tokeni.")

    # 2. Weight surgery on Model checkpoint
    if os.path.exists(model_path):
        print(f"  * Model ağırlık cerrahisi (Weight Surgery) uygulanıyor: {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        
        # Instantiate KristalLM with expanded vocab
        model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
        # Resize state dict
        from train import resize_state_dict
        new_sd = resize_state_dict(model, state_dict)
        model.load_state_dict(new_sd, strict=False)
        torch.save(model.state_dict(), model_path)
        print("  * Model kontrol noktası yeni kelime haznesi boyutuyla başarıyla kaydedildi.")

    return sorted_new_tokens


def main():
    parser = argparse.ArgumentParser(description="Öğretmen Veri Entegrasyonu ve Sözlük Genişletme")
    parser.add_argument("--dry-run", action="store_true", help="Değişiklik yapmadan simüle et")
    parser.add_argument("--recompile-bin", action="store_true", default=True, help="train_pedagogy_highschool.bin dosyasını yeniden derle")
    parser.add_argument("--train-steps", type=int, default=100, help="Eğitim adım sayısı (0 ise eğitimi çalıştırma)")
    parser.add_argument("--device", type=str, default="cpu", help="Hesaplama cihazı (cpu/mps/cuda)")
    args = parser.parse_args()

    print("=" * 65, flush=True)
    print(" KRİSTAL-VEKTÖREL: ÖĞRETMEN VERİ ENTEGRASYONU & SÖZLÜK CERRAHİSİ", flush=True)
    print("=" * 65, flush=True)

    # 1. Extract records
    print("\n[1/4] Öğretmen Kayıtları (Arena & Epistemik Kütük) Ayrıştırılıyor...", flush=True)
    teacher_recs = extract_teacher_records()
    print(f"  * Toplam {len(teacher_recs)} adet öğretmen bilgi/QA örneği çıkarıldı.", flush=True)

    if not teacher_recs:
        print("İşlenecek öğretmen verisi bulunamadı.", flush=True)
        return

    # 2. Merge into foundation dataset
    print("\n[2/4] Temel Lise / Pedagoji Veri Setine Entegre Ediliyor...", flush=True)
    added_count = merge_into_foundation_dataset(teacher_recs, dry_run=args.dry_run)
    print(f"  * {added_count} yeni benzersiz örnek 'data/pedagogy/high_school_foundation_dataset.jsonl' dosyasına eklendi.", flush=True)

    # 3. Discover new tokens & Weight surgery
    print("\n[3/4] Yeni Morfem/Kök Keşfi ve Model Ağırlık Cerrahisi...", flush=True)
    new_tokens = discover_new_tokens_and_expand(teacher_recs, device=args.device, dry_run=args.dry_run)

    # 4. Recompile and train
    if not args.dry_run and args.recompile_bin:
        print("\n[4/4] Temel Eğitim Kütüğü Yeniden Derleniyor...", flush=True)
        prep_cmd = [sys.executable, "-u", "scripts/prepare_pedagogy_high_school_dataset.py"]
        res = subprocess.run(prep_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode == 0:
            print("  * 'data/train_pedagogy_highschool.bin' başarıyla derlendi.", flush=True)
        else:
            print(f"  * Derleme hatası:\n{res.stdout}", flush=True)
            return

        if args.train_steps > 0:
            print(f"\nModel Genişletilmiş Dağarcıkla Eğitiliyor ({args.train_steps} Adım)...", flush=True)
            train_cmd = [
                sys.executable, "-u", "train.py",
                "--data", "data/train_pedagogy_highschool.bin",
                "--steps", str(args.train_steps),
                "--batch-size", "16",
                "--lr", "2e-4",
                "--device", args.device
            ]
            t_res = subprocess.run(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(t_res.stdout[-600:] if t_res.stdout else "", flush=True)

    print("\n[Tamamlandı] Öğretmen veri entegrasyonu, sözlük genişletme ve eğitim başarıyla sonuçlandı.", flush=True)


if __name__ == "__main__":
    main()
