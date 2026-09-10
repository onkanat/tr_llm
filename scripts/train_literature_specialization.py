#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: TÜRKÇE EDEBİYAT VE ŞİİR DERİNLEŞTİRME VE EĞİTİM BETİĞİ
========================================================================
Halk Şiiri, Divan Edebiyatı, Tanzimat, Servet-i Fünun, Fecr-i Ati, Milli Edebiyat
ve Cumhuriyet Dönemi Türk Şiiri konularında:
1. 220 altın soru-cevap ve bilgi kartını Qdrant'a (`simulasyon_bellek`) toplu enjekte eder.
2. Temel eğitim setine (`data/pedagogy/high_school_foundation_dataset.jsonl`) ekler.
3. Model kelime haznesini ve ağırlıklarını cerrahiyle günceller.
4. `train_pedagogy_highschool.bin` ikili dosyasını derler.
5. Modeli 150 adım eğiterek ağırlıkları günceller.
"""

import os
import sys
import json
import subprocess
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.agent_gateway import AgentGateway
from scripts.merge_teacher_dataset import (
    merge_into_foundation_dataset,
    discover_new_tokens_and_expand
)

LIT_DATASET_PATH = "data/pedagogy/literature_poetry_dataset.jsonl"
FOUNDATION_PATH = "data/pedagogy/high_school_foundation_dataset.jsonl"


def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL: TÜRKÇE EDEBİYAT & ŞİİR DERİNLEŞTİRME PROGRAMI")
    print("=" * 70)

    if not os.path.exists(LIT_DATASET_PATH):
        print(f"Hata: '{LIT_DATASET_PATH}' bulunamadı. Önce üreteci çalıştırın.")
        return

    records = []
    with open(LIT_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n[1/5] {len(records)} adet Edebiyat & Şiir kaydı yüklendi.")

    # 1. Qdrant RAG Enjeksiyonu
    print("\n[2/5] Qdrant 'simulasyon_bellek' Koleksiyonuna Toplu Bilgi Enjeksiyonu...")
    gateway = AgentGateway.create_default(device="cpu")
    injected_count = 0
    for idx, rec in enumerate(records):
        inp = rec.get("input", "")
        out = rec.get("output", "")
        text = f"{inp}\n{out}".replace("Soru:", "").replace("Cevap:", "").strip()
        
        try:
            doc_id = gateway.inject_knowledge(text, target_collection="simulasyon_bellek")
            injected_count += 1
            if (idx + 1) % 50 == 0 or idx == len(records) - 1:
                print(f"  * {idx + 1}/{len(records)} bilgi kartı başarıyla indekslendi.")
        except Exception as e:
            continue
            
    print(f"  * Toplam {injected_count} bilgi kartı Qdrant belleğe işlendi.")
    gateway.close()

    # 2. Temel Eğitim Setine Ekleme
    print("\n[3/5] 'data/pedagogy/high_school_foundation_dataset.jsonl' Dosyasına Entegre Ediliyor...")
    added = merge_into_foundation_dataset(records, foundation_path=FOUNDATION_PATH)
    print(f"  * {added} yeni edebi kayıt temel eğitim setine kalıcı olarak eklendi.")

    # 3. Yeni Morfem Keşfi ve Ağırlık Cerrahisi
    print("\n[4/5] Edebi Terminoloji Sözlük Taraması ve Ağırlık Cerrahisi...")
    new_tokens = discover_new_tokens_and_expand(records, device="cpu", dry_run=False)

    # 4. İkili Eğitim Kütüğünü Derleme ve 150 Adım Eğitim
    print("\n[5/5] 'data/train_pedagogy_highschool.bin' Derleniyor ve Model Eğitiliyor...")
    prep_res = subprocess.run([sys.executable, "scripts/prepare_pedagogy_high_school_dataset.py"], capture_output=True, text=True)
    if prep_res.returncode != 0:
        print(f"Hata: Veri derleme başarısız:\n{prep_res.stdout}")
        return

    print("  * Veri paketi başarıyla derlendi. Eğitim başlatılıyor (150 Adım)...")
    train_cmd = [
        sys.executable, "train.py",
        "--data", "data/train_pedagogy_highschool.bin",
        "--steps", "150",
        "--batch-size", "16",
        "--lr", "2e-4",
        "--device", "cpu"
    ]
    t_res = subprocess.run(train_cmd, capture_output=True, text=True)
    print(t_res.stdout[-700:] if t_res.stdout else "")

    print("\n" + "=" * 70)
    print(" TÜRKÇE EDEBİYAT & ŞİİR DERİNLEŞTİRME VE EĞİTİM TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
