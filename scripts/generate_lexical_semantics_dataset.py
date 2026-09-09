#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from typing import List, Dict

def generate_lexical_semantics(target_count: int = 25000):
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: GTS SÖZLÜK VE ANLAMSAL BAĞLAM ÜRETİCİ")
    print("=" * 60)

    gts_path = 'data/poems/gts.json'
    if not os.path.exists(gts_path):
        print(f"Hata: {gts_path} bulunamadı!")
        return

    dataset: List[Dict[str, str]] = []
    task_counts = {
        "definition": 0,
        "example": 0,
        "proverb": 0
    }

    with open(gts_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue

            madde = entry.get("madde", "").strip()
            if not madde or len(madde) < 2 or " " in madde:
                continue

            anlamlar = entry.get("anlamlarListe", [])
            atasozleri = entry.get("atasozu", [])

            # 1. Sözlük Tanımı Görevi
            for a in anlamlar:
                anlam_metni = a.get("anlam", "").strip()
                if anlam_metni and len(anlam_metni) >= 8 and not anlam_metni.startswith("bkz."):
                    dataset.append({
                        "instruction": "Kelimenin sözlük tanımını açıkla.",
                        "input": madde,
                        "output": anlam_metni
                    })
                    task_counts["definition"] += 1
                    break # First clear definition is enough

            # 2. Edebi Örnek Cümle Görevi
            for a in anlamlar:
                ornekler = a.get("orneklerListe", [])
                for o in ornekler:
                    ornek_metni = o.get("ornek", "").strip()
                    yazar = o.get("yazar", [{}])[0].get("tam_adi", "") if o.get("yazar") else ""
                    if ornek_metni and len(ornek_metni) >= 15:
                        out_str = f"\"{ornek_metni}\""
                        if yazar:
                            out_str += f" — {yazar}"
                        dataset.append({
                            "instruction": "Kelimenin edebi kullanımına bir örnek ver.",
                            "input": madde,
                            "output": out_str
                        })
                        task_counts["example"] += 1
                        break
                if task_counts["example"] >= target_count // 3:
                    pass

            # 3. Atasözü ve Deyim Görevi
            if atasozleri:
                atasozu_maddeleri = [at.get("madde", "").strip() for at in atasozleri if at.get("madde")]
                if atasozu_maddeleri:
                    selected_at = random.choice(atasozu_maddeleri)
                    dataset.append({
                        "instruction": "Kelimeyi içeren bir atasözü veya deyim belirt.",
                        "input": madde,
                        "output": selected_at
                    })
                    task_counts["proverb"] += 1

            if len(dataset) >= target_count:
                break

    random.seed(42)
    random.shuffle(dataset)

    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'lexical_semantics_dataset.jsonl')

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print(" SÖZLÜK VE ANLAMSAL VERİ SETİ TAMAMLANDI!")
    print("=" * 60)
    print(f"Toplam Örnek Sayısı: {len(dataset):,}")
    print("Görev Dağılımı:")
    for t_name, count in task_counts.items():
        print(f"  * {t_name:<14}: {count:,}")
    print(f"Kayıt Yolu: {out_path}")

if __name__ == '__main__':
    generate_lexical_semantics()
