#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADIM B1: KANAL NORMALİZASYONU (KANONİK ŞEMA DÖNÜŞTÜRÜCÜ)
======================================================
Tüm veri kümelerini kanonik şemaya dönüştürür:
  - instruction: Yalnızca rol / sistem görevi tanımı.
  - input: Soru veya içerik (asla boş kalmaz).
  - output: Mükerrer öneklerden arındırılmış saf yanıt.

Orijinal dosyalar korunur; çıktılar 'data/pedagogy_canonical/' altına yazılır.
"""

import os
import re
import json

CANONICAL_DIR = 'data/pedagogy_canonical'
os.makedirs(CANONICAL_DIR, exist_ok=True)


def clean_output_prefixes(text: str) -> str:
    cleaned = text.strip()
    # Remove doubled prefixes
    doubled_patterns = [
        r'^(Teknik Çözüm:\s*)+',
        r'^(Usta Marangoz:\s*)+',
        r'^(Cevap:\s*)+',
        r'^(Yanıt:\s*)+',
    ]
    for pat in doubled_patterns:
        cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def normalize_carpenter():
    src = 'data/pedagogy/carpenter_specialization_dataset.jsonl'
    dst = os.path.join(CANONICAL_DIR, 'carpenter_canonical.jsonl')
    if not os.path.exists(src): return 0
    count = 0
    with open(src, 'r', encoding='utf-8') as f_in, open(dst, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            d = json.loads(line)
            raw_inst = d.get('instruction', '').strip()
            raw_inp = d.get('input', '').strip()
            raw_out = d.get('output', '').strip()
            
            # Question is in raw_inp if not empty, otherwise raw_inst
            q = raw_inp if raw_inp else raw_inst
            q = re.sub(r'^(Soru:\s*)', '', q, flags=re.IGNORECASE).strip()
            out = clean_output_prefixes(raw_out)
            
            item = {
                'instruction': 'Ahşap ve marangozluk uzmanı olarak cevapla.',
                'input': q,
                'output': out
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  * Carpenter Canonical: {count:,} kayıt')
    return count


def normalize_turk_tarihi():
    src = 'data/pedagogy/turk_tarihi_sft.jsonl'
    dst = os.path.join(CANONICAL_DIR, 'turk_tarihi_canonical.jsonl')
    if not os.path.exists(src): return 0
    count = 0
    with open(src, 'r', encoding='utf-8') as f_in, open(dst, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            d = json.loads(line)
            raw_inst = d.get('instruction', '').strip()
            raw_inp = d.get('input', '').strip()
            raw_out = d.get('output', '').strip()
            
            # In turk_tarihi, raw_inst is the actual question
            q = raw_inst
            q = re.sub(r'^(Soru:\s*)', '', q, flags=re.IGNORECASE).strip()
            
            # If raw_inp has context that is informative (not just metadata label)
            if raw_inp and not raw_inp.startswith('Bağlam: Türk_tarih_seti'):
                inp = raw_inp + " " + q
            else:
                inp = q
                
            out = clean_output_prefixes(raw_out)
            item = {
                'instruction': 'Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.',
                'input': inp,
                'output': out
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  * Türk Tarihi Canonical: {count:,} kayıt')
    return count


def normalize_high_school():
    src = 'data/pedagogy/high_school_foundation_dataset.jsonl'
    dst = os.path.join(CANONICAL_DIR, 'high_school_canonical.jsonl')
    if not os.path.exists(src): return 0
    count = 0
    with open(src, 'r', encoding='utf-8') as f_in, open(dst, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            d = json.loads(line)
            raw_inst = d.get('instruction', '').strip()
            raw_inp = d.get('input', '').strip()
            raw_out = d.get('output', '').strip()
            
            q = re.sub(r'^(Soru:\s*)', '', raw_inp, flags=re.IGNORECASE).strip()
            out = clean_output_prefixes(raw_out)
            item = {
                'instruction': 'Lise fen ve temel bilimler uzmanı olarak açıkla.',
                'input': q if q else raw_inst,
                'output': out
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  * High School Canonical: {count:,} kayıt')
    return count


def normalize_literature():
    src = 'data/pedagogy/literature_poetry_dataset.jsonl'
    dst = os.path.join(CANONICAL_DIR, 'literature_canonical.jsonl')
    if not os.path.exists(src): return 0
    count = 0
    with open(src, 'r', encoding='utf-8') as f_in, open(dst, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            d = json.loads(line)
            raw_inst = d.get('instruction', '').strip()
            raw_inp = d.get('input', '').strip()
            raw_out = d.get('output', '').strip()
            
            q = raw_inp if raw_inp else raw_inst
            item = {
                'instruction': 'Türk edebiyatı ve kültür uzmanı olarak açıkla.',
                'input': q,
                'output': clean_output_prefixes(raw_out)
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  * Literature Canonical: {count:,} kayıt')
    return count


def normalize_middle_school():
    src = 'data/pedagogy/middle_school_chat.jsonl'
    dst = os.path.join(CANONICAL_DIR, 'middle_school_canonical.jsonl')
    if not os.path.exists(src): return 0
    count = 0
    with open(src, 'r', encoding='utf-8') as f_in, open(dst, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            d = json.loads(line)
            raw_inst = d.get('instruction', '').strip()
            raw_inp = d.get('input', '').strip()
            raw_out = d.get('output', '').strip()
            item = {
                'instruction': 'Ortaokul fen ve sosyal bilgiler dersi kapsamında açıkla.',
                'input': raw_inp if raw_inp else raw_inst,
                'output': clean_output_prefixes(raw_out)
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  * Middle School Canonical: {count:,} kayıt')
    return count


def normalize_parenting():
    src = 'data/pedagogy/parenting_dataset.jsonl'
    dst = os.path.join(CANONICAL_DIR, 'parenting_canonical.jsonl')
    if not os.path.exists(src): return 0
    count = 0
    with open(src, 'r', encoding='utf-8') as f_in, open(dst, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            d = json.loads(line)
            # Parenting is already canonical morphology instruction + input word + output
            item = {
                'instruction': d.get('instruction', '').strip(),
                'input': d.get('input', '').strip(),
                'output': clean_output_prefixes(d.get('output', '').strip())
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  * Parenting Canonical: {count:,} kayıt')
    return count


def main():
    print("=" * 60)
    print(" ADIM B1: VERİ KÜMELERİ KANONİK ŞEMAYA DÖNÜŞTÜRÜLÜYOR")
    print("=" * 60)
    c1 = normalize_carpenter()
    c2 = normalize_turk_tarihi()
    c3 = normalize_high_school()
    c4 = normalize_literature()
    c5 = normalize_middle_school()
    c6 = normalize_parenting()
    tot = c1 + c2 + c3 + c4 + c5 + c6
    print(f"\nToplam Kanonik Kayıt: {tot:,}")
    print(f"Tüm kütükler '{CANONICAL_DIR}/' dizinine kaydedildi.")

if __name__ == '__main__':
    main()
