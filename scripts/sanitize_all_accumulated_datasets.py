#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: TÜM BİRİKEN VERİLERİ VE EĞİTİM KÜTÜKLERİNİ TAM ARINDIRMA (SANITIZATION)
========================================================================================
1. data/pedagogy/arena_base_accumulated.jsonl
2. data/pedagogy/arena_carpenter_accumulated.jsonl
3. data/future_train_archive.jsonl
4. data/future_train_vector.jsonl
5. data/qdrant_db (yerel gömülü veritabanı)

Tüm kayıtlardan DUSUNCE, Öğretmen rolü, hedef kitle, pedagojik kontrol vb.
meta-öğretmen düşünce izlerini tamamen temizler, deklaratif bilgiye indirger.
"""

import os
import sys
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.pedagogical_supervisor import sanitize_teacher_card, extract_cot_and_card

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"

COT_TRIGGERS = [
    r"\[?Öğretmen Açıklaması\]?",
    r"\[?DUSUNCE\]?",
    r"\[?DÜŞÜNCE\]?",
    r"Soru Analizi",
    r"Öğretmen rol",
    r"Öğretmen kimliğ",
    r"hedef kitle",
    r"\*\*Düşünce",
    r"Derinlemesine Düşünce",
    r"\*\*Konu:",
    r"\*\*Hedef Kitle",
    r"\*\*Çıktı Formatı",
    r"Adım 1:",
    r"Adım 2:",
    r"Temel Kavramlar:",
    r"İlk taslak",
    r"Son kontrol",
    r"<think>",
    r"<thought>",
    r"<reasoning>",
    r"<DUSUNCE>",
    r"we need to",
    r"the user says",
    r"they want"
]
DELIM_REGEX = re.compile(r"(?i)(" + "|".join(COT_TRIGGERS) + ")")


def is_contaminated(text: str) -> bool:
    if not text:
        return False
    return bool(DELIM_REGEX.search(text))


def clean_record_output(raw_output: str) -> str:
    """Extracts pure declarative answer from contaminated output with zero leakage."""
    if not raw_output or not isinstance(raw_output, str):
        return ""

    text = raw_output.strip()

    # Strategy 1: If text starts with clean concept and then drops into CoT triggers
    parts = DELIM_REGEX.split(text)
    if parts:
        prefix = parts[0].strip(" \n\r\t-:*\"'“’")
        if len(prefix) >= 15 and not is_contaminated(prefix):
            # Clean leading labels if any
            prefix = re.sub(r'^(?:\[?(?:Bilgi Kartı|Cevap|Özet|Teknik Çözüm)\]?)\s*:\s*', '', prefix, flags=re.IGNORECASE)
            prefix = prefix.strip(" \n\r\t-:*\"'“’")
            if len(prefix) >= 15 and not is_contaminated(prefix):
                return prefix

    # Strategy 2: Look for explicit <BILGI_KARTI> or text BILGI_KARTI / Bilgi Kartı header
    card_match = re.search(r'(?i)<BILGI_KARTI>(.*?)</BILGI_KARTI>', text, flags=re.DOTALL)
    if card_match:
        c = card_match.group(1).strip(" \n\r\t-:*\"'“’")
        c = re.sub(r'^(?:\[?(?:Bilgi Kartı|Cevap|Özet|Teknik Çözüm)\]?)\s*:\s*', '', c, flags=re.IGNORECASE)
        c = c.strip(" \n\r\t-:*\"'“’")
        if len(c) >= 15 and not is_contaminated(c):
            return c

    header_match = re.search(r'(?i)(?:BILGI_KARTI|BİLGİ KARTI|Bilgi Kartı|Son kontrol)[\s*:]+([^\n]+(?:\n[^\n]+)?)', text)
    if header_match:
        c = header_match.group(1).strip(" \n\r\t-:*\"'“’")
        c = re.sub(r'^(?:\[?(?:Bilgi Kartı|Cevap|Özet|Teknik Çözüm)\]?)\s*:\s*', '', c, flags=re.IGNORECASE)
        c = c.strip(" \n\r\t-:*\"'“’")
        if len(c) >= 15 and not is_contaminated(c):
            return c

    # Strategy 3: sanitize_teacher_card fallback
    san = sanitize_teacher_card(text)
    if san and len(san) >= 15 and not is_contaminated(san):
        return san

    return ""


def sanitize_jsonl_file(file_path: str):
    source_path = file_path + ".bak" if os.path.exists(file_path + ".bak") else file_path
    if not os.path.exists(source_path):
        print(f"{C_YELLOW}[!] Dosya bulunamadı: {source_path}{C_RESET}")
        return

    print(f"\n{C_BOLD}Arındırılıyor: {file_path}{C_RESET}")

    cleaned_records = []
    dropped_count = 0
    total_count = 0

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            try:
                data = json.loads(line)
                raw_out = data.get("output", "")

                clean_out = clean_record_output(raw_out)
                if not clean_out or len(clean_out) < 15 or is_contaminated(clean_out):
                    dropped_count += 1
                    continue

                # Clean input field if it has contaminated context
                raw_in = data.get("input", "")
                if is_contaminated(raw_in):
                    # If input is 'belge: <doc> sorgu: <q>'
                    if "sorgu:" in raw_in:
                        sorgu_part = raw_in.split("sorgu:", 1)[1].strip()
                        raw_in = f"sorgu: {sorgu_part}"
                    else:
                        in_clean = DELIM_REGEX.split(raw_in)[0].strip()
                        raw_in = in_clean if len(in_clean) >= 5 else raw_in

                data["input"] = raw_in
                data["output"] = clean_out
                cleaned_records.append(data)
            except Exception:
                dropped_count += 1
                continue

    with open(file_path, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {C_GREEN}Tamamlandı:{C_RESET} Toplam {total_count} kayıt -> {len(cleaned_records)} tertemiz kayıt korundu ({dropped_count} kirli/kurtarılamaz elendi).")


def main():
    print(f"{C_BOLD}{C_GREEN}======================================================================")
    print("  KRİSTAL-VEKTÖREL: DERİN VERİ ARINDIRMA (ZERO LEAKAGE)")
    print("======================================================================\n" + C_RESET)

    files_to_clean = [
        "data/pedagogy/arena_base_accumulated.jsonl",
        "data/pedagogy/arena_carpenter_accumulated.jsonl",
        "data/future_train_archive.jsonl",
        "data/future_train_vector.jsonl"
    ]

    for fpath in files_to_clean:
        sanitize_jsonl_file(fpath)

    # Sanitize local embedded Qdrant db if present
    from qdrant_client import QdrantClient
    if os.path.exists("data/qdrant_db"):
        print(f"\n{C_BOLD}Yerel Qdrant (data/qdrant_db) Arındırılıyor...{C_RESET}")
        try:
            client = QdrantClient(path="data/qdrant_db")
            for col in ["kristal_bellek", "simulasyon_bellek"]:
                if client.collection_exists(col):
                    res, _ = client.scroll(col, limit=2000, with_payload=True)
                    dirty_ids = []
                    update_points = []
                    for p in res:
                        txt = p.payload.get("text", "")
                        if is_contaminated(txt):
                            clean_t = clean_record_output(txt)
                            if clean_t and len(clean_t) >= 15 and not is_contaminated(clean_t):
                                p.payload["text"] = clean_t
                                update_points.append((p.id, p.payload))
                            else:
                                dirty_ids.append(p.id)
                    if dirty_ids:
                        client.delete(col, points_selector=dirty_ids)
                        print(f"  * {col}: {len(dirty_ids)} kirli kayıt silindi.")
                    if update_points:
                        for pid, pld in update_points:
                            client.set_payload(col, payload=pld, points=[pid])
                        print(f"  * {col}: {len(update_points)} kayıt temizlenerek güncellendi.")
                    if not dirty_ids and not update_points:
                        print(f"  * {col}: Tertemiz (0 kirli kayıt).")
        except Exception as e:
            print(f"  [!] Qdrant temizleme hatası: {e}")

    print(f"\n{C_BOLD}{C_GREEN}Tüm veri arındırma süreci başarıyla tamamlandı!{C_RESET}\n")


if __name__ == "__main__":
    main()
