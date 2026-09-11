#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: VEKTÖREL BELLEK ARINDIRMA VE DÖKÜMAN GÜNCELLEME
==========================================================================
Bu betik;
1. kristal_bellek ve simulasyon_bellek koleksiyonlarını tarar.
2. Öğretmen LLM'den sızan ham Chain-of-Thought (CoT) ve İngilizce meta düşünce
   metinlerini tespit eder, temizler veya tamamen siler.
3. ahsap_ve_marangozluk_rehberi.md dosyasındaki güncel ve temiz ahşap bilgisi
   parçalarını (Gürgen, Meşe, Çam, Ceviz, Kırlangıç vb.) kristal_bellek'e indeksler.
"""

import os
import sys
import json
import re
from qdrant_client import QdrantClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector
from src.gateway.pedagogical_supervisor import sanitize_teacher_card

def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL: VEKTÖREL BELLEK ARINDIRMA (SANITIZATION)")
    print("=" * 70)

    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    client = QdrantClient(path="data/qdrant_db")
    
    cot_patterns = [
        r'\bthe user\b', r'\bwe need to\b', r'\bthey want\b', r'\blet\'s produce\b',
        r'\bhere is\b', r'\bso that should\b', r'\bmake it\b', r'\bfirst, let\b',
        r'\bin this case\b', r'\bconcise, pedagogically\b'
    ]
    cot_regex = re.compile('|'.join(cot_patterns), re.IGNORECASE)

    # 1. Purge / Sanitize Dirty Points
    for col_name in ["kristal_bellek", "simulasyon_bellek"]:
        if not client.collection_exists(col_name):
            continue
        print(f"\n[1] '{col_name}' taranıyor...")
        offset = None
        points_to_delete = []
        points_to_update = []
        total_scanned = 0

        while True:
            res, offset = client.scroll(col_name, limit=100, offset=offset, with_payload=True)
            total_scanned += len(res)
            for p in res:
                txt = p.payload.get("text", "")
                if cot_regex.search(txt) or "<think>" in txt or "<thought>" in txt:
                    sanitized = sanitize_teacher_card(txt)
                    if sanitized and len(sanitized) >= 20 and not cot_regex.search(sanitized):
                        points_to_update.append((p.id, sanitized, p.payload))
                    else:
                        points_to_delete.append(p.id)
            if offset is None:
                break

        print(f"  * Taranan Toplam: {total_scanned} kayıt.")
        if points_to_delete:
            print(f"  * Silinen Kirli Kayıt: {len(points_to_delete)} adet -> {points_to_delete}")
            client.delete(col_name, points_selector=points_to_delete)
        if points_to_update:
            print(f"  * Temizlenip Güncellenen: {len(points_to_update)} adet.")
            for pid, clean_txt, payload in points_to_update:
                payload["text"] = clean_txt
                client.set_payload(col_name, payload=payload, points=[pid])

    # 2. Re-index Knowledge Guide (Gürgen, Meşe, Çam, Ceviz vs.) into kristal_bellek
    print("\n[2] Ahşap ve Marangozluk Rehberi (Gürgen vb.) kristal_bellek'e indeksleniyor...")
    memory = VectorMemory(collection_name="kristal_bellek", vector_size=768, client=client)
    guide_path = "data/knowledge/ahsap_ve_marangozluk_rehberi.md"

    if os.path.exists(guide_path):
        with open(guide_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = [s.strip() for s in content.split("## ") if s.strip() and not s.startswith("# ")]
        added_count = 0

        for sec in sections:
            lines = sec.split("\n", 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else title
            doc_text = f"{title}: {body}"

            # Check if this exact text already exists in memory
            existing = memory.dense_recall(generate_kristal_vector(tokenizer.encode(doc_text), tokenizer.decode(tokenizer.encode(doc_text))), top_k=1)
            if existing and existing[0]["score"] >= 0.98:
                continue

            t_ids = tokenizer.encode(doc_text)
            t_tags = tokenizer.decode(t_ids)
            dense_v = generate_kristal_vector(t_ids, t_tags)
            sparse_v = generate_sparse_vector(t_ids, t_tags)

            memory.add_document(
                text=doc_text,
                dense_vector=dense_v,
                sparse_vector=sparse_v,
                metadata={
                    "source": "ahsap_ve_marangozluk_rehberi.md",
                    "title": title,
                    "crystal_tags": t_tags,
                    "token_ids": t_ids
                }
            )
            added_count += 1
            print(f"  + İndekslendi: {title}")

        print(f"\n[3] Toplam yeni indekslenen ahşap rehberi konusu: {added_count} adet.")

    final_kristal = client.count("kristal_bellek").count if client.collection_exists("kristal_bellek") else 0
    final_sim = client.count("simulasyon_bellek").count if client.collection_exists("simulasyon_bellek") else 0
    print(f"\nBELLEK DURUMU:")
    print(f"  * kristal_bellek   : {final_kristal} temiz döküman")
    print(f"  * simulasyon_bellek: {final_sim} temiz döküman")
    print("=" * 70)


if __name__ == "__main__":
    main()
