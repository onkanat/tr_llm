#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: 1931 TÜRK TARİHİ VERİ HAZIRLIK & ENTEGRASYON BORU HATTI
========================================================================
Bu betik:
1. HF onkanat/turk-tarihi-1931-sft-dpo veri setini (tr_sft, tr_chat, tr_dpo) yükler.
2. SFT kayıtlarını temizleyip `data/pedagogy/turk_tarihi_sft.jsonl` olarak hazırlar.
3. Chat diyaloglarını formatlayıp `data/pedagogy/turk_tarihi_chat.jsonl` olarak hazırlar.
4. DPO tercih çiftlerini tokenize edip `data/pedagogy/turk_tarihi_dpo_tokenized.jsonl` ve
   `data/pedagogy/dpo_all_tokenized.jsonl` içine entegre eder.
5. Yeni morfem/kök taraması yaparak gerekirse sözlüğü ve model ağırlıklarını cerrahiyle genişletir.
6. Qdrant 'simulasyon_bellek' koleksiyonuna tarihsel bilgi kartlarını indeksler.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.rag.vector_memory import VectorMemory
from src.gateway.agent_gateway import AgentGateway
from scripts.merge_teacher_dataset import discover_new_tokens_and_expand

DATA_DIR = "data/pedagogy/turk_tarihi_1931"
RAW_SFT = os.path.join(DATA_DIR, "tr_sft_dataset.jsonl")
RAW_CHAT = os.path.join(DATA_DIR, "tr_chat_dataset.jsonl")
RAW_DPO = os.path.join(DATA_DIR, "tr_dpo_dataset.jsonl")

OUT_SFT = "data/pedagogy/turk_tarihi_sft.jsonl"
OUT_CHAT = "data/pedagogy/turk_tarihi_chat.jsonl"
OUT_DPO_TOK = "data/pedagogy/turk_tarihi_dpo_tokenized.jsonl"
DPO_ALL_PATH = "data/pedagogy/dpo_all_tokenized.jsonl"


def prepare_sft():
    print("\n[1/5] Türkçe 1931 Tarih SFT Verisi Hazırlanıyor...")
    if not os.path.exists(RAW_SFT):
        raise FileNotFoundError(f"'{RAW_SFT}' bulunamadı!")

    sft_records = []
    with open(RAW_SFT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                inst = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                out = item.get("output", "").strip()
                if not inst or not out:
                    continue
                sft_records.append({
                    "instruction": inst,
                    "input": inp,
                    "output": out
                })
            except Exception:
                continue

    with open(OUT_SFT, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  * {len(sft_records):,} adet SFT kaydı '{OUT_SFT}' dosyasına yazıldı.")
    return sft_records


def prepare_chat():
    print("\n[2/5] Türkçe 1931 Tarih Chat Verisi Hazırlanıyor...")
    if not os.path.exists(RAW_CHAT):
        raise FileNotFoundError(f"'{RAW_CHAT}' bulunamadı!")

    chat_records = []
    with open(RAW_CHAT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                messages = item.get("messages", [])
                if len(messages) >= 2:
                    user_msg = messages[0].get("content", "").strip()
                    asst_msg = messages[1].get("content", "").strip()
                    if user_msg and asst_msg:
                        chat_records.append({
                            "instruction": "Tarih araştırmaları ve Cumhuriyet dönemi tarih tezleri uzmanı olarak kullanıcı ile diyalog yürüt.",
                            "input": user_msg,
                            "output": asst_msg
                        })
            except Exception:
                continue

    with open(OUT_CHAT, "w", encoding="utf-8") as f:
        for rec in chat_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  * {len(chat_records):,} adet Sohbet kaydı '{OUT_CHAT}' dosyasına yazıldı.")
    return chat_records


def prepare_dpo(tokenizer: KristalTokenizer, vocab: Vocabulary):
    print("\n[3/5] Türkçe 1931 Tarih DPO Tercih Çiftleri Tokenize Ediliyor...")
    if not os.path.exists(RAW_DPO):
        raise FileNotFoundError(f"'{RAW_DPO}' bulunamadı!")

    bos_id = vocab.encode("<BOS>")
    eos_id = vocab.encode("<EOS>")
    inst_start_id = vocab.encode("<INSTRUCTION>")
    inst_end_id = vocab.encode("</INSTRUCTION>")
    output_start_id = vocab.encode("<OUTPUT>")
    output_end_id = vocab.encode("</OUTPUT>")

    tokenized_dpo = []
    with open(RAW_DPO, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                prompt_q = item.get("prompt", "").strip()
                ctx = item.get("input", "").strip()
                chosen = item.get("chosen", "").strip()
                rejected = item.get("rejected", "").strip()

                if not prompt_q or not chosen or not rejected:
                    continue

                full_prompt = f"{ctx}\nSoru: {prompt_q}".strip() if ctx else prompt_q

                p_ids = tokenizer.encode(full_prompt)
                c_ids = tokenizer.encode(chosen)
                r_ids = tokenizer.encode(rejected)

                # Strip BOS/EOS if already present
                if p_ids and p_ids[0] == bos_id: p_ids = p_ids[1:]
                if p_ids and p_ids[-1] == eos_id: p_ids = p_ids[:-1]
                if c_ids and c_ids[0] == bos_id: c_ids = c_ids[1:]
                if c_ids and c_ids[-1] == eos_id: c_ids = c_ids[:-1]
                if r_ids and r_ids[0] == bos_id: r_ids = r_ids[1:]
                if r_ids and r_ids[-1] == eos_id: r_ids = r_ids[:-1]

                prompt_ids = [bos_id, inst_start_id] + p_ids + [inst_end_id, output_start_id]
                chosen_ids = c_ids + [output_end_id, eos_id]
                rejected_ids = r_ids + [output_end_id, eos_id]

                rec = {
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids
                }
                tokenized_dpo.append(rec)
            except Exception:
                continue

    with open(OUT_DPO_TOK, "w", encoding="utf-8") as f:
        for rec in tokenized_dpo:
            f.write(json.dumps(rec) + "\n")
    print(f"  * {len(tokenized_dpo):,} adet tokenize edilmiş DPO kaydı '{OUT_DPO_TOK}' dosyasına yazıldı.")

    # Merge into dpo_all_tokenized.jsonl (if not already present)
    if os.path.exists(DPO_ALL_PATH):
        with open(DPO_ALL_PATH, "a", encoding="utf-8") as f:
            for rec in tokenized_dpo:
                f.write(json.dumps(rec) + "\n")
        print(f"  * DPO kayıtları ana kütüğe ('{DPO_ALL_PATH}') eklendi.")
    return tokenized_dpo


def index_rag_knowledge(sft_records: List[Dict[str, str]], max_cards: int = 500):
    print(f"\n[4/5] Qdrant 'simulasyon_bellek' Koleksiyonuna Tarih Bilgi Kartları İndeksleniyor (Hedef: {max_cards} kart)...")
    try:
        gateway = AgentGateway.create_default(device="cpu")
        injected = 0
        sample_records = sft_records[:max_cards] if len(sft_records) > max_cards else sft_records
        for idx, rec in enumerate(sample_records):
            inp = rec.get("input", "")
            q = rec.get("instruction", "")
            out = rec.get("output", "")
            card_text = f"Tarihsel Belge / Konu: {inp}\nSoru: {q}\nBilimsel Açıklama: {out}".strip()
            try:
                gateway.inject_knowledge(card_text, target_collection="simulasyon_bellek")
                injected += 1
                if (idx + 1) % 100 == 0 or idx == len(sample_records) - 1:
                    print(f"  * {idx + 1}/{len(sample_records)} bilgi kartı Qdrant belleğe işlendi.")
            except Exception as e:
                continue
        print(f"  * Toplam {injected} adet 1931 Tarih bilgi kartı başarıyla indekslendi.")
        gateway.close()
    except Exception as e:
        print(f"  * Uyarı: Qdrant indeksleme atlandı/hata: {e}")


def main():
    print("=" * 70)
    print(" KRİSTAL-VEKTÖREL: 1931 TÜRK TARİHİ BORU HATTI ENTEGRASYONU")
    print("=" * 70)

    # 1. Prepare SFT and Chat
    sft_records = prepare_sft()
    chat_records = prepare_chat()

    # 2. Setup Compiler & Tokenizer
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 3. Prepare DPO
    prepare_dpo(tokenizer, vocab)

    # 4. Check & expand vocabulary if new tokens exist
    print("\n[5/5] Yeni Morfem & Kök Taraması...")
    sample_records = (sft_records[:1000] + chat_records[:1000])
    discover_new_tokens_and_expand(sample_records, vocab_path="data/vocab.json", device="cpu", dry_run=False)

    # 5. Index into RAG Memory
    index_rag_knowledge(sft_records, max_cards=300)

    print("\n" + "=" * 70)
    print(" 1931 TÜRK TARİHİ VERİ ENTEGRASYONU TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
