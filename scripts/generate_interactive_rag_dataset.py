#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: İNTERAKTİF SELF-RAG EĞİTİM VERİ ÜRETİCİ
================================================================
Bu betik, KristalLM dil modelinin kendi RAG (Vektörel Bellek) sistemini
otonom olarak kullanmasını öğreten pedagojik eğitim verisi üretir:

1. Sorgu Üretme (Query Generation): Bilgi gerektiren sorularda <ARA> ... </ARA> üretimi.
2. Belge Destekli Yanıtlama (Grounded Generation): <BELGE> ... </BELGE> metnini kullanarak doğru yanıt verme.
3. Bilgi Yok İtirafı (Abstention): Alakasız belgede halüsinasyon görmeme ("Belgelerimde bu bilgi yok").
4. Sohbet / RAG Ayrımı: Günlük diyaloglarda gereksiz RAG çağırmama.
"""

import os
import sys
import json
import random
from typing import List, Dict

def load_gts_samples(max_samples: int = 1500) -> List[Dict[str, str]]:
    """GTS sözlüğünden gerçek madde ve tanımları yükler."""
    samples = []
    gts_path = 'data/poems/gts.json'
    if not os.path.exists(gts_path):
        return samples

    with open(gts_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                madde = item.get("madde", "").strip()
                if not madde or len(madde) < 3 or " " in madde:
                    continue
                anlamlar = item.get("anlamlarListe", [])
                for a in anlamlar:
                    anlam = a.get("anlam", "").strip()
                    if anlam and len(anlam) >= 15 and not anlam.startswith("bkz."):
                        samples.append({"madde": madde, "tanim": anlam})
                        break
                if len(samples) >= max_samples:
                    break
            except Exception:
                continue
    return samples

def load_carpenter_samples() -> List[Dict[str, str]]:
    """Marangozluk alan uzmanlığı verilerini yükler."""
    samples = []
    c_path = 'data/pedagogy/carpenter_specialization_dataset.jsonl'
    if not os.path.exists(c_path):
        return samples

    with open(c_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                inp = item.get("input", "").strip()
                out = item.get("output", "").strip()
                if inp and out:
                    samples.append({"konu": inp, "bilgi": out})
            except Exception:
                continue
    return samples

def main():
    print("=" * 65)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: İNTERAKTİF SELF-RAG EĞİTİM VERİSİ")
    print("=" * 65)

    gts_items = load_gts_samples(2000)
    carpenter_items = load_carpenter_samples()
    print(f"Kaynak Veriler: {len(gts_items)} GTS maddesi, {len(carpenter_items)} marangozluk bilgisi yüklendi.")

    dataset: List[Dict[str, str]] = []

    # =========================================================================
    # GÖREV 1: SORGU ÜRETME / ARAMA ÇAĞRISI (<ARA> ... </ARA>)
    # =========================================================================
    print("[1] Otonom RAG Sorgusu Üretme Görevleri Oluşturuluyor...")
    
    # 1.A. GTS Kelime Anlamı Sorguları
    for item in gts_items:
        m = item["madde"]
        dataset.append({
            "instruction": "Kullanıcının sorusunu yanıtlamak için vektörel bellek arama sorgusu üret.",
            "input": f"{m} ne demektir?",
            "output": f"<ARA> {m} sözlük tanım anlam </ARA>"
        })
        dataset.append({
            "instruction": "Hafızadaki bilgileri taramak için arama belirteçlerini oluştur.",
            "input": f"{m} kelimesinin anlamını açıklar mısın?",
            "output": f"<ARA> {m} tanım </ARA>"
        })

    # 1.B. Alan Uzmanlığı (Marangozluk) Sorguları
    for item in carpenter_items:
        konu = item["konu"]
        if len(konu.split()) <= 6:
            dataset.append({
                "instruction": "Teknik soru için vektörel belleğe gönderilecek arama sorgusunu oluştur.",
                "input": f"{konu} hakkında bilgi verir misin?",
                "output": f"<ARA> {konu} ahşap marangozluk teknik </ARA>"
            })

    # 1.C. Doğa ve Bilim Merak Sorguları
    science_queries = [
        ("Ağaçlar kışın neden yaprak döker?", "<ARA> ağaç kış yaprak dökme neden </ARA>"),
        ("Gökyüzü neden mavidir?", "<ARA> gökyüzü mavi ışık saçılma atmosfer </ARA>"),
        ("Güneş neden sıcaktır?", "<ARA> güneş füzyon çekirdek ısı enerji </ARA>"),
        ("Bal arıları nasıl bal yapar?", "<ARA> arı bal nektar kovan </ARA>"),
        ("Deniz suyu neden tuzludur?", "<ARA> deniz okyanus tuz mineral akarsu </ARA>"),
        ("Yapay zeka nasıl öğrenir?", "<ARA> yapay zeka makine öğrenimi algoritma </ARA>"),
        ("Kitap okumak zihni nasıl geliştirir?", "<ARA> kitap okuma beyin zihin odaklanma </ARA>"),
        ("Doğa yürüyüşü yapmanın faydaları nelerdir?", "<ARA> doğa yürüyüş sağlık zindelik </ARA>")
    ]
    for q, a in science_queries:
        dataset.append({
            "instruction": "Soruyu yanıtlamak üzere vektörel bellekte aranacak terimleri üret.",
            "input": q,
            "output": a
        })

    print(f"  -> Toplam Sorgu Üretme Örneği: {len(dataset):,}")

    # =========================================================================
    # GÖREV 2: BELGE DESTEKLİ ÇIKARIM VE YANITLAMA (<BELGE> ... </BELGE>)
    # =========================================================================
    print("[2] Belge Destekli Çıkarım (Grounded Q&A) Görevleri Oluşturuluyor...")
    count_g2 = 0
    for item in gts_items[:1000]:
        m = item["madde"]
        t = item["tanim"]
        dataset.append({
            "instruction": "Verilen belgedeki bilgileri kullanarak soruyu yanıtla.",
            "input": f"<BELGE> {m}: {t} </BELGE> {m} ne anlama gelmektedir?",
            "output": f"{m}, {t.lower()} anlamına gelmektedir."
        })
        count_g2 += 1

    for item in carpenter_items:
        k = item["konu"]
        b = item["bilgi"]
        dataset.append({
            "instruction": "Hafıza belgesine dayanarak teknik soruyu yanıtla.",
            "input": f"<BELGE> {b} </BELGE> {k} hakkında ne söylenebilir?",
            "output": b
        })
        count_g2 += 1
    print(f"  -> Toplam Belge Destekli Çıkarım Örneği: {count_g2:,}")

    # =========================================================================
    # GÖREV 3: BELLEKTE BİLGİ OLMADIĞINDA DÜRÜSTÇE İTİRAF ETME (ABSTENTION)
    # =========================================================================
    print("[3] Dürüstlük ve Bilgi Yetersizliği İtirafı Görevleri Oluşturuluyor...")
    count_g3 = 0
    random.seed(42)
    shuffled_gts = list(gts_items[:500])
    random.shuffle(shuffled_gts)

    for i in range(len(shuffled_gts) - 1):
        item_a = shuffled_gts[i]
        item_b = shuffled_gts[i + 1]
        
        # Doc is about A, question is about B
        dataset.append({
            "instruction": "Verilen belgeyi incele ve belgede soruya cevap yoksa dürüstçe belirt.",
            "input": f"<BELGE> {item_a['madde']}: {item_a['tanim']} </BELGE> {item_b['madde']} nedir?",
            "output": f"Sağlanan hafıza belgesinde '{item_b['madde']}' hakkında bilgi yer almamaktadır."
        })
        count_g3 += 1
    print(f"  -> Toplam Bilgi Yok İtirafı Örneği: {count_g3:,}")

    # =========================================================================
    # GÖREV 4: DOĞRUDAN SOHBET (RAG ÇAĞIRMADAN DOĞAL DİYALOG)
    # =========================================================================
    print("[4] RAG İhtiyacı Olmayan Doğal Sohbet Görevleri Ekleniyor...")
    chat_path = 'data/pedagogy/chat_conversations.jsonl'
    if os.path.exists(chat_path):
        with open(chat_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        random.seed(42)
        sample_chat = random.sample(lines, min(1000, len(lines)))
        for cl in sample_chat:
            dataset.append(json.loads(cl))

    random.seed(42)
    random.shuffle(dataset)

    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'rag_interactive_dataset.jsonl')

    with open(out_path, 'w', encoding='utf-8') as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print("\n" + "=" * 65)
    print(" İNTERAKTİF SELF-RAG VERİ KÜMESİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 65)
    print(f"Toplam Üretilen RAG Kayıt Sayısı : {len(dataset):,}")
    print(f"Kayıt Dosyası                     : {out_path}")
    print("=" * 65)

if __name__ == '__main__':
    main()
