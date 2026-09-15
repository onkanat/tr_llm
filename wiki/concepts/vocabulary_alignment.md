---
tags: [concept]
date: 2026-09-15
sources: [data/vocab.json, src/compiler/lexicon.py, data/eval/t0027_model_state_2026-09-15.json]
status: active
---

# Sözlük Hizalaması, Noktalama Bloğu ve Token Kayması (wiki/concepts/vocabulary_alignment.md)

Bu konsept, kök morfem sözlüğü (`roots.tsv`) ile dil modelinin kelime dağarcığı (`vocab.json`) arasındaki ilişkiyi, noktalama karakterlerinin sözlükteki konumunu ve bayat veri derlemelerinin yarattığı token kayması riskini inceler.

---

## 🎯 Kök Sözlüğü ile Vocab Ayrımı

* **Kök Morfem Sözlüğü (`roots.tsv`):** Kristal Derleyicisi'nin temel ontolojisidir; dildeki atomik kök morfemleri barındırır.
* **Model Kelime Dağarcığı (`vocab.json`):** Modelin girdi/çıktı katmanlarında (`wte`, `lm_head`) tensör karşılığı bulunan token haritasıdır. Kök morfemlerin yanı sıra özel kontrol tokenlarını (`<BOS>`, `<EOS>`, `<OUTPUT>`), dilbilgisel ek morfemlerini ve noktalama karakterlerini içerir.

---

## 🔣 Noktalama Bloğu Konumu

Eylül 2026 envanter ölçümlerinde doğrulandığı üzere:
1. Taban sözlük (`data/vocab.json`) noktalama karakterlerini içermemektedir.
2. Genişletilmiş modellerde noktalama bloğu tam olarak **32.137** indeksinde başlamaktadır (`.`, `,`, `?`, `!`, `-`, `:`, `;`, `(`, `)`).
3. Noktalama karakterlerinin taban modele mi yoksa uzantı sözlüğe mi dahil edileceği mimari kararı (D-A..D-E kararları), modelin genel Türkçe metin devam ettirme başarısını doğrudan belirler.

---

## ⚠️ Bayat İkili Dosya (`.bin`) Tehlikesi

Eğitim verileri (`.bin`) derlenirken kullanılan sözlük ile modeli çalıştıran sözlük arasında token ID uyuşmazlığı olması durumunda token kayması (token ID shift) meydana gelir. Bu nedenle veri derleme betikleri her zaman güncel `vocab.json` ile senkronize edilmelidir.

---

## 📊 Ölçüm ve Raporlar

* 23 modelin sözlük ve tensör boyut envanteri: `data/eval/t0027_model_state_2026-09-15.json`
* Model yeniden üretim ve sözlük birleştirme planı: `data/eval/regeneration_plan_2026-09-15.json`

---

## 🔗 İlgili Sayfalar
* [[kristal_tokenizer]]: Tokenizer mimarisi.
* [[state_dict_resizing]]: Ağırlık boyutlandırma kuralı.
* [[base_and_jacket_architecture]]: Taban ve ceket mimarisi.
