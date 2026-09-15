---
tags: [concept]
date: 2026-09-15
sources: [data/eval/regeneration_plan_2026-09-15.json, data/eval/regeneration_recon_2026-09-15.json]
status: active
---

# Taban Değişmez, Ceket Değişken Mimarisi (wiki/concepts/base_and_jacket_architecture.md)

Bu konsept, modelin genel Türkçe dilbilgisi ve morfoloji yeteneğini temsil eden **değişmez taban model** ile alana özgü uzmanlıkları temsil eden **değişken ceket** katmanlarını tanımlar.

---

## 🏛️ Mimari Paradigma: İki Parçalı Minimum

Kullanıcı tarafından onaylanan D-A..D-E mimari kararları doğrultusunda, modelin tek aşamalı monolitik bir yapı yerine iki parçalı olarak inşa edilmesi kararlaştırılmıştır:

1. **Değişmez Taban Model (Base Model):**
   * Türkçe dilbilgisi, morfolojik analiz, heceleme ve genel söz dizimini öğrenir.
   * Tarihsel sınırlılıkların aksine noktalama işaretleri (`.`, `,`, `?` vb.), rakamlar ve kesme işareti doğrudan taban modelin eğitim kümesine dahil edilir.
   * Taban model eğitildikten sonra dondurulur ve genel dil referansı olarak korunur.

2. **Değişken Ceket Modeli (Jacket / Specialization):**
   * Taban modelin üzerine giydirilen parametre verimli veya hafif fine-tuning katmanıdır.
   * Zanaat terminolojisi (marangozluk, teknik üretim) ve diyalog stilleri ceket aşamasında modele kazandırılır.

---

## 📋 D-A..D-E Karar Matrisi

Projenin geleceğe dönük yeniden üretim yol haritası şu kararlarla kesinleşmiştir:
* **D-A (Sözlük Çıkış Noktası):** Genişletilmiş güncel sözlük mimarisinin esas alınması.
* **D-B (Noktalama & Rakam):** Noktalama ve rakamların taban modele taşınması.
* **D-C (B1.5 Bölmesi):** Ayrık iskelet sözleşmesiyle eğitimin yürütülmesi.
* **D-D (Bağlam Boyutu):** Model bağlam uzunluğunun optimize edilmesi.
* **D-E (İki Parçalı Yapı):** Taban-ceket ayrımının nihai mimari olarak benimsenmesi.

---

## 📊 Ölçüm ve Raporlar

* Yeniden üretim yol haritası ve karar detayları: `data/eval/regeneration_plan_2026-09-15.json`
* Ön araştırma ve keşif analizleri: `data/eval/regeneration_recon_2026-09-15.json`

---

## 🔗 İlgili Sayfalar
* [[vocabulary_alignment]]: Sözlük ve noktalama entegrasyonu.
* [[model_checkpoints]]: Checkpoint taksonomisi.
