# Skill 7: Pedagoji ve Uzmanlaşma (Pedagogy & Specialization) - Çatı

Bu doküman, Türkçe Kristal-Vektörel Mimarisi'nin insan öğreniminden ilham alan 3 aşamalı **Pedagojik Eğitim Müfredatını**, **Külliyat Konsolidasyonunu**, **Diyalog & Self-RAG Becerilerini** ve **Değerlendirme/Hizalama Süreçlerini** detaylandırır.

---

## 🏫 Üç Fazlı Pedagojik Yapı ve Külliyat Genişlemesi

Model, devasa veriyi tek aşamada ezberlemek yerine, kademeli bir pedagojik müfredat üzerinden eğitilir:

### 1. Faz 1: Bebeklik (Infancy) - Anlamsal Sınırlar
*   **Amaç:** Kavramların yasal sınırlarını pozitif ve negatif zıtlık örnekleriyle belirlemek.
*   **Hacim:** 1.000 örnek (`data/pedagogy/infancy_dataset.jsonl`).
*   **Örnek:** `Top: Pozitif: Top seker. Negatif: *Top içilir.`

### 2. Faz 2: Ebeveynlik (Parenting) - Derin Morfolojik ve Anlamsal Görevler
*   **Derinleştirilmiş 10 Morfolojik Görev:** Kök bulma, çoğul tespiti, hâl tespiti, zaman kipi, çatı dönüşümleri, olumsuzluk, soru eki, türetim vb. (30.000 örnek, `scripts/deepen_parenting_dataset.py`).
*   **TDK GTS Anlamsal Kök Sözlüğü:** 20.500 temel kökün sözlük tanımları ve semantik alanları (`scripts/generate_lexical_semantics_dataset.py`).

### 3. Faz 3: Sosyalleşme (Socialization) - Zanaat, Diyalog ve Otonom RAG
*   **Alan Uzmanlığı (Marangozluk & Zanaat):** 10.000 örnek ahşap işleme terminolojisi ve zanaat mantığı (`scripts/generate_specialization_dataset.py`).
*   **Doğal Türkçe Sohbet & Diyalog:** 6.000 çok turlu doğal konuşma örneği (`scripts/generate_deep_chat_dataset.py`).
*   **Otonom Self-RAG Etkileşimi:** Modelin bilgi açığını algılayıp `<ARA>` ve `<BELGE>` etiketleriyle çalıştığı 6.599 interaktif senaryo (`scripts/generate_interactive_rag_dataset.py`).

---

## 📊 Külliyat Dağılımı ve Eğitim Paketleri

| Veri Kümesi / Görev | Dosya Adı | Örnek Sayısı |
| :--- | :--- | :--- |
| **Bebeklik (Infancy)** | `infancy_dataset.jsonl` | 1.000 |
| **Ebeveynlik Derin (Parenting Deep)** | `parenting_deep_dataset.jsonl` | 30.000 |
| **TDK GTS Anlamsal Sözlük** | `lexical_semantics_dataset.jsonl` | 20.500 |
| **Marangozluk Uzmanlığı** | `carpenter_specialization_dataset.jsonl` | 10.000 |
| **Doğal Türkçe Sohbet** | `chat_conversations.jsonl` | 6.000 |
| **Otonom Self-RAG** | `rag_interactive_dataset.jsonl` | 6.599 |
| **DPO Tercih Çiftleri** | `dpo_tokenized.jsonl` | 5.792 |
| **TOPLAM KONSOLİDE SFT** | `data/train_deep_sft.bin` | **87.629** |

---

## ⚡ İnce Ayar, Hizalama ve Değerlendirme

### 1. İstem Maskeleme (Causal Prompt Masking)
Eğitimde modelin istemi ezberleyerek gradyan harcamasını önlemek için `<OUTPUT>` öncesindeki tüm tokenlar `-100` olarak maskelenir. Model yalnızca çıktı üretim kaybından sorumlu tutulur.

### 2. DPO Tercih Hizalaması (`train_dpo.py`)
5.792 adet tercih çifti (chosen vs. rejected) üzerinde modelin kaba veya halüsinatif ifadeler yerine akademik/zanaatkar ve doğru morfolojik yanıtları tercih etmesi sağlanır.

### 3. Değerlendirme ve Benchmark Motoru
*   **Sohbet Kalitesi Testi (`scripts/evaluate_chat.py`):** Selamlaşma, mantık, zanaat ve genel Türkçe sohbet kalitesini otomatik puanlar.
*   **Self-RAG İnteraktif Testi (`scripts/test_rag_interactive.py`):** Modelin otonom sorgu emisyonu ve belge sentez yeteneğini doğrular.
*   **Külliyat İçi Sınama (`scripts/evaluate_on_train_data.py`):** Modelin temel pedagojik görevlerdeki başarı oranını ölçer:
    *   **Morfoloji Başarısı:** %100
    *   **Marangozluk Uzmanlığı:** %80
    *   **Eğitim Kaybı (Final Loss):** 0.6601
    *   **Perplexity (PPL):** 2.40
