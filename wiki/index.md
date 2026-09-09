---
tags: [index]
date: 2026-06-14
sources: [src/]
status: active
---

# Kristal-Vektörel Mimari Fihristi (Wiki Index)

Bu dizin, Türkçe ve bitişken diller için özel olarak geliştirilmiş morfem tabanlı deterministik dil modeli mimarisinin anlamsal hafızasını barındırır.

---

## 🗺️ Mimari Dizin Yapısı

### 1. Temel Bileşenler (Entities)
Modeli oluşturan ana kod sınıfları ve modülleri:
* [[kristal_lm]]: Causal Transformer Decoder modeli ve katman yapıları.
* [[kristal_tokenizer]]: KristalCompiler ile sözlüğü bağlayan tokenizasyon katmanı.
* [[vector_memory]]: Qdrant tabanlı hibrit geri çağırma (hybrid search) bellek modülü.
* [[rag_pipeline]]: Vektörel Gezgin'in uçtan uca arama-üretim (RAG) akış yöneticisi.
* [[chat_prompt]]: Komut satırı etkileşimli sohbet ve test aracı.

### 2. Kavramlar ve Algoritmalar (Concepts)
Sistemde kullanılan özgün matematiksel ve morfolojik konseptler:
* [[sign_inversion]]: Olumsuz anlam taşıyan kelimelerin morfem embedding vektörlerinin işaret terslemesi.
* [[causal_prompt_masking]]: SFT eğitiminde talimat ve girdi kısımlarının kayıp (loss) hesabından muaf tutulması.
* [[rope]]: Causal self-attention içinde göreceli konumsal ilişkilerin rotasyon matrisleriyle kodlanması.

### 3. Geliştirme ve Analiz Günlükleri
* [[log]]: Kronolojik geliştirme günlüğü.
* [[synthesis]]: Teknik borçlar (technical debt), model yanlılıkları ve çözüm önerileri analizi.
