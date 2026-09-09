---
tags: [entity]
date: 2026-06-14
sources: [src/rag/vector_memory.py]
status: active
---

# VectorMemory Sınıfı (wiki/entities/vector_memory.md)

`VectorMemory`, Vektörel Gezgin (Vector Rover) prototipinin evrensel hafıza yönetimini ve bilgi okyanusundaki navigasyonunu kontrol eden Qdrant tabanlı RAG (Retrieval-Augmented Generation) bileşenidir.

---

## 📐 Hibrit Arama ve RRF (Reciprocal Rank Fusion)

Modül, anlam benzerliği (dense vector) ile morfolojik eşleşmeyi (sparse vector) bir arada değerlendiren bir hibrit arama mimarisi barındırır:
1. **Dense Search:** 768 boyutlu yoğun vektörler ile anlamsal (semantic cosine) yakınlık sorgusu yapar.
2. **Sparse Search:** TF-IDF tabanlı IDF modifikasyonu ile morfem/kelime düzeyinde kesin eşleşme sorgusu yapar.
3. **RRF:** İki sorgunun sonuçlarını Reciprocal Rank Fusion ile birleştirerek en iyi sıralamayı üretir.

---

## ⚖️ Morfolojik Kısıt Puanlaması (Token-Type Constraint)

Sorgudaki ayırt edici kökler (`distinctive_query_roots`) çıkarılarak arama sonuçlarındaki dokümanlarla karşılaştırılır. Eğer dokümandaki köklerin sorgudaki köklerle eşleşme oranı %50'nin altındaysa, dokümanın RRF skoru **%50 oranında cezalandırılır (penalty scoring)**. Bu sayede ilgisiz kelimelerin morfem benzerlikleri nedeniyle yukarı çıkması engellenir.

---

## 🔗 İlgili Dosyalar
* [src/rag/vector_memory.py](file:///Users/hakankilicaslan/Git/tr_llm/src/rag/vector_memory.py)
