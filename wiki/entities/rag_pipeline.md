---
tags: [entity]
date: 2026-06-14
sources: [src/rag/rag_pipeline.py]
status: active
---

# RAG Pipeline (wiki/entities/rag_pipeline.md)

`rag_pipeline.py`, Qdrant vektörel belleği (`VectorMemory`) ile Causal Transformer dil modelini (`KristalLM`) uçtan uca bir arama-üretim (Retrieval-Augmented Generation) akışında birleştiren ana yürütücü betiktir.

---

## 📐 İş Akışı ve Fonksiyonlar

1. **Girdi Tokenizasyonu:** Sorgu kelimesi `KristalTokenizer` ile tokenize edilerek morfem akışı elde edilir.
2. **Vektör Üretimi:** Sorgunun yoğun ve seyrek vektörleri `generate_kristal_vector` ve `generate_sparse_vector` fonksiyonlarıyla hesaplanır.
3. **Retrieval (Geri Çağırma):** Qdrant üzerinden hibrit arama (RRF + morfolojik kısıt filtrelemesi) yapılarak en uyumlu 1 belge geri çağrılır.
4. **Kompakt Prompt İnşası:** Kompakt bir JSON şablonu oluşturulur:
   `belge: <belge_morfemleri> sorgu: <sorgu_morfemleri>`
5. **Autoregressive Greedy Decoding:** Model, `<OUTPUT>` etiketinden başlayarak sırayla sonraki morfemleri tahmin eder ve `<EOS>` sınırına ulaşana kadar yanıtı üretir.


---

## 🔗 İlgili Dosyalar
* [src/rag/rag_pipeline.py](file:///Users/hakankilicaslan/Git/tr_llm/src/rag/rag_pipeline.py)
* [[vector_memory]]
* [[kristal_lm]]
