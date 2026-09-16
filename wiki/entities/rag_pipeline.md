---
tags: [entity]
date: 2026-09-15
sources: [src/rag/rag_pipeline.py]
sources_sha256:
  src/rag/rag_pipeline.py: af796b5fec837d098e6d9ed0b25f33293f9058dbd55206fc9bc4a7ec968b3b6f
status: active
---

# RAG Pipeline (wiki/entities/rag_pipeline.md)

`rag_pipeline.py`, Qdrant vektörel belleği (`VectorMemory`) ile Causal Transformer dil modelini (`KristalLM`) uçtan uca tek kanonik arama-koşullama-üretim (Retrieval-Augmented Generation) hattında birleştiren ana yürütücü ve servis dikişidir.

---

## 📐 İş Akışı ve Fonksiyonlar

1. **Girdi Tokenizasyonu ve Vektör Üretimi (`build_query_vectors`):** Sorgu kelimesi `KristalTokenizer` ile tokenize edilerek morfem akışı elde edilir; yoğun (dense) ve seyrek (sparse) vektörleri üretilir.
2. **Retrieval (Geri Çağırma, `retrieve_context`):** Qdrant (`VectorMemory`) üzerinden morfolojik kısıtlı hibrit arama yapılarak en uyumlu belge ve eşleşme skoru (`match_score`) temin edilir.
3. **Eşik Kapısı (`is_context_usable`):** Geri çağrılan belgenin eşleşme skoru `RAG_MATCH_THRESHOLD` (0.40) sınırını geçip geçmediği denetlenir (`match_score >= 0.40`). Eşik altındaysa model halüsinasyonu ve gürültüyü önlemek amacıyla koşullama yapılmaz (`conditioned = False`).
4. **Kanonik Prompt İnşası (`build_rag_prompt_tokens`):** Koşullama geçerliyse `build_rag_input` sözleşmesine uygun biçimde girdi düzenlenir:
   `<BELGE> {doc_text} </BELGE> {query}`
   Eşik altındaysa yalnızca yalın `{query}` kullanılır. JSON zarfı tokenize edilir ve `<OUTPUT>` belirtecinde kesilerek model girdisi hazırlanır.
5. **Autoregressive Greedy Decoding (`greedy_decode`):** Dil modeli, `<OUTPUT>` sınırından başlayarak deterministik argmax adımlarıyla sonraki morfemleri tahmin eder; tekrarlama cezası sliding window üzerinde uygulanır ve `<EOS>` belirtecine ulaşıldığında üretim sonlandırılır.

---

## 🔗 İlgili Dosyalar
* [src/rag/rag_pipeline.py](file:///Users/hakankilicaslan/Git/tr_llm/src/rag/rag_pipeline.py) (sha256: `af796b5fec837d098e6d9ed0b25f33293f9058dbd55206fc9bc4a7ec968b3b6f`)
* [src/rag/epistemic_agent.py](file:///Users/hakankilicaslan/Git/tr_llm/src/rag/epistemic_agent.py)
* [tests/test_rag_pipeline.py](file:///Users/hakankilicaslan/Git/tr_llm/tests/test_rag_pipeline.py)
* [[vector_memory]]
* [[kristal_lm]]
