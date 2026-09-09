---
tags: [concept]
date: 2026-06-14
sources: [scripts/train_step_demo.py]
status: active
---

# İşaret Terslemeli Embedding Mantığı (wiki/concepts/sign_inversion.md)

Bu konsept, bitişken dillerdeki olumsuzluk (negation) anlamını dil modelinin embedding katmanında doğrudan matematiksel bir vektör yön değişimi (sign inversion) olarak temsil etme yöntemidir.

---

## 📐 Çalışma Prensibi

Geleneksel LLM'ler olumsuzluk eklerini sadece istatistiksel yan yana geliş olasılığı olarak öğrenirken, `KristalLM` bu anlamı geometrik olarak temsil eder:
1. Kelime sınırları kontrol edilir.
2. Eğer bir kelimenin morfem dizilimi içinde `NEG` veya `IMPOTENTIAL_NEG` morfemlerinden biri varsa, o kelimenin **tüm morfem embedding vektörleri** $-1.0$ ile çarpılarak vektör uzayında tam ters yöne (inversion) çevrilir.
3. Bu sayede model, olumlu ve olumsuz kelimeleri vektör uzayında zıt koordinatlarda konumlandırır.

```python
# scripts/train_step_demo.py:KristalEmbedding.forward
if has_negation:
    embeddings[b, word_start_idx:seq_len] *= -1.0
```

---

## 🔗 İlgili Bileşenler
* [[kristal_lm]]: embedding katmanını barındıran ana model sınıfı.
