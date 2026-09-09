---
tags: [entity]
date: 2026-06-14
sources: [scripts/train_step_demo.py, train.py]
status: active
---

# KristalLM Sınıfı (wiki/entities/kristal_lm.md)

`KristalLM`, bitişken dillerin morfem yapılarını işlemek üzere tasarlanmış, Causal Self-Attention (Causal Transformer Decoder) tabanlı ana dil modelidir.

---

## 🏛️ Katman Yapısı ve Parametreler

* **Embedding Katmanı:** [[sign_inversion]] mantığı ile çalışan özel `KristalEmbedding` sınıfı. Mutlak konum embedding'i (`pos_embedding`) kaldırılmıştır; relative konumsal ilişkiler attention içinde [[rope]] ile çözülmektedir.
* **Attention Katmanı:** PyTorch `nn.MultiheadAttention` yerine, Q, K, V projeksiyonlarını ve attention matris çarpımını RoPE ile birleştiren özel `CausalSelfAttention` ve `RotaryEmbedding` sınıfları kullanılmaktadır.
* **Katman Sayısı (`n_layer`):** 4 adet Transformer bloğu.
* **Dikkat Başlığı Sayısı (`n_head`):** 4 adet kafa (Multihead Attention).
* **Gizli Boyut (`n_embd`):** 768 boyut.
* **Blok Boyutu (`block_size`):** 64 (Eğitim bağlam boyutu) veya 1024 (Model sınırı).

```python
class KristalLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, vocab: Vocabulary, n_layer=4, n_head=4):
        # Mimarinin inşası (RoPE ve CausalSelfAttention ile)
```

---

## 🔗 İlgili Dosyalar ve Kavramlar
* [scripts/train_step_demo.py](file:///Users/hakankilicaslan/Git/tr_llm/scripts/train_step_demo.py#L145-L184)
* [train.py](file:///Users/hakankilicaslan/Git/tr_llm/train.py)
* [[sign_inversion]]
* [[rope]]
