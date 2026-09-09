---
tags: [concept]
date: 2026-06-14
sources: [scripts/train_step_demo.py]
status: active
---

# Döner Konumsal Kodlama (Rotary Position Embeddings - RoPE) (wiki/concepts/rope.md)

Döner Konumsal Kodlama (RoPE), modelin kelimeler/morfemler arasındaki göreceli mesafeleri (relative positions) self-attention katmanında rotasyon matrisleri aracılığıyla doğrudan sorgu (Query) ve anahtar (Key) vektörlerine kodlamasını sağlayan modern bir konumsal kodlama yöntemidir.

---

## 📐 Matematiksel Tanım ve Uygulama

Mutlak konumsal kodlama (`pos_embedding`) yerine, sorgu ve anahtar vektörleri head boyutu bazında ikişerli koordinatlara ayrılır ve konumsal frekanslara göre rotasyona uğratılır:

$$q_{rotated} = (q \times \cos) + (\text{rotate\_half}(q) \times \sin)$$
$$k_{rotated} = (k \times \cos) + (\text{rotate\_half}(k) \times \sin)$$

### Kod Uygulaması (`scripts/train_step_demo.py`)
```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

---

## 🏆 Mimari Kazanımlar

1. **Bağlam Penceresi (Context Window) Esnekliği:** Model, sabit bir mutlak pozisyon sınırı olmaksızın göreceli konumları öğrenebildiği için bağlam penceresi genişlemelerine (32'den 64'e) mükemmel uyum sağlar.
2. **Kayıp Değerinde Düşüş:** RoPE entegrasyonu sonrasında modelin bitiş kaybı (loss) 4.189'dan **3.069** seviyesine (yaklaşık 1.1 puan) düşmüştür.
3. **Semantik Hizalama:** Attention matrisindeki konumsal ilişkilerin daha doğal kurulması sayesinde, SFT talimatlarındaki negatif örnek yanlılığı (PLURAL bias) çözülmüştür.

---

## 🔗 İlgili Sınıflar
* [[kristal_lm]]: Model mimarisi sınıfı.
