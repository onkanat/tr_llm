---
tags: [concept]
date: 2026-06-14
sources: [train.py]
status: active
---

# İstem Maskeleme (Causal Prompt Masking) (wiki/concepts/causal_prompt_masking.md)

Bu konsept, SFT (Supervised Fine-Tuning) eğitiminde modelin istem ve girdi kısımlarını ezberlemesini önleyerek, yalnızca çıktı (cevap) kısmındaki kayba (loss) odaklanmasını sağlayan maskeleme mekanizmasıdır.

---

## 📐 Çalışma Prensibi

1. Token akışında `<OUTPUT>` ve `</OUTPUT>` kontrol etiketleri aranır.
2. `<OUTPUT>` öncesinde yer alan tüm istem (`<INSTRUCTION>`) ve girdi (`<INPUT>`) tokenlarının hedef dizideki (`targets`) değerleri `-100` olarak değiştirilir (PyTorch CrossEntropyLoss fonksiyonunda `-100` varsayılan olarak göz ardı edilen indekstir).
3. Böylece model, istemi üretmek için gradyan güncellemesi yapmaz; sadece istem verildiğinde cevabı üretmeye zorlanır.

```python
# train.py
if not is_output:
    targets[b, i] = -100
```

---

## 🔗 İlgili Bileşenler
* [[kristal_lm]]: eğitim döngüsünün uygulandığı ana sınıf.
