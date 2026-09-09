---
tags: [entity]
date: 2026-06-14
sources: [src/llm/tokenizer.py]
status: active
---

# KristalTokenizer Sınıfı (wiki/entities/kristal_tokenizer.md)

`KristalTokenizer`, düz Türkçe metinleri kelime kelime alıp `CrystalCompiler` üzerinden morfem tag akışlarına dönüştüren ve bunları sözlükteki (vocabulary) sayısal ID'lere eşleyen tokenizasyon katmanıdır.

---

## 🔑 Özel Kontrol Tokenları (Special Tokens)

SFT ve veri sarmalama işlemlerinde aşağıdaki özel kontrol belirteçleri kullanılır:
* `<UNK>` (ID: 0): Sözlük dışı morfemler.
* `<PAD>` (ID: 1): Padding.
* `<BOS>` (ID: 2): Cümle/Belge başlangıcı.
* `<EOS>` (ID: 3): Cümle/Belge sonu.
* `<INSTRUCTION>` / `</INSTRUCTION>` (ID: 4/5): Talimat sınırları.
* `<INPUT>` / `</INPUT>` (ID: 6/7): Görev girdisi.
* `<OUTPUT>` / `</OUTPUT>` (ID: 8/9): Görev çıktısı.
* `<NUMBER>` (ID: 10): Sayısal ifadeler.
* `<PROPER_NOUN>` (ID: 11): Özel isimler ve derleyici dışı büyük harfli etiketler.

```python
class KristalTokenizer:
    def encode(self, text: str) -> List[int]:
        # Metni morfemlere ayırıp ID dizisi döndürür
```

---

## 🔗 İlgili Dosyalar ve Kavramlar
* [src/llm/tokenizer.py](file:///Users/hakankilicaslan/Git/tr_llm/src/llm/tokenizer.py)
* [[causal_prompt_masking]]
