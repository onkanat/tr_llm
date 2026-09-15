---
tags: [concept]
date: 2026-09-15
sources: [src/llm/prompt_contract.py, tests/test_prompt_contract.py]
status: active
---

# Kanonik İstem Sözleşmesi ve Çift BOS İzolasyonu (wiki/concepts/canonical_prompt_contract.md)

Bu konsept, modelin eğitim, değerlendirme ve çıkarım aşamalarında kullandığı girdi isteminin (prompt) tek bir kanonik zarfta standartlaştırılmasını ve çift başlangıç belirteci (`<BOS>`) hatasının önlenmesini tanımlar.

---

## 🎯 Amaç ve Ortaya Çıkış Gerekçesi

Eylül 2026 denetimlerinde (T-0013 ve T-0015), değerlendirme betiklerinin eğitim betiklerinden farklı olarak istem başına fazladan bir `<BOS>` belirteci eklediği saptanmıştır. Bu hata:
1. Değerlendirme sırasında istemin 1. token'dan itibaren eğitim bağlamından kopmasına neden olmakta,
2. Causal self-attention konumsal ilişkilerini bozarak model çıktılarında yapay bozulmalara yol açmaktaydı.

---

## 📐 Sözleşme Kuralları

1. **Tek Kanonik Giriş Kapısı:** İstem oluşturma işlemleri doğrudan dize birleştirme ile değil, [src/llm/prompt_contract.py](file:///Users/hakankilicaslan/Git/tr_llm/src/llm/prompt_contract.py) içindeki `render_prompt` fonksiyonu aracılığıyla gerçekleştirilir.
2. **Kanonik Zarf Standardı:** İstem yapısı her zaman aşağıdaki sırada ve tek `<BOS>` içerecek biçimde üretilir:
   `<BOS><INSTRUCTION> {instruction} <INPUT> {input_text} <OUTPUT>`
3. **Kanarya Testleri Koruması:** `tests/test_prompt_contract.py` altında tanımlanan kanarya testleri, eğitim ve değerlendirme kodlarında çift `<BOS>` üretilmesini sürekli olarak denetler.

---

## 📊 Ölçüm ve Raporlar

* İstem sözleşmesi sapması ve A/B kıyaslama sonuçları: `data/eval/exp_c_recheck_2026-09-15.json`
* Karşılaştırmalı çıkarım analizleri: `data/eval/t0027_model_state_2026-09-15.json`

---

## 🔗 İlgili Sayfalar
* [[causal_prompt_masking]]: İstem maskeleme prensibi.
* [[b1_5_split_contract]]: Veri bölme sözleşmesi.
* [[chat_prompt]]: Etkileşimli istem arayüzü.
