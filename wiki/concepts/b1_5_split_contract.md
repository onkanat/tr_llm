---
tags: [concept]
date: 2026-09-15
sources: [src/data/b1_5_splitter.py, scripts/prepare_b1_5_splits.py, data/eval/f0_identity_scope_2026-09-15.json]
status: active
---

# B1.5 Bölme Sözleşmesi ve Veri Sızıntısı İzolasyonu (wiki/concepts/b1_5_split_contract.md)

Bu konsept, modelin morfolojik genelleme yeteneğini ezberden (memorization) ayırt etmek amacıyla kurgulanan **B1.5 veri bölme sözleşmesini** ve cevap düzeyindeki sızıntıların önlenmesi mimarisini tanımlar.

---

## 🎯 Amaç ve Ortaya Çıkış Gerekçesi

Geleneksel veri bölme yaklaşımlarında, girdi talimatı farklı olsa dahi aynı morfolojik kök ve çekim kombinasyonunun hem eğitim (train) hem değerlendirme (val/test) kümelerinde yer alması durumunda model, dilbilgisel kuralları genellemek yerine hedef morfem dizisini ezberleyebilir.

T-0012 denetiminde tespit edildiği üzere:
1. Kapalı sınıf iskelet anahtarları ve ayrık kök sözleşmesi sıkı biçimde uygulanmadığında, test kümesindeki hedef morfem dizilerinin eğitim kümesinde birebir bulunması riski oluşmaktadır.
2. Bu durum değerlendirme metriklerini yapay olarak iyimser kılmakta, modelin gerçek OOV (sözlük dışı kök) veya görülmemiş kelime performansını maskelemektedir.

---

## 📐 Sözleşme Kuralları

1. **Ayrık Kök Bölmesi (Disjoint Lemma Split):** Eğitim kümesinde kullanılan ana kökler ile test/değerlendirme kümesindeki kökler kesin olarak ayrık olmalıdır.
2. **Cevap Düzeyinde Sızıntı Yasağı:** `<OUTPUT>` sonrasında üretilen morfem dizisinin tam kombinasyonu, test kümesinden bağımsız tutulmalıdır.
3. **Deterministik İskelet Doğrulaması:** Bölme işlemi rastgele çekirdeğe (random seed) terk edilmeksizin [src/data/b1_5_splitter.py](file:///Users/hakankilicaslan/Git/tr_llm/src/data/b1_5_splitter.py) üzerinden deterministik sözlük indekslerine göre ayrıştırılır.

---

## 📊 Ölçüm ve Raporlar

Bu sözleşmeye dair somut veri dağılımları, örtüşme oranları ve ayrıştırma envanteri kod tabanında sabitlenmiş ampirik raporlarda takip edilir:
* Ayrıntılı bölme kapsamı ve örtüşme analizleri: `data/eval/f0_identity_scope_2026-09-15.json`
* Checkpoint soykütüğü ve ezber testleri: `data/eval/t0018_why_no_memorization.json`

---

## 🔗 İlgili Sayfalar
* [[causal_prompt_masking]]: İstem maskeleme prensibi.
* [[canonical_prompt_contract]]: Kanonik zarf sözleşmesi.
* [[model_checkpoints]]: Checkpoint soykütüğü ve ağırlık durumları.
