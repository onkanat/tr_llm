---
tags: [entity]
date: 2026-06-15
sources: [chat_prompt.py]
status: active
---

# Etkileşimli Sohbet Arayüzü (wiki/entities/chat_prompt.md)

`chat_prompt.py` betiği, eğitilmiş `KristalLM` dil modelinin ağırlıklarını (`data/kristal_model.pt`) kullanarak komut satırından (CLI) gerçek zamanlı çıkarım (inference) ve test yapılmasını sağlayan etkileşimli bir araçtır.

---

## 🛠️ Temel Özellikler

1. **Çift Mod Desteği (Multi-Mode):**
   * **SFT Modu (Default):** Modelin ebeveynlik fazında öğrendiği talimat takip etme yeteneklerini sınar (kök bulma, hâl tespiti, çoğul tespiti).
   * **Normal Mod:** Serbest metin üretimi ve morfem tamamlama görevlerini test eder.
2. **Renkli Morfem Görselleştirmesi:**
   * Morfemler kategorilerine göre (kökler, ekler, kontrol etiketleri) ANSI renk kodlarıyla renklendirilir.
3. **Gerçek Zamanlı Çıkarım:**
   * Metal GPU (MPS) desteği sayesinde Apple donanımlarında hızlı çalışır.
   * Modelin tahmin ettiği morfemler anlık olarak ekrana akıtılır (streaming effect).
4. **Parametre Ayarı:**
   * Çıkarım esnasında sıcaklık (`temperature`) ve `top_k` örnekleme değerleri dinamik olarak güncellenebilir.

---

## 🚀 Kullanım Şekli

Betiği çalıştırmak için aşağıdaki komut kullanılır:

```bash
venv/bin/python chat_prompt.py
```

### Örnek SFT Çıkarımı:

```text
[Mod: SFT | Temp: 0.0 | Top-K: 5]
Lütfen SFT Görevi Seçin veya Kendi Talimatınızı Girin:
  1. Kelimedeki kök morfemini bul.
  2. Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.
  3. Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.
  4. Kelimedeki eylemin zamanını veya kipini tespit et.
  5. Özel Talimat Gir...
Seçiminiz (1-5) veya Komut: 1
Analiz Edilecek Kelimeyi Girin: kitaplarda

 Girdi Morfem Akışı:
<BOS> <INSTRUCTION> <PROPER_NOUN> kök morfem POSS_2SG CASE_ACC bul </INSTRUCTION> <INPUT> kitap PLURAL CASE_LOC </INPUT> <OUTPUT>

Model Çıktısı:
 <PROPER_NOUN> kitap <EOS>
```

---

## 🏫 Ortaokul Samimi Sohbet Desteği

Model, **Pedagojik İnce Ayar Fazı** sonrasında ortaokul düzeyinde samimi bir "kanka" jargonuyla sohbet edebilme yeteneği kazanmıştır. Bu özelliği test etmek için:

1. CLI üzerinde `mode` yazarak **NORMAL** moda geçiş yapın.
2. `Selam kanka naber?` veya `En sevdiğin ders hangisi?` gibi samimi sorular yöneltin.
3. Model, morfem seviyesinde ürettiği çıktıyı `MorphemeDecompiler` yardımıyla anlamlı Türkçe yüzey formuna çevirecektir (Örn: *"ben okuldan yeni geldim ya sen kanka"*).

---

## 🔗 İlgili Sayfalar
* [[kristal_lm]]: Modelin genel mimarisi.
* [[kristal_tokenizer]]: İstemlerin morfem bazında işlenmesi.
* [[causal_prompt_masking]]: SFT eğitimindeki maskeleme kuralları.
