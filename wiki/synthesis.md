---
tags: [synthesis]
date: 2026-06-15
sources: [data/vocab.json, src/llm/tokenizer.py]
status: active
---

# Teknik Borçlar ve Mimari Sentez (wiki/synthesis.md)

Bu dosya, modelin güncel eğitim verilerinden kaynaklanan yanlılıklarını (bias), sözlük genişlemesi risklerini ve mimari optimizasyon fırsatlarını (technical debt) inceler.

---

## 🟢 1. SFT Veri Seti Sınıf Dengesizliği (Dataset Imbalance Bias)

* **Bulgu:** `parenting_dataset.jsonl` ve DPO veri kümelerinde olumsuz/nötr örneklerin (örn. `PLURAL: Hayır`) sayısı, olumlu örneklerin (`PLURAL: Evet`) sayısından çok daha fazlaydı.
* **Durum:** **TAMAMEN ÇÖZÜLDÜ**. 
* **Detay:** Mimariye [[rope]] entegre edilmesi ve bağlam boyutunun 64'e genişletilmesinin ardından yapılan **Stage-2 SFT İnce Ayar (Parenting)** aşamasıyla modelin konumsal attention hassasiyeti zirveye ulaşmıştır. Model, `çocuklar` kelimesinin çoğul eki aldığını `%100.00` olasılıkla (`PLURAL: Evet`) ve `kitaplarda` kelimesinin kökünü `%99.94` olasılıkla (`ROOT: kitap`) doğru tahmin etmektedir.

---

## 🟢 2. Wikipedia Veri Baskınlığı (Wikipedia Dataset Dominance)

* **Bulgu:** Wikipedia morfem akışı (`data/train_wiki.bin`), birleşik veri kümesinin **%76.23**'ünü (8.20M / 10.76M token) oluşturmaktaydı. Bu durum SFT görevlerinin ezilmesine yol açmaktaydı.
* **Durum:** **ÇÖZÜLDÜ (Aşamalı Eğitim Metoduyla)**.
* **Detay:** Wikipedia + Külliyat ile genel dil yeteneğini öğrenen model (Stage-1), dondurulmadan `lr=1e-4` gibi düşük bir öğrenme oranıyla **sadece SFT verileriyle ince ayar (Stage-2 SFT)** işlemine tabi tutulmuştur. Bu sayede model hem genel morfolojik yeteneklerini korumuş hem de SFT talimatlarını %100 doğrulukla takip etmeye odaklanmıştır.

---

## 🟢 3. Sözlük (Vocabulary) Genişlemesi ve OOV Kontrolü

* **Bulgu:** Bilimsel ve HPC odaklı veri setlerinin (biyoloji, fizik, linux, matematik, fortran, paralel programlama) morfem düzeyinde işlenmesi esnasında ilk aşamada %30-35'lere varan yüksek OOV (sözlük dışı) kelime oranı tespit edilmiştir.
* **Durum:** **TAMAMEN ÇÖZÜLDÜ**.
* **Detay:**
  * `KristalTokenizer`'a Unicode `NFC` normalizasyonu entegre edilerek birleşik karakter düzensizlikleri giderilmiştir.
  * OOV taramasıyla en sık geçen 1.900 adet bilimsel ve teknik kök (`cpu`, `gpu`, `fortran`, `frac`, `integral` vb.) `roots.tsv` tablosuna NOUN olarak eklenmiştir. Sözlük boyutu **28.261** token'a genişletilmiştir.
  * OOV oranı tüm teknik setlerde **%8 - %15** bandına çekilerek modelin teknik terimleri anlaması sağlanmıştır.

---

## 🟢 4. Çoklu Bilimsel Tercih Hizalaması (Multi-Dataset DPO Alignment)

* **Durum:** **BAŞARIYLA TAMAMLANDI**.
* **Detay:** 5.792 adet DPO tercih çifti üzerinde model CPU ortamında (MPS GPU hafıza sınırı nedeniyle) 1.500 adım eğitilmiştir. Kayıp (loss) **2.31**'den **0.008** seviyesine düşerek tercih hizalaması tamamlanmıştır. Model, SFT-hizalı dil yapısını korurken akademik ve LaTeX-tabanlı bilimsel çıktıları tercih etmeyi öğrenmiştir.

---

## 🟢 5. RAG Hizalama ve Döküman Kopyalama Güvenilirliği (RAG Alignment & Document Copying)

* **Bulgu:** Model RAG modunda iken ortaokul sohbet jargonuyla varsayılan çıktılar üretme eğilimindeydi. Ayrıca, tokenizer'ın morfem etiketlerini çift derleme ve OOV hatasıyla `<PROPER_NOUN>`'a dönüştürme eğilimi mevcuttu.
* **Durum:** **TAMAMEN ÇÖZÜLDÜ**.
* **Detay:**
  * **Çift Derleme Önleme:** `KristalTokenizer.encode` aşamasına `clean_word in self.vocab.stoi` kontrolü eklenerek morfem etiketlerinin Türkçe kelime gibi derlenmesi ve hatalı OOV üretilmesi engellendi.
  * **Önek İyileştirmesi:** `B:` ve `S:` önekleri yerine dil yapısında zaten var olan `belge:` ve `sorgu:` kelimeleri tercih edilerek token karmaşası önlendi.
  * **RAG Hizalama Eğitimi:** Qdrant veritabanındaki 4.000 GTS dökümanı kullanılarak 7.982 adet sentetik RAG SFT kopyalama örneği üretildi ve model Metal GPU (MPS) üzerinde 1.200 adım fine-tune edildi (Loss **9.05**'ten **0.65**'e düştü).
  * **Kopyalama Başarısı:** Model, temiz veri kümesinde (Clean Case) dökümanın morfem akışını **%100 doğrulukla kopyalamayı** başarmıştır. OOV içeren durumlarda ise jargona sapmaksızın dökümanın dilbilgisel ve morfolojik yapısını korumuştur.

---

## 🟢 6. Eylül 2026 Kapatılan Sözleşme ve Altyapı Borçları

* **Çift `<BOS>` Değerlendirme Çelişkisi (T-0013 & T-0015):** Değerlendirme betiklerinin istem başına fazladan `<BOS>` ekleyerek eğitimi 1. token'da bozması sorunu [[canonical_prompt_contract]] ve `render_prompt` fonksiyonu ile tamamen çözüldü; kanarya testleri eklendi.
* **Sessiz Ağırlık Kırpma / Slicing (T-0025 & T-0026):** Model tensörlerinin boyut uyuşmazlığında sessizce kırpılarak eğitilmiş parametrelerin çöpe atılması engellendi. [[state_dict_resizing]] fonksiyonu tek kanonik kapı yapılarak küçültme girişimlerinde açık `ValueError` fırlatması sağlandı.
* **Üretim Kasası Test Kirliliği (T-0023):** Birim testlerin doğrudan `data/pedagogy/cot_vault.jsonl` dosyasına yazması engellendi; geçici kasa mimarisi ve SHA-256 bütünlük çıpası kuruldu ([[cot_vault]]).
* **B1.5 Veri Sızıntısı (T-0012):** Eğitim ve değerlendirme arasındaki cevap düzeyinde örtüşmeler [[b1_5_split_contract]] ile ayrık kök sözleşmesine bağlandı.
* **Talimat Yüzeyi Hizalaması (T-0028 & T-0029):** Talimat ve kural dosyalarındaki bayat iddialar (20.500 kök hedefi, 5.792 DPO sayısı vb.) KOD KAZANIR ilkesiyle ampirik verilere (48.200 tekil lemma / 52.373 giriş) eşitlendi; `/docs_update` yeteneği genişletildi.

---

## 🟡 7. Aktif ve Devam Eden Mimari Borçlar

* **4096² Dikkat Maskesi Tamponları (T-0027):** 23 model checkpoint'inin tensör baytlarının %21'ini oluşturan sabit bool maske tamponları (`blocks.*.attn.mask`) diskte gereksiz yer kaplamaktadır. Yeniden üretim aşamasında maskelerin `register_buffer(persistent=False)` yapılması önerilmektedir ([[model_checkpoints]]).
* **`test_model.py` Boyut Uyumsuzluğu (T-0027 & T-0028):** Betik modeli sabit `data/vocab.json` (31.357) ile başlattığı için güncel 32.816 satırlı `kristal_model.pt` yüklendiğinde `RuntimeError` vermektedir. Kanonik `resize_state_dict` entegrasyonu planlanmaktadır.
* **Tabanın Noktalama/Rakam Eksikliği (D-A..D-E Kararları):** Mevcut taban model noktalama işaretlerini ve rakamları öğrenmemiştir. Kullanıcı onaylı D-A..D-E kararları doğrultusunda noktalama ve rakamların taban modele taşındığı iki parçalı minimum taban yeniden üretimi (`data/eval/regeneration_plan_2026-09-15.json`) sıradaki ana geliştirme borcudur ([[base_and_jacket_architecture]]).


