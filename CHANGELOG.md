# Değişiklik Günlüğü (Changelog)

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenmektedir.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning (SemVer)](https://semver.org/spec/v2.0.0.html) ilkelerini benimser.

---

## [1.2.0] - 2026-09-09

### Eklendi (Added)
- **Otonom Self-RAG Pedagojik Eğitimi:**
  - Dil modelinin bilgi açığını fark ettiğinde otonom arama sorgusu tetiklemesi ve dönen belgeleri entegre etmesi için `<ARA>`, `</ARA>`, `<BELGE>`, `</BELGE>` kontrol belirteçleri (control tokens) sisteme kazandırıldı.
  - `scripts/generate_interactive_rag_dataset.py`: 6.599 interaktif Self-RAG diyalog örneği üreten sentetik veri hattı eklendi.
  - `scripts/test_rag_interactive.py`: Modelin otonom sorgu üretimi ve belge sentezleme yeteneklerini ölçen etkileşimli test betiği eklendi.
- **Doğal Türkçe Sohbet & Diyalog Külliyatı:**
  - `scripts/generate_deep_chat_dataset.py`: 6.000 yüksek kaliteli Türkçe çok turlu sohbet veri seti üretildi (`data/pedagogy/chat_conversations.jsonl`).
  - `scripts/prepare_chat_balanced_dataset.py`: Morfolojik kökler, uzmanlık ve diyalog verisini harmanlayan dengelenmiş SFT eğitim paketi (`data/train_chat_balanced.bin`) oluşturuldu.
  - `scripts/evaluate_chat.py`: Sohbet yanıt kalitesi, Türkçe gramer uyumu ve bağlam tutarlılığını test eden değerlendirme motoru yazıldı.
- **Konsolide Derin SFT Eğitimi:**
  - `scripts/prepare_deep_dataset.py`: 87.629 kayıt ve 11.216.512 morfolojik tokenden oluşan ana eğitim kümesi (`data/train_deep_sft.bin`) derlendi.
  - `scripts/train_chat_sft.py`: Sohbet ve Self-RAG odaklı hassas ayar (fine-tuning) döngüsü geliştirildi. Model kaybı 0.6601 seviyesine optimize edildi.
- **Dökümantasyon:**
  - `USER_GUIDE.md`: CLI kullanımı, eğitim parametreleri, RAG entegrasyonu ve sorun giderme adımlarını içeren kapsamlı kullanıcı rehberi eklendi.

### Değiştirildi (Changed)
- **Sözlük Genişletmesi (Vocab Expansion):**
  - `KristalTokenizer` ve `data/vocab.json`, Self-RAG kontrol belirteçlerini kapsayacak şekilde 31.310'dan 31.322 token seviyesine güncellendi.
  - `train.py` ve eğitim betiklerine `resize_state_dict` yeteneği eklendi; sözlük boyutu değiştiğinde mevcut model ağırlıkları bozulmadan dinamik olarak genişletilmektedir.
- **README.md Güncellemesi:**
  - Mimari akış şeması (Mermaid), bileşen dizin yapısı, 87.629 örneklik külliyat tablosu ve benchmark sonuçlarıyla güncellendi.
- **Morfolojik Geri Çözücü (Decompiler):**
  - `MorphemeDecompiler` sınıfı `<ARA>`, `<BELGE>` gibi özel XML meta etiketlerini algılayıp yüzey form üretiminde koruyacak şekilde güncellendi.

### Düzeltildi (Fixed)
- **macOS Metal (MPS) Deadlock:**
  - PyTorch'un Metal arka yüzünde büyük sözlük boyutlarında (~31k) AdamW adımlarında oluşan kilitlenme sorunu çözüldü; eğitim adımları CPU üzerinde kararlı hale getirildi, çıkarım için MPS hızlandırması korundu.

---

## [1.1.0] - 2026-09-09

### Eklendi (Added)
- **Morfolojik Görev Derinleştirmesi (Parenting Deep):**
  - `scripts/deepen_parenting_dataset.py`: 10 temel pedagojik görev için (kök bulma, ek analizi, birleştirme, olumsuzluk, soru, çatı dönüşümleri vb.) 30.000 örneklik derinleştirilmiş veri seti eklendi.
- **TDK GTS Anlamsal Sözlük Eğitimi:**
  - `scripts/generate_lexical_semantics_dataset.py`: TDK Güncel Türkçe Sözlük tanımlarını ve anlamsal ilişkilerini morfolojik eğitime kazandıran sentetik veri oluşturucu.
- **Alan Uzmanlığı (Marangozluk & Zanaat):**
  - `scripts/generate_specialization_dataset.py`: Ahşap işleme, zıvana, geçme, kereste kurutma ve geleneksel marangozluk terminolojisini içeren uzmanlık kümesi (`data/pedagogy/carpenter_specialization_dataset.jsonl`).
- **Morfolojik Ek ve Kural Genişletmeleri:**
  - İlgi eki olan `-ki` (`REL_ki`), `MorphotacticsGraph` durum makinesine (`State.NOUN_POST_CASE` ve `State.NOUN_ROOT` $\rightarrow$ `State.ADJ_ROOT`) entegre edildi.
  - `src/compiler/decompiler.py`: Morfem analiz çıktısından deterministik yüzey formu inşa eden `MorphemeDecompiler` motoru yazıldı.

### Değiştirildi (Changed)
- **Vektörel Bellek Esnekliği (VectorMemory Resiliency):**
  - `src/rag/vector_memory.py`: Qdrant sunucusunun erişilemez olduğu durumlarda çökme yaşanmaması için otomatik `:memory:` fallback desteği sağlandı.
- **Fonoloji Motoru Testleri:**
  - 67 birim testin tamamı (%100) yeşile çekildi; ünlü daralması, çift ünsüz türemesi ve kaynaştırma kuralları pekiştirildi.

---

## [1.0.0] - 2026-06-15

### Eklendi (Added)
- **Kristal Derleyici (Crystal Compiler):**
  - `src/compiler/lexicon.py`: O(1) / O(k) karmaşıklığında 20.500 köklük doğrulanmış TDK leksikonu.
  - `src/compiler/morphotactics.py`: Türkçenin ek kurallarını yönlü çizge (DAG) ve sonlu durum makinesi (FSM) olarak modelleyen yapı.
  - `src/compiler/phonology.py`: Büyük/küçük ünlü uyumu ve ünsüz yumuşaması/sertleşmesi fonolojik kuralları.
- **KristalLM Nöral Ağ:**
  - `src/llm/model.py`: RoPE pozisyon kodlamalı, SwiGLU aktivasyonlu ve RMSNorm katmanlı özgün dil modeli mimarisi.
  - `src/llm/tokenizer.py`: BPE yerine deterministik morfem ayrıştırmalı Kristal Tokenizer.
- **RAG & Epistemik Merak:**
  - `src/rag/vector_memory.py`: Qdrant tabanlı anlamsal vektör deposu.
  - `src/rag/merak.py`: Modelin bilgi eksikliğini tespit eden Merak Motoru (Curiosity Engine).
  - `src/llm/router.py`: İçsel bilgi, RAG erişimi ve pedagojik eğitim modu arasında geçiş yapan Tri-Modal Router.
- **Pedagojik Müfredat:**
  - 3 Aşamalı eğitim paradigması: Bebeklik (Infancy), Ebeveynlik (Parenting), Sosyalleşme (Socialization).
