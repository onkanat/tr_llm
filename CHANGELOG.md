# Değişiklik Günlüğü (Changelog)

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenmektedir.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning (SemVer)](https://semver.org/spec/v2.0.0.html) ilkelerini benimser.

---

## [1.4.0] - 2026-09-10

### Eklendi (Added)
- **`<UNK>` Epistemik Merak Tetikleyicisi (`src/rag/merak.py` & `src/rag/epistemic_agent.py`):**
  - Girdide `<UNK>` (bilinmeyen morfem/kök) tespit edildiğinde veya model bir sonraki adımda `<UNK>` kestirdiğinde entropi seviyesine bakılmaksızın otomatik `needs_retrieval = True` üretilerek epistemik merak motoru tetiklenmektedir.
- **Dinamik Sözlük Genişletmesi & Model Ağırlık Cerrahisi (`src/llm/tokenizer.py` & `src/gateway/retrain_pipeline.py`):**
  - `Vocabulary.register_new_tokens`: Çalışma zamanında karşılaşılan yeni morfemleri sözlüğe güvenli bir şekilde kaydeder.
  - `expand_model_vocabulary`: Modelin `embedding` ve `lm_head` katmanlarını sıfır-unutma (zero-forgetting) garantisiyle genişletir; eski ağırlıkları bit düzeyinde korurken yeni morfemleri semantik ortalama ile ilklendirir.
- **Evre 4 Alan Uzmanlaşması (Specialization) Veri ve Eğitim Boru Hattı:**
  - `scripts/prepare_carpenter_specialization_dataset.py`: Ahşap ve marangozluk dikey uzmanlık modülü için bağımsız ikili eğitim paketi (`data/train_carpenter_specialization.bin`, 12.775 örnek, 793K token) derlendi.
  - Model modüler olarak eğitilerek `data/kristal_carpenter_model.pt` ağırlıklarına kaydedildi (Bitiş Kaybı: 0.3778).
- **Google Gemini API Öğretmen Entegrasyonu (`src/gateway/pedagogical_supervisor.py`):**
  - Türk edebiyatı ve şiir gibi derin kültürel alanlarda usta model olarak `gemini-2.5-flash` entegre edildi.
  - `scripts/run_agent_arena.py` ile 200 turluk Türk edebiyatı ve şiir sınavı başarıyla icra edildi.

### Değiştirildi (Changed)
- **Pedagojik Aşama Ayrımı:**
  - Evre 1-3 (Bebeklik, Ebeveynlik ve Temel Lise Müfredatı) ile Evre 4 (Ahşap Alan Uzmanlığı) birbirinden net olarak ayrıldı.
  - `scripts/prepare_pedagogy_high_school_dataset.py` arındırılarak sadece genel lise, edebiyat, şiir, fen, tarih, GTS TDK sözlüğü ve Self-RAG verilerini içerecek şekilde düzenlendi (`data/train_pedagogy_highschool.bin`).
- **CLI Esnekliği:**
  - `train.py`: `--load-path` ve `--save-path` argümanları eklendi.
  - `chat_prompt.py`: `--model` argümanı eklendi; temel lise modeli (`kristal_model.pt`) ile uzmanlık modelleri (`kristal_carpenter_model.pt`) arasında geçiş imkanı sağlandı.
- **Birim Test Kapsamı:**
  - `tests/test_epistemic_unk_and_vocab_expansion.py` eklendi; toplam test sayısı 87'ye çıktı (%100 başarı, 87/87 passed).

### Düzeltildi (Fixed)
- **Mod Çökmesi (Mode Collapse) Giderildi:**
  - `data/pedagogy/infancy_dataset.jsonl` dosyasındaki aşırı tekrarlı (%50) `"negatif çelişik bağ ... uzayda yankı bulunamadı"` sentetik kalıbı dengelendi; modelin uzmanlık sorularında mantıklı terminoloji (`reçine, körelmiş, bıçak, temizlenmeli...`) üretmesi sağlandı.
- **Sözlük Boyutu ve Çift Kayıt Koruması:**
  - `Vocabulary.encode` ve `register_new_tokens` fonksiyonlarında token indeks taşması ve tekrarlı kayıt riskleri tamamen bertaraf edildi.

---

## [1.3.0] - 2026-09-10

### Eklendi (Added)
- **Agent Gateway & REST API (`src/gateway/agent_gateway.py`):**
  - Dış büyük ajan modellerinin (Antigravity Agent'ları, Google Gemini API, Ollama vb.) küçük KristalLM modeliyle otonom iletişim kurması için programatik arayüz ve gömülü HTTP REST sunucusu eklendi (`/api/query`, `/api/inject`, `/api/check`, `/api/status`, `/api/backlog`).
- **Pedagojik Denetçi (Pedagogical Supervisor - `src/gateway/pedagogical_supervisor.py`):**
  - Modelin pedagojik ve alan yeterliliğini otonom sınavdan geçiren öğretmen-öğrenci döngüsü.
  - Eksik veya hatalı bilgi tespitinde RAG sistemine (`kristal_bellek` ve `simulasyon_bellek`) otomatik bilgi enjeksiyonu.
  - Bilgi enjeksiyonu sonrası modelin bu veriyi arama/sorgulama ile bulabilirlik denetimi ($\ge 0.85$ benzerlik eşiği).
- **Epistemik Merak & Sürekli Öğrenme Döngüsü (Karpathy Continuous Learning Loop):**
  - Entropi tabanlı epistemik boşluk algılama ($H(z) > \tau$) ve $\ge 0.85$ alaka skoruna sahip arama sonuçlarının otomatik `data/pedagogy/future_train_vector.jsonl` tamponuna kaydedilmesi.
  - `src/gateway/retrain_pipeline.py`: Biriken `future_train_vector.jsonl` verisini otomatik olarak binary eğitim formatına dönüştürüp modeli pekiştiren yeniden eğitim hattı.
- **Temel Lise & Pedagoji Külliyatı (High School Foundation Dataset):**
  - `scripts/generate_high_school_dataset.py`: Edebiyat, Fizik, Kimya, Biyoloji, Tarih, Coğrafya ve Mantık/Matematik alanlarında 462 temel soru-cevap çifti üretildi (`data/pedagogy/high_school_foundation_dataset.jsonl`).
  - `scripts/prepare_pedagogy_high_school_dataset.py`: Pedagoji ve lise külliyatını birleştiren 45.384 örnek ve 1.623.547 morfemlik eğitim paketi (`data/train_pedagogy_highschool.bin`) derlendi.
- **Antigravity Ajan Rehber Skill'i (`kristal-pedagogical-arena`):**
  - Antigravity ekosistemindeki büyük ajanların küçük KristalLM'i otonom eğitebilmesi ve denetleyebilmesi için kapsamlı rehber skill tanımlandı (`kristal-pedagogical-arena`).
- **Etkileşimli Arena & CLI Desteği:**
  - `scripts/run_agent_arena.py`: Terminalden tek komutla otomatik arena sınavı ve denetim oturumu başlatıcı.
  - `chat_prompt.py`: `--gateway` argümanı ve `arena` etkileşimli komutu eklendi.

### Değiştirildi (Changed)
- **Birim Test Kapsamı:**
  - `tests/test_agent_gateway.py` eklenerek toplam test sayısı 83'e çıkarıldı (%100 başarı, 83/83 passed).
- **Dökümantasyon Güncellemeleri:**
  - `README.md` ve `USER_GUIDE.md`, Agent Gateway, REST API, Karpathy Continuous Learning döngüsü ve temel lise eğitimi referanslarıyla güncellendi.

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
