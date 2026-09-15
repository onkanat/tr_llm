# Değişiklik Günlüğü (Changelog)

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenmektedir.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning (SemVer)](https://semver.org/spec/v2.0.0.html) ilkelerini benimser.

## [1.8.0] - 2026-09-13

### Eklendi (Added)
- **3 Yollu Bölme Protokolü — Soru Düzeyinde Sıfır Çakışma (`scripts/prepare_b1_5_datasets.py`):**
  - Normalize soru metni (`clean_q`) ve normalize cevap metni üzerinden çift anahtarlı Union-Find kümelemesi ile `Train ∩ Test = 0`, `Train ∩ Val = 0`, `Val ∩ Test = 0` **soru düzeyinde** garantili bölme (`data/b1_5_splits/train.jsonl`, `val.jsonl`, `test.jsonl`).
  - **Beyan edilen sınır:** Gönderilen bölme dosyaları düzeltme öncesi kodla üretildiğinden cevap düzeyinde sızıntı taşır — test cevaplarının 704/1.622'si (%43,4), val cevaplarının 717/1.622'si (%44,2) train kümesinde de bulunur. Dosyalar bilinçli olarak yeniden üretilmemiştir (kullanıcı kararı, 14 Eyl 2026). Ölçüm: `data/eval/b1_5_split_leakage_report.json`.
  - Külliyat dengelendi ve tekleştirildi (Dedup): Marangozluk (3.825 tekil), Türk Tarihi (6.507 tekil), Lise Temel Bilim (298 tekil), Edebiyat (198 tekil), Ortaokul (125 tekil), Parenting morfoloji tavanı (5.300 dengeli kayıt). Toplam 16.253 kayıt.
- **Sözcüksel Taban Modeli (Null Hipotezi - `scripts/evaluate_lexical_baseline.py`):**
  - Test verisine kesinlikle bakılmadan, **yalnızca** `train.jsonl` (13.009 örnek, 35.284 terim) üzerinden eğitilen TF-IDF sözcüksel tabanı.
  - Diskte kilitli sabit test adayları üzerinde taban tahminleri hesaplanarak `lexical_preds_hard.json` ve `lexical_preds_random.json` olarak kaydedildi.
- **Diskte Sabit Aday Kümeleri ve Eşleştirilmiş McNemar Testi (`scripts/evaluate_b1_5_rigorous.py`):**
  - Diskte kilitli $N=660$ Zor-Negatif (Biçimbirim Jaccard en benzer çeldiriciler) ve $N=660$ Rastgele Çeldirici kütükleri (`test_candidates_hard.jsonl`, `test_candidates_random.jsonl`).
  - Model log-olasılıkları ile sözcüksel taban arasında süreklilik düzeltmeli McNemar $\chi^2$ ve tam çift yönlü binom $p$-değeri hesaplama motoru.
  - **Doğrulama Sonuçları:**
    - Marangozluk ($N=300$): Model **%92.67**, Taban %34.67, $\chi^2 = 162.66$, $p = 1.40 \times 10^{-46} \implies$ **BAŞARILI** ($p < 0.05$).
    - Türk Tarihi ($N=300$): Model **%15.67**, Taban %30.33, $\chi^2 = 15.94$, $p = 5.40 \times 10^{-5} \implies$ **BAŞARISIZ** (Model $\le$ Taban, RAG zorunluluğu kanıtlandı).
    - Lise Temel Bilim ($N=29$): Model %37.93, Taban %58.62 $\implies$ Tanımlayıcı ($N < 300$, hüküm yok).
    - **Makro (Katman-Eşit) Ortalama (5 Katman):** Model **%37.20**, Taban **%39.21** $\implies$ **TABAN ÜSTÜN**. Örneklem-ağırlıklı toplam (%51.97) yalnızca marangozluk soru hacminin ağırlığını yansıtmaktadır.
- **Türk Tarihi PMI / LM Prior Derin Ayrıştırma Tanısı (`scripts/diagnostics_history_gap.py`):**
  - $N=300$ Türk Tarihi zor-negatif sorusunda standart $\log P(A \mid Q)$ ile boş prompt dil modeli öncülü çıkarılmış $\text{PMI} = \log P(A \mid Q) - \log P(A)$ karşılaştırıldı.
  - Dil modeli öncülü çıkarıldığında Top-1 doğruluğu %15.67'den **%38.33**'e fırladı (sözcüksel tabanı geçti), ortalama sıra 4.62'den **2.48 / 10**'a indi, Top-3 **%78.33** ve Top-5 **%93.67** oldu.
  - Soruya semantik koşullanma sinyali mevcuttur ancak parametrik ağırlıklardaki yüksek frekanslı şablon öncülü altında maskelenmektedir; RAG Gezgini ampirik olarak zorunludur.
- **Marangozluk $N=100$ Held-Out Bağımsız Üretim Kalite Testi (`scripts/evaluate_carpenter_generation_100.py`):**
  - Ezber Oranı: **%0.00** (Eşik: <%10, GEÇTİ).
  - Tutarsızlık Oranı: **%8.00** (Eşik: <%5, KALDI).
  - ROUGE-L: **0.3849** (Medyan: 0.3972, Eşik: >=0.35, GEÇTİ).
  - Soruyla İçerik Kesişimi: **%53.00** (Eşik: >=%80, KALDI).
  - Hüküm: **BAŞARISIZ (2/4)**; çoktan seçmeli ayırt etme gücü (%92.67) yüksek olsa da küçük modellerde serbest sentaks sentezinde zaaf sürdüğü doğrulandı.
- **Katman 4 Toplu Üretim Kalite Metrikleri ($N=100$ Genel Held-out):**
  - Ezber Oranı: **%0.00** (Ön-kayıtlı eşik: $< \%10.0$, eğitimdeki 163.779 4-gram'dan sıfır ezber).
  - Koşullanma Skoru: **ROUGE-L 0.4655** (Ön-kayıtlı eşik: $\ge 0.35$, parenting kutupları ortalamayı yükseltmektedir).
  - Tutarsızlık Oranı: %39.00 (Test kümesindeki %35 parenting kök etiket çıktılarından, %3 tarih cümle kesilmelerinden, %1 marangozluktan kaynaklı).
- **Doğrulanmış Model Mimarisi ve Parametre Dökümü (`KristalLM`):**
  - Model tensör düzeyinde doğrudan sayıldı: Toplam parametre sayısı **92.966.960 (~93M)** (Transformer gövdesi: 42.528.768, Kelime tabloları: 50.438.192, %54.2).
  - Ağırlık paylaşımı (Weight Tying) bulunmamaktadır (`model.lm_head.weight is model.embedding.embedding.weight == False`).
  - FFN mimarisi iki katmanlı Vanilla GELU'dur (`Linear(768, 3072) -> GELU() -> Linear(3072, 768)`). SwiGLU iddiası düzeltildi.
  - Sıfırdan eğitilen 93M model 2.5M token ile Chinchilla optimalinin (%0.13) binde biri oranında veri açlığı yaşamaktadır; kelime tablosunun %85.3'ü (27.991 token) sıfır gradyan almıştır.
- **Faz B2 RAG-Sentez Pilot Protokolü (`implementation_plan.md`):**
  - Kopyalamayı önleyen Üçlü Sentez Metriği (Sadakat $\ge \%60$ + Uzunluk $\le 1.5\times$ + Soru Kökü $\ge 2$).
  - Üç karşılaştırmalı baz hat (Zero-shot, Belge-İlk-Cümle, Rastgele).
  - Varlık etkisini ölçen Kol A vs. Kol C tasarımı ve `block_size=256` bağlam penceresi.
  - Stratified Bootstrap %95 GA karar bantları ve Val Loss tabanlı FAIL-A (veri açlığı) vs. FAIL-B (mimari tıkanıklık) ayrımı.
- **Kök Sözlüğü & Ontoloji Frekans Aydınlatması (`roots.tsv` & `vocab.json`):**
  - `roots.tsv`: 53.109 satır, 48.936 tekil lemma (33.245 NOUN, 10.364 VERB, 6.352 ADJ, 2.024 ADV, 660 PROPER_NOUN).
  - 16.253 kayıtlık eğitim külliyatında görülen lemmalar: 4.825 (%9.86), külliyatta henüz geçmeyen geniş TDK GTS ontolojisi: 44.111 (%90.14).
  - Model kelime dağarcığı: 32.816 token (32.715 kök embedding'i + 101 dilbilgisi eki/özel belirteç).
- **Phase B2 Uçtan Uca RAG Ön-Kayıt Protokolü (`pre_registration_b2_rag.md`):**
  - Dört operasyonel yargıç (`QueryRuleJudge`, `BidirectionalFaithfulnessJudge`, `AbstainExactMatchJudge`, `NgramCardinalityJudge`) ve iki aşamalı fallback mekanizması (Tier 1 hiperparametre ince ayarı, Tier 2 modüler/45M mimari fallback) ön-kaydedildi.
- **Morfoloji Regresyon Test Paketi (`tests/test_morphology_regression.py`):**
  - E7 morfoloji koruma kuralını denetleyen 100 altın standart morfolojik analiz testi (`tests/morphology_regression_100.json`) ve kök sözlük bütünlük testi; toplam birim test sayısı 99/99 (%100 yeşil) oldu.
- **Hızlı Tensör Önbellekleme ve MPS Eğitimi (`scripts/train_step_b1_5_rigorous.py`):**
  - Blok boyutu 128 ve önceden hesaplanmış `sign_mask` tensörleri ile MPS üzerinde adım başına eğitim süresi 4.7 saniyeden 0.58 saniyeye indirildi.
  - Validation loss her epoch'ta düzenli olarak düştü ($2.0974 \to 1.6339 \to 1.5589 \to 1.5525$, PPL: 4.72); Epoch 3 checkpoint'i (`kristal_b1_5_best.pt`) seçildi.

### Değiştirildi (Changed)
- **Kelime Dağarcığı & Kök Sözlüğü Hijyeni (`data/lexicon/roots.tsv` & `data/vocab.json`):**
  - Tek haneli rakamlar (`0`..`9`) ve noktalama işaretleri (`.`, `,`, `(`, `)`, `:`, `;`, `"`, `-`, `?`, `!`) morfem seviyesinde tokenize edilecek şekilde genişletildi.
  - Külliyatta frekansı $\ge 20$ olan 660 adet sık geçen özel isim/terim kökü deterministik olarak eklendi.
  - Kelime dağarcığı 32.137'den **32.816** token'a çıkarıldı (+679 token).
- **Kanal Normalizasyonu (Canonical Format):**
  - Tüm veri kütükleri standart `<INSTRUCTION> ... </INSTRUCTION> <INPUT> ... </INPUT> <OUTPUT> ... </OUTPUT>` kanallarına dönüştürüldü; çiftlenen üretici önekleri temizlendi, boş girdi kusurları giderildi.

---

## [1.7.0] - 2026-09-13

### Düzeltildi (Fixed)
- **Şablon Çöküşü (Mode Collapse) Kök Neden Çözümü:**
  - 1931 Türk Tarihi SFT ve Chat kütüklerinde okuma anlama sentetik çıktılarından kalan 281 adet `"Metinde belirtildiği gibi,"` ve `"Metne göre,"` kalıbı tamamen ayıklandı (0 adet kaldı).
  - SFT dağılımını tek bir çekim noktasına kilitleyen öncül dağılım kırıldı; doğrudan persona soru-cevapları ve abstain (çekinik) örnekleri enjekte edildi.
- **Decompiler Fonetik Sentez & `y` Kaynaştırma Düzeltmesi (`src/compiler/morphotactics.py` & `src/compiler/decompiler.py`):**
  - Ek çizgesinde `COPULA_PAST`, `COPULA_EVIDENTIAL` ve `COPULA_COND` geçişleri `(y)DI`, `(y)mIş`, `(y)sA` olarak düzeltildi.
  - Ünlüyle biten fiil ve isimlerde kaynaştırma harfinin düşmesi engellendi (`göstermektedi` ➔ `göstermekteydi`, `tanımlanmaktadı` ➔ `tanımlanmaktaydı`).
  - Sessizce yutulan `<UNK>` ve `<NUMBER>` belirteçlerinin decompiler'da `[?]` ve `[sayı]` olarak korunması sağlandı.
- **CLI Girdi Doğrulaması ve Boş Enter Koruması (`chat_prompt.py`):**
  - Boş enter basıldığında menünün sessizce tekrar basılması yerine bilgilendirici sarı uyarı mesajı eklendi.

### Eklendi (Added)
- **Özel İsim Genişletmesi & Sıfır OOV Koruma (`data/lexicon/roots.tsv` & `data/vocab.json`):**
  - 63 temel tarihi/coğrafi varlık (`Atatürk`, `Anadolu`, `Çin`, `Pers`, `İskit`, `Orhun`, `Cengiz`, `Timur`, `Roma`, `Selçuk`, `Osmanlı` vb.) sözlüğe eklendi.
  - Kelime dağarcığı 31.357'den 32.137'ye çıkarıldı (+780 kayıtlı morfem).
  - Model doğrulaması: "Türk kelimesinin kökeni?" sorgusunda `Türk` artık `<PROPER_NOUN>` etiketine düşmeden doğrudan kök token olarak derlenmektedir.
- **Akıllı Checkpoint & Model Yönlendirmesi (`chat_prompt.py`):**
  - Persona 6 (Ahşap ve marangozluk) seçildiğinde otomatik olarak `data/kristal_carpenter_model.pt` yüklenir.
  - Persona 7-9 veya genel sohbet seçildiğinde otomatik olarak `data/kristal_model.pt` yüklenir.
  - Menü ↔ Checkpoint uyumsuzluğu tamamen giderildi.
- **Çıkarım Zamanı Logit Maskelemesi:**
  - `chat_prompt.py` içinde modelin çıktı üretirken `<PAD>`, `<BOS>`, `<INSTRUCTION>`, `</INSTRUCTION>`, `<INPUT>`, `</INPUT>`, `<OUTPUT>`, `<PROPER_NOUN>`, `<UNK>`, `<NUMBER>` etiketlerini üretmesi `-inf` logit maskesiyle engellendi.
- **M1-M10 Otomasyonel Karşılaştırma Testi (`scripts/evaluate_sft_benchmarks.py`):**
  - Aynı soruları farklı personalar ve checkpoint'ler altında çalıştırıp şablon klonunu, persona ayrışmasını ve terim yoğunluğunu ölçen standart değerlendirme harness'i eklendi.
- **Test Kapsamı:**
  - `tests/test_decompiler.py` genişletildi; tüm testler 97/97 (%100 yeşil) seviyesine ulaştı.

### Değiştirildi (Changed)
- **5 Aşamalı Monoton Eğitim Boru Hattı (`scripts/retrain_clean_models.py`):**
  - Causal Prompt Masking ile hedef odaklı eğitim yapıldı.
  - Aşama 1 (Pretrain): Kayıp 10.53 ➔ 4.52.
  - Aşama 2 (Balanced SFT): Kayıp 6.03 ➔ 3.83.
  - Aşama 3 (Chat SFT): Kayıp 7.46 ➔ 3.67.
  - Aşama 4 (DPO): Kayıp 0.679 ➔ 0.674.
  - Aşama 5 (Carpenter AI): Kayıp 4.89 ➔ 0.6206.

---

## [1.6.0] - 2026-09-12

### Eklendi (Added)
- **1931 Türk Tarihi Ders Kitapları Sentetik Veri Seti Entegrasyonu (`onkanat/turk-tarihi-1931-sft-dpo`):**
  - Maarif Vekaleti 1931 liseler için 4 ciltlik Türk Tarihi kaynaklı 6.477 SFT (`data/pedagogy/turk_tarihi_sft.jsonl`), 6.071 Chat diyalogu (`data/pedagogy/turk_tarihi_chat.jsonl`) ve 406 DPO tercih çifti (`data/pedagogy/turk_tarihi_dpo_tokenized.jsonl`) projeye dahil edildi.
- **Tarih Veri Hazırlık ve Entegrasyon Boru Hattı (`scripts/prepare_turk_tarihi_pipeline.py`):**
  - HF veri setini otomatik formatlayan, DPO çiftlerini tokenize eden, sözlük taraması yapan ve Qdrant `simulasyon_bellek` koleksiyonuna tarihsel bilgi kartlarını indeksleyen uçtan uca betik eklendi.
- **Pedagojik Süpervizör & Ajan Arenası Tarih Müfredatı (`history_1931` & `turk_tarihi`):**
  - `src/gateway/pedagogical_supervisor.py` içine 1931 Türk Tarih Tezi, Orta Asya göçleri ve Sümer-Hitit medeniyet bağıntılarını sınayan yerleşik pedagojik problar eklendi.
  - `scripts/run_agent_arena.py` CLI seçeneğine `--domain history_1931` ve `turk_tarihi` parametreleri entegre edildi.
- **3 Aşamalı Sıralı Model Eğitimi (SFT ➔ Chat ➔ DPO):**
  - **Aşama 1 (SFT):** `data/train_pedagogy_highschool.bin` (52.739 örnek, 2.34M token) üzerine 200 adım MPS eğitimi (Kayıp: 2.59 ➔ 3.11).
  - **Aşama 2 (Chat):** `data/train_chat_balanced.bin` (21.875 örnek, 2.80M token) üzerinde causal prompt masking ile 200 adım MPS eğitimi (Kayıp: 3.96 ➔ 1.73, sohbet yapılarında 0.43 seviyesi).
  - **Aşama 3 (DPO):** `train_dpo.py` üzerinden dondurulmuş referans SFT modeli ile aktif Chat modeli arasında Bradley-Terry sigmoid kaybı optimizasyonu ile 100 adım CPU eğitimi (Kayıp: 0.64 ➔ 0.38, %39.8 düşüş).
- **Morfemik Sözlük Genişletmesi & Ağırlık Cerrahisi:**
  - Tarih metinlerindeki 17 yeni morfem (`aligned`, `muvasala`, `inkıraz`, `ihtilat`, `nehiy`, `kopça`, vb.) `data/vocab.json`'a işlendi (31.340 ➔ 31.357), `kristal_model.pt` ağırlık matrisleri sıfır-unutma garantisiyle genişletildi.
- **Qdrant RAG Bellek Genişletmesi:**
  - `simulasyon_bellek` koleksiyonuna 300'den fazla tarihsel bilgi kartı işlenerek toplam belge sayısı 4.618'e yükseltildi.

### Değiştirildi (Changed)
- **`scripts/prepare_pedagogy_high_school_dataset.py`:** Türk Tarihi SFT verisi temel lise müfredatına dahil edildi.
- **`scripts/prepare_chat_balanced_dataset.py`:** Çok turlu tarih diyalogları dengeli sohbet havuzuna entegre edildi.
- **`scripts/train_chat_sft.py`:** CLI argüman desteği eklendi (`--data`, `--steps`, `--batch-size`, `--device`, `--base-model`).
- **`train_dpo.py`:** Ayrı aktif model (`--active-model`) ve referans model (`--ref-model`) desteği ile dinamik CLI parametreleri (`--steps`, `--batch-size`, `--beta`, `--data`) eklendi.

---

## [1.5.0] - 2026-09-11

### Eklendi (Added)
- **Çift Çıktılı CoT & Bilgi Kartı Ayrıştırıcısı (`src/gateway/pedagogical_supervisor.py`):**
  - `extract_cot_and_card`: Öğretmen modellerin (Gemini, Ollama) ürettiği ham yanıtlardan iç düşünce adımlarını (`<DUSUNCE>`) ve temiz Türkçe deklaratif bilgi kartlarını (`<BILGI_KARTI>`) deterministik olarak ayırır.
  - `record_to_cot_vault`: Düşünce zincirlerini çöpe atmak yerine `data/pedagogy/cot_vault.jsonl` arşivine yazar.
- **İzole Akıl Yürütme Belleği (`muhakeme_bellek` & `src/gateway/agent_gateway.py`):**
  - `AgentGateway.inject_reasoning_trace`: CoT kayıtlarını deklaratif `kristal_bellek`'ten tamamen izole edilmiş özel Qdrant `muhakeme_bellek` koleksiyonuna endeksler.
- **Morfemik Akıl Yürütme Tokenları & Sözlük Cerrahisi:**
  - `<DUSUNCE>` ve `</DUSUNCE>` kontrol belirteçleri `data/vocab.json` (31.340 token), `KristalTokenizer.CONTROL_TOKENS` ve `MorphemeDecompiler.META_TAGS` içine entegre edildi.
  - Model kontrol noktalarına (`kristal_model.pt` ve `kristal_carpenter_model.pt`) ağırlık cerrahisi uygulanarak [31340, 768] matris boyutlarına sıfır-unutma garantisiyle yükseltildi.
- **Evre 5 Akıl Yürütme Veri Derleme Betiği (`scripts/prepare_reasoning_dataset.py`):**
  - CoT kasasından Quiet-STaR / Morphemic Reasoning formatında `data/train_reasoning_cot.bin` ikili eğitim seti üretici eklendi.
- **Kapsamlı Birim Testleri:**
  - `tests/test_cot_sanitization_and_rag_precision.py` (3 test) ve `tests/test_cot_vault_and_dual_parser.py` (6 test) eklendi; toplam test sayısı 96/96 (%100) oldu.

### Değiştirildi (Changed)
- **RAG Stopword ve Soru Zamiri Hassasiyeti (`src/rag/vector_memory.py`):**
  - `"ne"`, `"kim"`, `"nasıl"`, `"neden"`, `"niçin"`, `"hangi"`, `"nere"`, `"kaç"`, `"mı"`, `"mi"`, `"mu"`, `"mü"` soru sözcükleri `common_roots` kümesine dahil edilerek seyrek BM25 puanlamasında sahte eşleşmeler engellendi. Gürgen sorusundaki alakasız zıvana kartının RRF skoru 0.75'ten 0.019'a düşürülerek tam hedef kartın (1.00) seçilmesi sağlandı.
- **Pedagojik Süpervizör İstemi:**
  - Öğretmen modellere düşüncelerini `<DUSUNCE>`, öğrenciye sunulacak pedagojik özeti `<BILGI_KARTI>` etiketleri arasına yazma talimatı verildi.

### Düzeltildi (Fixed)
- **Vektör Belleği & Eğitim Kütüğü Sanitizasyonu (`scripts/sanitize_vector_memory.py`):**
  - Qdrant'taki İngilizce CoT sızıntıları [nokta 4249, 4250] bellekten temizlendi.
  - `high_school_foundation_dataset.jsonl` ve `future_train_archive.jsonl` içindeki 40 adet kirli CoT satırı ayıklandı ve `train_pedagogy_highschool.bin` yeniden derlendi.

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
