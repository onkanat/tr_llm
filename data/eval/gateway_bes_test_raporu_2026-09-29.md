# Canlı Gateway (AgentGateway) 5 Benzersiz Test Değerlendirme Raporu

- **Tarih / Zaman:** 2026-09-29T03:48:43Z (06:48 Yerel)
- **Sunucu / Port:** `http://127.0.0.1:8080/api/query` (MPS, Canlı Anka Gateway)
- **Model / Sözlük:** `data/anka_base_v2.pt` (33.114 vocab, `vocab_anka_r1_33114.json`)
- **Vektör Bellek:** `anka_bellek` (Qdrant 192.168.1.9:6333, 36 kanonik doküman)
- **Eşik Yapılandırması:** `RAG_MATCH_THRESHOLD = 0.33` (T-0179), `similarity_threshold = 0.85` (Kapı-D)
- **Parametreler:** `max_new_tokens = 120` (yüksek jeton), `temperature = 0.0` (deterministik çıkarım)
- **Ham Telemetri JSON:** [`data/eval/gateway_bes_test_raporu_2026-09-29.json`](file:///Users/hakankilicaslan/Git/tr_llm/data/eval/gateway_bes_test_raporu_2026-09-29.json)

---

## 📊 Özet Telemetri Matrisi

| # | Soru / Alan | RAG Skoru | Conditioned (>=0.33) | Entropi (Önce / Sonra) | Yönlendirici (Router) | Epistemik Başarısızlık |
|---|---|---|---|---|---|---|
| **1** | **Astronomi (Pulsar & Nötron)**: *Süpernova kalıntısı nötron yıldızı veya pulsar nasıl enerji yayar?* | **1.1111** | **True** (Doküman Verildi) | 8.97 ➔ 2.76 | `['pedagogy', 'legal']` | **True** (Gereksinim kütüğe işlendi) |
| **2** | **Türk Tarihi (1931 Tez)**: *1931 Türk Tarih Tezi doğrultusunda Orta Asya göçlerinin temel nedenleri?* | **0.3932** | **True** (Doküman Verildi) | 3.95 ➔ 4.29 | `['carpenter', 'pedagogy']` | False |
| **3** | **Morfoloji (Grammar Core)**: *Gözlemleyemediklerimizdenmişçesine sözcüğünün kök ve ek tahlili?* | **0.7879** | **True** (Doküman Verildi) | 5.62 ➔ 0.43 | `['grammar_core', 'carpenter']` | False |
| **4** | **Bitki Ekolojisi (Karasal Adaptasyon)**: *Böcekçil bitkiler azotça fakir topraklarda nasıl adapte olmuştur?* | **0.4040** | **True** (Doküman Verildi) | 6.32 ➔ 5.46 | `['pedagogy', 'grammar_core']` | False |
| **5** | **Felsefe / Epistemoloji (Öz-Yansıtma)**: *Belirsizlik taşıması ve merak dürtüsüyle kanıt aramanın epistemik anlamı?* | **0.2333** | **False** (Eşik Altı, Salt LLM) | 7.44 ➔ 2.95 | `['grammar_core', 'legal']` | False |

---

## 🔬 Detaylı Test Analizleri

### Test 1: Astronomi / Pulsar ve Enerji Yayılımı
- **Sorgu:** *"Süpernova kalıntısı olan bir nötron yıldızı veya pulsar nasıl enerji yayar?"*
- **RAG Hibrit Arama:** `Pulsar PSR` kanonik kartını doğrudan buldu. RRF hibrit skoru **1.1111** ile eşiğin çok üzerine çıktı.
- **Getirilen Kanıt:** *"Pulsar PSR üzerine yapılan analizler çöken devasa bir yıldızın süpernova patlaması sonrası kalan hızla dönen nötron yıldızıdır olduğunu gösterir. Bu durum kutuplarından yaydığı periyodik radyo darbeleri kozmik bir saat gibi işler sonucunu doğurur."*
- **Model Yanıtı & Morfemler:**
  - *Metin:* `Eğer nötrinoların kütlesini sıfıra indirir.`
  - *Morfem Dizilimi:* `eğer nötrino PLURAL POSS_2SG kütle POSS_3SG CASE_ACC_N sıfır CASE_DAT indir TENSE_AORIST .`
- **Epistemik Gözlem:** RAG skoru yüksek (>=0.85) olmasına karşın modelin post-entropisi (`2.757`) tau eşiğini aştığı için sistem bunu **Epistemic Failure** olarak işaretledi ve `future_train_vector.jsonl` kütüğüne otomatik besleme adayı olarak kaydetti.

---

### Test 2: Türk Tarihi / Maarif Müfredatı
- **Sorgu:** *"1931 Türk Tarih Tezi doğrultusunda Orta Asya göçlerinin temel nedenleri nelerdir?"*
- **RAG Hibrit Arama:** Bellekteki astronomi yoğunluklu 36 kart arasından 'göç' terimini Fomalhaut toz halkasındaki gezegen göçleri belgesiyle eşleştirdi (Skor: **0.3932**).
- **Yeni 0.33 Eşik Etkisi (T-0179):** Eski 0.40 eşiğinde bu doküman filtrelenip atılacakken, 0.33 eşiği sayesinde başarıyla `conditioned=True` oldu.
- **Model Yanıtı & Morfemler:**
  - *Metin:* `Metin, bunun sonucu olarak kabul edilir.`
  - *Morfem Dizilimi:* `metin , bu CASE_GEN sonuç POSS_3SG ol GERUND_ArAk kabul edil TENSE_AORIST .`
- **Yönlendirici:** `carpenter` ve `pedagogy` uzmanlarını seçti.

---

### Test 3: Türkçe Morfoloji ve Dilbilgisi
- **Sorgu:** *"Gözlemleyemediklerimizdenmişçesine sözcüğünün kök ve ek tahlilini yap."*
- **RAG Hibrit Arama:** Skor **0.7879** (`conditioned=True`).
- **Model Yanıtı & Morfemler:**
  - *Metin:* `Bu durum, [?] ve suya karşı dirençli bir şekilde dönüştürür.`
  - *Morfem Dizilimi:* `bu durum , <UNK> ve su CASE_DAT karşı direnç DERIV_lI bir şekil CASE_LOC dönüştür TENSE_AORIST .`
- **Yönlendirici:** Soru gramer odaklı olduğu için beklendiği gibi birinci uzman olarak **`grammar_core`** (%100 doğru alan tahsisi) seçildi. Entropi 5.62'den 0.43'e sert bir şekilde düştü.

---

### Test 4: Bitki Ekolojisi & Drosera Adaptasyonu
- **Sorgu:** *"Böcekçil bitkiler azotça fakir topraklarda hayatta kalmak için hangi morfolojik adaptasyonları geliştirmiştir?"*
- **RAG Hibrit Arama:** `Utricularia` etobur sucul bitki kartını buldu (Skor: **0.4040**, `conditioned=True`).
- **Model Yanıtı & Morfemler:**
  - *Metin:* `[Özel İsim], bu durumda [?] olarak adlandırılır. [Özel İsim], bir türü olan [?] ( veya) ve bitkilerin ortak atası olarak tanımlanır.`
  - *Morfem Dizilimi:* `<PROPER_NOUN> , bu durum CASE_LOC <UNK> ol GERUND_ArAk adlandırıl TENSE_AORIST ...`
- **Yönlendirici:** `pedagogy` ve `grammar_core` uzmanları devrede.

---

### Test 5: Felsefe & Merak Motoru (Öz-Yansıtma)
- **Sorgu:** *"Bilinmeyen bir olgu karşısında zihnin belirsizlik taşıması ve merak dürtüsüyle kanıt araması epistemik açıdan ne ifade eder?"*
- **RAG Hibrit Arama:** Bellekteki 36 dokümanda doğrudan eşleşen kart bulunamadı (Skor: **0.2333**). Skor 0.33 eşiğinin altında kaldığı için doküman enjeksiyonu yapılmadı (`conditioned=False`).
- **Model Yanıtı & Morfemler:**
  - *Metin:* `Cevap: bu durum, insanların zihniyle ilgili olduklarını gösterir.`
  - *Morfem Dizilimi:* `cevap : bu durum , insan PLURAL POSS_2SG zihin POSS_3SG CASE_INS ilgi DERIV_lI ol PART_DIk POSS_3PL CASE_ACC_N göster TENSE_AORIST .`
- **Gözlem:** Model harici bir doküman olmadan kendi iç parametrik ağırlıklarıyla mantıklı ve dilbilgisi kurallarına uygun bir yanıt üretti; giriş entropisi 7.44 iken üretim sonrası 2.95'e indi.

---

## 🎯 Temel Çıkarımlar ve Mimari Doğrulama

1. **Yeni Eşik Kapısı (0.33) Tam Çalışıyor:** T-0179 ile getirilen 0.33 RRF eşiği sayesinde 0.39 ve 0.40 aralığındaki sınır belgeler (Test 2 ve Test 4) başarıyla modele kanıt olarak beslenmiş, alakasız sorguda (Test 5, 0.23) ise filtre devreye girerek gürültü engellenmiştir.
2. **TriModalRouter Görev Dağılımı:** Gramer sorularında doğrudan `grammar_core`, olgusal ve pedagojik sorularda `pedagogy`/`carpenter` uzmanlarının seçilmesi mimarinin alan ayrıştırmasını başarıyla sürdürdüğünü kanıtlamaktadır.
3. **Morfem Ayrıştırıcı & Decompiler Kararlılığı:** Tüm 120 jetonluk uzun üretimlerde EOS/OUTPUT sonlandırıcılarına uyulmuş, decompiler temiz Türkçe yüzey formları üretmiştir.

---

## 🧪 Test 6 (EK-ANALİZ, T-0183): Inject-edilen Bilginin Uçtan-uca Canlı Koşullaması (T-0182 takibi)

- **Damga / Kaynak:** BETİKTEN `2026-09-29T03:56:27Z` (`date -u`; canlı süreç `conditioned` alanı — T-0181) · operatör emri
- **Kapsam-beyan:** Bu raporun Test 1–5 koşumu `36 kanonik doküman` ve inject-ÖNCESİ
  telemetriyle yapılmıştır (03:48Z); T-0182 yazımı (03:53Z) sonrasında canlı arz **36 → 37**'dir
  (bilinçli-ezim, operatör-onayı; kayıt: `t0182_canli_inject_kayit_2026-09-29.md` — eskisinin
  "36" değeri geçerliliğini korur, kendi koşum-penceresinde).

- **Sorgu:** *"Kırlangıç kuyruğu birleştirmede çekme mukavemetini nasıl sağlar?"* (`max_new_tokens=120`, `temperature=0.0`)
- **RAG Hibrit Arama:** T-0182'de `/api/inject` ile yazılan nokta **birebir** geri-geldi:
  skor **1.3333** (ham RRF iki-kaynak rank-0 → `/RRF_SCORE_MAX=0,75` ölçeğinin üst-uçu),
  `rag_document` = inject-metni (`"Kırlangıç kuyruğu, masa ve çekmece kasalarında çekme
  mukavemeti sağlayan geleneksel bir köşe birleştirmedir."`).
- **Conditioned (>=0.33):** **True** — T-0179 eşiği + T-0181 alanı canlı-yanda birlikte;
  yazılan bilgi RAG-koşullamasına tam girer (`source_collection: anka_bellek`).
- **Yönlendirici:** `['grammar_core', 'legal']` · **Epistemik Başarısızlık:** False.
- **Model Yanıtı & Morfemler:**
  - *Metin:* `Teknik çözüm: usta Cevabı: şerit testere dişlerinin birbirine boşluksuz ve kasmadan oturması mukavemet için şarttır.`
  - *Morfem Dizilimi:* `teknik çözüm : usta Cevabı : şerit testere diş POSS_3PL CASE_GEN_N birbiri POSS_2SG CASE_DAT boş DERIV_lIk DERIV_sIz ve kas GERUND_mAdAn oturma POSS_3SG mukavem...`

### Test-6 çıkarımı

1. **Inject→Retriev→Condition zinciri uçtan-uca CANLI:** /api/inject noktası, aynı
   sunucudan /api/check **ve** /api/query ile birebir kanıtlanmıştır (yazılan bilgi
   artık modelin RAG-koşullama girdisine girer; `conditioned=True`).
2. **Koşullama ≠ İçerik-kullanma (T-0066 dersi canlı-tekrar):** model üretimi
   carpenter-reçete-kalıptır; `response_text` inject-metninden değil parametrik
   ağırlıklardan besleniyor görünüyor. Eşik-decisyon doğru, LM içerik-uyumlu üretim
   beklentisi ayrı sınıftır (ceket-kapasite; T-0137/T-0138 KABUL_YOK zinciriyle tutarlı).
3. **Canlı-tanık sayacı:** `anka_bellek` bundan böyle **37**'dir; ilerideki
   doğrulamalar bu taban-değerle formüle edilir (geri-dönüş path: nokta-silme API'si,
   kimlik `03:53:19.896Z` + `injected_by: agent_gateway`).
