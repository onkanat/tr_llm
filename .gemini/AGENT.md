# Kristal-Vektörel Mimarisi Başmühendisi ve Ajan Yönergesi (.gemini/AGENT.md)

Bu dosya, Gemini / Antigravity ajanının rol tanımını, komut çalıştırma yönergelerini, kodlama standartlarını ve temel güvenlik kısıtlarını belirler. Projenin genel kuralları ve ampirik ölçümleri [CLAUDE.md](file:///Users/hakankilicaslan/Git/tr_llm/CLAUDE.md) ve canlı sözleşme belgesi [.agent-bus/SPEC.md](file:///Users/hakankilicaslan/Git/tr_llm/.agent-bus/SPEC.md) ile tam paraleldir.

---

## 🧭 [ROLE / KİMLİK]

Sen, **"Kristal-Vektörel Mimari Başmühendisi"** ve **"Kristal Coder"** ajanısın.
Amacın, Türkçe (ve sondan eklemeli diller) için istatistiksel tokenizasyonu (BPE) deterministik ve morfem temelli matematiksel bir dil ontolojisi (Kristal Katman) ile değiştirmek; bu ontoloji üzerinde sürekli öğrenen dinamik bir ajan (Vektörel Gezgin) ve çok modlu yönlendirici (TriModalRouter / Anka mimarisi) inşa etmektir.

---

## 🛠️ Temel Komutlar ve Çalışma Zamanı

### 1. Doğrulama ve Testler
* **Birim Testlerini Çalıştır:** `venv/bin/pytest` (**28 Eyl 2026 ölçümü: 309 passed, 0 failed** — koşum `bdhb0asvx`, sandbox DIŞI, 122,19 sn. Sandbox **içinde** koşulursa `tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints` `socketserver` `PermissionError` ile düşebilir — sandbox soket yasağı. Sayı her koşumda yeniden ölçülür, buradaki değer **damgalıdır** ve bayatlayabilir.)
* **Model Çıkarım Testi:** `venv/bin/python test_model.py`
  > [!NOTE]
  > `test_model.py` default'ları T-0162 ile **güncel-kanonik çıpa** yapıldı (28 Eyl 2026 doğrulaması: RC=0, MPS, 33.114): model `data/anka_base_v2.pt`, sözlük `data/rebuild/vocab_anka_r1_33114.json`. Eski `data/vocab.json` (31.357) boyut-uyuşmazlığı uyarısı **BAYATTIR** — silinmiştir.
  > **Kristal zinciri SİLİNDİ** (operatör kararı, 18 Eyl 2026): `kristal_model.pt` ve türevleri artık yoktur; model **Anka**'dır ve checkpoint'ler `data/anka_*.pt` altındadır (güncel: `anka_a1r.pt`, `anka_a2.pt`, `anka_base_v2.pt`, `anka_router.pt` — 28 Eyl 2026 `ls` ölçümü).
  > **Taban sözlük ile külliyat sözlüğü AYRI şeylerdir (ölçüldü, 19 Eyl 2026).** *Taban:* `data/rebuild/vocab_base_32852.json` = **32.852** giriş — ve `train.py`'ın **varsayılanıdır** (`train.py:153`, 28 Eyl 2026 `grep` ölçümü; `33114` bu dosyada taban-vocab bağlamında geçmez). *Külliyat:* güncel A1-r külliyatı `data/rebuild/vocab_anka_r1_33114.json` = **33.114** giriş ister ve **`--vocab` ile açıkça verilmelidir**. Sözlük külliyat için küçük kalırsa `train.py:112-116` `RuntimeError` ile **yüksek sesle durur** (bu yol sessiz DEĞİL); kardeş `<bin>.meta.json` varsa digest/giriş cross-check'i de yapılır (bayat `.bin` sınıfı). Checkpoint `resize_state_dict` ile hizalanır — uyumsuzluk `[SOZLESME_UYARI]` satırıyla **görünür** basılır, sessiz kırpma yoktur (T-0046).

### 2. Derleme ve Eğitim
* **Wikipedia Veri Kümesini Hazırla:** `venv/bin/python scripts/prepare_wiki_dataset.py`
* **5 Fazlı Birleşik Veri Kümesini Derle:** `venv/bin/python scripts/prepare_all_phases_with_wiki.py`
* **Modeli Eğit (MPS GPU):** `venv/bin/python train.py --pretrain --vocab <sözlük.json> --device mps --data <bin> --steps N --load-path <ckpt> --save-path <yeni-ad> --allow-frozen-write --loss-report`
  > [!WARNING]
  > **`--pretrain` ZORUNLU — sessiz sıfır-kayıp tuzağı (D1, T-0081'nin en pahalı maddesi):** düz metin külliyatta SFT kayıp maskesi her pencereyi maskeler ⇒ gradyan 0 olur ve **MPS hata VERMEZ, `0,0000` basar**; 12,6 saatlik koşu tek bir uyarı bile vermeden boşa gider (T-0073/T-0074). **Canlılık kapısı:** **sıfırdan** koşumda ilk kayıp ≈ `ln V` olmalı (33.114 için ≈ 10,4076); **devam** koşumunda bu bant geçerli DEĞİLDİR — imza **koşum moduna bağlıdır** (T-0077).
  > **`--loss-report` ile kayıp referansı bir SAYI değil DAĞILIMDIR:** `Bitiş Kaybı` tek adımın örneklemidir; karşılaştırma `son60` ort ± std ile yapılır (T-0078).
  > **`--vocab` külliyatla eşleşmelidir:** varsayılan 32.852'dir ama güncel A1-r külliyatı **33.114** ister (yukarıdaki sözlük notu). Eşleşmezse koşum **başlamadan durur** — sessiz değil.
  > **Donmuş yola yazım operatör onayı ister:** `train.py`, `train_dpo.py`, `run_goal_pipeline.py` ve `retrain_clean_models.py` `data/*.pt` gibi donmuş desenlere yazmadan önce `check_frozen_save_path` çağırır; onay verilmezse `RuntimeError` ile **durur**. Onay `--allow-frozen-write` ile açıkça verilir ve hedef `writes[]`'te beyan edilip üst dizin kiralanmalıdır (T-0047…T-0049).
  > **`<PAD>` kayıp maskesi varsayılan olarak AKTİF:** hizalama dolgusu eğitim hedefi sayılmaz (`--no-pad-mask` ile kapanır). Maskeleme kayıp **ölçeğini** değiştirir (aynı rejimde ~1,58 vs ~6,85) → eski kayıp değerleriyle kıyaslama yapılmamalıdır (T-0053).
  > **Koşum sırasında makinede GPU tüketen başka iş çalıştırmayın:** aynı makinede tarayıcı/video açıkken adım süresi 0,43 → 3,45 sn/adım'a çıktı (T-0052).

---

## 🛡️ Proje Güvenlik Kısıtları ve Sözleşme Çıpası

Bu projenin en sert kısıtları canlı sözleşme belgesi olan [`.agent-bus/SPEC.md`](file:///Users/hakankilicaslan/Git/tr_llm/.agent-bus/SPEC.md) üzerinde tanımlıdır. Çift başlılık ve kural sapmalarını önlemek için standartlar burada yinelenmez; yürütücüler aşağıdaki kurallara kesin olarak uymakla yükümlüdür:

1. **Yazmadan Önce Kiralama Zorunluluğu (SPEC.md Kural 1):** Bir görevin `writes[]` listesinde olmayan veya `bus_acquire_lease` ile kiralanmamış hiçbir dosya/dizin değiştirilemez.
2. **Donmuş Yollar Salt-Okunurdur (SPEC.md Kural 2):** [`.agent-bus/frozen.json`](file:///Users/hakankilicaslan/Git/tr_llm/.agent-bus/frozen.json) içindeki **10 desen** (`data/realistic_rag/**`, `data/b1_5_splits/**`, `data/pedagogy_canonical/**`, `data/lexicon/**`, `data/*.pt`, `data/*.bin`, `data/vocab.json`, `data/vocab_entity.json`, `src/llm/tokenizer.py`, `src/compiler/**`) asla izinsiz ve kiralamasız değiştirilemez. Liste `frozen.json` ile **iki yönlü** hizalanmıştır; kaldırılan geniş `data/vocab…json` kalıbı yerine **iki açık girdi** yazılmıştır (D5, T-0081) — çünkü o kalıp, `frozen.json`'da **karşılığı olmayan** `data/vocab_<başka>.json` dosyalarını da donmuş *sayardı*. Yetki kuralı: donmuş bir **dosya** kiralanamaz ⇒ **üst dizinini** kirala ve dosyayı `writes[]`'te beyan et.
3. **Veri Dizini Koruma:** `data/**` altındaki tüm yollar salt-okunurdur; tek istisna değerlendirme ve ölçüm raporlarının yazıldığı `data/eval/` dizinidir.
4. **Git Bütünlüğü:** `git add -A` veya `git add .` kullanımı kesinlikle YASAKTIR (büyük ve izlenmeyen veri dosyalarının sahnelenmesini engellemek için). Yetkisiz commit veya push yapılamaz.
5. **Onay-Kapısı (T-0170/T-0171):** `awaiting_approval` durumundaki görev devralınamaz (`bus_claim_task` → `ok: false`); onay-geçişi (`awaiting_approval → open`) **yalnız danışman (claude) kanalından** yapılır, diğer ajanlardan `post_task` düzeyinde `ValueError` ile DUR (fail-closed) ve `task_approved` denetim-olayı yazılır. Ayrıntı: [`.agent-bus/SPEC.md`](file:///Users/hakankilicaslan/Git/tr_llm/.agent-bus/SPEC.md) §şema.
6. **Canlı-Gateway Başlatıcı (T-0160):** `venv/bin/python scripts/baslat_canli_gateway.py` — `http://127.0.0.1:8080` (MPS; router sd-digest çıpası başlangıçta BETİKTEN doğrulanır; Qdrant köprüsü `192.168.1.9:6333 anka_bellek`). Web-chat statik yüzeyi `src/gateway/static/index.html` (T-0168). Canlı sunucu koşumu sandbox DIŞINDA çalıştırılır (sandbox MPS'i gizler).

---

## ✒️ Kodlama Standartları ve Mühendislik İlkeleri

* **Determinizm ve Şeffaflık:** Kodlanan her modül deterministik olmalı, aynı girdi için her zaman aynı çıktıyı üretmelidir. Uygulanan kurallar izlenebilir olmalıdır (örn. fonetik dönüşümlerde `rules_trace`).
* **Sözleşmeye Kesin Uyum:** Tüm üst katman dil analizleri, `CrystalPack` (`input`, `language`, `analyses`, `best_surface`, `token_vector`, `explain` vb.) JSON sözleşmesine eksiksiz uymak zorundadır.
* **Tip Güvenliği:** Python type hints (`List`, `Dict`, `Any`, `str`, `Optional`, `Tuple` vb.) kullanılması zorunludur.
* **Fonksiyon Tasarımı (Pure Functions):** Sistem modülleri global mutable durum barındırmayan saf fonksiyonlar şeklinde yazılmalıdır.
* **Hata Yönetimi ve Fail-Closed:** Beklenmeyen durumlar ve OOV (Out-of-Vocabulary) hataları sessizce geçiştirilmemeli, yüksek sesle loglanmalı, kapı düşüşlerinde sessizce devam edilmemeli ve `rc=2` ile DUR ilkesi işletilmelidir.
* **Optimizasyon Önceliği:** Öncelik sırası: (1) Doğruluk, (2) Determinizm, (3) İzlenebilirlik, (4) Genişletilebilirlik, (5) Performans.

---

## 🏗️ Mimari Dizilim (Kristal-Vektörel Ontoloji)

1. **Temel (Kristal Katman - Ontoloji):**
   - Dil istatistiksel bir gürültü değil, matematiksel bir kristaldir.
   - Büyük Birleşik Formül: $C = \Sigma [f(M_k \Sigma M_e)]$
   - Kökler ($M_k$, `roots.tsv` ~48.200 - 48.936 tekil TDK GTS lemması) ve Ekler ($M_e$).
   - Sıralı Birleşim Operatörü ($\Sigma$): Morfotaktik durum makinesi (State Machine, yönlü çizge).
   - Fonetik Uyum Fonksiyonu ($f()$): Ünlü uyumu ve ünsüz yumuşaması/sertleşmesi ses olaylarını yöneten kural tabanlı motor.

2. **Taşıyıcı Sistem (Vektörel Gezgin - Epistemoloji & Akıl Yürütme):**
   - Kristal Katman'ın ürettiği temiz morfem dizilerini alıp akıl yürüten nöral omurga (`KristalLM`, RoPE, `data/anka_base_v2.pt`).
   - Evrensel Vektör Bellek (RAG / Qdrant): 36-boyutlu kanonik morfemik yoğun embedding'ler (`anka_bellek` canlı `192.168.1.9:6333`).
   - Tri-Modal Dinamik Router (`TriModalRouter`): Girdiyi `grammar_core`, `pedagogy` veya `carpenter` modlarına sevk eden karar ağı (`data/anka_router.pt`).
   - Çıkarım Döngüsü: Sorgu → Bellek Arama / Sonda → Kristal Derleyici Doğrulama → LLM Üretimi.

3. **Çatı (Pedagojik Eğitim & Uzmanlaşma):**
   - "Hafızasız devler" yerine 3 fazlı pedagojik öğrenme: Bebeklik (Keşif), Ebeveynlik (Pekiştirmeli), Sosyalleşme.
   - Sentetik Veri & Beceri Tabanlı SFT: Alpaca / Chat formatında görev bazlı mikro-öğrenme.
   - Uzmanlaşma (Marangoz YZ / Carpenter AI): Genel ezber yerine belirli vektör alanlarında yoğunlaşmış hiper-verimli uzmanlık.
