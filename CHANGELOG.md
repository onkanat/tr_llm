# Değişiklik Günlüğü (Changelog)

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenmektedir.

Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) standardına dayanır ve bu proje [Semantic Versioning (SemVer)](https://semver.org/spec/v2.0.0.html) ilkelerini benimser.

## [1.19.0] - 2026-11-04

### Eklendi (Added)
- **P2.6 tur-4 (zaman/kip dozu hükmü):** `anka_taban_v3r4.pt` (5.059 SFT + 1.265 pencere, 198 adım) — **'gideceğim'→TENSE top-1 çevrildi (0,9325**; tur-2'de top-5 dışı, tur-3'te top-2 0,056): merdivende TABAN dışında **ilk 4/4 top-1** (güvenler 0,9988/0,9325/0,9807/0,9218). Buna karşılık kapılar 1/4: KAPI-1 **2,4390** (açık 0,119; kapanış hızı −0,741→−0,153→−0,102→−0,044 nat/tur — doygun) ve sonda ödün verdi (72 morfem; kontrol sorgusu 45→24 erken-EOS; UNK %2,78). **Hüküm: v3r4 ceket tabanı DEĞİL**; ölçülen sıradaki kalem uzun-yanıt alan kartı (iki eksen — görev doğruluğu + üretim uzunluğu — artık aynı veri kalemiyle aynı anda dönmüyor). Merdiven tablosu (6 basamak) + kanıtlar: `data/eval/p26_tur4_kapi_olcum_2026-11-04.md`.

## [1.18.0] - 2026-11-04

### Eklendi (Added)
- **P2.9 faz-1 arşiv uygulaması (kullanıcı onayı):** 20 checkpoint / **6,616 GiB** → `data/_archive/p29-faz1-20261104/` (mv-only — silme yok; 20/20 sha önce=sonra; A/B tam paket **542 passed** her iki durumda; canlı soyağacı ve v3 ailesi dokunulmadı; `data/` kökü .pt hacmi 21,6→15,0 GiB). Faz-2 (C1 + `test_model.py:28` referans düzeltmesi) ayrı onay bekler.
- **Canlı gateway terfisi (kullanıcı talimatı):** router varsayılanı `anka_router_v2.pt` → **`anka_router_v3.pt`** (sd-sha çıpası `4016b210…`) + `legal` uzmanı `anka_legal_replay.pt` (marangoz+bahçıvan aynen; taban pure). Canlı 3-sorgu probe: hukuk→**legal**, marangozluk→carpenter, nötr→gardener; temiz kapanış kanıtlı. **Açık uyarı:** legal ceketi P2.5 kapılarından geçmedi — terfi kullanıcı kararı; uzmanı olmayan 'legal' sorgusunun sessizce tabana düştüğü tuzak (`epistemic_agent.py:459` `.get(primary_expert, self.model)`) terfiyle kapatıldı.
- **Legal ceket v3 (10× veri ölçeği):** `legal_canonical_sft_v2.jsonl` 463 → **4.813 kayıt / 877.721 token** (b1∪b3 = 4.052/4.053 non-mülga madde tam kapsamı; val-50 bayt-özdeş). Koşum: 547 adım, pencere %20 + KL β=0,2. Kapılar: **K1 İLK KEZ GEÇTİ** (ΔCE **−0,523** vs bar −0,50) fakat K2 0/20 (kart kimliği kurulmadı; madde numarası halüsinasyonu) ve K3 OOD **0,083** (KL öğretmeni 4,5× maruziyet altında tutamadı) → **KALDI 1/3**. Altı koşumun haritası: üç kapı bu tarif uzayında bir arada kapanmıyor — legal ceketi deneysel kalır, canlı uzman `anka_legal_replay.pt` (OOD 0,365 — v3'ten üstün) olarak kalır.
- **Kod hijyeni:** supervisor `--auto-retrain` KIRIKLIĞI onarıldı — `varsayilan_otonom_retrain_pipeline()` fabrikası (kanonik yollar, `save≠model` fail-closed kapısı, `RETRAIN_*_PATH` env geçilebilir; TDD 4/4) + `run_agent_arena.py` fabrikaya bağlandı. Bayat `prepare_reasoning_dataset.py` (v1.5.0, ölü 31.357 sözlüğü) `scripts/_legacy/prepare_reasoning_dataset_v150.py`'ye emekli; tek aktif test referansı + README/USER_GUIDE `_p22`'ye yönlendirildi (16/16 GREEN).
- **`simulasyon_bellek` canlı tohumlandı:** **4.073 nokta** (`data/simulasyon_bellek_export.jsonl` aynen; idempotent upsert 4.073→4.073; read-back birebir; `anka_bellek` 46 / `muhakeme_bellek` 91 değişmedi) — uzak fallback dalı artık ölü değil. Betik: `scripts/seed_simulasyon_bellek.py`.
- **ΔH sinyel seçeneği (opt-in):** `src/rag/merak.py`'ye `trigger_signal='entropy' (DEFAULT) | 'delta_h' | 'post_h'` — donmuş τ=2,5 varsayılanı bit-düzeyinde aynı (TDD 8/8; tam süit **542 passed**). Çevrimdışı kalibrasyon (P2.4'ün 36 sorguluk kayıtları, model koşmadan): en iyi ayrım `delta_h @ −1,50` (J +0,27; tetik %100→%31), hatırlama-öncelikli `post_h @ +2,50` (J +0,24; belirsiz TPR 0,70) — **öneri olarak belgelendi; eşik dondurulmadı (n=36 ince)**.
- **Zaman/kip doz verisi (tur-4 girdisi):** `data/pedagogy/pure_base_sft_v3.jsonl` — **+446 ölçümlü kayıt** (40 fiil × 30 biçim ızgarası; 467 çakışan/UNK aday tokenizer ölçümüyle elendi; zaman/kip kaydı 204→**650**; val-150 bayt-özdeş; UNK %0,00). Tur-4 komutu (`v3r3`→`v3r4`, pencere %20, KL β=0,2) hazır — donmuş kapılar aynı dörtlü.

## [1.17.0] - 2026-11-04

### Eklendi (Added)
- **P2.5 — legal ceketi hattı (FAZ-1/2A/2B/2C, ölçümlü hükümler):** Hukuk korpusu: 17 kanun / 4.151.408 karakter (mevzuat.gov.tr metin ucu kanıtlı; UTF-8/cp1254 meta tuzağı ve İ-harf indeks kayması ölçülerek aşıldı; UNK **%1,34** — donmuş 33.114 sözlükte genişletme gerekmedi) + `data/pedagogy/legal_canonical_sft.jsonl` (463 kayıt; **%99,7 birebir-alıntı** bağımsız denetimi). Ceket v1 (T-0209 tarifi, lr 1e-5, 3 epok): **KALDI 1/3** — ölçülen kök neden: biçim EDİNİMİ gerekiyor (init val CE 4,78; marangoz ceketi biçim-içi 1,07'den ince-ayar yapmıştı). Router v3 (`data/anka_router_v3.pt`, v2'ye yazılmadı): legal sınıfı **0→60 örnek**; 10 hukuk sorgusunda **0/10 → 8/10 legal**; diğer 4 sınıf v2-paritesi 1,0 (T-0210 çıpası birebir). Ceket replay karışımı (P2.6 tarifi: SFT + %19,95 pencere + KL β=0,2, 10 epok, 125 adım): **KALDI 0/3** (ΔCE −0,306/−0,50; kanonik-20 0/20; OOD 0,365/0,40) — ters-korelasyon (alan-öğrenim ↔ OOD sadakati) yumuşadı ama kalkmadı. **Toplam hüküm: kart şablonu edinimi VERİ ölçeğine bağlı** (legal 413 train vs marangoz 6.123) — yeni kalem: legal SFT setinin ~3-6K kayda büyütülmesi.
- **P2.7 — BPE-baseline benchmark repoda (tek komut, tekrarlanabilir):** `scripts/benchmark_bpe_baseline.py` — Kristal **2,065 tok/kelime** (UNK %1,18) vs BPE-32k 2,101 (in-domain); dağıtım-dışı benzersiz-kelime örnekleminde **1,70× üstünlük**; kaynak BPE eğiticisindeki PYTHONHASHSEED-nondeterministik merge sırası ölçülüp onarıldı (3 bağımsız süreç bayt-özdeş).
- **P2.8 — router ablasyonu (q_merak kanal katkısı):** kanal val top-1 doğruluğuna **+24,75 puan** katıyor (1,000 → 0,7525 sıfırlanınca; sınıf kırılımı: pedagogy 0,32, carpenter 0,70); 'kanal verilmmez' P-only yolu bit-özdeş kanıtlandı; diskte hiçbir ağırlık değişmedi.
- **P2.9 — checkpoint envanteri + arşiv politika taslağı:** 32 dosya / 10,8 GiB, sha-kanıtlı; **5 bayt-özdeş yinelenik çift = 1,74 GiB** saf tekrar; canlı soyağacı (pure + marangoz + bahcivan + router_v2) 1,05 GiB **DOKUNULMAZ** (kod düzeyinde teyitli); faz-1 arşiv önerisi 20 dosya / 6,6 GiB — mv listeleri YALNIZ TASLAK, uygulama `test_model.py:28` referans düzeltmesi önkoşuluyla kullanıcı onayına bağlı.

## [1.16.0] - 2026-11-04

### Eklendi (Added)
- **P2.2 — Evre-5 akıl yürütme derlemesi:** `scripts/prepare_reasoning_dataset_p22.py` + `data/train_reasoning_cot.bin` (1.126 uint16 token; 3 kalite-kapılı kayıt — kasada 22 benzersiz kayıttan 19'u ayrıştırıcı artefaktı nedeniyle elendi, gerekçeleri meta'da). UNK %0,8; sınır kapısı maks id 32.151; `KristalDataset` P2.3 tüketimi kanıtlı. Plan adı `prepare_reasoning_dataset.py` izlenmiş v1.5.0 betiğine ait olduğundan yeni betik `_p22` adıyla (emeklilik kararı ayrı kalem).
- **P2.3 — ilk gerçek otonom retrain döngüsü (uçtan uca kanıt):** planın 'eski-şema' önermesi git kanıtıyla çürütüldü (5 kayıt zaten 14-alanlı yeni şemada) — göç uydurulmadı; 7 testlik şema doğrulayıcısı + sigorta kopyası eklendi. Canlı vaka: 'Pulsarlar neden bu kadar düzenli radyo darbeleri yayar?' → entropi 5,46→6,02 (τ üzeri) → epistemik hat yeni şemayla yazdı (5→6) → backlog eşiği (6≥5) → `retrain_pipeline.run_training` 12 adım EXIT=0 (8,98 sn, scratch kaydı) → arşiv; `anka_taban_pure.pt` sha256 değişmedi. Hüküm: `data/eval/p23_otonom_retrain_hukum_2026-11-04.json`.
- **P2.4 — τ=2,5 ve 0,33 eşik kalibrasyonu (ölçüm → dondurma):** 36 sorguluk canlı-varı set (gerçek gateway + salt-okunur anka_bellek): τ=2,5 %100 tetikler (prompt-entropi tabanı ≥3,28 nat) ve alternatif τ masası ayrımı iyileştirmiyor (kategori dağılımları örtüşük, sıralama ters) → **τ=2,5 ölçümle dondu**; 0,33 alan sorgularının 8/10'unu geçiriyor, yükseltme kapsamı düşürüyor (0,40→%50, 0,60→%20), elelenenler eşikle kurtarılamaz → **0,33 ölçümle dondu**. Ham dağılım: `data/eval/p24_esik_dagilim_2026-11-04.json` + rapor `p24_esik_kalibrasyon_2026-11-04.md`.

### Bilinen sınırlar / arka-log (P2.4 bulguları)
- `pedagogical_supervisor.py:644` `--auto-retrain` varsayılan `RetrainPipeline()` ile KIRIK (`run_training` T-0087 açık yol ister — RuntimeError); canlı kullanım için konfigüre pipeline enjeksiyonu gerekli.
- Uzak Qdrant'ta `simulasyon_bellek` BOŞ (0 nokta) — fallback dalı ölü.
- Merak tetiğinde ayrım istenirse çözüm τ değil sinyal değişikliği (ΔH / post-entropi) — ayrı iş kalemi.

## [1.15.0] - 2026-11-04

### Eklendi (Added)
- **P2.1 — Qdrant parametreleşmesi + `muhakeme_bellek` canlı:** `VectorMemory`/gateway'deki `localhost:6333` sabitleri `QDRANT_URL` env/arg akışına bağlandı (varsayılan birebir korunur; mevcut testlerin `:memory:` sözleşmesi bozulmaz); `inject_reasoning_trace` gateway'in çözdüğü adrese yazar. `scripts/seed_muhakeme_bellek.py`: 91 CoT kaydını tam-kayıt-sha256 nokta kimliğiyle upsert eder (içerik-bazında yalnız 22 benzersiz — kimlik kayıt-bazlı seçilmeseydi ≥91 kabul düşerdi), idempotent (tekrar koşum 91→91). Canlı doğrulama: `QDRANT_URL=http://192.168.1.9:6333` → 91 nokta read-back, `anka_bellek` dokunulmadı. İstemci `timeout` parametresi (varsayılan 2,0s korundu — LAN'da create_collection yanıtı 2s'yi aşabiliyor, ölçüldü). Tam paket **517 passed + 1 xfailed**.
- **P2.6 tur-3 + SFT veri zenginleştirme:** `data/pedagogy/pure_base_sft_v2.jsonl` (+265 ölçümlü kayıt: 194 zaman/kip ızgarası — kristal tokenizer ile doğrulanmış, sözlüksel çakışan 106/300 aday elendi; 40 uzun nötr 'nedir ve nasıl' kaydı, 85-113 morfem; 17 durum + 14 çoğul; val-150 kuyruğu bayt-özdeş korunur, yeni kayıtlarda 0 UNK). Tur-3 `data/anka_taban_v3r3.pt`: masked SFT val CE **2,5848 → 2,4827**. Dört eksik aksın hepsinde ölçülebilir ilerleme (TENSE ilk kez top-5'te 2. sıra; UNK %3,00 → **%2,33** ilk kez eşik altı; görev-3/4 güvenleri 0,92/0,83; zımpara erken-EOS 14→30) ama donmuş kapılar çevrilmedi: **1/4** (yalnız pencere-CE ≤3,35) → **v3r3 ceket tabanı DEĞİL**; merdiven kapanış hızı −0,74 → −0,15 → −0,10 nat/tur ile protokolle durduruldu. Ceketler `anka_taban_pure` üzerinde kalır; hüküm kayıtları: `data/eval/p26_tur3_kapi_olcum_2026-11-04.md`.

## [1.14.0] - 2026-11-04

### Eklendi (Added)
- **P2.6 — SFT-replay merdiveni (`scripts/train_p26_sft_replay.py`):** `anka_taban_v3.pt`'ü ceket-uygun tabana çevirmek için T-0209 kalıbının ölçümlü uyarlaması: SFT jsonl + külliyat pencere karışımı (T-0209'un 4:1'i; 4.348 SFT + 1.087 pencere, %20), kaynak-seçimli GERÇEK KL çapası (β=0,2, öğretmen v3; MPS mikro-parti OOM onarımı — kayıp/gradyan birebir), lr 5e-5→1e-5, 170 opt-adım/epok (~6 dk/epok), atomik kayıtlar.
- Tur-1 `data/anka_taban_v3r.pt` + tur-2 `data/anka_taban_v3r2.pt`: masked SFT val CE **3,4789 → 2,7380 → 2,5848** (aşırı-eğitim sinyali yok; külliyat pencere-CE 3,2118 → 3,2181 — KL ödünü ~0,02 nat'ta sınırlı). 4 donmuş kapıdan yalnız pencere-CE (≤3,35) geçti; SFT bandı (≤2,32), paket 4/4 top-1 ve UNK ≤%2,5 kapanmadı → **ceket hattı açılmaz**; ceketler `anka_taban_pure` üzerinde kalır.
- Kapı ölçüm kayıtları: `data/eval/p26_kapı_olcum_2026-11-04.md`, `data/eval/p26_tur2_kapi_olcum_2026-11-04.md`, `data/eval/p26_replay_hukum_2026-11-04.json`.

### Düzeltildi (Fixed)
- **T-0209 KL beyanı düzeltmesi (ölçülmüş):** `scripts/train_t0209_taban_pure.py:163-169`'daki `len(batch) == 4` her iki veri dalında da doğruydu → `is_pencere` hep False → KL β=0,2 dalı **hiç koşmamış**; [1.11.0]'daki "KL β=0.2 korunarak" beyanı kod düzeyinde geçersizdir — T-0209'un ölçülen sonucu yalnız 4:1 karışımla alındı. P2.6 betiği kaynak tensorüyle satır-seçim yaparak KL'yi gerçekten ateşler (kanıt: 325/340 mikro-parti, tüm-koşum KL ort 0,09).

### Bilinen sınırlar (P2.6 hükmü)
- Görev-2 ('gideceğim' → TENSE) iki replay turunda da `root`'a yığıldı (TENSE top-5'te yok) → sorun epok değil `pure_base_sft.jsonl` veri bileşimi: zaman/kip tespiti örnekleri ve uzun alan yanıtları ('nedir ve nasıl kullanılır' kalıbı) eklenmeden KAPI-1/3 kapanmaz — yeni iş kalemi.
- UNK üretimi replay-bağışık (sorgu-başı 1'er sabit) → sözlük kapsamı ayrı kalem; erken-EOS 'zımpara' kartında sürüyor.

## [1.13.0] - 2026-11-04

### Eklendi (Added)
- **Faz-2 — sözlük-nitelik fonoloji hattı (Plan-1, inceleme K1-K3/Ö1 kapanışıyla):**
  - Lexicon nitelik altyapısı: `PhonologyEngine.resolve_affix` lemma niteliklerini alıyor (FRONT/BACK_HARMONY → ünlü uyumu çevirisi); compiler/decompiler çağrı noktaları kök niteliğini geçirir.
  - Sözlük verisi (satır 52.373 / lemma 48.200 sabit; çapa SHA beyanlı 3×): `et`/`fehmet` VOICING (`edeceğim`), `imkân`/`hâl`/`ihtimal` â-uyum istisnaları (`imkânı`, `hâli`, `ihtimali`), `de`/`ye` FRONT_HARMONY (ünlüsüz-kalan gövde kaynaşması: `diyor`, `yiyor`).
  - -(I)yor ünlü kaynaşması (`anlıyor`/`arıyor`/`oynuyor`/`okuyor` sınıfı, 8 vaka + regresyonlar); NEG+(I)yor budama düzeltmesi (`anlamıyor` sınıfı çözülür; fonoloji artık ölü kod değil).
  - Morfotaktik: isim-soylu şahıs kenarları (`iyiyim`, `nasılsın`, `evdeyim`; çıplak NOUN_ROOT bilinçli hariç — `evim` iyelik ikilemesi korunur) ve ADJ→isim türetimleri (DERIV_lIk/lI/sIz/CI: `hayranlık`, `güzellik`, `güzelsiz`).
  - Decompiler sağlamlaştırma: PAST_PERSON override önceliği (`geldik` sınıfı geri geldi), kök-bağlı aday seçimi (`iyiyim`/`hastayım` üretim roundtrip). Test paketi **495 passed / 1 xfailed**.
- **Faz-3 — külliyat v2 (29,1M token):** GTS tanım çıkarıcı (128.327 satır — kaynak tavanı kanıtlı), Wikipedia çok-parça akışı (27M token, resume'lu), birleşme betiği `scripts/assemble_kulliyat_v2.py` — sözlük kapsamı %14,7 → **%95,58**; katmanlı %2 val dilimi + `.bin.meta.json` provenance kapısı; `scripts/eval_ce_val.py` pencere-CE değerlendiricisi.
- **T3.4/T3.5 — devam-ön-eğitim `data/anka_taban_v3.pt` + değerlendirme:** 1 epok (14.226 adım, 12,9 sa), kayıp 3,59 → 3,31 (son200 ±0,17), LR 2e-4→2e-5 tam anneal. Düz metinde net iyileşme: kulliyat_v2 val CE 3,8205 → **3,1936** (PPL 45,6 → 24,4; −%16,4 CE); eski a1r pencere-CE 3,4899 → 3,2734. SFT biçiminde bilinçli/ölçülmüş gerileme: masked SFT val CE 2,3049 → 3,4789 (saf-korpus koşumu, SFT-replay yok), dil bilgisi paketi güveni 0,9998 → 0,0726 — **v3 ceket tabanı olarak KULLANILMAZ**; ceket hattı için SFT-replay (P2.6) şart. Ölçüm kayıtları: `data/eval/t34_v3_degerlendirme_2026-11-04.md`, `data/eval/t34_egitim_ozet_2026-11-04.md`.

### Değişti (Changed)
- Hüküm ölçütü netleştirmesi: T-0209 kayıtlarındaki "Val CE 2,3049" **masked SFT val CE**'dir (pencere-CE 3,4899 ayrı satırdadır); eşik denetimleri artık ölçüt-metrik adıyla yazılır.

## [1.12.0] - 2026-11-04

### Değişti (Changed)
- Ölçüm-kanıtlı doküman senkronu (inceleme: trllm-inceleme-sentez-20261103): Zero-OOV rozeti dürüstleştirildi (UNK %4–11), Mermaid lemma sayıları düzeltildi (48.200/52.373), anka_bellek nokta sayısı 36→46.
- **Faz-1 kod düzeltmeleri** (kalite-inceleme bulguları, TDD ile kapatıldı):
  - `src/llm/tokenizer.py` — U+2019 (tipografik kesme) encode regex sınıfına alındı; apostrof dalına boş-gövde koruması eklendi: kesme kelimeyi çerçeveliyorsa (ör. `’Merhaba’`) inflection dalına girilmez, kelime kesmeler soyularak genel derleme yoluna düşer — `<UNK>`/kelime kaybı (`'[?]'`) yaşanmaz (`tests/test_u2019_typographic.py`).
  - `src/compiler/decompiler.py` — ondalık birleştirme lookahead'i Türkçe virgüle genişletildi (`"3,14"` artık `"3, 14"` değil `"3,14"`); `'.'` ve `','` için ortak bilinçli sınırlama (etiket akışında `rakam . rakam` / `rakam , rakam` görünümü cümle/listeden ayırt edilemez) kod yorumunda belgelendi ve kasıtlı-takas pin testleriyle sabitlendi (`tests/test_decimal_roundtrip.py`).
  - Bayat varsayılanlar temizlendi: `verify_rag.py` taban modeli `data/anka_taban_pure.pt`'ye bağlandı; `rag_tool.py` koleksiyon varsayılanı `anka_bellek`, `storage_path=None` (remote/bellek-içi), sözlük yolu kanonik `data/rebuild/vocab_anka_r1_33114.json`; `scripts/prepare_wiki_dataset.py` aynı kanonik sözlüğe hizalandı.

## [1.11.0] - 2026-10-02

### Eklendi (Added)
- **T-0209 — Marangozluk Dikey Ceket Ayrışımı ve Saf Akıcı Taban Model (`anka_taban_pure.pt`):**
  - Taban model içindeki ahşap/marangozluk sızıntıları temizlenerek `data/pedagogy/pure_base_sft.jsonl` (4.498 kayıt) veri setiyle marangozluktan arındırılmış saf taban model `data/anka_taban_pure.pt` (Val CE 2.3049, pencere-CE 3.4899) 4:1 SFT:Pretrain ve KL $\beta=0.2$ korunarak sıfırdan eğitildi.
  - Bağımsız **Marangozluk Ceketi (`anka_marangoz.pt`)** üretildi: 6.123 kanonik marangozluk SFT kaydı (`data/pedagogy/marangoz_canonical_sft.jsonl`, meşe ağacı ontolojisi dahil) ile saf taban üzerine eğitildi (Val CE 1.0403).
  - Canlı Gateway (`scripts/baslat_canli_gateway.py`), saf taban (`anka_taban_pure.pt`), iki bağımsız uzman ceket (`gardener`: `data/anka_bahcivan.pt`, `carpenter`: `data/anka_marangoz.pt`) ve `anka_router_v2.pt` ile MoE mimarisinde ayağa kaldırıldı.
- **T-0208 — TriModalRouter v2 (5 Sınıf) & Dinamik Uzman Dağıtımı:**
  - TriModalRouter 3 sınıftan 5 sınıfa genişletildi (`grammar_core`, `pedagogy`, `carpenter`, `gardener`, `legal`).
  - Dinamik uzman seçimi: Gateway (`AgentGateway`), yönlendiricinin seçtiği birincil uzmana göre (`expert_models`) ilgili uzman ağırlıklarını (`KristalLM`) dinamik olarak devreye sokar.
  - `data/anka_router_v2.pt` 5-sınıflı yönlendirici modeli eğitilip tescillendi.
- **T-0205..T-0207 — Bahçıvanlık Dikey Ceketi (`anka_bahcivan.pt`) & Canlı Doğrulama:**
  - Bahçıvanlık dikey alanı için bağımsız uzman ceket `data/anka_bahcivan.pt` eğitildi.
  - Model tutarsızlık ve akıcılık testleri yapılarak botanik ve tarım sorgularında uzmanlaşma sağlandı.
- **T-0167..T-0183 — Gateway Yetkinlikleri ve Bellek İyileştirmeleri:**
  - `/api/query` arayüzüne `temperature`, `top_k`, `conditioned` alanları eklendi.
  - Web UI (`src/gateway/static/index.html`) güncellendi.
  - Qdrant RAG eşik değeri 0.33 olarak kalibre edildi. Test paketi 320 teste ulaştı.

## [1.10.0] - 2026-09-28

### Eklendi (Added)
- **T-0151 — `kristal_bellek` → `anka_bellek` Koleksiyon Yeniden-Adlandırması:**
  - Kod tabanı genelinde 7 dosyada 36 geçişle koleksiyon adı `anka_bellek` olarak güncellendi (`vector_memory.py`, `agent_gateway.py`, `pedagogical_supervisor.py`, `sanitize_vector_memory.py`, `sanitize_all_accumulated_datasets.py`, `run_agent_arena.py`, `test_agent_gateway.py`).
  - `generate_kristal_vector` fonksiyon adı ve tokenizer/model marka adları korundu; `VectorMemory()` kurucu varsayılanı ve `_resolve_target_memory` `anka_bellek` ile hizalandı; bilinmeyen koleksiyon adında fail-closed `ValueError` doğrulandı (commit `a9a5b03`, koşum-2 6/6 GEÇTİ, damga `2026-09-28T06:33:06Z`).
- **T-0152 — `data/qdrant_db/` Disk-Artefakt Temizliği:**
  - T-0148 Tur-B 4B gereği kurucu varsayılan `storage_path`'in kaldırılmasının ardından diskte kalan 6 dosya (toplam 304.437 bayt, ~304 KB) operatör onayıyla silindi (`.lock`, `meta.json`, `simulasyon_bellek`, `test_temp_coll`, eski-ad `kristal_bellek`, `muhakeme_bellek`). Canlı Qdrant'ta 36-nokta yeşil durum korundu (commit `256c9e5`).
- **T-0153 — Canlı-Migrate `kristal_bellek` → `anka_bellek`:**
  - Uzak Qdrant sunucusunda (`192.168.1.9:6333`) nokta kopyası (verbatim migrate) yöntemiyle 36 kanonik RAG belgesi birebir taşındı; dense 768 Cosine + sparse BM25/IDF + `on_disk_payload=true` şeması kuruldu, self-retrieval doğrulandı, eski koleksiyon `delete_collection` sarmalayıcısıyla silindi; 9 yabancı koleksiyon korundu (commit `9f509cb`, 6/6 GEÇTİ, damga `2026-09-28T06:44:43Z`).
- **T-0154 — TriModalRouter Eğitilmiş Ağırlıklar (3-Sınıf):**
  - Prompt embedding (768), merak vektörü (`CuriosityEngine`, 768) ve multimodal gating üzerinden 3 sınıflı (`grammar_core`, `pedagogy`, `carpenter`) yönlendirici CPU AdamW ile 3 epoch eğitildi (val top-1 **0,93**, baseline **0,2867**; train kayıp 0,3604 → 0,1890 azalan). İkinci koşumda deterministik bit-özdeşlik (`state_dict` SHA `9b85f951…`) sağlandı; `data/anka_router.pt` (SHA `64527c72…`) üretildi (commit `8bab7ae`, 6/6 GEÇTİ, damga `2026-09-28T06:58:13Z`).
- **T-0155 — Gateway TriModalRouter Eğitilmiş Ağırlık Entegrasyonu:**
  - `src/rag/epistemic_agent.py` içine `router_state_path` parametresi ve fail-closed `load_trained_router_state` fonksiyonu (dosya yoksa/şema uymazsa `RuntimeError`) eklendi.
  - `src/gateway/agent_gateway.py` `create_default` varsayılanı `"data/anka_router.pt"` olarak sabitlendi; 300-val kümesinde 279/300 (**0,93**) doğruluk birebir yeniden üretildi (commit `a6a20ad`, koşum-2 6/6 GEÇTİ, damga `2026-09-28T07:16:03Z`).
- **T-0156 — Pedagogy Daraltma ve Router Yeniden-Eğitimi:**
  - Pedagogy kaynakları 5'ten 4 dosyaya daraltıldı (`parenting_canonical.jsonl` çıplak-kelime morfolojisi ve `grammar_core` ile 1.097 örtük metin paylaştığı için elendi). 7.048 tekil metinlik havuzdan 600 örnek seçildi (`grammar_core` kesişimi 0, parenting üyeliği 0).
  - Model yeniden eğitilerek pedagogy train başarısı **416/500 → 498/500**'e çıkarıldı; val top-1 **1,00** (baseline 0,24), val karışım matrisi köşegen 300/300 oldu.
  - `data/anka_router.pt` operatör onayıyla güncellendi: eski SHA `64527c72…` → yeni SHA `44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a` (state_dict SHA `d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf`, commit `7eafe53`, 8/8 GEÇTİ, damga `2026-09-28T16:19:27Z`).
- **T-0157 — Gateway İnsan-Okunur Üretim Onarımı:**
  - `src/rag/epistemic_agent.py` içinde `YAPISAL_BASTIRMA_JETONLARI` (PAD, BOS, INSTRUCTION vb. 7 jeton) tanımlandı ve üretim sırasında argmax öncesi `-inf` ile maskelendi. Boş maske durumunda fail-closed `RuntimeError` konuldu.
  - `process_query` çıktısında UNK korunurken yapısal etiketler temizlendi ve `decompile_sentence(…, capitalize=True)` ile tüm çıktılar büyük harfle başlatıldı (commit `db5858e`, koşum-2 6/6 GEÇTİ, damga `2026-09-28T07:56:36Z`).
- **T-0158 — Canlı Gateway Başlatma Kanıtı:**
  - Canlı gateway `http://127.0.0.1:8080` üzerinde T-0156 router ağırlıkları (`d369b3cd…`) ve uzak Qdrant (`192.168.1.9:6333`, 36 belge) köprüsüyle arka planda ayağa kaldırıldı. REST `/api/query` çağrısında 17 anahtarlı yanıt, capitalize edilmiş temiz Türkçe çıktı ve deterministik çalışma doğrulandı (`.agent-bus/notes/T-0158.md`).
- **T-0159 — Belge Hizalaması (README.md + USER_GUIDE.md):**
  - `README.md` ve `USER_GUIDE.md` güncel test tabanı (99/99 → **307/307**), sözlük ayrımı (taban 32.852 ≠ külliyat 33.114), model soyağacı (`anka_base_v2.pt`, `anka_router.pt`), `/api/query` JSON gövdesi (`query`, `instruction`, `mode`) ve fail-closed `VectorMemory` mimarisiyle tam senkronize edildi (commit `9f227de`).
- **T-0160 — TUR-X: Kalıcı Başlatıcı ve Parametre Normalizasyonu:**
  - `scripts/baslat_canli_gateway.py` oluşturuldu; fail-closed router SHA çıpası (`d369b3cd…`) ve `--port` argümanı eklendi.
  - `src/rag/epistemic_agent.py:234` içindeki ölü varsayılan `max_new_tokens` 40'tan canlı çağrıyla uyumlu **45**'e normalize edildi (commit `366f5a3`).
- **T-0161 — Soru Filtresi Şartnamesi ve Router Koşul Doğrulaması:**
  - Pedagoji 4-dosya havuzunda soru işareti (`?`) dağılımı ampirik olarak ölçüldü: toplam 7.540 kayıt, 7.048 tekil (%94,40 soru içeren, %5,60 içermeyen).
  - Soru içerenleri dışlama hipotezinin (Kol-A: 395 tekil) router için gereken 600 tekil örneği karşılayamadığı (`395 < 600`) matematiksel olarak kanıtlandı. Soru filtresi kuralıyla (Kol-B: 6.653 tekil havuz, `?` oranı %100) model eğitimi ve doğrulama koşulları test edildi (val top-1 **0,9933**, train 500/500/500, state_dict SHA `c8ced1f0…` bit-özdeş determinizm); donmuş `data/anka_router.pt` başarıyla korundu (commit `366f5a3`, 8/8 GEÇTİ, damga `2026-09-28T17:23:05Z`).
  - **Operatör kararı (28 Eyl 2026, AskUserQuestion):** filtre **VAZGEÇİLDİ** — madde "uygulanabilir-değil" olarak kapatıldı (Kol-A kanıtı + Kol-B tanı-koşumu '?'-metinlerin eğitimde sorun çıkarmadığını gösterdi; kapanış-kaydı notes T-0161.md, damga `17:32:40Z` BETİKTEN). Şartname-yazım kusuru kaydı: kesin-çıkarma kuralı havuz-envanteri ölçülmeden sabitlenmişti (İLAN-formülasyon-kusuru dersi, şartname-yazarı tarafında).
- **T-0162 — TUR-A: `test_model.py` Bayat-Default Onarımı + p5_probe Kaydı:**
  - `test_model.py:25-26` bayat default'lar gündem-kanonik çıpayaya taşındı: `data/kristal_model.pt` (SİLİNMİŞ) + `data/vocab.json` (31.357) → `data/anka_base_v2.pt` (`d0f415f3…`, canlı-gateway'inki birebir) + `data/rebuild/vocab_anka_r1_33114.json` (**33.114** külliyat); CLI `--checkpoint/--vocab`, fallback zinciri ve `resize_state_dict` yolu aynen korundu. Koşum kanıtı (RC=0, MPS, sözlük `33114`, 4+4 olgu, hüküm-dışı): `data/eval/t0162_test_model_kosum_2026-09-28.log` (143 satır).
  - `data/eval/p5_probe_archive.jsonl`'daki T-0145 probe kalıntısı (+1 satır, append-only kanıt-günlüğü) REVERT yerine kayıt-commit ile kapatıldı (İLAN `t0162_tura_ilan_20260928.md`, damga `17:35:51Z` BETİKTEN).

## [1.9.0] - 2026-09-16

### Eklendi (Added)
- **F2 — Bayat `.bin`'lerin yeni sözlükle yeniden derlenmesi (eski artefaktlar korunarak):**
  - `data/train_chat_balanced.bin` yeniden derlendi (sözlük **32.852** + `literal_entity_mode=True`): 3.120.000 token, **max id 32.850**, noktalama bloğu **180.336**, kesme 39.706 — eski artefaktta noktalama **0** idi. Bağımsız yeniden koşum **bayt-özdeş** (T-0044).
  - `data/train_balanced_sft_v2.bin` **yeni ad** olarak 7 jsonl kaynaktan derlendi: 1.793.433 token, noktalama 180.336; eski `train_balanced_sft.bin` **korundu** (T-0045). 8. kaynak (`future_train_vector.jsonl`, diskte 0 B) kullanıcı kararıyla **hariç** tutuldu.
  - Eğitim hattı `_v2`'ye geçirildi; model artık **yüklenen sözlükten** kuruluyor (32.852) ve `resize_state_dict` +36 satır padding'i `[SOZLESME_UYARI]` ile **görünür** biçimde yapıyor (T-0046).
- **Donmuş-yazım kapıları:** `run_goal_pipeline.py`, `train.py`, `train_dpo.py`, `retrain_clean_models.py` yazımdan **önce** `check_frozen_save_path` çağırıyor ve operatör onayı (`--allow-frozen-write`) istiyor; kanonik yardımcı `src/llm/frozen_guard.check_frozen_save_path` (T-0047…T-0049).
- **`<PAD>` kayıp maskesi (`train.py`):** eğitim kaybında dolgu token'ları **varsayılan olarak maskelenir** (`--no-pad-mask` ile kapanır). Kontrollü ikiz deney (aynı taban/veri/adım, yalnız bayrak farkı): maskesizde `<PAD>` en sık üretilen token (**39-65**), maskelide **0** (T-0053).
- **F3 — tabana yetkinlik tamamlama:** önce **yedek** (`scratch/f3_backup_2026-09-16/`, 5 checkpoint, sha-özdeş + `torch.load` ile açılabilirlik kanıtı), sonra 2×1000 adımlık koşumlar (`kristal_model_f3_sft.pt`, `kristal_model_f3.pt`) ve maskeli yeniden koşum (`kristal_model_f3_clean_sft.pt`, `kristal_model_f3_clean.pt`) (T-0050/T-0052/T-0053/T-0054).
- **Test:** `tests/test_pad_loss_mask.py` — maskenin **iki yönlü** kanıtı (maskeli kayıp PAD konumlarına duyarsız, maskesiz duyarlı) + regresyon → paket **195 passed**.

### Bilinen sınırlar (Known limitations)
- `kristal_model_f3.pt` **`<PAD>` yozlaşması** taşır (maskesiz reçeteyle eğitildi) → **F4 ceket eğitiminin tabanı olarak kullanılmamalı**; maskeli yeniden koşum (`f3_clean`) esas alınmalı.
- **Kayıp değerleri maske durumuna bağlıdır:** aynı rejimde maskeli **1,5836** / maskesiz **6,8515** (~4×). Maskeleme öncesi kayıplarla (5,3955 / 3,6530 / 2,4404 / 0,9447) **doğrudan kıyaslama yapılmamalı**.
- Eğitim koşumları sırasında makinede GPU tüketen başka iş çalıştırılmamalı: aynı makinede tarayıcı/video açıkken adım süresi **0,43 → 3,45 sn/adım**'a çıktı (T-0052 ölçümü, nedeni kullanıcı tarafından teyit edildi).

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
