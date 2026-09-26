# G6a (T-0137) İLAN — Bahçıvan Dikeyi — 2026-09-26 (claude devralma)

Bu dosya İLANDIR, RAPOR DEĞİLDİR (ILAN ≠ RAPOR — G4 kusur sınıfı). Ölçümler koşum
ÖNCESİ ilanlanır; hükümler BETİKTEN yazılır, elle sayı yoktur. Taban: `data/anka_base_v2.pt`
(`d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`, MUHÜRLÜ 2.0-sealed —
T-0136; mimari 6/6/768/RoPE/tying-false DOKUNULMAZ). Şartname: ceket_tasarim_2026-09-26.md §8.

## İLAN-1 — TABAN-ÇIPA (koşum başlıyor)

- **Amaç:** eğitilmemiş muhürlü tabanın kanonik kapta beklendiği gibi KÖR olduğu kanıtı
  (taban-çıpa deseni; eğitilmemiş model ROUGE 0,0000 — T-0036/T-0106 çıpası).
- **Koşum:** kanonik kap `scripts/evaluate_carpenter_anka.py` DOKUNULMAZ; `--model
  data/anka_base_v2.pt --output scratch/t0137/taban_cipa_sonda.json` (kanonik varsayılanlar:
  vocab `data/rebuild/vocab_anka_r1_33114.json` çıpa zinciri, lexicon, heldout, n=100,
  seed=42, device=mps, max-new=128). Sandbox DIŞI (MPS gizli). Tek süreç (lsof temiz).
- **AYRAÇ (koşum-öncesi ilan):** sonda JSON ROUGE-L alanı betikten okunur;
  `ROUGE-L < 0,005` → **TABAN_CIPA_GECTI** · `≥ 0,005` → **TABAN_CIPA_DUSTU**
  (koşum durmaz ama fine-tune BAŞLAMAZ — kök neden ölçülür). Eşik TAVAN/ESIK
  sabitlerine (0,3221 / 0,9509) DOKUNMAZ.
- **Kabul koşusu artefaktı notu:** çıktı `scratch/` altında; data/eval'e yazım yok.
- **Düzeltme (ölçüm ÖNCESİ, 26 Eyl 11:05Z):** ilk koşum `--ceket-ekseni` olmadan yapıldı
  ⇒ betik `metrik["olculdu"]=False`, `hukum={}`, ROUGE ÜLÇÜLMEDİ (betikten okundu:
  üretim ROUGE yalnız `--ceket-ekseni` dalında üretilir — `evaluate_carpenter_anka.py`
  K2 bölümü). İlan İLK halinde "kanonik varsayılanlar" yazmıştı — kanonik TABAN-ÇIPA
  koşumu `--ceket-ekseni` bayrağıyla tanımlıdır (T-0036/T-0106 çıpa koşumlarıyla aynı).
  İlk sonda JSON (b891afc02a1ff55e29cd8ea3bcc690676f31abd255e6f3d3fc985f5e7d547820)
  ROUGE-siz olarak kanıta saklandı; koşum `--ceket-ekseni` ile TEKRARLANIYOR. Ayraç
  DEĞİŞMEDİ: `ROUGE-L < 0,005 → TABAN_CIPA_GECTI`.

## İLAN-2 — ARZ ÜRETİMİ PİLOT KAPISI (hüküm DAMGALI: PILOT_GECTİ — betikten, 2026-09-26; ayrıca rapor: bahcivan_g6a_arzpilot_sonuc_2026-09-26.md)

**ARAÇ GÜNCELLEME (operatör bildirimi 2026-09-26 ~12:40Z: "Ollama hazır değil; küçük
modeller set edilmiş — haiku, gpt-oss:120b; istersen bir dene"):** pilot aracı genişletildi —
gpt-oss:120b OLLAMA'da hazır değil (sunucu devre dışı); pilot claude subagent aracılığıyla
**haiku** modeliyle koşulur (4 subagent × 5 çift = n=20). gpt-oss:120b, Agent aracının
seçilebilir model listesinde YOK (ölçüldü: yalnız sonnet/opus/haiku/fable) — pilot dışı
beyan. **EŞİKLER DEĞİŞMEDİ** (araçtan bağımsız ilanlı ayraçlar); hüküm BETİKTEN.

**ARAÇ KİMLİĞİ notu (ölçüm sonrası beyan, 2026-09-26 16:35 — operatörün Ollama
ayarları ekran görüntüsünden okundu):** "haiku" etiketi Claude katmanıdır; Ollama
Claude entegrasyonu bunu **gemma4:31b-cloud**'a eşler (Sonnet 5 → glm-5.3-flash:cloud
— bu oturum). Fiilî üretim motoru Ollama bulut zinciridir; pilot ve üretim koşumları
bu eşlemeyle koştu. Sınıflandırıcı "timed out" arızaları `:cloud` model
meşkulliğidir. Eşikler araç kimliğinden bağımsızdır (İLAN-2 ilk beyanı geçerli).

**ARAÇ KİMLİĞİ düzeltmesi (operatör gözlemi, 2026-09-26 16:45):** operatör Ollama
ayar ekranında TÜM isteklerin glm-5.3-flash:cloud'a yazıldığını bildirdi —
`model: haiku` parametresi fiilî eşlemeye gitmiyor olabilir (gemma4:31b-cloud
devrede görünmüyor). Beyan şu şekilde düzeltildi: subagent üretimleri Claude
katman etiketi "haiku" ile İSTEMLENDİ, fiilî motor operatörün gözleminde
glm-5.3-flash:cloud'dur. Eşikler ve hüküm değişmez.

Arz üretimi aracı gemini-2.5-flash'tan (kota tükendi) yerel OLLAMA'ya geçiş adayıdır
(`192.168.1.14:11434`, operatör bildirdi: meşkul). Pilot koşum AŞAĞIDAKİ eşiklerle
yapılacaktır; pilot eşikten geçmeden arz külliyatına GİRİLMEZ (kayıt YOK, "üretti" ≠ "uygun"):

- **Örneklem:** pilot n=20 (alan: bahçıvanlık/bitki bakımı; D3-istisna kalıbı:
  soru-kalıp + "Ahşap uzmanı olarak cevapla." STİLİ UYARLANMIŞ sistem zarfı + cevap).
  İlanlı kalıp aileleri (şablonlar, `{K}` = gerçek bahçe konusu):
  1. nedir: "`{K}` nedir?" · 2. nasil: "`{K}` nasıl yetiştirilir?" ·
  3. ne_zaman: "`{K}` ne zaman budanır?" · 4. hastalik: "`{K}` hangi hastalıklara
  yakalanır?" · 5. islev: "`{K}` bahçede ne işe yarar?" — 20 örnek 5 aileye dağılır
  (4+4+4+2+2); aile eşikte "≥ 5 farklı aile" ölçülür. *(Dağılım yazımı düzeltildi —
  ölçüm SONRASI, 2026-09-26: "4+4+4+2+2" toplamı 16 eder, aritmetik hataydı; aile
  dağılımı AYRAÇ DEĞİL, eşik yalnız aile sayısı ≥5'tir. Ölçülen dağılım 5+5+5+2+3 —
  rapor: bahcivan_g6a_arzpilot_sonuc_2026-09-26.md, dürüst kayıt 1.)*
- **AYRAÇ (betik hükmü):**
  1. `kalip_koruma >= 18/20` (%90) — her örnek tanımlı soru-kalıbına + cevap-sonlandırma
     düzenine oturur; boşluk/tekrar/dağılım-dışı son yok.
  2. `kalip_cekimlilik`: soru-kalıp sayısı ≥ 5 farklı aile (T-0125 dersi: tek kalıp homojenliği
     kilit doğurur; ağırlık dengesi §5/G4 hükmü).
  3. Türkçe jeton dökümü/başka dil karışımı: 0 örnek (mutlak).
- Hüküm dalları: **PILOT_GECTI** (arz üretimi bu araçla açılır) · **PILOT_YETERSIZ**
  (kayıt; araç değişir veya arz ertelenir — KAYIT YAPILMAZ). Pilot sonucu RAPOR'da
  (ayrı dosya), ILAN burada tek kez.
- Gemini kotası dönerse karşılaştırmalı pilot (gemini vs OLLAMA aynı istemler) yapılabilir.

## İLAN-4 — BOŞ MODEL POZİTİF KONTROLÜ (operatör onayı 2026-09-26; FAZ-0 sonrası açıldı)

- **Amaç:** ölçütün ayırt ediciliği kanıtı — TABAN_CIPA_DUSTU hükmünün kök nedeni
  varsayım hatasıydı (base_v2 devralınmış); ROUGE ölçütünün kendisi boş modeli
  yakalamalıdır (taban-çıpa deseni: ölçüt için).
- **Boş model:** `KristalLM(33114, 768, blok 4096, 6 katman, 6 kafa)` rastgele ağırlık,
  tohum **42** (determinizm kuralı), `scratch/t0137/bos_model.pt` — checkpoint'i
  DEĞİŞMEZ, data/*.pt yazım YOK.
- **Koşum:** kanonik kap, `--model scratch/t0137/bos_model.pt --ceket-ekseni`,
  aynı varsayılanlar (n=100, seed 42, mps).
- **AYRAÇ (koşum-öncesi ilan):** `ROUGE-L < 0,005` → **POZITIF_KONTROL_GECTI**
  (ölçüt ayırt ediyor) · `≥ 0,005` → **POZITIF_KONTROL_DUSTU** (ölçüt körlemesine
  geçiyor — KÖK NEDEN ölçülür). Oracle kontrolü kimlik 1,0000 ister (koşumda
  fail-closed, betik `durdur`).

## İLAN-3 — DERLEME (arz derlendi: BIRLESIM_GECTİ; koşum ÖNCESİ damga, 2026-09-26)

İLAN-7 birleşim hüküm betikten **BIRLESIM_GECTİ** (rc=0): nihai külliyat
`data/pedagogy/bahcivan_arena.jsonl` — **2.199 çift** (`da7bdddd…`), alanlar
`soru/cevap/kaynak/aile`. Derleme İLAN'ı koşum ÖNCESİ (fine-tune tasarımın
taban-çıpa çıpası düzeltmesiyle birlikte İLAN-3b'de ayrıca damgalanacak):

- **Adapter sözleşmesi:** arena çifti → D3-istisna şeması (eski marangoz
  corpus şemasının — `{"instruction": "<zarf> Soru: …", "input": "", "output": …}`
  — birebir paraleli): `{"instruction": "Bahçıvan uzmanı olarak cevapla. Soru:
  {soru}", "input": "", "output": "{cevap}"}`. `kaynak`/`aile` adapter JSON'a
  GİRMEZ (şema üç alan; ara dosya `scratch/t0137/bahcivan_d3_adapter.jsonl`,
  2.199 satır, betikle digest'lenir).
- **Sistem zarfı (SFT normalize kaydı):** `system_prompt = "Bahçıvan uzmanı
  olarak cevapla."` — `"Ahşap uzmanı olarak cevapla."` paraleli
  (`tokenize_specialization_jsonl` parametresi; varsayılan DOKUNULMAZ).
- **Tokenizer kurulumu (kanonik kapla AYNI):** vocab
  `data/rebuild/vocab_anka_r1_33114.json` (`evaluate_carpenter_anka.py:55`
  `VOCAB_VARSAYILAN` — taban 33114 ister) · lexicon `data/lexicon/roots.tsv` ·
  `literal_entity_mode=True` (T-0134 marangoz kardeş bin kurulumu — bit-özdeşlik
  zinciri). Kanonik kod `src/llm.dataset_compiler` IMPORT edilir (kopya YASAK).
- **Çıktı:** `scratch/t0137/bahcivan_arena.bin` + `.meta.json` (data/ donmuş —
  scratch'ta; boundaries · total_records · total_tokens · sozluk_giris).
- **AYRAÇ (betik hükmü — `scratch/t0137/bahcivan_derle.py`):**
  1. parse hatası 0 · encode hatası 0 · normalize encode hatası 0 ·
     satır başına ≤2 kayıt (D4 invariant'ları fail-closed).
  2. Kayıt sayısı BETİKTEN; beklenen tavan 4.398 (2.199 ham + 2.199 SFT,
     hepsi len>2 ise; beklenen değer hüküm JSON'da betikten).
  3. **`bin_dekod_dogrula` ZORUNLU — 0 sapma kabul:** BOS/EOS/OUTPUT//OUTPUT
     sayımı == total_records · token toplamı == meta · sozluk_giris == 33114.
  4. SFT kayıtların zarf metni ("Bahçıvan uzmanı olarak cevapla.") betikle
     sayılır — 2.199 beklenir.
  Dallar: **DERLEME_GECTİ** (rc=0; fine-tune İLAN'ı ayrıca açılır) ·
  **DERLEME_SAPMA** (rc=2; kök neden ölçülür, fine-tune BAŞLAMAZ).
- `ROL_ZARFI` (`prompt_contract.py:72`, marangoz metni) DOKUNULMAZ — eval zamanı
  prompt zarfı ayrı bir sözleşme; fine-tune İLAN'ında ayrıca beyan edilir.
- Fine-tune tasarımı (önceki İLAN-3 damgası, taşındı): taban DONUK; LM bedeli
  kapısı **0,0230** (operatör onayı 2026-09-26: çıpa base_v2'nin KENDİ ROUGE'si
  0,1392); tutarsızlık çıpa %5; `--rol-zarf` beyanlı (0,1342 zarflı / 0,1354
  nötr çıpa); oversampling tavanlı (G4 hükmü). Checkpoint `scratch/` altında;
  doğrulama sonrası silinir. MPS disiplini: tek eğitici; koşum-öncesi `lsof` kanıtı.

## İLAN-5 — TAM ARZ ÜRETİMİ (pilot PILOT_GECTİ sonrası; koşum ÖNCESİ damga, 2026-09-26 13:05Z)

- **Hedef boyut:** **2.115 çift** — marangoz külliyatının (10.575, T-0134 kanıtı)
  **%20'si** (operatör emri 2026-09-26: "Marangozu %20 geçsin"; yorum operatör
  onayıyla: "%20'si kadar" — AskUserQuestion 2026-09-26).
- **Aile dağılımı (dengeli, T-0125/T-0126 dersi):** nedir 425 · nasil 425 ·
  ne_zaman 425 · hastalik 425 · islev 415 (toplam 2.115).
- **Koşum:** pilot ile AYNI araç ve kalıp şablonları (İLAN-2); subagent başına
  50 çift; dosyalar `scratch/t0137/arz_uretim_<aile>_<NN>.jsonl`; şema
  `{"soru","cevap"}` (pilotla birebir); her subagent 50 FARKLI konu (tema
  listesi: sebze/meyve/süs/ağaç/zararlı/hastalık/toprak/gübre/tohum/araç …).
- **AYRAÇ (betik hükmü, `scratch/t0137/arz_uretim_olc.py` — elle sayı YOK):**
  1. Pilotun üç ayracı TÜM külliyatta: kalip_koruma ≥ %90 (≥ 1.904/2.115),
     aile sayısı = 5, dil karışımı 0 (mutlak).
  2. **YENİ — birebir soru tekrarı ≤ %5** (≤ 106 çift): bağımsız subagent
     çakışma tavanı; üstünde aşım bölümü ölçülür ve KAYIT YOK.
  3. Konu-tekrarı (aynı konu farklı ailelerde) BEKLENİR ve hükümsüz aday
     beyanı olarak ölçülür (aynı konunun ailelerce sorulması arzu edilir).
- Hüküm dalları: **ARTZ_URETIM_GECTI** (doğrulanmış çiftler
  `data/pedagogy/bahcivan_arena.jsonl`'a birleştirilir — writes[] kiralanmış)
  · **ARTZ_URETIM_YETERSIZ** (aşım bölümü ayrı külliyata KAYIT YAPILMAZ;
  yeniden üretim ilanı ayrıca).
- Sistem zarfı üretim istemlerine YOK (pilotla aynı; derleme fazında İLAN-3
  gereği T-0134 zarfıyla uygulanır — İLAN-2 dürüst kayıt 2 ile tutarlı).

## İLAN-6 — YENİDEN ÜRETİM (İLAN-5 hüküm dalı: ARTZ_URETIM_YETERSIZ; koşum ÖNCESİ damga, 2026-09-26)

İLAN-5 hüküm betikten **ARTZ_URETIM_YETERSIZ** (rc=1): n 2.115 tuttu, kalıp
2.115/2.115, aile 5, dil 0 — ama **birebir soru tekrarı 274 (%12,96) > %5 eşiği
(≤106)**. Rapor: `bahcivan_g6a_arzuretim_sonuc_2026-09-26.md`; aşım bölümü
betikten ölçüldü (`arz_asim_analiz.py` → `arz_asim_bolumu.json`): **242 tekrarlı
soru → 274 fazla kopya** (ne_zaman 85 · hastalik 63 · nasil 59 · islev 38 ·
nedir 29). Külliyat `data/pedagogy/bahcivan_arena.jsonl`'a BİRLEŞTİRİLMEDİ.

- **Yeniden üretim hedefi:** 274 aşım çifti + %20 tampon = **330 yeni çift**;
  aile dağılımı aşım oranlarına göre: ne_zaman 102 · hastalik 76 · nasil 71 ·
  islev 46 · nedir 35. Dosyalar `scratch/t0137/arz_uretim_<aile>_<NN>.jsonl`
  DEVAM numaralarıyla (hastalik_10…, ne_zaman_10… vb.); şema ve kalıp
  şablonları değişmez; YENİ KONULAR (mevcut külliyatta geçen birebir sorular
  tekrarlanmaz; konu-tekrarı hükümsüz).
- **Birleşim kuralı (deterministik, betik hükmü `arz_birlestir.py`):** birebir
  soru tekrarında İLK görülen kopya kalır (dosya-adı + satır-sırası
  sıralaması); aşım çiftleri birleşime GİRMEZ (İLAN-5 dalı ile tutarlı);
  yeniden üretim çiftleri eklenir; çıktı `data/pedagogy/bahcivan_arena.jsonl`
  (kiralanmış writes[]).
- **AYRAÇ (birleşim hükmü, İLAN-5 ayracıyla AYNI):** nihai külliyatta n ≥ 2.115,
  birebir soru tekrarı ≤ %5, kalip_koruma ≥ %90, aile = 5, dil 0 (mutlak);
  parse hatası 0. Hüküm dalları: **BIRLESIM_GECTI** (fine-tune fazı İLAN-3
  açılır) · **BIRLESIM_YETERSIZ** (birleşim geri alınır — data/pedagogy çıktısı
  silinmez, KAYIT YAPILMAZ; kök neden ölçülür). Elle sayı YOK.
- Ölçüm betiği `arz_uretim_olc.py` DOKUNULMAZ (ilk hüküm kanıtı olarak durur);
  nihai hüküm birleşim betiğinden yazılır (`bahcivan_birlesim_hukum.json`).

## İLAN-7 — İKİNCİ YENİDEN ÜRETİM (İLAN-6 birleşim dalı: BIRLESIM_YETERSIZ; koşum ÖNCESİ damga, 2026-09-26)

İLAN-6 birleşim hüküm betikten **BIRLESIM_YETERSIZ** (rc=1): girdi 2.445 → 350
çıkarıldı (274 eski aşım + yeni çakışma), birleşim **2.095 < 2.115 hedef (20
eksi)**; kalıp koruma 2095/2095, dil 0, yeni tekrar 0 — sorun YALNIZ ARZ.
`data/pedagogy/bahcivan_arena.jsonl` YAZILMADI (dal kuralı işledi). Kök neden
ölçümü (betik): 330 yeni çiftin **74'ü eski külliyatla birebir soru çakıştı**
(ne_zaman 44 · nasil 11 · nedir 10 · islev 9) + 5 yeni-içi tekrar; tamponun
büyük payı çakışmaya gitti.

- **İkinci yeniden üretim hedefi:** **120 çift** — 20 n eksiği + 100 tampon;
  aile dağılımı kök neden ölçümüne göre: ne_zaman **40** · nasil **30** ·
  nedir **20** · islev **15** · hastalik **15**. Dosyalar
  `arz_uretim_<aile>_13/14.jsonl` DEVAM numaralarıyla; şema ve kalıp
  şablonları değişmez; subagent istemlerinde ÇAKIŞMA UYARISI sertleştirilir.
- **Birleşim betiği `arz_birlestir.py` DOKUNULMAZ** — ayracı zaten n ≥ 2.115 ·
  birebir tekrar ≤ %5 · kalıp ≥ %90 · aile = 5 · dil 0; hüküm dalları
  **BIRLESIM_GECTI** (data/pedagogy/bahcivan_arena.jsonl yazılır; fine-tune
  fazı İLAN-3 açılır) · **BIRLESIM_YETERSIZ (tekrar)** — yazım YOK; kök neden
  ölçülür; tampon/hedef revizyonu OPERATÖR kararına gider. Elle sayı YOK.

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · tokenizer + `src/compiler/**`
DOKUNULMAZ · kanonik eval betiği DOKUNULMAZ · elle sayı YOK · `git add -A` YASAK ·
commit operatör kapısıdır.