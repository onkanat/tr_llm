# VG-2 — Vektörel Gezgin v1: Yetenek Ağı (yetenek_agi) — 2026-11-04

## İLAN

VG-2 uygulanır: kullanıcı mimarisi (memory_archive/vektorel_gezgin_arch.md) gereği
statik skill.md/istem yüklemesi REDDEDİLİR; sistemin GERÇEK yetenekleri 36
yetenek kartı olarak **vektör uzayında** yaşar, sorgu YAKINLIĞI (kosinüs) ile
AKTİVASYON olur ve ağırlıklı etki (ağırlık × benzerlik) skoru belirler. Yeni
`yetenek_agi` Qdrant koleksiyonu (anka_bellek/muhakeme_bellek/simulasyon_bellek
DOKUNULMAZ), TDD (RED önce), idempotent tohumlama + read-back, 5-sorgu demo
probu, gateway'e YALNIZ-GÖZLEM kancası. Commit YAPILMAZ (controller'a aittir).

## HÜKÜM

**GEÇTİ.** Kanıtlar (hepsi koşum-çıktısı, uydurma yok):

| Kapı | Ölçüm | Sonuç |
|---|---|---|
| Koleksiyon tohumlama | `seed_yetenek_agi.py --url http://192.168.1.9:6333` | 36 kart yazıldı, **öncesi 0 → sonrası 36** (read-back: `points_count=36, status=green`) |
| Idempotensi | ikinci tohumlama (birim testi, yerel Qdrant) | sayı 36→36 SABİT (nokta kimliği sha256('yetenek:<kimlik>') 2^63-mod) |
| Demo aktivasyon | `yetenek_probe.py` 5 sorgu, eşik=0,02, top_k=5 | **hit@1: 5/5, hit@3: 5/5**; determinizm: aynı sorgu 2. koşumda birebir aynı tablo |
| TDD | `tests/test_yetenek_agi.py` (QDRANT_URL'li koşum) | **34/34 GEÇTİ** (gerçek koleksiyon dumanı dahil; Qdrant'sız koşumda duman SKIP) |
| Regresyon | tam süit `venv/bin/python -m pytest tests/ -q` | **583 passed, 1 skipped, 1 xfailed, 0 failed** (4:26) |
| Gateway kancası | kanca testleri + mevcut gateway/MoE/router testleri | kanca kapalıyken anahtar YOK (davranış birebir); açıkken gözlem eklenir; hatalıysa None |

## RAPOR

### 1) Tasarım — `src/rag/yetenek_agi.py` (yeni modül, ~450 satır)

- **(a) `YetenekKarti`** (frozen dataclass): `{kimlik, ad, açıklama/kapsam,
  etki_alanı, agirlik, meta}`; `doc_text()` embedding'e giren kanonik metin.
- **(b) Kayıt defteri**: `YetenekAgi` sınıfı, `yetenek_agi` Qdrant koleksiyonu
  (kalıp: `seed_muhakeme_bellek.py`; şema `VectorMemory._create_hybrid_collection`
  → dense 768 Cosine + sparse IDF). **Ad-kapısı**: farklı koleksiyonlu bellek
  kurulumda ValueError (SERT KURAL koda gömülü — yanlış koleksiyona yazılamaz).
  `tohumla()` idempotent upsert (nokta kimliği = sha256('yetenek:<kimlik>')
  2^63-mod) + read-back sayısı döner.
- **(c) Aktivasyon motoru**: sorgu → tokenize → `generate_kristal_vector(768)` →
  `dense_recall` (kosinüs) → **saf çekirdek** `aktivasyon_seti(adaylar, esik,
  top_k)`: eşik süz (>= dahil) → etki = agirlik × skor → sırala
  ((-etki, -skor, ad) — beraberlikte ad kırıcı) → top-k.
- **(d) Sonuç**: `{"aktif_yetenekler": [(ad, skor, etki), ...], "karar_notu": str}`
  — karar notu deterministik (zaman damgası YOK).
- **Determinizm (tohum sabit)**: tokenizer deterministik; kristal vektör morfem
  tohumları md5-tabanlı sabit; sıralama anahtarı deterministik; aynı sorgu +
  aynı koleksiyon → birebir aynı sonuç (prob ve duman testinde KANITLANDI).
- **Eşik kalibrasyonu (ölçülmüş)**: ham skor bandı — ilgili kart isabetleri
  0,033–0,262; ilgisiz kuyruk ~0,005–0,024. `DEFAULT_ESIK=0.02`: beş isabetin
  tamamını alır (min 0,033 > 0,02), derin kuyruğu eler; kalan gürültüyü top-k
  sınırlar. (0,15 gibi yüksek eşik tek-nadir-kelime sorgularda ağı kör ederdi —
  ölçmeden eşik konmadı.)

### 2) Kart envanteri — 36 GERÇEK kart (uydurma yok)

Kaynak: `src/` + `scripts/` + `modules/` + `data/pedagogy/` envanteri. Her
kartın `meta['kanit']` alanı repoda VAR OLAN dosyadır —
`test_her_kartin_kaniti_repo_icinde_gercek` bu sözleşmeyi ZORLAR (uydurma kart
testte ölür).

Alan dağılımı: `derleyici:5, tokenizer:2, rag:7, router:2, alan:4, gateway:4,
veri:7, ölçüm:5`. Ağırlıklar: alan ceketleri 1,25 / çekirdek rag+tokenizer
1,10–1,15 / router-merak 1,0 / veri-ölçüm araçları 0,85–0,95.

Örnekler (ad · kimlik · kanıt): Kristal Derleyici (kristal_derleyici ·
src/compiler/core.py) · Kristal Tokenizer (kristal_tokenizer · src/llm/tokenizer.py)
· RAG Boru Hattı (rag_boru_hatti · src/rag/rag_pipeline.py) · Anka/Muhakeme/
Simülasyon Bellek (scripts/seed_*.py) · Üç-Modal Router (uc_modal_router ·
src/llm/router.py) · Merak Motoru (merak_motoru · src/rag/merak.py) ·
Marangoz/Bahçıvan/Legal/Pedagoji ceketleri (scripts/train_t0209_marangoz.py,
train_t0207_bahcivan.py, train_legal_ceket.py, generate_high_school_dataset.py)
· Ajan Kapısı (src/gateway/agent_gateway.py) · Otomatik Yeniden Eğitim
(src/gateway/retrain_pipeline.py) · GTS Tanım Çıkarıcısı
(scripts/extract_gts_definitions.py) · Mevzuat Getirici
(scripts/fetch_mevzuat_corpus.py) · Eşik Kalibrasyonu
(scripts/dogrulama_t0178_rrf_esik_olcum.py) · Donmuş Yol Bekçisi
(src/llm/frozen_guard.py) · … (tam liste: `kart_envanteri()`).

### 3) Testler — `tests/test_yetenek_agi.py` (TDD: RED önce — ilk koşum
ModuleNotFoundError ile RED; sonra GREEN)

7 küme, 34 test: kart alanları/donukluk/nokta-kimliği; envanter sözleşmesi
(20–40 kart, benzersiz kimlik/ad, kanıt-yolu repo'da VAR, kapsam kelimeleri);
saf çekirdek (etki aritmetiği, **ağırlığın benzerliği gölgelemesi**, eşik dahil
eleme, top-k kırpma, beraberlikte ad kırıcı, boş sonuç, determinizm, karar notu);
aktivasyon motoru (mock bellek — Qdrant'sız: yapı, determinizm, top-k/eşik,
boş sonuç, etki uçuşu, tokenizer'sız fail-closed); yerel tohumlama (ağ-sız
gerçek Qdrant storage_path: idempotent + read-back + tam-yol determinizm);
**gateway kancası** (kapalıyken anahtar YOK; açıkken gözlem eklenir; kanca
hatası ask'i bozmaz → None); **gerçek koleksiyon dumanı** (erişilemezse SKIP:
36 nokta ≥ 20, aynı sorgu 2× birebir eşit, skor (0,1], etki ≤ 2,0).

### 4) Demo probu — `scripts/yetenek_probe.py` (eşik=0,02, top_k=5, 36 kart)

| Sorgu | Aktif yetenek (sıra · skor · etki=ağırlık×benzerlik) |
|---|---|
| 'zımpara nedir' | 1. **Marangoz Alan Ceketi** 0,0326 · **0,0408** ←beklenen; 2. Koşullandırma Ölçümü 0,0427 · 0,0363; 3. Prompt Sözleşmesi 0,0351 · 0,0316 |
| 'iş kanunu maddesi' | 1. **Legal Alan Ceketi** 0,1551 · **0,1938** ←; 2. **Mevzuat Getirici** 0,1516 · 0,1440 ← (eşik altında kalan 3 kuyruk kartı elendi) |
| 'gideceğim kelimesini çözümle' | 1. **Kristal Derleyici** 0,2330 · **0,2563** ←; 2. **Kristal Tokenizer** 0,1553 · **0,1786** ←; 3. GTS Tanım Çıkarıcısı 0,0575 · 0,0518 |
| 'ceviz ağacı özellikleri' | 1. **Bahçıvan Alan Ceketi** 0,1078 · **0,1348** ←; 2. Mevzuat Getirici 0,1138 · 0,1081; 3. Merak Motoru 0,0727 · 0,0727 |
| "Qdrant'ta ne var" | 1. **Vektör Bellek Yönetimi** 0,2616 · **0,2878** ←; 2. Morfolojik Taktikler 0,2827 · 0,2827; 3. Kristal Tokenizer 0,1990 · 0,2289; 4. **Hibrit RRF Araması** 0,1982 · 0,2180 ← |

**Anlamlılık ölçümü: hit@1 = 5/5, hit@3 = 5/5.** Vizyon kanıtı (ağırlıklı etki ≠
ham benzerlik): 'zımpara nedir' sorgusunda Marangoz ceketi ham skorda (0,0326)
Koşullandırma Ölçümü'nün (0,0427) ALTINDA ama 1,25 ağırlığı etkiyi 0,0408 > 0,0363
yapıp kartı ZİRVEYE taşır — benzerlik değil, ağırlıklı etki belirler. Ham tablo:
`scratch/vg2_probe_ham.json` (eşik=0), `scratch/vg2_probe_final.json`.

### 5) Gateway gözlem kancası (davranış DEĞİŞİKLİĞİ YOK)

`src/gateway/agent_gateway.py` (minimal diff): (1) import
`KANCA_TOP_K`(=3); (2) `__init__`'e opsiyonel `yetenek_agi=None` parametresi;
(3) `ask()` sonunda korunan blok: `yetenek_agi` yoksa anahtar HİÇ eklenmez
(mevcut sözleşmeler birebir); varsa `sonuc["yetenek_aktivasyonu"] =
yetenek_agi.aktivasyon(sorgu=query, top_k=3)`; aktivasyon hatası ask'i
BOZMAZ → dürüst `None` + `[VG2_KANCA]` satırı. `scripts/baslat_canli_gateway.py`:
çözülmüş QDRANT_URL varsa kanca AÇIK (kart sayısı basılır), yoksa/bağlanamazsa
KAPALI raporlanır — başlatma DURMAZ.

### 6) Koleksiyon durumu + read-back

`http://192.168.1.9:6333/collections/yetenek_agi`: **points_count = 36,
status = green** (ekim öncesi 0; 36 kart upsert, 36 benzersiz nokta; örnek
read-back payload: kimlik='kristal_vektor_uretme' alan='tokenizer'
agirlik=1.15 kanit='src/rag/embedding.py'). Tekrar ekim idempotent
(kimlik-türeli nokta: çoğaltma yok).

### 7) Değişen dosyalar (commit YOK — controller'a devredilir)

- **YENİ**: `src/rag/yetenek_agi.py` (modül: kart/defter/motor/envanter);
  `tests/test_yetenek_agi.py` (34 test); `scripts/seed_yetenek_agi.py`
  (idempotent ekim + read-back); `scripts/yetenek_probe.py` (5-sorgu demo);
  bu rapor.
- **DEĞİŞTİ (minimal)**: `src/gateway/agent_gateway.py` (+1 import, +1
  opsiyonel param, +6 satır gözlem bloğu — ask() dönüş anahtar seti kanca
  kapalıyken birebir); `scripts/baslat_canli_gateway.py` (+15 satır,
  tamamen try/except-korumalı opsiyonel blok).
- **NOT (dürüst)**: paralel çalışan bir kardeş subagent (VG-1,
  `auxiliary_memories`) `agent_gateway.py`'yi 23:42'de kendi tabanından
  yeniden yazdı; kancam ve onun değişikliği o anda düştü. Kanca temiz tabana
  yeniden uygulandı ve 54 testle yeniden doğrulandı; VG-1'in
  `auxiliary_memories` satırları bu dosyada YOK — controller birleştirme
  sırasında ikisini birlikte görmelidir.

### 8) Vizyon → v1 eşleşmesi (dürüst)

Uyanan: yetenekler vektörde; aktivasyon sorgu-yakınlığıyla; ağırlıklı etki
karar veriyor; yalnız roots.tsv statik; determinizm; kartlar gerçek işlevlerden.

v1'de YOK olan (v2+ için açık borç): (1) **ağırlıklar evrimleşmiyor** — başlangıç
değerleri elle; kullanım-sıklığına/telemetriye göre otomatik ağırlık evrimi yok;
(2) aktivasyon **yalnız gözlem** — gateway/router yönlendirmesini henüz
DEĞİŞTİRMİYOR (v3 router'la bağlantı v2 işi); (3) kart envanteri koddaki
`kart_envanteri()`'den tohumlanıyor — canlıda yeni kart önerme/kısıt
(constraint) kartları yok; (4) aktivasyonda yalnız yoğun kosinüs kullanıldı —
seyrek/hibrit RRF yolu kart uzayında denenmedi.
