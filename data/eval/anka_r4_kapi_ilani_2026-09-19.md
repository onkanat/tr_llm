# Anka A1-r · Faz 4 kapı İLANI (ölçümden ÖNCE yazıldı)

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Yürütücü:** claude
**Plan:** `enchanted-wiggling-moon.md` §Faz 4 · **Durum:** BU BELGE ÖLÇÜMDEN ÖNCE YAZILDI.

> Bu belge, Faz 4 külliyat derlemesinin kapılarını **ölçüm başlamadan** sabitler.
> Ölçümden sonra eşik **değiştirilmez**; bir eşiğin yanlış ilan edildiği
> anlaşılırsa bu **rapora** yazılır ([[tasarim-sayisi-betikle-hesaplanmali]]).

---

## 1. A1'e göre DEĞİŞEN girdiler

| Girdi | A1 | A1-r | sha256 |
|---|---|---|---|
| sözlük (lexicon) | `data/lexicon/roots.tsv` (DONMUŞ) | `data/lexicon/roots_anka_r1.tsv` | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| taban lexicon (referans) | — | `data/lexicon/roots.tsv` (okunur, değişmez) | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| vocab | `data/rebuild/vocab_base_32852.json` | `data/rebuild/vocab_anka_r1_33114.json` | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| vocab N | 32.852 | **33.114** | — |
| `literal_entity_mode` | `False` | `False` (DEĞİŞMEDİ) | — |

**Süperset özdeşliği (ölçüldü):** `stoi(33036) ⊂ stoi(33114)`, fark **tam 78**.
Yeni 78 kimlik il/il adı sınıfıdır (`Adıyaman`, `Ardahan`, `Bartın`, `Gaziantep`,
`Giresun`, `Hakkâri`, `Isparta`, `Iğdır`, `Kars`, `Ermenice`, …).

---

## 2. K1–K7 — A1 ile **birebir** aynı (değiştirilmedi)

| Kapı | Eşik | Not |
|---|---|---|
| K1 noktalama | `max_id ≥ 32145` **ve** noktalama oranı `≥ %5,0` | `max_id` artık 33.113 |
| K2 `<UNK>` | `≤ %4,0` | akış ölçümü |
| K3 `<PAD>` | `≤ %1,0` | yazılan `.bin` |
| K4 ölçek | train `≥ 90.000.000` jeton | |
| K5 tekrarsızlık | train∩val belge id = 0, `benzersiz_belge_id > 0` | |
| K6 izlenebilirlik | her kaynak parquet'in sha256'sı var | |
| K7 pozitif kontrol | noktalama hattı CANLI **ve** `literal=False` DEVREDE | fail-closed, rc≠0 |

### 2.1 YENİ — K6b köken (planın ölçülmüş provenance boşluğu)

A1 üreticileri K6'da yalnız parquet kaynaklarını, meta'da yalnız **sözlüğü**
hash'liyordu; **`roots.tsv` hiçbir kapıda, meta alanında veya sha256'da yoktu**
⇒ bir roots değişikliği **sessizce görünmezdi**. K6b bunu kapatır:

| Alt kapı | İddia |
|---|---|
| K6b-1 | meta'da `lexicon_sha256`, `lexicon_taban_sha256`, `sozluk_sha256` **dolu** |
| K6b-2 | koşum **sonunda** üç digest de başlangıçtakiyle **birebir** aynı |
| K6b-3 | donmuş taban `roots.tsv` digest'i ilan edilen `fe3005e5…` ile aynı |

---

## 3. K8 — **YENİDEN İLAN** (eski K8 D2 başarısında DÜŞER; bu kapının kusurudur)

Eski K8 (`data/eval/anka_a1_corpus_design_2026-09-18.md:74`): `PN ≥ %1,0`.
Bu bir **tabandı** (hattın canlı olduğunun kanıtı), kalite tavanı değil. D2 başarılı
olursa PN **düşer** ve eski K8 **düşer** — başarısızlık değil, kapı kusuru.
Eski metin **değiştirilmeden** duruyor; yeni ilan burasıdır:

> **YENİ K8 TEMSİL:** `ENT_adet == 0` **VE** `CAP_adet == 0` **VE** `PN_orani ≥ %0,5`

### 3.1 ⚠️ Plandaki `literal_ad_orani ≥ %4,0` maddesi — **UYGULANAMAZ** (ilan edilen kusur)

Plan Faz 0'ın yeni K8 metni `… VE literal_ad_orani ≥ %4,0` diyordu. Ama
`LITERAL_ENTITY = False` iken üretici **hiçbir literal-ad bloğu üretmez** ⇒ bu terim
**değerlendirilemez** (payda tanımsız). Kapıya sessizce katmak, sessizce düşürmek
veya eşiği gizlice çevirmek yerine burada **beyan edilir**: madde **kapsam dışıdır**,
gerekçesi budur. Yerine `PN_orani` **ayrıca** raporlanır ve **G3**'e karşı okunur.

---

## 4. G1…G6 — yazılan `.bin`'den ölçülür (kaba vekil yasak)

`G1` ve `G6` **derleyici** ölçümleridir (yüzey gerekir); diğerleri jeton sınıfıdır ve
**yazılan artefakttan** okunur. `G1`/`G6` için örneklem, "önce" değerleriyle
kıyaslanabilir olsun diye **aynı örnekleyici** (`scratch/anka_r1_on_kapsam.py`'nin
r4 kopyası: aynı 14 temizleme kuralı, aynı `MIN_BELGE_KAR=300`, aynı
`HEDEF_KABUL_PER_FILE=1500`, aynı iki parquet) ile üretilir.

| Kapı | Eşik | Yön | ÖNCE (ölçüldü) |
|---|---|---|---|
| G1 en-uzun-eşleşme ihlali | ≤ %1,0 | ↓ | **%2,3789** |
| G2 `<UNK>` oranı | ≤ %1,5 | ↓ | **%1,5733** |
| G3 `<PROPER_NOUN>` oranı | ≤ %3,0 | ↓ | **%8,0186** |
| G4 literal ad / yer tutucu | ≥ 1,0 | ↑ | **0,3753** |
| G5 sözlük kapsamı (gözlenen/giriş) | ≥ %95 | ↑ | **%86,3** (train) · %47,73 (val) |
| G6 tur özdeşliği (TAM+ALTERNATİF) | ≥ %99,0 | ↑ | **%96,9232** |

### 4.1 ⚠️ G5'in ilan edilen YÖNÜ D2'nin başarısıyla TERS (ilan edilen kusur)

`G5 = benzersiz_gözlenen_id / vocab_giriş`. **Payda büyüdüğü için** D2 başarılı
oldukça G5 **mekanik olarak düşer** (32.852→33.114 giriş; eklenen 262 kimliğin çoğu
seyrek). Bu, ölçümden **önce** beyan edilir: **G5 düşerse bu, değişikliğin değil
ilanın kusurudur.** Eşik değiştirilmez; sonuç olduğu gibi raporlanır ve G5'in
**iki payı da** (gözlenen id, giriş) yazılır ki okur kendisi hüküm verebilsin.

### 4.2 Çürütme maddeleri (her kapı için: "hangi sonuç cümlemi çürütür?")

| Kapı | Çürütme |
|---|---|
| G2/G3 | PN düşerken **`ANALIZSIZ` oranı artarsa**, kazanç yer tutucudan değil **sınıf kaymasından** gelir ⇒ hüküm `AYIRT EDEMEDİ` |
| G3 | PN düşüşü **yalnız eklenen 262 kimliğe** karşılık gelen yüzeylerden gelmiyorsa (yani düşüş başka bir mekanizmadan geliyorsa) çürür; bu yüzden **PN kütlesi iki parçaya ayrılır**: (a) yeni kimlikli yüzeyler, (b) geri kalan |
| G4 | LIT artışı **PN düşüşü olmadan** gelirse (yani adlar zaten literaldi) oran mekanik şişer ⇒ pay ve payda **ayrı ayrı** raporlanır |
| G1/G6 | Ölçüm **kapı geçmez** ve **ayrık çift sayısı 0** ise hüküm `GEÇMEDİ` değil **`KANITSIZ`** ([[tavan-artefakti-kapi-gecmez-kanitsizlik]]) |
| Tümü | Örneklem "önce" ile **birebir aynı** değilse kıyas geçersiz; örneklem kimliği (`dosya_bası`, `bin_jeton`) tabloya yazılır |

---

## 5. Değişmeyenler (fail-closed)

- `data/anka_a1.pt` · `data/anka_a1_pretrain.bin` · `data/anka_a1_pretrain_val.bin` ·
  `data/anka_pretrain.bin` — **okunur, üzerine YAZILMAZ**; koşum başında ve sonunda
  digest karşılaştırılır.
- `SOZLUK_BEKLENEN = 33114` **bilinçli ve beyan edilerek** güncellendi (32.852'den);
  `_next_id == 33114` fail-closed kapısı korunur ⇒ tokenizer'ın sessizce id eklemesi
  **düşürür**.
- Çıktı yolları **yeni**: `data/anka_a1r_pretrain{,_val}.bin` (T-0072 raporunu
  bayatlatmamak için; projenin kendi emsali a0b→a1 geçişidir).

---

## 6. EK — Faz 5 (yeniden eğitim) kapıları, koşumdan ÖNCE ilan edildi

Faz 4 kapı ilanı yazıldığında Faz 5 ayrıntılanmamıştı. Aşağısı **koşumdan önce**
yazılmıştır ve sonradan değiştirilmez.

### 6.1 Faz 5'in girdisi olan DEĞİŞİKLİK — `train.py` (T-0080'de bulunan kusur)

**Bulunan kusur (koşumdan önce, ölçüldü):** `train.py:49` sözlük yolunu
`data/rebuild/vocab_base_32852.json` olarak **sabit kodluyordu**; A1-r külliyatı ise
**33.113**'e kadar jeton kimliği taşır. `scripts/train_step_demo.py:20-39`
`KristalDataset` **hiçbir sınır kontrolü yapmaz** (`self.data[i:i+block_sz]` doğrudan
embedding'e gider). İki sessiz arıza mümkündü: (a) koşumun ortasında `RuntimeError`,
(b) MPS'in geçersiz indekste **hata vermeyip çöp değer döndürmesi** (T-0073'te ölçüldü).
İkisi de 12,6 saatlik koşuyu boşa çıkarırdı.

**Düzeltme (varsayılan davranış bit-bit korunur):**
`--vocab <yol>` argv geçersiz kılması (varsayılan **aynı**: `vocab_base_32852.json`)
+ `KristalDataset` yüklendikten **sonra** fail-closed sınır kontrolü
(`max jeton id < vocab_size` değilse `RuntimeError`)
+ kardeş `<bin>.meta.json` varsa `sozluk_giris`/`sozluk_sha256` **birebir** karşılaştırması;
meta alanları yoksa **sessizce geçilmez**, `ATLANDI` diye **basılır**.

**İkinci bulunan kusur (aynı aileden) — CIHAZ KAPISI.** `train.py:36-45`'te
`--device mps` istendiğinde MPS yoksa **sessizce CPU'ya düşüyordu**; tek ayırt edici
sinyal `"CPU modu seçildi"` satırıydı. **Ölçüldü (bu makinede, bu oturumda):**
`torch.backends.mps.is_available()` → sandbox **içinde `False`**, **dışında `True`**.
Yani sandbox içinden başlatılan bir koşum 12,6 saati CPU'da yakardı (T-0075'in
birebir tekrarı: *okunamayan sinyalde durmayan koruma kapı değildir*). Düzeltme:
açıkça istenen cihaz kullanılamıyorsa **`RuntimeError`**; `--device cpu` ve
varsayılan davranış **değişmedi**.

**Kontroller (`scratch/anka_r4_sinir_kontrol.py`, beş yol, hepsi `rc` ile):**

| Kontrol | Girdi | İlan edilen sonuç | Ölçülen |
|---|---|---|---|
| N1 meta uyuşmazlığı | gerçek A1 `.bin` + 33.114 sözlük | DURDURULDU | **DURDURULDU** (`sozluk_giris=32852 != 33114`) |
| N2 sınır ihlali | sahte `.bin`, id 65.000, meta **doğru** | DURDURULDU | **DURDURULDU** (`id=65000 >= vocab_size=32852`) |
| N3 meta alanı yok | sahte `.bin`, meta yalnız `block_size` | GEÇTİ + `ATLANDI` bastı | **GEÇTİ + ATLANDI** |
| P1 temiz | sahte `.bin`, id<taban, meta birebir | GEÇTİ + `BIREBIR` bastı | **GEÇTİ + BIREBIR** |
| N4 cihaz kapısı | `--device bozuk-cihaz` | DURDURULDU | **DURDURULDU** |

**Beş yolun beşi de ilan edilen davranışı verdi.** N2/N3 birlikte "kapı hem
düşebiliyor hem geçebiliyor"u kanıtlar; N4 ortamdan **bağımsız** bir addır
(sandbox içi/dışı aynı sonucu vermek zorundadır). N2'de id 65.000 seçildi çünkü
jeton tipi `uint16` (tavan 65.535).

`train.py` sha256: **`b87961af…`** (önce) → **`940b22b929eb576d68719fa1ce5cae1a72e043ee87ed5ec02656c485fd042c48`** (sonra).
`venv/bin/pytest`: **214/214 yeşil** (sandbox içinde `socket.bind` yasağı yüzünden 1
test düşer; sandbox **dışında** 7/7 geçti ⇒ kusur test kabında, kodda değil).

### 6.2 Canlılık imzası — MODA GÖRE ilan edilir (sıcak başlangıç)

T-0077'de ölçüldü: **sıfırdan** koşumda ilk kayıp ≈ ln V = canlıdır; **devam**
koşumunda tam **tersi** — ≈ ln V ise model sessizce sıfırdan başlamıştır. Bu koşum
**sıcak başlangıçtır** (`--load-path data/anka_a1.pt`), dolayısıyla:

| Kapı | İlan edilen | İhlalinde hüküm |
|---|---|---|
| **L1 (fail-closed)** | `ilk_kayıp < 8,0` | tavan **T-0077'nin devam koşumu için ilan ettiği tavanın AYNISI** (yeniden kullanıldı, uydurulmadı). Aşılırsa sıcak başlangıç kaybolmuş ⇒ ALARM, koşum durdurulur |
| **L1′ (mutlak taban)** | `ilk_kayıp < ln(33114) − 1,0 = 9,4077` | ikincil; `≈ ln V` sessiz-sıfırlama imzasıdır (T-0077) |
| **L2 (rapor, kapı değil)** | son 60 adım ort ± std, **n=60** | A1'in kendi son-60 düzeyi `3,9122 ± 0,2147` ve **ilk kaybı `4,3107`** ile **kıyaslanır**; külliyat ve sözlük **farklı** olduğu için bu **gösterge**, kapı değil |

**L1'in beklenen büyüklüğü ilan edilir (koşumdan önce, kaba hesap):** A1-r külliyatında
yeni kimlik bloğunu (262 id) kullanan konum oranı ≈ **%2**; o konumlar ilk adımda
rastgele satırlardan geçer ⇒ yapısal katkı ≈ `0,02 × (ln V − 4,3) ≈ +0,12` nat.
Dolayısıyla beklenen ilk kayıp ≈ **4,0–4,6** bandında; `8,0` tavanı bu bandın çok
üstünde ama `ln V`'nin çok altında ⇒ **hem sıfırlamayı yakalar hem yanlış ALARM üretmez**.

**L2'nin neden kapı olmadığı ilan edilir:** A1'in `3,9122`'si A1 külliyatında
ölçüldü; A1-r külliyatı D2 kimliklerini **literal** yapar ⇒ aynı metnin kayıp
düzeyi **yapısal olarak** değişir. Tek sayıdan bant kurmak
([[bitis-kaybi-tek-adim-orneklemi]]) bu projede **zaten bir kez** yanlış ALARM üretti.

### 6.3 Öğrenme oranı ve mod — açıkça verilir (sessiz geri düşme yasak)

`train.py:143` devam koşumunda `lr`'yi **sessizce** `2e-4`'e düşürür; `--lr` argv
döngüsü **ondan sonra** koştuğu için açık `--lr` kazanır. Bu koşumda `--lr` **açıkça**
verilir ve `--pretrain` **zorunludur** (düz-metin külliyatta SFT maskesi her pencereyi
öldürür → gradyan 0 → MPS `0,0000` basar; T-0073).

### 6.4 Değişmezler (fail-closed)

- `data/anka_a1.pt` · `data/anka_a1_pretrain.bin` · `data/anka_a1_pretrain_val.bin` —
  **okunur, üzerine YAZILMAZ**; koşum başı/sonu digest karşılaştırılır.
- Koşum sırasında makinede **GPU tüketen başka iş çalıştırılmaz** (ölçüldü:
  0,43 → 3,45 sn/adım; 2-3× sapma **yük sinyalidir**).
- AdamW momentleri **kaydedilmiyor** ⇒ her devam ağırlıktan ısınır (T-0077).
