# Unutma Ölçümü Tasarımı — DÜZELTİLMİŞ ZEMİN (18 Eyl 2026)

**Damga:** 2026-09-18T03:30Z · **Yazan:** claude (danışman) · **Statü:** TASARIM — **17 Eyl sürümünün YERİNE geçer**
**Önceki sürüm:** `data/eval/forgetting_measurement_design_2026-09-17.md` (zemini kirli olduğu için **geçersiz**)
**Dayanak:** Onkanat kararı (18 Eyl): *"önce ölçüm zeminini düzelt"* + *"A′/B′ tutulmuşluğu kanıtlanacak"* + *"C beş nokta için tam kafayla yeniden ölçülecek"*
**Görev:** T-0062

---

## 1. Neden düzeltildi (denetim bulguları, her biri ölçüldü)

### 1.1 A ve B dilimi **ezber dilimiydi** (asıl kusur)
A ve B, `data/train_balanced_sft_v2.bin` üzerinde ölçülüyordu. Bu dosya, `f3_clean`'in atası `f3_clean_sft`'in **run_a eğitim korpusunun kendisidir** — kanıt: `T-0054`, satır 21,
`train.py --device mps --data data/train_balanced_sft_v2.bin --steps 1000 --load-path data/kristal_model.pt --save-path data/kristal_model_f3_clean_sft.pt`.
Dosyada "tutma bölgesi" yoktur; eğitim `torch.randint` ile **tüm dosyadan** pencere çeker.

**Ölçülen sonuç:** tabanın bu dilimdeki CE'si **1,9625**; tutulmuş genel metinde **6,0655** (gts_dict) / **5,7032** (raw_full) — yani **3,1 kat**. Aracın "A" dediği şey ezberin geri kalanına ne kadar bağlı kaldığıydı, genel dil yetkinliği değil. Ezber diliminde CE tabanın dibinde olduğu için **her sapma göreli olarak şişiyordu.**

**Bağımsız teyit (benim ölçümüm değil):** `data/eval/forgetting_baseline_2026-09-17.json` satır 10, **aynı dilimde eğitilmemiş** `taban` checkpoint'i için CE **5,4899** ve top-1 **%42,4** kaydediyor. O dilimde eğitilmiş `f3_clean` ise **1,9625** / **%84,4**. Yani `f3_clean`'in **tutulmuş** metindeki CE'si (5,70 / 6,07), dilime **hiç dokunmamış** bir modelin **o dilimdeki** CE'siyle (5,49) aynı düzeyde. Bu, dilimin ezberlendiğini bağımsız bir ölçümle gösterir.

**Bu, hükmü tersine çevirdi:** eski zeminde A hiçbir özelleştirme kolunda geçmiyordu (+%23,7 … +%42,1); düzeltilmiş zeminde %5/%10/%25 geçiyor (bkz §4).

### 1.2 Tasarım dokümanı ile uygulama çelişiyordu
17 Eyl dokümanı dilimi "son %10" diye tarif ediyordu; uygulanan offset `100000 + b·1000` (dosyanın başından ~%5,6). **Düzeltildi:** yeni zeminde dilim tam olarak tarif edilir (§3).

### 1.3 C aracı **kırpılmış kafayla** ölçüyordu
`scripts/evaluate_carpenter_generation_100.py:58` `data/vocab.json` (**31.357**) yükler; beş checkpoint'in kafası da **32.852** satır → **1.495 satır kırpılıyor** (nadir sözcükler + noktalama 32.137–32.145 + rakamlar + `<ALL_CAPS>`).

Kırpılmanın **iki** sapması ölçüldü: (i) *girdi* — kanonik sözlükle 20 marangoz isteminin 830 token'ının **81'i (%9,8)** kırpılmış modelin temsil edemediği id'de; (ii) *çıktı* — kırpılmış üretimlerde noktalama **0**. Kırpılan aralıkta **morfolojik etiket yok** (TENSE_PAST=22, COPULA_AORIST=115, POSS_3SG=13, CASE_LOC=29 → hepsi düşük id).

### 1.4 C aracı **yanlış tokenizer sözleşmesini** kullanıyordu (yeni bulgu)
`scripts/evaluate_carpenter_generation_100.py:62`, `KristalTokenizer(compiler, vocab)` — `literal_entity_mode` **varsayılan `False`**. Oysa kanonik **veri üreticilerinin tamamı** `True` kullanır:

| site | değer |
|---|---|
| `scripts/prepare_balanced_sft_dataset.py:109` | `True` (varsayılan) |
| `scripts/prepare_chat_balanced_dataset.py:134` | `True` (varsayılan) |
| `scripts/prepare_carpenter_specialization_dataset.py:209` | `True` (varsayılan) |
| `scripts/recompile_datasets.py:45,61,75` | `True` |
| `scripts/run_goal_pipeline.py:246,261,360` | `True` |
| `src/llm/prompt_contract.py:36` | `False` (**servis** yolu) |

→ Model `True` sözleşmesiyle üretilmiş veriyle eğitildi; C aracı istemi `False` sözleşmesiyle tokenize ediyordu.

### 1.5 Yöntem bulgusu: **hizalı blok imzası bu iş için geçersiz**
Görev spec'i tutulmuşluğu "hizalı 128-token blok imzası (blake2b-64)" ile kanıtlamayı öngörüyordu. 17 eğitim `.bin`'ine karşı **altı adayın hepsinde ortak blok imzası 0 çıktı** — 32-gramlarının **%51,2'si** eğitimde olan `b1_5_splits/test.jsonl` için bile. Sebep: eğitim `.bin`'leri kayıtları 128-token bloklara doldurup hizalıyor, aday metinler ise hizasız duruyor. **Yerine stride-1 32-gram (sıraya duyarlı polinom hash) kanalı geçti.**

---

## 2. Yeni eksenler ve **ÖNCEDEN İLAN EDİLEN** eşikler

| Eksen | Ne ölçülür | Zemin | Eşik (bu dokümanla, **sonuç görülmeden** ilan) |
|---|---|---|---|
| **A′. Genel dil kaybı** | CE (maskesiz **ve** maskeli, aynı pencereler, tek forward) | **Tutulmuş**, iki bağımsız korpus: birincil `gts_dict`, ikincil `raw_full` | **Göreli artış ≤ %10** (17 Eyl'in eşiği, düzeltilmiş zemine aynen taşındı) |
| **B′. Noktalama yetkinliği** | Hedefi noktalama olan konumlarda top-1 | **Tutulmuş** `raw_full` (birincil; seçim ölçütü §3.2) | **Düşüş ≤ 5 puan** (17 Eyl'in eşiği, aynen) |
| **C. Ceket alan içi kazanç** | `t0062_carpenter_eval_full.py` — **tam kafa + kanonik tokenizer sözleşmesi** | marangozluk held-out (100 istem) | **Taban çizgisine göre iyileşme > 0** |

**Eşiklerin kaydırılmadığının kanıtı:** A′ ve B′ eşikleri 17 Eyl dokümanındaki sayıların **aynısıdır**; değişen tek şey **zemin**dir. C'nin mutlak kapısı (betiğin kendi hükmü: ezber <%10, tutarsızlık <%5, ROUGE ≥ 0,35, kesişim ≥ %80) **ayrı bir sorudur** ve hiçbir kol onu geçmez; F4 kararı için işleyen ölçüt tabana göre **iyileşmedir**.

**B′ eşiğinin okunuşu (iki okuma ayrışıyor — açıkça kayda geçirilir):** orijinal kural "top-1 düşüşü **≤ 5 puan**" yani **mutlak**tur. Aynı kural yeni zeminde de mutlak uygulanır. Alternatif **göreli** okuma (tabanın %94'ünü koru) taban %84,4 iken %5 puanlık düşüşe karşılık geldiği için türetilebilir, ama yeni tabanda (%52,9) **daha sıkı** olur. Her iki okuma da raporlanır; hüküm **mutlak** okumaya göre verilir (birebir ilan edilen kural).

---

## 3. Zeminin kurulumu

### 3.1 Tutulmuş korpus üretimi ve kanıtı
`scratch/t0062_corpus_lab.py` — 17 eğitim `.bin`'inin **tümü** stride-1 32-gram (sıraya duyarlı polinom hash, `h = Σ_k T[t_{i+k}]·BASE^k mod 2^64`) ve hizalı 128-blok imzasıyla taranır. Aday kayıtlar **tek tek** hash'lenir; "temiz" = **32-gram örtüşmesi sıfır**.

| aday | token | örtüşen 32-gram | kirli kayıt | TEMİZ+≥65 token | hüküm |
|---|---|---|---|---|---|
| `data/raw_corpus.txt` | 797.299 | %0,257 | %1,22 | 2.038 (208.359 tok) | kısmi temiz |
| `data/poems/gts.json` | 1.657.381 | %0,768 | %1,63 | 4.261 (1.218.030 tok) | kısmi temiz |
| `data/anythingllm_chats_cleaned.json` | 94.607 | %0,000 | %0 | 56 | **tam temiz** |
| `data/pedagogy/arena…` | 70.100 | %31,66 | %84,5 | 9 | kirli |
| `data/b1_5_splits/val.jsonl` | 112.374 | %46,64 | %61,1 | 10 | kirli |
| `data/b1_5_splits/test.jsonl` | 120.696 | **%51,22** | %60,9 | 9 | kirli |

**Yan bulgu:** donmuş `b1_5_splits/` test bölmesi **token düzeyinde %51,22** eğitimle örtüşüyor — bilinen cevap-düzeyi sızıntı bulgusunu (T-0012, %43,4) doğrulayıp büyütüyor.

**Kayıtlar birleştirilmedi** (birleştirme, kayıt sınırını aşan ve taranmamış yeni 32-gramlar üretirdi). `≥65 token` kısıtı A′ penceresinin (SEQ 64) gereğidir.

### 3.2 B′ zemini neden `raw_full`
Seçim ölçütü **kollara değil, tabanın aracı çalıştırıp çalıştırmadığına** bakar: `gts_dict`'te tabanın top-1'i **%10,6** — 6 sınıf için şans düzeyi %16,7'nin **altında**, yani taban bu beceriyi sergilemiyor (taban etkisi). `raw_full`'da **%52,9**, şansın belirgin üstünde. Tablo her iki zemin için de raporlanır.

### 3.3 Pencereler
`ri = (b·BS+i)·37 mod N`, `s = (b·11+i·5) mod (len−SEQ)`. **Aynı pencereler beş checkpoint için de kullanılır** → karşılaştırma **eşleştirilmiş**tir ve GA eşleştirilmiş bootstrap ile alınır.

---

## 4. Sonuçlar (düzeltilmiş zemin) ve karar

### 4.1 A′ — genel dil kaybı (tutulmuş zemin, iki bağımsız korpus)

Dört kol da **tutulmuş** metinde unutuyor; ama miktar örtüşme oranıyla **monoton** azalıyor:

| kol | A′ `gts_dict` (birincil) | tabana göre | A′ `raw_full` (ikincil) | tabana göre | eşleştirilmiş GA95 (gts) | p(Δ>0) |
|---|---|---|---|---|---|---|
| %0 `carpenter_v2` | 7,0336 | **+%15,96** | 6,5496 | +%14,84 | [+0,9569, +0,9796] nat | 1,00 |
| %5 `f4_r05` | 6,5495 | **+%7,98** | 6,2076 | +%8,84 | [+0,4746, +0,4937] nat | 1,00 |
| %10 `f4_r10` | 6,3320 | **+%4,39** | 5,9741 | +%4,75 | [+0,2565, +0,2767] nat | 1,00 |
| %25 `f4_replay` | 6,3122 | **+%4,07** | 6,0018 | +%5,24 | [+0,2375, +0,2558] nat | 1,00 |
| taban `f3_clean` | 6,0655 | — | 5,7032 | — | — | — |

**Hüküm (A′, eşik ≤ +%10):** **%5, %10 ve %25 GEÇİYOR; %0 KALIYOR.**

İki bağımsız korpusta işaret **ve** büyüklük tekrarladı; dört kolun dördünde de eşleştirilmiş GA sıfırı geniş marjla dışlıyor (`p = 1,00`) → unutma gürültü değil, ve kollar arası sıralama da istatistiksel olarak anlamlı (GA'lar birbirleriyle kesişmiyor).

**Bu, 17 Eyl hükmünü tersine çevirdi.** Eski (ezber) zeminde A hiçbir oranda geçmiyordu (+%23,7…+%42,1). Sebep: tabanın o dilimdeki CE'si **1,9625** — kendi dibi; oradan her sapma göreli olarak şişiyordu. Tutulmuş zeminde taban CE'si 6,0655 (gts) / 5,7032 (raw), yani **3,1 kat** yukarıda; ölçüm artık ezberin geri kalanına bağlılığı değil, **gerçek genelleme kaybını** ölçüyor.

### 4.2 Maskeli/maskesiz çapraz kontrol

A′ iki eksende ölçüldü (aynı pencereler, tek forward): pencerenin 64 hedefinin tamamı (*maskesiz*) ve yalnız son 32'si (*maskeli*, gradyan görmeyen istem bölgesi dışarıda). İki eksen **aynı yönde ve aynı büyüklükte** (|fark| ≤ 0,02 nat) → ölçüm PAD/istem artefaktı değil.

### 4.3 C — ceket alan içi kazanç (tam kafa + kanonik sözleşme)

| kol | ezber % | tutarsızlık % | ROUGE-L | kesişim % | ΔROUGE (tabana göre) |
|---|---|---|---|---|---|
| taban `f3_clean` | 0,0 | 8,0 | 0,3425 | 49,0 | — |
| %0 `carpenter_v2` | 0,0 | 8,0 | 0,3796 | 54,0 | **+0,0371** |
| %5 `f4_r05` | 0,0 | 8,0 | 0,3667 | 52,0 | **+0,0242** |
| %10 `f4_r10` | 0,0 | 7,0 | 0,3625 | 53,0 | **+0,0200** |
| %25 `f4_replay` | 0,0 | 13,0 | 0,3554 | 52,0 | **+0,0129** |

**Hüküm (C, iyileşme > 0):** **dört kolun dördü de geçiyor**; kazanç ceket oranıyla monoton artıyor (%0 > %5 > %10 > %25).

Betiğin kendi **mutlak** kapısı (tutarsızlık <%5, ROUGE ≥ 0,35, kesişim ≥ %80) **beş kolun hiçbirinde** geçmiyor — taban dahil. Bu ayrı bir sorudur (ölçeğin kendisi bu tabanda tutmuyor) ve F4 kararı için işleyen ölçüt değildir.

**Sözleşme düzeltmesinin etkisi ölçüldü ve küçük:** tam kafa `False` → `True` ROUGE'u en çok +0,004 oynatıyor ve **sıralamayı değiştirmiyor**. Materyal olan sapma kırpılmış kafaydı (taban 0,0651 → 0,3425); sözleşme düzeltmesinin asıl görünür etkisi **üretilen metinde** — kırpılmış koşumda noktalama **0**, tam kafada noktalama geri geliyor.

### 4.4 B′ — noktalama yetkinliği: **eksen geçerli değil** (ölçümle bulundu)

**Önce örnekleme kusuru düzeltildi ve ölçüldü.** Eski `punct_heldout` ilk `cap` konumu **dosya sırasından** alıyordu:

| korpus | kayıt | eski kapsam | yeni kapsam (yayılmış) |
|---|---|---|---|
| `gts_dict` | 4.261 | 1.001 konum / 129 kayıt (**%3,0**) | 1.000 konum / 1.000 kayıt (%23,5) |
| `raw_full` | 2.038 | 1.004 konum / 154 kayıt (**%7,6**) | 1.000 konum / 1.000 kayıt (%49,1) |

**Cihaz izolasyonu (karışık eksen değil):** tazeleme koşusu `device=cpu` raporladı, öncekiler `mps`. Tek bir checkpoint MPS'te yeniden koşuldu ve CPU koşumunu **birebir** üretti (top-1 29,2 · ort. rank 78,8 · A′ 6,5496) → B′'deki büyük değişim **cihazdan değil, örneklemeden** geliyor.

Düzeltilmiş örneklemeyle B′ (top-1 %):

| kol | `gts_dict` | `raw_full` (birincil) | düşüş (puan) | goreli koruma |
|---|---|---|---|---|
| taban `f3_clean` | 18,3 | **26,1** | — | — |
| %0 `carpenter_v2` | 20,9 | **29,2** | **−3,10** (iyileşme) | %111,9 |
| %5 `f4_r05` | 18,3 | 23,5 | +2,60 | %90,0 |
| %10 `f4_r10` | 16,8 | 24,2 | +1,90 | %92,7 |
| %25 `f4_replay` | 17,0 | 22,5 | +3,60 | %86,2 |

**İlan edilen (mutlak) kurala göre dört kolun dördü de B′'yi geçiyor.** Göreli okumada yalnız %0 geçiyor. İki okuma üç kolda ayrışıyor (yukarıdaki tablo) — bu ayrışma §2'de öngörülmüştü.

> ⚠️ **Bu tablo GEÇERSİZ eksene aittir** (aşağıda gerekçesi). Özellikle **`%0 −3,10 puan / %111,9 iyileşme`** satırı **onarılmış ve geçerli** eksende (T-0063, dengeli 3-sınıf kısıtlı top-1) **tersine döner**: %0 orada iki korpusta da tabanın **altında** ve kayıp **anlamlı** (raw −1,011 puan p=5,5e-03 · gts −3,867 puan p=5,6e-20). Ham top-1'deki "iyileşme" **çoğunluk-id bedava kazancının** artefaktıdır. Kaynak: `data/eval/t0063_b_prime_axis_2026-09-18.json`. **Düzeltme turu: T-0065** — `data/eval/t0065_verification_processing_2026-09-18.md`.

**AMA eksenin kendisi geçersiz çıktı — ve bu, örnekleme düzeltmesiyle ortaya çıktı.** Top-1 doğru sayımı, örneklemdeki **çoğunluk id'nin payı kadar bedava** kazanır:

| korpus | örnekleme | sabit-tahmin tabanı (= çoğunluk payı) | model tabanı top-1 |
|---|---|---|---|
| `gts_dict` | eski | %47,7 (`,`) | %10,6 |
| `gts_dict` | yeni | %54,8 (`,`) | %18,3 |
| `raw_full` | eski | %57,7 (`,`) | %52,9 |
| `raw_full` | yeni | **%70,1** (`.`) | **%26,1** |

**Dört durumun dördünde de taban, "hep aynı işareti tahmin et" diyen sabit bir tabanın ALTINDA.** Yani B′ ekseni, tabanın **hiç sergilemediği** bir beceriyi ölçüyor; üzerinde "koruma oranı" hesaplamak anlamsızdır. Bu bir **tanı**dır; §2'de ilan edilen kapı **değiştirilmedi** ve yukarıdaki geçiş hükmü aynen raporlanıyor — ama **B′ artık unutma hükmü taşıyamaz.**

**Eski "raw_full'da dört kolun tamamı tabanın altında" bulgusu bir ÖRNEKLEME ARTEFAKTIYDI** (129/154 kayıtlık dar bölge). Düzeltilmiş örneklemede %0 kolu tabanın **üstünde** (29,2 > 26,1). Bu, `scratch/t0062_ab_probe.py`'nin `punct_heldout` çekilişinin kusuruydu; ölçüldü, düzeltildi ve eski sayılar duyarlılık kontrolü olarak saklandı.

### 4.5 Karar tablosu (ilan edilen kurallarla)

| kol | A′ `gts_dict` (≤ +%10) | B′ `raw_full` (düşüş ≤ 5 puan) | C (ΔROUGE > 0) | A′+C |
|---|---|---|---|---|
| %0 `carpenter_v2` | **+%15,96 KALDI** | −3,10 puan GEÇTİ *(tanı: geçersiz)* | +0,0371 GEÇTİ | **KALDI** |
| %5 `f4_r05` | +%7,98 GEÇTİ | +2,60 puan GEÇTİ *(tanı: geçersiz)* | +0,0242 GEÇTİ | **GEÇTİ** |
| %10 `f4_r10` | +%4,39 GEÇTİ | +1,90 puan GEÇTİ *(tanı: geçersiz)* | +0,0200 GEÇTİ | **GEÇTİ** |
| %25 `f4_replay` | +%4,07 GEÇTİ | +3,60 puan GEÇTİ *(tanı: geçersiz)* | +0,0129 GEÇTİ | **GEÇTİ** |

**Hüküm:** Tutulmuş zeminde **%5, %10 ve %25 unutmayı kabul edilebilir düzeyde tutuyor**; **%0 tutmuyor**. C kazancı ceket oranıyla monoton **artıyor**, unutma monoton **azalıyor** → gerçek bir Pareto cephesi var; **%5 (`f4_r05`) A′'yı geçen kollar arasında en yüksek C kazancına sahip.**

> **DÜZELTME (bağımsız doğrulama, 18 Eyl 04:17Z — bulgu B1):** Yukarıdaki "dirsek" ifadesi **istatistiksel olarak desteklenmiyor**; bu düzeltme doğrulayıcının bulgusudur ve kabul edilmiştir. C ekseninde n=100 held-out üzerinde eşleştirilmiş bootstrap (95% GA):
> - r05 (%5) vs r10 (%10), fark +0,0662 → GA **[−0,0257, +0,1581]** — **sıfırı kesiyor**
> - r05 (%5) vs taban, fark +0,0574 → GA **[−0,0314, +0,1462]** — **sıfırı kesiyor**
> - v2 (%0) vs r05 (%5), fark +0,0257 → GA **[−0,0055, +0,0570]** — **sıfırı kesiyor**
>
> Yani C ekseninde **%5 ile %10 ve %5 ile taban ayırt edilemiyor**; sıralama yalnız **nokta tahminlerine** dayanıyor. **%5 tercihi katı bir istatistiksel sınır değil, ortalama skorlar üzerinden seçilmiş bir Pareto çalışma noktasıdır.** Karar tablosu okunurken bu böyle okunmalıdır: "C'de %5 en iyi" değil, "C'de %0–%25 arasındaki farklar bu n ile ölçülemedi". C'yi kesinleştirmek istiyorsa **n artırılmalıdır** (ayrı iş).

**B′ geçersiz olduğu için hüküm A′ ve C üzerinde durur.** (B′'nin yerine yeni bir kapı sonradan **kondu**: T-0063 onarılmış B′ ekseni — aşağıdaki karar notuna bakınız.)

> **KARAR VERİLDİ (kullanıcı, 18 Eyl 2026): F4 çeket noktası = `%5 f4_r05`. F4 kararı KAPANDI.**
> Tam kayıt: **`data/eval/f4_decision_2026-09-18.md`**. Özet: %5, üç ilan edilen eşikten de
> **değiştirilmeden** geçiyor (A′ ≤+%10: +%7,98/+%8,84 · onarılmış B′ ≥%90: %98,6/%101,1 ·
> C ΔROUGE>0: +0,0242). %0 A′'dan kaldığı için elendi; %10 ve %25 de geçiyor ama hiçbir eksen
> onları %5'e tercih ettirmiyor. `f4_r05` üç ölçümün de aleyhine olmadığı tek koldur.

**Karar kuralı (17 Eyl'den devralındı, değişmedi):**
- **A′ ve B′ eşikleri İÇİNDE** → unutma kabul edilebilir düzeyde; yalın F4 yeterli.
- **A′ veya B′ AŞILDI** → unutma var; F4 kompozisyonuna chosen/SFT kütlesi tekrarı eklenir (ayrı tur, kullanıcı onayıyla).
- **C iyileşmedi** → ceket öğrenilmemiş; F4 reçetesi (adım/lr) yeniden kurulur.

**Ayrıca raporlanır, kapı değildir:** eşleştirilmiş GA (95%), maskeli/maskesiz eksen farkı, B′ top-3 ve ortalama rank, B′ id-kırılımı, C istatistikleri.

---

## 5. Araçlar (hepsi `scratch/`, görev T-0062)

| Araç | İş |
|---|---|
| `t0062_corpus_lab.py` | Tutulmuş korpus adayı üretimi + 32-gram/blok örtüşme taraması |
| `t0062_ab_probe.py` | A′/B′ ölçümü; `--basis old` (pozitif kontrol) ve `--basis heldout` |
| `t0062_carpenter_eval_full.py` | C (tam kafa; `--vocab` ve `--literal-entity-mode` ile iki sapma kapatılır) |
| `t0062_verdict.py` | Eşikleri uygular, karar tablosunu üretir |

**Pozitif kontrol (araç sadakati):** `t0062_ab_probe.py --basis old`, `scratch/forgetting_probe.py`'nin dilimini birebir kopyalar ve beş checkpoint'te kayıtlı A/B sayılarını 4 ondalığa kadar üretti (1,96256 / 2,78903 / 2,42820 / 2,68475 / 2,56169 ve %84,4 / %74,8 / %79,2 / %80,8 / %79,2) → farklar zeminden gelir, araçtan değil.

**Araç kusurları (ölçülüp düzeltildi):** (i) kayıt başına `np.isin` 39M diziyi her çağrıda yeniden sıralıyordu → 28+ dk kilitlenme; `np.searchsorted` ile değiştirildi, tarama 28 sn'ye indi. (ii) İlk XOR hash sıraya duyarsızdı → 3,77M ayrı n-gram'ı (%8,8) yanlışlıkla birleştiriyordu; polinom hash ile değiştirildi.
