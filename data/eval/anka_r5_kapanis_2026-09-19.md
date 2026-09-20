# Anka A1-r · Faz 5 (yeniden eğitim) — KAPANIŞ RAPORU

**Tarih:** 19 Eyl 2026 (UTC) · **Görev:** T-0080 · **Yürütücü:** claude (tek yürütücü)
**Koşum kaydı:** `data/eval/anka_r5_kosum_2026-09-19.json` · **Ham log:** `scratch/anka_r5_kos.log`
**Plan:** `~/.claude/plans/enchanted-wiggling-moon.md`

> **Bu rapor ölçüm raporudur, anlatı değil.** Faz 0–4'ün eşzamanlı kaydı
> `data/eval/anka_r0*…r4*` serisidir; Faz 4 kapı raporu
> `data/eval/anka_r4_build_2026-09-19.md`. Buradaki her sayı ya `--kapanis`
> koşumunun JSON çıktısından ya log'un kendisinden **birebir** alındı.

---

## 0. Hüküm — önce sonuç

| Soru | Ölçülen cevap |
|---|---|
| Koşum tamamlandı mı? | **EVET** — `EĞİTİM BAŞARIYLA TAMAMLANDI`, `rc=0` |
| L1 canlılık kapısı | **GEÇTİ** (ilk kayıp 4,1704 < devam tavanı 8,0) |
| Değişmezler (A1 artefaktları) | **BİREBİR** — 4/4 digest aynı |
| Kayıp | son60 **3,6509 ± 0,2170** · son200 **3,6877 ± 0,2399** · medyan **3,7979** |
| Çıktı | `data/anka_a1r.pt` · **373.735.137 B** · `b93cc1cd…` (tam digest §4) |
| Süre | **45.440,01 sn = 12,62 sa** (ilan edilen ~12,6 sa **tuttu**) |
| Eğitim hedefi (kabul edilebilir sözlük zenginliği) | **KISMEN** — G1…G6 **5/5 KALDI** (Faz 4 raporu §12) |

**Dürüst cümle:** koşum **teknik olarak kusursuz** kapandı; ama bu, hedefin
tutulduğu anlamına **gelmez**. Faz 4'ün hükmü değişmedi: değişiklik her niceliği
istenen yöne taşıdı, **hiçbiri ilan edilen eşiği geçmedi** ve en büyük kaldıraç
(D1 en-uzun-kök) operatör kararıyla **kapsam dışı** bırakıldı.

---

## 1. Koşum künyesi

| Alan | Değer |
|---|---|
| Mod | **devam (sıcak başlangıç)** — `--load-path data/anka_a1.pt` |
| Adım | **48.828 / 48.828** · `--batch-size 8 --block-size 256` |
| Öğrenme oranı | `--lr 0.001` (**açıkça** verildi — `train.py:135,143` sessiz 2e-4 düşüşü engellendi) |
| `--pretrain` | **verildi** (zorunlu; yoksa SFT maskesi gradyanı sıfırlar, MPS `0,0000` basar) |
| Sözlük | `data/rebuild/vocab_anka_r1_33114.json` (33.114) · `f9940a8d…` |
| Külliyat | `data/anka_a1r_pretrain.bin` · 100 M jeton · `jeton_max = 33113` (< 33114 ✓) |
| Cihaz | MPS (sandbox **dışında** koşturuldu — sandbox MPS'i gizler) |
| Başlangıç / bitiş (UTC) | **2026-09-19T11:17:45Z → 2026-09-19T23:55:08Z** |
| `save-every` | 500 — son atomik kayıt adım 48828 |

Hız: **0,9305 sn/adım** ort. (45.440,01 / 48.828). Koşum boyunca makinede GPU
tüketen başka iş çalıştırılmadı (T-0052: yükte 0,43 → 3,45 sn/adım ölçülmüştü).

---

## 2. L1 — canlılık kapısı (koşum SÜRERKEN yazıldı)

| Alan | Değer |
|---|---|
| İlk kayıp (adım 1) | **4,1704** |
| Devam tavanı | **8,0** |
| Mutlak taban (soğuk koşum imzası) | 9,4077 |
| Hüküm | **GEÇTİ** |

**Neden bu bant:** T-0077 — canlılık imzası **koşum moduna bağlıdır**. Sıfırdan
koşumda ilk kayıp ≈ `ln V` (33.114 için 10,4076) **beklenir**; **devam** koşumunda
tam tersi — sıcak başlangıç *kaybolmuşsa* kayıp `ln V`'ye **sıçrar**. Bu koşum
devam modundadır, dolayısıyla ölçüt `≈ ln V` **değil**, `ln V`'nin **belirgin
altıdır**. 4,1704 bu bandın içindedir ve L1 kaydıyla **birebir** aynıdır.

---

## 3. Kayıp — SAYI değil DAĞILIM (T-0078)

`--loss-report` çıktısı, log'dan birebir:

```
Başlangıç Kaybı:   4.1704
Bitiş Kaybı:       3.9335
Kayıp Dağılımı:    son60 3.6509 ± 0.2170 (n=60)
                   son200 3.6877 ± 0.2399 (n=200)
                   medyan 3.7979 (n=48828)
```

⚠ **`Bitiş Kaybı` tek adımın örneklemidir** (`loss_history[-1]`). Bu koşumda o
sayı (3,9335), son60 ortalamasının (3,6509) **0,28 üstünde** — yani tek sayıyı
hüküm sanmak, sonucu **olduğundan kötü** okuturdu. Referans **dağılımdır**.

**Bağımsız çapraz kontrol (iki ayrı ayrıştırıcı):** `scratch/anka_r5_durum.py`
(kendi regex'i) ve kapanış öncesi hesapladığım pencere istatistikleri **aynı
son60 ortalamasını (3,6509)** verdi; log'daki adım satırı sayısı **4.883**
(`loglanan_kayip_sayisi`) ile `--kapanis`'in bulduğu sayı da uyuştu.

**Eğilim (kapanış öncesi, dört çeyrek ortalaması):**
3,9870 → 3,8085 → 3,7508 → 3,6994 · son çeyrek dışı kuyruk **3,5605** — tekdüze iniş.

> Ara ölçümlerden birinde `son60 = 3,9400` okundu ve bu, bir önceki raporda
> **gerileme gibi görünme riski** taşıyordu. Aynı pencere koşum sonunda
> **3,6860'a** indi ⇒ o okuma **pencere örneklemesiydi** (std ≈ 0,22–0,25),
> gerçek bir bozulma değil. Bu, tek pencereyi hüküm yapan her okumanın neden
> yasak olduğunun **bu koşumdaki canlı örneğidir**.

---

## 4. Değişmezler ve çıktı — TAM digest (önek değil)

`--kapanis`, koşumdan **önce** alınan `degismezler_once` ile **sonra** alınan
`degismezler_sonra` tablolarını **birebir** karşılaştırır. Hüküm: **BİREBİR**.

| Yol | Boyut (B) | sha256 (TAM) | Durum |
|---|---:|---|---|
| `data/anka_a1.pt` | 372.124.278 | `f32d492d9361c668bc86e4e4b0f99f5c473af552486837452475f8b0aa3eb5b4` | **değişmedi** |
| `data/anka_a1_pretrain.bin` | — | `383a9c890b75ec96195122b9981d7a73371d3bcaa3b9f6e7c2a188510d917cb8` | **değişmedi** |
| `data/anka_a1_pretrain_val.bin` | — | `02f52063e73e4ac362297c9ecad41247aa621e168572b9cb3f6d302f734da801` | **değişmedi** |
| `data/anka_pretrain.bin` | — | `0e55f0c0c24617a9323059c71563a9ce943177ccc90cd56824be1a7f26d2b164` | **değişmedi** |
| **`data/anka_a1r.pt`** (ÇIKTI) | **373.735.137** | **`b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293`** | yeni |

### 4.1 Sıcak başlangıç gerçekten devralındı mı? — ÖLÇÜLDÜ (tahmin değil)

İki checkpoint'in `state_dict`'i CPU'da yüklenip **anahtar anahtar** karşılaştırıldı:

| Ölçüm | a1 | a1r |
|---|---:|---:|
| Anahtar sayısı | 101 | 101 |
| Yalnız a1'de / yalnız a1r'de | `[]` / `[]` | — |
| Toplam tensor baytı | 372.089.168 | 373.699.944 |

**Şekli değişen tek üç tensor** (33.114 − 32.852 = **262** yeni satır):

| Tensor | (önce) → (sonra) | Bayt farkı |
|---|---|---:|
| `embedding.embedding.weight` | (32852, 768) → (33114, 768) | **+804.864** |
| `lm_head.weight` | (32852, 768) → (33114, 768) | **+804.864** |
| `lm_head.bias` | (32852,) → (33114,) | **+1.048** |
| **Toplam** | | **+1.610.776** |

Ölçülen tensor farkı **+1.610.776 B** — beklenen **+1.610.776 B** ⇒ **BİREBİR**.
Dosya boyutları arasındaki fark ise **+1.610.859 B**; kalan **83 B** pickle/zip
üstverisidir (tensor değil).

**Sonuç:** sıcak başlangıç **devralınmıştır** — anahtar kümesi birebir aynı, mimari
aynı, yalnız sözlük eksenindeki üç tensor büyümüştür. `lm_head.weight`'in **ayrı**
bir anahtar olarak durması weight-tying'in **olmadığını** da bağımsız olarak
doğrular (mimari değişmezi korunmuştur).

> **Kendi kusurum (kayda geçer):** bu maddenin **ilk sürümünde** mekanizmayı
> ölçmeden yazdım — `lm_head.bias`'ı atladım ve dosya-baytı farkını tensor-baytı
> farkıyla kıyaslayıp artakalan 405 B'yi *"AdamW/üstveri"* diye **uydurdum**.
> Ölçüm onu çürüttü: gerçek artık **83 B** ve nedeni üstveri. Sayı doğruyken
> **mekanizma** yanlıştı — [[denetim-kapsami-iddiadan-dar]] ailesinin bir örneği
> daha; kural: **ölçmediğin mekanizmayı yazma.**

---

## 5. Ölçüm aracının KENDİ kusurları (kapanıştan ÖNCE bulundu, kayıtlı)

`scratch/anka_r5_kos.py --kapanis`'in önceki sürümünde **üç** kusur vardı; üçü de
log metinleri `train.py` ile **karşılaştırılarak** bulundu (tahminle değil) ve
düzeltildi. Kayda geçer, çünkü üçü de **sahte hüküm üreticisiydi**:

| # | Kusur | Üreteceği yanlış hüküm |
|---|---|---|
| a | `son` tespiti `"Kaydedildi"` (büyük K) arıyordu; `train.py:329` küçük k ile basıyor ve `[KAPANIS]` diye bir metin **hiç basmıyor** | **başarılı koşum bile "BELIRSIZ"** yazılırdı (yanlış alarm) |
| b | `loss_report_ham` regexi `"Son 60 adim"` arıyordu; `train.py:318-320` `son60 … ± … (n=60)` basıyor | hiç eşleşmez ⇒ L2 kapısı **sahte yokluk** okurdu |
| c | `bitis_kaybi_satiri` regexi `"Başlangıç Kaybı"`nı yakalayıp ona **"bitiş"** adı veriyordu | **başlangıç kaybı bitiş sanılırdı** (T-0078 ailesi) |

Ek olarak `scratch/anka_r5_durum.py`'nin ilk deseninde `Adım (\d+)` tek boşluk
şartı vardı; `train.py` adım numarasını **sağa yaslı** bastığı için 1–990 arası
satırlar **atlanıyordu** (ölçüldü: 684 satırın 584'ü) ve "ilk kayıp" 3,5814
görünüyordu. `\s+` ile düzeltildi; gerçek ilk kayıp **4,1704** (L1 kaydıyla birebir).
⇒ **Ortak ders:** aynı sınıf — *aracın kapsamı iddiadan dar*.

### 5.1 Bu kapanışın KENDİ kusuru (kayda geçer)

**Raporun digest'i, rapor yazılırken ÜÇ KEZ değişti** ve üçüncüsüne kadar bir
kısmı başka belgelere **referans olarak** yazılmıştı. Sıra:

1. Rapor yazıldı → `ddf9f2ea…`; bu digest **hafızaya ve plana** yazıldı.
2. §4.1'deki **tahminim ölçümle çürütüldü** (§4.1 notu) ⇒ metin düzeltildi → digest değişti.
3. §1'e başlangıç/bitiş damgası eklendi → digest yine değişti.

⇒ `ddf9f2ea…` **hiçbir anda nihai değildi**; ona atıf yapan iki belge **bayat**
kaldı ve düzeltildi. Bu tam olarak *"kabul koşusu ölçtüğü artefaktı değiştirir"*
sınıfıdır: **bir artefaktın digest'ini, o artefakt kapanmadan başka yere yazma.**
Doğru sıra: içerik **donduktan sonra** digest al, **sonra** atıf yaz.

Ayrıca: ilk `bus_send` mesajında atıflar **yalnız yol** ile verildi; SPEC'in
`bus_send` atıf kuralı **yol + sha256** ister ⇒ eksiklik ayrı bir mesajla
**düzeltildi**, sessizce geçiştirilmedi.

---

## 6. Kabul kriterleri (T-0080) — madde madde

| # | Kriter | Durum | Kanıt |
|---|---|---|---|
| 1 | Faz 0 artefaktı + eşikler ölçümden **önce** ilan edildi | **✓** | `anka_r0_kapsam_2026-09-19.{json,md}`; sürüm notları |
| 2 | Ölçüm kapları lexicon'u yükler + fail-closed pozitif kontrol | **✓** | `anka_r4_g_kapilari…json` → `G6_pozitif_kontrol` 3/3 TAM; `K0.TEMIZ` |
| 3 | `roots.tsv` yerinde değişmedi; yeni dosya digest'li | **✓** | yeni: `data/lexicon/roots_anka_r1.tsv` `ea874a73…` |
| 4 | `vocab_base_32852.json` digest'i **değişmedi** | **✓** | `7611b6a5…` korundu; önek değişmezliği assertion'ı geçti |
| 5 | Yeni `.bin` gerçek dosyadan ölçülür; roots+vocab digest'leri meta'da | **✓** | `bin_sha256_meta` `9f987576…` ≡ `anka_a1r_pretrain.bin` |
| 6 | A1 artefaktlarının sha256'ları değişmedi | **✓** | §4 — 4/4 **BİREBİR** |
| 7 | Canlılık bandı koşum moduna göre ilan edilir | **✓** | §2 — devam tavanı 8,0; soğuk imzası 9,4077 **ayrı** yazıldı |
| 8 | `pytest` geçer + gerçek assertion eklenir | **✓** | `tests/test_morphology_regression.py` `65a6cd38…` (commit `6bca90b`) |
| 9 | Kapanış: mesaj + kiralar boş + `changed_files` **iki yönlü**, `len==len(set)` | **✓** | §7 + `state/results/T-0080.json` |

---

## 7. Değişen dosyalar — iki yönlü

Ölçüm: `var` kümesi = T-0080 yüzeylerinde **var olan** dosyalar; `len(var) ==
len(set(var))` ⇒ **tekillik GEÇTİ**; eksik dosya **0**.

| Kategori | Adet |
|---|---:|
| Kod (`src/compiler/*.py` ×4, `train.py`, `tests/test_morphology_regression.py`) | 6 |
| Veri (`roots_anka_r1.tsv`, `vocab_anka_r1_33036.json`, `vocab_anka_r1_33114.json`) | 3 |
| Çıktı (`anka_a1r.pt`, `anka_a1r_pretrain{,_val}.bin`) | 3 |
| `data/eval/` raporları (Faz 0–5 serisi) | 35 |
| `scratch/anka_*.py` üreticileri | 30 |
| **Toplam** | **77** |

**Bilinçli olarak DIŞARIDA bırakılanlar** (sessizce eklenmez, beyan edilir):
`data/eval/anka_a0*` ve `data/eval/anka_a1_*` (dosya adı tarihi **2026-09-18**;
kendilerini **T-0071 / T-0072** olarak ilan ediyorlar ⇒ **önceki görevin**
artefaktları, bu görev onlara **dokunmadı**, yalnız **okudu**).

> **`git status` bu listenin kanıtı DEĞİLDİR.** `*.pt` ve `*.bin` gitignore'da,
> `scratch/` gitignore'da ⇒ kapanış kanıtı **üreticinin listesine + tam-digest
> tablosuna** bağlanır. `git add -A` **kullanılmadı ve kullanılmayacak.**

---

## 8. Açık kalan işler (kapatılmadı — beyan edilir)

1. **AdamW momentleri kaydedilmiyor.** `train.py` optimizer durumunu
   checkpoint'e yazmıyor ⇒ `data/anka_a1r.pt`'den **devam** eden bir koşum
   optimizer durumunu **kaybeder** (momentler sıfırdan birikir). Kayıp eğrisi
   bu yüzden devamda geçici olarak bozulabilir. **Bu koşumun kaydı sağlamdır**;
   sınır **bir sonraki** koşum içindir.
2. **D4 (kesme/rakam) hâlâ açık.** G6'nın HATA'sının **%86,26'sı** kesme
   sözleşmesidir; `decompile` `Ankara'da` yazıyor, `compile` okuyamıyor (**asimetrik**).
   Planın D4 hedefi `src/llm/tokenizer.py` idi: **ölçüldü — bu dosya T-0080'de
   DEĞİŞMEDİ** (`git log` son commit `4251c53`; çalışma ağacı temiz). Külliyat
   `literal=False` ile derlendi. ⇒ D4 **kapanmadı**, sözleşme asimetrisi sürüyor.
3. **G3'ün KALDI'sı önceden ilan edilmişti** — D2 yer tutucu kütlesinin yalnız
   **%11,1'ini** kapatabilir; gözlenen ~%10 göreli düşüş o tavana uyuyor. Kusur
   değişikliğin değil, **G3'ü D2'ye bağlayan ilanın**.
4. **G5'in ilan edilen YÖNÜ kusurlu** — kapı yeniden ilan edilmeden anlamlı değil.
5. **En büyük kaldıraç D1 (en-uzun-kök kuralı) kapsam dışı bırakıldı**
   (ölçülmüş: TAM %81,24→%84,90). Bir sonraki turun **ilk** maddesi budur.
6. Bu raporla Faz 5 **kapandı**; `data/anka_a1r.pt` **tek geri dönüşsüz noktaydı**
   ve yazımı **beyanlı + kiralı** yapıldı (`--allow-frozen-write`, üst dizin kiralandı).

---

*Raporun bütün sayıları `data/eval/anka_r5_kosum_2026-09-19.json` (üretici çıktısı)
ve `scratch/anka_r5_kos.log` (ham log) dosyalarından birebir alınmıştır.*
