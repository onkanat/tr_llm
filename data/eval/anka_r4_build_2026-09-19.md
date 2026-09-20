# Anka A1-r · Faz 4 (külliyat yeniden derleme) — KAPI RAPORU

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Durum:** kapılar ölçüldü, hüküm verildi, Faz 5 başlatıldı

## 0. Özet — önce hüküm, sonra kanıt

| Ne | Sonuç |
|---|---|
| **İlan edilen ölçüm kapıları (G1·G2·G3·G4·G6)** | **5/5 KALDI** — ikisi tabanda da |
| G5 sözlük kapsamı | **KALDI** (ve ilan edilen yönü **kusurlu**, §8) |
| **Operatörün durdurma koşulu: K1 9-etiket D1 kapısı** | **GEÇTİ** (iki tabanda da parmak izi birebir) |
| Faz 4 üretici kapıları K1…K8 | **9/9 PASS** |
| Ölçüm kabı kimlik kontrolleri | **TEMİZ** (kaynak 5/5 birebir, örneklem 4/4 birebir) |
| Yön | **istenen yönde, iki bağımsız tabanda işaret uyumlu** |
| Büyüklük | **eşiklerin ALTINDA kaldı** |

**Dürüst cümle:** değişiklik **her** niceliği istenen yöne hareket ettirdi, ama
**hiçbiri ilan edilen eşiği geçmedi**. Yer tutucu kütlesi 9,5919 → 8,1760 düştü
(düşüşün **%89,89'u** yeni kimlikli konumlarla kütle dengesinde açıklanıyor) —
yani "kazanç var" doğru, "kabul edilebilir zenginlik" **henüz yok**. Eşikler
ölçümden sonra **değiştirilmedi** (plan kuralı); bu rapor onları değiştirmez,
yalnız **olduğu gibi** yazar.

**Ve şu da dürüstlüğün parçası:** G3'ün KALDI'sı **sürpriz değildi, önceden
ilan edilmişti.** Faz 0'da ölçülmüştü ki D2 yer tutucu kütlesinin yalnız **%11,1'ini**
kapatabilir (`data/eval/anka_r3_on_olcum_2026-09-19.json`; %56,7'si açık sınıf kişi/yer
adı, %15,2'si D4 kesme). Gözlenen düşüş (~**%10 göreli**) o tavana **uyuyor**.
⇒ Kusur değişikliğin **yetersizliği değil**, **G3'ü D2'ye bağlayan ilanın**
kusurudur (§8'deki G5 ile aynı aile). Bu, "eşiği düşürelim" demek **değildir** —
ilan edilen sayı **olduğu gibi** kalır; kayda geçen, ilanın hangi mekanizmaya
bağlandığıdır.

Operatörün kararı yalnız **bir** koşula bağlıydı (*"9-etiket kapısı koşulur;
düşerse rc≠0 ile durur"*) ve o koşul **sağlandı** (§3, ve gücünün sınırı §3.1'de)
⇒ Faz 5 operatör kararı gereği yürür.

---

## 1. Ölçüm tabanları — neden İKİ taban var (ilanın kendi iç tutarsızlığı)

İlan §4 önsözü "G2…G5 **yazılan artefakttan** ölçülür" diyor; **ama** §4'ün kendi
"ÖNCE (ölçüldü)" kolonu `r1_on`'un **örnek** değerlerini yazıyor. A1'in *yazılan*
`.bin`'i PN **%7,9365** iken örnek tabanı PN **%8,0186**'dır. Yani ilanın önsözü
ile kendi kolonu **farklı tabanlardır**. Bu bir **ilan kusurudur** (§9'da kayıtlı);
sessizce düzeltilmedi — **iki taban da raporlanır** ve hükmün işaretinin iki
tabanda da aynı olduğu gösterilir.

- **Taban A** — yazılan 100 M `.bin` (`data/anka_a1_pretrain.bin` → `anka_a1r_pretrain.bin`).
  **Nüfus kusuru var** (§7): belge kümeleri %99,04 örtüşür.
- **Taban B** — **aynı örnekleyici**, aynı 3.212 makale, aynı 236.573 yüzey.
  **Nüfus kusuru yok** ⇒ hüküm bu tabana dayanır, Taban A **destekleyici**dir.

### 1.1 Kaynak kontrolü — bağımsız iki ölçüm birebir mi?

```
## -1. Kaynak kontrolü (bağımsız iki ölçüm birebir mi?)

| Nicelik | kiyas_taban | üretici JSON | Birebir? |
|---|---|---|---|
| A1 PN | 7936521 | 7936521 | **EVET** |
| A1 UNK | 2629227 | 2629227 | **EVET** |
| A1r PN | 7136316 | 7136316 | **EVET** |
| A1r UNK | 2376511 | 2376511 | **EVET** |
| A1r LIT | 3088160 | 3088160 | **EVET** |
```

Beş nicelik **iki bağımsız ölçüm yolundan** (`anka_r4_kiyas_taban.py` vs
üreticinin kendi JSON'u) birebir çıktı ⇒ tabloların kaynağı güvenilir.

### 1.2 Örneklem kimliği — kıyas geçerli mi?

İki sınıf **ayrı** raporlanır. Bunları aynı kefeye koymak "örneklem bozuk" diye
**yanlış** hüküm verirdi (bu benim ilk taslak kusurumdu, §9 #2).

```
**Sözlükten bağımsız — BİREBİR olmak zorunda:**

| Alan | ÖNCE (r1_on) | SONRA (r4) | Birebir? |
|---|---|---|---|
| `benzersiz_yuzey` | `236573` | `236573` | **EVET** |
| `dosya_basi` | `{'tr-00000.parquet': 1688, 'tr-00001.parquet': 1524}` | `{'tr-00000.parquet': 1688, 'tr-00001.parquet': 1524}` | **EVET** |
| `kabul_makale` | `3212` | `3212` | **EVET** |
| `kelime_jetonu` | `2449120` | `2449120` | **EVET** |

**Sözlükten türetilen — farklı çıkması BEKLENİR (ihlal değil, ölçülen etki):**

| Alan | ÖNCE (r1_on) | SONRA (r4) | Değişim |
|---|---|---|---|
| `akis_sure_sn` | `236.4` | `237.4` | `+0.4230%` |
| `bin_jeton` | `4738026` | `4762703` | `+0.5208%` |

→ Belge seçimi BİREBİR aynı ⇒ aynı-örnekleyici kıyası GEÇERLİ.
```

`bin_jeton`'un **farklı çıkması doğrudur**: aynı belgeler yeni sözlükle
kodlandığında jeton sayısı **tanım gereği** değişir. Bunu "kimlik ihlali"
saymak yanlış olurdu.

---

## 2. İlan edilen ölçüm kapıları — **5/5 KALDI**

```
## 1. G1 / G6 — aynı-örnekleyici tabanı (yüzey gerekir, tek taban)

- **G1 en-uzun-eşleşme ihlali (%)**: 2.3789 → **2.3666** · kapı `<= 1.0` · **KALDI**
- **G6 tur özdeşliği (%)**: 96.9232 → **96.5771** · kapı `>= 99.0` · **KALDI**

## 2. G2 / G3 / G4 — İKİ taban birlikte (ilan §4 iç tutarsızlığı)

**Taban A — YAZILAN 100M `.bin`** (ilan §4 önsözü; nüfus kusuru: belge kümeleri %99,04 örtüşür, A1'in %0,44–1,11'i A1-r'de yok):

| Kapı | ÖNCE (A1 100M) | SONRA (A1-r 100M) | Eşik | Sonuç |
|---|---|---|---|---|
| G2 `<UNK>` (%) | 2.6292 | **2.3765** | `<= 1.5` | **KALDI** |
| G3 `<PROPER_NOUN>` (%) | 7.9365 | **7.1363** | `<= 3.0` | **KALDI** |
| G4 literal ad / yer tutucu | 0.2902 | **0.4327** | `>= 1.0` | **KALDI** |

  Pay/payda ayrı (ilan §4.2 G4 çürütmesi): LIT 2,302,966 → 3,088,160 · PN 7,936,521 → 7,136,316

**Taban B — AYNI ÖRNEKLEYİCİ** (ilan §4'ün 'ÖNCE (ölçüldü)' kolonunun tabanı; popülasyon-kontrollü, nüfus kusuru YOK):

| Kapı | ÖNCE (r1_on) | SONRA (r4) | Eşik | Sonuç |
|---|---|---|---|---|
| G2 `<UNK>` (%) | 1.5733 | **1.523** | `<= 1.5` | **KALDI** |
| G3 `<PROPER_NOUN>` (%) | 8.0186 | **6.653** | `<= 3.0` | **KALDI** |
| G4 literal ad / yer tutucu | 0.3753 | **0.6617** | `>= 1.0` | **KALDI** |

**İşaret uyumu (iki taban) — hükmün tabandan bağımsızlığı:**

| Nicelik | Taban A farkı | Taban B farkı | İşaret aynı mı? |
|---|---|---|---|
| PN (%) | -0.8002 | -1.3656 | **EVET** |
| UNK (%) | -0.2527 | -0.0503 | **EVET** |
| LIT (%) | +0.7852 | +1.3929 | **EVET** |
```

**İşaret uyumu tablonun en önemli satırıdır:** altı farkın **altısı** iki bağımsız
tabanda aynı yönde ⇒ nüfus kusuru hükmü **çeviremez**.

**Not (küçük ama gerçek):** G2 Taban B'de **1,5733 → 1,5230**, eşik 1,5 — yani
**0,023 puan** ile kaçtı. "Neredeyse geçti" demek **spindir**: kapı KALDI'dır.
Ama bu, G2'nin D2 ile *zaten* hedefe çok yakın olduğunu gösterir.

---

## 3. Operatör kapısı — K1 9-etiket D1 dokunulmadı

Operatör kararı (aynen): *"D1'e ve tie-break'e DOKUNULMAZ. Her partiden sonra
9-etiket kapısı koşulur; düşerse rc≠0 ile durur."*

Kapı tanımı `scratch/anka_r2_d3_mekanizma.py:131-155`'ten **birebir** alındı
(yeniden yazılmadı): 5 DOĞRU okuma + 4 YANLIŞ (D1 kusuru) = 9 etiket; parmak izi
sabitlenmiş `beklenen` ile birebir olmak zorunda. **YANLIŞ'ların YANLIŞ kalması da
kapının parçasıdır** — `ağacı→ağaç` düzelseydi bu da "D1'e dokunuldu" demekti.

```
[ONCE]  sozluk=32,852 · lexicon_cocuk=62 · K0 TEMIZ
[K1-ONCE]  DEGISMEDI · DOGRU_hepsi_ok=True
[SONRA] sozluk=33,114 · lexicon_cocuk=63 · K0 TEMIZ
[K1-SONRA] DEGISMEDI · DOGRU_hepsi_ok=True
        iz: bunun:bu:2|onun:o:2|ölmeden:öl:2|topraklar:toprak:2|günümüzde:gün:3|
            ağacı:ağa:2|aklı:ak:2|kanadı:kana:2|gönlü:gön:2

[K1] K1 GECTI — D1'e dokunulmadi (iki tabanda da parmak izi BIREBIR)
```

Her iki tabanda **K0 pozitif/negatif kontrolü önce** koştu (T-0078'in ölüm sebebi:
`load_from_tsv` atlanırsa derleyici **boş trie** ile çalışır ve bu **sessizdir**);
`lexicon_kok_cocuk` sayısı da ayrıca rapora yazıldı (62 / 63) — boş trie 0 verirdi.

**Kanıt:** `data/eval/anka_r6_k1_kapisi_2026-09-19.json` · `rc=0`

### 3.1 ⚠ K1'in GEÇTİ'sinin ayırt edici gücü KÜÇÜKTÜR (kendi uyarım)

Bu kapı **9 etiket** ile konuşur; aynı ölçüm hattında **34.832 pencere** var ve tam
karar haritası çıkarıldığında `seçimi değişen = 16` bulunmuştu. Yani K1'in GEÇTİ'si
**"D1 bu 9 okumada kaymadı"** demektir; **"D1 hiç kaymadı"** demek **değildir**.
Oracle olmadan `değişti ≠ iyileşti` (kayıtlı: `kapi-orneklem-zayifsa-gecmek-kanit-degil`).
Bu yüzden K1'i **operatörün durdurma koşulu** olarak raporluyorum, **kalite kanıtı**
olarak değil.

---

## 4. G3 çürütme maddesi + ANALIZSIZ bacağı

İlan §4.2 (ölçümden **önce** yazıldı): *"düşüş yeni kimlikli yüzeylerden
gelmiyorsa **VEYA ANALIZSIZ artıyorsa** hüküm `AYIRT EDEMEDİ`."* İkinci bacak
`ONCE` tarafında **ölçülmemişti** ⇒ kapanması için taban koşumu yapıldı
(`scratch/anka_r4_taban_ornek.py`, aynı örnekleyici, aynı birim).

```
**ONCE tabanı kontrolleri** (fail-closed):

- pozitif(Ankara): `True` · negatif(zzqqxx): `True` · yeni-blok-boş: `True`
- `İstanbul` bu tabanda: `['İstanbul']` ← **elle seçtiğim negatif kontrol YANLIŞTI** (kendi kusurum #4); kontrol dosya+trie çiftine bağlandı

- ONCE liveness (ANALIZSIZ bacağı canlı mı): `True` (745,142)

| Morfem pozisyonu (yüzey-sıklığı ağırlıklı) | ÖNCE (taban) | SONRA (A1-r) | Değişim |
|---|---|---|---|
| yeni kimlikli konum | 0 | 53,576 | +53,576 |
| kökü sözlükte eksik konum | 24,841 | 4,986 | -19,855 |
| sözlükte VAR olan konum | 2,774,113 | 2,795,475 | +21,362 |
| **ANALIZSIZ (büyük+küçük)** | **745,142** | **703,524** | **-41,618** (-5.59%) |

→ **ANALIZSIZ DÜŞTÜ** ⇒ ilan edilen çürütme maddesinin ANALIZSIZ bacağı **ATEŞLEMEDİ**
```

**Çürütme maddesi ATEŞLEMEDİ** ⇒ hüküm `AYIRT EDEMEDİ` değil.

Taban koşumu ayrıca **ilanın kendi "ÖNCE" kolonunu bağımsız olarak yeniden üretti**:
PN %8,0186 · UNK %1,5733 · G1 %2,3789 · G6 %96,9232 — ilan §4'te yazılan
değerlerin **birebir aynısı**. Bu, ilanın sayılarının kaynaksız olmadığının kanıtıdır.

---

## 5. Atıf — yer tutucu düşüşünün kaynağı (jeton kütlesi dengesi)

```
## 4. Atıf (jeton kütlesi) ve nüfus kusuru

- Yeni blok: **262** id (254 büyük + 8 küçük) · kütle **756,058** jeton (%0.7561)
- Büyük-harf yeni kimlik kütlesi: **719,265** (%0.7193)
- PN düşüşü **800,205** jeton; bunun **%89.89'i** yeni kimlikli konumlarla açıklanır (açıklanmayan 80,940)
- LIT artışı **785,194**; bunun **%91.60'i** yeni kimlikli
- Kütle dengesi kontrolü: `{"kiyas_sozluk_giris_A1": 32852, "kiyas_sozluk_giris_A1r": 33114, "build_benzersiz_id_A1": 28352, "build_benzersiz_id_A1r": 28599, "PN_eslesme": true, "UNK_eslesme": true, "yeni_blok_eslesme": true}`
```

**Kütle dengesi kapanıyor:** PN'nin kaybettiği 800.205 jetonun **%89,89'u** tam
olarak artık yeni büyük-harf kimlik taşıyan jetonlardır (719.265). LIT artışının
**%91,60'ı** aynı kaynaktan. Üç bağımsız oran (PN, UNK, yeni-blok) üreticinin
JSON'larıyla **bit-özdeş**.

Birim uyarısı: bu **jeton kütlesi** atfıdır, **konum-konum eşleşmesi değil** —
iki `.bin` farklı morfem sınırlarına sahip olduğu için hizalı değildir.

### 5.1 Yeni 262 id'nin kökeni (D2 mi, id onarımı mı?)

`scratch/anka_r4_koken_dagilim.py` → `data/eval/anka_r4_koken_dagilim_2026-09-19.json`:

| Köken | Adet | Örnek |
|---|---|---|
| **YALNIZ YENİ ROOTS** (D2/D3 eklemesi) | **173** | Abu Dabi, Abuja, Addis Ababa, Afganistan, Akra, Almanya |
| **TABANDA BİREBİR** (id onarımı: Faz 3) | **78** | Adıyaman, Aliağa, Ardahan, Bartın, Bayburt, Ermenice, Gaziantep |
| **TABANDA normalize ile** (İ/I onarımı) | **11** | Arjantin, Germen, Irak, Kabil, Katar, Kore, Panama, Sudan, Tiran, Tokyo, Umman |
| Hiçbirinde yok | **0** | — |

Kontroller: `İstanbul_blokta=True` · `Ankara_blokta=False` · `id_siz_yeni_lemma=0`.
Bağımsız ölçüm: `roots_anka_r1.tsv` = `roots.tsv` + **207 lemma**, **0 silme**.

**Bu ayrım rapor için zorunludur:** 262 kimliğin **89'u** yeni veri *değil*,
zaten var olan donmuş lemmaların **erişim/id onarımıdır**. İkisini birleştirip
"D2 kazancı = 262" yazmak şişirme olurdu.

---

## 6. Üretici kapıları K1…K8 ve korunan artefaktlar

```
| Kapı | PASS |
|---|---|
| `K1_noktalama` | **PASS** |
| `K2_unk` | **PASS** |
| `K3_pad` | **PASS** |
| `K4_olcek` | **PASS** |
| `K5_tekrarsizlik` | **PASS** |
| `K6_izlenebilirlik` | **PASS** |
| `K6b_koken` | **PASS** |
| `K7_pozitif_kontrol` | **PASS** |
| `K8_temsil` | **PASS** |

- `koken_degismedi`: `{"lexicon": true, "lexicon_taban": true, "sozluk": true}`
- `korunanlar dokunulmadi`: `{"data/anka_a1.pt": true, "data/anka_a1_pretrain.bin": true, "data/anka_a1_pretrain_val.bin": true, "data/anka_pretrain.bin": true}`
```

**K1…K8'in hepsi PASS olması "külliyat iyi" demek DEĞİLDİR.** Bunlar
*taban* kapılarıdır (hattın canlı olduğunun kanıtı), kalite tavanı değil — ilan
§Faz 0 bunu zaten yazmıştı: *"K8 bir taban, kalite tavanı değil."* Kalite
sorusunu G1…G6 cevaplar ve onlar **KALDI**.

---

## 7. Nüfus kusuru (kendi ölçüm tasarım kusurum — beyan edilir)

```
- Nüfus kusuru: belge örtüşmesi %99.0386 · A1'in **%0.4408–%1.105** dilimi A1-r'de YOK
```

İki 100 M külliyat aynı sıralı akıştan, aynı 14 kuralla, aynı `MIN_BELGE_KAR=300`
ile derlendi; ama jeton sayıları değiştiği için **durma noktası belge sınırında
farklı**: A1 146.560 kabul edilen belgeye, A1-r 145.151'e kadar gitti.
A1'in **%0,44–1,11** dilimi A1-r'de **yoktur**.

Bu dilim, atıf kütlesinden **küçük değildir**. Bu yüzden **100 M ↔ 100 M kıyası tek
başına hüküm vermez**; hüküm aynı belgeleri okuyan örnekleyici kıyasına dayanır
ve Taban A **destekleyici** olarak raporlanır (§2'de işaret uyumu bunu doğrular).

---

## 8. G5 — ilan edilen YÖNÜN kendisi kusurlu

```
- ÖNCE train: `%86.3` (gözlenen 28,352 / giriş 32,852)
- SONRA train: **%86.3653** (28,599/33,114)
- SONRA val: **%47.572** (15,753/33,114) · ÖNCE val %47.73
- İlan eşik `>= 95.0` ⇒ **KALDI**
- İlan edilen kusur: G5'in ilan edilen YONU D2 basarisiyla TERS: payda buyudukce G5 mekanik olarak duser (ilan §4.1). Duserse bu ILANIN kusurudur, degisikligin degil.
```

G5 **yapısal olarak D2 ile ters**tir: D2 sözlüğe lemma **ekler** ⇒ payda büyür ⇒
gözlenen/giriş oranı **mekanik olarak düşer**. D2'nin başarısı G5'i düşürür.
Ölçülen: gözlenen id 28.352 → 28.599 (**+247 artmış**), oran ise %86,3 → %86,37'de
**neredeyse sabit** — çünkü pay ile payda birlikte büyüdü. G5'in KALDI'sı
değişikliğin değil, **ilanın** kusurudur.

---

## 9. Bu pencerede ölçülen KENDİ KUSURLARIM (yedi madde, hepsi kayıtlı)

| # | Kusur | Nasıl yakalandı / etki |
|---|---|---|
| **1** | Tablo üreticim örneklem kimliğini **tek sınıf** sayıyordu ⇒ `bin_jeton`/`akis_sure_sn`'i "ihlal" sayacaktı | YANLIŞ "kıyas geçersiz" hükmü üretirdi. İki sınıfa ayrıldı (KİMLİK / ETKİ) |
| **2** | Tablo üreticim G2/G3/G4 "SONRA"sını örnekten alırken ilan §4 önsözü yazılan artefaktı istiyordu | İki taban + işaret uyumu tablosu eklendi |
| **3** | Refütasyon maddesinin ilk yeniden yazımında **birim karıştırdım** (jeton sınıfı vs morfem pozisyonu) | Birim alanı ayrı yazıldı; "iki birim karıştırılmaz" notu koda gömüldü |
| **4** | **`yeni_kimlik_pozisyon` SAHTE SIFIR:** `c.compile()` çıktısındaki `token_vector` **karakter dizisi** taşır (`['İstanbul']`), id değil; kümeyi **id kumesi** kurmuştum ⇒ `m in yeni_kimlik` **her zaman False** | Declared çürütme maddesini **yanlış yöne** ateşlerdi (`AYIRT EDEMEDİ`). Düzeltildi; kusurlu artefakt **silinmedi, kanıt olarak saklandı**: `data/eval/anka_r4_g_kapilari_KUSURLU_atif_2026-09-19.json`. Düzeltilmiş ölçüm: **53.576** (0 değil) |
| **5** | Taban koşumunda NEGATİF kontrolü **elle** `İstanbul` seçtim ("tabanda çözülemez") — **İ/I onarımından sonra tabanda ÇÖZÜLÜYOR** | Fail-closed **kontrol düştü ve koşumu durdurdu** (doğru davranış). Kontrol dosya+trie çiftine bağlandı. Aynı sınıf hatanın **dördüncüsü** (r0 raporunda "kontrol etiketi ≠ beklenen değer (3. kez)") |
| **6** | İlan §1 girdi tablosu **`src/compiler/**`'i listelemiyor**; oysa derleyici mtime'ları (09-19 12:05–12:25) A1 derlemesi (09-18 16:55) ile A1-r derlemesi (09-19 13:49) **arasında** | A1↔A1-r farkı lexicon∪sözlük **∪ derleyici**'dir. İlan eksik beyan etti; **sessizce düzeltilmedi**, buraya yazıldı |
| **7** | Nüfus kusuru (farklı durma noktası) **geç fark edildi** | §7'de beyan edildi; hüküm Taban B'ye bağlandı |

**Ek (sandbox, kod kusuru değil):** `pytest` sandbox içinde
`test_agent_gateway.py::test_gateway_http_server_endpoints` `socket.bind` →
`PermissionError [Errno 1]` ile düştü; **sandbox dışında 7/7 passed**. Paket
işlevsel olarak **214/214**. `/sandbox` ile yönetilir.

---

## 10. Tam digest tablosu (önek değil, TAM)

| Yol | Bayt | sha256 (TAM) |
|---|---|---|
| `data/lexicon/roots.tsv` | 870,426 | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/lexicon/roots_anka_r1.tsv` | 873,755 | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| `data/rebuild/vocab_base_32852.json` | 741,650 | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` |
| `data/rebuild/vocab_anka_r1_33114.json` | 747,458 | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| `data/anka_a1_pretrain.bin` | 200,000,000 | `383a9c890b75ec96195122b9981d7a73371d3bcaa3b9f6e7c2a188510d917cb8` |
| `data/anka_a1_pretrain_val.bin` | 4,000,000 | `02f52063e73e4ac362297c9ecad41247aa621e168572b9cb3f6d302f734da801` |
| `data/anka_a1r_pretrain.bin` | 200,000,000 | `9f9875762518829dec8f5baa41bd012ee1abf72fb630e24840bafedf691d4f22` |
| `data/anka_a1r_pretrain_val.bin` | 4,000,000 | `b3fad8da2d6b3c197aafda52bf7d69ac4cfc17b42d4332a60936633eb21fc5d6` |
| `data/anka_a1.pt` (devralınan, DONMUŞ) | 372,124,278 | `f32d492d9361c668bc86e4e4b0f99f5c473af552486837452475f8b0aa3eb5b4` |
| `src/compiler/core.py` | 7,502 | `abbc78433c1b5fc35498c05d727bd461cf3936b06ad777c8d84338a81f3acfa4` |
| `src/compiler/lexicon.py` | 5,193 | `9e2986f870204634b4c8fd17dac1c6bb1020a9aed5631b4cad6bf6258e1f51e7` |
| `src/compiler/morphotactics.py` | 10,905 | `1b03902db77533b627558c218d62ee3553582af4b6a4c5d6c271d06c0494d4aa` |
| `src/compiler/phonology.py` | 8,128 | `0ced6097adc47c3581f6ae19756d40102850ddb6e2aca53e105fd29421e85f95` |
| `src/llm/tokenizer.py` (DONMUŞ) | 15,178 | `560a3bfbf32c394a23559c9d3645cfbc3b554223a032156924749dcfa896a7af` |
| `train.py` | 17,447 | `940b22b929eb576d68719fa1ce5cae1a72e043ee87ed5ec02656c485fd042c48` |
| `scratch/anka_r5_kos.py` | 10,155 | `25ddc6d1e9bd26abd00e63e2f5cb9dee4ed6357fe227ca6d9b6f0b7cc153ac60` |
| `scratch/anka_r4_g_kapilari.py` | 29,517 | `0961c583f6396b7fbe0bbf263609cb31d11699c059a244751432d911193ef0a8` |
| `scratch/anka_r4_taban_ornek.py` | 32,061 | `0fc379b36f0eea317b8e6c2f661f8549c2174b2d73bff8d47184a1fbb0f44bd5` |
| `scratch/anka_r6_k1_kapisi.py` | 7,218 | `ac15ef2ca3a89a35f15468192ff4079a63511aae8d0a0cb8855eb83233265236` |
| `scratch/anka_r4_tablo.py` | 14,785 | `9ae2edb5ea5796f970cdc640df58af3d2a00cd9cc64264597ccbfcb935c01790` |

**Ölçüm artefaktları** (`data/eval/`, bu raporla birlikte):

| Artefakt | Ne |
|---|---|
| `anka_r4_g_kapilari_2026-09-19.json` | SONRA, aynı-örnekleyici (düzeltilmiş atıf) |
| `anka_r4_g_kapilari_KUSURLU_atif_2026-09-19.json` | **kusurlu koşum, kanıt olarak saklandı** (sahte sıfır) |
| `anka_r4_taban_ornek_2026-09-19.json` | ÖNCE tabanı + ANALIZSIZ bacağı |
| `anka_r1_on_kapsam_2026-09-19.json` | ÖNCE (ilan §4 kolonunun kaynağı) |
| `anka_r4_kiyas_taban_2026-09-19.json` | 100 M ↔ 100 M tam `.bin` sınıf kıyası |
| `anka_r4_atif_bin_2026-09-19.json` | atıf + nüfus kusuru |
| `anka_r4_koken_dagilim_2026-09-19.json` | 262 id'nin kökeni (173/78/11) |
| `anka_r6_k1_kapisi_2026-09-19.json` | operatör K1 9-etiket kapısı |
| `anka_r4_kapi_ilani_2026-09-19.md` | **ölçümden ÖNCE** yazılan kapı ilanı |

---

## 11. Faz 5 — başlatıldı, canlılık imzası GEÇTİ

Operatörün durdurma koşulu (K1) sağlandığı için Faz 5 operatör kararı gereği yürüdü.

```
[cmd] train.py --vocab data/rebuild/vocab_anka_r1_33114.json --data data/anka_a1r_pretrain.bin
      --device mps --pretrain --allow-frozen-write --loss-report
      --load-path data/anka_a1.pt --save-path data/anka_a1r.pt
      --steps 48828 --batch-size 8 --block-size 256 --save-every 500 --lr 0.001
[SOZLESME_UYARI] 'embedding.embedding.weight' (32852, 768) -> (33114, 768)  (262 satır dolduruldu)
[SOZLESME_UYARI] 'lm_head.weight'              (32852, 768) -> (33114, 768)  (262 satır dolduruldu)
[SOZLESME_UYARI] 'lm_head.bias'                (32852,)     -> (33114,)      (262 satır dolduruldu)
Model mimarisi kuruldu ve cihaza taşındı. (Öğrenme Oranı: 0.001)
Eğitim Başlatılıyor -> Adım Sayısı: 48828, Batch: 8, Block: 256, Hedef: ON-EGITIM
Adım    1/48828 | Kayıp (Loss): 4.1704 | Adım Süresi: 2.06s
Adım   10/48828 | Kayıp (Loss): 4.1002 | Adım Süresi: 0.71s

[L1] ilk kayip: adim 1 -> 4.1704 · tavan 8.0000 (mutlak taban 9.4077)
[L1] GECTI — sicak baslangic canli
```

Kanıtlanan dört şey, dördü de **ölçülmüş**:

1. **Sıcak başlangıç canlı.** İlk kayıp **4,1704** < 8,0 tavanı. Sessizce sıfırdan
   başlasaydı ilk kayıp ≈ ln(33114) = **10,4076** olurdu (T-0077 imzası).
2. **Hizalama SESSİZ DEĞİL.** `[SOZLESME_UYARI]` üç parametre için **262 satır
   dolduruldu** diye bastı — sessiz kırpma yok.
3. **LR sessizce düşmedi.** `Öğrenme Oranı: 0.001` — `train.py`'nin devam
   koşumunda lr'yi sessizce 2e-4'e indiren satırı `--lr` ile **açıkça** ezildi.
4. **Maske ölü değil.** Hedef `ON-EGITIM` (düz sonraki-jeton) ⇒ T-0073'ün
   "her pencereyi maskele ⇒ gradyan 0 ⇒ MPS 0,0000 basar" tuzağına düşülmedi.

**Yük sinyali:** adım süresi **0,71 sn** (referans 1,55). 2-3× sapma yük sinyali
olurdu; sapma yok ⇒ makinede GPU tüketen başka iş **yok** (T-0052).
**Süre:** 48.828 adım × ~0,71 sn ≈ **9,6 saat** (T-0072 dersi: `adım × sn/adım`,
bölme değil). Kira bitişi 2026-09-20T11:58Z ⇒ **marj var**.

---

## 12. Açık kalan işler (kapatılmadı, beyan edilir)

1. **G1…G6 KALDI** ⇒ "kabul edilebilir zenginlik" hedefi **kısmen** karşılandı.
   En büyük tek kaldıraç **D1**'dir (en-uzun-kök kuralı; ölçülmüş: TAM
   %81,24→%84,90) ve bu pencerede operatör kararıyla **kapsam dışı** bırakıldı.
2. **D4 (kesme/rakam)** hâlâ açık: G6'nın HATA'sının **3.371'i (%86,26)** kesme
   sözleşmesidir — decompiler `Ankara'da` yazıyor ama derleyici okuyamıyor
   (`compile → []`). Sözleşme **asimetrik**; ilanda bu **ayrıştırılmamıştı**, bu
   raporda ayrıştırıldı. (`HATA_ayristirma`: kesme 3.371 · geri_boş 0 · diğer 537)
3. **G5 ilan yönü kusurlu** — kapı yeniden ilan edilmeden anlamlı değil.
4. **İlan §1 girdi tablosu eksik** (`src/compiler/**` yok).
5. `data/eval/anka_r4_g_kapilari_2026-09-19.json` **düzeltilmiş** koşumun
   üzerine yazdı; kusurlu hali `..._KUSURLU_atif_...json`'da duruyor.
6. Faz 5 koşumu `data/anka_a1r.pt` yazacak (donmuş, `--allow-frozen-write` ile
   beyan edildi, üst dizin kiralandı). **Tek geri dönüşsüz nokta burasıdır.**

---

*Bu raporun bütün sayıları `scratch/anka_r4_tablo.py` çıktısından **birebir**
alınmıştır; hiçbir sayı elle yazılmamıştır. Tablolar `scratch/anka_r4_tablo.out`
dosyasında ham hâlde durur.*
