# T-0072 — Anka A1-a: külliyat **düz metin** temsiliyle YENİDEN derlendi

**Görev:** T-0072 · **ilan** 2026-09-18T13:16:56Z · **başlangıç** 2026-09-18T13:20:28Z ·
**bitiş** 2026-09-18T13:55:32Z
**Operatör kararı (Onkanat, 18 Eyl 2026):** *"literal=False ile yeniden derle."*

**Bu görevin yeri:** T-0071'in **tam külliyatında** ölçülen temsil kusurunun
(`data/anka_pretrain.bin`'in **%39.1**'i varlık işaretlemesi;
`<ENT><CAP>i n g i l t e r e</ENT>` = "İngiltere" 12 jeton) operatörce
kapatılması. Kapılar **ölçümden önce** ilan edildi (`data/eval/anka_a1_corpus_design_2026-09-18.md`); burada ölçüldü, **değiştirilmedi**.
*(Bu oran T-0071'in yazılı dosyasından yeniden ölçüldü. §1'deki %35.6
farklı bir kümedir — 400 makalelik örnek; ikisi karıştırılmamalı.)*

---

## 1. Neden: kararın ölçülmüş dayanağı

Aynı örnek (400 makale) iki modda **yeniden tokenize edildi** (tasarım betiği):

| | `literal=True` (T-0071) | `literal=False` (bu görev) |
|---|---|---|
| jeton/makale | 5,314.2 | **3,397.8** |
| **karakter/jeton** | 2.5951 | **4.0587** |
| UNK | %1.2056 | %1.9955 |
| `<ENT>` | 85,030 | **0** |
| `<PROPER_NOUN>` | 0 | **80,990 (%5.96)** |

⇒ **%36.06 daha az jeton**; aynı bütçeyle **1.56× daha fazla metin**.

⚠️ **Kümeyi ayırt et:** bu tablo **400 makalelik ÖRNEK**tir (parquet başı = uzun makaleler). Tam
külliyatta ölçülen `<PROPER_NOUN>` oranı **%7.94**'dir (§3.1); örnekteki
%5.96 **aynı sayı değildir** ve olması da beklenmez — iki farklı kümedir.

**✅ Kabul edilen bedel (kayda geçirilir):** OOV büyük-harf kelime tek `<PROPER_NOUN>`
yer tutucusuna çöker ⇒ **model özel ad ÜRETEMEZ.** Ölçülen oran — tam külliyat
**%7.94**, örnek %5.96.
Bu bir yetenek kaybıdır; bu rapor onu "sorun yok" saymaz.

---

## 2. Değişen tek şey ve değişmeyenler

| | T-0071 | T-0072 |
|---|---|---|
| tokenizer | `literal_entity_mode=True` | **`False`** |
| çıktı | `data/anka_pretrain.bin` | **`data/anka_a1_pretrain.bin`** |
| derleyici | `scratch/anka_a0b_build.py` | **`scratch/anka_a1_build.py`** (yeni) |

**Değişmedi:** kurtarılan külliyat (repo dışı HF önbelleği), arınma kuralları, kanonik sözlük
(`32852` giriş), belge düzeyi `<BOS>…<EOS>`, min 300
karakter eleme, sıra (glob sorted + satır sırası), hedef 100,000,000 +
2,000,000 jeton, seri referans yolu, artımlı yazma.

**Neden yeni betik/çıktı:** T-0071'in betiğinin **digest'i kapalı raporda kayıtlı**; düzenlemek o
raporu bayatlatırdı. Aynı şekilde `data/anka_pretrain.bin` **ölçülmüş bir artefakt** — üzerine
yazılmadı, **yan yana durur** (aşağıda kanıtlandı).

---

## 3. 🚩 ÖLÇÜLEN KAPILAR (ilan edildi — değiştirilmedi)

| kapı | hüküm |
|---|---|
| **K1_noktalama** | GEÇTİ |
| **K2_unk** | GEÇTİ |
| **K3_pad** | GEÇTİ |
| **K4_olcek** | GEÇTİ |
| **K5_tekrarsizlik** | GEÇTİ |
| **K6_izlenebilirlik** | GEÇTİ |
| **K7_pozitif_kontrol** | GEÇTİ |
| **K8_temsil** | GEÇTİ |

**K8 (YENİ) — bu görevin asıl kapısı.** T-0071'de kapıların hiçbiri **akışın neyden oluştuğunu**
sormuyordu; bu yüzden külliyatın %39.1'i varlık işaretlemesi olduğu hâlde **7/7 "GEÇTİ"** çıkmıştı.
K8 iki hattı **ayırt eder**: `literal=True` ile derlenseydi `ENT` milyonlarca olur ve K8 düşerdi.

Ölçülen: `ENT`=**0** · `CAP`=**0** ·
`<PROPER_NOUN>`=**%7.9365** (eşik ≥%1.0).

**K7 iki kontrollü:** K7a noktalama hattı canlı mı (`12`),
K7b **temsil gerçekten devrede mi** — OOV büyük-harf referans metni `<PROPER_NOUN>` üretmeli ve
`<ENT>` üretmemeli (ölçülen `PROPER_NOUN`=1,
`ENT`=0). **Hat ölseydi ya da mod yanlış olsaydı K1–K6
"KANITSIZ" ilan edilecekti.**

### 3.1 Derlenen artefakt (yazılan dosyadan yeniden ölçüldü)

| | jeton | sha256 | PAD | UNK | noktalama | benzersiz id | `<PROPER_NOUN>` | `<ENT>` |
|---|---|---|---|---|---|---|---|---|
| `data/anka_a1_pretrain.bin` | 100,000,000 | `383a9c890b75ec96195122b9981d7a73371d3bcaa3b9f6e7c2a188510d917cb8` | %0.0000 | %2.6346 | %11.0213 | 28,352 | %7.9365 | 0 |
| `data/anka_a1_pretrain_val.bin` | 2,000,000 | `02f52063e73e4ac362297c9ecad41247aa621e168572b9cb3f6d302f734da801` | %0.0000 | %2.9046 | %11.1694 | 15,679 | %8.8232 | 0 |

`max_id` = **32850** · `min_id` = 0 · `<EOS>` = 146,559

**Belge akışı:** 194,345 makale okundu · kullanılan **146,560** (train) +
**3,899** (val) · elenen (kısa) **43,884** ·
tekrar id **0** · tekrar metin **2**

**T-0071 ile verim kıyası (aynı kaynak, aynı arınma):**

| | T-0071 (`literal=True`) | T-0072 (`literal=False`) |
|---|---|---|
| belge — **train** | 73,767 | **146,560** |
| belge — val | 2,009 | 3,899 |
| jeton (train) | 100,000,000 | 100,000,000 |
| **jeton (train) ÷ belge (train)** | **1,355.6** | **682.3** |
| **aynı 100 M jetonla okunan belge** | 73,767 | **146,560 (×1.99)** |

**Ana bulgu (tam külliyat, örnek değil):** aynı 100 M jeton bütçesiyle T-0071 **73,767**,
T-0072 **146,560** belge okudu ⇒ **×1.99 daha fazla belge**. Örnek üzerinde
ölçülen `metin_carpani` (1.56×) bunun **örnek ölçeğindeki** karşılığıdır.

### 3.2 Kaynak izlenebilirliği

| `tr-00000.parquet` | 102,001,047 | okundu | `17df2980dd5a28ce8f449ab7b199e875165bb7ad81305c1c544896d3c3f2b732` |
| `tr-00001.parquet` | 0 | okunmadı | `a358fb4fd5179e8ac282ff266c7ad19d53eafe680911986a7b2c04d941127ca9` |

### 3.3 Sözlük kapsamı

| | train | val |
|---|---|---|
| benzersiz id | **28,352** | **15,679** |
| kapsam | **%86.3** | %47.73 |
| hiç görünmeyen | 4,500 | 17,173 |
| noktalama bloğu | 9/9 | 9/9 |

---

## 4. 🚩 ÇÜRÜTME ŞARTI — **ölçümden ÖNCE ilan edildi** (T-0071'in tasarım eksiği kapatıldı)

> **literal=False kulliyatinin karakter/jeton orani literal=True'ninkinden KUCUK cikarsa, 'ayni butcede daha fazla metin okur' gerekcesi CURUR ve karar yeniden acilir.**

| | değer |
|---|---|
| örnekte `karakter/jeton` true | 2.5951 |
| örnekte `karakter/jeton` false | 4.0587 |
| **çürüdü mü?** | **False** |

T-0071 kendi çürütme şartını **ölçümden sonra** yazmış ve bunu tasarım eksiği olarak kaydetmişti.
Bu görev o eksiği **tekrarlamadı**: şart tasarım belgesinde, derleme başlamadan önce kayda geçti.
Sıra kanıtı: `ilan_utc` = 2026-09-18T13:16:56Z ≤ `baslangic_utc` = 2026-09-18T13:20:28Z.

---

## 5. Kapanış kanıtları

- **T-0071 artefaktı dokunulmadı:** `data/anka_pretrain.bin` koşum öncesi/sonrası sha256 **aynı**
  (`dokunulmadi` = True); külliyat **yan yana** durur.
- **Kanonik sözlük değişmedi:** `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` → `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad`
  (`sozluk_degismedi` = True).
- **id uzayı genişlemedi:** inşa öncesi/sonrası 32852, `next_id`=32852
  (`SystemExit` bekçisi).
- **Hiçbir `.pt` yazılmadı**; GPU kullanılmadı (CPU derlemesi, 6 işçi).
- **Artımlı yazma:** 71 ara kontrol noktası; sentinel `rc=0`'a bağlı.
- **Paralelleştirme:** 6 işçi süreç; akış hızı 48,627 jeton/sn,
  toplam 35.0 dakika.
- **Seri ↔ paralel bit-özdeşliği (yeni betik için, duman ölçeğinde):** **DOĞRULANDI** — `anka_a1_smoke_train.bin` AYNI · `anka_a1_smoke_val.bin` AYNI (seri koşum logu mevcut, 2 çıktı)
  Özdeşlik hükmü **seri koşumun gerçekten yapıldığı** kanıtına bağlandı; yoksa "dosyalar
  zaten aynıydı" diye kendini doğrulayan bir assert olurdu.

---

## 6. BU GÖREVİN İDDİA ETMEDİĞİ ŞEYLER

1. **"Düz metin daha iyi bir model verir" DEĞİL** — yalnız *jeton başına daha fazla gerçek metin*
   ölçüldü. Kalite A1 sonrası ayrı bir kapıdır; hiçbir model eğitilmedi, hiçbir `.pt` yazılmadı.
2. **`<PROPER_NOUN>` yetenek kaybı GİDERİLMEDİ** — yalnız kabul edildi ve ölçüldü
   (%7.94). Modelin özel ad üretip üretemediği A1 sonrası ölçülmelidir.
3. **K5 boş bir kapıdır** — `train ∩ val = 0` inşa gereği imkânsızdır (val, train kesildikten
   *sonraki* makalelerden alınır); "GEÇTİ" hükmü **külliyat hakkında bir şey söylemez**.
   Tekrar eleme: **id yolu 0 kez**, **metin yolu 2 kez** tetiklendi.
   Bu, mekanizmaların *çalıştığını* gösteren zayıf bir kanıttır (birkaç vaka); külliyatın
   tekrarsız *olduğunu* göstermez.
4. **UNK %2.6346 bir "iyi" değil**, ölçülmüş bir orandır.
5. **Noktalama *yeteneği* ölçülmedi** — K1 noktalamanın **veride bulunduğunu** ölçer.
6. **Ceketin T-0067/T-0068 kusuru** ve **T-0066'nın "okuyor ama kullanamıyor"** bulgusu bu
   görevle ilgili değildir; giderilmiş sayılmaz.
7. **Bu külliyat hâlâ "doğal metin" değil** — morfoloji etiketleri (`CASE_ABL` vb.) ve
   `<PROPER_NOUN>` yer tutucusu içerir. Ölçülen: `<PROPER_NOUN>` %7.94.

---

## 7. Artefaktlar (tam 64 karakter sha256; bu belgenin kendisi ve bus durum dosyaları hariç)

| yol | bayt | sha256 |
|---|---|---|
| `scratch/anka_a1_design.py` | 15,969 | `de0d7cb78b4d80f1e0943abdb61722902d49406e09023400ff0b2a8d3630ff43` |
| `scratch/anka_a1_build.py` | 19,497 | `6632f56789647d7b9515e357168d460e745c5f42c839b8f90f1a9bf4a54632b1` |
| `scratch/anka_a1_report.py` | 24,461 | `10211b3d64588adbd13780b1447768c687cca1e7f2fac268e721b87f232404f0` |
| `scratch/anka_a1_close.py` | 8,752 | `7a1942af70ceea014ad2ef208fc3a863bdd1504feb4098d36117e21cd95e9184` |
| `data/anka_a1_pretrain.bin` | 200,000,000 | `383a9c890b75ec96195122b9981d7a73371d3bcaa3b9f6e7c2a188510d917cb8` |
| `data/anka_a1_pretrain.bin.meta.json` | 1,028 | `6afd761dc49f57a908acc04eccbc1d50a02902817cc0702ed02a12afe4468320` |
| `data/anka_a1_pretrain_val.bin` | 4,000,000 | `02f52063e73e4ac362297c9ecad41247aa621e168572b9cb3f6d302f734da801` |
| `data/anka_a1_pretrain_val.bin.meta.json` | 1,026 | `abc8817bb6356ba417361b5739e531e5c06bdb91fc91da65c89ff708cdfa289b` |
| `data/eval/anka_a1_build_2026-09-18.json` | 12,360 | `ee2531370534856b28bbc3e23e0cdcc48e027f16f6813940660366098f3e91a6` |
| `data/eval/anka_a1_corpus_design_2026-09-18.json` | 3,336 | `7aafa9db51560fcc35a1ccef70d9e2d76b2fdc78014798415cdab8690488a13f` |
| `data/eval/anka_a1_corpus_design_2026-09-18.md` | 5,315 | `44bd6f131a99ef06942fa380803e0121ce41213701cbbc5ef0d9482e75561145` |
| `data/anka_pretrain.bin` | 200,000,000 | `0e55f0c0c24617a9323059c71563a9ce943177ccc90cd56824be1a7f26d2b164` |
| `data/anka_pretrain_val.bin` | 4,000,000 | `4be459821814e9221f731d8a20c03ab790b460beb05619d68ba331028978141b` |

Bu belgenin kendi sha256'sı tabloya **konmadı** (kendine referans); üreticinin `[md]` satırında
basılır. Bus'a ait `state/tasks|results` dosyaları da konmadı — kapanış çağrısı onları kendisi
günceller ve tablo sahte ihlal üretirdi (T-0068 dersi).

---

## 8. Sonraki adım

**T-0073 (A1 planlama):** GPU verim sondası (sandbox **DIŞI**) — blok 128/256/512 × yığın 8/16/32
üzerinde **jeton/sn ölçümü**; adım bütçesi **ölçümden** ilan edilsin.

**Basit aritmetik (yalnızca ölçülmüş ~1,55 sn/adım @blok 128 ile):**

| | değer |
|---|---|
| adım (blok 128) | 100,000,000 ÷ 128 = **781,250 adım** |
| süre | 781,250 × 1,55 sn = **336 saat ≈ 14.0 gün** |
| blok 512 olsaydı (adım süresi *aynı* kalsaydı) | 195,312 adım × 1,55 sn = 84 saat |

⇒ Kullanıcının **~1 gün** bütçesinin çok üzerinde. **Blok büyütmek adım sayısını düşürür ama adım
süresini de büyütür**; bu yüzden "blok 512 ⇒ 84 saat" gibi bir
ölçekleme **yapılamaz** — sonda tam da bunu ölçer. *(Boyut kontrolü: 78.125 adım = **10 M** jeton
eder; 100 M jeton **781.250** adımdır — bu iki sayı sık karıştırılıyor.)*

**T-0074 (A1):** ön-eğitim koşumu (MPS, sandbox DIŞI, makinede başka GPU işi **YOK** — T-0052).

---

## 9. Ölçüm SONRASI bulunan ve düzeltilen kusurlar — **bu raporun kendisi**

Bu rapor taslak hâlinde denetlendi; **beşi de raporun kendi metninde** olan kusurlar bulundu ve
düzeltildi. Kapılara **dokunulmadı** (kural: ölçüm sonrası bulunan kusur rapora yazılır, kapıya değil).

| # | kusur | tür | düzeltme |
|---|---|---|---|
| 1 | §1'de T-0071'in bulgusu **%35,6** (400 makalelik örnek) diye yazılmıştı; T-0071'in *tam külliyat* ölçümü **%39.1** | **küme karışması** | tam külliyat değeri T-0071'in dosyasından yeniden ölçüldü; §1'de küme ayrımı açıkça yazıldı |
| 2 | §6/3 "tekrar eleme **tetiklenmedi**" diyordu, ama `tekrar_metin`=2 | **iddia ölçümle çelişiyor** | "id 0 · metin 2 kez tetiklendi ⇒ zayıf kanıt" olarak düzeltildi |
| 3 | `jeton/belge` etiketi train jetonunu **train+val** belgeye bölüyordu | **etiket/ölçü uyuşmazlığı** | tablo train-only ve val-only satırlara ayrıldı |
| 4 | §8'de süre `jeton/128/1.55` ile hesaplanmıştı ⇒ **1,55'e bölünmüş**; 140 saat | **boyut hatası** (adım × sn/adım) | **336 saat**; formül ve boyut kontrolü açıkça yazıldı |
| 5 | §8 metni "blok 512 ⇒ **42** saat" diyordu; tablo **84** saat (42, blok **1024**'ün değeri) | **metin ↔ tablo uyuşmazlığı** | metin artık tabloyla **aynı formülden** üretiliyor |

**Neden kayda geçiyor:** 4. kusur sayıyı **2,4× küçük** gösteriyordu ve doğrudan operatörün bütçe
kararını besliyor; sessizce düzeltilseydi "140 saat" rakamı bir yerde yaşamaya devam ederdi.

İlgili: [[anka-kulliyati-duz-metin-degil]], [[anka-yeni-model-adi]], [[hf-onbelleginde-wikipedia-kurtarildi]],
[[tavan-artefakti-kapi-gecmez-kanitsizlik]], [[ilan-edilen-kural-ayirt-edici-olmali]],
[[tasarlayan-kendi-iddiasini-yanlislayabilmeli]].
