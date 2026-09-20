# Anka A1-r · Faz 1 (sözlük) + Faz 3 (sözlük büyütme) — ölçüm ve üretim raporu

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Yürütücü:** claude
**Plan:** `enchanted-wiggling-moon.md` · **Operatör kararı (bu pencere):** D2 + D3-mekanizma
yürür; **D1'e ve tie-break'e dokunulmaz**.

> **Bu raporun en önemli satırı bir ÜRETİM değil bir ÇÜRÜTMEdir:** planın Faz 0
> tablosunda G3 kapısı (`<PROPER_NOUN> ≤ %3,0`) D2'ye bağlanmıştı. Ölçüm, sözlük
> işinin yer tutucu kütlesinin **yalnız %11,1'ini** kapatabildiğini gösterdi ⇒
> **G3 bu turda yapısal olarak ulaşılamaz.** Ayrıntı §4.

---

## 1. Üretilen artefaktlar (tam digest)

| Yol | sha256 (tam) | Satır/boyut |
|---|---|---|
| `data/lexicon/roots.tsv` **(DONMUŞ · DEĞİŞMEDİ)** | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` | 52.373 veri |
| `scratch/anka_r1_yedek/roots_kaynak.tsv` (yedek) | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` | kaynakla **birebir** |
| **`data/lexicon/roots_anka_r1.tsv`** (YENİ) | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` | 52.581 veri (+208) · 873.755 bayt |
| `data/rebuild/vocab_base_32852.json` **(DEĞİŞMEDİ)** | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` | = `data/vocab_entity.json` (birebir) |
| **`data/rebuild/vocab_anka_r1_33036.json`** (YENİ) | `b2bb4c6fb4c43423a3070ed29af013a9326264ca72a4bbb0c6ede1c8d27bbfee` | 33.036 giriş · 732 K |
| `scratch/anka_r1_eklemeler2.tsv` (plan) | `0ad3b7c629f365fcb8cb287fcd56bf790cd6233218875280eb02d701ffb1aec7` | 244 satır |
| `scratch/anka_r1_lexicon.py` (üretici) | `data/eval/anka_r1_lexicon_2026-09-19.json` içinde | — |

**Dönüşümsüzlük:** donmuş `roots.tsv` **okundu, üzerine yazılmadı**; yedek digest'i
kaynakla birebir. `vocab_base_32852.json` üzerine yazılmadı. Çıktı yolları yeni.

---

## 2. Faz 1 — üreticinin kapıları

| Kapı | Ne iddia eder | Sonuç |
|---|---|---|
| S1 şema | tam 3 alan · TAB · LF · CR yok · başlık sabit | **GEÇTİ** |
| S2 round-trip | kaynak bu şemayla **birebir** yeniden üretilebilmeli | **GEÇTİ** (52.373 satır) |
| S3 nitelik | nitelik kümesi ilan edilen 6 değerden oluşur | **GEÇTİ** |
| S4 sayım | satır = kaynak + \|EKLE\| | **GEÇTİ** 52.373 → 52.581 (+208) |
| S5 küme | her EKLE kaynakta YOK · her GÜNCELLE kaynakta VAR | **GEÇTİ** (208/15) |
| S6 sınır | çekirdek/kuyruk kırığı tam `50473+208` | **GEÇTİ** (kırık 50.681) |
| S7 sonrası | **YAZILAN dosyadan** 35 ilan edilmiş hedef doğru kökle çözülür | **GEÇTİ 35/35** |

Nitelik dağılımı (yazılan dosya): `-` 49.782 · `VOICING` 2.633 · `VOWEL_DROP` 152 ·
**`GEMINATION` 11** · **`VOICING,GEMINATION` 2** · **`VOWEL_DROP,GEMINATION` 1**.

> Birleşik nitelik değerleri (`VOICING,GEMINATION`, `VOWEL_DROP,GEMINATION`) bu
> dosyada **ilk kez** kullanıldı; okuyucu (`in` ile kontrol) destekliyordu, veride
> 0 satır vardı.

**Çıktı deterministik:** `--yaz` iki kez koşuldu, digest **aynı** (`ea874a73…`).

### 2.1 Bağımsız doğrulama (üreticinin kendi S7'sinden ayrı)

Çok-kümeli fark: kaynakta olup çıktıda değişen **yalnız 15 satır** (hepsi ilan
edilen GÜNCELLE) · yeni **223 satır** (208 EKLE + 15 güncellenmiş biçim) ·
**52.358 satır birebir korundu**. Çekirdek sıralı, kırık indeksi 50.681, başlık
aynı, CR yok.

---

## 3. Faz 3 — sözlüğü büyütme kararı ÖLÇÜMLE verildi

`tokenizer.py:308-317` morfemin `stoi`da olmasını şart koşar ⇒ **Faz 1 tek başına
hiçbir şey yapmaz**. Ölçüm (`scratch/anka_r3_on_olcum.py`, 236.573 yüzey ×
2.449.120 ham jeton, sıklık ağırlıklı):

| Kol | N | yer tutucuya düşen morfem pozisyonu kurtarılan |
|---|---|---|
| TABAN | 32.852 | 0 (taban) |
| **DAR** (yeni D2 lemma kimlikleri) | **33.036** | **33.720** (%57,6) |
| GENİŞ (tüm eksik meşru lemma) | 49.185 | 58.562 (%100), marjinal **+24.842** |

**Karar: DAR.** GENİŞ, +16.149 embedding satırı (tablo ≈ %49 büyür) karşılığında
yalnız +24.842 pozisyon (kelime pozisyonlarının **%0,7'si**) veriyor. DAR, aynı
sınıfın %57,6'sını alıyor.

**Çapraz doğrulama:** kelime-süzgeçli TABAN yer tutucu kütlesi
`EKSIK_BUYUK 56.860 + ANALIZSIZ_BUYUK 363.712 = 420.572`; bağımsız ölçülen
(`anka_r1_on_kapsam_2026-09-19.json`) PN kütlesi `8,0186% × 4.738.026 ≈ 379.942`.
Fark **%10,7** ve artık açıklanıyor: D4-kesme sınıfının bir kısmını gerçek
tokenizer kesme yoluyla çözüyor. İki bağımsız ölçüm **aynı mertebede**.

---

## 4. ⚠️ ÇÜRÜTME — planın D2 öncülü ölçümle yanlışlandı

Kelime-süzgeçli **468.968** analizsiz/yer-tutucu pozisyonun sınıf dağılımı:

| Sınıf | Pozisyon | Pay | Sözlükle düzelir mi |
|---|---|---|---|
| Açık sınıf kişi/yer adı (`Mustafa`, `Robert`, `William`…) | 265.953 | %56,7 | **HAYIR** — D2 kapsam kuralım açık sınıfı dışarıda bırakıyor |
| Küçük harf analizsiz (`yürürlüğe`, `yönetici`, `işgali`…) | 100.870 | %21,5 | **KISMEN** — çoğu morfofonoloji (D1/D3 ailesi), sözlük değil |
| **D4 kesme** (`Türkiye'nin`, `İstanbul'da`, `Atatürk'ün`) | 71.256 | %15,2 | **HAYIR** — tokenizer işi |
| Kısaltma/Roma rakamı (`II`, `I`, `ABD`, `MÖ`) | 25.894 | %5,5 | kısmen |
| TÜM-BÜYÜK (`UNESCO`, `TÜBİTAK`) | 609 | %0,1 | kısmen |

⇒ Sözlük işi (GENİŞ kol bile) kütlenin **%11,1'ini** kapatıyor.
⇒ **G3 (`PN ≤ %3,0`) D2 ile ulaşılamaz**; baskın kaldıraçlar D4 (kesme) ve açık
sınıf adlar. Bu, planın "eksik lemma sınıfı PN'in ana nedenidir" varsayımının
**çürütülmesidir** — sayı planın kendi Faz 0 tablosuna karşı ölçüldü.

### 4.1 Teşhis sırasında bulunan, kapsam DIŞI bırakılan gerçek kusur

`yürürlüğe`, `işgali`, `yönetici`, `hâline`, `demiryolu` gibi yaygın kelimeler
**kök bulunuyor ama yol tamamlanmıyor**. Sürüldü: `PhonologyEngine.resolve_affix`
`A` arkifonemini `phonology.py:139`'da **uyumluyor**; `final`+`DA` → `finalda`
çıkıyor çünkü `final`/`işgal`/`hâl` **alıntı sözcük istisnalarıdır** (son seslem
`a` olmasına karşın ön-uyum: `finalde`, `işgalde`, `hâlde`). Bu **D1/D3 ailesi** ve
operatör kararıyla **ertelenmiştir** ⇒ bu pencerede dokunulmadı, kapsam
genişletilmedi.

---

## 5. Bu pencerede ölçülen KENDİ kusurlarım

| # | Kusur | Nasıl yakalandı | Etki |
|---|---|---|---|
| 1 | S2 round-trip'te **başlık satırını atladım** | S2 düştü, fark tam 21 bayt = `len("lemma\tpos\tattributes\n")` | Üretici durdu; 1. koşum boşa gitmedi ama kapı olmasa yanlış dosya yazılırdı |
| 2 | `[S4/S6] kuyruk … DEGISMEDI` iddiası **yalnız uzunluğa** bakıyordu | bağımsız doğrulama: kuyrukta **1 satır** değişmiş (`sed`) | **Zayıf denetim**: uzunluk eşitliği içerik eşitliğini KANITLAMAZ. İçerik karşılaştırması eklendi |
| 3 | O2 süzgecim `l.isalpha()` idi ⇒ `Abu Dabi`, `Bosna-Hersek`, `Addis Ababa` elendi | O2 düştü, DAR ⊆ GENİŞ ihlali | Kapı **doğru** çalıştı, kusur süzgeçteydi; ilan edilen `mesru()` yüklemi konuldu |
| 4 | DAR kümesine **zaten stoi'da olan** lemmaları kattım | O2 ikinci kez düştü | `dar = (yeni − taban) − stoi` olarak düzeltildi |
| 5 | **`TABAN` kolunu hiç doldurmadım** (sıfır bıraktım) ⇒ `hukum` "kazanç"ı **işareti ters** ve tabansız üretti | sayı r1_on ile çelişti, kaynağa inildi | Sessiz çöp sayı; TABAN da bir kol yapıldı |
| 6 | `ANALIZSIZ_KUCUK=338.212`'i "yer tutucu" okumak üzereydim | r1_on'un UNK'sı 74.540 ile çelişti | Kova **rakam/noktalama yığınıyla** dolu (`d` 28.301, `ö` 17.988); kelime-süzgeçli ikinci sayım eklendi |

**Ortak ders:** 1, 2 ve 5, aynı ailedendir — *bir denetimin kapsamı iddiasından
darsa, denetim "temiz" der ve bu sessizdir.* Üçünde de sayı doğruydu, **iddia**
yanlıştı.

---

## 6. Hüküm

- **Faz 1: TAMAM.** 35/35 hedef yazılan dosyadan doğrulandı; donmuş dosya değişmedi.
- **Faz 3: TAMAM.** N = 33.036, önek değişmezliği (id 0..32851) doğrulandı,
  yazılan dosyadan 5/5 yüzeyin tüm morfem kimlikleri `stoi`da.
- **Çürütülen iddia:** D2 ⇒ G3. Gerekçe ve sayılar §4'te; kapı **değiştirilmedi**,
  bulgu rapora yazıldı ([[tasarim-sayisi-betikle-hesaplanmali]] kuralı).
- **Kapsam dışı bırakılanlar (bilinçli):** D1 · tie-break · D4 (kesme) · açık sınıf
  kişi adları · alıntı sözcük morfofonolojisi · GENİŞ kol (ölçülmüş maliyetiyle
  reddedildi, `hukum` bloğunda saklı).

**Sıradaki:** Faz 4 (külliyat derleme, ~35 dk CPU) → Faz 5 (sıcak başlangıçlı
yeniden eğitim, ~12,6 sa MPS).

---

## 7. ⚠️ EK — §3 kararı YÜRÜRLÜKTEN KALKTI (aynı oturumda, Faz 4 öncesi ölçüldü)

**Tetikleyici:** Faz 4 hazırlığında pozitif kontrol, `İstanbul`'un derleyicide
**çözüldüğünü** (`Kok = İstanbul`) ama `stoi`'da **olmadığını** gösterdi ⇒ yüzey
`<PROPER_NOUN>` basıyor. `İstanbul` **yeni** bir lemma değil (temel `roots.tsv`de,
İ/I onarımından sonra çözülüyor) ⇒ §3'ün DAR kuralı onu **atlamıştı**.

### 7.1 Kusur: denetimin kapsamı iddiasından DARDI (§3'ün kendi kusuru)

| | §3'ün yaptığı | Olması gereken |
|---|---|---|
| Aday kümesi | `lem_yeni − lem_taban − stoi` = **"yeni lemmalar"** | **"kök kimliği `stoi`da olmayan lemmalar"** |
| Eksen | lemmanın **kimliği** (yeni mi?) | **mekanizma** (kimlik eksik mi?) |

İki küme **aynı değil** ve fark tam olarak baskın sınıfta: `İngiliz`, `Fransız`,
`İtalyan`, `İstanbul`, `Hollanda`, `Kanada`, `İspanya`… hepsi temel sözlükte **var**,
kimlikleri **yok**. (Aile: [[denetim-kapsami-iddiadan-dar]].)

### 7.2 Yeniden ölçüm (`scratch/anka_r4_ek_kimlik.py`, aynı 236.573 yüzey önbelleği)

Kural **ölçümden önce ilan edildi**: aday = DAR (koşulsuz) ∪ `stoi`-dışı **kök
kimlikleri**; N\* = kurtarılan ≥ %90 × erişilebilir en iyi.

| | §3 DAR | §3 GENİS | **§7 düzeltilmiş (N\*)** |
|---|---|---|---|
| ek kimlik | 184 | 16.333 | **262** (184 ∪ 182) |
| N | 33.036 | 49.185 | **33.114** |
| kurtarılan pozisyon | 33.720 (%57,6) | 58.562 (%100) | **52.714 (%90,0)** |
| tablo büyümesi | +%0,56 | +%49,2 | **+%0,80** |

**§3'ün GENİS maliyet hesabı yanlış değildi, ama ekseni yanlıştı:** `KOK_EKSIK`
sınıfını taşıyan **yalnız 1.146 kimlik** var; GENİS'in fazladan 15.187 kimliğinin
önbellekteki görülme sayısı **sıfır** ⇒ ölçülen "GENİS = %100" ile "1.146 kimlik =
%100" aynı şeydir. Yani GENİS'i reddetme **gerekçesi** doğruydu; **DAR'ı seçme**
gerekçesi yanlıştı.

> **Bedel karşılaştırması tek cümlede:** §3 DAR'a göre **+78 satır** karşılığında
> **+18.994 pozisyon** (%57,6 → %90,0). §3'ün "GENİS reddedildi" hükmü duruyor;
> §3'ün "DAR seçildi" hükmü **bu ek ile değiştirildi**.

### 7.3 §4'ün sınıf tablosu — yalnız bir terim düzeltmesi, yüzdeler DEĞİŞMEZ

§4'teki tablo "Açık sınıf kişi/yer adı **HAYIR** — D2 kapsam kuralım açık sınıfı
dışarıda bırakıyor" diyordu. Bu, `Mustafa`/`Robert`/`William` (derleyici **yol
bulamıyor** = `ANALIZSIZ`) için doğrudur; ama `İstanbul`/`Londra` sınıfı **derlenir**
(`KOK_EKSIK`) ve **sözlük kimliği eklenerek düzelir**. §4 bu iki alt sınıfı tek
terimde toplamıştı.

**Yüzdeler doğru kalıyor:** §4'ün `468.968` paydası `ANALIZSIZ_BUYUK + ANALIZSIZ_KUCUK`
(kelime-süzgeçli) idi ve `%11,1` oranı `58.562 / (58.562 + 468.968) = %11,10` olarak
tutuyor — yani §4 **doğru paydayı** kullanmış. Düzeltilen tek şey **atıf**tır:
`58.562`'nin `56.860`'ı kimlik eklenerek **kapanabilir**, `468.968`'in
`363.712`'si **kapanamaz**. **§4'ün hükmü (G3, D2 ile yapısal olarak ulaşılamaz;
baskın kaldıraç D4 kesme + açık sınıf adlar) DEĞİŞMEDİ.**

### 7.4 Yeni artefakt

| Yol | sha256 (tam) | N |
|---|---|---|
| `data/rebuild/vocab_anka_r1_33114.json` (**YÜRÜRLÜKTE**) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` | 33.114 |
| `data/rebuild/vocab_anka_r1_33036.json` (§3, **artık kullanılmıyor**) | `b2bb4c6fb4c43423a3070ed29af013a9326264ca72a4bbb0c6ede1c8d27bbfee` | 33.036 |

Süperset özdeşliği **ölçüldü**: `stoi(33036) ⊂ stoi(33114)`, fark **tam 78**
(`Adıyaman`, `Ardahan`, `Bartın`, `Gaziantep`, `Giresun`, `Hakkâri`, `Isparta`,
`Iğdır`, `Kars`, `Ermenice`, …). §3'ün dosyası **silinmedi** — kusurun kanıtı
olarak duruyor.

**Faz 4/5 bu dosyayla koşar.** Kapı ilanı: `data/eval/anka_r4_kapi_ilani_2026-09-19.md`.

