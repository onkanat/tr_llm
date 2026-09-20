# Faz 0 · Ölçüm Kabı ve İlan Edilen Kapılar — SONUÇ

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Yürütücü:** claude (tek yürütücü)
**Aşama:** Faz 0 **TAMAMLANDI** · **Hüküm: kapıların hiçbiri geçmiyor**

Bu belge, plandaki **Faz 0**'ın (`.claude/plans/enchanted-wiggling-moon.md`) kapanış
raporudur. Kapılar **ölçümden önce** ilan edildi ve sonra **değiştirilmedi**.

---

## 1. İlan edilen kapılar ve sonuçları

Ölçüm **A1-sadık** kapla yapıldı (`anka_r0e_kapsam_kesin.py`): A1 üreticisinin
14 temizleme kuralı **birebir**, belge bütünüyle `tok.encode()` ile kodlanır
(A1'in kendi yolu), **iki parquet'in ikisi** okunur.

| Kapı | İlan eşik | Ölçülen | Sonuç |
|---|---|---|---|
| **G1** en-uzun-eşleşme ihlali | ≤ %1,0 | **%2,3943** (2.668/111.432) | ✗ KALDI |
| **G2** `<UNK>` külliyat oranı | ≤ %1,5 | **%1,7200** | ✗ KALDI |
| **G3** `<PROPER_NOUN>` oranı | ≤ %3,0 | **%8,0446** | ✗ KALDI |
| **G4** literal ad / yer tutucu | ≥ 1,0 | **0,3747** | ✗ KALDI |
| **G5** sözlük kapsamı | ≥ %95 | Faz 4'e ertelendi (yazılan `.bin`'den) | — |
| **G6** tur özdeşliği (TAM+ALTERNATİF) | ≥ %99,0 | **%96,9892** | ✗ KALDI |

### G6'nın tek kusuru var — ve onarılırsa kapı GEÇER

G6'nın `HATA` dalı ilk kez **ayrıştırıldı**:

| HATA alt sınıfı | Adet | Pay |
|---|---|---|
| **kesme sözleşmesi** | **2.831** | **%84,4** |
| geri-boş | 0 | %0 |
| diğer | 524 | %15,6 |

> **Kesme sözleşmesi onarılırsa G6 = %99,5298 → ilan edilen %99,0 eşiğini GEÇER.**
> Yani "tur özdeşliği kötü" değil; **tek bir yapısal kusur** G6'yı düşürüyor.

---

## 2. Kesme sözleşmesi — bu turun en ağır bulgusu

```
c.compile("Ankarada")        -> ["Ankara", "CASE_LOC"]
dec.decompile_sentence(...)  -> "Ankara'da"     ← decompiler KESME KOYUYOR
c.compile("Ankara'da")       -> []              ← derleyici OKUYAMIYOR
```

**Ama kusur "asimetri"den daha ağır.** Örnekler:

| Yüzey | Seçilen analiz | Decompiler çıktısı | Türkçe yazımı |
|---|---|---|---|
| `Kıpçaklar` | `[Kıpçak, PLURAL]` | `Kıpçak'lar` | **`Kıpçaklar`** — çoğulda kesme YOK |
| `Bulgarları` | `[Bulgar, POSS_3PL]` | `Bulgar'ları` | **`Bulgarları`** |
| `kuzeyine` | `[Kuzey, POSS_2SG, CASE_DAT]` | `Kuzey'ine` | **`kuzeyine`** — küçük harfli! |

Yani decompiler **(a)** Türkçe yazımının kesme kullanmadığı yerde kesme koyuyor
(çoğul/türev eklerinden önce), **(b)** küçük harfli bir kelimeyi büyütüyor.
Kırılma tam **D4 (kesme, 17.398 benzersiz yüzey)** sınıfında.

**Sonuç:** "kesme zaten token (32850)" demek "kesme doğru çözülmüş" demek değildir.
Faz 2'nin D4 maddesi bu bulguyla **yeniden çerçevelendi**.

---

## 3. V1 kuralı ÇÜRÜDÜ — D1 açık tasarım sorusudur

Planın Faz 2/D1 kuralı `score = len(kök)*100 - len(morphemes)` (V1) idi.
**Ölçümle çürütüldü** (`anka_r0d_v1_curutme`, rc=0):

- Bugün **doğru** olan **5/5** okumayı bozuyor:
  `bunun→bun+POSS` · `onun→on+POSS` · `ölmeden→ölme+CASE_ABL` ·
  `topraklar→toprakla+TENSE_AORIST` · `günümüzde→günü+POSS_1PL+CASE_LOC`
- ROOT oracle **24/25 → 23/25**
- **2.403 değişimin 2.403'ünde morfem sayısı EŞİT** (azalma 0) ⇒ V1 yapısal
  iyileştirme değil, **beraberlik kırıcı** — ve yanlış tarafa kırıyor
- Bu koşumda `V1 kök değiştirir` = **2.668** = **G1 ihlal sayısı** (yapısal özdeşlik:
  V1 tam olarak "daha uzun geçerli kök varken" seçimi değiştirir)

**İhlal örnekleri V1'in neden tutarsız olduğunu gösteriyor:**

| Yüzey | Bugün | V1 | V1 haklı mı? |
|---|---|---|---|
| `olan` | `[o, DERIV_lAn]` | `[ol, PART_An]` | **evet** |
| `kurdu` | `[kur, COPULA_PAST]` | `[kurt, POSS_3SG]` | hayır |
| `belli` | `[bel, DERIV_lI]` | `[bell, POSS_3SG]` | hayır |
| `günümüzde` | `[gün, POSS_1PL, CASE_LOC]` | `[günü, POSS_1PL, CASE_LOC]` | hayır |

Ne "en uzun kök" ne "en kısa kök" D1'i tek başına tutuyor.

### D1'in yeni çerçevesi (tartışma belgesinden)

D1 bir **skorlama** değil **ad alanı (namespace)** problemidir. `aklı→ak+DERIV_lI`
ile `bunun→bun+POSS` **aynı kök nedendir**: skorlayıcı, sözlükte duran kısa/türemiş
gövdeyi meşru kök görüyor ve beraberliği ona kırıyor. Ayrıntı:
`data/eval/anka_derleyici_istisna_tartisma_2026-09-19.md`

---

## 4. İki ölçüm — neden ikisi de duruyor

| Kapı | `anka_r0_kapsam` (ilk kap) | `anka_r0e_kapsam_kesin` (A1-sadık) |
|---|---|---|
| G1 | %10,713 *(şişik ölçüt)* | **%2,3943** |
| G2 `.bin` | %1,8417 | **%1,7200** |
| G3 `.bin` | %9,9207 | **%8,0446** |
| G4 | 0,3441 | **0,3747** |
| G6 | %97,3223 | **%96,9892** |
| Örneklem | 3.000 makale / 215.881 yüzey | 3.212 makale / 236.573 yüzey |

**Hangisi geçerli, ölçüt ne?** Gerçek artefakt: A1 külliyatının kendi `.bin`'i
(ölçüldü: PN **%7,9365** · UNK **%2,6346**).

- **r0e'nin PN'si (8,0446) A1'in gerçek külliyatını 0,108 puan içinde yeniden
  üretiyor** ⇒ kap, modellediği artefakta karşı **bağımsız olarak doğrulandı**.
- r0'un 9,9207'si bunu tutmuyor ⇒ `.bin` paydası A1'in yolunda değildi.
- G1 için ilk ölçüt ("daha uzun aday **var mı**") yanlıştı; doğrusu ("geçerli
  **analizde** daha uzun kök var mı") **%2,3943** veriyor — r0c'nin kısaltılmış
  temizlemeyle bulduğu %2,434'e yakın ⇒ **temizleme kısaltması G1'i oynatmamış**
  (ama yüzey sayısını oynatmıştı: 215.881 / 208.971 / 211.703).

> **r0e bu sayıları geçersiz kılmıyor, kapsamını daraltıyor.** İlk kap, kusurları
> *bulan* kaptır; r0e onları *doğru ölçen* kaptır. İkisi de kanıt olarak duruyor.

⚠️ **LIT karşılaştırması tanıma bağlıdır.** r0e'nin "literal ad" tanımı
(büyük harfle başlayan, `_` içermeyen, etiket olmayan jeton) A1 raporundaki
%2,30 tanımıyla **aynı olduğu doğrulanmadı** ⇒ 3,0141 vs 2,3029 farkı
**tanım farkı olabilir**, bulgu değil.

---

## 5. Mekanizma tabloları (`anka_r0_kapsam`, 3.000 makale)

| Mekanizma | Benzersiz yüzey | Kelime jetonu | % |
|---|---|---|---|
| `D2_eksik_lemma` | **68.934** | 310.527 | %14,62 |
| `D3_morfofonoloji` | **18.799** | 46.330 | %2,18 |
| `D4_kesme` | **17.398** | 49.066 | %2,31 |
| `D4_ek_parcasi` | 16 | 827 | %0,04 |
| `ARTIK_isaretleme` | 348 | 5.772 | %0,27 |
| `RAKAM` | 1.027 | 1.783 | %0,08 |

### Sınıf payları (r0e, jeton ölçeği)

| Sınıf | r0e `.bin` | A1 gerçek külliyat |
|---|---|---|
| diğer | %87,2213 | — |
| `<PROPER_NOUN>` | %8,0446 | %7,9365 |
| `<UNK>` | %1,7200 | %2,6346 |
| literal ad | %3,0141 | %2,3029 *(tanım farkı olabilir)* |

---

## 6. Kontrol tablosu (fail-closed; düşen her kontrol `rc=2` ile DURdurur)

| Kontrol | Beklenen | Sonuç |
|---|---|---|
| Derleyici lexicon'la kurulu (poz. kontrol) | `başkenti→[başkent,POSS_3SG]` | **TEMİZ** (4 poz + 2 neg) |
| G4 filtresi (poz. + neg. küme) | poz. kaybolmaz, neg. sızmaz | **TEMİZ** · literal ad id 1.354 |
| V1 çürütme kontrolü | bugün-doğru 5/5 bozulmalı | **5/5 BOZULDU** ⇒ V1 çürüdü |
| G6 araç kontrolü | decompiler tur atmalı | **3/3 TAM** (özel ad kesme'li) |
| İki parquet de okundu | eksik dosya yok | **TEMİZ** (1.688 + 1.524) |
| `.bin` dtype | `uint16` (poz. kontrol) | **doğrulandı** |

---

## 7. Bu turda ölçülen KENDİ kusurlarım

| # | Kusur | Nasıl çıktı | Etki |
|---|---|---|---|
| 1 | **Üç ayrı temizleme.** r0c/r0d A1'in 14 kuralını **7'ye kısaltmıştı** | Üç betik üç farklı yüzey sayısı verdi (215.881/208.971/211.703); plan "BİREBİR kopyalanır" diyordu | r0c/r0d'nin sayıları **geçersiz**; r0e'de düzeltildi |
| 2 | **`.bin` paydası A1'in yolunda değildi** | r0 G3 %9,9207 ↔ A1 gerçek %7,9365 | r0e belgeyi bütünüyle kodlayarak düzeltti; **PN 0,108 puan içinde tuttu** ⇒ kap doğrulandı |
| 3 | **G1 ölçütü şişik** ("aday var mı" ≠ "geçerli analizde daha uzun kök") | `anlamlara`: `anlamla` aday ama `ra` geçersiz | %10,713 → **%2,3943** (~4,4× şişme) |
| 4 | **`c.decompile_sentence` yok** — o metot `MorphemeDecompiler`'da | `except` yutuyordu ⇒ **her şey HATA** çıkardı | Pozitif kontrol eklendi; kontrol eklenmeden koşsaydı **sahte bulgu** üretirdi |
| 5 | **Örnekleyici ikinci parquet'i hiç açmıyordu** | dış `break` dosya döngüsünü de kırıyordu | `eksik` kapısı yakaladı; iki parquet'in ikisi okunuyor |
| 6 | **Kontrol etiketi ≠ beklenen değer (3. kez)** | `Ankarada` bekliyordum, araç `Ankara'da` döndürdü | fail-closed düştü **ve bulguyu üretti** (§2) |
| 7 | G4 `w in buyuk_id` (str vs int kümesi) | daima `False` ⇒ G4 **0,0** | sessiz tip hatası; düzeltildi |
| 8 | G6 sınıflandırıcısı **asla düşemezdi** (`alternatif` else dalında) | `tur_ok+alternatif ≡ toplam` | üç yollu sınıflandırıcı + ulaşılabilir HATA dalı |
| 9 | "Bitişik önek yanlı" örnekleme yanlılığı hipotezi | Bölgesel sonda: PN %16,02/13,72/16,55/16,10/15,16 | **ÇÜRÜDÜ** — fark payda farkıydı |
| 10 | Kuyruk bloğu "sahte lemma" hipotezi | 52.373 lemma, yalnız **3** ayrışabilir | **ÇÜRÜDÜ** — `anlamla`/`camla` meşru VERB lemmaları |
| 11 | Regresyon oracle ayrıştırması | `split("ROOT:")[-1]` ROOT olmayanları sahte yaptı | "75/100 ulaşılamaz" **sahte bulgu**; gerçek ROOT oracle **25** |
| 12 | **Bellekten yazdığım sayı bayattı** | "14/14 `.bin` ≤ 32136 ⇒ noktalama YOK" | Bugün **21 `.bin`, 9'u noktalama taşıyor** (`anka_a1_pretrain` %11,021). Bellek düzeltildi |

---

## 8. Örneklem sapması (beyan)

İlan edilen örneklem **3.000 makale** (dosya başı 1.500) idi; gerçekleşen
**3.212** (1.688 + 1.524). Nedeni: `break` **parti sınırında** çalışıyor
(batch=200), parti taşması kadar fazla okunuyor. **Kusur değil, granülerlik** —
ama ilan edilen sayıdan saptığı için yazılıyor.

---

## 9. Hüküm

1. **Faz 0 tamamlandı.** Beş kapının beşi de **KALDI**; hiçbiri geçmiyor.
2. **G6'nın tek kusuru var** (kesme sözleşmesi, HATA'nın %84,4'ü). Onarılırsa
   **%99,53 ⇒ GEÇER.**
3. **V1 (en uzun kök) ÇÜRÜDÜ.** Faz 2/D1 **yeniden tasarlanmalı**; planın
   "TAM %81,24→%84,90" gerekçesi **geçersiz**.
4. **D1 = ad alanı problemi**, skorlama problemi değil (bkz. tartışma belgesi).
5. **Faz 3, zekirlik temsil kararına bağımlı** (`roots.tsv` nitelik alanı yalnız
   3 değer taşıyor; allomorf yok) ⇒ skor değişikliğinden **önce** çözülmeli.

### Çürütme maddesi

| İddia | Onu çürütecek ölçüm |
|---|---|
| "G6'nın tek kusuru kesme" | Kesme onarıldıktan sonra G6'nın %99'un altında kalması |
| "D1 ad alanı problemidir" | Ad alanı ayrıldıktan sonra beraberliklerin **hâlâ** yanlış kırılması |
| "r0e'nin kabı doğru" | PN'nin A1 gerçek külliyatından >0,5 puan sapması |

---

## 10. Tam digest tablosu

| Artefakt | sha256 (tam) |
|---|---|
| `scratch/anka_r0_kapsam.py` | `b94c1317011e0e8cb57eca32c89c6c10878addd01fda12ecd406441444d519c5` |
| `scratch/anka_r0e_kapsam_kesin.py` | `50e60120a24387a49c7e29873be1bae1965ca915a109ef64ac8398739d16faa6` |
| `data/eval/anka_r0_kapsam_2026-09-19.json` | `ed42aafa0647a1c3f4e022bf128ffe6fdef6508bafda63c8638e64cb555304cb` |
| `data/eval/anka_r0b_ihlal_temizligi_2026-09-19.json` | `b80c9a979b0bf52e3f9742519a214c279a8f95e7496e06dede17645d07150616` |
| `data/eval/anka_r0c_g1_duzeltme_2026-09-19.json` | `3305cb143570af339faa0ba166b19e6f2316c707a484f48916b00c93d1f03464` |
| `data/eval/anka_r0d_v1_curutme_2026-09-19.json` | `badcbf0b707d3153b35a2bb6fc90f42d40ceadd7cc2503fc0cca72ee03f0366b` |
| `data/eval/anka_r0e_kapsam_kesin_2026-09-19.json` | `b93e3e614d175faceb5beb56ab4fdb4a995011ebaf6eb7b4679f2b4c8f1884d4` |
| `data/lexicon/roots.tsv` (donmuş) | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/rebuild/vocab_base_32852.json` (donmuş) | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` |
| `tests/morphology_regression_100.json` | `cbe00f31c4dcc34ba2baed5317f1bf38b4e69b130c5271205c1dd5709dab18da` |

**Donmuş hiçbir yola yazılmadı.** Bu fazda üretilen her şey `scratch/` ve
`data/eval/` altındadır (`data/eval/` `data/**` kuralının tek istisnasıdır).

---

## 11. EK (19 Eyl 2026, Faz 2 ön-ölçümü sırasında) — G1'in ÖNCÜLÜ ÇÜRÜDÜ

> Bu ek, yukarıdaki tabloyu **silmez**; G1 satırının nasıl **okunması gerektiğini**
> değiştirir. Gerekçe: `scratch/anka_r2_on_olcum.py` (Faz 2 ön-ölçümü) G1'i
> **bilinen yer-gerçeği** olan 9 vaka üzerinde sınadı.

G1'in öncülü *"daha uzun kökle çözülebilen bir yüzeyde kısa kök seçilmesi = hata"*dır.
Bu öncül **ölçümle çürüdü**:

| Vaka | Gerçek durum | G1 ihlali sayıyor mu? |
|---|---|---|
| `bunun` `onun` `ölmeden` `topraklar` `günümüzde` | **DOĞRU okunuyor** | **EVET (5 yanlış pozitif)** |
| `ağacı` `aklı` `kanadı` `gönlü` | **YANLIŞ okunuyor** | EVET (4 gerçek pozitif) |

**9 vakanın 5'inde ölçüt, doğru okumayı ihlal sayıyor.** Yani G1 *"hata"*yı değil,
*"kısa kökle çözülmüş"*ü ölçüyor.

### Sonuç: G1 sayısının düşmesi tek başına KANIT DEĞİLDİR

1. G1 `max(rz) > rz(seçilen)` diye sorar. Bir aday yolunu **silmek** G1'i tanım
   gereği düşürür — seçim iyileşmeden. ⇒ G1'i düşüren her tasarım, düşüşün ne
   kadarının *sildiği alternatiften* geldiğini **ayrıca** göstermek zorundadır.
2. Bu yüzden §9'daki "G1 ≤ %1,0 kapısı" **hedef olarak kalır** ama bir tasarımın
   *doğruluğunu* göstermez; yalnız **tutarlılık** gösterir.
3. Doğruluk için **etiketli oracle** gerekir. Elimizde yalnız **9 etiket** var;
   aday tasarımlar **18.000+ kararı** değiştiriyor ⇒ *etiket/karar oranı ~1/2000*.

> **Faz 2, oracle olmadan açılamaz.** Bu, §9/5'in (Faz 3 zekirlik kararına bağımlı)
> yanına **ikinci bir ön koşul** olarak eklenir. Ayrıntı:
> `data/eval/anka_r2_on_olcum_2026-09-19.md`.
