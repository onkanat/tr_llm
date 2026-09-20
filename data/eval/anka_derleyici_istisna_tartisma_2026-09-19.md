# Derleyici İstisna Mimarisi — Tartışma Açılışı

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Durum:** AÇIK — karar bekliyor
**Yazan:** claude (tek yürütücü) · **Danışman:** Antigravity'ye danışma gönderildi
(`2026-09-19T08-24-05Z-claude.json`, GÖREV DEĞİL)

---

## 1. Soru

> *"Türkçede istisnalar var; bu istisnaları compiler'a gömmek yerine token atamak
> daha iyi değil mi?"*

Bu soru iki seçenekli kurulmuş: **kural (compiler)** ya da **token (sözlük)**.
Ölçüm, bu çerçevenin eksik olduğunu gösteriyor — ama operatörün sezgisi tamamen
yanlış değil, **yanlış sınıfa** uygulanmış.

---

## 2. Ölçüm (hepsi 19 Eyl 2026, bu oturumda, fail-closed kap ile)

Kap: derleyici `LexiconManager.load_from_tsv` ile kuruldu; pozitif kontrol TEMİZ
(`başkenti → [başkent, POSS_3SG]`). `LexiconManager` **boş trie** ile başlar;
`load_from_tsv` atlanırsa her şey sessizce boş döner (T-0078'in ölüm sebebi).

### Ö1 — Sınıf bazında derleyicinin bugünkü davranışı

| Sınıf | Geçen | Çöken |
|---|---|---|
| 1 kaynaşma (voicing) | `kitabı`✓ `ekmeği`✓ `yemeği`✓ | `ağacı`→`[ağa,DERIV_CI]` · `kanadı`→`[kana,TENSE_PAST]` |
| 2 ünlü düşmesi | `burnu`✓ `oğlu`✓ | **`ağzı` → 0 YOL** · `aklı`→`[ak,DERIV_lI]` · `gönlü`→`[gön,DERIV_lI]` |
| 3 ünsüz ikizleşmesi | — | **`hakkı` · `hissi` · `affı` · `zammı` → 4/4 0 YOL** |
| 4 tek heceli t/d | `gidiyor`✓ `tadı`✓ `gidecek`✓ | **`ediyor` → 0 YOL** · `edecek`→`[edecek]` (tek jeton) |
| 5 **zekirlik** | — | **`bana`→`['bana']` · `sana`→`['sana']` · `suyun`→`['suyun']` (TEK JETON)** · `bunu`→`[bun,POSS_3SG]`✗ · `neyin`→`[ney,POSS_2SG]`✗ |
| 6 iyelik+araç `-yla` | — | **`adıyla` · `hâline` → 0 YOL** · `aracıyla`→`[ara,DERIV_CI,CASE_INS]`✗ |

### Ö2 — Veri modeli ne ifade EDEBİLİYOR

`roots.tsv` nitelik alanındaki **tüm** değerler:

```
'-'          49.587
'VOICING'     2.635
'VOWEL_DROP'    151
```

**Toplam 3 farklı değer.** `allomorf` / `supplet` / `irregular` / `zekir` içeren
değer: **YOK.** Yani veri modeli zekirliği (suppletion) **ifade edemiyor** —
allomorf için ayrı bir alan yok.

### Ö3 — ÇEKİRDEK BULGU: zekirlik ZATEN "token" olarak çözülmüş — ve zarar veriyor

`roots.tsv`'de **lemma olarak** duran çekimli yüzeyler (grep ile doğrulandı):

```
satır  5699: bana    PRON  -
satır 35408: sana    PRON  -
satır 51631: suyun   NOUN  -
satır  8684: bun     NOUN  -        (8685: bun VERB)
satır 32023: on      ADJ   -        (32024 on NOUN, 32025 on VERB)
satır 31298: ney     NOUN  -
satır  3318: anlamla VERB  -        51631 dışı örnekler: gölgele 17418, toprakla 41102
```

Yani **proje zaten operatörün önerdiği şeyi yapıyor**: istisnayı sözlüğe girdi
olarak atıyor. Zekirlik bir kural değil, bir **lemmadır**.

**Ve tam bu mekanizma D1'i zehirliyor.** "En uzun kök kazanır" (V1) kuralı
`bunun → bun`, `topraklar → toprakla`, `günümüzde → gölgele` seçiyor; **bugün
DOĞRU olan 5/5 okumayı bozuyor** (ölçüldü, rc=0, `anka_r0d_v1_curutme`).

Sonuç: **token-atama bu mimaride "kural koymama" değildir — derleyicinin KÖK
sandiği ad alanını kirletmektir.** Aynı ad alanına giren her çekimli yüzey,
skorlayıcı için meşru bir kök adayı olur.

### Ö4 — "Kapalı küme" iddiası ölçekle sınanınca

`D2_eksik_lemma` 68.934 · `D3_morfofonoloji` 18.799 · `D4_kesme` 17.398 benzersiz
yüzey. `VOICING` tek başına **2.635** giriş ve sınıf **AÇIK** (t/k/p/ç-final her
yeni ad). "Her istisnaya token" kapalı liste varsayar; envanter kapalı değil.

---

## 3. Ayrım: üç sınıf, üç ayrı cevap

| Sınıf | Ne | Doğru yer | Gerekçe (ölçülmüş) |
|---|---|---|---|
| **A** | Sözlüksel-koşullu **düzenli** değişimler: kaynaşma, ünlü düşmesi, ikizleşme, tek-heceli t/d | **Kural + nitelik** (compiler + roots) | Bunlar kuralın **örneği**, kural değil. `VOICING` 2.635 girişi çözüyor; `kitabı`/`burnu`/`gidiyor` **çalışıyor**. Çökenler eksik **nitelik** (`ağız` `-`, `et` `-`) veya eksik **mekanizma** (ikizleşme hiç yok) veya **D1 skoru** (`aklı`,`gönlü`,`ağacı`,`kanadı`) |
| **B** | **Zekirlik**: `ben→bana`, `sen→sana`, `su→suyun` — gerçekten düzensiz, **kapalı** küme | **Ayrı ad alanı** (ne lemma ne token) | Bugün lemma olarak duruyor ve Ö3'teki hasarı üretiyor. Veri modeli (Ö2) ifade edemiyor ⇒ temsil kararı gerekiyor |
| **C** | **Dizge dışı**: rakam, noktalama, kesme işareti | **Opak token** ✓ **ZATEN YAPILMIŞ** | Ölçüldü (bu oturum, `data/*.bin`): **21 `.bin`'in 9'u noktalama taşıyor**; `anka_a1_pretrain.bin`'de noktalama 32137–45 **%11,021** (11,02 M jeton) · rakam 32146–55 **%9,143** · kesme 32850 %0,006 · 32816–19 `<ENT>/<CAP>` **0**. Yani opak token bu sınıf için **öneri değil, mevcut durum** |

**Özel adlar (D2) bu üçlüde C'ye girmez.** Ölçülmüş olarak başarısız: `<PROPER_NOUN>`
yer tutucusu doğru addan **283× olası**, üretimde külliyatın **4,96 katı**, model
`Erzurum` üretti. Yer tutucu kimliği **korumaz — tutucuya çöker**. Kimlik taşıyan
bir şeyi opak token'a çevirmek, korunmak istenen bilgiyi tam da siler.

> **KENDİ BULGUMUN DÜZELTMESİ (ölçüm sırasında yakalandı).** Bu belgeyi yazarken
> C sınıfı için *"14/14 eğitim `.bin`'inde maks id ≤ 32.136 ⇒ hiç noktalama yok"*
> yazmıştım — **bellekten**. Ölçtüm ve **bayat çıktı**: bugün 21 `.bin` var, **9'u
> noktalama taşıyor**, Anka A1'de **%11,021**. Sayı T-0077 döneminde doğruydu
> (o zaman 14 dosya vardı ve Anka derlemeleri henüz yapılmamıştı); Anka A1
> noktalama ve rakamı **kapsıyor**. Bu, C sınıfının tartışmaya *açık* değil
> *kapalı ve uygulanmış* olduğu anlamına gelir.
> **Ayrıca:** `<PROPER_NOUN>` için verilen *283×* ve *4,96 katı* sayıları T-0079
> raporundandır (bu oturumda yeniden ölçülmedi); Ö1–Ö4 ise bu oturumun kendi
> ölçümüdür. İkisi aynı çerçevede sunulmamalıdır.

> **C sınıfı tartışmaya kapalıdır — çünkü zaten uygulanmıştır.** Noktalama ve rakam
> hem sözlükte token olarak var (32137–32155) hem de Anka A1 külliyatında
> (%11,02 / %9,14). Operatörün "token atamak" sezgisi **bu sınıf için doğru ve
> yürürlükte**. Tartışılacak olan A ve B sınıflarıdır.

### 3.1 C sınıfında ÖLÇÜLMÜŞ BİR KUSUR: kesme sözleşmesi asimetrik

C sınıfı "çözülmüş" demek "doğru çözülmüş" demek değil. Bu turda ölçüldü:

```
c.compile("Ankarada")        -> ["Ankara", "CASE_LOC"]
dec.decompile_sentence(...)  -> "Ankara'da"        ← decompiler KESME KOYUYOR
c.compile("Ankara'da")       -> []                 ← derleyici OKUYAMIYOR
```

**Tur sözleşmesi asimetriktir ve kırılma tam D4'ün (kesme, 17.398 benzersiz yüzey)
yaşadığı yerdedir.** Sonuçları:

- G6'nın (tur özdeşliği) **HATA** dalı bu olguyu saklıyordu: Faz 0 HATA'yı
  ayrıştırmıyordu, tek sayı veriyordu (2.688). `anka_r0e` artık HATA'yı
  **kesme sözleşmesi / geri-boş / diğer** olarak üçe ayırıyor.
- Bu, B sınıfı (ayrı ad alanı) kararını da bağlar: yeni bir ad alanı açmak
  `decompile_sentence()` turunu **daha da** asimetrik yapabilir. S3 sorusu tam
  bunu soruyor.

> Bu kusur **benim ilk G6 pozitif kontrolümün yanlış kurulması sayesinde** çıktı:
> kontrol `Ankarada`yı bekliyordu, araç `Ankara'da` döndürdü ve fail-closed düştü.
> Üçüncü kez aynı sınıf hata (kontrol etiketi ≠ beklenen değer) — ve üçüncü kez
> **kontrolün kendisi bulguyu üretti**.

---

## 4. Karar kuralı (öneri)

> **İstisna "aynı kuralın bir örneği" mi, yoksa "kuralın dışında bir şey" mi?**
> - Örnek ise → sözlüğe girdi **+ nitelik** (kural derleyicide kalır).
> - Kural dışı **ve** dizge dışı ise → opak token.
> - Kimlik taşıyorsa → **token değil**.
> - **Kural dışı ama kimliksiz ve kapalı kümeyse → ayrı ad alanı (B).**
>
> Ve üstüne mimari kısıt:
> **Hiçbir çözüm, çekimli bir yüzeyi derleyicinin KÖK ADAYI olarak görünür
> kılmamalıdır.** Bu kısıt hem B'nin temsilini hem D1'in onarımını belirler.

---

## 5. Bu tartışma D1'i nasıl yeniden çerçeveliyor

Planın Faz 2/D1 maddesi "en uzun kök kazanır" (V1) idi ve **ölçümle çürütüldü**
(`anka_r0d_v1_curutme`, rc=0). V1 çürüdükten sonra D1 "açık tasarım sorusu" olarak
kalmıştı. **Bu ölçüm o soruyu daralttı:**

D1 bir **skorlama** problemi değil, bir **ad alanı (namespace)** problemidir.
`aklı`'nın `ak+DERIV_lI` seçilmesi ile `bunun`'un `bun+POSS` seçilmesi **aynı kök
nedendir**: skorlayıcı, sözlükte duran kısa/türemiş gövdeyi meşru bir kök görüyor
ve beraberliği ona kırıyor. Skor fonksiyonunu değiştirmek semptomu oynatır; ad
alanını ayırmak nedeni kaldırır.

**Ölçülecek hipotez (Faz 2'ye aday):** *sözlükte çekimli/türemiş yüzeyler kök ad
alanından çıkarılırsa hem D1 ihlali hem zekirlik hasarı aynı anda düşer.* Bu,
V1'in aksine **iki bağımsız kusuru tek nedenle** açıklar; V1 ise yalnız 4/4 hedefi
düzeltip 5/5 doğruyu bozuyordu.

---

## 6. Açık sorular (Antigravity'ye gönderildi)

- **S1 İkizleşme:** doğru yer morphotactics koşullu kopyalama mı, `GEMINATION`
  niteliği mi, ayrı istisna tablosu mu? Tetikleyici ek kümesi nedir (POSS_3SG
  yeterli mi; CASE_ACC/DAT de mi)? Sınıf kapalı mı, açık mı?
- **S2 Zekirlik:** `roots.tsv`'e 4. kolon `ALLOMORF=` mi, ayrı istisna tablosu mu,
  derleyicide kapalı harita mı? Kısıt: hangisi "çekimli yüzey kök adayı olmaz"
  garantisini veriyorsa o kazanır.
- **S3 Kaçırılan sınıf var mı?** Ayrı ad alanı `decompile_sentence()` tur
  özdeşliğini (bugün G6 **%97,32**) bozar mı, bozarsa nasıl ölçülür?

Cevap **hipotez** olarak alınacak ve ölçülecek; cevabın hedef damgası doğrulanacak
(16 Eyl kuyruk dersi).

---

## 7. Çürütme maddesi

**Hangi sonuç bu belgedeki hangi cümleyi çürütür?**

| İddia | Onu çürütecek ölçüm |
|---|---|
| "Zekirlik zaten token olarak çözülmüş ve zarar veriyor" | `bana`/`bun`/`on` lemma kaldırıldığında D1 ihlalinin **düşmemesi** |
| "İkizleşme kapalı küme değil" | Türkçe ikizleşen köklerin sayılabilir ve küçük (birkaç düzine) çıkması |
| "D1 skor değil ad alanı problemidir" | Ad alanı ayrıldıktan sonra beraberliklerin **hâlâ** yanlış tarafa kırılması |
| "Özel ad token'lanmamalı" | Opak token'lı bir temsilin yer tutucudan **daha az** çökmesi |
| "C sınıfı zaten çözülmüş" | Noktalama/rakam token'larının külliyattan **çıkarılmasının** bir kapıyı iyileştirmesi |

---

## 8. Durum ve sıra

**Operatör 19 Eyl: sorusunu geri çekti, Antigravity'yi kendisi bilgilendirdi.**
Bu belge **kapanmış tartışmanın kaydı** olarak duruyor; §6'daki danışma cevabı
gelirse **hipotez** olarak ölçülür, görev açılmaz.

- **Hiçbir donmuş dosyaya yazılmadı.** Bu belge yalnız ölçüm + çerçeve.
- **Faz 2 açılmadan S1/S2/S3 cevapları ölçülür.** Özellikle S2'nin temsil kararı
  Faz 3'ün (sözlük büyütme) şemasını belirler ⇒ **Faz 3, S2'ye bağımlı**.
- Zekirlik girdileri (`bana`,`sana`,`suyun`,`bun`,`on`,`ney`) mevcut hâliyle
  bırakılırsa **hiçbir skor değişikliği güvenli değildir** — bu, Faz 2'nin ön
  koşulu olarak kayda geçiyor.

### Ölçüm çıpası

| Ne | Yol | sha256 |
|---|---|---|
| Kanonik sözlük | `data/rebuild/vocab_base_32852.json` | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` |
| Lexicon | `data/lexicon/roots.tsv` | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| V1 çürütme | `data/eval/anka_r0d_v1_curutme_2026-09-19.json` | `badcbf0b707d3153b35a2bb6fc90f42d40ceadd7cc2503fc0cca72ee03f0366b` |
| F0 kapsam | `data/eval/anka_r0_kapsam_2026-09-19.json` | `ed42aafa0647a1c3f4e022bf128ffe6fdef6508bafda63c8638e64cb555304cb` |
