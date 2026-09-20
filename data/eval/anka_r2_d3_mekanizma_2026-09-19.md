# Faz 2 · D3 Mekanizma Onarımı — Önce/Sonra Ölçümü

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Yürütücü:** claude (tek yürütücü)
**Kapsam:** `src/compiler/morphotactics.py` + `src/compiler/phonology.py`
**Hüküm:** **D3 uygulandı ve kapılar geçti — 1.919 yeni yüzey çözülüyor, mevcut
TEK BİR karar bile değişmiyor.** Üç plandan ikisi yazıldığı gibi yapılamadı;
biri (circumflex) **iki eksene ayrılarak** kurtarıldı, biri (ADV_ROOT) bedeli
ölçülmediği için **uygulanmadı**. İkisi de aşağıda beyanlıdır.

> **İki cümle:** (1) D3 **tamamen eklemeli** çıktı: `seçimi değişen = 0`,
> `kaybolan = 0` ⇒ operatörün "D1'e dokunulmaz" kısıtı 9-etiket kapısından
> **daha güçlü** biçimde kanıtlandı. (2) Planın circumflex kuralı (`â`→ön)
> **yanlıştı**; `â` uyum bilgisi taşımıyor (`efkâr`→ön · `kâr`→art · `hâl`→ön,
> üçü de özdeş yapı) ⇒ düzeltme **iki eksene bölünerek** yapıldı.

---

## 1. Operatör kararı ve kapsam sınırı

D1 (kısa kök seçimi / tie-break) **Faz 2'den AYRILDI** (operatör kararı, 19 Eyl):
*"D1'e ve tie-break'e DOKUNULMAZ. Her partiden sonra 9-etiket kapısı koşulur;
düşerse rc≠0 ile durur."*

Bu belge yalnız **D3 mekanizma** işini kapsar: ikizleşme (gemination) · iyelikli
adın araç durumu (`-yla`) · circumflex ünlüler · `ADV_ROOT`. `_score_paths`
(`core.py:169-180`) **değiştirilmedi**; `compile()`/`_find_paths_recursive`
sıralama mantığı **değiştirilmedi**.

---

## 2. İlan edilen kapılar (ölçümden ÖNCE yazıldı, sonra değiştirilmedi)

| Kapı | Eşik / kural | Gerekçe |
|---|---|---|
| **K0** | derleyici 4 poz + 2 neg ile kurulu | T-0078'in ölüm sebebi: `load_from_tsv` atlanması sessizce boş trie |
| **K1** | **9-etiket parmak izi `once` ile BİREBİR** | **operatör kapısı** — D1 dokunulmadı kanıtı |
| **K2** | G1 `once` değeri r2'nin ilan ettiği **%2,3943 ile birebir** | kabın kendisi doğru mu (örnekleyici birebirliği) |
| **K3** | 4 mevcut-doğru vaka korunur | regresyon |
| **kaybolan** | **0** | aşağıda: gerekçesi düzeltildi (§9.3) |
| **D3-B** | in-memory yamalı lexicon ile 6/6… beklenen ≥5/6 | mekanizma veriyle çalışıyor mu |

**Ön kontrol (değişiklikten önce):** donmuş bir dosyaya yazmadan önce her
düzeltmenin **adreslenebilir kazancı** ölçüldü — çözülemeyen 125.141 yüzey
üzerinde: `yla/yle` **2.387** · circumflex **1.796** · gemination-kalıp **113**.

---

## 3. Ölçüm düzeni

`scratch/anka_r2_d3_mekanizma.py <once|sonra>` — fail-closed, artımlı yazan kap.
Örnekleyici r2/r0e'den **birebir** kopyalandı (`MULTILINE|DOTALL` arınma, 3.212
makale, 236.573 benzersiz yüzey). Her yüzey **TEK geçişte** derlenir; seçilen
kök + morfem sayısı + beraberlik + G1 durumu **karar haritasına** yazılır
(`scratch/anka_r2_d3_secim_<etiket>.tsv`). `sonra`, `once` haritasıyla
karşılaştırılır — **`once` haritası yoksa `rc=2` ile durur.**

**Neden harita:** G1 kaba bir özettir ve r2 §9'da **hata ölçüsü olmadığı**
gösterildi (özgüllük %0). "D1 dokunulmadı" iddiası tek bir oranla
kanıtlanamaz; **yüzey yüzey seçim karşılaştırması** gerekir.

---

## 4. Uygulanan düzeltmeler

| # | Yer | Değişiklik | Durum |
|---|---|---|---|
| **D3-A** | `morphotactics.py:162-176` | `n_cases`'e `("CASE_INS", "(y)lA", "-")` eklendi | **uygulandı** |
| **D3-B** | `phonology.py` `mutate_stem` | `GEMINATION` dalı (VOICING'den **sonra**) | **mekanizma uygulandı**, veri Faz 1 |
| **D3-C** | `phonology.py` | circumflex ünlüler `VOWELS`'a; **`HARMONY_VOWELS` ayrı** | **uygulandı (bölünmüş)** |
| **D3-D** | `morphotactics.py` | `ADV_ROOT`'a geçiş | **UYGULANMADI** (§11.3) |

### D3-A — kimlik neden `CASE_INS` (yeni id değil)
Grubun öteki kimlikleri `*_N` (n-tamponu: `nDA`, `nDAn`…). Araç durumunun
tamponu **`y`**'dir ve **yüzeyi çıplak addaki `CASE_INS` ile AYNIDIR**. Yeni bir
id, sözlükte **aynı ek için ikinci bir morfem jetonu** açardı. Saptırma
bilinçli, gerekçesi kodda yazılı.

### D3-B — neden kural değil NİTELİK
İkizleşme ek ile değil **lemma ile** belirlenir (`hak`→`hakkı` ama `ak`→`akı`).
Kural yazılsaydı yanlış lemmaları da ikizlerdi ⇒ veri olarak taşınır. Sıra
önemli: `ret`→**`redd`** için önce VOICING (`t`→`d`) sonra ikizleşme.

---

## 5. ÖNCE (temel) ve SONRA — sayılar

| Ölçüm | ÖNCE | SONRA | Fark |
|---|---|---|---|
| **K1** 9-etiket parmak izi | `bunun:bu:2\|…\|gönlü:gön:2` | **birebir aynı** | **DEĞİŞMEDİ** ✓ |
| **K3** regresyon | TEMİZ | TEMİZ | ✓ |
| çözülen yüzey | 111.432 | **113.351** | **+1.919** |
| çözülemeyen yüzey | 125.141 | 123.222 | −1.919 (tutarlı) |
| G1 ihlal (adet) | 2.668 | 2.696 | **+28** |
| G1 oranı | %2,3943 | **%2,3785** | −0,0158 puan |
| beraberlik sayısı | 34.742 | 34.832 | +90 |
| **kaybolan** | — | **0** | ✓ |
| **seçimi değişen** (kök/morfem) | — | **0** | ✓ |
| beraberlik durumu değişen | — | 12 | yeni aday, kazanan aynı |
| G1 durumu değişen | — | 15 | aynı sınıf |
| ilgisiz etki (yla/circumflex dışı) | — | **0** | ✓ |
| `cozulemeyen` yla/yle | 2.387 | 469 | −1.918 |
| `cozulemeyen` circumflex | 1.796 | 1.788 | −8 |
| `cozulemeyen` gemination-kalıp | 113 | 112 | −1 |

> ⚠️ **G1 oranı düşerken MUTLAK ihlal sayısı 2.668 → 2.696 (+28) ARTTI.**
> Oran, paydayı 1.919 büyüttüğü için düşüyor. Tek başına "oran düştü" cümlesi
> yanıltıcıdır (`[[olcut-kendi-payini-yiyor]]`); **iki sayı birlikte** raporlanır
> ve `G1` zaten hata ölçüsü değildir (r2 §9).

> ⚠️ **FIRSAT satırları BİR BÖLÜNTÜ DEĞİLDİR** — üçü de aynı `cozulemez` kümesi
> üzerinde **bağımsız** yüklemlerdir (`'yla' in w` · `â/î/û in w` · regex).
> Kesişirler (bir yüzey hem `yla` hem circumflex taşıyabilir) ⇒ **toplanamazlar**.
> Toplamın tutmaması (2.387+1.796+113 ≠ 125.141) kusur değil, ölçütün doğasıdır.

### 5.1 En güçlü bulgu: `seçimi değişen = 0`
113.351 çözülen yüzeyin **111.432'si** zaten çözülüyordu ve bunların
**hiçbirinin** seçilen kökü/morfem sayısı değişmedi. ⇒ D3 **yalnız eklemeli**;
D1 kararları **kanıtlanmış biçimde** dokunulmadı. Bu, 9-etiket kapısından
**daha güçlü** bir kanıttır (9 etiket 34.742 beraberliğin küçük bir örneğidir).

Değişen 12 beraberlik + 15 G1 durumu **yeni adayların** ürünüdür (kazanan aynı
kaldı) ⇒ külliyatta **12 yeni belirsiz-pencere** açıldı. Planın
`needs_disambiguation` maddesi için küçük ama **gerçek** bir maliyettir; Faz 4
raporuna yazılır.

---

## 6. D3-B pozitif kontrolü (in-memory yamalı lexicon)

Gerçek veri (`roots_anka_r1.tsv`) Faz 1'de üretilir; **mekanizmanın** çalıştığı
bu turda kanıtlanır — yamalı lexicon **yalnız sondayı** besler, ana sözlük değildir.

| Yüzey | ÖNCE | SONRA | Beklenen |
|---|---|---|---|
| `hakkı` | 0 yol | **`hak`** ✓ | `hak` |
| `hissi` | 0 yol | **`his`** ✓ | `his` |
| `affı` | 0 yol | **`af`** ✓ | `af` |
| `zammı` | 0 yol | **`zam`** ✓ | `zam` |
| `hakkımız` | 0 yol | **`hak`** ✓ | `hak` |
| `hakkında` | `hakkında` | `hakkında` | kendi lemması — **doğru davranış** |

**5/6** ✓. Doğrudan `mutate_stem` sınaması: `hak→hakk` · `his→hiss` · `af→aff` ·
`zam→zamm` · `ret`+`VOICING,GEMINATION`→**`redd`** · `su`+`GEMINATION`→`su`
(ünlüyle biten gövde **ikizlenmez** — yanlış-pozitif kontrolü).

---

## 7. D3-C — circumflex: plan kuralı YANLIŞTI, iki eksene bölündü

### 7.1 Ölçülen olgu: `â` uyum bilgisi TAŞIMAZ
`hâl` ve `kâr` **ikisi de sözlükte** ve **ikisinin de tek ünlüsü `â`** — ama
**zıt** uyum isterler. `efkâr` da aynı sınıfta:

| Kök | Doğru çekim | ESKİ (`â` atlanır) | YENİ (`â`→art) |
|---|---|---|---|
| `efkâr` | `efkâri` (ön; önceki `e`) | ✓ | **✗** |
| `kâr` | `kârı` (art; yedek `a`) | ✓ | ✓ |
| `hâl` | `hâli` (ön) | ✗ | ✗ |

ESKİ **2/3**, YENİ **1/3** ⇒ planın eşlemesi **regresyondu**. Üçü de yapısal
olarak **özdeş** ⇒ harf ayırt edici değil — **D1'in önek/uzatma beraberliğiyle
aynı biçim** (`[[d1-onek-sira-artefakti]]`).

### 7.2 Düzeltme: kümeyi eksene göre ayır
`VOWELS` (düzeltme işaretli) **yalnız tampon/sonluluk** için;
`HARMONY_VOWELS` (işaretsiz) **yalnız uyum** için. Eski uyum **birebir** korunur.

**A/B kanıtı (aynı süreç, tek değişken `VOWELS`)** — tampon ekseni:

| Yüzey | ESKİ | YENİ |
|---|---|---|
| `Balâya` | 0 yol | **1** ✓ |
| `Habeşîye` | 0 yol | **1** ✓ |
| `Habeşînin` | 0 yol | **2** ✓ |
| `Balâ` | 1 | 1 (değişmedi) |

Ve uyum ekseni **geri geldi**: `efkâri`→`efkâr` ✓ · `Efkâri`→`efkâr` ✓ ·
`kârı`→`kâr` ✓.

### 7.3 Adıyla bilinen istisna: `hâl`
`hâli`/`hâline` **0 yol kalıyor** — `â` uyumu atlandığı için yedek `a`→art
veriyor. **Bu bilinçli ve gizlenmemiştir:** istisna **leksiktir** (harften
çıkmaz), `hâl`'e Faz 1'de lemma niteliği verilerek çözülecek. Yanlış tarafa
sessizce yazmak yerine **adlandırıldı**.

---

## 8. Hedef listesi `0/13 → 1/13` — kalem kalem NEDEN

Hedef liste **ölçümden önce** ilan edildi; sonuçları gördükten sonra
**değiştirilmedi**. Yalnız `adıyla` düzeldi; kalan 12 kalemin nedeni **ölçüldü**:

| Yüzey | Neden hâlâ 0 | Sınıf |
|---|---|---|
| `adıyla` | **DÜZELDİ** → `[ad, POSS_3SG, CASE_INS]` | **D3-A** ✓ |
| `aracıyla` | çözülüyor ama kök `ara` seçiliyor (doğrusu `araç`) | **D1** (ertelendi) |
| `hakkıyla` | çözülüyor ama `hakkıyla` **kendi lemması** olarak (kuyruk bloğu) | **Faz 1 kuyruk** |
| `suyuyla` | `su`'nun düzensiz iyesi (`suyu`), `-yla` değil | **yeni kalem** (beyan) |
| `hâli` · `hâline` | `hâl` istisnası (§7.3) | **Faz 1 veri** |
| `hakkı` `hissi` `affı` `zammı` `hakkımız` | `GEMINATION` **niteliği** sözlükte yok | **Faz 1 veri** |
| `hissetti` | çözülüyor (`hisset`, 4 yol) — kök seçimi | **D1** |
| `ağzı` | `ağız`'ın `VOWEL_DROP` niteliği yok | **Faz 1 veri** |

> **Hedef listesinin 1/13'ü D3'ün başarısızlığı DEĞİLDİR.** 13 kalemin
> **7'si Faz 1 verisi**, 2'si D1, 1'i kuyruk bloğu, 1'i ayrı bir düzensizlik.
> D3'ün **gerçek** kazancı bu listeyle ölçülmez — hedef liste **elle seçilmiş**
> ve ağırlıkla veri eksikliği taşıyor. Gerçek kazanç **külliyat üzerinde**
> ölçüldü: **+1.919 yüzey**.

---

## 9. Bu turda ölçülen KENDİ kusurlarım

| # | Kusur | Nasıl çıktı | Etki |
|---|---|---|---|
| 1 | **`â`→art eşlemesi regresyondu** (§7.1) | `kaybolan 1` kapısı: `Efkâri` | Geri alındı, iki eksene bölündü |
| 2 | **Kapının gerekçesi yanlıştı:** *"geçiş eklemek yol SİLMEZ"* | `Efkâri` kaybı | Yalnız **graf** ekseni için doğru; **fonoloji** ekseni önek eşleşmesini değiştirir ⇒ yol silebilir. Gerekçe koda düzeltildi |
| 3 | **İki düzeltmeyi tek kaba koydum** | `kaybolan` graf'tan mı fonolojiden mi belirsizdi | Teşhis `$TMPDIR` sondasıyla ayrıldı; ders: eksen başına ayrı ölçüm |
| 4 | Aynı çıktı yoluna iki `sonra` koşumu yazdım | Reddedilen ilk koşumun JSON'u **üzerine yazıldı** | Kanıt olarak **log kopyaları** `scratch/anka_r2_d3_kosum_*.log` altına alındı; reddin sayıları bu belgede |
| 5 | Ön kontrolü yoktu, sonradan eklendi | İlk `once` koşumu `firsat` bloğu olmadan koştu | Yeniden koşuldu; **ikinci** koşum kanoniktir (harita digest'i **aynı** ⇒ determinist) |
| 6 | `once` ve `sonra` **farklı betik revizyonlarıyla** koştu | `betik_sha256` 8f09e65a… vs 53d9833e… | Karşılaştırılan nicelikler **aynı kod yolundan**; fark yalnız *yorum satırları* + `sonra`'ya özel DIFF bloğu. **Beyan edildi** |
| 7 | JSON anahtar adı ile print etiketi tutarsız (`ilgili_olmayan_*` vs `ilgisiz_etki`) | Kendi sorgum `None` döndü | Yalnız adlandırma; **değer 0** her iki yoldan da doğrulandı |
| 8 | Sondamda yanlış anahtar adı sordum | `ilgisiz_olmayan_sayi: None` | Artefakt kusuru değil; doğru adlar `ilgili_olmayan_{sayi,etki}` |
| 9 | Raporda `ADV_ROOT` terminalini **`:239`** diye gösterdim | Satır **246** (`graph.mark_terminal(State.ADV_ROOT)`) | Atıflar ölçülerek düzeltildi; iddia (0 çıkış) doğruydu, **atıf** yanlıştı |
| 10 | **Sahte plan kusuru yazmaya yakındım:** `grep "def pos_to_state"` boş dönünce *"plan var olmayan simgeye atıf yapıyor"* sonucuna vardım | `pos_to_state` `core.py:35-44`'te **yerel sözlük** (fonksiyon değil) ⇒ planın `CONJ`/`INTERJ` iddiası **DOĞRU** (ikisi de sözlükte yok, satır 44'ün `.get(…, NOUN_ROOT)` varsayılanına düşüyorlar) | Kusur **aramamdaydı**, planda değil. **"Erişemedim ≠ yok"** — bulguyu yazmadan önce **okuyarak** doğrula (`[[iki-sayi-celisiyor-sanma-once-kume]]`) |

---

## 10. Ölçülen PLAN kusurları (Faz 1 tasarımını etkiler)

### 10.1 D2'nin POS kuralı yanlış: milliyet adları çekimlenir
Sözlükte **6.352 `ADJ` lemma**; bunların **3.627'sinin `NOUN` satırı YOK**.
`ADJ_ROOT`'tan çıkan geçişler yalnız `DERIV_lAş` · `DERIV_lAn` · copula ·
`GERUND_KEN` ⇒ **çoğul/iyelik/hâl YOK**. Ölçüldü:

| Yüzey | Sonuç |
|---|---|
| `Türkler` · `Türkün` · `Türke` | ✓ (3 yol — `Türk` **NOUN** satırı var) |
| `Fransızlar` · `İngilizler` · `Fransızın` · `Fransıza` | **0 yol** |
| `Fransız` (lemma) | **sözlükte HİÇ YOK** |
| `İngiliz` `İtalyan` `Alman` `Türk` | **`NOUN(-)`** (çekimlenir) |
| `Amerikan` | **yalnız `ADJ`** (çekimlenemez) |

> Plan *"milliyet sıfatları `ADJ`"* diyor. **Çekimlenen milliyet adları**
> (`Fransızlar`, `İngilizler` — yer tutucu listesinin başı) `ADJ` verilirse
> **kazanç gelmez**. ⇒ D2'de çekimlenen sınıfa **`NOUN`** (gerekirse
> `NOUN`+`ADJ` ikisi) verilmelidir.

### 10.2 ADV sayısı şişik
Plan: *"2.024 ADV lemma çekimlenemez"*. Ölçüldü: **2.024 ADV lemmasının yalnız
1.277'sinin NOUN/ADJ/VERB satırı yok** ⇒ gerçekten çekimsiz olan **1.277**.

### 10.3 D3-D (ADV_ROOT) — UYGULANMADI
`ADV_ROOT` `morphotactics.py:246`'da terminal ve **0 çıkışı var** — yeniden
ölçüldü: `build_default_graph()` ⇒ `ADV_ROOT` için çıkış kenarı **0**,
`terminal=True` (`noun_copula_states` de onu içermiyor). **Ama bedeli ölçülmedi:**
- D3-D sondayı (`elbetteydi`·`belkiydi`·`şimdiyken`·`hızlıca`·`yalnızca`·
  `güzelce`) **6/6 zaten çözülüyor** (4·4·1·2·4·3 yol) — çünkü bu lemmaların
  başka POS satırı var.
- Önceki turda 600/600 veren sondam **kendi artefaktımdı** (üretilen yüzeyler
  kelime değildi; "kök == lemma" sınaması meşru alternatif okumaları reddediyordu).
- Geçiş eklemek **yeni aday** üretir ⇒ beraberlik yapısını oynatır (§5.1'de 12
  yeni belirsiz-pencere). **Ölçülmemiş bir değişiklik yapılmaz.**

⇒ **Açık kalem**, Faz 1'e taşındı. Ölçüm tarifi: çözülemeyen yüzeyler içinde
`find_stems` bir ADV lemma döndürenlerin sayısı (harness'ın `firsat` bloğuna
eklenir).

---

## 11. Çürütme maddesi

| İddia | Onu çürütecek ölçüm |
|---|---|
| "D3 tamamen eklemeli, D1'e dokunmadı" | `secim_degisen > 0` **veya** K1 parmak izinin değişmesi |
| "İki eksen ayrılmalı" | `HARMONY_VOWELS` ayrımı **kaldırıldığında** tampon kazancının ve uyum kaybının **aynı** işaretli çıkması |
| "`hâl` leksik istisnadır, harften çıkmaz" | `hâl`'i `kâr`'dan ayıran bir **harf/konum** kuralı gösterilmesi |
| "D2 milliyet adları `NOUN` olmalı" | `ADJ` verilen milliyet adlarının çoğul/hâl çekimini **kazandığının** gösterilmesi |
| "ADV_ROOT bedeli küçük" | çözülemeyen yüzeylerde ADV-kök kaynaklı payın **büyük** çıkması (§10.3) |
| "Hedef liste 1/13 D3'ün başarısızlığı" | 13 kalemin D3-dışı nedenlerinin **yanlış** olduğunun gösterilmesi |

---

## 12. Tam digest tablosu

| Artefakt | sha256 (tam) |
|---|---|
| `src/compiler/phonology.py` (D3-B + D3-C) | `0ced6097adc47c3581f6ae19756d40102850ddb6e2aca53e105fd29421e85f95` |
| `src/compiler/morphotactics.py` (D3-A) | `1b03902db77533b627558c218d62ee3553582af4b6a4c5d6c271d06c0494d4aa` |
| `scratch/anka_r2_d3_mekanizma.py` | `53d9833e9fbe2642b986e19590c681bcdbf39a6dbb87e30c2042557419407c5a` |
| `data/eval/anka_r2_d3_once_2026-09-19.json` | `413421554ec907ae9a20030f0ef00b1e17a281f5b7a46eaee17ce153b46054af` |
| `data/eval/anka_r2_d3_sonra_2026-09-19.json` | `79a4d9772e5df688b1b76ff85e088cb3562af76b931ac20e4fd97372875ebe0e` |
| `scratch/anka_r2_d3_secim_once.tsv` | `c3630d28cf3eb81e32b7619901020fd12ac503e7b586f755bd514025eba1c5f5` |
| `scratch/anka_r2_d3_secim_sonra.tsv` | `9d826084b8f55154813f52d701586dee80055219e1f58adbfd582ce54ea1e3a8` |
| `data/lexicon/roots.tsv` (donmuş) | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `tests/morphology_regression_100.json` | `cbe00f31c4dcc34ba2baed5317f1bf38b4e69b130c5271205c1dd5709dab18da` |
| `scratch/anka_r2_d3_kosum_d3_sonra_2026-09-19.log` | **reddedilen** ilk `sonra` koşumu (kanıt) |

**Provenance zinciri (doğrulandı):** `sonra` diff'inin okuduğu `once_sha256`
== diskteki `_once.tsv` == `once` JSON'unun kaydettiği harita digest'i
(`c3630d28…c5f5`) ✓. **İki koşumun `betik_sha256`'sı farklıdır** (8f09e65a… vs
53d9833e…) — beyan edildi (§9.6).

**Tablo denetimi (belge yazıldıktan SONRA, ayrı adım):** §12'deki **9 digest'in
dokuzu** `shasum -a 256` ile yeniden ölçüldü ⇒ **9/9 birebir** (önek *ve* kuyruk;
`[[hash-iddialari-tam-digest-ile-denetlenir]]`). §5'teki **her sayı** iki JSON
artefaktından **yeniden okundu** (kendi transkripsiyonuma güvenilmedi) ⇒
**14/14 birebir**: K1 parmak izi · çözülen 111.432/113.351 · G1 ihlal
2.668/2.696 · G1% 2,3943/2,3785 · beraberlik 34.742/34.832 · çözülemeyen
125.141/123.222 · yla 2.387/469 · circumflex 1.796/1.788 · gem 113/112 · diff
(1.919 · 0 · 0 · 12 · 15 · 0) · D3-B 5/6 · D3 hedef 0/13→1/13 · `DURDURULDU`
yok. **İki ölçüm de raporun kendi sayılarını bağımsız kanaldan doğruladı.**

**Test durumu (bu turda koşuldu):** `venv/bin/pytest` → **213 passed, 1 failed**.
Tek başarısızlık `test_agent_gateway.py::…test_gateway_http_server_endpoints`,
nedeni `socket.bind` → `PermissionError: [Errno 1] Operation not permitted` =
**sandbox yasağı**; bu değişikliğin ürünü **değildir** (değişiklik öncesi de
aynıydı). Morfoloji regresyon testleri **geçiyor**.

**Donmuş hiçbir YOLA yazılmadı:** `src/compiler/**` donmuş ama `T-0080`
kiralaması altında (12:08:21Z'ye kadar) ve `writes[]`'te beyanlı;
`data/lexicon/roots.tsv` **hiç değiştirilmedi** (digest sabit). Üretilen her şey
`scratch/` + `data/eval/` altındadır.
