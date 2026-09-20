# Faz 2 · İ/I Normalizasyonu Onarımı — Ölçüm ve Sevk Kanıtı

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Yürütücü:** claude (tek yürütücü)
**Kapsam:** `src/compiler/lexicon.py` + `src/compiler/core.py` (yalnız `_turkish_lower`)
**Hüküm:** **Kusur gerçek, ölçüldü ve YALNIZ-EKLEMELİ olarak sevk edildi.** Örneklemde
**+145 çözülen yüzey, 0 kayıp**; 236.573 yüzeyin **hiçbiri** çözülemez olmadı. Bedel
**1 yüzey** + **24 yeni beraberlik penceresi**; ikisi de aşağıda adıyla beyanlıdır.
Operatörün ilan ettiği **9-etiket kapısı geçti**.

> **İki cümle:** (1) `compile` aramayı Türkçe-normalize ederken (`İ`→`i`), `load_from_tsv`
> trie anahtarını düz `.lower()` ile yazıyordu; `'İ'.lower()` **2 karakter** (`i`+U+0307)
> verdiği için **sözlükte VAR olan 79 satır ölüydü** — `İstanbul`, `İngiliz`, `İtalyan`,
> `İspanyol` dâhil. (2) Düzeltme **kapalı bir kusuru açtı, açık bir kusuru kapatmadı**:
> anahtar farkı **+61 / −0**, anahtar sözlüğünden bağımsız olarak karar haritasında da
> `kaybolan = 0`.

---

## 1. Ölçülen kusur (mekanizma)

| Yer | Ne yapıyor |
|---|---|
| `core.py` `compile()` | girdiyi `_turkish_lower` ile normalize eder: `İ`→`i`, `I`→`ı` |
| `lexicon.py` `load_from_tsv()` | trie **anahtarını** düz Python `.lower()` ile yazar |

İkisi **tam olarak `İ` ve `I`'da** ayrışır:

```
'İ'.lower()          == 'i' + U+0307 (COMBINING DOT ABOVE)   → 2 karakter
_turkish_lower('İ')  == 'i'                                   → 1 karakter
'I'.lower()          == 'i'          ·  _turkish_lower('I') == 'ı'
```

⇒ trie'de `i̇stanbul` (2 karakterli `i`) anahtarı var, arama `istanbul` istiyor ⇒
`find_stems` `i` düğümünün altında `n` bulamayıp **ilk karakterde `break`** eder.
Sonuç: **`İ`/`I` ile başlayan her lemma erişilemez — sözlükte VAR olsa bile.**

**Ölçüm (sözlük `data/lexicon/roots.tsv`, sha256 `fe3005e5…`):** İ/I-başlangıçlı
**79 satır** (72 `İ` + 7 `I`); **79'u da sıralı ÇEKİRDEKTE** (kuyruk bloğunda 0) ⇒
küratörlü, gerçek özel adlar. Ölü oldukları hâlde:

| Yüzey | Düzeltmeden ÖNCE | SONRA |
|---|---|---|
| `İstanbul` | **0 yol** | `['İstanbul']` |
| `İstanbulda` | 0 yol | `['İstanbul','CASE_LOC']` |
| `İngilizler` | 0 yol | `['İngiliz','PLURAL']` (3 yol) |
| `İtalyanlar` · `İspanyollar` | 0 yol | 3 yol |
| `Isparta` · `Iğdır` | 0 yol | 1 yol |

---

## 2. İlan edilen kapılar (ölçümden ÖNCE yazıldı, sonuçtan sonra DEĞİŞTİRİLMEDİ)

| Kapı | Kural | Rol |
|---|---|---|
| **K0** | derleyici 4 poz + 2 neg ile kurulu, **iki kolda** | T-0078'in ölüm sebebi: `load_from_tsv` atlanırsa sessiz boş trie |
| **İ** | 11 hedef yüzey A→B kazanır | dar sonda |
| **K1** | **9-etiket parmak izi iki kolda AYNI** | **OPERATÖR KAPISI** ("D1'e ve tie-break'e dokunulmaz") |
| **A-sadakat** | A kolunun karar haritası DİSKTEKİ durumun digest'ine **birebir** | örnekleyici+lexicon+graf byte sadık mı |
| **kaybolan** | **0** | yalnız ekleme beklenir |
| **G1** | diskteki `load_from_tsv` == ölçülen B kolu digest'i | **(onay kabında, ölçümden sonra ilan edildi)** sevk edilen = ölçülen |
| **G2** | düz `.lower()` elle kurgu == bozuk A kolu digest'i | **kap AYIRT EDİCİ mi** — G1'in negatif kontrolü |

> **Dürüstlük notu:** **G1/G2 ölçümden SONRA, sevk adımından ÖNCE** ilan edildi.
> İlk beş kapı (K0/İ/K1/A-sadakat/kaybolan) ölçüm betiğinin başlığına **önceden** yazıldı;
> G1/G2 ölçülen B kolunu **diskteki kaynağa bağlama** ihtiyacından doğdu. Sıra burada
> açıkça yazılıdır, "baştan ilan edilmişti" denmez.

---

## 3. Ölçüm düzeni — İKİ ayrı kap

| Kap | Ne yapar | Neden ayrı |
|---|---|---|
| `scratch/anka_r5_i_norm.py` | düzeltmeyi **bellekte** kurar (lexicon elle doldurulur), A/B karşılaştırır | düzeltmenin **etkisini** ölçer |
| `scratch/anka_r5b_i_norm_dogrula.py` | **diskteki gerçek kaynağı** (`load_from_tsv`) çalıştırır | sevk edilen kaynağın **ölçülenle aynı** olduğunu kanıtlar |

`anka_r5b`'nin G2 kolu, `anka_r5_i_norm`'un A kolunu **birebir** yeniden kurar (aynı 3 alan,
aynı düz `.lower()` anahtarı) ⇒ iki kap birbirinin kontrolüdür.

**A-kol sadakat kanıtı:** A kolunun karar haritası DİSKTEKİ (D3 uygulanmış) durumun
haritasıyla **birebir** (`9d826084…a3a8`) çıktı ⇒ örnekleyici + lexicon + graf **byte
sadık**; B koluyla fark **yalnız** İ-normalizasyonudur.

---

## 4. ÖNCE / SONRA — 236.573 yüzeylik örneklem (3.212 makale)

| Ölçüm | A (mevcut) | B (düzeltilmiş) | Fark |
|---|---|---|---|
| **K1** 9-etiket parmak izi | `bunun:bu:2\|…\|gönlü:gön:2` | **birebir aynı** | **DEĞİŞMEDİ** ✓ |
| K3 regresyon | TEMİZ | TEMİZ | ✓ |
| çözülen yüzey | 113.351 | **113.496** | **+145** |
| **kaybolan yüzey** | — | **0** | ✓ |
| G1 ihlal (adet) | 2.696 | **2.700** | **+4** |
| G1 oranı | %2,3785 | **%2,3789** | +0,0004 puan |
| beraberlik sayısı | 34.832 | 34.893 | +61 |
| **seçimi değişen** (kök/morfem) | — | **16** | aşağıda kalem kalem |
| beraberlik durumu değişen | — | 24 | **hepsi 0→1**, kayıp yok |
| G1 durumu değişen | — | 4 | hepsi 0→1 |

> ⚠️ **G1 oranı ve mutlak sayı AYNI yöne gitmedi** (oran +0,0004, mutlak +4) — payda da
> büyüdü. Tek başına "oran arttı" cümlesi yanıltıcıdır; **iki sayı birlikte** raporlanır
> (`[[olcut-kendi-payini-yiyor]]`). `G1` zaten hata ölçüsü **değildir** (r2 §9: özgüllük %0).

---

## 5. Oynayan 16 karar — kalem kalem, YÖNÜYLE

Tam liste, JSON'un sakladığı ilk-15 örnekten **değil**, diskteki iki haritadan
(`anka_r5_i_norm_A.tsv` / `_B.tsv`) bağımsız olarak yeniden türetildi.

| # | Yüzey | A (kök:morfem) | B (kök:morfem) | Sınıf |
|---|---|---|---|---|
| 1 | `IR` | `ır`:1 | **`Ir`**:1 | onarım |
| 2 | `Ir` | `ır`:1 | **`Ir`**:1 | onarım |
| 3 | `ır` | `ır`:1 | **`Ir`**:1 | onarım |
| 4 | `Ilgın` | `ılgın`:1 | **`Ilgın`**:1 | onarım |
| 5 | `Ilıca` | `ılıca`:1 | **`Ilıca`**:1 | onarım |
| 6 | `ılıca` | `ılıca`:1 | **`Ilıca`**:1 | onarım |
| 7 | `Ilıcalı` | `ılıca`:2 | **`Ilıca`**:2 | onarım |
| 8 | `ılıcası` | `ılıca`:2 | **`Ilıca`**:2 | onarım |
| 9 | `İsa` | `isa`:1 | **`İsa`**:1 | onarım |
| 10 | `İSAM` | `isa`:2 | **`İsa`**:2 | onarım |
| 11 | `İsam` | `isa`:2 | **`İsa`**:2 | onarım |
| 12 | `İncili` | `In`:3 | **`İncil`**:2 | onarım |
| 13 | `İğdir` | `iğ`:2 | **`İğdir`**:1 | onarım |
| 14 | `İkizler` | `ikiz`:2 | **`İkizler`**:1 | onarım |
| 15 | `ının` | `ını`:2 | **`In`**:2 | onarım (kuyruk-çöp → çekirdek lemma) |
| 16 | `ikizler` | `ikiz`:2 | `İkizler`:1 | **MALİYET** |

**15 onarım · 1 maliyet.** 1–8: `Ilıca`/`Ir` gibi **çekirdek** (satır 773-776)
gerçek özel adlar artık seçiliyor. 9–13: `İsa`/`İncil`/`İğdir` (çekirdek, satır 49.250-49.320).
15: A'nın kazananı `ını` **kuyruk bloğunda** (satır 51.948) çöp bir satırdı; B onun
yerine **çekirdekteki** `In`'i seçiyor.

### 5.1 Tek maliyet: `ikizler` — ve neden YENİ bir kusur sınıfı DEĞİL

`ikizler` (küçük harf, "ikizler") A'da `ikiz`+`PLURAL` (2 morfem); B'de tek lemma
`İkizler` (1 morfem). Nedeni D1'in bilinen skorudur (`10.0 - len(morphemes)`) — tek
lemmalık okuma çok-morfemli okumayı yener. **D1'e dokunulmadı; bu davranış zaten vardı.**
`İkizler` lemması erişilebilir hâle gelince o davranış İ için de işliyor.

**Pozitif kontrol (İ-DIŞI ⇒ düzeltmeden ETKİLENMEZ, tabanın kendi davranışı):**

| Yüzey | Kök | Morfem |
|---|---|---|
| `akrep` | **`Akrep`** | 1 |
| `terazi` | **`Terazi`** | 1 |
| `aslan` | **`Aslan`** | 1 |
| `kova` | **`Kova`** | 1 |

Taban, büyük-harfli lemmayı **zaten** küçük-harf yüzeye tercih ediyor. ⇒ Düzeltme
İ lemmalarını sözlüğün geri kalanıyla **pariteye** getiriyor; `ikizler` bu paritenin
**bedeli**dir, onarımın yarattığı yeni bir yara değil. Sayı: **1.832 büyük-harfli lemma**
düz `.lower()` anahtarı alıyor (tabanın **mevcut** kuralı) — bu kural D2/D3 kapsamında
**değiştirilmedi**.

---

## 6. Anahtar muhasebesi — iddia koddan doğrulandı

Üç bağımsız kanaldan aynı sayı:

| Ölçüm | Kanal | Sonuç |
|---|---|---|
| satır → benzersiz lemma | `roots.tsv` okuma | 79 satır → **70** benzersiz (9 tekrar: `İbranice`,`İngilizce`,`İskandinavca`,`İskitçe`,`İskoçça`,`İslami`,`İspanyolca`,`İsveççe`,`İtalyanca`) |
| **anahtar farkı** | iki trie'nin benzersiz anahtar kümeleri | diskteki **49.696** − eski **49.635** = **+61 / −0** |
| mutabakat | 70 − 9 = 61 | ✓ **tam** |
| yüzey kazancı | karar haritası diff'i | **+145 / −0** |

⇒ 79 = 70 + 9 (tekrar) ve 70 = 61 + 9 (ilki zaten yazmıştı). **Anahtar ekseni ile
karar ekseni bağımsız olarak kapalı ve kayıp iki eksende de sıfır.**

> **Kendi kusurum (§9.4):** ilk enstrümantasyonum "yeni anahtar"ı **mutlak** saydı
> (49.696 ≈ bütün dosya) ve düzeltmeye atfetmedi; ayrıca İ/I filtresi yanlış soruyu
> sordu. Doğru ölçüm **iki kümenin farkı**dır. Rakam yukarıdaki düzeltilmiş ölçümdür.

---

## 7. Jeton-ağırlıklı etki — yüzey sayısı kararı vermez

+145 **yüzey** sayısı küçük görünür; ama karar jeton ağırlığıyla verilir. r0'ın
(`data/eval/anka_r0_kapsam_2026-09-19.json`) sıralı `<PROPER_NOUN>` listesinde:

| | ilk-60 PN jetonu |
|---|---|
| toplam | 46.924 |
| İ/I-başlangıçlı kalem | **13 kalem · 16.109 jeton = %34,33** |
| çöp ayıklanınca (`I` 1.531 · `III` 618) | **%29,75** |

Listenin **1., 3., 5. ve 7.** sırası (`İngiliz` 5.270 · `İtalyan` 2.875 ·
`İspanyol` 1.293 · `İstanbul` 979) A kolunda **tamamen ölüydü** ⇒ yer tutucuya düşüyordu.

> ⚠️ **İki sayı iki ayrı ölçüm tabanından geliyor ve karıştırılmamalıdır**
> (`[[iki-sayi-celisiyor-sanma-once-kume]]`): **+145 yüzey** 236.573 yüzeylik
> **Wikipedia örnekleminden**; **%34,33** r0'ın **A1 külliyatı** jeton sıralı
> ilk-60 PN listesinden. İkincisi bir **üst sınır değil**, ilk-60 dilimidir.

---

## 8. Sevk edilen düzeltme

```
# lexicon.py — YALNIZ EKLEME
lower_lemma = lemma.lower()
if lower_lemma != lemma:            # tabanın MEVCUT kuralı, KORUNUR
    self._insert(lower_lemma, row)
tr_lemma = turkish_lower(lemma)     # compile'ın ARADIĞI anahtar
if tr_lemma != lemma and tr_lemma != lower_lemma:
    self._insert(tr_lemma, row)
```

**Tek kaynak:** `turkish_lower` `lexicon.py`'de tanımlı; `core.py:_turkish_lower` ona
**delege eder** (`core.py` bu modülü import eder, ters yön döngü olurdu). Projenin
kendi emsali: T-0057'de maskeleme "tek kaynağa indirildi".

**Neden "değiştirme" değil "ekleme":** ilk denemem mevcut düz anahtarı **sildi** ve
`kaybolan 1 = ['ir']` çıktı — `Ir` lemmasının `ir` anahtarı erişilebilirdi, silinince
yüzey düştü. Düzeltmenin işi **erişim açmak**, kapatmak değil.

---

## 9. Bu turda ölçülen KENDİ kusurlarım

| # | Kusur | Nasıl çıktı | Etki |
|---|---|---|---|
| 1 | Düzeltmeyi **"değiştirme"** biçiminde yazdım | `kaybolan 1 = ['ir']` kapısı | Geri alındı; yalnız-ekleme. **Kapı bunu yakaladı** |
| 2 | Örnekleyiciyi kopyalarken **girinti kaybı** (`break` 8→4 boşluk) | dosya 0 tamamen okundu, `kabul=344.403`, `eksik=['tr-00001']` | Kesin-değer kapısı (1.688/1.524 · 236.573) eklendi |
| 3 | A koluna **yanlış digest** sabitledim (D3 `once` yerine `sonra` gerekirken) | A-sadakat kapısı ateşledi | Doğrusu diskteki durumdur; sabit düzeltildi, gerekçe koda yazıldı |
| 4 | Enstrümantasyonum **yanlış niceliği** ölçtü (mutlak "yeni anahtar" ≈ 49.696) ve düzeltmeye atfetmedi | kendi sorgum tutarsız çıktı | Doğru ölçüm iki kümenin **farkı**; §6 rakamı düzeltilmiş olandır |
| 5 | BSD `sed` `\b` desteklemiyor ⇒ `s/ONCE_HARITA\b/…/g` **sessizce hiçbir şey yapmadı** | hata dalında `NameError` kalacaktı, gerçek mesajı maskelerdi | Python ile değiştirildi; `kalan: 0` + `py_compile` ile doğrulandı |
| 6 | Dar sondanın parmak izi listeleri iç içe döngü yüzünden **iki kez** ekleniyordu | yazdırılan dize şişkin | `tb==tf` karşılaştırması geçerli kaldı; **dize alıntılanmamalı** |

---

## 10. Çürütme maddesi

| İddia | Onu çürütecek ölçüm | Sonuç |
|---|---|---|
| "Düzeltme yalnız eklemeli" | anahtar farkında **−** çıkması veya karar haritasında `kaybolan > 0` | **tutmadı**: +61/−0 · +145/−0 ✓ |
| "Sevk edilen kaynak ölçüleni üretir" | G1 digest eşitliğinin düşmesi | **tutmadı**: `91b3d6af…` birebir ✓ |
| "Kap ayırt edicidir" (G1 boş bir kap değil) | G2'nin A digest'ini **tutmaması** | **tutmadı**: `9d826084…` birebir ✓ |
| "İ lemmaları gerçekten ölüydü" | `İstanbul`'un düzeltmeden ÖNCE yol vermesi | **tutmadı**: A kolunda 0 yol ✓ |
| "`ikizler` maliyeti YENİ bir kusur sınıfıdır" | İ-dışı bir büyük-harf lemmada **aynı** gölgelemenin bulunmaması | **tutmadı**: `akrep`→`Akrep` · `kova`→`Kova` tabanda zaten var ✓ |
| "D1'e dokunulmadı" | K1 9-etiketin değişmesi | **tutmadı**: iki kolda birebir (operatör kapısı) ✓ |
| **"9-etiket kapısı yeterli bir D1 kanıtıdır"** | 9 etiketin **dışında** karar oynaması | **ÇÜRÜDÜ**: `seçimi değişen = 16`. Kapı **geçti** ama **zayıftır**; asıl kanıt 34.832 pencerelik karar haritasıdır |

---

## 11. Tam digest tablosu

| Artefakt | sha256 (tam) |
|---|---|
| `src/compiler/lexicon.py` (sevk edilen düzeltme) | `9e2986f870204634b4c8fd17dac1c6bb1020a9aed5631b4cad6bf6258e1f51e7` |
| `src/compiler/core.py` (`_turkish_lower` delegasyonu) | `abbc78433c1b5fc35498c05d727bd461cf3936b06ad777c8d84338a81f3acfa4` |
| `src/compiler/phonology.py` (D3-B/D3-C, değişmedi) | `0ced6097adc47c3581f6ae19756d40102850ddb6e2aca53e105fd29421e85f95` |
| `src/compiler/morphotactics.py` (D3-A, değişmedi) | `1b03902db77533b627558c218d62ee3553582af4b6a4c5d6c271d06c0494d4aa` |
| `data/lexicon/roots.tsv` (DONMUŞ — **hiç değiştirilmedi**) | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `scratch/anka_r5_i_norm.py` (ölçüm kabı) | `4efb919413d5b10f023300b86b7b0a256ea739079c85aff9c6e12f5baa14cd9f` |
| `scratch/anka_r5b_i_norm_dogrula.py` (onay kabı) | `0b89806c2cc9f37c965cf3bb5f5344a7e9c1e8f3b637267b78db3a23a506b247` |
| `data/eval/anka_r5_i_norm_2026-09-19.json` | `fa08a91628bbf52fd78a1b44acaac964f7c6607afc65b0385b6bd2cdb02aea6f` |
| `data/eval/anka_r5b_i_norm_dogrulama_2026-09-19.json` | `187b45f7da792f12cd0d6b4d38da10b8d8294d2d76114646967db19ac275d0b9` |
| `scratch/anka_r5_i_norm_A.tsv` (bozuk kol) | `9d826084b8f55154813f52d701586dee80055219e1f58adbfd582ce54ea1e3a8` |
| `scratch/anka_r5_i_norm_B.tsv` (düzeltilmiş kol) | `91b3d6afefcd3dc53a14869d671e079c0b3869e6b8457e9178470f45e5f6bd31` |
| `scratch/anka_r5b_diskteki.tsv` (= B, birebir) | `91b3d6afefcd3dc53a14869d671e079c0b3869e6b8457e9178470f45e5f6bd31` |
| `scratch/anka_r5b_elle_duz.tsv` (= A, birebir) | `9d826084b8f55154813f52d701586dee80055219e1f58adbfd582ce54ea1e3a8` |

**Provenance zinciri:** `anka_r5b`'nin beklediği B digest'i (`91b3d6af…`) == diskteki
`anka_r5b_diskteki.tsv` == `anka_r5_i_norm_B.tsv` ✓ · beklediği A digest'i
(`9d826084…`) == `anka_r5b_elle_duz.tsv` == `anka_r5_i_norm_A.tsv` ✓ ·
`roots.tsv` iki kapta da aynı sabitle (`fe3005e5…`) ✓ · iki JSON'da `DURDURULDU` **yok**.

**Tablo denetimi (belge yazıldıktan SONRA, ayrı adım):** §11'deki **13 digest'in
13'ü** `shasum -a 256` ile yeniden ölçüldü ⇒ **13/13 birebir** (önek *ve* kuyruk;
`[[hash-iddialari-tam-digest-ile-denetlenir]]`). §4–§7'deki **20 niceliğin 20'si**
iki JSON + iki TSV'den **yeniden okundu** (kendi transkripsiyonuma güvenilmedi) ⇒
**20/20 birebir**: A/B çözülen · A/B G1 adet ve oran · A/B beraberlik · diff
(145·0·16·24·4) · İ hedef 10/11 · A-sadakat · K1 · 16 karar (haritadan yeniden
türetildi) · 24 beraberliğin **hepsi** 0→1 · 4 G1'in **hepsi** 0→1 · G1/G2/G3/G7 ·
iki JSON'da `DURDURULDU` yok · örneklem 3.212/236.573. **İki bağımsız kanal raporun
kendi sayılarını doğruladı.**

**Test durumu:** `venv/bin/pytest` → **213 passed, 1 failed**. Tek başarısızlık
`test_agent_gateway.py::…test_gateway_http_server_endpoints`, nedeni `socket.bind` →
`PermissionError: [Errno 1] Operation not permitted` = **sandbox yasağı**; bu
değişikliğin ürünü değildir (değişiklik öncesi de aynıydı). Morfoloji regresyon testleri
**geçiyor**.

---

## 12. Faz 1'e devredilen kararlar

1. **D2'nin İ ile başlayan kazançları bu düzeltme olmadan BOŞA GİDERDİ.**
   `İstanbul`/`İngiliz`/`İtalyan`/`İspanyol` sözlükte **zaten var** ve artık erişilebilir;
   D2'nin ekleyeceği İ-başlangıçlı adlar da aynı yoldan geçiyor.
2. **Kuyruk bloğu sınırı ölçüldü: satır 50.475** (52.373 veri satırının ≈%96'sı çekirdek).
   `ını` (satır 51.948) bu blokta ve bir kararın kazananıydı ⇒ tasfiye ölçütü kanıtlı.
3. **Büyük-harfli lemma gölgelemesi (1.832 lemma)** D2/D3'te değiştirilmedi; *karar
   verilmiş bir kusur değil*, ölçülmüş bir tasarım özelliğidir ve `ikizler` sınıfı
   buradan doğar. Ayrık ele alınırsa D1/tie-break tartışmasına girer ⇒ **dokunulmadı**.
4. **`İsveçli` hâlâ 0 yol** (11 hedeften tek düşen): `İsveç` ülke lemması sözlükte yok
   ⇒ türetme (`İsveç`+`DERIV_lI`) kurulamıyor. Bu bir **D2 veri** kalemidir, İ
   normalizasyonu değil — iki kolda da 0 yol olması bunu kanıtlar.

**Donmuş hiçbir YOLA izinsiz yazılmadı:** `src/compiler/**` donmuş ama T-0080 kiralaması
(12:08:21Z'ye kadar, `src/compiler/` dizin kiralaması) altında ve `writes[]`'te beyanlı;
`data/lexicon/roots.tsv` **hiç değiştirilmedi** (digest sabit). Üretilen her şey
`scratch/` + `data/eval/` altındadır. **`git add -A` kullanılmadı.**
