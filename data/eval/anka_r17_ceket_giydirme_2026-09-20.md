# T-0094 — Ceketi Anka'nın derleyicisiyle giydirme (Faz 2 + Faz 3a)

**Görev:** T-0094 · **Yürütücü:** claude · **Damga (UTC):** `2026-09-20T17:05Z`
**İlan (ölçümden ÖNCE yazıldı, sonra değiştirilmedi):**
`data/eval/anka_r17_ceket_kapi_ilani_2026-09-20.md` — ekleri §6b · §6c · §6d · §6e
**Build raporu:** `data/eval/anka_r17_build_2026-09-20.md`

> Bu dosya **artımlı** yazılır (T-0063): her faz bittiğinde bölümü eklenir, sayılar
> ölçümden gelir, kopya taşınmaz. Sonda bölümü (§2) ölçüm sürerken **boş bırakılır**.

---

## 1. Faz 2 — Anka'ya bağlı ölçüm kabı

**Ürün:** `scripts/evaluate_carpenter_anka.py` (yeni dosya; eski
`evaluate_carpenter_generation_100.py` **değiştirilmedi**).

### 1.1 — Eski kabın Anka'ya bağlanamama sebebi (ölçüldü, kopya değil)

| kusur | eski kap | sonuç |
|---|---|---|
| sözlük | `vocab.load("data/vocab.json")` = **31.357** | Anka 33.114 ⇒ kafa uyuşmaz |
| lexicon | `roots.tsv` — `load_from_tsv` atlanırsa her girdi **sessizce boş** döner | ölçüm sessizce ölür |
| model yolu | varsayılan `kristal_b1_5_best.pt` / `kristal_model.pt` | **silinmiş** |
| `--output` | verilmezse donmuş `data/b1_5_splits/` altına yazar | donmuş yüzeye **izinsiz** yazım |
| `<OUTPUT>` id | `vocab.stoi.get("</OUTPUT>", 9)` — **sessiz varsayılan** | id **1705** gerçekte var; varsayılan sessiz yanlış |
| zarf | `:128-132` elle kurulmuş | `render_prompt` kanonik değil |
| çıktı şeması | **model kimliği YOK** | 13 Eyl'deki n=100 sonucu bugün **hiçbir checkpoint'e atfedilemiyor** |

### 1.2 — Yeni kabın kapıları

| # | kapı | davranış |
|---|---|---|
| **G1** | sözlük + lexicon yüklenir | lexicon **trie kök/girdi sayısı** basılır; `0` ise **`rc=2`** |
| **G2** | özel jetonlar | `<BOS>`/`<EOS>`/`<OUTPUT>`/`</OUTPUT>` **sessiz varsayılansız** okunur |
| **G3** | kafa = sözlük | `lm_head.weight.shape[0] != len(stoi)` ⇒ **`rc=2`** |
| **K4** | oracle katmanları | kimlik ROUGE = **1,0** değilse veya distraktör ≥ 0,35 ise ⇒ **`rc=2`** |
| **çözünürlük** | iki eşik de çözülebilir mi | A/B yarı-genişliği eşiği aşarsa ⇒ **`rc=2`** |

> **G1 neden `trie_kok_sayisi`:** `LexiconManager.__init__` `self.root` kurar; **`self.roots`
> YOK**. `len(lex.roots)` **AttributeError** verirdi. Doğrulama `is_word` düğümleri DFS ile
> sayılarak yapılır ve **iki yönlü** sınanır (kök **ve** girdi > 0).

### 1.3 — Kabın ölçtüğü şeyler

| eksen | tanım | eşik |
|---|---|---|
| **A** (unutma) | Wikipedia diliminde **maskesiz** CE, tabana karşı **eşli değişim** | artış **≤ +%10** |
| **B** (unutma) | noktalama bloğu `32137..32145`'te top-1, **eşli** düşüş | düşüş **≤ 5,0 puan** |
| **ceket** | ezber (4-gram) · tutarsızlık · ROUGE-L · soru-içerik kesişimi | `<%10` · `<%5` · `≥0,35` · `≥%80` |

**Kanonik yardımcılar** (kopya yasak): `render_prompt` + `resize_state_dict`
→ `src/llm/prompt_contract.py`; `KristalLM` → `scripts/train_step_demo.py`;
`rouge_l_score` → `scripts/evaluate_b1_5_rigorous.py`; `wilson_ci` →
`scripts/evaluate_mcq_conditioning.py` (**yüzde döner**).

### 1.4 — Kap kendi üzerinde doğrulandı (pozitif kontrol)

**Kimlik eşlemesi** (`--model` = `--baseline` = `data/anka_a1r.pt`):

```
A artış %+0,00 (eşik ≤ +%10,0) · A yarı-genişlik ±0,0984 < pay 0,3535 ✓
B düşüş +0,00 puan (eşik ≤ 5,0) · B yarı-genişlik ±1,95 < eşik ✓
```

Eşli mantık **tam sıfır** veriyor ⇒ kendini doğruluyor. Oracle katmanı (**n=100**,
bu raporun kendi koşumlarından **yeniden üretilebilir**):
**kimlik 1,0000** (tavan tuttu) · **distraktör 0,0848** · **sabit-tahmin 0,0780** —
ikisi de 0,35'in **çok altında** ⇒ ölçüt **AYIRT EDİYOR**. Bu, T-0091'in *"42.459 vakada
0 karar değiştiren kör araç"* kusurunun bu kapta **bulunmadığının** kanıtıdır.

> **DÜZELTME (K11, §5).** Bu bölüm (ve ilan §6e) önce **distraktör 0,1379 · sabit-tahmin
> 0,1576** yazıyordu. **O çift hiçbir saklı ölçüm dosyasında YOKTUR** — `scratch/t0094_sonda/`
> altındaki **10** sonuç JSON'unun **hiçbiri** o değerleri taşımıyor. Oracle sayıları
> **`n`'e bağlıdır** ve `n` o satırda **yazılı değildi**: ölçülen çiftler `n=100` → 0,0848/0,0780 ·
> `n=4` → 0,1061/0,1223. **Hükmün kendisi değişmiyor** (üç değerde de ≪ 0,35), ama sayı
> **ölçülmemiş** olduğu için düzeltildi. Kural: bir oracle sayısı **`n` yazılmadan** kanıt değildir.

### 1.5 — Protokol, ölçüm gücüne göre sıkılaştırıldı (ilan §6d)

İlk tasarım T-0059'un örneklemini kopyalıyordu ve **kendi eşiğini çözemiyordu**:

| eksen | eski | yarı-genişlik | eşik | yeni | yarı-genişlik |
|---|---|---|---|---|---|
| **A** | 64 pencere | ±0,2205 | pay 0,346 | **256 pencere** | **±0,0984** (3,6×) |
| **B** | **250 konum** | **±6,05 puan** | **5,0 puan** | **2.500 konum** | **±1,95 puan** (2,6×) |

**Yan kanıt:** aynı model, aynı dilim, yalnız örneklem büyüyünce B top-1 **%41,04 → %49,56**.
Küçük örneklemin verdiği sayı **artefaktmış**. İki **fail-closed** kapı eklendi: yarı-genişlik
eşiği aşarsa **`rc=2`**, hiçbir hüküm verilmez (T-0075/T-0093 ailesi: *çözemeyeceği eşiği
sınayan kap, kap değildir*).

### 1.6 — A ekseninin MUTLAK değeri ilan §4.1 ile çelişmiyor — farklı **kaptır**

§4.1 `anka_a1r.pt` için **4,0132** ilan etmişti; bu kap **3,5352 ± 0,8032** ölçüyor. Aynı
dosya, **farklı pencere örnekleme protokolü**. Dağılım geniş ⇒ iki sayı aynı dağılımın iki
örneklemidir. **Bağlayıcı sonuç:** A'nın mutlak sayısı protokole bağlıdır, ileride
kıyaslanamaz; **eşiği taşıyan şey EŞLİ FARKTIR** — kap tabanı `--baseline` ile **kendisi**
ölçtüğü için fark protokol-uyumludur.

---

## 2. Faz 3a — Kısa unutma sondası

### 2.1 — Düzen (ilan §5.1'in uygulanışı)

Dört kol, **eşit hesaplama** (1.000 adım × blok 128 × yığın 8 = **1.024.000** jeton-yuvası),
tek tohum (`42`), sabit `--lr 2e-4`, aynı yükleme noktası (`data/anka_a1r.pt`), MPS,
sandbox **dışında**. Tek değişen: replay oranı. Replay karışımları
`scripts/build_replay_mix.py` ile, kaynak `data/train_chat_balanced.bin` (`443f93d3…`).

| kol | replay | veri | toplam blok | ceket blok | replay blok | ölçülen replay payı |
|---|---|---|---|---|---|---|
| **R0** | yok | `data/train_carpenter_specialization_anka.bin` | 6.131 | 6.131 | 0 | %0,0 |
| **R4** | `--replay-every 4` | `scratch/t0094_sonda/mix_r4.bin` | 8.175 | 6.131 | 2.044 | %25,0 |
| **R10** | `--replay-every 10` | `scratch/t0094_sonda/mix_r10.bin` | 6.813 | 6.131 | 682 | %10,0 |
| **R20** | `--replay-every 20` | `scratch/t0094_sonda/mix_r20.bin` | 6.454 | 6.131 | 323 | %5,0 |

> **Ölçülen replay payları beyan edilen oranlarla birebir tutuyor** (`output_stats.
> measured_replay_ratio_pct` = 25,0 / 10,0 / 5,0). Karışım üreticisi kendi oranını
> **ölçüp** yazıyor; beyan ile ölçüm ayrışmıyor.

### 2.2 — Koşumlar: dördü de canlı, taban dokunulmadı

| kol | `rc` | duvar | ilk kayıp | `son60` | arm sha256 (tam) |
|---|---|---|---|---|---|
| **R0** | 0 | 439 s | 3,1376 | 0,8969 ± 0,1583 | `ba313677466108ba0e18dbb851390a0635c592da8a631cf95eb4729532df2120` |
| **R4** | 0 | 502 s | 2,7157 | 1,0191 ± 0,2467 | `f3bd5be6aa71702bbaa60a8622e1be2cf00d0b9c15d4737cedcf667bdca9e574` |
| **R10** | 0 | 519 s | 2,7550 | 0,9763 ± 0,2031 | `1163b918abd6f52ced42b916d8edd085daa835ef912fe37d459cfa9750f125a4` |
| **R20** | 0 | 530 s | 3,0635 | 0,9547 ± 0,1760 | `1d04365518cb067abca89a122913dd1911158196cfb1c9149b7cdba4f675d6fa` |

**Canlılık kapısı GEÇTİ (T-0073 tuzağı YOK):** dört kolda da ilk kayıp **sonlu ve > 0**
(2,72…3,14); hiçbir kolda `0,0000` görülmedi. Bu, `--pretrain`'siz SFT rejiminde maskelemenin
**no-op olmadığının** kanıtıdır: kayıp ~3,1'den ~0,95'e **düşüyor** ⇒ gerçek hedef kaldı.
Adım süresi 0,39–0,53 s/adım (T-0052'nin 3,45 s/adım artefaktı yok — makinede GPU'ya başka
iş bindirilmedi).

> **İlan §3.1'in istediği İKİ değer birlikte raporlanıyor:** ilk kayıp **ilk partideki
> eğitilmemiş modelin kaybıdır** (sondanın kendi tabanı) ve `son60` **düşüş yönünü** verir.
> İlan **sabit sayı bandı ilan etmedi** (gerekçesi: maske rejiminde ölçülmüş bant yok;
> uydurulmuş bant kanıtsız vaat olurdu) — onun yerine **karar kuralı** ilan etti ve kural
> **iki değerle** uygulandı. Burada `0,00` **hiç** görülmedi.

**`data/anka_a1r.pt` DOKUNULMADI:** `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293`
(koşum dizisi öncesi = sonrası). Kollar `data/` altına **hiç** yazmadı; hepsi `scratch/`'te.

> **`son60` KOLLAR ARASI KIYASLANAMAZ.** Her kol **farklı bir karışım** görüyor ⇒ hedef
> dağılımı farklı ⇒ kayıp ölçeği farklı. R0'ın en düşük kaybı ("0,8969") onun **daha iyi**
> olduğunu göstermez; R0 yalnız ceketi görüyor, diğerleri ceket+sohbet görüyor. Bu sayı
> **yalnız canlılık** için okunur (T-0077/T-0078: kayıp bir **dağılım**dır).

### 2.3 — Unutma eksenleri: **hiçbir kol eşiği geçmiyor**

Her kol, **aynı tabana** (`data/anka_a1r.pt`) karşı **eşli** ölçüldü. A ekseni aynı 256
pencereyi, B ekseni aynı 2.500 konumu görür ⇒ fark protokol-uyumludur (§1.6).

| kol | A: taban CE → kol CE | A artış | A yarı | A_gec | B: taban top-1 → kol | B düşüş | B yarı | B_gec | **UYGUN** |
|---|---|---|---|---|---|---|---|---|---|
| **R0** | 3,5352 → 4,8008 | **+%35,80** | ±0,1294 | ✗ | %49,56 → %48,06 | +1,51 p | ±1,95 | ✓ | **hayır** |
| **R4** | 3,5352 → 4,4020 | **+%24,52** | ±0,1152 | ✗ | %49,56 → %49,29 | +0,28 p | ±1,95 | ✓ | **hayır** |
| **R10** | 3,5352 → 4,5858 | **+%29,72** | ±0,1244 | ✗ | %49,56 → %48,06 | +1,51 p | ±1,95 | ✓ | **hayır** |
| **R20** | 3,5352 → 4,6255 | **+%30,84** | ±0,1251 | ✗ | %49,56 → %47,74 | +1,82 p | ±1,95 | ✓ | **hayır** |

Eşikler: A **≤ +%10** · B **≤ 5,0 puan**. **Çözünürlük kapıları geçti** (A yarı-genişlik
0,115–0,129 < pay 0,354; B yarı-genişlik ±1,95 < eşik 5,0) ⇒ sayılar **hüküm verebilir**;
"ölçemedim" değil, "ölçtüm ve **kaldı**".

**İki eksen ayrışıyor:** B (noktalama top-1, **sığ** beceri) dört kolda da **korunuyor**
(en kötü +1,82 puan). A (Wikipedia CE, **derin** dil modeli) dört kolda da **bozuluyor**
(en iyi +%24,52). ⇒ Bağlayıcı kısıt **A eksenidir**; noktalama kaybı bu ölçekte sorun değil.

### 2.4 — HÜKÜM: ilan edilen §5.1 kural 3 ATEŞLEDİ ⇒ **DURULUR**

> İlan §5.1 (ölçümden **önce** yazıldı): *"Bir kol **UYGUN**'dur ancak ve ancak
> **A ≤ +%10** **VE** **B ≥ −5,0 puan**. … **Hiçbir kol uygun değilse ⇒ DURULUR** ve
> operatöre bildirilir. 'Daha iyi bir kol üret' DENMEZ; replay oranı **uydurulmaz**."*

**Hiçbir kol UYGUN değil ⇒ Faz 3b BAŞLATILMADI.** Replay oranı seçilmedi, uydurulmadı.
En iyi kol (R4, +%24,52) bile eşiğin **2,45 katı**. Kural ölçümden önce ilan edildiği ve
yol tuttuğu için hüküm **tasarımdan değil ölçümden** geliyor
([[tasarlayan-kendi-iddiasini-yanlislayabilmeli]]).

### 2.5 — Ve sonda **kolları SIRALAYAMIYOR** — kendi tasarımımın kusuru

R4 (%25) en iyi, R10 (%10) ve R20 (%5) ondan kötü, R0 (replay yok) en kötü. Ama sıra
**doza göre değil**: R10 < R20 (yani %10 replay, %5'ten **kötü**). Doğrusal fit:

```
A% = 30,796 + (−0,0677)·replay%      R² = 0,0162
```

**R² = 0,0162 ⇒ doz, A'daki değişimin %1,6'sını açıklıyor; %98,4'ü başka bir şey.**
Dört nokta arasındaki 11,3 puanlık yay örnekleme gürültüsü **değil** (A yarı-genişliği
±0,13) — yani kollar **gerçekten farklı**, ama fark **dozdan gelmiyor**.

**Sebep ölçüldü — üç değişken birlikte oynuyor, ben yalnız birini niyet ettim.**

Önce **doğru sordum:** karışımların blok düzeni gerçekten farklı mı? Her karışım bloğu
kaynaklarıyla **byte düzeyinde** karşılaştırıldı (blake2b; ceket 6.131 blok · replay 24.375
blok · **kesişim 0**):

| kol | blok | ceket | replay | **sınıflanamayan** | ölçülen desen | atlanan ceket bloğu |
|---|---|---|---|---|---|---|
| **R4** | 8.175 | 6.131 | 2.044 | **0** | `CCCR` × sürekli | **her 4.** |
| **R10** | 6.813 | 6.131 | 682 | **0** | `CCCCCCCCCR` × sürekli | **her 10.** |
| **R20** | 6.454 | 6.131 | 323 | **0** | `C`×19 `R` × sürekli | **her 20.** |

**Serpiştirme RASTGELE DEĞİL — tam periyodik** (ölçülen replay blokları arası mesafe:
R4'te hep **4**, R10'da hep **10**, R20'de hep **20**; sapma **0**). Ve **hiçbir blok
sınıflanamadı** demek: karışımdaki her blok kaynağıyla **bit-özdeş**, üretici veriyi
**yeniden yazmıyor**, yalnız **yerleştiriyor**.

**Bu, çakışmayı tahminden ÖLÇÜME çevirdi:** replay eklemek, ceket bloğunu **silmiyor** —
**her n'inci bloğu atlıyor**. ⇒ Kollar **aynı ceket önekini** (indeks ~0…999) **farklı
alt kümelerle** görüyor. Yani "sıra" değil, **görülen VERİ AKIŞI** değişiyor:

| eksen | R0 | R4 | R10 | R20 | niyet ettim mi |
|---|---|---|---|---|---|
| replay **miktarı** | %0 | %25 | %10 | %5 | **evet** |
| **görülen veri akışı** | kesintisiz 1.000 | her 4. atlanmış 750 | her 10. atlanmış 900 | her 20. atlanmış 950 | **hayır** |
| replay blok **konumu** | — | 3,7,11,15… | 9,19,29… | 19,39,59… | **hayır** |

**Sonuç: bu sonda, replay oranının unutmaya etkisini ölçemez** — çünkü oranı değiştirmek
**her kolda farklı bir jeton dizisi** üretiyor; 1.000 adımlık kısa koşumda **hangi bloğun
görüldüğü** baskın hâle geliyor. Hüküm (§2.4) yine de geçerlidir — kural "hiçbiri
geçmiyorsa dur" diyor ve **hiçbiri geçmiyor**; üstelik "sıralayamıyorum" bulgusu,
*"en yüksek geçen oranı seç"* adımının bu veriyle **zaten yapılamayacağını** gösteriyor.
İki bağımsız gerekçe aynı kapıya çıkıyor.

> **Kendi kusurum (K10) — ve bu kusur ÖLÇÜMLE bulundu, tahminle değil.** Doğru tasarım:
> replay'i **blok yuvası** düzeyinde kurup (atanan yuvalar sabit, yalnız *doldurulan oran*
> değişken) ya da sıra tohumunu **ayrı** bir eksen yapmak. Bu düzeltme **yapılmadı**
> (ölçüm bitti); kusur burada kayıtlıdır.

### 2.6 — Ceket ekseni: **hiçbir kol ceketin bir epoch'unu bile görmedi**

| kol | 1.000 adımda ceket blok | **ceket epoch** | ROUGE-L | ezber | tutarsızlık | kesişim |
|---|---|---|---|---|---|---|
| **TABAN** (ceketi hiç görmedi) | 0 | **0** | **0,0000** | **%0** | **%100** | **%0** |
| **R0** | 1.000 / 6.131 | **0,16** | 0,0969 | %16 | %32 | %0 |
| **R4** | 750 / 6.131 | **0,12** | 0,1058 | %1 | %20 | %0 |
| **R10** | 900 / 6.131 | **0,15** | 0,0997 | %9 | %28 | %0 |
| **R20** | 950 / 6.131 | **0,15** | 0,1075 | %6 | %45 | %0 |
| eşik | — | — | **≥ 0,35** | **<%10** | **<%5** | **≥%80** |

**TABAN satırı ölçüldü** (`eval_TABAN_fix.json`, `device: "mps"`, `model_sha256`
`b93cc1cd…` — yani `data/anka_a1r.pt`'nin **kendisi**, ceketi hiç görmemiş hâli).
Üretimi **`, [?], [?], [?], [?], …`** — saf **yer tutucu tekrarı**; Türkçe yok, önek yok
(5/5 örnekte `teknik çözüm` = 0, `</OUTPUT>` = 0). Bu satır iki iş görür:

1. **Ölçüte gerçek bir sıfır tabanı verir.** Ceket, ROUGE'u **0,0000 → 0,097–0,108**
   taşımıştır; tutarsızlığı **%100 → %20–45** düşürmüştür. Yani *"0,16 epoch"* iddiası artık
   eşiğe bakarak değil, **ölçülmüş bir önce/sonra** ile söyleniyor: ceket **hiç yoktan**
   ~0,10 üretti, eşik **0,35**.
2. **Ölçüm kabının ayırt ettiğini kanıtlar** (T-0061/T-0091 dersi). Eğitilmemiş model
   ROUGE **tam 0,0000** veriyor ⇒ ölçüt tabanı **görebiliyor**; taban da 0,35 çıksaydı
   **ölçüt ölü** olurdu. İki uç (0,0000 ve eşik) arasında **ayrık** bir ölçek var.
   *Tabanı ölçmeyen bir kap, kendi körlüğünü göremez.*

> **VAKUM GEÇİŞ UYARISI — TABAN'ın `unutma_gec: True`'su KANIT DEĞİLDİR.** Taban
> koşumunda `--model` = `--baseline` = `data/anka_a1r.pt`'dir ⇒ A **+%0,00** ve B
> **+0,00 puan** **inşa gereği** çıkar; bu bir başarı değil, **özdeşliğin kendisidir**
> ([[on-kontrol-fail-closed-olmali]], T-0093'ün "VAKUM geçen denetim" yüzü). Taban
> satırının **tek** bilgi taşıyan kolonu `ceket_ekseni` bloğudur. A/B kolonları yalnız
> **pozitif kontrol** olarak okunur: eşli mantığın tam sıfır verdiğini burada da
> doğruluyor (n=100, ±0,0984 yarı-genişlik).

Ceket ekseni **dört kolda da kaldı** (ROUGE 0,097–0,108 « 0,35; kesişim **%0** « %80).
**Ama bu sayı bir hüküm değildir:** model ceketi **altıda bir kez** görmüş. ROUGE 0,10
@ 0,16 epoch, *"ceket öğrenilemez"* demek **değildir** — *"0,16 epoch yetmedi"* demektir;
TABAN satırı bu okumayı **destekliyor** (0,0000'dan 0,10'a çıkmış, yani öğrenme **başlamış**).
Sonda **ceket sorusunu cevaplayamaz**; yalnız **unutma** sorusunu cevaplar (ve orada da
hiçbir kol geçmez).

Tek olumlu iz: ezber oranı R4'te **%1** (R0 %16) — replay, ezberi **düşürüyor**. Ama
tutarsızlık %20'de kalıyor ve kesişim %0 ⇒ üretim **soruyu kullanmıyor**.

**Nitelik kanıt — biçimin YALNIZ ÖNEKİ öğrenilmiş, bitişi öğrenilmemiş.** Aynı held-out
kaydı (cevap: *"Raspa: … esnek çelik plakadır"*), dört kolun üretimi — ve **sıfır noktası**:

| kol | üretim | ne öğrenmiş |
|---|---|---|
| **TABAN** | `, [?], [?], [?], [?], [?], [?], [?], [?], …` | **hiçbir şey** — yer tutucu tekrarı, Türkçe yok |
| **R0** | `usta Cevabı: teknik çözüm: Tanımlanan özellikler doğrudan **ardıç** ağacına aittir.` | **önek doğru**, içerik **şablon** |
| **R4** | `teknik çözüm: teknik çözüm: teknik çözüm: … **kırlangıç** ağacına aittir.` | önek doğru, **döngüye** giriyor |
| **R10** | `usta Cevabı: teknik çözüm: **Ahşabın**` | önek başlıyor, **cümle kurulmuyor** |
| **R20** | `usta Cevabı: teknik çözüm: **zıvana**` | önek başlıyor, **erken kesiliyor** |

Sıfır noktası ile kollar arasındaki fark **nitelik olarak da** açık: TABAN'da **Türkçe
kelime yok**, kollarda cümle kuruluyor. Yani ceket, 0,16 epoch'ta bile **dil üretimini
ceket bağlamına çekmiş**; eksik olan **içerik** ve **sonlandırma**.

**Ölçüm (20 üretimin tamamı, kırpmasız):** `teknik çözüm` öneki **14/20 = %70**;
`</OUTPUT>` bitişi **0/20 = %0**. ⇒ Ceketin **giriş** biçimi öğrenilmiş, **çıkış** biçimi
**hiç** öğrenilmemiş. Bu, tablodaki `tutarsızlık` metriğiyle (%20–45) **tutarlıdır**: model
sonlandırmayı öğrenmediği için döngü/kesme üretiyor. Gövde ise eğitimden **ezberlenmiş bir
şablon** (*"Tanımlanan özellikler doğrudan X ağacına aittir"*), yalnız ağaç adı değişiyor;
sorunun **nesnesi** (raspa/rende/keser) hiç geçmiyor ⇒ **kesişim %0**'ın mekanizması bu.
0,16 epoch'ta **giriş biçimi** öğrenilir, **çıkış biçimi ve içerik** öğrenilmez — bu ayrım,
düşük ROUGE'un *"kap bozuk"* değil *"eğitim kısa"* olduğunu **nitelik olarak** da
doğruluyor (§2.7).

> **K13 — bu paragrafın ilk sürümü YANLIŞTI.** *"Üç kol da `teknik çözüm:` önekini **ve
> `</OUTPUT>` bitişini** üretiyor"* yazmıştım. Bitiş iddiası **ölçülmemişti** ve ölçüm onu
> **çürüttü** (0/20). Önek iddiası da "üç kol" diyordu ama tablo **dört** kol gösteriyor.
> İkisi de düzeltildi. *Bu, K11'in (ölçülmemiş oracle çifti) **aynı sınıftan ikinci**
> örneğidir ve **aynı raporda** çıkmıştır* ⇒ kusur tek seferlik değil, **yazma alışkanlığı**;
> düzeltme de aynı: **iddiayı ölçüme bağla**.

### 2.7 — K7: ölçüm kabının üretim istemi **dağılım dışıydı** (kap kendi kendini yanlışladı)

Ceket ekseninin ilk koşumu **geçersizdi** ve kusur **modelde değil kaptaydı**: `uret()`
istemini `tokenizer.encode(render_prompt(...))` ile kuruyordu; `encode` sona **`<EOS>`
ekliyor** ⇒ model `<BOS>…<OUTPUT> <EOS>` görüyordu. Eğitimde `<EOS>` **her zaman kayıt
sınırıdır** ve `<OUTPUT>`'tan **sonra hiç gelmez** ⇒ son konum dağılım dışı, üretim
**çöküyordu**.

**Ölçülmüş kanıt — aynı model (`arm_R0.pt`), aynı kayıtlar, tek fark sondaki `<EOS>`**
(`eval_R0.json` = düzeltmesiz · `eval_R0_fix.json` = G4 düzeltmeli; ikisi de n=100, tohum 42):

| kayıt | düzeltmesiz koşum (`<BOS>…<OUTPUT> <EOS>`) | düzeltmeli koşum (`<BOS>…<OUTPUT>`) |
|---|---|---|
| idx 1 | `[Özel İsim] [Özel İsim]: POSS_3SG: genel [?] kerestesinde ve suya çok dayanıklıdır; …` | `bu işlem için marangozlukta zımparalama tercih edilir.` |
| idx 2 | `[Özel İsim] [Özel İsim] ( anlam ayrımı: teknik çözüm: teknik çözüm: teknik çözüm: …` | `usta Cevabı: teknik çözüm: Tanımlanan özellikler doğrudan ardıç ağacına aittir.` |
| idx 3 | `[Özel İsim] [Özel İsim], [Özel İsim]: POSS_2SG CASE_GEN merkezine göre daha fazla olmalıdır.` | `teknik çözüm: teknik çözüm: Tanımlanan özellikler doğrudan sedir ağacına aittir.` |

Düzeltmesiz koşumda üretim **yer tutucu ve etiket jetonlarına** çöküyor
(`[Özel İsim]`, `POSS_3SG`, `CASE_GEN`, `[?]`) ve döngüye giriyor. **Düzeltmesiz hüküm
YANLIŞ olurdu:** *"ceket öğrenilmedi"*. Oysa ceket **öğrenilmişti** — aynı checkpoint,
EOS kırpılınca `usta Cevabı: teknik çözüm: …` üretiyor. Kusur **ölçüm kabındaydı**.

Düzeltme: sondaki `<EOS>` **kırpılır** + **iki dalı da** sınayan **G4** kapısı (model
yüklenmeden **önce**). Geçersiz kanıt **silinmedi** — `_fix` ekiyle ayrı ad alanına yazıldı
(T-0092: kanıt ezilmez).

> **K14 — §2.7'nin ilk sürümünde ÖLÇÜLMEMİŞ bir örnek vardı.** Tabloya
> `<BOS>…<OUTPUT>` → *"`teknik çözüm : … ait COPULA_AORIST . </OUTPUT>`"* yazmıştım.
> **Bu dizge T-0094'ün hiçbir artefaktında yoktur** — `COPULA_AORIST` T-0059/T-0062/T-0078
> gibi **eski ceket** koşumlarının artefaktlarında geçiyor; yani örnek **başka bir
> modelden/koşumdan** gelmiş ve **bu koşumun kanıtı gibi** sunulmuştu. Gerçek ölçüm
> yukarıdaki tablodur. *Bu, K11 ve K13 ile **aynı sınıftan ÜÇÜNCÜ** vakadır — hepsi aynı
> raporda.* İlk ikisinde kusur **ölçülmemiş** bir sayıydı; bu kez **başka bir koşumdan
> taşınmış** bir örnekti. Ortak düzeltme aynı: **her cümlenin kaynağını artefakta bağla.**
> ([[denetim-kapsami-iddiadan-dar]], [[iki-sayi-celisiyor-sanma-once-kume]])

**Düzeltme unutma eksenlerine DOKUNMADI — bit düzeyinde kanıt:**

| kol | A (önce → sonra) | B (önce → sonra) |
|---|---|---|
| R0 | %+35,8001 → **%+35,8001** | +1,5067 → **+1,5067** |
| R4 | %+24,5194 → **%+24,5194** | +0,2775 → **+0,2775** |
| R10 | %+29,7195 → **%+29,7195** | +1,5067 → **+1,5067** |
| R20 | %+30,8421 → **%+30,8421** | +1,8239 → **+1,8239** |

Sekiz sayının **sekizi de özdeş** ⇒ A/B eksenleri ham pencerelerle çalıştığı için
etkilenmedi; yalnız **üretim** ekseni bozuktu. *"Bir düzeltmeden sonra her ekseni ayrı oku"*
([[ayni-harf-iki-eksende-zit-calisir]]).

### 2.8 — K8: ilanın epoch sayıları ölçümle çürüdü (§6 düzeltmesi)

Bu bölümün §6'sı *"R0: **1,30** epoch ceket; R4: **0,98**"* diyordu. **Ölçüm: R0 0,16,
R4 0,12** — yaklaşık **8 kat** sapma. İlginç olan: **oran doğruydu** (1,30/0,98 = 1,33;
1.000/750 = 1,33), yalnız **ölçek** yanlıştı. Yani sayı, ceketin gerçek blok sayısından
(6.131) değil **başka bir varsayımdan** türetilmişti.

> **T-0067 kuralı uygulandı:** ölçümden sonra kusur **tasarıma geri düzeltilmedi**;
> sapma **rapora yazıldı**. §6'daki sayı **düzeltilmiş hâliyle** bırakılmadı, burada
> **çürütüldü** olarak işaretlendi.

---

## 3. Faz 3b — Tam giydirme koşumu

**BAŞLATILMADI.** İlan §5.1 kural 3 ateşledi (§2.4): hiçbir kol `A ≤ +%10` **ve**
`B ≥ −5,0 puan` koşulunu birlikte sağlamadı. Kural ölçümden **önce** yazıldı; replay oranı
**seçilmedi, uydurulmadı**. `data/anka_a1r_ceket.pt` ve
`data/train_carpenter_specialization_anka_replay.bin` **üretilmedi** (beyan edilen
donmuş hedefler **yazılmadı**).

Karar **operatöründür**; bu rapor seçenekleri **ölçülmüş hâlleriyle** sunar, öneri
üretmez (§2.5 ayrıca gösteriyor ki bu sonda **oranları sıralayamaz**).

### 3.1 — Operatörün önündeki ölçülmüş tablo (öneri DEĞİL)

Karar için gereken üç sayı **ölçülmüştür**; hiçbiri türetilmemiştir:

| eksen | ölçülen | eşik | kaynak |
|---|---|---|---|
| ceket ROUGE-L @ ≤0,16 epoch | **0,097–0,108** | ≥ 0,35 | §2.6 (4 kol) |
| ceket ROUGE-L @ **0 epoch** (TABAN) | **0,0000** | ≥ 0,35 | §2.6, `eval_TABAN_fix.json` |
| unutma A (genel CE) @ ≤0,16 epoch | **+%24,5 … +%35,8** | ≤ +%10 | §2.3 (4 kol) |
| unutma B (noktalama) @ ≤0,16 epoch | **+0,28 … +1,82 puan** | ≤ 5,0 puan | §2.3 (4 kol) |

**Bu tablonun söylediği:** ceket **öğrenme başlatmış** (0,0000 → 0,10) ama **hem ceket
eşiğine hem unutma eşiğine aynı anda** ulaşan bir kol **yok**. A ekseni (derin Wikipedia CE)
**dört kolda da** bozuluyor; B ekseni (sığ noktalama) **dört kolda da** duruyor.

**Yapılmayanlar (beyan):** Sonda `--steps 1000` ile **sabit** tutuldu ⇒ daha uzun bir koşumun
unutmayı **küçültüp küçültmediği ölçülmedi**; replay oranları **sıralanamadı** (§2.5);
ceketin **kaç epoch** gerektirdiği **ölçülmedi**. Bu üç soru **açıktır** ve bu rapor
onları **cevaplamaz**.

---

## 4. Bu görevde ölçülen kapı boşlukları (beyan edilir, kapatılmaz)

**B1 — `data/*.bin` kardeş `.bin.meta.json`'u KAPSAMIYOR.** Ölçüldü: `fnmatch`
`data/X.bin.meta.json` ↔ `data/*.bin` = **False**; `data/` altındaki **20** meta'nın
**hiçbiri** 10 desenden hiçbirine uymuyor. Meta, `.bin`'in **tazelik çıpasıdır**
(`bin_sha256`, `sozluk_sha256`, `jeton`) ve bayat-`.bin` çapraz kontrolü onu **okur** ⇒
donmuş liste kendi denetim yüzeyinin yarısını açıkta bırakır.

**B2 — `scripts/build_replay_mix.py` donmuş-yol denetimi ÇAĞIRMIYOR.** `grep`: `frozen`
**0 eşleşme**; oysa varsayılan `--output`'u `data/train_f4_replay_mix.bin` — **donmuş
`data/*.bin` deseninin içinde**. Betik `data/` altına **kapısız** yazabilir. Sonda bu yüzden
konteynerini `scratch/`'e aldı; Faz 3b karışımı `writes[]`te **beyan edilmiş** olduğu için
SPEC Kural 2 karşılanır (beyan + `data/` dizin kirası) — **boşluk yine de beyan edilir**.

**B3 — `$TMPDIR` sandbox sınırında değişiyor** (içeride `/tmp/claude-501`, dışarıda
`/var/folders/…`). Sonda konteyneri bu yüzden `scratch/t0094_sonda/` (iki dünyada **aynı
mutlak yol**). Kapı kendi kendini yakaladı: ilk sandbox-dışı koşum
`RuntimeError: Parent directory … does not exist` ile **durdu**.

**B4 — denetimimin POZİTİF KONTROLÜ ateşledi** (kendi kusurumu buldu). `scratch/t0094_donmus_denetim.py`
ilk koşumda `rc=2` verdi: *"POZİTİF KONTROL DÜŞTÜ"*. Sebep **dünyada değil, ölçütümdeydi** —
`.bin.meta.json`'u donmuş saymıştım; **uymuyor**. Kusur benim beklenti kümemdeydi; kapı onu
rapor edilmeden **önce** yakaladı. Düzeltildi; ikinci koşum **TEMIZ**
(taranan **1.438** · donmuş eşleşen **69** · beyanlı **1** · ihlal **0**).

**B5 — kap SESSİZCE CPU'ya düşüyordu; yakalandı ve KAPATILDI.** İlk satır
`device = "mps" if torch.backends.mps.is_available() else "cpu"` idi. Sandbox'ta MPS
**görünmez** (ölçüldü: `built=True / avail=False`) ⇒ koşum **sessizce** CPU'ya düşer;
**uyarı yok**, yalnız 10× yavaşlama. Sayılar da MPS koşumlarıyla **kıyaslanamaz** hâle
gelir (üretim RNG'si cihaza bağlı). **Nasıl yakalandı:** taban ölçümünü sandbox içinde
başlattım; kollar ~100 s sürerken bu koşum **13 dakikada bitmedi** ⇒ süre uyuşmazlığı
kapıyı ele verdi. **Düzeltme:** açık `--device` argümanı + **G5 fail-closed** kapısı
(MPS istenip yoksa **`rc=2`**) + `device` çıktı şemasına yazıldı. **İki dal da sınandı:**
`--device mps` sandbox içinde **`rc=2`**, `--device cpu` açıkça verilince **uyarıp geçti**.

> **Bu boşluk bir ölçümü kurtardı:** sandbox içindeki o koşum bitseydi, taban ceket ekseni
> **CPU sayılarıyla** raporlanacak ve MPS ile alınmış kol sayılarıyla **yanlışlıkla**
> kıyaslanacaktı. Uyuşmazlık **sessiz** olurdu ([[on-kontrol-fail-closed-olmali]],
> [[sandbox-hides-mps-device]]).

---

## 5. Kendi kusurlarım (bu fazlarda)

**K4 — "0 bulgu" ile "0 tarama" tuzağına **düştüm**, pozitif kontrol kurtardı.** İlk
beklenti kümem yanlıştı ve denetim bunu *"denetim KÖR"* diye bağırarak durdurdu (B4).
Kayıtlı kural T-0093/K3'ün aynısı: sayı "temiz" görünse de **taranan sayısı** basılmadan
denetim kanıt değildir. Bu kez kural **işe yaradı**.

**K5 — Faz 2 ilk protokolü kendi eşiğini çözemiyordu.** T-0059'un `n=250`'sini kopyaladım;
B ekseninde yarı-genişlik **±6,05** > eşik **5,0**. Ölçüt "ilan edilmiş ama sınanamaz"
durumdaydı. Düzeltme **ölçüme başlamadan** yapıldı ve ilan §6d'ye yazıldı.

**K6 — `$TMPDIR`'i iki dünyada aynı sandım** (B3). Kayıtlı
`kopyalanan-betik-kabini-degistirir` dersinin tekrarı. Fark: bu kez koşum **kendi kendine
durdu** (eksik dizin `RuntimeError`) ⇒ sessiz bozulma olmadı.

**K7 — ÖLÇÜM KABIM YANLIŞ HÜKÜM VERİYORDU** (§2.7, bu görevin **en pahalı** kalemi).
Üretim istemine `encode`'un eklediği `<EOS>` **kırpılmıyordu** ⇒ istem dağılım dışı ⇒
üretim çöküyor ⇒ *"ceket öğrenilmedi"* hükmü. Oysa ceket **öğrenilmişti**. Kapı **G4**
eklendi (iki dal). Ders: **yardımcı fonksiyon sınır jetonu ekliyorsa çağıran kırpmalı**.

**K9 — kendi kabımda SESSİZ cihaz düşüşü bıraktım** (B5). `"mps" if available else "cpu"`
satırı fail-**open**'dı; sandbox'ta uyarısız CPU'ya düşüyordu. **Yakalanma yolu dikkate
değer:** sayı **yanlış çıkmadı**, koşum **bitmedi** — 100 s yerine 13 dakika. Yani kusuru
**süre** ele verdi, **değer** değil. Kapı **G5** ile fail-closed yapıldı; `device` çıktı
şemasına yazıldı ([[sandbox-hides-mps-device]]).

**K10 — sonda tasarımımda replay oranı ile sıra/ceket-dozu **ayrıştırılmadı** (§2.5).
R² = 0,0162 ⇒ eksen boşta kaldı. Doğru tasarım: `replay_every`'yi sabit tutup **sıra
tohumunu** değiştiren bir koldizisi, ya da karışımı **blok yuvası** düzeyinde kurup
sırayı sabitlemek. Bu düzeltme **yapılmadı** (ölçüm bitti); kusur burada kayıtlıdır.

**K11 — raporuma ÖLÇÜLMEMİŞ bir oracle çifti yazmıştım.** §1.4 ve ilan §6e
*"distraktör 0,1379 · sabit-tahmin 0,1576"* diyordu; **saklı 10 sonuç JSON'unun hiçbiri**
bu değerleri taşımıyor. Sebep: oracle sayıları **`n`'e bağlı** ve o satırda `n` **yazılı
değildi** (`n=100` → 0,0848/0,0780 · `n=4` → 0,1061/0,1223). **Hüküm değişmedi**
(üç değerde de ≪ 0,35) ama **sayı ölçülmemişti** ⇒ düzeltildi (§1.4) ve ilan §6f'ye
düzeltme eki yazıldı. Kayıtlı dersin tekrarı: *ölçülmemiş satıra "PASS" yazma*
([[ara-artefakt-bayatligi-ve-kendini-dogrulayan-assert]]); ek kural: **bir oracle sayısı
`n` yazılmadan kanıt değildir** ([[denetim-kapsami-iddiadan-dar]]).

**K13 — AYNI SINIFTAN İKİNCİ KEZ: raporuma ölçülmemiş bir NİTELİK iddiası yazdım.**
§2.6'da *"Üç kol da `teknik çözüm:` önekini **ve `</OUTPUT>` bitişini** üretiyor"*
demiştim. Ölçüm (20 üretim, kırpmasız): önek **14/20 = %70**, bitiş **0/20 = %0** ⇒
**bitiş iddiası ÇÜRÜDÜ**; ayrıca "üç kol" derken tablo **dört** kol gösteriyordu.
Düzeltildi (§2.6). **Neden önemli:** K11 ile **aynı raporda**, **aynı sınıfta** (ölçülmemiş
iddia) ikinci kez düştüm ⇒ bu bir **yazma alışkanlığı**, tek seferlik dalgınlık değil.
**İç tutarsızlık sinyali gözden kaçtı:** tablodaki `tutarsızlık` metriği (%20–45, yani
sonlandırma başarısız) zaten "bitiş öğrenilmedi" diyordu; iddiam **kendi tablomla
çelişiyordu** ve bunu ancak dış denetim yakaladı. Kural: **nitelik iddiası da sayı gibi
ölçülür**; üretim örnekleri zaten artefaktta duruyor, okumak yeterliydi.
([[olcut-kendi-payini-yiyor]], [[denetim-kapsami-iddiadan-dar]])

**K14 — ÜÇÜNCÜ KEZ, bu kez BAŞKA BİR KOŞUMDAN örnek taşıdım.** §2.7'nin ilk sürümünde
K7'nin iki dalını *"`teknik çözüm : … ait COPULA_AORIST . </OUTPUT>`"* dizgesiyle
gösteriyordum. **Bu dize T-0094'ün hiçbir artefaktında yoktur**; `COPULA_AORIST`
T-0059/T-0062/T-0078 gibi **eski ceket** koşumlarında geçer ⇒ örnek **eski bir modelden**
alınmış, **bu koşumun kanıtı gibi** sunulmuştu. Yerine gerçek **eşleştirilmiş** ölçüm
kondu (`eval_R0.json` ↔ `eval_R0_fix.json`, aynı checkpoint, aynı kayıtlar, n=100, tohum 42).
**K11 · K13 · K14 — üçü de aynı raporda, aynı sınıfta.** İlk ikisi *ölçülmemiş sayı*,
bu *başka koşumdan taşınmış örnek*. Kök neden ortak: **cümleyi artefakta bağlamadan yazmak**;
çare de ortak: **her iddiayı kaynağıyla birlikte kur** ([[iki-sayi-celisiyor-sanma-once-kume]],
[[ozet-bayat-olabilir-kaynagi-oku]]).

---

## 6. Sınırlar

* Faz 3a bir **sonda**dır: 1.000 adım, tek tohum, tek replay ekseni. **Genel marangozluk
  yetkinliği** hükmü bu ölçekle verilemez.
* `--steps 1000` sabit tutulduğu için kollar **eşit hesaplama** görür, **eşit epoch**
  görmez. ~~(R0: 1,30 epoch ceket; R4: 0,98)~~ → **ÇÜRÜTÜLDÜ (K8, §2.8): ölçülen
  R0 0,16 · R4 0,12.** Oran doğruydu (1,33), **ölçek ~8 kat yanlıştı**. Sayı ölçümden
  sonra **tasarıma geri düzeltilmedi**; sapma §2.8'de kayıtlıdır (T-0067).
* **Sonda ceket sorusunu CEVAPLAYAMAZ** (§2.6): hiçbir kol ceketin bir epoch'unu bile
  görmedi (≤0,16). ROUGE 0,10 @ 0,16 epoch, *"ceket öğrenilemez"* **değildir**.
* **Sonda replay oranlarını SIRALAYAMAZ** (§2.5): doz, A eksenindeki değişimin **%1,6**'sını
  açıklıyor (R² = 0,0162); karışımın **sırası** ve **ceket maruziyeti** de eksenle birlikte
  oynuyor. *"En yüksek geçen oran"* seçimi bu veriyle yapılamaz — §2.4 hükmünden **bağımsız**
  ikinci gerekçe.
* Unutmanın **adım sayısıyla** nasıl ölçeklendiği **ölçülmedi** (tek bütçe: 1.000 adım).

---

## 7. Kapanış kanıtları

### 7.0 — Bu görevin ürettiği dosyalar (TAM digest)

| dosya | sha256 (tam) |
|---|---|
| `data/eval/anka_r17_ceket_kapi_ilani_2026-09-20.md` (Faz 0, ilan) | `1f32c72b3058483a047b68a0d19917b9500dc1cb579846026359f7ae4903e40d` |
| `data/train_carpenter_specialization_anka.bin` (yeni ceket, **donmuş desene uyar**) | `8cd0696d90ef3816613cdc1c452580a6e9ab381085462e1d3f3e0e19e4f9c25a` |
| `scripts/evaluate_carpenter_anka.py` (Anka'ya bağlı ölçüm kabı) | `0f017773bb015ef79f7fac02f51b2a14ed9d27a213fef72574ce16357c37188b` |
| `data/eval/anka_r17_ceket_giydirme_2026-09-20.json` (rapor ikizi) | `9cab49100698a75b310884c36146babb4e36362f54f8eeefba073b54afe0bdef` |

> **Bu raporun KENDİ digest'i bilerek YAZILMADI.** Bir dosya, kendi digest'ini içine
> yazdığı anda **kendi kendini bayatlatır** (yazmak dosyayı değiştirir ⇒ satırdaki değer
> yanlış olur). Okur, digest'i **kendisi** hesaplar:
> `shasum -a 256 data/eval/anka_r17_ceket_giydirme_2026-09-20.md`
> Yukarıdaki diğer dört satır bu raporun **dışındadır**, dolayısıyla **stabildir**.
> ([[kabul-kosusu-olctugu-artefakti-degistirir]] — *artefakt DONMADAN digest yazma*.)

### 7.1 — Birim testler: taban **DEĞİŞMEDİ**

```
venv/bin/pytest -q  →  1 failed, 237 passed in 141.57s
FAILED tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints
       PermissionError: [Errno 1] Operation not permitted
```

**238 test toplandı**; düşen **tek** test ilan edilenin **aynısı** ve sebebi
**sandbox soket yasağı** (T-0093'ten beri bilinen, sandbox dışında geçen hâl). Bu görev
`tests/` altına **dokunmadı** ⇒ sayı tabanla **birebir**.

**Kanarya ön-kontrolü:** pytest'ten önce **tüm** `scratch/*.py` dosyaları `py_compile` ile
derlendi (ölçüldü: **160 dosya, 160 OK, 0 bozuk**) — bozuk bir `scratch/` betiği kanaryayı
düşürüp test sayısını **yanlış biçimde** değiştirebilirdi ([[scratch-dosyasi-test-yuzeyinde]]).

### 7.2 — Donmuş yüzey denetimi: **TEMIZ**

```
[sekıl]       frozen.json sozluk · patterns LISTE · len=10 ✓
[tarama]      taranan=1.464 · donmus desene eslesen=69
[beyanli]     data/train_carpenter_specialization_anka.bin  (data/*.bin)   ← pozitif kontrol TUTTU
[ihlal]       beyan EDILMEMIS donmus yazim: 0
              ✓ YOK (beklenen): data/anka_a1r_ceket.pt
              ✓ YOK (beklenen): data/train_carpenter_specialization_anka_replay.bin
rc=0
```

`tarama == 0 ⇒ rc=2` kapısı **aktif**; pozitif kontrol (beyanlı donmuş yazım) **ateşledi**
⇒ denetim **kör değil** (T-0093/K3 dersi).

> **Denetime bu turda eklenen:** §3'ün *"Faz 3b başlatılmadı ⇒ üretilmedi"* cümlesi
> **düzyazıydı** — kanıt değildi. Denetime `YAZILMAYAN_HEDEFLER` assert'i eklendi: bu iki
> beyanlı hedefin **yokluğu ölçülür**, varsa `rc=2` ile **DURULUR**. Sonuç **2/2 YOK**.
> *Yokluk da bir ölçümdür; "yapmadım" cümlesi ancak ölçülürse kanıt olur.*
> ([[denetim-kapsami-iddiadan-dar]], [[on-kontrol-fail-closed-olmali]])

### 7.3 — KORUNAN artefaktlar: dokunulmadı

| artefakt | sha256 (tam) | durum |
|---|---|---|
| `data/anka_a1r.pt` | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | **dokunulmadı** |
| `data/train_carpenter_specialization_anka.bin` | `8cd0696d90ef3816613cdc1c452580a6e9ab381085462e1d3f3e0e19e4f9c25a` | **yazıldı** (beyanlı; §1) |

Sonda kolları `data/` altına **hiç** yazmadı — dördü de `scratch/t0094_sonda/` içinde.

### 7.4 — Kendi raporumun sayıları denetlendi (K11 sınıfına karşı)

Raporda geçen **5 tam digest** betikle yeniden hesaplandı: **5/5 eşleşti**, eşleşmeyen **0**.
Ayrıca oracle çiftinin **ölçülmemiş** olduğu bu denetimde ortaya çıktı ve düzeltildi (§1.4,
§5/K11, ilan §6f). *Bir sayıyı rapora yazmak, o sayıyı ölçmüş olmak değildir.*

**§2.3 tablosunun tamamı** `eval_ozet_fix.jsonl`'e karşı **sayısal** olarak yeniden okundu:
4 kol × 8 sayı = **32 değer**, **32/32 örtüştü**, uyuşmazlık **0** (A: taban CE, kol CE, artış,
yarı-genişlik · B: taban top-1, kol top-1, düşüş, yarı-genişlik). *Bu denetim, kendi denetçimin
ilk sürümünde **yanlış negatif** verdi* — dizge karşılaştırması `+%35,80` ile `+35,80`'i
eşleştirmiyordu; `%` işareti yüzünden 4 satır "uyuşmuyor" göründü. **Kusur denetçimdeydi,
raporda değil**; sayısal karşılaştırmaya geçilince hepsi tuttu. Ders: *bir uyuşmazlık
görünce önce ölçütün kendisini doğrula* — aksi hâlde olmayan bir kusuru rapora yazardım.

### 7.5 — Ölçüm ortamı (beyan)

| kalem | değer |
|---|---|
| cihaz | **`mps`** — `--device mps` **açıkça**, G5 kapısı **fail-closed** (§4/B5) |
| sandbox | eğitim ve değerlendirme **sandbox DIŞINDA** (MPS sandbox'ta görünmez) |
| adım süresi | 0,39–0,53 s/adım (T-0052: makinede GPU'ya başka iş bindirilmedi) |
| konteyner | `scratch/t0094_sonda/` (iki dünyada **aynı mutlak yol**; §4/B3) |
| **kayıtlı cihaz** | **yalnız `eval_TABAN_fix.json`** → `"device": "mps"` · `"device_istenen": "mps"` |

> **Taban koşumu G5'ten SONRA yapıldı ve cihazı KAYITLI:** `eval_TABAN_fix.json` içinde
> `device` alanı **vardır** ve `mps` der. Ayrıca `model_sha256` alanı tam digest taşır
> (`b93cc1cd54093fc6…`), yani **ölçülen model kimliği kanıtlıdır**. TABAN satırının (§2.6)
> sayıları bu dosyadan okunmuştur ⇒ *taban iddiası kanıtlanmış, kol iddiaları çıkarılmıştır.*

> **Kanıt sınırı (beyan):** dört **kol** değerlendirmesi, `device` alanı çıktı şemasına
> **eklenmeden ÖNCE** koştu ⇒ `eval_R*_fix.json` dosyalarında `device` **YOKTUR**; o
> koşumların cihazı **artefakttan doğrulanamaz**. MPS olduğu **ortamdan** bilinir
> (sandbox dışı + adım süresi) ama **kayıtlı değildir** — bu yüzden "kanıtlandı" değil
> **"ortamdan çıkarıldı"** diye yazılır. `eval_TABAN_fix.json` ise `device: "mps"` alanını
> **taşır** (G5 sonrası koştu). *Bir alanı sonradan eklemek, ondan önceki koşumlara
> geriye dönük kanıt kazandırmaz.*
* `src/llm/frozen_guard.py:59` mutlak-yolda **FAIL-OPEN** açığı **kapatılmadı** (T-0092'den
  beri açık, operatör kararı bekliyor). Kap ona **güvenmez**.
* D1 (en-uzun-kök) ve D4 (kesme/rakam) **kapsam dışı**.
