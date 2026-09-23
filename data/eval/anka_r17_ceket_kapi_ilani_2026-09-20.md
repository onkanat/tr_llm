# T-0094 · Faz 0 — KAPI İLANI (ölçümden ve `.bin` yazımından ÖNCE)

**Görev:** T-0094 · **Yürütücü:** claude · **Damga (UTC):** `2026-09-20T14:05Z` ·
**`claimed_at`:** T-0094

> **Bu dosya ölçümden ÖNCE yazılır ve sonra DEĞİŞTİRİLMEZ.**
> İlan edilen ölçümler, eşikler, çürütme maddeleri ve tasarım sayıları burada sabitlenir;
> ölçüm sonrası bir kusur bulunursa **tasarıma düzeltilmez, rapora yazılır** (T-0067).

---

## 0. Operatör kararları (bu oturumda, AskUserQuestion ile — yetkili)

| # | karar |
|---|---|
| 1 | *"Yeniden derle, sonra eğit"* — ceket Anka'nın derleyicisiyle yeniden derlenir |
| 2 | *"Anka'ya bağlı yeni test kabı yaz"* |
| 3 | *"SFT — `--pretrain`'i ÇIKAR"* |
| 4 | *"Önce kısa unutma sondası"* |

---

## 1. Girdi digest tablosu (TAM digest — önek DEĞİL)

Ölçüldü, `sha256` tam 64 karakter:

| dosya | bayt | sha256 |
|---|---|---|
| `data/rebuild/vocab_anka_r1_33114.json` | 747.458 | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| `data/lexicon/roots_anka_r1.tsv` | 873.755 | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| `data/lexicon/roots.tsv` (donmuş, taban) | 870.426 | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/pedagogy/carpenter_specialization_dataset.jsonl` | 2.001.426 | `146a73dce0bf95f11ad9647510c8b500ecabb2f69308e39f597148e2660946fd` |
| `data/pedagogy_canonical/carpenter_canonical.jsonl` | 2.018.336 | `d0d9703970d8d41eda64f8a0b8f3064d3c45177bdf77250bab7c71e02fc38ef0` |
| `data/train_chat_balanced.bin` (replay kaynağı) | 6.240.000 | `443f93d352b07defca91b2a620f2e4154c3ed0b14f7c695535c959209e7608fa` |
| `data/train_carpenter_specialization.bin` (eski ceket — KORUNUR) | 1.735.300 | `13dd81bf700690b78ca15246a2fc1582ee6f8fea587c6d4e78f8ab1e1017b109` |
| `data/anka_a1r.pt` (giydirme tabanı — KORUNUR) | 373.735.137 | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `data/anka_a1.pt` (KORUNUR) | 372.124.278 | `f32d492d9361c668bc86e4e4b0f99f5c473af552486837452475f8b0aa3eb5b4` |
| `data/anka_a1r_pretrain.bin` (KORUNUR) | 200.000.000 | `9f9875762518829dec8f5baa41bd012ee1abf72fb630e24840bafedf691d4f22` |

---

## 2. İlan edilen ölçümler (V1–V6)

Kapılar **yazılan `.bin`'den** ölçülür (biriktiriciye güvenilmez — `anka_r4_build.py` deseni).

| # | ölçüm | ne sorar |
|---|---|---|
| **V1** | sözlük kimliği | yeni ceket metasındaki `sozluk_sha256` = `f9940a8d…` mi |
| **V2** | **önek invaryantı** | id 0…32851 yeni sözlükte jeton-joton aynı mı (fark **0** olmalı) |
| **V3** | `literal_entity_mode` etkisi | yeni cekette `<ENT>`/`<CAP>`/`<ALL_CAPS>` **0** mı |
| **V4** | temsil yakınsaması | yeni `literal/PN` eski **8.428,5**'e karşı ne oldu |
| **V5** | ölçek korunumu | jeton sayısı eski **867.650**'ye karşı ne oldu |
| **V6** | muhasebe özdeşliği | `kept` kimliği tutuyor mu; `json_errors=encode_errors=short_skipped=0` mı |

### 2.1 — İLAN EDİLEN ÇÜRÜTME MADDELERİ (ölçümden önce yazıldı, sonra değiştirilmez)

1. *"V2'de fark **> 0** çıkarsa ⇒ önek invaryantı kırıldı, ceket Anka'ya giydirilemez;
   'kısmen uyar' yazılmaz."*
2. *"V3'te `<ENT>`/`<CAP>` **sıfırlanmazsa** ⇒ `literal_entity_mode` beklenen yerde devreye
   girmiyor; hüküm `AYIRT EDEMEDİ`."*
3. *"V4'te yeni `literal/PN` **eskisinden büyükse** ⇒ yeniden derleme temsili yaklaştırmadı
   **uzaklaştırdı**; sonraki adım giydirme DEĞİL, sebep araştırmasıdır."*
4. *"V6'da `kept` kimliği tutmazsa ⇒ kaynak okuma yolu değişmiş; ölçüm kabı kusurlu,
   `.bin` geçersiz."*

### 2.2 — Fail-closed kontrol kümeleri: **ÖLÇÜLDÜ** (plandaki ifade DÜZELTİLDİ)

Planda kontrol kümeleri *"Ankara/İstanbul literal değil, `<PROPER_NOUN>`"* diye taslak
yazılmıştı. **Ölçüm bu ifadeyi yanlışladı** ve düzeltme burada, **uygulamadan önce** beyan
edilir. `literal_entity_mode=False` + `roots_anka_r1.tsv` + sözlük 33.114 ile ölçüldü:

| sözcük | `stoi`'da mı | id | `<PROPER_NOUN>` | `<ENT>` | `<UNK>` | çözüm |
|---|---|---|---|---|---|---|
| `Ankara` | **VAR** | 31511 | 0 | 0 | 0 | `[<BOS>, 31511, <EOS>]` |
| `İstanbul` | **VAR** | 33102 | 0 | 0 | 0 | `[<BOS>, 33102, <EOS>]` |
| `Fransa` | **VAR** | 32907 | 0 | 0 | 0 | `[<BOS>, 32907, <EOS>]` |
| `İngiliz` | **VAR** | 33087 | 0 | 0 | 0 | `[<BOS>, 33087, <EOS>]` |
| `Ankara'da` | yok | — | 0 | **0** | 0 | `[<BOS>, 31511, 29, <EOS>]` (kesme dalı ATLANDI) |
| `Ankara'nın` | yok | — | 0 | **0** | 0 | `[<BOS>, 31511, 56, <EOS>]` |
| `Zyxwvukent` | yok | — | **1** | **0** | 0 | `[<BOS>, 4, <EOS>]` |

**Doğru kontrol kümeleri (ilan edilen):**

* **Negatif küme** = `{Ankara, İstanbul, Fransa, İngiliz}` — sözlükte **var** ⇒ **kendi
  jetonunu** basar; `PN=0` **ve** `ENT=0` beklenir.
* **Pozitif küme** = `{Zyxwvukent}` — sözlükte yok, derlenemez, büyük harf ⇒ `PN=1`,
  `ENT=0` beklenir.
* **Kesme kümesi** = `{Ankara'da, Ankara'nın}` — kesme dalı `literal_entity_mode` ile
  kapılı olduğu için **atlanır**; çözüm morfem hattından gelir (`31511 + ek`), `ENT=0`.

**Kapı:** üç kümeden biri beklentiyi tutmazsa **`rc=2` ile DURULUR**. Bu, T-0078'in
*"`load_from_tsv` atlanınca her girdi sessizce boş döner"* ölüm sebebine karşı kurulmuştur.

**Ölçülen özel id'ler:** `<PROPER_NOUN>=4` · `<ENT>=32816` · `<CAP>=32818` ·
`<UNK>=0` · `<OUTPUT>=1706` · `<EOS>=3` · `<BOS>=2` · `next_id=33114`.

---

## 3. Rejim ve canlılık — `<OUTPUT>` ölçümü (T9 gerekçesi doğrulandı)

`train.py:372` SFT yolunda `mask_prompt_targets(...)` çağırır; `train_step_demo.py:58`
**kural 5**: *"`<OUTPUT>` içermeyen pencerede TÜM hedefler -100"* — maskeleme **pencere
başına**dır. Ölçüldü:

| külliyat | `<OUTPUT>` | 128'lik pencere | `<OUTPUT>` içeren pencere |
|---|---|---|---|
| ceket `.bin` (eski) | **12.763** | 6.778 | **6.776 = %99,97** |
| replay `train_chat_balanced.bin` | 24.374 | 24.375 | **24.374 = %100,00** |
| `anka_a1r_pretrain.bin` (Wikipedia) | **2** / 100.000.000 | 781.250 | 2 = **%0,00** |

**Üç hüküm:**
1. SFT maskelemesi ceket külliyatında **NO-OP DEĞİL** (T-0073 tuzağı bu külliyatta yok;
   o tuzak Wikipedia satırındadır, `<OUTPUT>`=2/100M). ⇒ **`--pretrain` ÇIKARILIR** (karar 3).
2. **Replay kaynağı SFT altında ÖLÜ DEĞİL** (`%100,00`) ⇒ replay gerçekten gradyan üretir.
   Bu, karışımın varlık sebebini kurtarır ve **ölçülmeden varsayılamazdı**.
3. `train.py:376-380` hedefsiz partide `RuntimeError` ⇒ **fail-closed**; sessiz sıfır kayıp yok.

### 3.1 — Canlılık bandı: maske rejimi için **YENİDEN İLAN** edilir

Maskeleme kayıp **ölçeğini** değiştirir ⇒ **eski düz-LM sayıları bu koşum için referans
DEĞİLDİR** ve kıyaslanmaz (T-0077).

| imza | ölçüt |
|---|---|
| **ÖLÜ** | kayıp tam **`0,0000`** ⇒ **DUR** (T-0073: MPS geçersiz hedefte hata vermez) |
| **CANLI** | ilk kayıp **sonlu** ve **`> 0`** |
| **Rapor biçimi** | `son60` **ort ± std** — *"Bitiş Kaybı" tek adımın ÖRNEKLEMİDİR* (T-0078) |

> **Sabit bir sayı bandı ilan EDİLMEZ.** Gerekçe: maske rejiminde bu model için ölçülmüş
> bir bant **yok**; uydurulmuş bir sayı (ör. "3,5–4,5") kanıtsız bir vaat olurdu. Bunun
> yerine **karar kuralı** ilan edilir: ölü imzası + sonluluk + **sondanın kendi tabanına
> göre düşüş yönü**. Sonda taban kaybını da ölçer; iki değer birlikte raporlanır.

---

## 4. Unutma eksenleri — eşikler T-0059'dan **kanonik kaynaktan** doğrulandı

`data/eval/t0059_verification.json` okundu (hafızadan alınmadı). Ölçülmüş referans:

| eksen | T-0059 ölçümü | eşik | sonuç |
|---|---|---|---|
| **A** genel CE (maskesiz) | f3_clean `1,962547` → carpenter_v2 `2,789015` | **≤ +%10** (≤ 2,1588) | **+%42,11 — AŞILDI** |
| **A'** maskeli (hedefle örtüşen) | `1,8153` → `2,8337` | — | **+%56,1** (maskesizden DAHA BÜYÜK) |
| **B** noktalama top-1 | %84,4 → %74,8 (250 konum, seed 7) | **≥ %79,4** (düşüş ≤ 5,0 puan) | **−9,6 puan — AŞILDI** |
| ortalama rank | 2,144 → 37,336 | — | çöküş |
| ceket öğrenimi (C) | maskeli CE `0,3285` → `0,0975` | — | öğrendi |

**Sonuç: T-0059'da unutma KATASTROFİKTİ ve ceket öğrenimi gerçekti — aynı koşumda.**

### 4.1 — Anka için A ekseni dilimi: **ÖLÇÜLDÜ**, T-0059 dilimi Anka'ya UYMUYOR

`anka_a1r.pt` iki dilimde ölçüldü (33.114 sözlükle, **kırpma YOK**):

| dilim | Anka A1-r maskesiz CE | okuma |
|---|---|---|
| `data/anka_a1r_pretrain.bin` (Wikipedia) | **4,0132** | Anka'nın **kendi taban dağılımı** (T-0093 A1 bandı 3,9833 ± 0,2867 ile tutarlı) |
| `data/train_balanced_sft_v2.bin` (T-0059 dilimi) | **6,7882** | Anka için **dağılım DIŞI** (Kristal verisi; A1-r hiç görmedi) |

⇒ **İlan edilen A ekseni = Wikipedia dilimi.** Gerekçe: unutma, modelin **kendi** taban
yetkinliğinin bozulmasıdır; model için zaten çok kötü olduğu bir dağılımda (6,79) ölçülen
değişim unutmayı değil gürültüyü ölçer.

**Beyan edilen sınır:** bu yüzden **T-0059'un MUTLAK sayıları bu koşumla kıyaslanamaz**
(farklı taban modeli, farklı dilim). Eşikler **göreli** oldukları için taşınır:
**A: artış ≤ +%10** · **B: düşüş ≤ 5,0 puan**. İkincil olarak T-0059 dilimi de **eşiksiz**
ölçülür ve raporlanır (yön bilgisi).

**B tanımı:** hedefi noktalama olan konumlarda top-1. Noktalama bloğu ölçüldü:
`32137..32145` = `. , ? ! - : ; ( )` (önek invaryantı ⇒ T-0059'un `(32137, 32138, 32142)`
kümesi aynen geçerli). Deterministik, `seed 7`, ≥ 250 konum.

> **Ölçüt kendi payını yiyor uyarısı (T-0078/H6):** B ekseni **yalnız düşüş** için eşikli;
> bir **iyileşme** "ceket işe yaradı" diye okunmaz — pozitif kontrol olmadan `değişti ≠ iyileşti`.

---

## 5. Faz 3a — Sonda tasarımı (kol seçimi ÖNCEDEN ilan edilir)

**Kollar (replay oranı):**

| kol | karışım | `--replay-every` |
|---|---|---|
| **R0** | **replay YOK** — düz ceket `.bin`'i doğrudan | — (karışım üretilmez) |
| **R4** | %25 | 4 |
| **R10** | %10 | 10 |
| **R20** | %5 | 20 |

> `build_replay_mix.py` `replay_every ≥ 2` ister ⇒ **R0 karışım gerektirmez**; doğrudan
> ceket `.bin`'i eğitilir. Dört kol da aynı tabandan (`anka_a1r.pt`), aynı tohumla.

**Sabit sonda hiperparametreleri:** `--steps 1000` · `--block-size 128` · `--batch-size 8`
· `--lr 2e-4` · **SFT (`--pretrain` YOK)** · `--device mps`.

**`--steps 1000` gerekçesi (ölçülmüş):** T-0059 unutmayı **tam 1.000 adımda** ölçtü ve her
iki eşik de aşıldı. Sonda, duyarlılığı **kanıtlanmış noktada** çalışır; daha uzun bir sonda
daha fazla bilgi vermez, yalnız saat harcar.

**`--lr 2e-4` gerekçesi ve PLAN DÜZELTMESİ (beyan edilir):** plan taslağı `--lr 0.001`
yazıyordu. Ölçülen tek ceket-eğitimi presedanı T-0059'dur ve o koşum `--lr` **vermediği**
için sessizce `2e-4` kullanmıştır (T1 tuzağı) ve ceket öğrenmiştir (maskeli CE
0,3285→0,0975). Tasarım **`2e-4`** olarak sabitlenir; T1 tuzağı değeri **açıkça vererek**
kapatılır. Bu düzeltme **uygulamadan önce** yapılmıştır, ölçümden sonra değil.

### 5.1 — Seçim kuralı (ÖNCEDEN ilan — ölçümden sonra değiştirilmez)

1. Bir kol **UYGUN**'dur ancak ve ancak **A ≤ +%10** **VE** **B ≥ −5,0 puan**.
2. Uygun kollar arasında kazanan **ceket ekseninde en yüksek ROUGE-L**'i olandır.
   ROUGE-L farkı **≤ 0,005** ise **en düşük replay oranı** kazanır (taban kayması azdır).
3. **Hiçbir kol uygun değilse ⇒ DURULUR** ve operatöre bildirilir. *"Daha iyi bir kol üret"*
   DENMEZ; replay oranı **uydurulmaz**.

> Kural 2, T-0060'ın *"%25 ceket sinyalini boğar"* bulgusunu **elle seçmeden** kodlar:
> %25 kolu ancak gerçekten en iyi ROUGE'u verirse kazanır. Bu, T-0060'ı **yanlışlanabilir**
> kılar.

### 5.2 — Sonda çıktıları

`$TMPDIR/t0094/sonda/` altına yazılır — **donmuş yüzeye değil**. Ara `.pt`'ler de
`$TMPDIR`'e. Sonuçlar artımlı (`save()` her koldan sonra) yazılır (T-0063).

---

## 6. Faz 3b — Tam koşum: tasarım sayıları **BETİKLE** hesaplanır

**`--steps` elle yazılmaz** (T-0067: *kaynak dışı sayı*). İlan edilen formül:

```
steps = ceil(jeton / (block_size × batch_size)) × EPOCH
```

* `jeton` → **yazılan `.bin`'in kendi meta'sından** (`total_tokens`); kopya sayı taşınmaz.
* `block_size = 128`, `batch_size = 8` ⇒ adım başına **1.024 jeton**.
* **`EPOCH = 3`** (ilan edilen tasarım parametresi).

**`EPOCH = 3` gerekçesi:** bu koşumda risk **eksik öğrenme değil, UNUTMADIR** — T-0059'da
1.000 adımda unutma iki eşiği de aştı. Bu yüzden epoch sayısı **küçük** seçilir; ceket
külliyatı küçüktür ve A1-r zaten güçlü bir tabandır. Sonda, adım sayısının yetersiz
kaldığını gösterirse bu **rapora yazılır**, tasarıma geri dönülmez.

**Replay oranı:** §5.1'in seçtiği kol.

### 6.1 — Ölçülen tuzaklar ve karşı önlemler

| # | tuzak | önlem |
|---|---|---|
| T1 | `--lr` verilmezse resume dalı sessizce `2e-4` | `--lr 2e-4` **açıkça** (ayrıştırma `:282-284` resume'dan sonra ⇒ açık değer kazanır) |
| T2 | `--block-size` üst-düzey `block_size` arar; ceket/replay meta'sında **yok** ⇒ sessizce 64 | `--block-size 128` **açıkça** |
| T3 | `--save-path` ZORUNLU (`:219-223`) | göreli yol + `--allow-frozen-write` |
| T4 | `frozen_guard.py:59` **mutlak yolda FAIL-OPEN** | `--save-path` **göreli**; kapıya güvenilmez |
| T5 | `anka_a1r.pt` için `.opt.pt` **YOK** ⇒ `--load-optimizer` rc=2 | `--load-optimizer` **verilmez**; ilk güncelleme **1,73×** (T-0092, beyan edilen bedel) |
| T6 | `--data` yoksa çıplak `return` ⇒ **rc=0** | `rc` tek başına kanıt sayılmaz; artefakt digest'i de ölçülür |
| T7 | `--vocab` varsayılanı 32.852 | `--vocab vocab_anka_r1_33114.json` **açıkça** |
| T8 | MPS sandbox'ta görünmez | gerçek koşum **sandbox DIŞINDA** |

---

## 6b. **EK (Addendum) — 2026-09-20T14:35Z · held-out düzeltmesi (operatör kararı)**

> Bu ek, **hiçbir ilan edilmiş kapıyı, eşiği, kolu veya seçim kuralını DEĞİŞTİRMEZ.**
> Yalnız Faz 1'in **girdisini** değiştirir ve gerekçesi ölçümdür. Eklenen tek şey **yeni ve
> daha sıkı bir kapıdır (V7)**.

### Neden — ölçüm (planın K1'i ÇÜRÜDÜ)

Plan K1, held-out zeminin `data/pedagogy_canonical/carpenter_canonical.jsonl`'dan
türetileceğini ve *"spec∩canon input = 145/145"* olduğunu yazıyordu. **Ölçüm bunu yanlışladı:**

| zemin | eğitimle çıktı kesişimi | hüküm |
|---|---|---|
| `pedagogy_canonical/carpenter_canonical.jsonl` | **4.635/5.400 = %85,8** (ham) · önek-normalize ile **%100,0** | **KULLANILAMAZ** |
| `pedagogy/arena_carpenter_accumulated.jsonl` (derleme kaynaklarında YOK) | **297/302 = %98,3** | **KULLANILAMAZ** |

Kanonik dosya, spec'in **satır hizalı normalleştirilmiş biçimidir** (önek temizlendikten
sonra çıktı eşleşmesi **4.950/5.400 = %91,7**). Yani o dosya **eğitim verisidir**, held-out
değil. Önek-normalize ölçütle temiz kalan kanonik kayıt **85/5.400 = %1,6**; iki ölçütün
**birleşimiyle 0/5.400 = %0,0**.

⇒ **Bu depoda temiz marangoz zemin YOK**; tek yol eğitim kaynağından **pay ayırmaktır**.

### Operatör kararı (AskUserQuestion, bu oturum)

*"Carve-out + yeniden derle"* — seçenekler: (A) carve-out, (B) tam külliyat + ceket ekseni
ölçülmez, (C) tam külliyat + kontamine zemin beyan edilerek. Operatör **(A)**'yı seçti.

### Değişen girdi (yeni ilan)

| kalem | eski ilan | **yeni ilan** |
|---|---|---|
| carpenter kaynağı | 5.400 satırın tamamı | **4.860 satır** (5.400 − 540 held-out) |
| held-out | kanonik dosyadan türetilir | **spec'ten `seed 42` ile ayrılan 540 satır (%10)** |
| held-out yolu | — | `data/eval/anka_r17_heldout_2026-09-20.jsonl` |
| `--steps` | formül aynı | **formül AYNI**; `jeton` yeni artefaktın meta'sından okunur |

Bölme **deterministiktir** (`random.Random(42)`, satır indeksleri üzerinde permütasyon).
Held-out satırları eğitim külliyatına **hiç girmez** ⇒ sızıntı **tanım gereği 0**; yine de
**ölçülerek doğrulanır**.

### Eklenen kapı — **V7 (sızıntı testi, daha sıkı)**

Aynı kesişim testi (bu ekteki kanonik/arena ölçümünün ta kendisi) **yeni üretilen külliyata
karşı** koşulur: hem **ham** hem **önek-normalize** biçimde, hem `input` hem `output`
ekseninde. **Beklenen: 0/540.** Sıfır değilse **`rc=2` ile DURULUR**.

> **Gerekçe:** held-out'u öldüren test, held-out'u doğrulayan test olmalıdır. Kanonik zemini
> "temiz" sanmamın sebebi bu testi **koşmamış olmamdı**; aynı hata tekrarlanmaz.

---

## 6c. **EK 2 — 2026-09-20T15:05Z · carve-out BİRİMİ düzeltildi (§6b'nin bir iddiası ÇÜRÜDÜ)**

> Bu ek **hiçbir eşiği, kolu veya seçim kuralını değiştirmez.** V7'nin **beklentisi 0'da
> kalır**. Değişen tek şey: V7'nin o 0'ı **hangi birimde** aradığı. §6b'nin *"sızıntı tanım
> gereği 0"* cümlesi **ölçümle yanlışlandı** ve burada kayda geçirilir.

### Neden — V7 ilk koşumda **ATEŞLEDİ** (kapı çalıştı)

§6b'nin satır-indeksi permütasyonu uygulandı ve V7 koşuldu. Sonuç **sıfır DEĞİL**:

```
[V7 ON-KONTROL] metin kesismesi: {'output_ham': 319, 'output_onek_norm': 274,
                                  'input_ham': 110, 'input_onek_norm': 110,
                                  'instruction_ham': 84, 'instruction_onek_norm': 84}
[V7 ON-KONTROL] kayit kesismesi: 300 (held-out 1051 kayit ↔ korpus 11712 kayit)
DURDURULDU: V7 SIZINTI ... rc=2
```

**Neden:** §6b *"held-out satırları külliyata hiç girmez"* diyordu — bu **satır düzeyinde
doğru**, ama **içerik düzeyinde yanlış**. Kaynak **yoğun biçimde tekrarlıdır** (ölçüldü):

| ölçüm | değer |
|---|---|
| satır | 5.400 |
| **benzersiz `output`** (ham) | **2.225** |
| **benzersiz `output`** (önek-normalize) | **1.775** |
| **cevabını başka satırla paylaşan satır** | **4.050 / 5.400 = %75,0** |
| benzersiz `instruction` | 3.396 (şablonlar **450×** tekrar) |
| benzersiz `input` | 146 (en kalabalığı **3.300 satır = %61**) |
| grup boyut dağılımı | `{1: 1350, 9: 400, 18: 25}` |

Yapı çözüldü: kaynak **450 soru × 9 yeniden-ifade şablonu** (4.050 satır) + 1.350 singleton.
Aynı cevap, dokuz farklı talimat kalıbı altında duruyor. ⇒ Bir satırı tutmak, **cevabını
kardeş satırdan eğitime bırakır**; "held-out" görünürde temiz, içerikte değil.

> **Bu, kanonik zemini öldüren testin aynısının beni yakalamasıdır** — ve tam da bu yüzden
> V7 ilan edilmişti. *"Held-out'u öldüren test, held-out'u doğrulayan test olmalıdır."*

### Düzeltilen birim (yeni ilan)

| kalem | §6b ilanı | **§6c ilanı** |
|---|---|---|
| bölme birimi | satır indeksi | **önek-normalize `output` GRUBU** |
| held-out ölçüsü | 540 satır | **178 grup ⇒ 539 satır** (%10 grup · %9,98 satır) |
| eğitim | 4.860 satır | **4.861 satır** |
| `--steps` | formül aynı | formül **AYNI**; `jeton` yeni artefaktın meta'sından |

Bir grup **bütün olarak** tutulur ⇒ held-out cevabının **hiçbir yeniden-ifadesi** eğitimde
kalmaz. Determinizm: grup anahtarları **sıralanır** + `random.Random(42)`.

### V7'nin eksen politikası — §6b'nin *"her iki eksende 0"* iddiası **ÇÜRÜDÜ**

§6b, kapıyı `input` ve `output` eksenlerinin **ikisinde de** 0'a bağlamıştı. Ölçüm bunun
**`input` ekseninde ulaşılamaz** olduğunu gösterdi (146 benzersiz `input`, en kalabalığı
külliyatın %61'i). Politika **ölçüme göre ayrılır**:

| eksen | politika | gerekçe |
|---|---|---|
| **`output` (CEVAP)** | **KAPILI — 0 olmalı** | Cevap **ölçülen şeydir**; cevap metni eğitimde kalırsa iyileşme **kopyalamadan ayırt edilemez** (T-0062) |
| **`input`/`instruction` (SORU)** | **ölçülür, KAPILMAZ** | Bunlar **görev dağılımıdır**; paylaşılan soru = **genelleme**, paylaşılan cevap = **sızıntı** |

> **Dürüstlük:** §6b'yi yazarken eksen yapısını **ölçmedim**; "0"ı iki eksene birden
> koydum. Bu, K1/K2 ile **aynı sınıftan** bir kusurdur (ilan edilen ölçüt ölçülmeden
> yazıldı) ve **burada beyan edilir**. Kapı **zayıflatılmadı**: kararı **veren** eksende
> (cevap) en sıkı hâlinde, 0'da bırakıldı.

---

## 6d. **EK 3 — 2026-09-20T15:40Z · Faz 2 kabı: protokol ÖLÇÜM GÜCÜNE göre sıkılaştırıldı**

> Bu ek **hiçbir eşiği değiştirmez.** Eşikler (A ≤ +%10 · B ≤ −5,0 puan) **aynen kalır**.
> Değişen: kap, o eşikleri **çözebilecek** örneklemle koşar. Ve kap bunu **kendisi
> doğrular** — çözemiyorsa **`rc=2` ile DURUR**.

### Neden — ölçüldü: T-0059'un örneklemi bu tabanda eşiği ÇÖZEMİYOR

B ekseni `n = 250` konumda, bu tabanın noktalama top-1'i (p ≈ %41) için Wilson %95
yarı-genişliği **±6,05 puan** — **eşik 5,0 puan**. Yani kap, ayırt etmesi gereken şeyi
ayırt edemeyecek kadar gürültülüydü. (T-0059'un kendi `n=250`'si p = %84,4'te ±4,94 ile
**sınırda** geçmiş; kusur T-0059'da da vardı, orada görünmedi.)

| eksen | eski örneklem | yarı-genişlik | eşik | hüküm |
|---|---|---|---|---|
| **A** (CE) | 64 pencere | ±0,2205 | pay 0,346 | sınırda (1,57×) |
| **B** (noktalama) | **250 konum** | **±6,05 puan** | **5,0 puan** | **ÇÖZEMİYOR** |
| **A** (yeni) | **256 pencere** | **±0,0984** | pay 0,3535 | **3,6×** ✅ |
| **B** (yeni) | **2.500 konum** | **±1,95 puan** | 5,0 puan | **2,6×** ✅ |

**Yan kanıt:** aynı model, aynı dilim, yalnız örneklem büyüyünce B top-1 **%41,04 → %49,56**
oldu. Küçük örneklemin verdiği sayı **artefaktmış**; bu, güç düzeltmesinin gerekçesidir.

### Eklenen iki fail-closed kapı

1. **B:** `Wilson yarı-genişliği ≥ 5,0 puan` ⇒ **`rc=2`**, hiçbir hüküm verilmez.
2. **A:** `%95 yarı-genişlik ≥ (0,10 × taban CE)` ⇒ **`rc=2`**.

> **Gerekçe:** *çözemeyeceği bir eşiği sınayan kap, kap değildir.* Bu, T-0075/T-0093'ün
> **fail-closed** ailesindendir: ölçüm gücü yetmiyorsa **durmak** gerekir, "geçti" demek
> değil.

### Kabın ölçtüğü A ekseni MUTLAK değeri — ilan §4.1 ile **çelişmiyor**, farklı **kaptır**

§4.1, `anka_a1r.pt` için Wikipedia diliminde maskesiz CE'yi **4,0132** diye ilan etmişti.
Faz 2 kabı aynı dosyada **3,5352** ölçüyor. **Bu bir çelişki değildir:** aynı dosya,
**farklı pencere örnekleme protokolü** (farklı pencere sayısı/seçimi). Dağılım geniş
(±0,8032) ⇒ iki sayı aynı dağılımın iki örneklemidir.

**Bağlayıcı sonuç:** A ekseni **mutlak** sayısı protokole bağlıdır ve **illeride
kıyaslanamaz**; **eşiği taşıyan şey EŞLİ FARKTIR** (taban ve ceketli model **aynı kapta,
aynı pencerelerle**). Kap, tabanı `--baseline` ile **kendisi ölçer** ⇒ fark protokol
uyumludur. §4.1'in 4,0132'si artık **yön bilgisi**dir, referans sayı değil.

### Kabın doğrulanmış pozitif kontrolü (Faz 2 duman testi)

`--model = --baseline = anka_a1r.pt` (kimlik eşlemesi) koşuldu: **A artış %+0,00 · B
düşüş +0,00 puan** — eşli mantık **tam sıfır** verir, yani kendi kendini doğrular.
Oracle katmanı: kimlik ROUGE **1,0000** (tavan tuttu) · distraktör **0,1379** · sabit-tahmin
**0,1576** — ikisi de eşik 0,35'in **çok altında** ⇒ ölçüt **AYIRT EDİYOR** (T-0091'in
"42.459 vakada 0 karar değiştiren kör araç" kusuru bu kapta **yok**).

---

## 6e. **EK 4 — 2026-09-20T17:05Z · Sonda konteyneri: `$TMPDIR` DEĞİL `scratch/t0094_sonda/`**

§5.2 *"`$TMPDIR/t0094/sonda/` altına yazılır"* diyordu. **Uygulanmadı** — sebep **ölçüldü**:

> `$TMPDIR` **sandbox sınırında değişiyor.** Sandbox içi: `/tmp/claude-501`.
> Sandbox dışı: `/var/folders/v0/8z_jjnds4rbdtth51qmwysnw0000gn/T/`.
> Karışım `.bin`'leri sandbox içinde yazıldı; ardından **sandbox dışında** koşan ilk eğitim
> denemesi `RuntimeError: Parent directory …/T//t0094/sonda does not exist` ile **durdu**.

Bu, kayıtlı **`kopyalanan-betik-kabini-degistirir`** sınıfının ta kendisidir: *kabı iki
dünyada aynı sanmak*. Düzeltme **`scratch/t0094_sonda/`** — depo içinde, **her iki dünyada
aynı mutlak yol**, `scratch/` zaten `.gitignore`'da ve T-0094 kirası kapsamında.

**Doğrulama (kaza eseri bedava çapraz kontrol):** karışımlar iki ayrı dizinde, ayrı koşumlarda
üretildi ve **sha256'lar birebir aynı** çıktı (`r4 a31a58b0…`, `r10 9d446311…`, `r20 0bca16bd…`)
⇒ `build_replay_mix.py` **deterministik**.

**Beyan edilen yan bulgu (kapı boşluğu):** `scripts/build_replay_mix.py` **hiçbir donmuş-yol
denetimi çağırmıyor** (`grep`: `frozen` **0 eşleşme**), oysa varsayılan `--output`'u
`data/train_f4_replay_mix.bin` — yani **donmuş `data/*.bin` deseninin içinde**. Bu betik
`data/` altına **kapısız** yazabilir. Sonda bu yüzden `scratch/`'e yazdı; **Faz 3b'nin
karışım yolu bu bulguya göre seçilecektir**. Boşluk **kapatılmadı** (kapsam dışı) — beyan edilir.

---

## 7. Ölçüm sınırları (baştan beyan)

* Bu görev **ceketi giydirir ve ÖLÇER**; "Anka iyi marangoz oldu" hükmü **vermez** —
  held-out zemin ve oracle katmanları olmadan `değişti ≠ iyileşti` (T-0080/F2, T-0091).
* `data/train_carpenter_specialization.bin` (eski ceket) **değiştirilmez** — kanıt olarak durur.
* `src/llm/frozen_guard.py:59` FAIL-OPEN açığı **kapatılmaz**; yalnız beyan edilir.
* D1 (en-uzun-kök) ve D4 (kesme/rakam) **kapsam dışı**.
* `scripts/prepare_carpenter_specialization_dataset.py` ve
  `scripts/evaluate_carpenter_generation_100.py` **değiştirilmez** (yeni dosyalar yazılır).

---

## 6f. **EK 5 — 2026-09-20T18:0xZ · DÜZELTME: §6e'nin oracle çifti ÖLÇÜLMEMİŞ**

§6e *"distraktör **0,1379** · sabit-tahmin **0,1576**"* diyordu. **Bu çift hiçbir saklı
ölçüm dosyasında YOKTUR.** Denetim: `scratch/t0094_sonda/` altındaki **10** sonuç JSON'unun
**hiçbiri** bu değerleri taşımıyor; ölçülenler:

| `n` | dosya | distraktör | sabit-tahmin |
|---|---|---|---|
| **100** | `eval_R0_fix.json` … `eval_R20_fix.json` | **0,0848** | **0,0780** |
| 4 | `_smoke_fix.json` | 0,1061 | 0,1223 |

**Kök neden:** oracle sayıları **`n`'e bağlıdır** ve §6e o satırda **`n`'i yazmıyordu**.
Saklı olmayan bir duman koşumundan gelmiş olması **muhtemeldir** ama **doğrulanamaz** ⇒
"doğrulanamaz" diye kaydedilir, tahmin **kanıt sayılmaz**
([[yurutucu-raporlari-bagimsiz-dogrulanmali]]).

**Hüküm değişmiyor:** üç değerde de ölçüt eşiği **0,35**'in çok altında ⇒ *ölçüt ayırt
ediyor* sonucu **ayakta**. Yanlış olan **sayı**, **hüküm değil**.

**Yeni kural (bu ekin bağlayıcı maddesi):** bir oracle/metrik sayısı, **örneklem boyutu
(`n`) yanında yazılmadan** kanıt değildir. §6e'nin sayıları bu ekle **geçersiz** sayılır;
yerine **§1.4'ün `n=100` çifti** geçer.

> **İlanın geri kalanı DEĞİŞTİRİLMEDİ** — §1–§5 ve §6b–§6e metinleri olduğu gibi durur;
> bu ek **yanlış olanı işaretler**, tarihini silmez.
