# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — YETENEK EKSENİ: adım bütçesi

**İlan:** `data/eval/anka_r32_adim_ilani_2026-09-22.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099g_zincir.log` · 3000 adım · 1024 sn (0,36 sn/adım) · `rc=0` · 08:20:33Z→08:38:34Z

---

## 1. Hüküm: BİRİNCİL okuma "gelmedi" ⇒ içerik GELMEDİ, ve kısıt **mekanizmadır**

| # | okuma | ilan: geldi ise | ölçülen (r30 → r32) | sonuç |
|---|---|---|---|---|
| **1 (birincil)** | **kesişim** | ≥ %10 | **%1,00 → %0,00** | **GELMEDİ** |
| 2 | tutarsızlık | < %60 | %98,00 → %88,00 | GELMEDİ |
| 3 | ROUGE-L | ≥ 0,096 | **0,0696 → 0,1168** (+0,0472 · **2,9 SE**) | **GELDİ** |

**Birincil okuma hükmü verir** (ilan §4) ⇒ **içerik gelmedi**. İlan §5'in **ikinci dalı**
uygulanır: *"kısıt **bütçe değil, mekanizmadır** ⇒ 'rank-sınırlı delta donuk tabana **biçim**
enjekte edebilir, **bilgi** edemez' hipotezi güçlenir ve **mimari ekseni kapanır**."*

## 2. Ayırt edici okuma — `ezber` ile `kesişim` AYRIŞTI

İlan §4'ün önceden yazdığı imza: *"ezber yükselirken kesişim sabit kalırsa ⇒ yalnız biçim."*

| | r30 | r32 | değişim |
|---|---|---|---|
| **ezber** | %28 | **%38** | **+10** |
| **kesişim** | %1 | **%0** | **−1** |
| eğitim kaybı (son60) | 3,8178 | **3,2479 ± 0,2947** | **belirgin düşüş** |

⇒ **AYRIŞTI. Yalnız biçim pekişti.** Model veriye çok daha iyi sığıyor (kayıp −%15), ama
sığdırdığı şey **soruya değmiyor**.

## 3. Ham üretimler — ROUGE neden yükseldi

```
idx 1  'bu işlem için'                                      rouge 0,5455
idx 2  'teknik çözüm: teknik çözüm:'                        ezber
idx 3  'teknik çözüm: teknik çözüm: Tanımlanan özellikler doğrudan ahşap tutulur.'  tutarsız=YOK
idx 4  'teknik çözüm: teknik çözüm:'                        ezber
referans (idx 1): "Teknik Çözüm: Bu işlem için 'kalınlık makinesi' kullanılmalıdır."
```

`idx 1`'in ROUGE'u **0,5455** çünkü üretilen şey referansın **içinden bir kalıp** (`bu işlem
için`). Bu bir **yankı**, bilgi değil. `idx 3` ilk kez `tutarsız=YOK` — ama ürettiği cümle
soruya cevap değil, **genel bir kalıp**.

⇒ **ROUGE yükseldi çünkü model eğitim metninden kalıplaşmış parçalar üretiyor.** Hiçbiri
sorunun içeriğine değmiyor (`kesişim %0`).

## 4. BEDEL: unutma kapısı DÜŞTÜ

| | r30 | r32 |
|---|---|---|
| A artışı | +%9,98 ✓ (pay yarı-genişliğin 0,147 katı) | **+%13,35 ✗** (pay **−3,35 puan**) |
| B düşüş | +1,51 ✓ | +0,67 ✓ |
| **`unutma_gec`** | True | **False** |

r24/r25'ten (chat replay dönemi) sonra **ilk kez** bir modül unutma kapısını düşürdü.
⇒ Bütçe ekseni **bedava değil**: 3× adım hem (yalnız) biçimi pekiştirdi hem de tabanı bozdu.

## 5. Tablo

| koşum | adım | ROUGE-L | tutarsızlık | **kesişim** | ezber | A artış | unutma_gec |
|---|---|---|---|---|---|---|---|
| r28 varsayılan | 1000 | 0,0191 | %100 | %0 | %0 | +%6,08 | ✓ |
| r29 `r`=64 | 1000 | 0,0248 | %100 | %0 | %0 | +%6,19 | ✓ |
| r30 +`lm_head` | 1000 | 0,0696 | %98 | %1 | %28 | +%9,98 | ✓ |
| r31 +`embedding` | 1000 | 0,0595 | %98 | %0 | %20 | +%9,45 | ✓ |
| **r32 +`lm_head` · 3000 adım** | **3000** | **0,1168** | %88 | **%0** | %38 | **+%13,35 ✗** | **✗** |
| *tam ince ayar seg_1 (1000 adım)* | 1000 | 0,1097 | **%24** | %0 | %20 | +%23,31 ✗ | ✗ |
| *tam ince ayar seg_6 (6000 adım)* | 6000 | **0,3035** | **%5** | **%10** | %0 | +%75,62 ✗ | ✗ |

**Modül ROUGE'da tam ince ayarı yakaladı (0,1168 > 0,1097)** — ama **nitelik olarak ayrı**:
tam ince ayar %24 tutarsızlık ve %20 ezberle; modül %88 tutarsızlık ve %38 ezberle.
Aynı sayı, farklı şey.

## 6. Ne kapandı, ne kaldı

**KAPANAN — mekanizma ekseni.** Kapasite (r29) · giriş tarafı (r31) · **ve şimdi bütçe (r32)**
elendi. Rank-sınırlı delta **donuk** bir tabana **biçim** enjekte edebiliyor, **bilgi** edemiyor.
Bütçeyi 3× yapmak bunu değiştirmedi — yalnız ezberi ve unutmayı artırdı.

**KALAN — ve ölçülmüş bir gerekçesi var:** **taban talimat izleyemiyor.** Ölçüldü (r23):
taban ROUGE **0,0000**, tutarsızlık **%100**, çıktı `, [?], [?], …`. Elimizdeki iki checkpoint
(`anka_a1.pt`, `anka_a1r.pt`) da **ön-eğitilmiş**; talimatla ince ayar görmüş bir temel **yok**.

⇒ Modül bugüne kadar **iki işi birden** yapmaya çalıştı: talimat izlemeyi öğret **ve** alan
bilgisi enjekte et. Sonuç, bu iki işin **ilkini** (biçim) başardığını, **ikincisini** (içerik)
başaramadığını gösteriyor. Operatörün mimarisi ("temel bir kez eğitilir, yetenekler modül
olur") tam olarak şunu söyler: **temelin işi talimat izleyebilmek**, modülün işi yetenek.
Şu anki temel o eşiğin **altında**.

**Ölçülmüş bir "çalışan temel" adayı var:** `scratch/t0096_kos/seg_6.pt` — ROUGE 0,3035,
tutarsızlık **%5**, ezber %0. Yani **talimat izleyebilen** bir model. Bedeli: Wikipedia
**+%75,62** bozuk. Bir sonraki ilanın doğal adayı: **temeli bu niteliğe getirip modülü onun
üstüne kurmak** — ya da genel talimat verisiyle bir kez eğitilmiş bir temel üretmek.

## 7. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `modules/marangoz_lmhead_3000.mod.pt` | `c9bbf2c137f7765a0e4298635c6f41c3c999288e5ee6b9cd886addebfae38ed0` |
| `data/eval/anka_r32_adim_yetenek_2026-09-22.json` | `9ccccd3dc2c5f3bfda46d46ff9c8eb0928d868ff664ea6de1e8f7843edd49329` |
| `data/eval/anka_r32_adim_yetenek_2026-09-22.modul.json` | `e0f0b2dd4a79269fd513a86ab8c882770b91e28d735dca6222d09a0d324a6202` |
| `scratch/t0099g_zincir.log` | `271f93b213b6df6307ee1e4c91049b5391f36c285d3882c776fce8082a275363` |
| `data/anka_a1r.pt` (taban, değişmedi) | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |

Modül meta: `37 katman · 1.869.216 parametre · adım=3000 · kayıp ilk 6,2606 → son60 3,2479 ±
0,2947 · 7,5 MB`. Vakum **19,436** ≠ 0 · değişen taban tensörü **0**.

**Commit yok · `git add -A` kullanılmadı · donmuş kalıplara yazılmadı · `src/**` bu koşumda değişmedi.**
