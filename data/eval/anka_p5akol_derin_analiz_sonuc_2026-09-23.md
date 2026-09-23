# P5-A · A-KOLU DERİN ANALİZ SONUCU — JETON çıktıları decompiler'da (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Görev:** T-0100 (üst: T-0099) · **Koşum YOK** (CPU
ölçüm, ~1 dk) · İlân: `anka_p5akol_derin_analiz_ilani_2026-09-23.md` (ölçümden
ÖNCE, sha256 `fa1f6d11…`) · Sonda rc=**0** · **TANISAL** — eşik önerisi YAPILMADI.

## 1. Ana bulgu — RAW tavan, temsil katmanı artefaktıdır

Aynı orneklem (heldout 539, n=100, seed 42 — P5 sonda betiğiyle birebir aynı
dizi), üç temsil:

| temsil | ROUGE-L ort | not |
|---|---|---|
| YUZEY | 1,0000 | referansın kendisi |
| RAW (P5 tavanı) | 0,4164 | `vocab.decode` dizisinin ham metni (ECA:481'in gördüğü) |
| **DECOMP** | **0,9509** | kanonik decompiler yüzey metni; **Δ +0,5345** |

**0/100 kayıtta DECOMP < RAW** — decompiler her kayıtta iyileştirdi. P5'in
0,4164 "tavanı"nın ~%87'si (0,5345/0,5836) **ham decode kaybıdır**;
`MorphemeDecompiler` ekleri yüzey Türkçesine çözümleyerek onarıyor
(`kullanıl TENSE_NECESS COPULA_AORIST` → `kullanılmalıdır`).

## 2. Kullanıcının üç eksenine ölçülmüş cevaplar

### 2a. Sözlük boyutu (33.114) — SINIRLAYICI DEĞİL (bu örneklemde)
* UNK: **31 token / 2.138** roundtrip token'ının ~%1,45'i; 30/100 kayıtta en
  az bir UNK.
* **UNK'lu kayıtların DECOMP ROUGE'u (0,9613) UNK'sızlardan (0,9465) YÜKSEK**
  — UNK etkisi bu örneklemde ölçülebilir sinyal değil (30 kayıt gürültü
  bandında). 33.114 boyutu bu külliyatın referans metinlerini tutuyor;
  `patinası`/`tezgah` sınıfı kayıplar tek başına tavanı cimrileştirmiyor.

### 2b. root.tsv (52.582 satır) — kök vuruş %93,26; vuruşsuzluk zararsız sınıflarda
* Kök adayı 2.138; vuruş **1.994** (%93,26) · vuruşsuz **144** (34 özgün).
* Vuruşsuz sınıf ayrışması (betikle): **79 rakam** (`[sayı]` yer tutucu —
  lexicon'da OLMAYACAK; tasarım gereği) + **65 çekimli/başharf-büyük bütün
  kelime** (`Ahşabın` ×5, `Cevabı` ×7, `Tahtaların` ×3, `Kerestenin` ×1 —
  tokenizer çözümlemedi; decompiler kök adayı olduğu gibi geçiriyor,
  ROUGE'u çoğunlukla koruyor) + `l ×3` (**kesme sınıfı** —
  decompiler-kesme dersinin canlı örneği).

### 2c. Jeton listesi kalitesi — decompiler TAM DETERMİNİSTİK
* Ek etiketler: **1.190/1.190 (%100) `affix_info`'dan çözümlendi**, `COMMON_
  FALLBACK` 0, `decompile_sentence` istisnası **0** — morphotactics graph +
  PhonologyEngine bu külliyatta tek istisnasız çalışıyor.
* Yani DECOMP tavanının 1,0'a kalan ~%5'lik boşluğu UNK (birleşen `patinası`
  sınıfı) + çekimli-bütün-kelime sınıfı + küçük LCS farklarından; ek
  çözümlemesi tam.

## 3. Mimari bağ (README:73 akışı: prompt → compiler → model → modül → decompile)

Üretim akışının **son aşaması decompiler'dır** (README:73
`Transformer → MorphemeDecompiler (decomp0)`; README:86 akıcı yüzey
sentezi). P5 ölçüt kabı ROUGE'u RAW `gm`'den sayıyor (ECA:481) — yani kapı,
akışın decompile aşamasını **atlıyor** ve modelin üretim akışında görülmeyecek
bir temsili ölçüyor. Ölçülmüş sonuç: kapı ROUGE tavanı RAW 0,4164 iken
üretim akışının göreceği temsilin tavanı **0,9509**.

**AÇIK KONU (ilan §4 beyanı — bu analizde YAPILMAZ):** kapı ROUGE'unu RAW
yerine DECOMP'tan saymak (ECA:481'de `gw = kelimeler(gm)` → decomp metni).
Bu bir **ölçüt kararıdır**: ilan gerektirir, tüm zincir (ECA → olcum_kabi →
test → tablo) güncellenir ve P3/P4/P5 kıyasları çift hükümle yeniden okunur.
Öneri kanıtının tamamı bu raporda: tavan 2,3× yukarı çekilir ve ölçüt
üretim akışına (kullanıcının hatırlattığı mimari) hizalanır.

## 4. Model karşılaştırmasına etkisi (sadece bağlam)

P4 0,1346 RAW tavanın %32'si. DECOMP tavanıyla okuma, modelin decompile-aşama
sonrası konumunu ölçmeyi gerektirir — P4 sonda JSON'ları RAW'dan sayıldığı
için mevcut sayılar DEĞİŞMEDİ; geçiş kararı verilirse çift-hüküm deseni
(P5'te kanıtlandı) yeniden kullanılır.

## 5. Digest tablosu (betikle hesaplandı)

| artefakt | sha256 |
|---|---|
| `scratch/anka_p5akol_derin_analiz.py` | `936d4229…` |
| `scratch/anka_p5akol_derin_analiz.json` (kayıt başına RAW/DECOMP + kayıt metinleri) | `dc6dc910…` |
| `data/eval/anka_p5akol_derin_analiz_ilani_2026-09-23.md` | `fa1f6d11…` |

*Kaynak çıpası: heldout `c397eb08…` (P4 raporuyla birebir) · vocab 33.114 ·
lexicon `roots_anka_r1.tsv` 52.582 satır.*

## 6. Kapanış

**P5-A KAPANDI (tanısal):** A-kolu üç temsil tavanı ölçüldü — RAW 0,4164 /
DECOMP **0,9509** / YUZEY 1,0; kayıp ayrışması tamamlandı (sözlük sınırlayıcı
değil · kök vuruş %93,26 vuruşsuzluk zararsız sınıflarda · ek çözümleme
%100). Eşik değişikliği yok; DECOMP geçişi AÇIK KONU olarak operatöre bırakıldı.