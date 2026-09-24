# T-0102 · SAPMA BİRİKİMİ AYRIŞMASI SONUCU — küçük sapmaların üst üste gelmesi (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Görev:** T-0102 (üst: T-0101 KAPANDI) ·
**Koşum YOK** (CPU) · İlân: `anka_t0102_sapma_ayrisma_ilani_2026-09-23.md`
(ölçümden ÖNCE) · Sonda rc=**0** · **TANISAL** — eşik önerisi YAPILMADI.

## 1. Ilanlı beklenti kıyası

| ölçüm | ilanlı beklenti | ölçülen | hüküm |
|---|---|---|---|
| (A) S2 (ek yüzeyli türev) | %80-90 | **%86,75** (13.064) | BAND İÇİ |
| (A) S1 (self-root, köksüz teknik terim) | %10-20 | **%13,25** (1.995) | BAND İÇİ |
| (A) S2 compile-recover | ≈ 0 | **4/13.064 (%0,03)** | BAND İÇİ |
| (B) bin UNK oranı | %2,38 | **%2,3765** (2.376.511) | BİREBİR |
| (B) PROPER_NOUN-bağlam payı | ~%20-30 | **%17,6** | bandın kenarı (ayrışma §3) |
| (B) `[sayı]` placeholder | ölçülür | **0 — FALSE-ALARM beyanı** | anahtar yok (§3) |
| (C) token-düzeyi fark ≪ kayıt-düzeyi | küçük | 309 / 11.217 kelime (%2,75) | BAND İÇİ |

## 2. (A) 'diger' tam sayımı — köke gelen ek sayısı sorusunun tam cevabı

| sınıf | adet | oran | içerik |
|---|---|---|---|
| S2 — katı-önek kökü VAR + ek yüzeyi | 13.064 | %86,75 | `bıçaklanış` → `bıçaklan` + `ış` |
| S1 — kendisi kök, ek YOK (teknik terim) | 1.995 | %13,25 | `strafor`, `nükleofilik`, `pgas`, `advertorial` |

* S2 ek yüzeyi dağılımı: 2 harf zirve; en sık ek `-me/-ma` (isim-fiil) ·
  `-abil/-ebil/-yabil` (POTENTIAL) · `-ış/-iş` · ettirgen kalıntıları
  (`-tır/-tir/-t/-k`). Tam külliyatta **compile-recover %0,03** (4/13.064) —
  zincirler graph'ta yok; S2'nin köklerinin vocab'da olma sayısı
  JSON'da (`s2_kok_vocabda`).
* S1'in tanımı gereği vocab-dışı (0/1.995 token) — önerinin
  (`tech<id>`) hedef sınıfı **külliyat lemma kapsamında %13,25**.

## 3. (B) Külliyat UNK kitlesinin gerçek bileşeni — beklentiyi değiştiren bulgu

2.376.511 UNK / 100 M jeton (%2,3765); 5 pencerede %2,06-2,49. Sınıf
payları (ÇAKIŞAN ölçütler — toplam %100 aşar, beyanlı):

| sınıf | pay | yorum |
|---|---|---|
| **İngilizce-komşu** (ASCII-alpha) | **~%46 (1,09 M jeton)** | külliyatta İngilizce kalıntı; `('the' → 'of') × 401` bağlamı — veri temizliği sınıfı |
| noktalama-komşu | %29,4 | |
| ek-etiket-komşu (çekimli) | %16,6 | |
| PROPER_NOUN-bağlam | %17,6 | kumanda beklentisinin (%20-30) altında |
| rakam-komşu | **%9,7** | `[sayı]` placeholder külliyatta KULLANILMAMIŞ |
| UNK-koşusu | %13,3 | çok-jetonlu bilinmeyenler |

**FALSE-ALARM beyanı:** `[sayı]` placeholder payı 0 çıktı — çünkü vocab'da
`'[sayı]'` tokenı **yok** (yalnız `sayı` kökü, ×57.953). Korpus rakamları
**digit jetonlar** olarak duruyor (`('0'→'1') × 691` bağlamları); P5-A'nın
`[sayı]` görüntüsü eval-ref yoluna özgüydü. Anahtar-yok sınıfı sessiz
geçilmedi — beyan edildi.

## 4. (C) Sapma birikiminin ROUGE'a bedeli — DECOMP 0,9473 → 1,0

`kelimeler` (ECA:113) **casefold** yapıyor ⇒ harf-büyük-küçük farkları
ROUGE'a **hiç girmiyor** (T-0101 kayıt-düzeyi buyuk_harf %43,6 sınıfı
ölçütle görünümdü; hüküm etkisi 0). Token-düzeyi gerçek fark havuzu:
**309 fark kelime / 11.217 (%2,75)**:

| sınıf | adet | örnek |
|---|---|---|
| diger_kelime (çoğu **kesme/tırnak kaybı**) | 279 | `'şerit` → `şerit` · `keskin'liğine` |
| çözülmemiş-etiket (graph-dışı ek) | 30 | `POSS_3SG CASE_LOC_N` |

DECOMP boşluğu (0,0527) token-düzeyinde **iki sınıfın** birikimi:
kesme/tırnak kaybı + graph-dışı etiketler. `[sayı]`/UNK katkısı bu
örneklemde 0.

## 5. T-0101 + T-0102 birlikte değerlendirme (operatörün istediği)

**Temel teori sağlam (T-0101):** compile determinizmi 539/539 · eğitim
meta digest hizalaması birebir · vocab→lexicon %99,4 · roundtrip DECOMP
0,9473. **Sapma birikimi ÜÇ bağımsız katmanda (T-0102):**

| katman | ölçülmüş kitle | sahibi | çözüm sınıfı |
|---|---|---|---|
| 1. S2 türev zincirleri graph'ta yok | 13.064 lemma, recover %0,03 | **compiler (morphotactics graph)** | zincir ekleme — eklemeli-dil öğrenimini destekler; `tech<id>` BLOKLAR |
| 2. Külliyat UNK %2,38 — **İngilizce kalıntı ~%46** | 2,38 M jeton | **veri (külliyat derleme temizliği)** | İngilizce parçaların ayrışması/temizliği |
| 3. Kesme/tırnak kaybı + graph-dışı etiket | 309 kelime (%2,75; ROUGE boşluğunun token-düzeyi kaynağı) | **decompiler** | kesme işareti korunumu + graph zincirleri |

Sıralama önerisi (operatör kararı — ilan bu raporu bağlamaz): katman 3 en
küçük ve en ucuz (decompiler'da kesme işareti korunumu; graph zincirleri
S2'nin 1-2 ek baskın sınıfını compile-reachable yapar); katman 2 veri
işi (İngilizce kalıntının külliyattan ayrışması — en büyük tek kale);
katman 4 (S1 entity tasarımı, öneriniz) küçük ama kalıcı — ancak
külliyat-yansıması ölçülmeden kurulmamalı (S1 lemma'ları korpus'ta
görünmüyorsa id boşa gider).

## 6. Digest tablosu (betikle)

| artefakt | sha256 |
|---|---|
| `scratch/anka_t0102_sapma_ayrisma.py` | `875ebd1d…` |
| `scratch/anka_t0102_sapma_ayrisma.json` | `236a9162…` |
| `data/eval/anka_t0102_sapma_ayrisma_ilani_2026-09-23.md` | `9210919b…` |

*Kaynak çıpası: T-0101 sonda JSON `46bf1537…` · vocab 33.114
(`f9940a8d8…`) · lexicon 48.407 özgün lemma · bin 100 M jeton
(`9f987576…` meta). Zincirde DEĞİŞİKLİK YOK.*

## 7. Kapanış

**T-0102 KAPANDI (tanısal):** sapma birikimi üç katmana ayrıştı — S2
graph zincirleri (13.064 lemma), külliyat İngilizce kalıntı (UNK'nın
~%46'sı), decompiler kesme/tırnak kaybı (309 kelime). Önerinin
(`tech<id>`) hedef sınıfı ölçüldü: S1 %13,25 — küçük; külliyat UNK
kitlesinin gerçek kralı İngilizce kalıntı.