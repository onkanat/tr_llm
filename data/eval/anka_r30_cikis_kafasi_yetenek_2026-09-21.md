# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — YETENEK EKSENİ: çıkış kafası

**İlan:** `data/eval/anka_r30_cikis_kafasi_ilani_2026-09-21.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099e_zincir.log` · 1000 adım · 301 sn · `rc=0` · 19:29:28Z→19:36:11Z

---

## 1. Hüküm: birincil okuma ATEŞLEDİ — ama kazanç **biçim**, içerik değil

| # | ilan edilen okuma | eşik | ölçülen | sonuç |
|---|---|---|---|---|
| **1 (birincil)** | ROUGE-L | ≥ 0,060 | **0,0696** (fark +0,0505 · **5,3 SE**) | **ATEŞLEDİ** |
| 2 | tutarsızlık | < %60 | **%98,00** | ateşlemedi |
| 3 | ezber | ≥ %5 | **%28,00** | **ATEŞLEDİ** |

İlan edilen kurala göre: **çıkış kafası bağlayıcıdır.** Çıkış-kafası ekseni **kapanmaz**.

**Ama üretimleri okumadan bu hüküm yanlış anlaşılır.** İlk üç örnek:

```
idx 1  üretim: 'teknik çözüm:'                     rouge 0,4000  tutarsız
idx 2  üretim: 'teknik çözüm: teknik çözüm:'       ezber  rouge 0,1600
idx 3  üretim: 'teknik çözüm: teknik çözüm:'       ezber  rouge 0,0000
referanslar: "Teknik Çözüm: Bu işlem için 'kalınlık makinesi' kullanılmalıdır." …
```

⇒ Model **zarfı** üretiyor (`teknik çözüm:`), sonra duruyor ya da zarfı tekrarlıyor.
**ROUGE kazancı paylaşılan zarftan geliyor**; **ezber**, zarfın eğitim 4-gramlarına
uymasından. Üç okumanın ikisi "bağlayıcı" dedi, ama **işaret ettikleri şey**
(bkz. ilan §4: *"görev öğreniliyor"*) **gerçekleşmedi**.

**İçeriği ölçen tek metrik kımıldamadı:** `kesişim` **%0 → %1** (eşik **%80**).

## 2. Doğru hüküm

> **Çıkış kafası, BİÇİMİN kazanılmasının bağlayıcı kısıtıydı. İÇERİĞİN değil.**

Bu, `T-0095`'in arka kapı bulgusuyla aynı desendir (*"biçim öğrenilmiş, içerik değil"*) —
orada da model zarfları üretip içerik üretemiyordu.

## 3. Tablo

| koşum | katman | ROUGE-L | tutarsızlık | **kesişim** | ezber | A artış | B düşüş |
|---|---|---|---|---|---|---|---|
| taban | — | 0,0000 | %100 | %0 | %0 | — | — |
| r28 · r=16 varsayılan | 36 | 0,0191 | %100 | %0 | %0 | +%6,08 ✓ | +0,28 ✓ |
| r29 · r=64 kapasite | 36 | 0,0248 | %100 | %0 | %0 | +%6,19 ✓ | −1,78 ✓ |
| **r30 · +`lm_head`** | **37** | **0,0696** | %98 | **%1** | **%28** | **+%9,98 ✓** | +1,51 ✓ |
| *tam ince ayar seg_1 (1000 adım)* | *tümü* | *0,1097* | *%24* | *%0* | *%20* | *+%23,31 ✗* | *−0,08 ✓* |
| *tam ince ayar seg_6 (6000 adım)* | *tümü* | *0,3035* | *%5* | *%10* | *%0* | *+%75,62 ✗* | *−8,29 ✗* |

**Modülün tam ince ayara oranı 1000 adımda: %17 → %63.** Çığır açıcı bir sıçrama — ama
sıçrayan şey **zarf**.

Çıpalar: sabit-tahmin **0,0780** · distraktör 0,0848 · kimlik 1,0000 · eşik 0,35.
r30 (0,0696) **hâlâ sabit-tahmin tabanının ALTINDA** — "her zaman tek cümle söyle"nin bile
gerisinde.

## 4. Unutma kapısı: kıl payı, ve gürültünün İÇİNDE

`unutma_gec: True` — **ama**:

| eksen | ölçülen | eşik | pay | yarı-genişlik |
|---|---|---|---|---|
| A artış | **+%9,9821** | +%10,0 | **0,0179 puan** | ±0,1218 ⇒ pay **0,147 kat** |
| B düşüş | +1,5067 | 5,0 | 3,4933 puan | ±1,95 ⇒ pay 1,79 kat |

A birincil hükmü şudur: modül **kapıyı geçti** (kural öyle diyor), ama **geçiş yarı-genişliğin
0,147 katıdır** ⇒ "geçti" demek "eşiğin altında olduğu gösterildi" demek **değildir**.
Bu, biçimi öğrenen modülün unutmayı da artırdığı yönünde bir işarettir: **r30, yetenek
kazancı ile unutmanın ilk kez aynı yöne gitmeye başladığı koşumdur.** (Yön okuması
yapılmıyor — bkz. `gurultuden-kucuk-farkin-isareti-yok`.)

## 5. Elenen ve açılan

**ELENEN (kalıcı):** kapasite (`r` 16→64, ×4 — r29'da elendi).
**AÇILAN:** çıkış kafası **biçim** için bağlayıcı. Yani modül mimarisinin **eksik parçası
bulundu**: yalnız attn/MLP sarmak, zarfları öğrenmeye yetmiyor.

**AÇIK KALAN (ilan edilmemiş):** **içerik** neden kazanılmıyor? `kesişim %1` (eşik %80) ve
`tutarsızlık %98` bunu söylüyor. Aday eksenler: **gömme katmanı** (`embedding` hâlâ donuk —
`lm_head` açıldı ama giriş tarafı kapalı), adım sayısı (1000 adım ≈ 1 epoch; ceket meta'sının
kendi formülü **2301** adım diyor), ve veri (11.708 kayıt).

## 6. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `modules/marangoz_lmhead_1000.mod.pt` | `451b06192576bc44f9c3a2a22c4e6ff923ddb9e0d3de3e50a83c784b4e73862b` |
| `data/eval/anka_r30_cikis_kafasi_yetenek_2026-09-21.json` | `9f7a6652f37148c0dbf891acc3fe042f7e2fcb2658d2a47f663f7f3998515229` |
| `data/eval/anka_r30_cikis_kafasi_yetenek_2026-09-21.modul.json` | `96c482096469aba96f139605460b0396ff2ccd3c0adce70d623304d90b512057` |
| `scratch/t0099e_zincir.log` | `bfa2031f7459f1266e0da28342f08c08159d9e09d6d0bcd53cf8ff42704564ad` |

Modül meta: `adim=1000 · r=16 · alpha=32 · 37 katman · 1.869.216 parametre (%1,9615) ·
hedefler …,lm_head · blok=128 · kayıp ilk 6,2606 → son60 3,8178 · 7,5 MB`.
Vakum kontrolü **15,218** ≠ 0 · taban `b93cc1cd…` başta=sonda · değişen tensör **0**.

**Commit yok · `git add -A` kullanılmadı · donmuş yollara yazılmadı.**
