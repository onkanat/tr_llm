# ANKA_i1 · ÖĞRENME ORANI DENEMESİ — ön-kayıtlı tek değişken

**Damga:** 22 Eyl 2026 · **Bu belge KOŞUMDAN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.
**Önceki tur:** `data/eval/anka_i1_faz1_sonuc_2026-09-22.md` (kapı İLK segmentte düştü).

---

## 1. Neden bu deneme — ölçülmüş gerekçe

Önceki koşum `lr 2e-4` ile **500 adımda** Wikipedia CE'yi 3,5352 → **4,0003 (+%13,16)** yaptı ve
kapı düştü. O ölçüm **iki adayı eledi**:

| elenen | kanıt |
|---|---|
| **yetenek içeriği** | genel talimat verisiyle de **aynı hızda** bozuluyor (**+%2,63** vs T-0096'nın +%2,33 /100 adım) |
| **istem maskesi kusuru** (`train.py:115`) | bu koşumda maske **DOĞRU** uygulandı (`--no-pad-mask`) ve hasar yine oldu |

⇒ Kalan tek aday **öğrenme oranı** (ve tam-parametre güncellemesi).

**`2e-4` bu taban için TİPİK DEĞİL.** Tam ince ayar SFT için yaygın aralık **1e-5…2e-5**;
`2e-4` ön-eğitim/LoRA ölçeğidir. Bu değer T-0096'dan **aynen devralınmıştı** ve önceki koşum
o devralmanın **bedelini ölçtü**.

## 2. Tek değişken

**`--lr`: 2e-4 → 2e-5** (10× düşüş). Başka hiçbir şey değişmez.

| eksen | önceki koşum (düştü) | **bu deneme** |
|---|---|---|
| `--lr` | 2e-4 | **2e-5** |
| külliyat | `scratch/anka_i1_talimat.bin` (ceket hariç, `<EOS>` onarımlı) | **aynı** |
| taban · sözlük · blok · batch · rejim | `data/anka_a1r.pt` · 33.114 · 128 · 8 · `--no-pad-mask` | **aynı** |
| segmentler · tohum | 6 × 500 = 3000 adım · 42+i | **aynı** |
| **kapı** | CE ≤ 3,8887 her segmentte | **aynı (DEĞİŞMEZ)** |
| çıktı | `scratch/anka_i1_kos/` | **`scratch/anka_i1_lr2e5_kos/`** (önceki artefakt EZİLMEZ) |

## 3. İlan edilen okumalar — ÖLÇÜMDEN ÖNCE

| # | okuma | GEÇTİ ise | DÜŞTÜ ise |
|---|---|---|---|
| **1 (kapı)** | Wikipedia CE her segmentte | **≤ 3,8887** — koşum **6 segmenti tamamlar** | koşum durur; durduğu segment yazılır |
| **2 (zarf)** | `tutarsızlık` (nihai ckpt) | **< %10** (taban %100) | ≥ %10 |
| 3 (kayıt) | ezber · ROUGE · kesişim · B | hükme girmez | — |

**Karar anahtarı (önceden):**
* **1 VE 2 geçti** ⇒ **H-İ desteklendi**: talimat aşaması **kurulabilir**; Faz 2 (yetenek modülü
  `anka_i1` üzerinde) başlatılır — mimarinin son sınanmamış hücresi.
* **1 geçti, 2 düştü** ⇒ LM korunuyor ama zarf **bu bütçede** öğrenilmiyor ⇒ karar **bütçe
  eksenine** kayar (adım sayısı), lr ekseni **kapanır**.
* **1 düştü** ⇒ 10× lr düşüşü bile yetmiyor ⇒ kısıt **tam-parametre güncellemesidir**
  (parametre-verimli aşama gerekir) ⇒ lr ekseni **kapanır**.

## 4. Yan hipotez — İLAN EDİLİR ve SINANIR

Hasarın **`lr × adım` ile doğrusal** ölçeklendiği varsayımı (tek ölçüm noktasından
ekstrapolasyon, **iddia değil hipotez**): 2e-4'te +%2,63/100 adım ⇒ **2e-5'te +%0,263/100 adım**
⇒ 3000 adımda **≈ +%7,9** ⇒ kapı **geçilmeli**.

* Koşum **6 segmenti tamamlarsa** ⇒ doğrusallık **desteklendi**.
* **Erken durursa** ⇒ hasar lr'den **yavaş** düşüyor (doğrusal değil) ⇒ bu da yazılır.

## 5. Ölçümün sınırı (ilan edilir)

* **Optimizer taşınmıyor** (`--save-optimizer` kapalı, disk) ⇒ segmentler arası AdamW momentleri
  sıfırlanır. Her segment kendi içinde tam bir eğitimdir.
* **Kayıp ölçeği** `--no-pad-mask` rejiminde; `seg_*` loglarıyla kıyas **kurulamaz**.
* **Tek tohum zinciri** (42+i); lr dışında hiçbir şey taranmıyor, **tek deneme**.

## 6. Bu bir "ayar kovalama" DEĞİLDİR

* Önceki tur **olumsuz** yazıldı; kapı **düşürüldü**, eşik **değiştirilmedi**.
* Aday, ölçümün **kendisinden** daraltıldı (veri ve maske elendi, geriye lr kaldı) ve
  `2e-5` **literatürdeki tam-ince-ayar aralığından** seçildi, sonuç görülerek değil.
* **Üç dalın üçü de karar üretir** (§3) — hepsi bir ekseni **kapatır**.

## 7. Uygulama

```
venv/bin/python scratch/anka_i1_surucu.py --device mps --lr 2e-5 \
    --kos scratch/anka_i1_lr2e5_kos
→ kapanış: scratch/anka_i1_kapanis.py (üç sinyal + kosum_bekcisi)
→ zarf:    scripts/evaluate_carpenter_anka.py --model <nihai> --baseline data/anka_a1r.pt --ceket-ekseni
```

**DOKUNULMAZ:** `data/**` (donmuş) · `CLAUDE.md` · `scratch/t0096_*` · `scratch/t0097_*` ·
**`scratch/anka_i1_kos/` (önceki koşumun kanıtı — SALT OKUNUR)** · kapanmış
`data/eval/anka_r17…r34*` · `src/**`. `git add -A` yasak; commit yok.
