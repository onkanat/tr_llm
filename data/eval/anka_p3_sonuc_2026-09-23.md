# P3 · AŞAMA 1+2 — ÜÇ KAYNAKLI YETENEK KOŞUMU SONUCU (23 Eyl 2026)

**Damga:** 23 Eyl 2026 08:16 (+03, betikten) · **rc=0** · Sentinel:
`BITTI 2026-09-23T05:15:47Z nihai=scratch/anka_p3_kos/seg_3.pt
sha256=7296aab14109…` · sonuc.json `durum=BITTI` — kapanış üç sinyali BİREBİR.
İlân: `anka_p3_ilani_2026-09-23.md` (koşumdan ÖNCE yazıldı, sha256 `48da9694…`).
Nihai model: **Anka P3** (`scratch/anka_p3_kos/seg_3.pt`, taban seg_2 → 6.000 adım).

## 1. Koşum kanıtı

* Süre **3.042,6 sn (~50,7 dk)** · 6.000 adım + 3 sonda · ~0,506 sn/adım
  (söndalar dâhil) · sandbox DIŞI MPS · H1 tek eğitici (0 tutucu).
* Kaynak dağılımı (ölçülmüş, pencere sayacı): wiki **9.602** / ceket **19.204** /
  SFT **19.203** = **%20,0/%40,0/%40,0** — ilanlı patern birebir.
* Carry: taban sidecar digest'i `ca8b007b…` ile birebir (CARRY kapısı geçti;
  101 parametre, adim=2000).
* L1: ilk kayıp 3,3903-bandı (devam koşumu — ln V imzası yok, model sağlam devraldı).
* **Held-out çıpası BİREBİR:** `c397eb08…` baş = son (V7 carve sızıntı kanıtı;
  koşum boyunca held-out 539 hiç okunmadı/yazılmadı).

## 2. Açık onarım beyanları (koşum 3 denemede tamamlandı)

1. **Disk-dolu çökmesi (1. deneme):** seg_1 opt sidecar yazımında
   `ios_base::clear` — disk %100. Onarım: `anka_p3_smoke/seg_1.pt + .opt.pt`
   ve `anka_p2_ince_ayr_kos/seg_1.pt + opt` silindi (**kayıt dosyaları korundu**;
   smoke ckpt CPU 50 adımdan türetilebilir) + sürücüye **ara-taşıyıcı silme
   kuralı** eklendi (yalnız SON segmentin `.opt.pt`'si korunur — ilanlı P2
   tasarımı).
2. **YALANCI TEPE (2. deneme, koşumu DURDURDU):** `tepe_karari`'daki
   `ezber_orani * 100.0` çarpımı — **alan zaten yüzde ölçeğinde**
   (`scripts/evaluate_carpenter_anka.py:513`: `100.0 * ezber / n`; ESIK_EZBER=10,0
   yüzde). seg_1'de 5,0 (%5) → 500,00 sanılıp DURDU. Onarım: çarpım kaldırıldı,
   kapı sınaması yeniden ölçüldü (**4 dal rc=0**: ezber 10,0 ⇒ DUR · 9,9 ⇒ geç).
   Kusurlu koşumun kayıtları kanıt olarak
   `scratch/anka_p3_yalanci_tepe_kayit/`'ta (sonuc.json `68b3cce7…`).
3. **3. deneme = TAM TEKRAR** (seed 42 deterministik; kayıp eğrisi ilk koşumla
   birebir: adım 1000'de 3,5408 vs 3,5386 — MPS gürültü bandında). Onarım
   koşum tekrarı SAYILMADI (sürücü kusuru; planın 1 onarım + 1 tekrar kolu
   disk çökmesinde kullanılmıştı).

## 3. Ölçüm — segment/sonda eğrisi (kanonik kap, n=100, seed 42, baseline=taban)

| segment | CE (ppl) | ROUGE-L | tutarsızlık | kesişim (F1) | ezber | LM bedeli A | TEPE |
|---|---|---|---|---|---|---|---|
| taban (Aşama 3.1) | 3,4438 | 0,0056 | %87 | %0 | %0 | — | — |
| seg_1 | 3,4994 (33,09) | **0,0955** | %41 | %0 | %2 | +%1,61 | geç |
| seg_2 | 3,4939 (32,92) | **0,1015** | %39 | %3 (0,0058) | %0 | +%1,46 | geç |
| **seg_3 (nihai)** | **3,4931 (32,89)** | **0,1014** | **%40** | **%2** | %2 | **+%1,43** | geç |

* TEPE kapısı üç segmentte de doğru hükmetti; ROUGE seg_2→seg_3'te düzleşti
  (−0,0001, gürültü altı) — tepe pratikte seg_2/3'ün ortasında, nihai ckpt =
  seg_3 (son geçen segment).
* **CE düz bantta korundu:** 3,4978→3,4939→3,4931 (tavan 3,8887; tabandan
  birikimli bedel **+%1,43** — T-0096'nın +%75'ine karşı **%10 iddiası ÖLÇÜLDÜ:
  GEÇTİ**).

## 4. Kıyas tablosu (kalibre eşiklere hüküm: kesişim 10,92 · ROUGE 0,35 · tutarsızlık 5,0 · ezber 10,0)

| eksen | Temel 2.0 zemin | P2 Aşama 3.1/3.2 (%0 noktası) | **P3 tepe (seg_3)** | T-0096 referans | eşik hükmü |
|---|---|---|---|---|---|
| ROUGE-L | 0,0056 | 0,0068 (gürültü) | **0,1014** | 0,3035 | ALTINDA (0,35) |
| tutarsızlık | %87 | %90 | **%40** | %5 | ALTINDA (5,0) |
| kesişim | %0 | %0 | **%2** | %10 | ALTINDA (10,92) |
| ezber | %0 | %0 | %2 | %0 | < %10 GEÇTİ |
| Wikipedia CE | 3,4438 | 3,4438 | **3,4931 (+%1,43)** | 6,2086 (+%75) | ≤ 3,8887 GEÇTİ |

## 5. Hüküm

**P3 KAPANDI — ve teşhis ölçüldü:**

* **"Daha çok veri (ceket carve)" hipotezi KISMEN DOĞRULANDI:** V7 carve geri
  dönüşü yetenek eksenini sıfırdan çıkardı — ROUGE **0,0056 → 0,1014 (18×)**,
  tutarsızlık **%87 → %40** (−47 pp), kesişim %0 → %2. Yetenek artık ölçülebilir
  bantta ama kalibre eşiklerin (ROUGE 0,35 · tutarsızlık 5,0 · kesişim 10,92)
  tamamının ALTINDA.
* **"Aşama 1 altyapısı T-0096'nın LM bedelini +%75'ten +%10'a indirir" iddiası
  ÖLÇÜLDÜ: GEÇTİ** (+%1,43). P2 Aşama 1'in maske/scheduler/clip/carry onarımı +
  Temel 2.0 tabanı, ceket r18 %40 payıyla bile LM'i tavanın altında tuttu.
* **İlanlı beklenti karşılaştırma:** ROUGE 0,1-0,3 bandı → 0,1014 (**bandın alt
  ucu**); tutarsızlık %20 bandı → **%40 (band üstü)**; CE → GEÇTİ.
* **Döngü-sonlandırma tablosu kolu:** kaplar geçti (ezber + CE), kesişim/ROUGE
  tavanın çok altında ⇒ **SONUÇ YAZILDI**; ölçüt ancak yeni insan tavanı
  ölçümüyle değişir. Dallar koşulmadı (CE aşmadı, ROUGE 0,05 üstünde tepe yok).
* **Kalan teşhis (yeni ilan konusu):** ceket r18 ile 1,6 epoch ROUGE'u 0,10
  bandına getirdi; T-0096'nın 0,30 bandı ~16 epoch'taydı — **ceket epoch arzı**
  (daha çok adım TEK BAŞINA kapandı: T-0097; bu koşumda seg_2→seg_3 ROUGE düz)
  kalan boşluğun en olası kaldıracı; kesişim %2-4'ün n=100 örneklemde kayıt
  düzeyi olduğu unutulmamalı (±2 kayıt = gürültü).

## 6. Digest tablosu (betikle hesaplandı)

| artefakt | sha256 |
|---|---|
| `scratch/anka_p3_kos/seg_3.pt` (nihai Anka P3) | `7296aab14109cb47ddb0b08b7c32dba6da86e5ad3e16d50d037131182e0bdd3a` |
| `scratch/anka_p3_kos/sonda_seg_3.json` (kanonik kap) | `2ced71a855f04c940567bf790541a6a93eb31fb40e92df8353b671a309a0a51f` |
| `scratch/anka_p3_kos/sonuc.json` | `216ff336f14a0c7322778935c0329589fa42b61ca7a1922f8640c4e83e1e8572` |
| `scratch/anka_p3_kos/segments.jsonl` | `c3d9c94eae0de86dda4f812aac55b7bcea33ede3981a7901ebaecced3f68e269` |
| `data/eval/anka_p3_ilani_2026-09-23.md` (ilân) | `48da9694959f12258280225574d130c620f49e658610e6a081be8665dfea125a` |
| `scratch/anka_p3_yalanci_tepe_kayit/sonuc.json` (kusurlu-tepe kanıtı) | `68b3cce74f262fe5892ee2e8119e2af79766543be6df534c2d779f77060e13e5` |
| `scratch/anka_p3_yalanci_tepe_kayit/sonda_seg_1.json` | `bee4299569e7243a8a7b3d889cac92125a9b3f58ce809e6de76b96bb16e73dd9` |
| ceket r18 bin (sonuc.json'dan) | `c9d964d98777358d6acb3db2c04d18a4e9b5bf515f1d8125077ce097857dfb78` |
| taban seg_2.pt (sonuc.json'dan) | `ca8b007b9601dc71…` (Aşama 3.1 nihai ile birebir) |
| held-out 539 (baş = son) | `c397eb08218937ea000cc6120f42a8783e1dcab7948c70b33537af74b50d096e` |

*Nihai digest iki kanaldan çıpalı: sentinel + sonuc.json aynı `7296aab1…`'i yazıyor.*

## 7. Kalan (operatör kararı bekliyor)

* P3 kayıtlarının commit/push onayı (sürücü + ilan + bu rapor + koşum kayıtları;
  `git add -A` yasak — dosya listesiyle).
* Disk: 3.0G'den seg_3 kayıtlarına düştü; koşum ckpt'leri (~1,35G) sonraki koşum
  tabanı/tanı arzı — korunması önerilir, temizlik onayı gerekir.
* Sıradaki ilan (P3 teşhisinden): ceket epoch arzı ölçümü (r18 külliyatla
  ~10-16 epoch) — kesişim/ROUGE tavanına karşı.