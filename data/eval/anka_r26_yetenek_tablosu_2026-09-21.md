# Anka · Taban mı, Modül mü? — Adım Artırma ve Yetenek Tablosu (21 Eyl 2026)

**Operatör sorusu (birebir):** *"Adım sayısını artıralım ve yeteneği ölçelim; temel model Anka mı,
eski kristal mı?"*

**İki cevap, ikisi de ölçümle:**

1. **Temel model ANKA'dır** (`data/anka_a1r.pt`). "Kristal" yok — `data/kristal_model.pt`
   mevcut değil ve ağaçta `kristal*.pt` **hiç** bulunmuyor. Üç modül dosyasının da kaydettiği
   `base_sha256` = `b93cc1cd…` = `data/anka_a1r.pt`'nin **ölçülen** tam digest'i; yükleyicinin
   digest kapısı yanlış tabanı reddettiği için üç başarılı yükleme **mekanik kanıttır**.
2. **Adım artırmak kapıyı getirmedi.** A ekseni (unutma) **+%13,02 → +%21,23** *kötüleşti*;
   "donuk taban sürüklenmeyi sınırlar" **hipotezi ÇÜRÜDÜ** (aşağıda).

---

## 1. Ana tablo — tek ölçüm kabı, tek cihaz (mps), n=100, max_new=128

Sayıların **hepsi** kanonik `scripts/evaluate_carpenter_anka.py` tarafından üretildi; modül
satırlarında betik *değiştirilmedi*, `model_yukle` sarıldı (`scripts/modul_ile_olcum.py`).

| koşum | adım | A CE | **A artış** (eşik ≤ +%10) | B top-1 | **B düşüş** (eşik ≤ 5,0) | ezber (<%10) | tutarsızlık (<%5) | **ROUGE-L** (≥0,35) | kesişim (≥%80) |
|---|---|---|---|---|---|---|---|---|---|
| **TABAN** `anka_a1r.pt` | 0 | 3,5352 | — (referans) | %49,56 | — | %0,00 ✓ | %100,00 ✗ | **0,0000** ✗ | %0,00 ✗ |
| T-0096 **seg_1** · tam ince ayar | 1000 | 4,3592 | +%23,31 ✗ | %49,48 | −0,08 ✓ | %20,00 ✗ | %24,00 ✗ | 0,1097 ✗ | %0,00 ✗ |
| **MODÜL @1000** · `marangoz_mixr4` | 1000 | 3,9953 | **+%13,02** ✗ | %47,15 | −2,42 ✓ | %0,00 ✓ | %100,00 ✗ | 0,0258 ✗ | %0,00 ✗ |
| T-0096 **seg_6** · tam ince ayar | 6000 | 6,2086 | +%75,62 ✗ | %41,28 | −8,29 ✗ | %0,00 ✓ | %5,00 ✗ | 0,3035 ✗ | %10,00 ✗ |
| **MODÜL @6000** · `marangoz_mixr4_6000` | 6000 | 4,2858 | **+%21,23** ✗ | %42,11 | −7,45 ✗ | %0,00 ✓ | %72,00 ✗ | 0,0505 ✗ | %4,00 ✗ |

**Hiçbir satır `ceket_ekseni_gec` = True değil.** Taban da, iki modül de, tam ince ayarın altı
segmenti de dört yetenek kapısının **hiçbirini** geçmiyor.

### 1a. Tabanın yetenek sıfır noktası — ilk kez ölçüldü

`anka_a1r.pt` bugüne kadar kanonik yetenek ekseninde **hiç** ölçülmemişti. Sonuç:

* ROUGE-L **birebir 0,0000** (± 0,0000, 100 örnekte tek bir örtüşme bile yok)
* tutarsızlık **%100** · kesişim **%0** · ezber %0
* üretim çıktısı dejenere: `, [?], [?], [?], …` (128 jetonun tamamı yer tutucu)

Yani taban, yetenek ekseninde "zayıf" değil, **hiç yok**: soru sorulunca cevap üretmiyor.
Bu, T-0096'nın seg_1'deki ROUGE 0,1097'sinin **tabandan değil eğitimden** geldiğini kanıtlar.

### 1b. Oracle katmanı — ölçüt ayırt ediyor mu?

| oracle | değer | hüküm |
|---|---|---|
| kimlik (referans ↔ kendisi) | **1,0000** | ✓ tavan tutuyor (`:425` fail-closed) |
| distraktör (komşu referans) | 0,0848 | ✓ eşik 0,35'in altında (`:427` fail-closed) |
| **sabit-tahmin** (eğitimden tek cümle) | **0,0780** | **taban ve iki modül bunun ALTINDA** |

**Keskin bulgu:** modül @1000 (0,0258) ve @6000 (0,0505), "her zaman tek bir sabit cümle söyle"
tabanının (0,0780) **altında** kalıyor. Modül yalnız "öğrenememiş" değil — **önemsiz tabanın da
altında**. Tam ince ayar seg_6 (0,3035) sabit tabanın 3,9 katı; modül ise 0,65 katı.

---

## 2. Adım artırma: hipotez ÇÜRÜDÜ

`data/eval/anka_r20_modul_marangoz_2026-09-21.md:29-31` ölçülmemiş tek şeyi açıkça yazmıştı:

> *"Modül 6000 adım koşulsaydı A ekseni nereye giderdi **ölçülmedi** — taban donuk olduğu için
> sürüklenmenin rank-16 deltasıyla sınırlı kalacağı **hipotezi** test edilmedi."*

**Test edildi. Hipotez yanlış.**

| | tam ince ayar (aynı veri/adım/tohum) | modül | hasarın kesilen payı |
|---|---|---|---|
| **1000 adım** | +%23,31 | +%13,02 | **%44,2** |
| **6000 adım** | +%75,62 | +%21,23 | **%71,9** |

Modül **göreli** olarak iyi: 6 kat adımda tam ince ayar +%75,62'ye tırmanırken modül +%21,23'te
kalıyor ⇒ hasarın %71,9'u kesiliyor. **Ama kapı mutlaktır (≤ +%10) ve modül 6 kat adımda
+%13,02'den +%21,23'e çıktı** — donuk taban sürüklenmeyi *sınırlamıyor*, yalnızca *yavaşlatıyor*.
Rank-16 deltası 6000 adımda hâlâ büyümeye devam ediyor.

### 2a. Yeteneğin kendisi de adımla artıyor — ama uçurum büyüyor

| adım | modül ROUGE-L | tam ince ayar ROUGE-L | modülün yakaladığı pay |
|---|---|---|---|
| 1000 | 0,0258 | 0,1097 | **%23,5** |
| 6000 | 0,0505 | 0,3035 | **%16,6** |

Modül, **stabiliteyi yeteneğin karşılığında satın alıyor** ve takas oranı adımla **kötüleşiyor**
(%23,5 → %16,6). 6000 adımda modül, tam ince ayarın ROUGE'sinin **altıda birini** alıyor.

### 2b. Ezber ekseninde modül gerçekten farklı

Tam ince ayar seg_1'de eğitim satırlarının **%20'sini** ezberliyor; modül **iki adım sayısında da
%0,00**. Modül hiç ezberlemiyor — ama ezberlememek yetenek demek değil (ROUGE 0,0505).

### 2c. Kayıp ↔ unutma birlikte hareket etti (T-0096 deseninin aynısı)

| | son60 kayıp | A artışı |
|---|---|---|
| modül @1000 | 5,0598 ± 0,2714 | +%13,02 |
| modül @6000 | 4,0100 ± 0,3323 | +%21,23 |

Kayıp **düştü** (5,06 → 4,01), unutma **arttı**. Yani "daha iyi uyum ⇒ daha çok unutma" —
T-0096'nın tam ince ayarda bulduğu *"ayrım yok"* deseninin modüldeki tekrarı. Modül bu
bağıntıyı **kesmiyor**, yalnızca eğimini azaltıyor.

---

## 3. Kabın doğrulanması (kendi kendini denetleme)

| kontrol | sonuç |
|---|---|
| **Çapraz doğrulama:** modül @1000, `scripts/modul_olcum.py`'nin kaydettiği +%13,02'yi **birebir** üretti (13,0161) | ✓ iki bağımsız kap aynı sayıyı verdi |
| **Baseline tuzağı:** `--model` = `--baseline` = `anka_a1r.pt`; sarma yalnız ilk çağrıya takıldı | ✓ `A_ekseni_taban` = 3,5352 = bağımsız taban koşumunun CE'si |
| **Vakum kapısı:** takma öncesi/sonrası logit farkı | ✓ 9,031 (1000) · 10,926 (6000) — **≠0**, modüller canlı, no-op değil |
| **Tohum tekrarı:** iki modül de ilk kayıpta **tam 6,8004** | ✓ aynı tohum + aynı takma noktası |
| **Taban dokunulmazlığı:** `base_sha256` başta=sonda | ✓ `b93cc1cd…` değişmedi |
| **Kayıtlı `ornekler` filtrelenmiş değil** (`:464` `if i < 5`) | ✓ boş `input` **tasarım gereği**: 343/539 satırda `input` boş ama `instruction` dolu ⇒ kusur değil, ölçüldü |

---

## 4. Bu turda bulunan kusurlar

| # | kusur | durum |
|---|---|---|
| 1 | `train_module.py` blok boyutunu **sessizce 64'e düşürüyordu** (`mix_r4.bin.meta.json` `block_size`'ı `mix_parameters` İÇİNE koyuyor) ⇒ 128'lik külliyat 64'lük pencereyle eğilir, kayıt da 64 yazar, uyumsuzluk **görünmez** | **KAPATILDI** — fail-closed kapı; `train_module.py`'nin repo genelinde **çağıranı yok** (grep boş) ⇒ kimseyi öldürmez |
| 2 | Aynı tuzak `train.py:135`'te (`if "block_size" in meta` — yalnız üst düzey) | **BEYAN EDİLDİ, DÜZELTİLMEDİ** — kapalı kayıtların eğiticisi; 9 karma külliyatın meta'sı iç içe şema kullanıyor |
| 3 | Sarma kendini çağırıyordu (`RecursionError`, 995 seviye) | **KAPATILDI** — orijinal `model_yukle` sarmadan **önce** yakalanıyor |
| 4 | İlk koşum blok 64 ile başlatıldı (1000'lik modül 128) ⇒ kıyaslanamaz | **KAPATILDI** — zincir durduruldu, `lsof` ile yetim süreç denetlendi (temiz, yarım modül yok), 128 ile yeniden koşuldu |

**Kusur 1 ve 2'nin kapalı kayıtlara etkisi: YOK (ölçüldü).** T-0094 ilanının kanonik komutu
(`data/eval/anka_r17_marangoz_ilani_2026-09-20.md:69`) `--block-size 128`'i açıkça geçiyor;
`scratch/t0097_marangoz.py:75` de `BLOCK = 128`.

---

## 5. Artefakt digestleri (TAM sha256)

`modules/*.pt` `.gitignore:12` (`*.pt`) kapsamında ⇒ git'te görünmez; kanıt kanalı **bu tablo**.

| dosya | boyut | sha256 |
|---|---|---|
| `data/anka_a1r.pt` (taban) | 373,7 MB | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `modules/marangoz_mixr4.mod.pt` | 5,1 MB | `2ef7d59ce586265a9aef32fc110b11c04c38007cd11bccb36c76856e18db1825` |
| `modules/marangoz_mixr4_6000.mod.pt` | 5,1 MB | `25fb7f2924cb48612511b069bc6babb9f087f356fb677bf47c5c9211e9b6a91b` |
| `data/eval/anka_r23_taban_yetenek_2026-09-21.json` | 5,4 KB | `9373ce510f3143348687811bf28c6ae3b54e852b5dcd22c28a6f0ffb44938fc8` |
| `data/eval/anka_r24_modul1000_yetenek_2026-09-21.json` | 6,5 KB | `0e9c67160c5e5709cc75ce660d01cb9a20335555907d9f6a7bbd1f191078bc1a` |
| `data/eval/anka_r25_modul6000_yetenek_2026-09-21.json` | 6,3 KB | `6e8c8ea3fbd9b719805b746132c5221b5e6d2d75cbdf74f1da0d3240f193beef` |
| `data/eval/anka_r24_modul1000_yetenek_2026-09-21.modul.json` | 4,6 KB | `e4ceb2e3af682a7a12649a92f04d6e945da3be2e091666950ea08b9d1800edc8` |
| `data/eval/anka_r25_modul6000_yetenek_2026-09-21.modul.json` | 4,7 KB | `b8c53064017ac880a14674d74408249d68510bd36ff98eed73d4ffbbb66dfaf9` |
| `scratch/t0099_zincir.log` (koşum günlüğü) | — | `198ee898aaf186a259f510f81340961a02bcd27d69833b3096b468b8534dda31` |

Koşum: 4 faz, `rc=0`, 13:26:58Z → 14:14:14Z. Eğitim 6000 adım · 1856 sn · blok 128 · batch 8 ·
lr 2e-4 · seed 43 · modül 1.327.104 parametre (%1,401) / 36 katman.

---

## 6. Hüküm ve açık

**Hüküm:** Ne taban ne modül, ne 1000 ne 6000 adım — **yetenek kapısı geçilmedi**. Modül
unutmayı göreli olarak kesiyor (6000 adımda hasarın %71,9'u) ama **mutlak eşiği geçmiyor**
(+%21,23 > +%10) ve karşılığında yeteneğin **altıda birini** alıyor. Modül @6000'in çıktısı
sabit-tahmin tabanının bile altında.

**Ölçülmüş ama açıklanmamış:** modül neden 1000→6000 adımda hem daha çok unutuyor hem de
yetenekte tam ince ayarın gerisinde kalıyor? Aday eksen **kapasite**: r=16 / 36 katman,
1,33 M parametre. 6000 adımda A artışının hâlâ tırmanıyor olması "kapasite yetersiz, delta
tabanı bozuyor" ile "kapasite yeterli, veri/adım oranı yanlış" arasında **ayırt etmiyor**.

**Açık kaldıraçlar (henüz denenmedi, sırayla):** (a) `r` **düşürmek** (8/4) — kapasiteyi
kısarak unutmayı sınamak; (b) replay oranını yükseltmek (şu an `replay_every: 4` ≈ %25);
(c) modülü **hedef katmanlardan** daraltmak (şu an 6 hedef × 6 blok).
Ölçüm sonrası kusur tasarıma göre onarılmaz (T-0067) ⇒ bunlar **ön-kayıtlı tek deneme** olmalı.

**Yapılmayan:** commit yok · `git add -A` kullanılmadı · donmuş yollara yazılmadı ·
`CLAUDE.md` dokunulmadı.
