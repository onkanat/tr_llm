# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — yetkin temel (`seg_6`) üzerine modül

**İlan:** `data/eval/anka_r33_yetkin_temel_ilani_2026-09-22.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099h_zincir.log` · 1000 adım · 299 sn · `rc=0` · 09:56:36Z→10:03:20Z

---

## 1. Hüküm: birincil okuma "EKLENMEDİ" — içerik yine gelmedi

| # | okuma | ilan: eklendi ise | ölçülen (seg_6 → seg_6+modül) | sonuç |
|---|---|---|---|---|
| **1 (birincil)** | **kesişim** | ≥ %18 | **%10,00 → %10,00** | **EKLENMEDİ** |
| 2 | ROUGE-L | ≥ 0,348 | 0,3035 → 0,3063 (fark +0,0028 · **0,13 SE**) | **harf: KISMİ · gerçek: DEĞİŞMEDİ** |
| 3 | tutarsızlık | < %10 | %5,00 → %5,00 | **KORUNDU** |

⇒ **Modül yetkin bir temele içerik EKLEMEDİ.** İlan §3 gereği: **H-B güçlenir** (rank-sınırlı
delta bilgi enjekte edemiyor), ama **H-A tam elenmez** — test edilen şey "yetkin modele
ekleme"ydi, "yetkin tabana enjeksiyon" değil. H-A'nın tam elenmesi için **yetkin ama yeteneksiz**
bir temel gerekir (aday `seg_1`; bu ilanda koşulmadı).

**Okuma 3 önemli:** modül yetkin modeli **bozmadı** (tutarsızlık %5, ezber %0 sabit).

## 2. ⚠ BEKLENMEDİK VE BÜYÜK BULGU: modül Wikipedia'yı ONARDI

| | A CE | tabana göre |
|---|---|---|
| taban Anka `anka_a1r.pt` | 3,5352 | — |
| `seg_6` (tam ince ayar hasarı) | 6,2086 | **+%75,62** |
| **`seg_6` + modül** | **4,3902** | +%24,18 |
| **seg_6'ya göre değişim** | **−%29,29** | **onarılan hasar payı: %68,0** |

**B ekseni de düzeldi:** top-1 %41,28 → **%44,61** (+3,33 puan).

Ve bu **yetenek pahasına olmadı**: kesişim %10 → %10, tutarsızlık %5 → %5, ezber %0 → %0,
ROUGE 0,13 SE içinde sabit.

> **Bu, ilan edilen soru DEĞİLDİ** (soru "içerik eklenir mi"ydi, cevap "hayır"). Ama aynı
> koşumdan çıkan, ölçülmüş ve büyük bir yan bulgudur.

## 3. Mekanizma sınaması — hipotezim ÇÜRÜDÜ

**Hipotezim:** "modül, tam ince ayarın sürüklemesini **tabana geri çekerek** onarıyor."
**Ölçüm:** ağırlık uzayında Frobenius uzaklığı.

| | ‖· − taban‖_F |
|---|---|
| `seg_6` | 674,17 |
| `seg_6` + modül | **804,42** (**+%19,3**) |

⇒ **Modül tabandan UZAKLAŞTI, yaklaşmadı.** Yani onarım "geri dönüş" değil: rank-16 delta,
ağırlık uzayında tabandan **uzaklaşan** bir yönde Wikipedia'yı **düzeltiyor** — ve o yön
yetenek metriklerine **dokunmuyor**.

**Aday açıklama (hipotez, ölçülmedi):** `seg_6` T-0096'nın eğiticisiyle üretildi ve o eğitici
**MPS maskeli kayıp kusuruyla** koştu (maskeli konumlar paydaya tam katılıyor — bilinen kusur).
Yani `seg_6`'nın hasarı **kısmen bozuk bir hedefin** artefaktı olabilir. Modül ise
`train_module.py` ile **doğru maskeyle** (−100 gerçekten yok sayılıyor) eğitildi ⇒ aynı veride
doğru hedefe doğru atılan adım, bozuk hedefin sürüklemesini **kısmen geri alıyor** olabilir.
**Bu bir iddia değil, sıradaki ilanın adayıdır.**

## 4. Kendi ölçüm aracım bir kez yanlış cevap verdi (ve yakalandı)

İlk Frobenius ölçümüm **iki kolu da birebir aynı** buldu (674,1743 = 674,1743) — yani "modül
hiçbir şey değiştirmiyor" gibi. Sebep benim betiğimdeydi: LoRA katmanının etkin ağırlığını
(`W + ölçek·B@A`) yazdıktan **sonra**, aynı anahtarla ham `weight`'i tekrar yazıp düzeltmeyi
**eziyordum**. Fark edildi çünkü "sıfır fark" **fiziksel olarak imkânsızdı** (vakum kapısı
logit farkı 14,1 veriyordu). Düzeltilince gerçek fark 270,47 çıktı.
— Aynı sınıf: [[kendi-sondamin-kusuru-sahte-bulgu-uretir]].

## 5. Tablo

| koşum | temel | ROUGE-L | tutarsızlık | **kesişim** | ezber | A artış | `unutma_gec` |
|---|---|---|---|---|---|---|---|
| r30 · `anka_a1r` + modül | ön-eğitilmiş | 0,0696 | %98 | %1 | %28 | +%9,98 | ✓ |
| r32 · `anka_a1r` + modül · 3000 adım | ön-eğitilmiş | 0,1168 | %88 | %0 | %38 | +%13,35 | ✗ |
| *seg_6 tek başına* | *tam ince ayar* | *0,3035* | *%5* | *%10* | *%0* | *+%75,62* | *✗* |
| **r33 · `seg_6` + modül** | **tam ince ayar** | 0,3063 | %5 | **%10** | %0 | **−%29,29** | **✓** |

## 6. Bu ne demek — "çalışan temel + modül" arayışında

**İlk kez bir modül, tabanı BOZMADAN büyük bir onarım yaptı.** `seg_6` + **7,5 MB**'lık modül:

* yetenek **korunuyor** (kesişim %10, tutarsızlık %5, ezber %0 — hepsi seg_6 ile aynı),
* Wikipedia hasarının **%68'i onarılıyor** (6,2086 → 4,3902),
* ve unutma kapısı **geçiliyor** (`unutma_gec: True`; `--baseline` seg_6'nın kendisiydi).

Yani mimari **kapasite/bütçe/giriş tarafı** eksenlerinde bilgi **ekleyemiyor**, ama
**onarabiliyor**. İki iş ayrı: *ekleme* rank-sınırlı deltaya kapalı görünüyor; *onarım* açık.

**Sınır:** temel `seg_6`'nın kendisi 356 MB'lık bir **tam ince ayar checkpoint'i** — "modül
dostu" bir taban değil. Yani bu bir "çalışan **çift**" ama "çalışan **temel** (bir kez eğitilmiş)
+ modül" mimarisinin tam karşılığı değil.

## 7. Açık — sıradaki ölçümler

1. **Onarım tekrarlanabilir mi?** Farklı tohumla aynı koşum ölçülmedi.
2. **`seg_1…seg_5`'te de oluyor mu?** Onarım hasarın büyüklüğüyle mi ölçekleniyor?
3. **Aday açıklama (§3):** onarım, T-0096 eğiticisinin **maskeli kayıp kusurundan** mı geliyor?
   Ayrım yolu: aynı modülü **doğru maskeyle eğitilmiş** bir tam ince ayar checkpoint'i üzerinde
   koşmak — o zaman onarılacak "bozuk hedef artefaktı" olmamalı.
4. **H-A'nın tam elenmesi:** `seg_1` (tutarsızlık %24 · **kesişim %0**) temelinde aynı modül ⇒
   "yetkin ama yeteneksiz" tabana enjeksiyon sınanır.

## 8. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `modules/seg6_marangoz_1000.mod.pt` | `afdae58007411d277de2eecd286ff243942fb7ce7f11987b38d3cfd3b3f0a7ed` |
| `scratch/t0096_kos/seg_6.pt` (**SALT OKUNUR — değişmedi**) | `136dda76da419e281cf8b800a0e562381c5e9c7db8ed85136ac3ee4f19578b8b` |
| `data/eval/anka_r33_yetkin_temel_yetenek_2026-09-22.json` | `61420c5c2a1cbd98971083213b6de30f3d78c4e7fcd3dde69b0d57b6c7d20af8` |
| `scratch/t0099h_zincir.log` | `b1b40f60a840f06a9f69fcf46c467b36276d0bd93794d074b8c6d4a9ea022395` |

Modül meta: `37 katman · 1.869.216 parametre · adım=1000 · blok=128 · kayıp ilk 0,1537 →
son60 0,7826 ± 0,6207 · 7,5 MB`. Vakum **14,105** ≠ 0 · taban tensörü değişen **0**.

**Commit yok · `git add -A` kullanılmadı · donmuş kalıplara yazılmadı · `src/**` değişmedi.**
