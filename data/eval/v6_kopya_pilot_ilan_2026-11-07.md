# İLAN — KOPYA-HATTI PİLOTU (v6 hazırlığı): 600 adım, kopya-formatlı pencereler (2026-11-07)

**Durum:** PİLOT — koşumdan ÖNCE donmuş karar kuralı. Tam koşum bu pilotun hükmüne bağlı.
**Amaç:** RAG koşullandırmasının belgeyi kopyalamadığı ölçüldü (4-gram örtüşme **0/34**;
`scratch/rag_fide_tani.py`, 07.11.2026). Bu pilot tek soruyu yanıtlar: **kopya-formatlı
ön-eğitim pencereleri kopya devresini bu ailede KURUYOR MU?**

## 1. Veri (üretildi — `scripts/build_copy_korpus.py`)

- Pencere biçimi **çıkarım sözleşmesinin ta kendisi**: `json{instruction, input:"<BELGE> doc </BELGE> sorgu", output: span}` + `<EOS>`, ≤128 jeton.
- Cevap = **belgenin birebir jeton dizisi** ve pencerede **iki kez** geçer (kopya sinyali); test zorlar (`tests/test_copy_korpus.py`, 10 test).
- Karışım: her 8 özgün 128-blokta **2 kopya penceresi** (%20 blok payı).
- Sızıntı: val 8-gram kapısı (satır düzeyi) + %2 hash-tutulan satırlar **yalnız K4 seti** için.
- K4 ölçüm seti: `data/eval/k4_kopya_tutulan.jsonl` (200 örnek; hiçbiri kopya penceresi olarak kullanılmadı).

## 2. Pilot tarifi (donmuş)

```
venv/bin/python scripts/train_v4_ogrenci.py \
  --init-model data/anka_ogrenci_v5.pt \
  --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
  --pencere-bin data/kulliyat_v3_copy.bin \
  --kl-kip sft --kl-beta 0.2 --pencere-kayip plain --pencere-pad-maske \
  --peak-lr 1.5e-4 --min-lr 1e-5 --warmup 200 \
  --batch 16 --accum 2 --pencere-satir 10 \
  --toplam-adim 600 --save-every 200 \
  --save-path scratch/v6_pilot.pt --allow-frozen-write \
  --history-path scratch/v6_pilot_history.json \
  --hukum-path scratch/v6_pilot_hukum.json
```
- Öğretmen = init = v5 (öz-çapa; kazancı geri almamak).
- `--pencere-pad-maske` ZORUNLU: kopya pencereleri 128'e PAD ile dolar; maskesiz model PAD yazmayı öğrenir (TDD: `test_pencere_pad_maske_pad_konumlarini_paydadan_cikarir`).
- Süre beklentisi: ~3,2 s/adım → **~35 dk**.

## 3. KARAR KURALI (pilot bittikten sonra değişmez)

Ölçüm: `scripts/k4_kopya_olcum.py --model scratch/v6_pilot.pt --tutulan data/eval/k4_kopya_tutulan.jsonl --n 100`
(Taban: aynı alet, `--model data/anka_ogrenci_v5.pt` — beklenen ≈0.)

| `kopya_orani` (çıktıda belgeden ≥1 4-gram) | HÜKÜM |
|---|---|
| **≥ 0,30** | **KOPYA DEVRESİ KURULUYOR** → tam koşum ilanı (1 epok) |
| 0,10 – 0,30 | Kısmi: kopya payını %20 → %27,3 yap (grup 8, kopya 3) ve pilotu TEKRARLA |
| < 0,10 | Tarif yetersiz: teşhis raporu (kopya devresi kurulmadı) — tam koşum YOK |

**K1 sigortası:** her nöbette SFT val CE ≤ 2,45 (v5 2,2219). Aşılırsa koşum durur.
**K2/K3 pilotta ÖLÇÜLMEZ** (600 adım temsili değil; tam koşum kapıları).

## 4. Dürüst risk beyanı

- Kopya **SFT ile değil ön-eğitimle** kurulur (P1 dersi); bu pilot onu test ediyor.
- %20 kopya payı okuryazarlığı bozabilir — K1 nöbeti ve tam koşumda K2/K3 kapıları yakalar.
- Kopya öğrenilse bile **K4'ün gerçek RAG doc'larında** (200-300 morfem, prompt >128) çalışması ayrı iş: retrieval-taraflı chunking gerekir (≤45 morf parçalar). Pilot yalnız devrenin varlığını ölçer.

## 5. PİLOT-1 SONUCU (600 adım, 22 dk, 07.11.2026) — K4=0 · HÜKÜM: TAM KOŞUM YOK

| Ölçüm | v5 (taban) | pilot-1 (600 adım) |
|---|---|---|
| K4 `kopya_orani` (100 tutulan örnek) | **0,00** | **0,00** |
| K4 `span_tam_orani` | 0,00 | 0,00 |
| **eğitim penceresinde** cevap bölgesi CE | 3,912 | **3,519** (−0,39) |
| aynı pencerede belge bölgesi CE | 6,819 | **2,383** (−4,44) |
| eğitim penceresinde kopyalayan | 0/10 | 0/10 |
| K1 SFT val CE | 2,2219 | 2,3236 (+0,10; eşik 2,45) |

**Donmuş kural işledi: 0,00 < 0,10 → tam koşum YOK.** Ama teşhis "tarif ölü" DEĞİL:
model **formatı öğrenmeye başladı** (belge bölgesi CE 6,82 → 2,38; cevap bölgesi 3,91 → 3,52)
ama 600 adımda (≈12.000 kopya penceresi) kopya hizalaması kurulmadı; greedy üretim hâlâ
soru metnini ve `</INPUT> <OUTPUT>` zarfını tekrar ediyor (yapısal ezber sürüyor).

**Karar: aynı tarifle 2.500 adımlık PİLOT-2** (tek değişken = adım sayısı; ~1,6 sa).
K4 ≥ 0,30 ise tam koşum ilanı; yine < 0,10 ise kopya payı deneyleri (grup 8 → kopya 3) ve
gerekirse "önce yalnız-kopya ısınma" (curriculum) ilanı.

**Beyan (ölçek değişimi):** nöbet avlusunun pencere-CE'si yeni .bin'den örneklendiği için
v5 ile aynı ölçekte DEĞİL (v5 3,3197 → pilot başlangıcı 4,9797; 25 pencerenin ~%20'si taze
kopya formatı). **K2 karşılaştırması resmi `kulliyat_v2_val.bin` ile yapılır — bozulmadı.**

## 6. PİLOT-2 (2.500 adım, 100 dk) — K4 yine 0,00 · TEŞHİS: TEKRAR YOK

| Ölçüm | v5 | pilot-1 (600) | pilot-2 (2.500) |
|---|---|---|---|
| K4 kopya_orani | 0,00 | 0,00 | **0,00** |
| eğitim penceresinde cevap CE | 3,912 | 3,519 | **3,326** |
| eğitim penceresinde belge CE | 6,819 | 2,383 | **1,777** |
| eğitim penceresinde kopyalayan | 0/10 | 0/10 | **0/10** |
| K1 SFT val CE | 2,2219 | 2,3236 | 2,2858 |

**Kök neden (ölçüm):** 2.500 adım × 20 pencere/adım = 50.000 pencere çekilişi; kopya payı
%20 → **~10.000 kopya penceresi, her biri ~1 kez** (bin'de 56.904 BENZERSİZ pencere var).
Model kopyayı **tekrar görmeden** öğrenemez; üstelik `<OUTPUT>` konumunda SFT çapasının
kısa-etiket önseli (`TENSE_PROG` gibi) hâlâ baskın.

## 7. PİLOT-3 — TEKRAR (curriculum) testi · DONMUŞ KURAL

`scratch/kopya_isinma.bin`: **3.000 pencere, %100 kopya**. Koşum: v5'ten, 1.192 adım
(≈8 tur; her pencere ~8 kez) ≈ 55 dk, SFT çapası ve KL aynen (10 kopya + 6 SFT satırı).

| K4 kopya_orani (1.192 adım, ~8 tur) | HÜKÜM |
|---|---|
| **≥ 0,30** | **TEKRAR = kaldıraç** → v6 tam tarifi: kopya ısınması (≈8-10 tur) + karışık koşum |
| 0,10 – 0,30 | Kısmi: 12–15 tura çıkar (aynı tarif) |
| < 0,10 | **Bu ailede kopya ağırlıklarda KURULMUYOR** → dürüst rapor; tek yol ekstraktif RAG (seçenek 1) |

K1 sigortası aynı (≤2,45); bu koşumda pencere-CE artık %100 kopya avlusu (ölçek yine değişti).

## 8. PİLOT-3 ve PİLOT-4 — K1 SİGORTASI İKİ KEZ TETİKLENDİ (149 adım)

| Deneme | Veri | 149. adım SFT val CE | Hüküm |
|---|---|---|---|
| pilot-3 | %100 kopya, **morfoz metin** | **2,4593** > 2,45 | K1 erken duruş |
| pilot-4 | %100 kopya, **doğal metin** (düzeltme) | **2,4666** > 2,45 | K1 erken duruş |

Pilot-3'te morfem-morfoz metin hatası vardı (`vocab.decode` ile kurulmuş "domates fide si
dikim i") ve düzeltildi — **doğal metinle de aynı sonuç çıktı.** Yani okuryazarlık kaybının
nedeni metnin yüzeyi değil, **biçim egemenliği**: pencere akışının %100'ü `<BELGE>` json
zarfı olunca, 149 adımda 0,245 CE kaybı oluyor. K1 bütçesi (2,45 − 2,2219) **0,228**;
yani bütçe, kopya devresinin ihtiyaç duyduğu ~8 turdan **çok önce** tükeniyor.

Not: pilot-4'te pencere-CE 13,36 → 11,37 (149 adım) — kopya öğrenmesi de yavaş.
%20 pay + tek geçiş (pilot-2) K4=0 veriyor; tekrar (pilot-3/4) ise K1'i yıkıyor.

## 9. DÖRT ÖLÇÜMÜN BİLEŞKE HÜKMÜ

1. Kopya davranışı, **bitmiş bir okuryazarlık durumunun üstüne** kurulamıyor: 20% pay ve
   tek geçiş yetmiyor (K4=0), tekrar ise K1'i yıkıyor.
2. Mimari ders (P1) ile tutarlı: davranış devreleri **ön-eğitimde** kurulur; v5 gibi
   oturmuş bir tabana sonradan eklenen biçim, okuryazarlıkla doğrudan yarışıyor.
3. Geriye mimariye uygun tek yol kalıyor: kopya pencerelerini **pure'dan başlayan
   ön-eğitimin İÇİNE** katmak (K1 kapısı o aşamada anlamsız; taban CE 4+).



