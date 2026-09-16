# Unutma Ölçümü Tasarımı — F4 öncesi (17 Eyl 2026)

**Damga:** 2026-09-17T02:20 local (23:20Z) · **Yazan:** claude (danışman) · **Statü:** TASARIM (uygulama aracı hazır, taban çizgisi ölçüldü)
**Dayanak:** Onkanat kararı (16 Eyl): *"F4 kompozisyonu → ÖNCE UNUTMA ÖLÇÜMÜ; iki varyant denenip unutma ölçülecek."*

---

## 1. Soru
Ceket (F4) eğitimi, tabanın **kazandığı genel yetkinliği ve noktalama becerisini** geriletir mi? Gerileme ölçülürse F4 kompozisyonuna **chosen/SFT kütlesi tekrarı** eklenir; ölçülmezse yalın (ceket verisi) yeterlidir.

## 2. Üç eksen (hepsi AYNI betikle, önce/sonra)
| Eksen | Ne ölçülür | Dilim/topluluk | Eşik (ÖNCEDEN ilan) |
|---|---|---|---|
| **A. Genel dil kaybı** | Teacher-forcing CE (MASKEsiz mod — T-0053/T-0058 ile kıyaslanabilir) | `data/train_balanced_sft_v2.bin` son %10, 6 batch (offset 100000+b·1000, BS 4, SEQ 64) | Artış **≤ %10** |
| **B. Noktalama yetkinliği** | Hedefi noktalama olan ≥250 konumda top-1 / ilk-3 / ortalama rank (T-0058'in B ekseni) | `data/train_balanced_sft_v2.bin` (deterministik seed 7) | top-1 düşüşü **≤ 5 puan** |
| **C. Ceket alan içi kazanç** | `scripts/evaluate_carpenter_generation_100.py` (ezber oranı + alan içi başarı) | marangozluk held-out | **İyileşme > 0** (ceket öğrenilmiş olmalı) |

**Kontroller:** dilimler, seed'ler ve betik **önce/sonra aynı**; artefaktlar salt okunur; ölçüm kodu tek (`scratch/forgetting_probe.py`); **eşikler sonuç görülmeden ilan edildi** ve sonuçtan sonra değiştirilmez.

## 3. Ölçüm anları
1. **TABAN ÇİZGİSİ (F4 öncesi):** `data/kristal_model_f3_clean.pt` — **ölçüldü** (bkz `data/eval/forgetting_baseline_2026-09-17.json`).
2. **F4 SONRASI:** yeni ceket checkpoint'i (`data/kristal_model_carpenter_v2.pt` önerilir).
3. (ölçüm gerekirse) F3'ün maskesiz sürümü `kristal_model_f3.pt` ile kıyas — "maskeleme farkı mı, unutma mı" ayrımı için.

## 4. Karar kuralı (önceden)
- **A ve B eşikleri İÇİNDE, C iyileşmiş** → yalın F4 yeterli; kompozisyona müdahale YOK.
- **A veya B eşiği AŞILDI** → unutma var; F4'ün kompozisyonuna **chosen/SFT kütlesi tekrarı** eklenir (ayrı tur, kullanıcı onayıyla; kütle `train_all_chosen.bin`'dir ama **kaynakları diskte yok** → yalnız mevcut dosya olduğu gibi karıştırılabilir, noktalamasız olacağı bilinerek).
- **C iyileşmedi** → ceket öğrenilmemiş; F4 reçetesi (adım/lr) yeniden kurulur.

## 5. Araç
`scratch/forgetting_probe.py <checkpoint...>` — her checkpoint için A ve B'yi basar. C için mevcut değerlendirme betikleri kullanılır (`evaluate_carpenter_generation_100.py`).
