# HÜKÜM — v4d sonda (5e-5 sabit, 1.000 adım): A SEÇENEĞİ REDDEDİLDİ (2026-11-07)

**İlan:** `data/eval/v4d_sonda_ilan_2026-11-07.md` (karar kuralı koşumdan ÖNCE dondu)
**Koşum:** `scratch/v4d_sonda_run.log` · hüküm JSON `scratch/v4d_sonda_hukum_2026-11-07.json` · çıktı `scratch/v4d_sonda.pt` (sha `09cb99e2…8736d`)
**Init/öğretmen:** `data/anka_ogrenci_v4c.pt` sha `de5ad556…c42b9c` (öz-çapa) · süre 2.917,5 sn / 2,82 s-adım

## Ölçüm

| Nokta | SFT val CE (K1) | pencere-CE (TRAIN avlusu, 25) | okula |
|---|---|---|---|
| init (v4c) | 2,234496 | **3,3495** | case @ 0,9993 |
| adım 500 | 2,240445 | 3,3529 | case @ 0,9993 |
| adım 1000 | 2,249763 | **3,3480** | case @ 0,9991 |

**Δ₁₀₀₀ = −0,0015 nat** (per-adım eğim −0,00101 nat/1000; ilk 100 adım 3,3802 → son 100 adım 3,4036 = yükseliş)

## Karar (donmuş kurala göre)

| Ölçüt | Değer |
|---|---|
| Gereken (0,0651 nat × 1000/2500) | **≥ 0,0260** |
| Ölçülen | **0,0015** → **17× eksik** |
| 2.500 adım karşılığı (aynı hız) | −0,0025 … −0,0038 nat = 0,065'in **%4–6'sı** |
| Hüküm | **A REDDEDİLDİ** (Δ < 0,0130 bandı) |

Ek: devam koşumu K1'i de hafif geri götürdü (2,2345 → 2,2498, +0,0153) — bilgi kazancı sıfırken okuryazarlık maliyeti var.

## Teşhis (hüküm değil, neden)

1. **Darboğaz adım sayısı değil, çapa↔külliyat dengesi.** Aynı 1 epokta: T3.4 (lr 2e-4, **çıpasız**) resmi VAL 3,1936 ama okuryazarlık 0,9998→0,0726 öldü; v4c (lr 1e-4, **çıpalı**) 3,2951 ve okuryazarlık İYİLEŞTİ. Çapanın bilgi eksenine maliyeti ≈0,10 nat. 0,065 nat'lık açık tam olarak bu faturanın içinde.
2. **Aynı ağırlıklardan devam, hiçbir lr'de bu açığı kapatmaz** — model 227.600 pencerelik dağılımdan çıkaracağını çıkarmış; sonda bunu en bayat (dolayısıyla en kazançlı) pencerelerde ölçtü.
3. Kalan tek eksen **lr'yi yükseltip K1 payını (0,2155 nat) harcamak**: lr 1e-4→3,2951 ve 2e-4→3,1936 noktaları arasında 1,5e-4 için kaba kestirim 3,24 (bar 3,23) — **yazı-tura**; kestirim T3.4'ün çıpasız olduğu için A lehine iyimser.

## Açık seçenekler

- **C:** lr 1,5e-4 tam koşum (11.380 adım, ~8,5 sa, çıpa ve karışım aynen) — K2 için tek dürüst deneme, beklenti 3,24–3,28.
- **B:** v4c'yi iki kapı istisnasıyla terfi (K2 0,0651 + K3 PLURAL 0,0219) — K1 geçti ve iyileşti; bilgi ekseni barın %89'unda; ağırlıklar semantik bilgi taşımıyor (Chinchilla %1,5, kopya devresi yok) → bilgi zaten vektör katmanında.
