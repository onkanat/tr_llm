# İLAN — v4d ÖLÇÜM SONDASI: yakınsamış rejimde 5e-5 marjinal hız (2026-11-07)

**Durum:** KOŞUMDAN ÖNCE DONMUŞ. Karar kuralı bu dosyada yazılı; sonda bittikten sonra eşik DEĞİŞMEZ.
**Amaç:** "v4d = v4c'den +2.500 adım @5e-5" (A) seçeneğinin K2'nin 0,0651 nat'ını kapatıp kapatmayacağını TAHMİN değil ÖLÇÜM ile karara bağlamak.

## 1. Gerekçe (ölçülmüş zemin)

v4c (11.380 adım, 30.493 sn) yakınsamış rejimde bitirdi — pencere-CE binerli ortalamalar
(`data/eval/v4c_ogrenci_history.json`, 1.000 adım/bin):

| adım | pencere-CE ort | Δ |
|---|---|---|
| 1–1000 | 3,6093 | — |
| 2001–3000 | 3,4654 | −0,0471 |
| 5001–6000 | 3,4214 | −0,0100 |
| 9001–10000 | 3,4065 | −0,0030 |
| 10001–11000 | 3,3975 | −0,0091 |
| 11001–11380 | 3,4130 | **+0,0156** |

Son 2.500 adımın lineer eğimi **−0,00025 nat/1000 adım ≈ 0**; son 1.000 adımda CE yükseldi.
Bu, "+2.500 adım devam" varsayımını ölçümsüz bırakıyor: sonda tam bu rejimin hızını ölçer.

## 2. Sonda tarifi (donmuş — v4d'nin ilk 1.000 adımı, BİREBİR)

`scripts/train_v4_ogrenci.py`, v4c tarifi AYNEN (karışım 10P+6S, `--pencere-kayip plain`,
`--kl-kip sft` β=0,2, batch 16 × accum 2, pencere tavanı 128), tek farklar:

- `--init-model data/anka_ogrenci_v4c.pt` → **öğretmen de v4c** (öz-çapa; `--teacher-model` verilmez). Çıpa artık pure'a değil, v4c'nin bulunduğu yere sabitlenir — devam koşumunun doğru semantiği (kazancı geri almamak).
- **lr SABİT 5e-5** (`--peak-lr 5e-5 --min-lr 5e-5 --warmup 0`; `get_lr` peak==min_lr'de sabit döner) → 2.500-adım cosine'in ortalama hızını tek noktada temsil eder.
- `--toplam-adim 1000 --save-every 500` → nöbetler: init, 500, 1000(final).

**Beyanlı yanlılık (dürüstlük):** `planla_pencere_baslar(seed=42, epok=0)` v4c ile AYNI permütasyonu üretir → sonda, v4c'nin ilk 2.000 penceresini (11.380 adım önce görülmüş, en bayat küme) yeniden görür. Bu, gerçek bir 2. epoğun başlangıcıyla birebir aynıdır; ama **A lehine İYİMSER** bir kurgudur (en bayat pencere = en çok yeniden-uyum payı). Sonda "yetmedi" derse, A'nın gerçek 2.500 adımı daha da az getirir.

## 3. KARAR KURALI (koşumdan önce dondu)

Ölçüm: `pencere_ce_train` (TRAIN avlusu, 25 pencere, tohum 7 — v4c ile aynı rig).
Zemin (v4c final nöbeti): **3,3495**. Resmi VAL zemini: **3,2951** (bar 3,23 → gereken **−0,0651 nat**).
Tek-ölçek varsayımı (beyanlı): train avlusu ile resmi VAL aynı yönde ve aynı büyüklükte hareket eder.

A'nın 2.500 adımı için gereken 1.000-adım hızı = 0,0651 × (1000/2500) = **0,0260 nat/1000 adım**.

| Δ₁₀₀₀ (train avlusu) | HÜKÜM |
|---|---|
| **≥ 0,0260** | **A onaylanır** — 2.500 adım K2 barını kapatma beklentisi karşılanıyor |
| **0,0130 – 0,0260** | **A tek başına YETMEZ** — en az 5.000 adım gerekir (yeni ilan) |
| **< 0,0130** | **A REDDEDİLİR** — devam koşumu ekseni tüketmiş; sıradaki adım C (lr 1,5e-4 tam koşum) veya B (istisnalı terfi) |

**K1 sigortası aynen:** her nöbette SFT val CE > 2,45 → koşum DURUR (`ERKEN_DURUS_K1`). Zemin 2,2345.
**K3 sonda ile ÖLÇÜLMEZ** (paket tripwire nöbette var, resmi `test_model.py` sonda sonrası ayrı).

## 4. Komut

```
venv/bin/python scripts/train_v4_ogrenci.py \
  --init-model data/anka_ogrenci_v4c.pt \
  --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --kl-kip sft --kl-beta 0.2 --pencere-kayip plain \
  --peak-lr 5e-5 --min-lr 5e-5 --warmup 0 \
  --batch 16 --accum 2 --pencere-satir 10 \
  --toplam-adim 1000 --save-every 500 \
  --save-path scratch/v4d_sonda.pt --allow-frozen-write \
  --history-path scratch/v4d_sonda_history.json \
  --hukum-path scratch/v4d_sonda_hukum_2026-11-07.json
```

Beklenen süre: 1.000 × 2,66 sn ≈ 44 dk + ön-hesap ~60 sn. Çıktı `scratch/` (data/ DOKUNULMAZ).
