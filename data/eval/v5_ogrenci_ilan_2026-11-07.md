# İLAN — V5 ÖĞRENCİ: pure'dan lr 1,5e-4 tam koşum (2026-11-07)

**Durum:** TAM KOŞUMDAN ÖNCE DONMUŞ — controller onayı ALINDI (07.11.2026, seçenek C). Kapı eşikleri ve aletler v4c'den AYNEN; hiçbiri kaydırılmadı.
**Zemin:** v4d sonda (07.11.2026) devam koşumu eksenini ÖLÇÜP kapattı: v4c'den sabit 5e-5 ile 1.000 adım → pencere-CE 3,3495→3,3480 (**−0,0015 nat**; gereken 0,0260). Aynı ağırlıklardan devam hiçbir lr'de K2'nin 0,0651 nat'ını kapatmaz. Kalan iki eksenden biri olan **lr** denenir: v4c lr 1e-4'te 3,2951'e indi, T3.4 lr 2e-4'te (çıpasız) 3,1936'ya indi → iç nokta 1,5e-4.
**Amaç:** K2 barını (≤3,23) kapatmak; K1'i (≤2,45) ve K3'ü koruyarak. v4c'ye DOKUNULMAZ — bu koşum ayrı bir öğrenci üretir; kaybederse en iyi öğrenci v4c olarak diskte kalır.

---

## 1. Zemin ölçümleri (her sayı kayıtlı)

| Ölçüm | Değer | Kaynak |
|---|---|---|
| TABAN (pure) SFT val CE | 2,304304 | v4b/v4c pilot `[BAŞLANGIÇ]` birebir |
| TABAN resmi VAL pencere-CE | 3,8205 | `eval_ce_val.py` |
| T3.4 (2e-4, çıpasız, saf korpus) | resmi VAL **3,1936**, okuryazarlık 0,9998→0,0726 ÖLDÜ | `data/eval/t34_egitim_ozet_2026-11-04.md` |
| v4 (5e-5, tüm-satır KL) | resmi VAL 3,905 (bilgi bloklandı) | `v4_ogrenci_hukum_2026-11-05.json` |
| v4b (1e-4, sm-kayıp patolojisi) | 6,0382 / K1 3,0168 | `v4b_ogrenci_hukum_2026-11-05.json` |
| **v4c (1e-4, plain + SFT-çıpa)** | **resmi VAL 3,2951** (bar 3,23 → 0,0651 KALDI); K1 2,2345 GEÇTİ+iyileşti; K3 PLURAL 0,9721 | `v4c_ogrenci_hukum_2026-11-06.json` |
| v4d sonda (5e-5 devam) | Δ₁₀₀₀ −0,0015 nat → devam ekseni TÜKENDİ | `v4d_sonda_hukum_2026-11-07.md` |
| çapanın bilgi faturası | ≈0,10 nat (T3.4 ↔ v4c, aynı 1 epok) | aynı dosyalar |

**Çıpa SHA'leri (koşumdan önce terminalden programatik okundu, 07.11.2026 07:23):**
```
5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt   (init VE KL öğretmeni)
f90cf88b9f79037571c055dadc4f6322ee5bd2ead6a64796b32b4e3172368369  data/pedagogy/pure_base_sft_v4.jsonl
7f91317aa58b17e777304b70619dc48b912c47ad563a478f5ca89cd95b57745c  data/kulliyat_v2/kulliyat_v2.bin
9f363f2f89e80bbb1364f9de5f6b50b9886b83da7caacdf35bcdfaaffeafad1  data/kulliyat_v2/kulliyat_v2_val.bin (resmi K2 avlusu)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json
```

## 2. Tarif (donmuş) — v4c'den TEK değişken

`--peak-lr 1e-4 → 1,5e-4` (min-lr 1e-5 ve warmup 200 AYNEN; cosine). Bunun dışında v4c tarifi birebir:

- Karışım: mikro-parti 10 pencere + 6 SFT, accum 2 → hedef-bazlı %80,1/19,9 (ölçülen ort SFT hedefi 52,86).
- `--pencere-kayip plain` (pencere satırlarında standart CE; SFT satırlarında sm-formu aynen; sm gömme girdisi olarak yaşar).
- `--kl-kip sft` β=0,2, öğretmen = init = pure (öz-çapa), yalnız SFT satırları.
- 11.380 opt-adım = 1 külliyat epoğu (kaplama %99,99), save-every 500, batch 16 × accum 2, pencere tavanı 128.

## 3. ÜÇ DONMUŞ KAPI (eşikler ve aletler DEĞİŞMEDİ)

| Kapı | Metrik | Eşik | Zemin |
|---|---|---|---|
| **K1 okuryazarlık** | masked SFT val CE (val-150 rig) | **≤ 2,45** | pure 2,3043; v4c 2,2345 (geçti) |
| **K2 bilgi** | resmi VAL pencere-CE (`eval_ce_val.py`, 500 pencere, tohum 1337) | **≤ 3,23** | pure 3,8205; v4c 3,2951 |
| **K3 nöbet** | `test_model.py` 4 görev top-1 VE her görevde ≥ 0,99 | 4/4 + ≥0,99 | v4c: root/TENSE/case ✓, PLURAL 0,9721 ✗ |

**K1 erken-duruş aynen:** her nöbette (500 katları, tüm koşum) SFT val CE > 2,45 → koşum DURUR, son ara-checkpoint diskte.
**Hüküm kuralı aynen:** yetmeyen kapıya dürüst KALDI; eşik sonradan kaydırılmaz.

## 4. ÖNCEDEN DONMUŞ TAHMİN (kapı DEĞİL — sonuç görülünce değişmez)

- **K2 beklentisi: 3,24–3,28 → yazı-tura.** Gerekçe: 1e-4→3,2951 ve 2e-4→3,1936 noktaları arasında doğrusal kestirim 3,2443; kestirim T3.4'ün ÇIPASIZ olması nedeniyle **iyimser**.
- **K1 beklentisi: 2,20–2,35** (v4c 1e-4'te iyileşti; çıpa aynı güçte). K1 payı 0,2155 nat.
- **K3 beklentisi: PLURAL 0,97–0,99** — bant hâlâ marjinal; K3 tek başına KALDI verebilir.
- **Dürüst olasılık:** K2 geçme şansı ~%40–50; K1 geçme ~%90; üçünün birden geçmesi ~%35–45.

## 5. Komut

```
venv/bin/python scripts/train_v4_ogrenci.py \
  --init-model data/anka_taban_pure.pt \
  --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --kl-kip sft --kl-beta 0.2 --pencere-kayip plain \
  --peak-lr 1.5e-4 --min-lr 1e-5 --warmup 200 \
  --batch 16 --accum 2 --pencere-satir 10 \
  --toplam-adim 11380 --save-every 500 \
  --save-path data/anka_ogrenci_v5.pt --allow-frozen-write \
  --history-path data/eval/v5_ogrenci_history.json \
  --hukum-path data/eval/v5_ogrenci_hukum_2026-11-07.json
```

Beklenen süre: v4c gerçekleşmesi 2,66 s/adım → **≈8,5 sa** (sonda 2,82 s/adım ölçtü → üst uç 8,9 sa).

Koşum sonrası K2/K3 resmi ölçümleri (v4c ile aynı aletler):
```
K2: venv/bin/python scripts/eval_ce_val.py --model data/anka_ogrenci_v5.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json \
      --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337
K3: venv/bin/python test_model.py --checkpoint data/anka_ogrenci_v5.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json
```

## 6. Dürüst risk beyanı

- **Asıl risk K1'dir (v4c'de tersiydi):** lr %50 yükseldi; T3.4 bu lr'nin 1,33 katında okuryazarlığı öldürdü. Sigortalar: %19,9 SFT CE çapası + SFT-satır KL β=0,2 + her-500-adım K1 nöbeti (erken-duruş). Nöbet erken tetiklenirse kayıp ≤2–3 sa; v4c elde kalır.
- **K2 yine kalabilir:** tahmin bandının alt ucu (3,28) barın 0,05 üstünde. Bu durumda üç eksen de (lr, karışım, süre) ölçülmüş olur ve karar B'ye (istisnalı terfi) veya mimari eksene (kapasite/veri) kalır.
- **K3 marjinal kalabilir:** PLURAL güveni devam-eğitimlerinde sistematik olarak bandın altında seyrediyor (v4 0,9441 → v4c 0,9721).
- **Disk:** 105 GB boş, checkpoint 374 MB — sorun yok.
