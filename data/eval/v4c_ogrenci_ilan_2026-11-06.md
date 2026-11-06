# İLAN — V4C ÖĞRENCİ: pencere plain-CE dalı + K1 nöbeti tüm koşuma (2026-11-06)

**Durum:** TAM KOŞUMDAN ÖNCE DONMUŞ — controller onayı bekler. Onay öncesi tam koşum YOK (pilot hariç).
**Zemin hipotez:** v4'ün tüm-satır KL'ı bilgiyi blokladı (K2 KALDI); v4b çıpayı SFT satırlarına daralttı ama pencere satırlarının KAYIP AĞIRLIĞI hâlâ ±1 işaret maskeliydi (T-0209 formu): −1 işaretli konumlar kayıp payından ÇIKARIR, gradyanı yukarı çevirir — pencere bileşeni negatifa inebilir (sm-kayıp patolojisi). v4c TEK cerrahi değişiklikle bilgiyi serbest bırakır: pencere satırlarında PLAIN CE (T3.4'ün bilgi girişini sokan dalı), SFT satırlarında mevcut sm-formu + KL yalnız orada (β=0,2 — format yaşadığı yerde korunur). K1 nöbeti tüm koşuma genişletilir (v4b dersi: ilk %20 penceresi yetmedi).
**Sert kurallar:** commit YOK (controller işler); çıktı `data/` altına `--allow-frozen-write` ile; Qdrant YOK; TAM KOŞUMU controller başlatır.

---

## 1. Ölçülmüş zemin (her sayı kayıtlı ölçümden)

| Ölçüm | Değer | Kaynak |
|---|---|---|
| TABAN (pure) SFT val CE | 2,304304 | v4b/v4c pilotları `[BAŞLANGIÇ]` birebir + p26 kapı rig'i (2,3043) |
| TABAN pencere-CE (TRAIN nöbet avlusu, 25, tohum 7) | 3,9633 | v4b/v4c pilotları birebir; P1 kaydıyla uyumlu |
| TABAN resmi VAL pencere-CE | 3,8205 | `eval_ce_val.py` (K2 referans bandı) |
| TABAN okula paket güveni | `case` @ 0,9998 | pilotlar; test_model.py TABAN bandıyla birebir |
| T3.4 ihlali (2e-4, çıpasız, saf-korpus) | okuryazarlık 0,9998→**0,0726 öldü**; pencere-CE 3,8205→3,1936 (**bilgiyi PLAIN CE soktu — v4c'nin dayanağı**) | `data/eval/t34_egitim_ozet_2026-11-04.md` |
| v4 (5e-5, tüm-satır KL β=0,2) | K1 geçti (2,3926) ama pencere-CE 3,9633→**4,1485 GERİ sardı**; resmi VAL 3,905 → **K2 KALDI** | `data/eval/v4_ogrenci_hukum_2026-11-05.json` |
| v4b (1e-4, SFT-only KL, sm-ağırlıklı pencere dalı) | pilot −0,037 nat @50; tam koşum sonuçları İLAN'da değil — v4c, pencere KAYIP AĞIRLIĞINI da düzeltir | `scratch/v4b_ogrenci_pilot_hukum.json` |
| sm-kayıp patolojisi | sm dalında pencere bileşeni NEGATİF mümkün (TDD kanıtı: `test_pencere_kayip_sm_dali_patoloji_negatif_mumkun`); plain dalda negatif İMKÂNSIZ (matematik: CE≥0, ağırlık≥0) | tests/test_v4_ogrenci.py |

**Çıpa SHA'leri (provenans — v4c pilot hüküm JSON'u ile birebir):**
```
5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt        (init VE KL öğretmeni; öz-çapa)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json
7f91317aa58b17e777304b70619dc48b912c47ad563a478f5ca89cd95b57745c  data/kulliyat_v2/kulliyat_v2.bin
f90cf88b9f79037571c055dadc4f6322ee5bd2ead6a64796b32b4e3172368369  data/pedagogy/pure_base_sft_v4.jsonl
9f363f2f89e80bbb1364f9de5f6b50b9886b83da7caacdf35bcdfaaffeafad1  data/kulliyat_v2/kulliyat_v2_val.bin (resmi val avlusu)
```

## 2. Tarif (donmuş)

**Veri karışımı — v4'ten AYNEN (tek satır değişmedi):** mikro-parti 16 satır = **10 külliyat penceresi + 6 SFT kaydı**; opt-adım (accum 2): 20P+12S. Hedef-bazlı **%80,1 külliyat / %19,9 SFT** (ölçülen ort SFT hedefi 52,86/kayıt); bağlam-jeton bazlı %62,5/%37,5 — iki taban da beyanlı. Pencereler `kulliyat_v2.bin`'den rastgele, epok permütasyonu tekrarsız (227.600/227.618 blok, kaplama %99,99). SFT çapası 5.585 train kaydı, tur başına yeniden karıştırılan sonsuz döngü (tam tur ≈ 466 adım); SFT val: son-150.

**Pencere KAYIP dalı (v4c TEK CERRAHİ DEĞİŞİKLİK):** `--pencere-kayip plain` → pencere satırlarında (kaynak==1) kayıp ağırlığı = geçerli-hedef göstergesi (gerçek pencerede TÜM 128 konum) → **standart ortalama CE, ≥0 her zaman (negatif İMKÂNSIZ)**. SFT satırlarında (kaynak==0) **mevcut sm-formu AYNEN kalır** (iki değişken birden değişmez). **sm MODELE gömme girdisi olarak yaşamaya devam eder** — KristalEmbedding her forward'da ±1 işareti uygular; mimari aynen, yalnız KAYIP AĞIRLIĞI değişti. Varsayılan `sm` = v4/v4b formu birebir (geriye uyum; eski komutlar değişmeden koşar).

**KL çıpası — v4b'den AYNEN:** `--kl-kip sft` → forward-KL, β=0,2, T=1, **YALNIZ kaynak==0 (SFT) satırlarında**; pencere (bilgi) satırları ÇIPASIZ. Öğretmen = init = pure (öz-çapa). SUB_CHUNK=8, öğretmen yığını KL_CHUNK=4, satır-seçim TEK NOKTA `kl_satirlari_sec`. Beklenen desen: mikro-parti başına 6/16 satır (%37,5 satır-bazlı; SFT hedef payı %19,9).

**LR:** peak **1e-4**, warmup **200**, cosine → min **1e-5**. AdamW wd 0,01; clip 1,0; tohum 42. Pencere tavanı 128 (aile dersi).

**Adım planı:** **11.380 opt-adım** (= 1 külliyat epoğu), save-every **500** (atomik ara-checkpoint her nöbette). VAL-nöbeti aynen: SFT val CE (val-150 rig) + külliyat pencere-CE (TRAIN avlusu, 25 pencere, tohum 7; resmi VAL K2'de AYRI) + okula tek-görev tripwire + pencere-kayıp bileşeni (plain: ≥0 patoloji tanığı, her Adım satırında) + history JSON.

## 3. ÜÇ DONMUŞ KAPI — eşikler ve aletler değişmedi

| Kapı | Metrik (resmi alet) | Eşik | Zemin |
|---|---|---|---|
| **K1 okuryazarlık** | masked SFT val CE (val-150 rig) | **≤ 2,45** | pure 2,3043; v4 final 2,3926 (geçti); P2.6 2,4271 |
| **K2 bilgi** | külliyat pencere-CE, **resmi VAL** (`eval_ce_val.py`, `kulliyat_v2_val.bin`, 500 pencere, tohum 1337) | **≤ 3,23** | pure 3,8205; T3.4 3,1936 (ihlalle); P2.6 3,2118; v4 3,905 KALDI |
| **K3 nöbet** | paket güveni: `test_model.py` 4 SFT görevi top-1 VE güven | **4/4 top-1 VE her görevde ≥ 0,99** | TABAN 0,9997/0,9981/0,9998/0,9986 (4/4) |

**K1 ERKEN-DURUŞ (v4c DEĞİŞİKLİĞİ — kural genişletildi, eşik AYNEN):** HER nöbet noktasında (save-every 500 katları, **tüm koşum**) SFT val CE 2,45'i AŞARSA koşum DURUR (hüküm `ERKEN_DURUS_K1`; son ara-checkpoint diskte). v4b'deki ilk-%20 izni YOK — v4b dersi: okuryazarlık sürüklenmesi sonraki adımlarda da yakalanabilmeli. Kural tek noktada: `k1_erken_durus_karar` (TDD'li).

**Hüküm kuralı aynen:** üç kapı da ölçülür; tarife yetmeyen kapı için dürüstçe **KALDI** verilir, eksikler sayısal tanıyla haritalanır. Kapı yetersizliği İLAN'ı değiştirmez.

## 4. PİLOT ölçümleri (50 adım, 06.11.2026; `scratch/v4c_ogrenci_pilot_*`, hüküm: `v4c_ogrenci_pilot_hukum.json`)

| Gösterge | Değer |
|---|---|
| opt-adım / eğitim süresi / ön-hesap | 50 / 187,1 sn / 51,4 sn |
| s/adım (son-20) | **2,36 ± 0,13 sn** |
| MPS tepe | 1,773 GB (önerilen sınır 5,33 GB — bol pay; v4b ile aynı) |
| **Pencere-satırı kayıp bileşeni (KRİTİK kanıt)** | **POZİTİF: min 3,0125 / ort 3,7620 / max 4,4163** (100/100 mikro-parti; adım-bazlı 50/50, min 3,4120) — **negatif SIFIR**, normal bant 2–6 içinde. Koşum içi nöbet: plain dalda negatif ölçülürse koşum V4_DUR ile durur (canlı patoloji testi) |
| Nöbet ilk değerleri | adım 0: **2,304304 / 3,9633 / case@0,9998** (init=pure BİREBİR ✓) · adım 25: 2,310909 / 3,9531 / case@0,9998 · adım 50: 2,322589 / 3,9261 / case@0,9998 |
| Pencere-CE erken sinyal | **−0,0372 nat @ 50 warmup adımı** (lr 2,5e-5) — v4b pilotu −0,0368: **"en az o kadar" beklentisi sağlandı**. Asıl plain etkisi warmup sonrası beklenir: v4b'de −1 işaretli konumların gradyanı yukarı çeviren katkısı v4c'te tamamen kaldırıldı |
| SFT CE @50 | +0,0183 (2,3043→2,3226) — v4b ile aynı karakterde; K1 payı 0,127 |
| **KL ateşlenme deseni** | **satır-bazlı 600/1600 = %37,5 (mikro-parti başına 6/16)** — sft kipi AYNEN ✓; mikro-parti-bazlı 100/100; KL ort 0,1449 (v4b aynı noktada 0,1450) |
| K1 tripwire | adım 25 nöbeti 2,45'in altında (2,3109) — erken-duruş yok |

**TAM koşum ETA (İLAN kuralı aynen):** pilot s/adım × 3 × 11.380 → **nokta 22,4 sa; dürüst bant 21,2–23,6 sa**. (Kalibrasyon dipnotu v4b'den miras: v4'te ×3 kuralı aşırı söyledi — gerçek/pilot ≈ ×1,25 çekişme düzeltmesiyle ~9–13 sa bandı mümkün; beyan kuralı değişmez, notla birlikte.)

## 5. TAM KOŞUM KOMUTU (İLAN onayı SONRASI; controller başlatır)

```
venv/bin/python scripts/train_v4_ogrenci.py \
  --init-model data/anka_taban_pure.pt \
  --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --kl-kip sft --kl-beta 0.2 --pencere-kayip plain --peak-lr 1e-4 --min-lr 1e-5 --warmup 200 \
  --batch 16 --accum 2 --pencere-satir 10 \
  --toplam-adim 11380 --save-every 500 \
  --save-path data/anka_ogrenci_v4c.pt --allow-frozen-write \
  --history-path data/eval/v4c_ogrenci_history.json \
  --hukum-path data/eval/v4c_ogrenci_hukum_2026-11-06.json
```

Koşum sonrası kapı ölçümleri (K1 koşum içi zaten her-500-nöbette; K2/K3 resmi):
```
K2: venv/bin/python scripts/eval_ce_val.py --model data/anka_ogrenci_v4c.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json \
      --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337
K3: venv/bin/python test_model.py --checkpoint data/anka_ogrenci_v4c.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json
```

## 6. Dürüst risk beyanı

- **Okuryazarlık yine risk altında (ANA risk, değişmedi):** pencere plain-CE T3.4'te 2e-4'te bilgiyi soktu ama SFT çapası YOKTU — okuryazarlık 0,9998→0,0726 çöktü. v4c'de %19,9 SFT CE çapası + SFT-only KL (β=0,2) + 1e-4 (2e-4'ün yarısı) bu çöküşün önündeki üç sigortadır; **K1 nöbeti (her 500 adımda, tüm koşum) son savunmadır** — SFT CE 2,45'i her aşarsa koşum durur, son ara-checkpoint diskte kalır. Bu sigortaların YETMEYECEĞI açık bir olasılıktır; sonuç K1/K3'ten çıkmazsa dürüst KALDI.
- **Bilgi girişi beklenti, garanti DEĞİL:** pilot −0,0372 nat @50 (v4b ile eşit) — asıl test warmup sonrası 1e-4 platosunda. Plain dal −1 gradyan-yukarı katkısını kaldırdı (patoloji testiyle kanıtlı: sm dalında negatif bileşen MÜMKÜN, plain'de İMKÂNSIZ) ama 1 epok (11.380 adım) hedef −0,59 nat (3,8205→≤3,23) için yetmeyebilir → **dürüst KALDI olasılığı K2'de hâlâ var**; eksik gelirse pencere-CE eğrisi + lr-merdiven önerisi AYRI İLAN'la gelir.
- **SFT dalında sm seyreltmesi bilinçli korundu:** SFT satırlarında ±1 işaret-ağırlıklı form değişmedi (iki değişken birden değişmez kuralı) — p26 seyreltme dersi (hedef/sm ≈ 0,417) SFT tarafında aynen geçerli; çapa ~25 tur tekrarlandığından underfit riski taşınır.
- **Patoloji canlı nöbeti:** plain kipte pencere bileşeni negatif ölçülürse koşum V4_DUR ile durur (matematikçe imkânsız — ölçülürse kayıp formu bozulmuş demektir). Pilot: 100/100 mikro-parti pozitif.
- **Bellek/OM:** KL öğretmeni yalnız 6/16 satırda → MPS baskısı v4'ten düşük (pilot tepe 1,773 GB); pencere-kayıp tanığı ek hesap maliyeti iki küçük .item() senkronu (pilot s/adım v4b ile aynı bantta).
