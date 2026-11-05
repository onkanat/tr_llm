# İLAN — V4 ÖĞRENCİ: külliyat bilgisi + SFT okuryazarlık çapası TEK koşumda (2026-11-05)

**Durum:** TAM KOŞUMDAN ÖNCE DONMUŞ — controller onayı bekler. Onay öncesi koşum YOK.
**Felsefe:** öğrenci külliyat bilgisiyle sınıf atlar ama okuryazarlığını ASLA kaybetmez (T-0209 deseni; "temel model ceketlere göre bükülmez" — T3.4'ün saf-korpus ihlali baştan işlenmez, P2.6'nın geri-alma merdiveni tekrarlanmaz).
**Sert kurallar:** commit YOK (controller işler); çıktı `data/` altına `--allow-frozen-write` ile; Qdrant YOK.

---

## 1. Ölçülmüş zemin (kanıtlar; her sayı komut çıktısından)

| Ölçüm | Değer | Kaynak |
|---|---|---|
| TABAN (pure) SFT val CE | 2,304304 | pilot `[BAŞLANGIÇ]` (05.11.2026); p26 kapı rig'i 2,3043 ile birebir |
| TABAN pencere-CE (TRAIN nöbet avlusu, 25, tohum 7) | 3,9633 | pilot `[BAŞLANGIÇ]`; P1 kaydıyla birebir |
| TABAN resmi VAL pencere-CE | 3,8205 | `eval_ce_val.py` (p26 KAPI-2 referans bandı) |
| TABAN okula paket güveni | `case` @ 0,9998 | pilot `[BAŞLANGIÇ]`; test_model.py TABAN bandıyla birebir |
| T3.4 ihlali | saf-korpus 2e-4, 14.226 adım → okuryazarlık 0,0726; pencere-CE 3,8205→3,1936 | `data/eval/t34_egitim_ozet_2026-11-04.md` (3,27 sn/adım, 12,9 sa) |
| P2.6 | SFT-ağırlıklı replay: CE 2,4271'e (2,30'a ASLA); pencere kazancı korundu (3,30 TRAIN) | `p26_tur5_hukum` (5e-5, 218 adım) |
| v4 SFT verisi | 5.735 kayıt; val-150 kuyruğu orijinalle özdeş (alan bazlı DOĞRU); UNK %0 (üretim kaydı) | `scratch/v4_zemin_probe.py` (05.11.2026) |
| v4 kayıt başına GERÇEK hedef konumu | ort 52,86 (std 34,19; min 4, maks 128); sm.sum() ort 126,71 → seyreltme 0,417 | aynı probe (FastSampleAlignedDataset ölçümü) |
| külliyat_v2.bin | 29.135.254 uint16; 227.619 blok; maks id 33.113 < 33.114 | aynı probe + meta kapısı |
| lr kanıtı | 5e-5 GÜVENLİ (p26 tur-1/4/5 + P1 dört koşum); 2e-4 YIKICI (T3.4); arada değer KANITSIZ → ALINMADI | hüküm kayıtları |

**Çıpa SHA'leri** (pilot provenans bloğu + probe):
```
5479a2f35a88975cae3cd8e66a4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt        (init VE KL öğretmeni; p26 ölçümüyle birebir)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json
7f91317aa58b17e777304b70619dc48b912c47ad563a448f5ca89cd95b57745c  data/kulliyat_v2/kulliyat_v2.bin
f90cf88b9f79037571c055dadc4f6322ee5bd2ead6a64796b32b4e3172368369  data/pedagogy/pure_base_sft_v4.jsonl
9f363f2f89e80bbb1364f9de5f6b50b9886b83da7caacdf35bcdfaaffeafad1  data/kulliyat_v2/kulliyat_v2_val.bin (resmi val avlusu)
```

## 2. Tarif (donmuş)

**Veri karışımı — token-bazlı oran NET beyanı:**
- "Token-bazlı" = **gerçek hedef konumu** (kayıp sinyalinin birimi, y ≠ −100): pencere satırı 128 hedef, SFT satırı ölçülen ort **52,86** hedef taşır.
- Mikro-parti (16 satır): **10 külliyat penceresi + 6 SFT kaydı** → opt-adım (accum 2): 20P + 12S.
- **Hedef-bazlı: %80,1 külliyat / %19,9 SFT** (ölçülen; pilot `[KARIŞIM]` birebir). Bağlam-jeton bazlı (satır=128 konum): %62,5/%37,5 — İLAN'da gizli DEĞİL, iki taban da beyanlı.
- Pencereler `kulliyat_v2.bin`'den rastgele, epok başına permütasyon **tekrarsız** (227.600/227.618 blok = kaplama %99,99) → 1 epok = külliyatın ~tümü bir kez. SFT çapası (5.585 train kaydı) her turda yeniden karıştırılarak **sonsuz döngü** (tam tur ≈ 466 adım → epokta ~24 tur).
- SFT val: son-150 (orijinal pure_base_sft kuyruğuyla özdeş; ölçüldü).

**KL çıpası:** öğretmen = init = `anka_taban_pure.pt`; forward-KL, **β = 0,2, TÜM satırlarda** (yalnız pencere değil — v4'te tüm model pure'a çıpalanır); öğretmen no_grad; mikro-parti disiplini SUB_CHUNK=8, öğretmen yığını KL_CHUNK=4 (MPS OOM dersi); satır-seçim kaynak tensorüyle (p26 B1; T-0209 ölü-kod tekrarı YOK). Kayıp formu p26'dan BİREBİR (payda sm.sum(); seyreltme 0,417 ölçülü — çapa ~24 tur tekrarlandığından dar-set underfit dersi aşıldı). *Dürüst not:* başlangıç KL ~0,125 (ağırlıklar özdeş; train/eval dropout asimetrisi — ölçüldü, çelişki değil).

**LR:** peak **5e-5**, warmup **200**, cosine → min **1e-5** (5e-5 güvenli-kanıtlı; 2e-4 yıkıcı-kanıtlı; ara değer kanıtsız — bilinçli alınmadı). AdamW wd 0,01; clip 1,0; tohum 42. Pencere tavanı 128 (aile dersi).

**Adım planı:** **11.380 opt-adım = 1 külliyat epoğu** (227.600 pencere / 20 pencere-adım). batch 16 × accum 2 = 32 satır/adım. save-every **500** (her nöbette atomik `.tmp`+replace ara-checkpoint).

**VAL-NÖBETİ (her save-every'de, üç gösterge + kayıp eğrisi):**
1. SFT val CE (val-150, FastSampleAlignedDataset, BLOCK=128 — TABAN 2,3043 rig'i);
2. külliyat pencere-CE (TRAIN bin, 25 pencere, tohum 7 — p26 avlusu; **resmi VAL avlusu bundan AYRI**, K2 ölçümünde eval_ce_val.py kullanılır);
3. okula paket güveni (tek görev, `case` bekle; K3'ün hızlı tripwire'ı — tam paket K3 kapısında ölçülür).
Kayıp eğrisi: history JSON'da adım-adım loss/KL.

## 3. ÜÇ DONMUŞ KAPI (koşum SONUNDA; eşikler önceden donmuş)

| Kapı | Metrik (resmi alet) | Eşik | Zemin |
|---|---|---|---|
| **K1 okuryazarlık** | masked SFT val CE (val-150 rig) | **≤ 2,45** | pure 2,3043; P2.6 bile 2,4271'e ulaşabildi — bant pure+0,15 |
| **K2 bilgi** | külliyat pencere-CE, **resmi VAL** (`eval_ce_val.py`, `kulliyat_v2_val.bin`, 500 pencere, tohum 1337) | **≤ 3,23** | pure 3,8205; T3.4 ihlali 3,1936'ya indi; P2.6 3,2118'i korudu |
| **K3 nöbet** | paket güveni: `test_model.py` 4 SFT görevi top-1 VE güven | **4/4 top-1 VE her görevde ≥ 0,99** | TABAN 0,9997/0,9981/0,9998/0,9986 (4/4) |

**K1 izleme izni (ERKEN DURUŞ):** ilk %20 penceresinde (adım ≤ 2.276; nöbetler 500/1000/1500/2000'de) SFT val CE **2,45'i AŞARSA koşum DURUR** (hüküm `ERKEN_DURUS_K1`; son ara-checkpoint diskte kalır). Sonrasındaki nöbetler yalnız kaydeder — kapılar koşum sonunda hükmedilir.

**Hüküm kuralı:** üç kapı da ölçülür; tarife yetmeyen kapı için dürüstçe KALDI verilir, eksikler sayısal tanıyla haritalanır (T3.4/P2.6 geleneği). Kapı yetersizliği İLAN'ı değiştirmez.

## 4. PİLOT ölçümleri (50 adım, 05.11.2026; `scratch/v4_ogrenci_pilot_*`)

| Gösterge | Değer |
|---|---|
| opt-adım / süre / ön-hesap | 50 / 229,5 sn / 66,8 sn |
| s/adım (son-20) | **2,64 ± 0,16 sn** (II. koşum, sakin makine) — I. koşumda 5,05 ± 1,33 sn (yüksek çekişme) |
| MPS tepe | **1,655 GB** (önerilen sınır 5,33 GB — bol pay) |
| KL ateşlenme | **100/100 mikro-parti** (tüm-satır kipi kanıtlandı); KL ort 0,1252 |
| Nöbet ilk değerleri | adım 0: 2,304304 / 3,9633 / case@0,9998 · adım 25: 2,307266 / 3,9579 / case@0,9998 · adım 50: 2,314865 / 3,9425 / case@0,9998 |
| Erken sinyal | pencere-CE (TRAIN) −0,021 nat @ 50 warmup adımı (lr henüz 1,25e-5) — bilgi ekseni HAREKETLİ; SFT CE +0,011 (K1'e pay 0,135) |

**TAM koşum ETA bandı (dürüst):** ETA kuralı = pilot s/adım × 3 (sürekli-koşum çarpanı; T3.4 kanıtı: pilot 1,18 → gerçek 3,27) × 11.380 adım.
- II. koşum noktası: **25,0 sa** (bant 23,5–26,6 sa); I. koşum noktası: **47,9 sa** (bant 35,3–60,5 sa).
- **Beyan: 24–48 sa (1–2 gün) plan bandı**; makine çekişmesine bağlı. K1 erken-duruş penceresi (~2.276 adım) ≈ 5–10 sa içinde tamamlanır — orada üç-gösterge nöbetleri ilk hüküm fırsatını verir. Ara-checkpoint her 500 adımda diskte: durdurma talimatı her nöbette kanıtlı verir.

## 5. TAM KOŞUM KOMUTU (İLAN onayı SONRASI; ~24-48 sa)

```
venv/bin/python scripts/train_v4_ogrenci.py \
  --init-model data/anka_taban_pure.pt \
  --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --kl-beta 0.2 --peak-lr 5e-5 --min-lr 1e-5 --warmup 200 \
  --batch 16 --accum 2 --pencere-satir 10 \
  --toplam-adim 11380 --save-every 500 \
  --save-path data/anka_ogrenci_v4.pt --allow-frozen-write \
  --history-path data/eval/v4_ogrenci_history.json \
  --hukum-path data/eval/v4_ogrenci_hukum_2026-11-05.json
```

Koşum sonrası kapı ölçümleri (K1 koşum içi zaten; K2/K3 resmi):
```
K2: venv/bin/python scripts/eval_ce_val.py --model data/anka_ogrenci_v4.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json \
      --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337
K3: venv/bin/python test_model.py --checkpoint data/anka_ogrenci_v4.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json
```

## 6. Dürüst risk beyanı

- **K2 iddialı:** hedef −0,59 nat (3,8205→≤3,23); kıyas T3.4: −0,63 nat ama 2e-4/100% sinyal/14.226 adımla VE okuryazarlık feda edilerek. v4: 5e-5 + %80 sinyal + β=0,2 çıpa pure'a (3,82 dağılımı) — adım-başına kazanç daha küçük; 1 epokda K2'ye yetmeyebilir → o durumda dürüst KALDI + sayısal tanı (KL denge noktası, lr-merdiven önerisi AYRI İLAN'la).
- **K1 korumalı:** çapa 24 tur + tüm-satır KL; P2.6'nın 2,30'a dönememe dersi başlangıç-farkı kaynaklıydı (v4 zATEN 2,3043'ten başlar — tutmak, geri almak değil).
- **K3 tam paket:** nöbet yalnız okula görevini izler (tripwire); nihai K3 test_model.py 4-görev paketiyle donmuş eşikte ölçülür.
- Pilot-1/pilot-2 s/adım farkı (2,64 vs 5,05) makine çekişmesi; bandın genişliği bundan — gizlenmedi.
