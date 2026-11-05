# İLAN — V4B ÖĞRENCİ: SFT-only KL + lr 1e-4 (aday A; 2026-11-05)

**Durum:** TAM KOŞUMDAN ÖNCE DONMUŞ — controller onayı bekler. Onay öncesi tam koşum YOK (pilot hariç).
**Zemin hipotez:** v4'ün tüm-satır KL'ı okuryazarlığı korudu ama bilgiyi blokladı; v4b çıpayı daraltır (yalnız SFT satırları) ve lr'yi orta noktaya çıkarır (1e-4). **Çıpa yalnız öğrencinin bildiğini korur** — pencere (bilgi) satırlarının KL ile frenlenmesi v4'te ölçülen hataydı.
**Sert kurallar:** commit YOK (controller işler); çıktı `data/` altına `--allow-frozen-write` ile; Qdrant YOK; TAM KOŞUMU controller başlatır.

---

## 1. Ölçülmüş zemin (dün ölçülen üç nokta + TABAN; her sayı kayıtlı ölçümden)

| Ölçüm | Değer | Kaynak |
|---|---|---|
| TABAN (pure) SFT val CE | 2,304304 | v4 pilot `[BAŞLANGIÇ]` + p26 kapı rig'i (2,3043) birebir |
| TABAN pencere-CE (TRAIN nöbet avlusu, 25, tohum 7) | 3,9633 | v4 pilot `[BAŞLANGIÇ]`; P1 kaydıyla birebir |
| TABAN resmi VAL pencere-CE | 3,8205 | `eval_ce_val.py` (K2 referans bandı) |
| TABAN okula paket güveni | `case` @ 0,9998 | v4 pilot; test_model.py TABAN bandıyla birebir |
| T3.4 ihlali (2e-4, çıpasız, saf-korpus) | okuryazarlık 0,9998→**0,0726 öldü**; pencere-CE 3,8205→3,1936 (bilgi girdisi gerçek) | `data/eval/t34_egitim_ozet_2026-11-04.md` (3,27 sn/adım, 12,9 sa) |
| v4 (5e-5, tüm-satır KL β=0,2, 11.380 adım) | K1 GEÇTİ (final SFT CE **2,3926** ≤ 2,45); bilgi BLOKLANDI: pencere-CE TRAIN 3,9633→**4,1485** (geri sarma), resmi VAL **3,905 > pure 3,8205 → K2 KALDI**; KL ort 0,125→**0,44** (tüm-satır çıpa öğretmenden uzaklaşmayı cezalandırdı) | `data/eval/v4_ogrenci_hukum_2026-11-05.json` (10,43 sa) + controller K2 ölçümü |
| lr merdiveni | 5e-5: okuryazarlık GÜVENLİ ama bilgi tavanı düşük (v4); 2e-4: çıpasız YIKICI (T3.4); **1e-4: ORTA NOKTA — KANITSIZ**, K1 erken-duruş sigortasıyla | v4/T3.4 hükümleri |

**Çıpa SHA'leri (provenans bloğu — pilot `data/eval/v4b_ogrenci_pilot_hukum.json` + v4 İLAN):**
```
5479a2f35a88975cae3cd8e66a4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt        (init VE KL öğretmeni; öz-çapa)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json
7f91317aa58b17e777304b70619dc48b912c47ad563a448f5ca89cd95b57745c  data/kulliyat_v2/kulliyat_v2.bin
f90cf88b9f79037571c055dadc4f6322ee5bd2ead6a64796b32b4e3172368369  data/pedagogy/pure_base_sft_v4.jsonl
9f363f2f89e80bbb1364f9de5f6b50b9886b83da7caacdf35bcdfaaffeafad1  data/kulliyat_v2/kulliyat_v2_val.bin (resmi val avlusu)
```

## 2. Tarif (donmuş)

**Veri karışımı — v4'ten AYNEN (tek satır değişmedi):** mikro-parti 16 satır = **10 külliyat penceresi + 6 SFT kaydı**; opt-adım (accum 2): 20P+12S. Hedef-bazlı **%80,1 külliyat / %19,9 SFT** (ölçülen ort SFT hedefi 52,86/kayıt); bağlam-jeton bazlı %62,5/%37,5 — iki taban da beyanlı. Pencereler `kulliyat_v2.bin`'den rastgele, epok permütasyonu tekrarsız (227.600/227.618 blok, kaplama %99,99). SFT çapası 5.585 train kaydı, tur başına yeniden karıştırılmalı sonsuz döngü (tam tur ≈ 466 adım); SFT val: son-150 (orijinal kuyrukla özdeş).

**KL çıpası (v4b TEK DEĞİŞİKLİĞİ #1):** `--kl-kip sft` → forward-KL, β=0,2, T=1, **YALNIZ kaynak==0 (SFT) satırlarında**; **pencere (bilgi) satırları ÇIPASIZ** — v4'ün "öğretmen her satırda" ayarı bilgisayar ölçülen bloklama hatasıydı. Öğretmen = init = pure (öz-çapa; adım-0'da KL≈0). Mikro-parti disiplini AYNEN: SUB_CHUNK=8, öğretmen yığını KL_CHUNK=4 (MPS OOM dersi), satır-seçim kaynak tensorüyle TEK NOKTA `kl_satirlari_sec` (p26 B1; nonzero CPU'da). Kayıp formu p26'dan BİREBİR. Beklenen KL ateşlenme deseni: **mikro-parti başına 6/16 satır** (satır-bazlı %37,5; SFT hedef payı %19,9) — v4'teki 16/16 TÜM-satır deseninden FARKLI; nöbet history'lerinde `kl_satir` kanıtı.

**LR (v4b TEK DEĞİŞİKLİĞİ #2):** peak **1e-4**, warmup **200**, cosine → min **1e-5**. 5e-5 (v4) bilgi tavanına takıldı; 2e-4 (T3.4) çıpasız yıkıcıydı; 1e-4 kanıtsız orta nokta — K1 erken-duruş bu koşumun canlı sigortasıdır. AdamW wd 0,01; clip 1,0; tohum 42. Pencere tavanı 128 (aile dersi).

**Adım planı:** **11.380 opt-adım** (= 1 külliyat epoğu), save-every **500** (atomik `.tmp`+replace ara-checkpoint her nöbette). VAL-nöbeti aynen v4: SFT val CE (val-150 rig) + külliyat pencere-CE (TRAIN avlusu, 25 pencere, tohum 7; resmi VAL K2'de AYRI) + okula tek-görev tripwire + history JSON.

## 3. ÜÇ DONMUŞ KAPI — v4'ten AYSEN (eşikler ve aletler değişmedi)

| Kapı | Metrik (resmi alet) | Eşik | Zemin |
|---|---|---|---|
| **K1 okuryazarlık** | masked SFT val CE (val-150 rig) | **≤ 2,45** | pure 2,3043; v4 final 2,3926 (geçti); P2.6 2,4271 |
| **K2 bilgi** | külliyat pencere-CE, **resmi VAL** (`eval_ce_val.py`, `kulliyat_v2_val.bin`, 500 pencere, tohum 1337) | **≤ 3,23** | pure 3,8205; T3.4 3,1936 (ihlalle); P2.6 3,2118; **v4 3,905 — KALDI** |
| **K3 nöbet** | paket güveni: `test_model.py` 4 SFT görevi top-1 VE güven | **4/4 top-1 VE her görevde ≥ 0,99** | TABAN 0,9997/0,9981/0,9998/0,9986 (4/4) |

**K1 izleme izni (ERKEN DURUŞ) AYNEN:** ilk %20 penceresinde (adım ≤ 2.276; nöbetler 500/1000/1500/2000'de) SFT val CE **2,45'i AŞARSA koşum DURUR** (hüküm `ERKEN_DURUS_K1`; son ara-checkpoint diskte). v4b'de bu izin ek önemdedir: 1e-4 çıpasız-pencere lr riskinin canlı sigortasıdır.

**Hüküm kuralı aynen:** üç kapı da ölçülür; tarife yetmeyen kapı için dürüstçe **KALDI** verilir, eksikler sayısal tanıyla haritalanır. Kapı yetersizliği İLAN'ı değiştirmez.

## 4. PİLOT ölçümleri (50 adım, 05.11.2026; `scratch/v4b_ogrenci_pilot_*`, hüküm: `v4b_ogrenci_pilot_hukum.json`)

| Gösterge | Değer |
|---|---|
| opt-adım / eğitim süresi / ön-hesap | 50 / 186,6 sn / 51,7 sn |
| s/adım (son-20) | **3,52 ± 0,94 sn** (adımlar 2,3–4,3 sn; makine çekişmesi görünür — v4 pilotunda da iki koşum 2,64/5,05 ayrışmıştı) |
| MPS tepe | **1,773 GB** (önerilen sınır 5,33 GB — bol pay; v4 tüm-satır: 1,655 GB) |
| **KL ateşlenme deseni (v4b'in ÖLÇÜM KANITI)** | **satır-bazlı 600/1600 = %37,5 (mikro-parti başına 6/16 satır) — TÜM-değil** ✓ (v4 tüm-satır: 16/16 = %100). Ateşlenen pay = SFT tarafı: hedef-bazlı **%19,9** (karışımın SFT oranı), bağlam-bazlı %37,5. Mikro-parti-bazlı 100/100 ateşlendi — kompozisyon garantisi (her parti 6 SFT satırı içerir; beklenen davranış, kusur değil). KL ort 0,1450 (v4 pilot aynı noktada 0,1252; SFT satırlarında 1e-4 sürüklenme başlangıcı) |
| Nöbet ilk değerleri | adım 0: **2,304304 / 3,9633 / case@0,9998** (init=pure BİREBİR ✓) · adım 25: 2,310661 / 3,9531 / case@0,9998 · adım 50: 2,321840 / 3,9265 / case@0,9998 |
| Erken sinyal | pencere-CE (TRAIN) **−0,037 nat @ 50 warmup adımı** (lr 2,5e-5) — bilgi ekseni HAREKETLİ; v4 pilotu aynı noktada −0,021 (v4b ~1,8× hızlı — serbest pencere gradyanı işliyor; v4 tam koşumda bu ekseni 4,15'e GERİ SARMIŞTI, v4b'de fren kaldırıldı). SFT CE +0,018 → K1 payı 0,128 |
| K1 tripwire | adım 25 nöbeti 2,45 altında (2,3107) — erken-duruş yok, çapa tutuyor |

**TAM koşum ETA (İLAN kuralı aynen):** pilot s/adım × 3 (sürekli-koşum çarpanı) × 11.380 adım → **nokta 33,4 sa; dürüst bant 24,5–42,3 sa**.
**Kalibrasyon dipnotu (dürüst):** v4'te ×3 kuralı AŞIRI söyledi — v4 pilot 2,64 s/adım → tahmin 25,0 sa (bant 23,5–26,6); **gerçek 10,43 sa** (gerçek/pilot ≈ ×1,25; hüküm: `sure_sn` 37.560). Aynı çekişme düzeltmesi v4b'ye uygulanırsa ~11–15 sa bandı çıkar — ama İLAN kuralı değişmez: beyan **24,5–42,3 sa**, kalibrasyon notuyla birlikte. K1 erken-duruş penceresi (~2.276 adım) bu bantta ~5–7 sa'da tamamlanır; her 500 adımda nöbet + atomik ara-checkpoint → durdurma talimatı nöbet başına kanıtlı verir.

## 5. TAM KOŞUM KOMUTU (İLAN onayı SONRASI; controller başlatır)

```
venv/bin/python scripts/train_v4_ogrenci.py \
  --init-model data/anka_taban_pure.pt \
  --sft-path data/pedagogy/pure_base_sft_v4.jsonl \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --kl-kip sft --kl-beta 0.2 --peak-lr 1e-4 --min-lr 1e-5 --warmup 200 \
  --batch 16 --accum 2 --pencere-satir 10 \
  --toplam-adim 11380 --save-every 500 \
  --save-path data/anka_ogrenci_v4b.pt --allow-frozen-write \
  --history-path data/eval/v4b_ogrenci_history.json \
  --hukum-path data/eval/v4b_ogrenci_hukum_2026-11-05.json
```

Koşum sonrası kapı ölçümleri (K1 koşum içi zaten; K2/K3 resmi):
```
K2: venv/bin/python scripts/eval_ce_val.py --model data/anka_ogrenci_v4b.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json \
      --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337
K3: venv/bin/python test_model.py --checkpoint data/anka_ogrenci_v4b.pt \
      --vocab data/rebuild/vocab_anka_r1_33114.json
```

## 6. Dürüst risk beyanı

- **1e-4 çıpasız-pencerelerde okuryazarlık riski (bu koşumun ANA riski):** T3.4 kanıtı — 2e-4 çıpasız koşum okuryazarlığı 0,9998→0,0726 öldürdü. v4b pencere satırlarında KL'yı bilinçli KALDIRIYOR (bilgi bloklamasını çözmek için); okuryazarlığı koruyacak olanlar: SFT-only KL çapası (%19,9 hedef payı pure'a bağlı), SFT CE sinyalinin kendisi ve **K1 erken-duruş** (ilk %20'de SFT CE 2,45'i aşarsa koşum durur — okuryazarlık ölmeden yakalanır; son ara-checkpoint diskte). 1e-4'te bu sigortanın YETMEYECEĞİ açık bir olasılıktır: nöbetler 500'er adımda, pencere içinde ~2000 adımlık sürüklenme penceresi kalır. Sonuç K1/K3'ten çıkmazsa dürüst KALDI.
- **SFT-only KL formatı koruyamazsa:** tüm-satır çıpanın v4'te işi formatı betonlamaktı (K1 2,3926); SFT satırlarına daraltılmış KL + %19,9 CE payı aynı işi yapıp yapamayacağı KANITSIZ. K1 erken-duruş yakalarsa kalan risk sıfırlanır; kaçırsa bile K3 paketi (4/4 + ≥0,99) final hükümde dürüstçe ölçülür — **format korunamazsa KALDI, merdiven/ek-tarif önerisi AYRI İLAN'la gelir.**
- **K2 iddialı:** hedef −0,59 nat (3,8205→≤3,23). v4b'nin serbest pencere gradyanları T3.4'ün −0,63 kazancına yakınsayabilir (1e-4, 2e-4'ten yarı hız), ama 1 epok (11.380 adım) yetmeyebilir → dürüst KALDI + sayısal tanı (pencere-CE eğrisi, lr-merdiven önerisi AYRI İLAN).
- **Bilgi-okuryazarlık GERİ BESLEMESİ:** v4 ölçümü — tüm-satır KL altında pencere-CE 3,96→4,15'e GERİ SARDI (bilgi kazancı koptu). v4b'de pencere satırları çıpasız olduğundan bu geri sarma beklenmez; nöbet history pencere-CE eğrisi bunu kanıtlar. Aksi yönde (pencere-CE düşerken SFT CE'nin fırlaması) K1 tripwire'ının işi.
- **Bellek/OM:** sft kipi öğretmeni mikro-partide 6/16 satırda çağırır (v4: 16/16) → MPS baskısı v4'ten DÜŞÜK (pilot tepe kanıtı bölüm 4'te); OOM riski azalır, disiplin yine de aynen.
