# P3 · AŞAMA 1+2 — ÜÇ KAYNAKLI YETENEK KOŞUMU İLANI (ön-kayıtlı)

**Damga:** 23 Eyl 2026 06:41 (+03) · **Yazım:** koşumdan ÖNCE · Plan:
`~/.claude/plans/enchanted-wiggling-moon.md` (P3 — onaylı). Operatör kararı
(23 Eyl): **V7 carve ile geri dön** — P2'nin "ceket eğitimde OLMAZ" kuralı, kendi
T-0094/T-0097 carve politikasından (held-out 539 = eval · 4.861 = yasal eğitim;
r17_build V7: cevap ekseni 0/0, kayıt kesişimi 0, ezber %0) KATIYDI ve tek
yetenek-uyumlu külliyatı eğitimden çıkarmıştı.

## 1. Girdiler (meta-digest'li, ölçülmüş)

| girdi | değer | kaynak |
|---|---|---|
| taban | `scratch/anka_p2_ince_ayr_kos/seg_2.pt` (`ca8b007b…`, Aşama 3.1 nihai, CE 3,4438) + **carry** (101 param, adim=2000) | koşum başında digest kapısı |
| wiki | `data/anka_a1r_pretrain.bin` (781.250 pencere) | kanonik |
| ceket | `data/train_carpenter_specialization_anka_r18.bin` — sha256 `c9d964d9…`, 1.568.123 jeton / 22.283 kayıt / **12.250 pencere**, vocab 33114 digest `f9940a8d…`, **held-out carve AYNI** (V8 ölçek kanıtı) | T-0097 r18 derlemesi |
| SFT mix | `data/rebuild/anka_p2_sft_mix.bin` — sha256 `233ca2e3…` (125.814 pencere) | ceketsiz kanıtlı derleme |
| held-out çıpası | `data/eval/anka_r17_heldout_2026-09-20.jsonl` sha256 **`c397eb08…`** — koşum başı/sonu birebir olmak ZORUNDA (T-0096 deseni) | V7 carve |

**Sızıntı beyanı:** ceket r18 bin'in içeriği 5.400 kaynak ceket külliyatının
held-out'suz (V7) kısmı + T-0097'nin 125 yeni olgu ×450 perspektif büyütmesidir;
held-out 539 cevap-grubu düzeyinde carve'lıdır (cevap ekseni 0/0 kanıtlı). Soru
ekseni çakışması (input 48 / instruction 83) bilinçli ve beyanlıdır —
"paylaşılan soru = genelleme, paylaşılan cevap = sızıntı" politikası.

## 2. Ölçülmüş çalışma girdisi — ceket penceresi maskesiz hedef oranı

n=200 pencere örnekleme (seed 42): **ort %45,53 · medyan %44,14 · std %14,39 ·
sıfır-hedef pencere 0/200** — talimat penceresinin %3,8/pencere değerine karşı
~12× sinyal yoğunluğu (cevap, pencerenin yarısını dolduruyor). T-0073 sıfır-hedef
tuzağı ceket dalında yok.

## 3. Tarif (sürücü: `scratch/anka_p3_surucu.py` — YENİ; P2 desenini İMPORT eder)

* **Patern (pencere idx % 5, deterministik):** 0 ⇒ wiki (%20) · 1,2 ⇒ ceket
  (%40) · 3,4 ⇒ SFT mix (%40). Smoke kanıtı (CPU 50 adım): dağılım
  **80/160/160 = %20/%40/%40 birebir**.
* lr peak **1e-4** · warmup 50 + cosine (toplam 6.000) · min_lr 1e-6 · clip 1,0 ·
  weight_decay 0,01 · b8 blok 128 · seed 42 · **6.000 adım = 3 × 2.000 segment**.
* Ceket maruziyeti: 6.000 adım × b8 × %40 ≈ 19.200 ceket pencere ≈ **1,6 epoch**
  (T-0096'nın ~16 epoch'unun 1/10'u — seyrelti riski ilanlı DAL ile yönetilir).
* Kapılar: H1 · L1 (devam: ln V imzası BOZUKLUK) · CARRY digest · **CE** (her
  segmentte ≤ 3,8887, kanonik assert'li) · **TEPE** (aşağıda) · artımlı yazım
  (`scratch/anka_p3_kos/`) · held-out çıpası başı/sonu.
* **Mini sonda:** her segment sonunda KANONİK kap alt süreçte
  (`evaluate_carpenter_anka.py --n 100 --seed 42 --ceket-ekseni`,
  baseline = koşum BAŞLANGICI — birikimli LM bedeli ölçülür).
* Süre: ~6.000 × 0,42-0,46 sn ≈ 45 dk + 3 sonda ≈ 1 saat · sandbox DIŞI MPS.

## 4. TEPE kapısı (önceden ilanlı, fail-closed)

1. **ezber ≥ %10** ⇒ DUR (tepe = önceki segment; ezber kapısı kabul şartı).
2. **ROUGE-L düşüşü > 0,05** önceki sonda göre ⇒ DUR (tepe = önceki segment).
   Gerekçe: ölçüm SE ≈ std/√100 (r33'te 0,0154); −0,05 ≈ >2 SE — işaret
   ölçülebilir (gürültüden-küçük-farkın-işareti-yok dersi). T-0097 24K'da tepe
   yapıp gerilemişti.
3. Kapı sınaması İKİ DALI ölçüldü (rc=0): +0,01 ⇒ DUR=False · −0,06 ⇒ DUR=True ·
   ezber %10 ⇒ DUR=True.

## 5. Kabul ve ilanlı beklenti

* Koşum kabulü: CE kapısı + TEPE kapısı + held-out çıpası birebir + üç sinyal
  (rc/sentinel/sonuc.json).
* **Beklenti (ölçümden, tahmin değil):** ROUGE 0,0056 → 0,1-0,3 bandı
  (T-0096 referans 0,3035); tutarsızlık %87 → %20 bandı; kesişim %0 → T-0096
  bandı (%10); **CE ≤ 3,8887** — "Aşama 1 altyapısı T-0096'nın LM bedelini
  +%75'ten +%10'a indirir" iddiasının ölçümü.
* Kalibre eşiklere hüküm: kesişim 10,92 · ROUGE 0,35 · tutarsızlık 5,0 · ezber 10,0.

## 6. Dallar (önceden ilanlı; aşama başına en fazla 1)

* **CE DÜŞERSE:** ceket %40→%25 + wiki %20→%35, carry ile TEK tekrar
  (wiki arz kaldıracı: tr-00001.parquet hazır ama bu dalda büyütülmez).
* **ROUGE < 0,05 kalırsa:** ceket epoch yetersiz ⇒ carry ile +6.000 adım TEK
  devam; hâlâ yoksa program BİTER, teşhis "carve arzı yetersiz" yazılır.
* **TEPE gerilemesi:** tepe segment nihai — geriye koşum yok.

## 7. Çıktılar

* Koşum: `scratch/anka_p3_kos/` (segments.jsonl append · sonuc.json atomik ·
  SONUC_BITTI sentinel yalnız rc==0).
* Sondalar: `scratch/anka_p3_kos/sonda_seg_N.json` · kapanış:
  `data/eval/anka_p3_sonuc_2026-09-23.md` — digest'ler betikle, damga betikten.

## 8. Beyanlar

* Genel SFT karışımı ve ölçüt eşikleri BU programda DEĞİŞMEZ (T-0097 seyrelti
  kanıtı olmadan büyütme yok; döngü-sonlandırma: ölçüt ancak yeni insan tavanı
  ölçümüyle değişir).
* Modül yolu bu programda koşulmaz (P2 Aşama 3.2 ölçüldü: delta içerik taşımıyor).
* Smoke dizini `scratch/anka_p3_smoke/` kanıt olarak korunur (CPU 50 adım).