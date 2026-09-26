# G6a (T-0137) RAPOR — KÖK-NEDEN ÖLÇÜM GÖREVİ (marangoz vs Bahçıvan kıyası) — 2026-09-26 (claude)

Operatör emri: "kök neden ölçüm görevini aç — marangoz verisi ile Bahçıvan
veri setini kıyasla". ILAN ≠ RAPOR: ilan `bahcivan_g6a_kokneden_ilan_2026-09-26.md`
(koşum ÖNCESİ damga). Hüküm BETİKTEN (`scratch/t0137/bahcivan_kok_neden_kiyas.py`
`587301ed…` → `bahcivan_kok_neden_hukum.json` `64bdc41e…`, rc=0; elle sayı YOK).
Bu TANISAL görevidir — kabul ayracı YOK.

## Hüküm ölçekleri (üç kısım, betikten)

### Kısım C — cevap-CE: **CE_DUSTU** (öğrenme VAR — ayırt edici bulgu)

| Model | maskeli CE (Bahçıvan heldout cevap hedefleri, n=213 pencere) | PPL karşılığı |
|---|---|---|
| taban `anka_base_v2.pt` (`d0f415f3…`) | **5,2103** ± 0,8242 | ≈ 183 |
| ft `seg_3.pt` (`bd68c450…`) | **3,3682** ± 0,5084 | ≈ 29 |
| **fark (taban − ft)** | **1,8421 ± SE 0,0664** (≈28σ) | — |

- **Aynı istemler, iki model** (istemsizlik beyanlı) — fark yalnız parametreler.
- Model Bahçıvan cevaplarını **CE ekseninde öğrenmiş** (PPL 183 → 29).
  **Dozaj/çekicilik hipotezi ÇÜRÜDÜ**: 6,8 epoch yeterliydi; veri öğrenildi.
- A13 tutarsızlık %43 ve A9 BAND_ALTINDA ile birleşince kök neden
  **ÜRETİM/decoding'dedir** — model cevabı öğrenmiş ama üretirken kaybediyor.

### Kısım B — ROUGE uzunluk-kesişim ayrıştırması (beyanlı simülasyon `K ≈ ROUGE × (ref+hyp)/2`)

| | ROUGE-L ort | üretim ort (kelime) | referans ort | **K_mutlak ≈** |
|---|---|---|---|---|
| ÖLÇÜM-1 marangoz | 0,1374 | 38,62 | 20,66 | **4,07** |
| ÖLÇÜM-2 Bahçıvan | 0,0365 | **61,32** | 33,40 | **1,73** |

- **K_oranı B/M = 0,4245**: modelin Bahçıvan cevaplarıyla mutlak kelime
  kesişimi marangozdakinin %42'si — ROUGE farkı yalnız uzunluk
  normalizasyonundan DEĞİL; mutlak kesişim de düşük.
- **Üretim uzuyor:** Bahçıvan istemlerinde üretim ort **61 kelime**
  (referans 33) — model marangoz uzun-cevap biçimini kısa-kalıp Bahçıvan
  istemine taşıyor; tutarsızlık %43 (yüklemsiz-son + jeton-döngü —
  `evaluate_carpenter_anka.py:514-516` tanımı) bu kaymanın yan izi.
- Simülasyon hükmü DEĞİŞTİRMEZ (İLAN beyanı) — ROUGE dalları KABUL_YOK'tan
  değişmez; bu kısım etkisini ayıran tanıdır.

### Kısım A — veri kıyas (Bahçıvan arena vs marangoz külliyat)

| Eksen | Bahçıvan (2.199) | marangoz katalog (5.400) | marangoz carpenter (5.400) |
|---|---|---|---|
| soru uzunluk ort/medyan | **4,14 / 4** | 11,94 / 12 | 12,37 / 12 |
| cevap uzunluk ort/medyan | **33,88 / 33** (20-61) | 22,04 / 22 (5-40) | 21,08 / 22 |
| benzersiz soru oranı | 1,0000 | 0,6289 (tekrar %37,1) | 0,6289 |
| kalıp homojenliği (max payı) | 0,206 | 0,1783 | 0,182 |

- Bahçıvan: **kısa kalıp soru (4 kelime) + uzun cevap (34 kelime)** —
  soru-cevap uzunluk asimetrisi marangozun tersine (12→22).
- Cevap-uzunluk hipotezi (referanslar kısa ⇒ ROUGE tavanı düşük) **ÇÜRÜDÜ**:
  Bahçıvan referansları marangozdan daha UZUN.
- Marangoz katalog soru tekrarı %37,1 (olgu-turları) — Bahçıvan %0,0; kalıp
  homojenliği benzer (~%18-21). Veri-kalite ekseninde kök-neden YOK.

## Bulgu ölçeği (ölçülenler; tahmin beyanı DEĞİL)

1. **Öğrenme VAR** (CE fark 1,8421 ± 0,0664; ≈28σ; PPL 183→29) — dozaj ve
   veri-kalite hipotezleri ölçüyle ÇÜRÜNDÜ.
2. **Kusur üretimde**: CE'de bilinen cevabı üretirken model 61 kelimelik
   marangoz-tarzı uzunluk + düşük mutlak kesişim (K 1,73) + %43
   yüklemsiz/döngü üretir. Bahçıvan üretimi kısa-kalıp istemde uzun-cevap
   biçimine kayıyor.
3. **Ölçümlenmiş adaylar (izleyen görev için beyanlı):** üretim
   sıcaklığı/decoding (greedy vs beam), `<OUTPUT>` zarf uyumu Bahçıvan
   isteminde, cevap-sonlandırma (İLAN-3b ceket §4 G4 kaydı: "cevap-sonlandırma
   dengesi"). Bu üçü ÖLÇÜLMEMİŞTİR — bu rapor bunları ölçmez.

## Kanıt digest tablosu (tam sha256)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/bahcivan_kok_neden_hukum.json` (hüküm) | `64bdc41e3eaf8c9a151ff33047631ef87614f2ba642d634a456dbe9c6b1323b5` |
| `scratch/t0137/bahcivan_kok_neden_kiyas.py` (betik) | `587301edf68f46ab9712d08f17b892ff1afab7892c04e6e83208ec05e89eae11` |
| `scratch/t0137/kok_neden_kiyas.log` (log) | `4d14385e063a8fa2a0ff51d04383cde65aca1a230a19a75742718aab1c48f552` |
| `data/eval/bahcivan_g6a_kokneden_ilan_2026-09-26.md` (İLAN) | `0e0bc5d67548c21076cfc39f0afc656359c73510d9cc3f938387f624d0c59534` |
| girdiler (salt-okunur): arena `da7bdddd…` · taban `d0f415f3…` · ft `bd68c450…` | (önceki faz kanıtlarıyla birebir) |

## Dürüst kayıtlar

1. **İlk koşum rc=1 — CE şekil kusuru:** `F.cross_entropy(logits[0], y)`
   KristalLM'in `(logits, loss)` TUPLE çıktısını yanlış çözdü
   ("input batch_size (1) vs target 128"); düzeltme kanonik CE yoluna geçti
   (`model(x, targets, ignore_index)` — train.py deseni). Ayraç DEĞİŞMEDİ;
   hüküm ikinci koşumdan (rc=0). İki koşum da log'da.
2. **Kısım C n=213** (220 değil): 7 pencere atlandı (kayıt > blok 128 veya
   boş hedef) — beyanlı `atlanan_kayit=7`, ayrac kapsamı daraltmaz (fark 28σ).
3. Kısım B K_mutlak SİMÜLASYONDUR (ortalama-toplamı yaklaşımı) — hüküm
   bağlamaz; İLAN'da beyanlı.
4. Marangoz katalog `instruction` alanı soru; `input` alanı olgu-girdisi
   (şema katalog dosyasından okundu, varsayılmadı).

## Sonuç dalları

- **CE_DUSTU + üretim-uzunluk taşıması → kök neden ÜRETİM fazında**; izleyen
  ölçüm eksenleri (sıcaklık/decoding/OUTPUT-zarf/sonlandırma) AYRI görevdir.
- ROUGE hükümleri değişmez: A9 BAND_ALTINDA + A13 IHLAL KABUL_YOK kalır.
- Checkpoint silme + commit OPERATÖR kapıları bekliyor. G6b (T-0138) antigravity'ye açık.

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · kanonik kod IMPORT ·
marangoz heldout DOKUNULMAZ (salt-okunur) · elle sayı YOK · data/** salt-okunur ·
`git add -A` YASAK · commit operatör kapısıdır.