# P2.5 FAZ-2A — LEGAL CEKETİ EĞİTİMİ: HÜKÜM RAPORU (2026-11-04)

**Görev**: T-0209 marangoz kalıbıyla legal ceketi eğitimi → `data/anka_legal.pt` + donmuş kapılarla değerlendirme.
**Veri**: `data/pedagogy/legal_canonical_sft.jsonl` — 463 kayıt / 80.455 token / UNK %1,62 (FAZ-1).
**Betiği kalıp**: `scripts/train_p26_sft_replay.py` (T-0209 kayıp formu BİREBİR, MPS mikro-parti, atomik kayıt, donmuş yol kapısı). **Yeni betik**: `scripts/train_legal_ceket.py`.
**Kural**: commit YOK (yapılmadı), Qdrant YOK, izlenen dosyalara dokunulmadı (`git status --porcelain | grep -v '^??'` → TEMİZ).

---

## 1. İLAN (koşumdan ÖNCE dondurulanlar)

Kayıt: `data/eval/p25_legal_ceket_kriterler_2026-11-04.md` (eğitim koşumundan önce yazıldı; kapı sabitleri betik yazımında `scripts/train_legal_ceket.py` içinde donduruldu).

**Tarif (T-0209 marangoz betiğinden ÇIKARILDI, `train_t0209_marangoz.py:54-64`)**: başlangıç `data/anka_taban_pure.pt` (v3 ailesi DEĞİL — P2.6 hükmü) · pencere/KL YOK (T-0209'un ölçülmüş davranışı saf SFT — KL dalı ölü koddu, P2.6-B1; ölü kod kopyalanmadı, dal kaldırıldı) · **peak_lr 1e-5 / min_lr 2e-6** (birebir) · warmup 3 (marangoz 15/234 = %6,4 → 39 adımda oransal) · batch 16 × accum 2 · wd 0,01 · clip 1,0 · kayıp formu `(CE_none*sm).sum()/(sm.sum()+1e-8)` (T-0209 BİREBİR) · val = **son-50**/463 (T-0209 son-150'nin ölçekli karşılığı), train = 413 · epok 3 + val-CE-artış nöbeti (en iyi epok geri yüklemeli) · tokenizer `literal_entity_mode=True` (FAZ-1 meta sözleşmesi; UNK %1,62 bu modda ölçüldü).

**Taban çıpaları** (ayrı koşum, `--mod taban-cipasi`; çıpa dosyası: `data/eval/p25_legal_taban_cipasi_2026-11-04.json`):

| Çıpa | Ölçüm |
|---|---|
| legal val CE (masked SFT, son-50, FastSampleAlignedDataset, BLOCK=128) | **4,782289** (PPL 119,38) |
| kanonik-20 (tohum 42; idx 12…456; val'deki 1/20): kart4 / etiket16 / ortak-önek | **0,00 / 0,00 / 0,0** |
| OOD CE: marangoz#0 / marangoz#1 / bahçıvan#0 | 1,2356 / 0,5890 / 4,8032 (gen 48/48/39) |

**Donmuş kapılar**: KAPI-1: final val CE ≤ 4,782289 − 0,25 = **≤ 4,5323** · KAPI-2: kanonik-20 ilk-16 birebir (kart+etiket) ≥ **0,50** VE > taban · KAPI-3: OOD ort ΔCE ≤ **+0,15** VE ilk-32 argmax taban-uyumu ≥ **0,70**. Genel hüküm: üçü de geçerse GEÇTİ. *Koşum sonrası hiçbir eşik değiştirilmedi.*

**Rig doğrulaması**: 4 adımlık pilot (TMPDIR'e kayıt) çıpayı birebir yeniden üretti (4,782289 ✓ deterministik); eğitim döngüsü, atomik kayıt, donmuş-yol kapısı, nöbet mantığı sağlıklı.

---

## 2. HÜKÜM

### TARİF KOŞUMU (marangoz LR 1e-5/2e-6, warmup 3, 3 epok = 39 opt-adım): **KALDI**

| Kapı | Ölçüm | Sonuç |
|---|---|---|
| **KAPI-1** val CE | 4,7823 → 4,7341 (Δ **−0,0482**; bar −0,25) | **KALDI** |
| **KAPI-2** kanonik | kart4 0,00 · etiket16 0,00 · ortak-önek 0,0 | **KALDI** |
| **KAPI-3** OOD nöbeti | ΔCE ort −0,0017 ✓ · uyum32 ort **0,71** (1,00/1,00/0,22) | **GEÇTİ** (sınırda) |

### Tanı koşumları (tarif-dışı, kapı mesafesini haritalamak için; AYNI donmuş kapılar)

| Koşum | LR / epok | val CE | Δ | kart4 | etiket16 | OOD uyum32 | K-1 | K-2 | K-3 |
|---|---|---|---|---|---|---|---|---|---|
| 1 TARİF | 1e-5 / 3 | 4,7341 | −0,048 | 0,00 | 0,00 | 0,71 | ✗ | ✗ | ✓ |
| 2 tanı | 5e-5 / 3 | 4,5882 | −0,194 | 0,00 | 0,00 | 0,44 | ✗ | ✗ | ✗ |
| 3 tanı | 5e-5 / 12 | 4,3189 | −0,463 | 0,05 | 0,00 | 0,12 | ✓ | ✗ | ✗ |

**Genel hüküm: KALDI** — tarif koşumu 3 kapıdan 1'ini geçti. Tanı koşumları bunun rig hatası değil **tarif-yetersizliği** olduğunu ve kapıların aynı koşumda bir arada erişilmediğini ölçtü:

- **val CE ~doğrusal düşüyor LR×epok ile, nöbet hiç ateşlenmedi** (12 epokta bile monoton: 4,692→4,319). KAPI-1'e ilk varış: 5e-5'de **epok 4** (4,5295 ≤ 4,5323); tarife sadık LR (1e-5) ile tahmini ~20-23 epok gerekir (reçetenin ~8 katı).
- **KAPI-3 ters korelasyonlu**: legal öğrenimi arttıkça OOD argmax sadakati monoton bozuluyor (uyum32 0,71 → 0,44 → 0,12; marangoz#1 CE +0,20'ye kadar). Görev 'pencere/karışım YOK' şartıyla P2.6'nın KL/pencere çıpasını bilinçli devre dışı bıraktı; marangoz bunu hissetmedi çünkü **init'i zaten biçim-içindeydi** (marangoz val CE 1,073'ten başlıyordu; legal 4,78'den başlıyor — biçim edinimi, ince-ayar değil).
- **KAPI-2 hiçbir koşumda oluşmuyor**: 12 epokta bile kart4 ancak 0,05 (tek kayıt, önek-4), etiket16 0,00. Üretim hâlâ pedagoji desenini yankılıyor (örn. üretim: `PLURAL : hayır` — pure_base_sft görev-4 kalıntısı); ilk-çıkış top-5'i `case/belge/borç/hesap/bilgi` — **hukuk-adjasansı kelime dağılımına kayış VAR, kart ŞABLONU YOK** (altın ilk jeton `prosedür` top-5'te bile değil). 463 kayıtta kart şablonu edinimi, T-0209 kayıp formunun prompt-payda seyreltmesi (~0,5×) altında, 2-3 epokta (hatta 12'de) ölçülebilir gerçekleşmiyor.

**Verim matematiği**: marangoz ölçülmüş verimi −0,033/234 adım = −1,4e-4 nat/adım; tarif koşumu −0,0482/39 = −1,2e-3 nat/adım (yüzey dikleşmiş, 8,8× verim) — yine de barın 1/5'i. Marangoz verimiyle 39 adımın projeksiyonu −0,0055'ti; ölçülen −0,048 bunun 8,8 katı ama KAPI-1'e yetmez.

### Karşılaştırma: marangoz ceketi neden işledi, legal neden (bu reçeteyle) olmadı

| | Marangoz (T-0209) | Legal (bu koşum) |
|---|---|---|
| Veri | 6.123 SFT + 1.530 pencere | 413 SFT (pencere yasak) |
| Init val CE | **1,073** (biçim biliniyor) | **4,782** (biçim bilinmiyor) |
| Opt-adım | 234 (1 epok) | 39 (3 epok) |
| Sonuç | −0,033 (ince-ayar) | −0,048 (biçim edinimi gerekiyor) |

---

## 3. RAPOR

### 3.1 Betik farkları (marangoz tarifi vs uygulanan — `scripts/train_legal_ceket.py`)

| Öğe | Marangoz betiği | Uygulanan | Gerekçe |
|---|---|---|---|
| Başlangıç | anka_taban_pure.pt | aynı | CEKET TABANI pure (P2.6 hükmü) |
| Öğretmen/KL | anka_base_v2 + β=0,2 KL (ama **ölü kod** — hiç koşmadı) | dal KALDIRILDI | Ölçülmüş tarife sadakat; ölçülmemiş KL uydurulmaz; pencere yoksa KL'nin dayanağı da yok |
| Pencere karışım | 1.530 pencere (4:1) | YOK | Görev şartnamesi: saf alan SFT |
| peak/min lr | 1e-5 / 2e-6 | **aynı** | Görev: tarifi betikten ÇIKAR ve uygula |
| warmup | 15 (234 adımda) | **3** (39 adımda) | Oransal ölçekleme (%6,4) |
| batch/accum/wd/clip | 16/2/0,01/1,0 | aynı | — |
| Epok | 1 | **3** (val-CE nöbetli, en iyi epok geri yükleme) | Görev (e): 2-3 epok + nöbet |
| Val | son-150 | **son-50** | Ölçekli karşılık (~174 token/kayıt) |
| Kayıp formu | `(CE_none*sm).sum()/(sm.sum()+1e-8)` | **BİREBİR** (+ P2.6 not-7 mikro-parti: payda tam partiden, SUB_CHUNK=8) | MPS bellek; değer/gradyan birebir |
| Kayıt | düz `torch.save` | **atomik** (tmp+replace) + `--allow-frozen-write` donmuş-yol kapısı + save-every + epok ckpt'leri | p26 iskeleti |
| Tokenizer | mode'suz | **literal_entity_mode=True** | FAZ-1 meta sözleşmesi (UNK %1,62 o modda; mode=False apostrof siler) |
| Ölçüm | val CE + pencere CE | val CE + **kanonik-20 (argmax)** + **OOD-3 nöbeti** | Görev 3(b)/(c) |

### 3.2 Eğitim ölçümleri

| Koşum | Süre (eğitim) | Adım | s/adım (son-20) | val CE eğrisi (epok) |
|---|---|---|---|---|
| 1 TARİF (1e-5, 3e) | 81,1 sn | 39 | 1,80±0,36 | 4,7583 → 4,7411 → 4,7341 (monoton; nöbet yok; en iyi epok 3) |
| 2 tanı (5e-5, 3e) | 66,3 sn | 39 | 1,39±0,06 | 4,6797 → 4,6116 → 4,5882 (monoton) |
| 3 tanı (5e-5, 12e) | 283,6 sn | 156 | 1,61±0,06 | 4,692→4,599→4,530→4,478→4,436→4,404→4,379→4,359→4,345→4,334→4,326→4,319 (12 epok monoton; **nöbet hiç ateşlenmedi**) |

Train loss (tarif koşumu): 3,95 (adım 1) → 3,28-4,24 bandında (adım başına kayıt zorluğu varyansı; val CE sinyali monoton). Ara kayıtlar: save-every 8 (atomik) + her epok sonu ayrı ckpt (`_e<n>.pt`). Bellek: çökme/OOM yok (137 sessiz-kill, komut-tampon hatası hiç olmadı); tek model + AdamW durumu, öğretmen forward'u yok (P2.6'nın GPU wired-limit tuzağı bu koşumda yapısal olarak yok — KL/pencere dalı kaldırıldı).

### 3.3 Kapı ölçümleri (detay)

**KAPI-1**: çıpa 4,782289 → tarif 4,734058 (−0,0482) · 5e-5/3e 4,588212 (−0,1941) · 5e-5/12e 4,318863 (**−0,4634 ✓**). Metrik kimliği: masked SFT val CE (T-0209 2,3049 ile AYNI ölçüm; pencere-CE değil).

**KAPI-2** (kanonik-20, tohum 42, idx: 12,15,16,44,47,52,57,71,111,114,125,140,216,279,302,327,346,377,379,456; 19'u train, 1'i val): taban 0,00/0,00 · tarif 0,00/0,00 (ortak-önek 0,0) · 5e-5/12e kart4 0,05 (kayıt 71, ortak-önek 4), etiket16 0,00 (ortak-önek ort 0,65). Argmax üretim tavanı 48 jeton (ilk-16 metrik için yeterli).

**KAPI-3** (OOD: marangoz_canonical_sft#0, #1, bahcivan_canonical_sft#0 — donmuş dosya#indeks):
| Sorgu | ΔCE: tarif / 5e-5-3e / 5e-5-12e | uyum32: tarif / 3e / 12e |
|---|---|---|
| marangoz#0 | −0,0049 / −0,0164 / **+0,1086** | 1,00 / 0,19 / 0,19 |
| marangoz#1 | +0,0039 / +0,0482 / **+0,2020** | 1,00 / 0,91 / 0,00 |
| bahçıvan#0 | −0,0040 / −0,0743 / −0,0801 | 0,22 / 0,22 / 0,19 |
| **ort** | −0,0017 / −0,0142 / +0,0768 | **0,71 ✓ / 0,44 ✗ / 0,12 ✗** |

### 3.4 Öneriler (FAZ-2B karar noktaları — üst ajana)

1. **KAPI-1 en düşük maliyetle**: 5e-5 × 4-5 epok yeterli (val CE ~4,44-4,53; KAPI-1 ✓, KAPI-3 ✗). Hazır checkpoint'ler: `data/anka_legal_tani12e.pt_e4.pt` (4,5295 — barın altında) / `_e5.pt` (4,4360).
2. **KAPI-3'ü koruyarak öğrenim imkânsız görünüyor** (pencere/KL çıpası şartnamesi dışında bırakıldı): alan-dışı sadakat ile alan öğrenimi ölçülen biçimde ters korelasyonlu. Spec revizyonu (P2.6 pencere/KL mekanizmasının legal ceketine izni) önerilir.
3. **KAPI-2 mevcut veri ölçeğinde erişilmez**: kart şablonu edinimi için ~marangoz ölçeği veri (5-6K kayıt) VEYA çok daha uzun eğitim gerekir; ya da kapı CE-tabanlı biçim ölçütüne (ör. ilk-16 jeton CE) çevrilir — 463 kayıtta birebir üretim katı.
4. **Deliverable kararı**: `data/anka_legal.pt` = TARİF ceketi (şartnameye sadık; KAPI-3'ü geçen tek koşum). Daha iyi val CE'li modeller tanı dizininde hazır (`anka_legal_p26lr.pt`, `anka_legal_tani12e.pt` + epok ckpt'leri); terfi (yeniden adlandırma) üst ajanın hükmüdür.

### 3.5 Dosya yolları + sha256

| Artefakt | sha256 |
|---|---|
| `scripts/train_legal_ceket.py` (YENİ betik; taban-cipasi/egitim modlu, nöbetli, kapı-sabitli) | — |
| `data/anka_legal.pt` (**TARİF ceketi — deliverable**) | `6456be4ea1bebe4c5f35429c61ee34ecc2104920a066f91e9ae5e37872b7df3b` |
| `data/anka_legal.pt_e1/e2/e3.pt` (tarif epok ckpt) | — |
| `data/anka_legal_p26lr.pt` (tanı: 5e-5/3e) | `8794708c444ddf07b277dc187bf40c8aa69c6c1f50aa3669bb98d61e8d524455` |
| `data/anka_legal_tani12e.pt` (tanı: 5e-5/12e; KAPI-1 ✓) | `6e60710e03a4cae04cee82619b171c3991d78b16409d01e00684b90a7aad67b3` |
| `data/anka_legal_tani12e.pt_e5.pt` (KAPI-1 crossing +pay) | `7be4e37f856e5d3eeeda6913460fc1d890d990c2b6fda4859d8c40b7aaa7c1cc` |
| `data/eval/p25_legal_ceket_kriterler_2026-11-04.md` (İLAN) | — |
| `data/eval/p25_legal_taban_cipasi_2026-11-04.json` (çıpa) | — |
| `data/eval/p25_legal_ceket_hukum_2026-11-04.json` (tarif hükmü) | — |
| `data/eval/p25_legal_ceket_hukum_p26lr_2026-11-04.json` (tanı hükmü) | — |
| `data/eval/p25_legal_ceket_hukum_tani12e_2026-11-04.json` (tanı hükmü) | — |
| `data/eval/p25_legal_history.json` / `_p26lr.json` / `_tani12e.json` (kayıp eğrileri) | — |
| `data/eval/p25_legal_egitim_marangozlr_log_2026-11-04.txt` (tarif koşum günlüğü) | — |
| Girdi çıpaları: `anka_taban_pure.pt` `5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360` · `legal_canonical_sft.jsonl` `69670bc6ddbc033cff4a38a044ef0ccd5a82c278cfd8f564755ea70634734e9c` · `vocab_anka_r1_33114.json` `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` | — |

### 3.6 Yeniden üretim

```bash
# çıpa
venv/bin/python scripts/train_legal_ceket.py --mod taban-cipasi
# TARİF koşumu (deliverable)
venv/bin/python scripts/train_legal_ceket.py --mod egitim --epochs 3 --save-every 8 \
  --save-path data/anka_legal.pt --allow-frozen-write
# tanı koşumları (aynı kapılar; yalnız --peak-lr/--min-lr/--warmup/--epochs ve --save-path değişir)
```