# P2 · AŞAMA 1 — EĞİTİM KODU ONARIMI SONUCU (train.py)

**Damga:** 23 Eyl 2026 · Operatör onaylı P2 planının Aşama 1'i tamamlandı ("eğitim kodu
serbest" kararı bağlayıcı). Ölçüm kapıları ve donmuş veri koruması **aynen** kaldı —
dokunan yalnızca `train.py` ve testleridir.

---

## 1. Yapılan dört onarım (plan tablosunun 1-5 maddesi)

| # | onarım | uygulama |
|---|---|---|
| 1 | **Maske çakışması** | `MASKE = -100` tek sözleşme; `maske_pad_hedefleri()` PAD hedeflerini
−100'e yazar, forward'a yalnız `ignore_index=-100` geçilir. Eski `ignore_index=<PAD>` yolu
SFT'nin −100'ünü geçersiz sınıf bırakıyordu — CPU'da IndexError FIRLATIR (testle kanıtlı),
MPS'te sessiz 0,0/garbaj dönerdi (T-0073 sınıfı). `--no-pad-mask` davranışı bit-bit korunur. |
| 2 | **Scheduler** | `get_lr()` — B1 şablonu (`train_step_b1_canonical.py:201-242`). Yeni
bayraklar `--warmup-steps` (0=kapalı), `--toplam-adim`, `--min-lr` (1e-6). İkisi de 0 ⇒ lr
sabit, logda LR alanı YOK (mevcut davranış bit-bit). Şablonun ilerleme tuzağı KAPANDI:
cosine `toplam_adim` sonunda min_lr'de DURUR, geri yükselmez (progress 1.0'da kırpılır). |
| 3 | **Grad clip** | `--clip` bayrağı, VARSAYILAN 1,0 (plan onaylı); `--clip 0` kapatır.
`optimizer.step()`'ten ÖNCE uygulanır (B1 sırası: lr güncelle → clip → step). |
| 4 | **`--weight-decay`** | Varsayılan 0,01 = eski sabit ⇒ davranış değişmedi, artık ölçülebilir. |
| 5 | **Scheduler carry** | Yan dosya payload'ına `"scheduler"` bloğu eklendi (warmup_steps,
toplam_adim, min_lr, peak_lr). Devam koşumu bayrak vermediğinde eğri SÜRÜR — cosine tepeyi
yeniden başlatmaz (ölçülen kusur sınıfı: carry'siz ilk güncelleme 1,73×, T-0092). |

## 2. Ölçüm kanıtı

**CPU smoke** (sandbox içi, 30 adım, seed 7): kayıp sonlu 10,4459 → 6,6419 — sıfırdan
koşumun canlılık imzası ≈ ln 33.114 = 10,41 ✓ (başlangıç kaybı bantta). Scheduler aktif
koşumda LR loglandı (adım 30: 0,000883 — cosine ilerleyişiyle uyumlu).

**Yan dosya** (`--save-optimizer` + `--warmup-steps 2 --toplam-adim 8`):
`scheduler={'warmup_steps': 2, 'toplam_adim': 8, 'min_lr': 1e-06, 'peak_lr': 0.001}` ✓.
Warmup ilk adım lr'si = peak×1/2 = 0,0005 ✓.

## 3. Testler (10 yeni/güncel)

* `tests/test_train_scheduler.py` (YENİ, 6 test): sabit-peak (scheduler kapalı eşlenik) ·
  warmup monoton artış + peak aşmama + sonrasında sabit · cosine yarıda (peak+min)/2 ·
  **toplamı aşan adımlar min_lr'de DURUR** (şablonun geri yükselme tuzağının kapısı) ·
  warmup+cosine birlikte · carry eğrisi kalıcılığı.
* `tests/test_pad_loss_mask.py` (+3 test): PAD→−100 yazımı + girdi mutasyon yemez ·
  `--no-pad-mask` bit-bit · **çakışma kapısı**: yeni tek-maske yolu −100 konumlarını yok
  sayar (logit manipülasyonu kaybı değiştirmez) VE PAD ölçeğini korur; eski yolun
  CPU'da IndexError fırlattığı testte kanıtlanır (MPS'te sessiz 0,0 sınıfı).
* `tests/test_train_optimizer_state.py` (+1 test, 1 genişleme): G2 payload'da `scheduler`
  bloğu + alanları + kapalı-koşumda 0/0 yazımı; yeni G3b: devam koşumu scheduler durumunu
  yükler ve LR alanını basar.

## 4. Tam suite

`venv/bin/pytest`: **290 passed, 1 failed** — tek düşen bilinen sandbox istisnası
(`test_agent_gateway_http_server_endpoints`, socketserver PermissionError; sandbox dışında
geçer). Değişiklik öncesi taban 280 passed ⇒ +10 net.

## 5. Digest tablosu (değişen dosyalar)

| dosya | sha256 |
|---|---|
| `train.py` | `0edd5e2cbef48ae5a51cb1b3f65f14afeea14729eaa0486584ece8748cdf2558` |
| `tests/test_train_scheduler.py` | `eca9b329c2396b228c7825f54932ebd904a65aa1b9ee1ad575c4236d6f249912` |
| `tests/test_pad_loss_mask.py` | `ab7ef5569d7a368723728e9ca2edcd3740b060f0c01d30bdc724c2453059c07e` |
| `tests/test_train_optimizer_state.py` | `81c24fca3616f2c4c9afd21faef1aacf9903490f7063b1b7e2fab5a011591631` |

## 6. Davranış değişimi beyanı (kim etkilendi)

* **SFT + PAD-maskeli koşumlar** (varsayılan): kayıp ARTIK DOĞRU hesaplanır — eski yol
  −100 hedefleri geçersiz sınıf sayıyordu (CPU: çökme; MPS: sessiz bozuk değer). Eski
  değerlerle kıyaslama YAPILMAMALI (ölçek değişimi beklenir).
* **Pretrain koşumlar**: PAD yok sayımı aynı konumları hedefler ⇒ ölçek değişmez.
* **Bayraksız koşum**: yalnız "Grad clip: 1.0" satırı yeni; clip varsayılan 1,0 AKTİF
  (plan onaylı onarım — eski koşumlar gradientsız-clip davranışında koşuyordu).
  Scheduler kapalı, weight_decay 0,01 (değişmedi).

## 7. Sıradaki

Aşama 2a — SFT külliyatını vocab 33114'e yeniden derleme (`data/rebuild/anka_p2_sft_mix.bin`):
ceket 5.400 + held-out 539 satır HARİÇ; sonra ilan + gece koşumu.