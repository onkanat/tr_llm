# T-0075 — SFT koruması parti-başına: meşru koşumu öldürür mü?

Ölçüm `2026-09-18T14:36:18Z` → `2026-09-18T14:36:24Z` · **CPU-only (GPU'ya dokunulmadi)** · Tohum 0 · 60,000 pencere/küme · 2,000 parti/(küme,B)

**Soru:** T-0074'te eklenen koruma (`train.py`: maskeleme 0 hedef bırakırsa `RuntimeError`) **parti başına** ateşliyor. `get_batch` rastgele pencere seçtiği için, sağlıklı bir SFT külliyatında parti şans eseri `<OUTPUT>`'suz pencerelere denk gelirse **meşru koşum ölür**. Antigravity'nin 14:29:32Z yanıtı bunu "SFT'de 0 hedef **asla** meşru değildir, RuntimeError %100 doğru" diye **ölçmeden** onayladı; bu rapor o hükmü ölçer.

## 0. Vekil doğrulaması (ucuz istatistik gerçek fonksiyona karşı)

| ölçüm | değer |
|---|---|
| `has_output(pencere)` ↔ gerçek `mask_prompt_targets` sonrası hedef>0 uyum | **1.0** (400/400) |
| denek içinde `<OUTPUT>` içeren | 381 |
| çelişki | 0 |

## 1. Kümeler

| külliyat | blok | blok kaynağı | p (`<OUTPUT>`'lu pencere) | ateşleme B=8 | ateşleme B=16 | ateşleme B=32 |
|---|---|---|---|---|---|---|
| `data/train_chat_balanced.bin` | 128 | meta | **95.1850%** | 0.000% | 0.000% | 0.000% |
| `data/train_chat_balanced_clean.bin` | 128 | meta | **95.0367%** | 0.000% | 0.000% | 0.000% |
| `data/train_balanced_sft_v2.bin` | 64 | meta | **72.3883%** | 0.000% | 0.000% | 0.000% |
| `data/train_chat_sft.bin` | 64 | GERI DUSUS (64) | **42.1800%** | 1.000% | 0.000% | 0.000% |
| `data/train_balanced_sft.bin` | 64 | meta | **23.3400%** | 10.800% | 1.250% | 0.000% |
| `data/train_deep_sft.bin` | 128 | meta | **98.0667%** | 0.000% | 0.000% | 0.000% |
| `data/train_pedagogy_highschool.bin` | 64 | meta | **92.6333%** | 0.000% | 0.000% | 0.000% |
| `data/train_future_finetune.bin` | 64 | meta | **7.2317%** | 57.500% | 30.000% | 8.850% |
| `data/train_all_chosen.bin` | 64 | GERI DUSUS (64) | **0.1183%** | 99.000% | 98.250% | 96.450% |
| `data/train.bin` | 64 | GERI DUSUS (64) | **0.1800%** | 98.100% | 97.400% | 93.450% |
| `data/train_corpus.bin` | 64 | GERI DUSUS (64) | **0.0000%** | 100.000% | 100.000% | 100.000% |
| `data/anka_a1_pretrain.bin` | 128 | meta | **0.0017%** | 99.950% | 99.850% | 99.900% |

Not: `<OUTPUT>` id'si kanonik sözlükte **1706**; kontrol edildi — depodaki **her** sözlük neslinde
(31.357 · 31.322 · 32.137 · 32.156 · 32.816 · 32.852) `<OUTPUT>` **1706**'dır (özel token bloğu önde paylaşılır),
bu yüzden bayat sözlüklü külliyatlarda da id geçerlidir.

## 2. HÜKÜM (ilan edilen kural, ölçümden önce yazıldı)

> **HUKUM CURUDU: en az bir SFT kulliyatinda koruma mesru kosumu >%0,1 oranda oldurur**

- En yüksek ateşleme oranı: **100.0000%** (data/train_corpus.bin)
- Kural: `> %0,1` ⇒ çürütür · `= 0` ⇒ destekler · arada ⇒ seyrek/şansa bağlı

## 3. Kontrol (ön-eğitim külliyatı — `--pretrain` yolunda koruma zaten çağrılmıyor)

- `data/anka_a1_pretrain.bin` B=8: ateşleme **100.0%**
- `data/anka_a1_pretrain.bin` B=16: ateşleme **99.9%**
- `data/anka_a1_pretrain.bin` B=32: ateşleme **99.9%**

Beklenen: `%100` (düz metin külliyatında `<OUTPUT>` yok ⇒ T-0073'te ölçülen ölü hedef).
Bu, ölçümün **pozitif kontrolüdür**: koruma orada ateşlemeli, burada ateşlememeli.

## 4. İddia edilmeyenler

1. **Model kalitesi ölçülmedi.**
2. Ölçüm **örnekleme**dir (60,000 pencere/küme); p çok küçükse (`~1e-6`) `%0` sonucu **yokluk değil kanıtsızlık** olabilir — bu yüzden ham `p` ve analitik `(1-p)^B` birlikte verilir.
3. Korumanın `--pretrain` yolunu etkilemediği **kod okumasıyla** söylenir (dallanma `else` içinde), koşumla sınanmadı.
4. Bu ölçüm GPU kullanmadı; T-0073 sondasının ölçümüne çakışmadı.

Üreticiler (tam sha256):

| dosya | sha256 | bayt |
|---|---|---|
| `scratch/t0075_guard_tani.py` | `d9a7cd9d9ea4869755fd596f461683a49ed6d09c4c953d2cc990cf1be2a99c55` | 15211 |
| `scripts/train_step_demo.py` | `1bac5e9263faca86e513d83ed1256dcc04305f741f86b714bc1f115b4e7f5dd7` | 15192 |
| `train.py` | `ab66c3a3db2ab34a376de180b5d3e1a8b5bf46e79ffc8889965c2f1c637c273d` | 9366 |

İlgili: [[sessiz-no-op-mps-sifir-kayip]], [[yurutucu-raporlari-bagimsiz-dogrulanmali]],
[[ilan-edilen-kural-ayirt-edici-olmali]], [[cift-yonunu-alan-adindan-oku]].
