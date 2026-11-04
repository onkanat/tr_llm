# P2.5 FAZ-2A — LEGAL CEKETİ EĞİTİMİ: DONMUŞ KAPILAR (İLAN)

**Tarih**: 2026-11-04 | **Durum**: EĞİTİM KOŞUMUNDAN ÖNCE DONDURULDU (İLAN)
**Taban çıpası**: `data/eval/p25_legal_taban_cipasi_2026-11-04.json` (ölçüldü, aşağıda)
**Tarif kaynağı**: `scripts/train_t0209_marangoz.py` (T-0209 ölçülmüş ceket tarifi) +
`scripts/train_p26_sft_replay.py` (kayıp formu/mikro-parti/atomik kayıt iskeleti)
**Kapı sabitleri `scripts/train_legal_ceket.py` yazımında donduruldu; bu dosya
çıpalarla somutlaştırır. Koşum sonrası değiştirilmez.**

## Tarif (donmuş — ölçülmüş T-0209 marangoz ceketinden ÇIKARILDI)

| Parametre | Değer | Gerekçe |
|---|---|---|
| Başlangıç | `data/anka_taban_pure.pt` (sha `5479a2f3…`) | CEKET TABANI pure — P2.6 hükmü; v3 ailesi DEĞİL |
| Öğretmen / pencere / KL | YOK | Saf alan SFT; T-0209'un ölçülmüş davranışı (KL dalı ölü koddu, P2.6-B1) — ölçülmemiş KL uydurulmaz, dal kaldırıldı |
| peak_lr | **1e-5** | marangoz: `PEAK_LR = 1e-5` (train_t0209_marangoz.py:56) |
| min_lr | **2e-6** | marangoz: `MIN_LR = 2e-6` (:57) |
| warmup | **3** | marangoz 15/234 adım (%6,4) → 39 adımlık koşuma oransal ölçekleme |
| batch / accum | 16 / 2 | marangoz (:59-60) |
| weight_decay / clip | 0.01 / 1.0 | marangoz/p26 |
| epochs | **3** | küçük set (413 train); val CE izlenir, artarsa nöbet |
| Kayıp formu | `(CE_none*sm).sum()/(sm.sum()+1e-8)` | T-0209 BİREBİR (sm işaret maskesi; -100 CE'de 0; prompt paydada) |
| Val dilimi | 463 kayıttan **son-50** | T-0209 son-150'nin ölçekli karşılığı (~174 token/kayıt); train = ilk 413 |
| Tokenizer | `literal_entity_mode=True` | FAZ-1 veri sözleşmesi (meta.json); UNK %1,62 bu modda |

## Taban çıpaları (ölçüldü — kapı sayıları bunlara bağlı)

| Ölçüm | Taban (anka_taban_pure) |
|---|---|
| **KAPI-1 çıpası**: legal val CE (masked SFT, son-50, FastSampleAlignedDataset rig, BLOCK=128) | **4.782289** (PPL 119,38) |
| **KAPI-2 çıpası**: kanonik-20 (tohum 42, idx: 12,15,16,44,47,52,57,71,111,114,125,140,216,279,302,327,346,377,379,456; val'deki 1/20) kart4 / etiket16 / ortak-önek | **0.00 / 0.00 / 0.0** |
| **KAPI-3 çıpası**: OOD CE — marangoz#0 / marangoz#1 / bahçıvan#0 | **1.2356 / 0.5890 / 4.8032** (gen: 48/48/39 jeton) |

## DONMUŞ KAPILAR

- **KAPI-1 (val CE)**: ceket legal val CE ≤ 4.782289 − 0.25 = **≤ 4.5323** ("belirgin düşüş" = −0.25 nat; masked SFT val CE metrik kimliği — pencere-CE DEĞİL).
- **KAPI-2 (kanonik doğruluk)**: 20 örneklemde **ilk-16-jeton birebir** (kart türü + kânun/kavram etiketi) oranı **≥ 0.50** VE taban oranından (0.00) katı yüksek. [kart4 = kart türü ilk-4; ikincil renk ölçümü]
- **KAPI-3 (alan sızıntısı nöbeti)**: 3 OOD sorguda (marangoz#0, marangoz#1, bahçıvan#0 — donmuş dosya#indeks) ort (CE_ceket − CE_taban) **≤ +0.15 nat** VE ilk-32 argmax üretim taban-uyumu **≥ 0.70**.

**Genel hüküm**: üç kapı da geçerse **GEÇTİ**; aksi halde **KALDI** — koşum sonrası hiçbir eşik değiştirilmez.