# T3.4 Eğitim Özeti — anka_taban_v3 (2026-11-04)

İLAN: Bu dosya, T3.4 devam-ön-eğitiminin tamamlanma bildirimindeki ölçülmüş son-durum bloğunu repoya kalıcı kaydeder. Tam stdout, tamamlanan arka plan sürecinin (proc_fe4b285af6c6) tamponundadır; bu özet son-durum bloğunu aynen taşır.

## Koşum

```
venv/bin/python train.py --device mps --seed 34 \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --data data/kulliyat_v2/kulliyat_v2.bin \
  --block-size 128 --pretrain \
  --base-model data/anka_taban_pure.pt \
  --save-path data/anka_taban_v3.pt --allow-frozen-write \
  --lr 2e-4 --steps 14226 --toplam-adim 14226 \
  --warmup-steps 200 --min-lr 2e-5 \
  --batch-size 16 --save-every 2000 --save-optimizer --loss-report
```

- Veri: `data/kulliyat_v2/kulliyat_v2.bin` = 29.135.254 uint16 token (text 2.121.627 + wiki 27.013.627; meta kapısı BİREBİR).
- Başlangıç: 01:50, bitiş: 14:45 (bildirim EXIT=0).
- 1 nominal epok = 14.226 adım (B=16 × T=128 → 2.048 hedef/adım).

## Son-durum bloğu (bildirimden aynen)

```
Toplam Süre:       46511.58 saniye  (≈12,9 sa; ort. 3,27 sn/adım)
Başlangıç Kaybı:   3.5928
Bitiş Kaybı:       3.3141  (tek adım örneklemi — dağılım kullanılmalı)
Kayıp Dağılımı:    son60  3.3300 ± 0.1397 (n=60)
                   son200 3.3116 ± 0.1706 (n=200)
                   medyan 3.3741 (n=14226)
Son LR:            0.000020 (cosine tam anneal, min-lr'a ulaştı)
[MASKE] Maskesiz Hedef Oranı: %100.00 (pretrain modu — T-0177 beklenen davranış)
[ckpt] adım 14226: 'data/anka_taban_v3.pt' güncellendi (son kayıt, atomik)
[ckpt] adım 14226: 'data/anka_taban_v3.pt.opt.pt' güncellendi (AdamW momentleri, atomik)
```

## Çıktılar

- `data/anka_taban_v3.pt` — 373.735.672 bayt, 14:45
- `data/anka_taban_v3.pt.opt.pt` — 747.486.705 bayt (AdamW + scheduler durumu)

## Notlar

- Ara checkpoint'ler save-every 2000 ile yazıldı; son koşum yalnızca son kaydı bırakır (üzerine yazım).
- Pilot kalibrasyonu 1,18 sn/adım demişti; gerçek koşum 3,27 sn/adım (fan'sız M2 ısıl kısılma + bellek baskısı ~3×; ders skill'de).
- Değerlendirme hükümleri: bkz. `t34_v3_degerlendirme_2026-11-04.md` (aynı dizin).