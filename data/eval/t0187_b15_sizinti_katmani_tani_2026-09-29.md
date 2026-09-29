# T-0187 HÜKÜM — B1.5 sızıntı-KATMANI tanısı

- **Damga:** BETİKTEN 2026-09-29T05:16:54Z (`date -u`) · **Yürütücü:** claude
- **Claim/kiralama:** `ok:true` (data/eval + notes) · splits DONMUŞ salt-okuma

## K1 — %43,4 bugün-tekrarı (BETİKTEN, /tmp ölçüm-betiği `t0187_b15_katmani_olcum.py`)

- `test_in_train: 704/1622 (43.4) — etiket=408 icerik=296` — **T-0012 bağımsız
  doğrulamasıyla BİREBİR** (t0012_verification.json PASS); doğrulama-zamani
  (14 Eyl) ile bugün arasında splits 0 bayt değişmiş değil (donmuş).
- `val_in_train: 717/1622 (44.2) — etiket=398 icerik=319` — birebir.
- **(a) TOKEN-NORMALIZE katmanı: 0** — bit-eşit DIŞI ama normalize-eşit sızıntı
  `0/1622`; sızıntının taşıyıcısı encoding değil, bit-birebir aynı string.

## K3 — sızıntı-KATMANI yargısı

**SIİZNTİ-KATMANI = (b) ŞABLON/ARZ-KAPSAMI katmanı** — iki belirgin bileşen:

1. **408 kisa-etetiket:** kapalı-sınıf etiket-arzı (`CASE: CASE_DAT`, `ROOT: iyi`,
   `TENSE: TENSE_PROG` gibi tek-tablolar); farklı tek-kelime sorma aynı etiket-cevabı
   ürettiğinde soru-ayrı ama cevap-birebir-eşit. "Paylaşılan soru = genelleme,
   paylaşılan cevap = sızıntı" politikası (anka_r17 §6b) kapalı-etiket-arzında
   DOĞAL olarak sızdırıyor — tokenizer/encode katmanının KUSURU DEĞİL.
2. **296 içerik-cevabı:** 18-21 kelimelik şablon-önek-i-farklı-fakat-output-birebir
   aynı (ök-ölçüme örnek iki satır, BETİKTEN dökümü raporda). Kök-neden BETİKTEN
   (T-0012 kök-neden kaydından birebir kalmış): `prepare_b1_5_datasets.py:264-268`
   ölü-kod şablonla-çeşitlendirilmiş aynı-cevabı farklı bölmeye dağıtmış.

(c) SPLIT-İNDEX katmanı: soru-düzeyi kesişim 0'dır (split_statistics BETİKTEN;
T-0012'de 0/1622) — **değil.**

## K4 — regen-reçetesinin İLAN-maddesi-önerisi

- **(1) Union-Find çıması:** regen betiğinin bölme-adımı öncesi output-birebir
  kümelerini union-find ile aynı bölmeye indirecek (T-0012 onarım-yolu; mevcut
  splits bunun ÖNCESİ donmuş durumda).
- **(2) Etiket-arzı grubu:** ≤4-kelimelik kapalı etiket-cevabı da union
  yörüngesine girmeli (aynı-cevap farklı-soru ayrımı etiket-sınıfında anlamsız).
- **(3) Eşik:** regen sonrası cevapt-yazılma hedef %0 (soru-düzeyi 0'da kalmalı);
  ölçüm BETİKTEN, İLAN-öncesi.
- **(4) Regen koşumu ve arz-derinliği AYRI operatör-emri** (bu turda koşum YOK).

## Beyan

KOD mutasyonu YOK; canlı 8080'e 0 istek. Kanıt-örnekler BETİKTEN dökümü bu
raporun kaynağındaki ölçüm-betiğinden (T-0012'de zaten bağımsız-PASS).