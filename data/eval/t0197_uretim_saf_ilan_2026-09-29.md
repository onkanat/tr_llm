# T-0197 İLAN — Faz-2B üretim-saf veri onarımı

- **Damga:** BETİKTEN 2026-09-29T15:37:15Z · claim `ok:true` · kiralama 4-yol `ok:true` (conflict 0)
- **Onay-geçişi:** operatör emri "T-0197 koşumunu başlat" → awaiting_approval → open

## Kanıt-çıpa TEKRAR-sayım (koşum-öncesi BETİKTEN)
- **PASS** — V1 İngilizce-baskın train/val/test (1599, 298, 268) birebir; V2 meta-prompt 1; V3 zarf 3-form toplam 430 ({'ikili': 79, 'tek_teknik': 86, 'tek_usta': 265}); V4 ≤4-kelime 5036

## 6-çıpa (koşum-başı BETİKTEN — tam-digest)
- test `f106e7d2c7854ea653a6039ac6fc2578ee7b8e2d36926b2763bc7ae2e2926260`
- val `eb96241534e71ae12329de84998cfdc04aaaa50fb8cbcd6236a922d32812f6a4`
- train `c2d8480b866f9fd0b57d7e671eb091c91e5c80cca5a4f133b62e61cdafaae39e`
- vocab `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`
- roots `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`
- halef `e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7`

## Zincir (şartname birebir)
1. `scripts/t0197_uretim_saf_onarim.py` (CPU): T-0196 curated (`data/eval/t0196_kulliyat_arz_dengeli.jsonl`, sha-çıpalı) ÜSTÜNE 3-eylem: P1 dil-dışlama + P2 zarf-canon ('Usta Cevabı: Teknik Çözüm:' tek-önek; 3 map BETİKTEN) + P3 meta-regex-pano atla; P4 = rapor-kolon (×≥20 etiket). Çıktı `data/eval/t0197_uretim_saf.jsonl` + meta.
2. `scripts/train_t0197_uretim_saf.py` (MPS, sandbox-dışı arkaplan): T-0196 eğitim-betik birebir; load=halef (e5eb114e… YALNIZ-OKU); epoch=2 peak_lr=1e-4; init_val-regen-50b; save `data/anka_b1_5_usaf_epoch{1,2}.pt` + best `data/anka_b1_5_usaf.pt` (YENİ ad); PROBE 200-adım; history artımlı `data/eval/t0197_training_history.json`; hüküm `data/eval/t0197_egitim_hukum_2026-09-29.json`.
3. `scripts/t0197_faz3_kabul.py` (MPS): T-0192 kalıbı arm-A eşli-100 + **dil-hizalı alt-küme ROUGE (yeni-birincil gösterge)**: ref-İngilizce-baskın olmayan öğelerde ayrı özet + İngilizce-hedef öğelerde ayrı özet. Eşikler: Tutarsızlık <%5 VE decomp ≥0,35 VE Ezber ≤%5 → `T0197_ONARIM_GECTI`; aksi DUR rc=2. Çıpa karşılaştırma: T-0196 26% / 0,1368.

## Risk-beyan
- 0,35 eşiği kanıtlı-DEĞİL (T-0196: curation tek yetmedi). Dil-hiza kapısı ROUGE'un ~%16 kısmını yapısal olarak hedefliyor — alt-küme karşılığı İLAN-birincil.
- P2 zarf-canon 265 'Usta Cevabı:'-hedefe 'Teknik Çözüm:' etiket-kompozisyonu ekler (prefix-birleştirme; içerik bit-birebir) — beyanlı.
