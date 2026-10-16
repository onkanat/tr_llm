# T-0210 — TriModalRouter 5-Sınıf Saf Taban Model (anka_taban_pure.pt) Hizalama Raporu

**Hüküm:** **T0210_ROUTER_V2_PURE_GECTI** (betikten; deterministik ölçüm)
**Damga:** 2026-10-15T18:16:02Z (UTC) — koşum sonu
**İlan:** `data/eval/t0210_router_v2_pure_ilan_2026-10-15.md` (sha256 `fe3bc26e319095ea2179db747e4c441509446eeec06a66d8f4a7232f721bb774`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 SIFIR-KILIÇ (kaynak-sha birebir eşleşme) | `{"data/anka_taban_pure.pt": "5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598", "data/pedagogy/bahcivan_arena.jsonl": "9fa356b44ebfc7f6d48e387d7dceb12330b36287c94d95d0e8c15b66646baa13", "sha_sapma": []}` | GEÇTİ |
| CİHAZ (özellik-çıkarımı: MPS) | `{"cihaz": "mps"}` | GEÇTİ |
| K5 VERİ-ENVANTER (500/100 x 4; kesişim-0; legal-0) | `{"train_kirilim": {"grammar_core": 500, "pedagogy": 500, "carpenter": 500, "gardener": 500}, "val_kirilim": {"grammar_core": 100, "pedagogy": 100, "carpenter": 100, "gardener": 100}, "val_train_kesisim": 0, "legal_ornek": 0}` | GEÇTİ |
| K2 BASELINE (fresh-init val top-1; kayıt-değer) | `{"baseline_val_top1": 0.0, "rastgele_taban_4aktif": 0.25}` | GEÇTİ |
| K3 EĞİTİM (val>=0,95 VE baseline+0,50 VE kayıp-azalıyor) | `{"val_top1": 1.0, "val_top2": 1.0, "baseline": 0.0, "epoch_kayiplar": [0.1951, 0.042, 0.0358, 0.0344, 0.0322], "train_per_sinif_dogru": {"grammar_core": 500, "pedagogy": 498, "carpenter": 499, "gardener": 499}, "val_per_sinif": {"grammar_core": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}, "pedagogy": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}, "carpenter": {"correct": 100, "total": 100, "acc": 1.0, "precision"` | GEÇTİ |
| K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir) | `{"sha_1": "a2a95957072efbd52b097d711854c1b427132fa679ee30ac20f2c2daba45f25c", "sha_2": "a2a95957072efbd52b097d711854c1b427132fa679ee30ac20f2c2daba45f25c"}` | GEÇTİ |
| K6 ŞEMA + ÇIKTI (5x256 w_g + birebir anahtarlar + kayıt) | `{"anahtar_sayi": 9, "w_g_shape": [5, 256], "cikti": "data/anka_router_v2.pt", "cikti_sha256": "74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2"}` | GEÇTİ |

## Çıktı ve Doğrulama

- Checkpoint: `data/anka_router_v2.pt` — sha256 `74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2`
- State Dict SHA-256: `a2a95957072efbd52b097d711854c1b427132fa679ee30ac20f2c2daba45f25c`
- Validation Top-1 Accuracy: `1.0000` (%100.00)
- Validation Top-2 Accuracy: `1.0000` (%100.00)

