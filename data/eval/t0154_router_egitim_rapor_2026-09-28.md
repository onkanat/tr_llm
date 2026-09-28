# T-0154 — TriModalRouter eğitilmiş-ağırlık (sonuç)

**Hüküm:** **T0154_ROUTER_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T06:58:13Z (UTC) — koşum sonu
**İlan:** `data/eval/t0154_router_egitim_ilan_2026-09-28.md` (sha256 `86acf99a41f1679520838192101b16b17d089b8042ef4f79b9b32b005ea02e7a`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 SIFIR-KILIÇ (çıktı-yok + kaynak-sha birebir) | `{"cikti_var": false, "data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598", "sha_sapma": []}` | GEÇTİ |
| CİHAZ (özellik-çıkarımı: MPS) | `{"mps": true}` | GEÇTİ |
| K5 VERİ-ENVANTER (500/100 x 3; kesişim-0; legal-0) | `{"train_kirilim": {"grammar_core": 500, "pedagogy": 500, "carpenter": 500}, "val_kirilim": {"grammar_core": 100, "pedagogy": 100, "carpenter": 100}, "val_train_kesisim": 0, "legal_ornek": 0}` | GEÇTİ |
| K2 BASELINE (fresh-init val top-1; kayıt-değer) | `{"baseline_val_top1": 0.2867, "rastgele_taban_3sinif": 0.3333}` | GEÇTİ |
| K3 EĞİTİM (val>=0,75 VE baseline+0,15 VE kayıp-azalıyor) | `{"val_top1": 0.93, "baseline": 0.2867, "epoch_kayiplar": [0.3604, 0.2205, 0.189], "train_per_sinif_dogru": {"grammar_core": 500, "pedagogy": 416, "carpenter": 500}, "kirpilan_4096_ustu": 0, "ozellik_sure_sn": 31.9}` | GEÇTİ |
| K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir) | `{"sha_1": "9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500", "sha_2": "9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500"}` | GEÇTİ |
| K6 ŞEMA + ÇIKTI (anahtar-küme birebir + yazım) | `{"anahtar_sayi": 9, "cikti": "data/anka_router.pt", "cikti_sha256": "64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720", "canli_istek": 0}` | GEÇTİ |

## Çıktı

- `data/anka_router.pt` (donmuş-desen yazım; operatör-onayı T-0154) — sha256 `64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720`
- state_dict sha256 `9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500` (determinizm-kanıtı)

## Entegrasyon (AYRI onarım maddesi — bu turda YAPILMADI)

Gateway `create_default` şu an router'ı fresh-init kurar; eğitilmiş
state_dict'in gateway'de yüklenmesi ayrı İLAN'lı kod-onarımıdır.

