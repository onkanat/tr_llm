# T-0161 — Soru Filtresi Şartnamesi ve Router Koşul Raporu

**Hüküm:** **T0161_SORU_FILTRESI_SARTNAMESI_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T17:23:05Z (UTC)
**İlan:** `data/eval/t0161_soru_filtresi_ilan_2026-09-28.md` (sha256 `67854aca450b110c9d84239ebddf83d64d5d06904c13a0285d7950695c4e3d25`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 KAYNAK-DİGEST & DEVİR KORUMASI (4-dosya; anka_router.pt korunur) | `{"sha_kontrol": {"data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598", "data/anka_router.pt": "44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a"}, "sha_sapma": [], "sinif_kaynak_1_dosya_sayi": 4}` | GEÇTİ |
| CİHAZ (özellik-çıkarımı: MPS) | `{"mps": true}` | GEÇTİ |
| K5 VERİ-ENVANTER (Kol-A: 395<600 yetersizlik; Kol-B: 6653 tekil; ? oranı=1.0) | `{"pedagogy_toplam_kayit": 7540, "pedagogy_tekil_toplam": 7048, "kol_a_soru_icermeyen_tekil": 395, "kol_a_yetersiz_395_kucuk_600": true, "kol_b_soru_iceren_tekil": 6653, "secilen_600_soru_isareti_orani": 1.0, "secilen_pedagogy_parenting_uye": 0, "grammar_core_havuz_kesisim": 0, "train_kirilim": {"grammar_core": 500, "pedagogy": 500, "carpenter": 500}, "val_kirilim": {"grammar_core": 100, "pedagogy": 100, "carpenter": 100}, "val_train_kesisim": 0, "legal_ornek": 0}` | GEÇTİ |
| K2 BASELINE (fresh-init Kol-B val top-1; kayıt-değer) | `{"baseline_val_top1": 0.2267, "rastgele_taban_3sinif": 0.3333}` | GEÇTİ |
| K3 EĞİTİM (val>=0,75 VE baseline+0,15 VE kayıp-azalan VE >=490/500) | `{"val_top1": 0.9933, "baseline": 0.2267, "epoch_kayiplar": [0.105, 0.0118, 0.0053], "train_per_sinif_dogru": {"grammar_core": 500, "pedagogy": 500, "carpenter": 500}, "train_alt_sinir": 490, "ozellik_sure_sn": 36.3}` | GEÇTİ |
| K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir) | `{"sha_1": "c8ced1f0cb989c08adc3ed41dec47f4f3fc0a18936d10cbb7e5c05a70ddac1ee", "sha_2": "c8ced1f0cb989c08adc3ed41dec47f4f3fc0a18936d10cbb7e5c05a70ddac1ee"}` | GEÇTİ |
| K7 KARIŞIM-MATRİSİ (train+val 3x3; satır-toplamı kırılım birebir) | `{"train": {"matris_gercek_satir_tahmin_sutun": [[500, 0, 0], [0, 500, 0], [0, 0, 500]], "per_sinif_dogruluk": {"grammar_core": 1.0, "pedagogy": 1.0, "carpenter": 1.0}, "satir_toplamlari": [500, 500, 500]}, "val": {"matris_gercek_satir_tahmin_sutun": [[100, 0, 0], [0, 98, 2], [0, 0, 100]], "per_sinif_dogruluk": {"grammar_core": 1.0, "pedagogy": 0.98, "carpenter": 1.0}, "satir_toplamlari": [100, 100, 100]}}` | GEÇTİ |
| K6 ŞEMA & DONMUŞ-MODEL KORUMASI (9 anahtar; anka_router.pt korunur; canlıya 0 istek) | `{"anahtar_sayi": 9, "sema_ok": true, "anka_router_pt_korundu": true, "anka_router_pt_sha256": "44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a", "canli_istek": 0, "vector_memory": "KULLANILMADI (betik ağ istemcisi kurmaz)"}` | GEÇTİ |
| K8 ENTEGRASYON DOĞRULAMASI (in-memory TriModalRouter yükleme → sd-sha teyit) | `{"yuklenen_sd_sha256": "c8ced1f0cb989c08adc3ed41dec47f4f3fc0a18936d10cbb7e5c05a70ddac1ee", "beklenen_sd_sha256": "c8ced1f0cb989c08adc3ed41dec47f4f3fc0a18936d10cbb7e5c05a70ddac1ee", "anahtar_sayi": 9, "hata": null}` | GEÇTİ |

## Kök-Neden ve Havuz Karşılaştırması (Kol-A vs Kol-B)

- **Kol-A (Soru işaretli metinleri eleme):** 4 dosyada geriye kalan tekil metin sayısı **395**'tir. Router için gerekli tekil örnek sayısı **600** (`500 train + 100 val`) olduğundan, `395 < 600` yetersizlik durumunu oluşturur.
- **Kol-B (Soru filtresi / Soru işaretli metinleri tutma):** 4 dosyada soru işareti içeren tekil metin sayısı **6.653**'tür. 600 örnek başarıyla seçilmiştir (`?` oranı: `%100,00`).

## Model ve Ağırlık Koruması

- `data/anka_router.pt` **korundu** (üzerine yazılmadı; sha256 `44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a`).
- Yeni eğitilen model `state_dict` sha256: `c8ced1f0cb989c08adc3ed41dec47f4f3fc0a18936d10cbb7e5c05a70ddac1ee` (deterministik kanıt).

## Karışım-Matrisi (K7)

```json
{
  "train": {
    "matris_gercek_satir_tahmin_sutun": [
      [
        500,
        0,
        0
      ],
      [
        0,
        500,
        0
      ],
      [
        0,
        0,
        500
      ]
    ],
    "per_sinif_dogruluk": {
      "grammar_core": 1.0,
      "pedagogy": 1.0,
      "carpenter": 1.0
    },
    "satir_toplamlari": [
      500,
      500,
      500
    ]
  },
  "val": {
    "matris_gercek_satir_tahmin_sutun": [
      [
        100,
        0,
        0
      ],
      [
        0,
        98,
        2
      ],
      [
        0,
        0,
        100
      ]
    ],
    "per_sinif_dogruluk": {
      "grammar_core": 1.0,
      "pedagogy": 0.98,
      "carpenter": 1.0
    },
    "satir_toplamlari": [
      100,
      100,
      100
    ]
  }
}
```

