# T-0156 — Pedagogy daraltma: router yeniden-eğitimi (sonuç)

**Hüküm:** **T0156_PEDAGOGY_DARALTMA_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T16:19:27Z (UTC) — koşum sonu
**İlan:** `data/eval/t0156_pedagogy_daraltma_ilan_2026-09-28.md` (sha256 `f1d3f2e9f061841fe4edee5bcccff8c1d2a496fc25cc85ffad9b77d31ecd2060`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 DEVİR-ÇIPASI + KAYNAK-DİGEST (devir==64527c72…; 4-dosya) | `{"cikti_var": true, "sinif_kaynak_1_dosya_sayi": 4, "mevcut_cikti_sha256": "64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720", "devir_cikti_sha256": "64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720", "data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598", "sha_sapma": []}` | GEÇTİ |
| CİHAZ (özellik-çıkarımı: MPS) | `{"mps": true}` | GEÇTİ |
| K5 VERİ-ENVANTER (4-dosya; havuz=7.048; parenting-0; gc-0) | `{"sinif_kaynak_1_dosya_sayi": 4, "pedagogy_havuz_tekil": 7048, "pedagogy_havuz_beklenti": 7048, "secilen_pedagogy_parenting_uye": 0, "grammar_core_havuz_kesisim": 0, "train_kirilim": {"grammar_core": 500, "pedagogy": 500, "carpenter": 500}, "val_kirilim": {"grammar_core": 100, "pedagogy": 100, "carpenter": 100}, "val_train_kesisim": 0, "legal_ornek": 0, "pedagogy_secilen_soru_isareti_orani": 0.9433}` | GEÇTİ |
| K2 BASELINE (fresh-init daraltılmış-val top-1; kayıt-değer) | `{"baseline_val_top1": 0.24, "rastgele_taban_3sinif": 0.3333, "eski_baseline_notu": "T-0154 0,2867 sayısı TAŞINMAZ (İLAN)"}` | GEÇTİ |
| K3 EĞİTİM (val>=0,75 VE baseline+0,15 VE kayıp-azalan VE >=490/500) | `{"val_top1": 1.0, "baseline": 0.24, "epoch_kayiplar": [0.1154, 0.0195, 0.0184], "train_per_sinif_dogru": {"grammar_core": 500, "pedagogy": 498, "carpenter": 499}, "train_alt_sinir": 490, "beklenti_notu": "500/500 beklenti-beyanı (KAPI değil; İLAN)", "kirpilan_4096_ustu": 0, "ozellik_sure_sn": 27.8}` | GEÇTİ |
| K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir) | `{"sha_1": "d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf", "sha_2": "d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf", "devir_sd_sha256": "9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500"}` | GEÇTİ |
| K7 KARIŞIM-MATRİSİ (train+val 3×3; satır-toplamı kırılım birebir) | `{"train": {"matris_gercek_satir_tahmin_sutun": [[500, 0, 0], [0, 498, 2], [0, 1, 499]], "per_sinif_dogruluk": {"grammar_core": 1.0, "pedagogy": 0.996, "carpenter": 0.998}, "satir_toplamlari": [500, 500, 500]}, "val": {"matris_gercek_satir_tahmin_sutun": [[100, 0, 0], [0, 100, 0], [0, 0, 100]], "per_sinif_dogruluk": {"grammar_core": 1.0, "pedagogy": 1.0, "carpenter": 1.0}, "satir_toplamlari": [100, 100, 100]}}` | GEÇTİ |
| K6 ŞEMA + ÇIKTI (anahtar-küme birebir; anka_router.pt yeniden-yazım) | `{"anahtar_sayi": 9, "cikti": "data/anka_router.pt", "cikti_sha256": "44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a", "devir_cikti_sha256": "64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720", "canli_istek": 0, "vector_memory": "KULLANILMADI (betik ağ istemcisi kurmaz)"}` | GEÇTİ |
| K8 ENTEGRASYON-ÇIPASI (dosyadan yükleme → sd-sha teyit) | `{"yuklenen_sd_sha256": "d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf", "beklenen_sd_sha256": "d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf", "anahtar_sayi": 9, "t0155_bayatlik_beyani": "scripts/dogrulama_t0155_router_entegrasyon.py K2 (9b85f951…) ve K3 (0,93 / 279-300) çıpaları ESKİ router'a bağlıdır; yeni yazım sonrası o betiğin yeniden koşumu K2/K3'te DÜŞER — BEKLENEN durumdur; betik DOKUNULMAZ, tarihsel hüküm artefaktı DEĞİŞTİRİLMEZ.", "hata": null}` | GEÇTİ |

## Devir kaydı

- `data/anka_router.pt` **yeniden yazıldı**: eski `64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720` → yeni `44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a`
- state_dict sha256 `d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf` (determinizm-kanıtı; devir `9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500`)

## T-0155 bayatlık beyanı (bilinçli)

scripts/dogrulama_t0155_router_entegrasyon.py K2 (9b85f951…) ve K3 (0,93 / 279-300) çıpaları ESKİ router'a bağlıdır; yeni yazım sonrası o betiğin yeniden koşumu K2/K3'te DÜŞER — BEKLENEN durumdur; betik DOKUNULMAZ, tarihsel hüküm artefaktı DEĞİŞTİRİLMEZ.

## Karışım-matrisi (K7)

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
        498,
        2
      ],
      [
        0,
        1,
        499
      ]
    ],
    "per_sinif_dogruluk": {
      "grammar_core": 1.0,
      "pedagogy": 0.996,
      "carpenter": 0.998
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
        100,
        0
      ],
      [
        0,
        0,
        100
      ]
    ],
    "per_sinif_dogruluk": {
      "grammar_core": 1.0,
      "pedagogy": 1.0,
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

