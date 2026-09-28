# T-0153 — canlı-migrate kristal_bellek → anka_bellek (sonuç)

**Hüküm:** **T0153_MIGRATE_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T06:44:43Z (UTC) — koşum sonu
**İlan:** `data/eval/anka_bellek_canli_migrate_ilan_2026-09-28.md` (sha256 `6b33266f07049d6349baaca6cad94bda3cd573ec2d3fbf1087442e38a7e36823`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 B0 + ÇIPA-ŞEMA | `{"anka_bellek_var": false, "status": "green", "nokta": 36, "dense_ad": "dense", "dense_size": 768, "dense_distance": "Cosine", "sparse_adlar": ["sparse"], "sparse_modifier": "idf", "on_disk_payload": true}` | GEÇTİ |
| K2 ARŞİV | `{"dosya": "data/eval/anka_bellek_canli_migrate_eski_nokta_arshivi_2026-09-28.jsonl", "nokta": 36, "sha256": "13c928194a1004266aed1b0436759983bf313e6db9063b2cb731cbf64268bbe4"}` | GEÇTİ |
| K3 KOPYA-BİREBİR (36/36) | `{"eski_n": 36, "yeni_n": 36, "uyusmayan": []}` | GEÇTİ |
| K4 SELF-RETRIEVAL | `{"sorgu_id": 1, "top1": [1]}` | GEÇTİ |
| K5 DOKUNULMAZLIK (9 foreign SABİT) | `{"once": {"elektor_articles": 90122, "octave_articles": 100, "rapberry_pi_pico_all_articles": 436, "rendergit_01_articles": 32, "rendergit_02_articles": 100, "rendergit_03_articles": 39, "rp2040_articles": 106, "sdr_articles": 54, "türk_articles": 5}, "sonra": {"elektor_articles": 90122, "octave_articles": 100, "rapberry_pi_pico_all_articles": 436, "rendergit_01_articles": 32, "rendergit_02_articles": 100, "rendergit_03_articles": 39, "rp2040_articles": 106, "sdr_articles": 54, "türk_articles": 5}}` | GEÇTİ |
| K6 ESKİ-SİLME (wrapper; fail-closed) | `{"wrapper_sildi": true, "kristal_bellek_yok": true, "anka_status": "green", "anka_nokta": 36}` | GEÇTİ |

## Son canlı envanter (koleksiyon → nokta)

| Koleksiyon | Nokta |
|---|---|
| anka_bellek | 36 |
| elektor_articles | 90122 |
| octave_articles | 100 |
| rapberry_pi_pico_all_articles | 436 |
| rendergit_01_articles | 32 |
| rendergit_02_articles | 100 |
| rendergit_03_articles | 39 |
| rp2040_articles | 106 |
| sdr_articles | 54 |
| türk_articles | 5 |

## Arşiv

Silme-öncesi 36 nokta tam-arşivi: `data/eval/anka_bellek_canli_migrate_eski_nokta_arshivi_2026-09-28.jsonl` (sha256 hüküm-JSON K2 içinde).

## Kaynak-düzeltmesi (İLAN'da koşum-öncesi beyanlı)

RAPOR2 §7'nin `data/pedagogy_canonical/**` kaynak-beyanı yanlıştı;
gerçek kaynak P2'de `data/realistic_rag/test_natural_150.jsonl` idi.
Yöntem nokta-kopyasıdır (T-0150 core.py onarımı sonrası
yeniden-derleme sapma riski; birebirlik K3'te ölçüldü).

