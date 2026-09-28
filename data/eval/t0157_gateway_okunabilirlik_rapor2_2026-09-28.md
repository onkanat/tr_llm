# T-0157 — gateway insan-okunur üretim (sonuç)

**Hüküm:** **T0157_OKUNABILIRLIK_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T07:56:36Z (UTC) — koşum sonu
**İlan:** `data/eval/t0157_gateway_okunabilirlik_ilan_2026-09-28.md` (sha256 `79827b7589bc91bad359d06435664894b1d2b96f543a0649c4e5b3c03f0e99d5`)
**İlan-2:** `data/eval/t0157_gateway_okunabilirlik_ilan2_2026-09-28.md` (sha256 `f8170578a233d9dc584cbe334d1a47c6f44d665300138e79810f4d4a57dfad1d`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 KAYNAK-ÇIPASI (3 dosya + koşullu router-çıpası) | `{"data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598", "data/anka_router.pt": "64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720", "router_cipasi_kosulu": "T-0156-öncesi", "sha_sapma": []}` | GEÇTİ |
| K5 STATİK-ÖN (py_compile + AST + pyflakes beyaz-listeli) | `{"py_compile": true, "sabit_yapisal": true, "sabit_temizlik": true, "gt_bastirma": true, "gt_fail_closed": true, "pq_temizlik": true, "pq_capitalize_true": true, "yukleyici_tanimli": true, "pyflakes_tum_bulgu": 1, "pyflakes_yeni_bulgu": []}` | GEÇTİ |
| K2 CANLI-ÜRETİM (3 sorgu; sızıntı-∅ + regex-∅ + capitalize) | `{"sorgular": [{"sorgu": "Kırlangıç kuyruğu nedir?", "uretilen_jeton": 17, "yapisal_sizinti": [], "tag_regex_eslesme": [], "decompiled_ilkkarakter": "S", "bos_cikti": false, "entropy_post": 0.4935}, {"sorgu": "Osmanlı Devleti ne zaman kuruldu?", "uretilen_jeton": 2, "yapisal_sizinti": [], "tag_regex_eslesme": [], "decompiled_ilkkarakter": "[", "bos_cikti": false, "entropy_post": 2.8918}, {"sorgu": "Meşe ağacı nedir?", "uretilen_jeton": 4, "yapisal_sizinti": [], "tag_regex_eslesme": [], "decompiled_ilkkarakter": "R", "bos_cikti": false, "entropy_post": 0.6995}], "bastirma_id_sayi": 7}` | GEÇTİ |
| K3 MUTASYON-KANITI (boş→RuntimeError; zayıf→PAD VAR; tam→YOK) | `{"tam_cikti": [1, 2], "tam_yapisal_sizinti": [], "bos_sabir_istisna": "RuntimeError: DURDURULDU (T-0157): vocab dolu ama yapısal-jeton bastırma listesi çözülemedi (boş); sessiz-bastırma", "zayif_sizinti": [0], "zayif_cikti": [0, 0, 0, 0, 0]}` | GEÇTİ |
| K4 DAVRANIŞ-KORUMA (ask 17-anahtar; PQ 16-anahtar; UNK sinyali) | `{"ask_anahtar_sayi": 17, "ask_anahtar_kume_esit": true, "response_text_str": true, "response_text": "", "pq_anahtar_sayi": 16, "pq_anahtar_kume_esit": true, "unk_korunur": true, "epistemic_failure": true, "future_train_recorded": true}` | GEÇTİ |
| K6 CANLI-DOKUNULMAZLIK (ağ istemcisi kurulmadı) | `{"ag_istemcisi": "YOK", "vector_memory": ":memory: (storage_path=None)", "canli_192_168_1_9_istek": 0, "localhost_istek": 0, "create_default_cagrildi": false}` | GEÇTİ |

## Onarım yüzeyi (kod-değişikliği)

- `src/rag/epistemic_agent.py`: `YAPISAL_BASTIRMA_JETONLARI` /
  `TEMIZLIK_JETONLARI` modül-sabitleri; `generate_tokens`'ta
  yapısal-jeton bastırması (-inf; fail-closed boş-liste)
  + `process_query`'de id-düzeyi temizlik + `capitalize=True`

