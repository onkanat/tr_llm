# MİMARİ DOĞRULAMA PAKET-5 SONUÇ — Gateway→Öğrenme Kanalı (T-0145 İLAN-1 / T-0148 TUR-B İLAN-2 onarım-doğrulama)

**Damga:** 2026-09-27T16:25:16Z · **Hüküm (BETİKTEN):** **DUR (rc=2)**

| Kapı | Durum |
|---|---|
| K1_FAZ_A_ENVANTER | GEÇTİ |
| K2_ARZ_CIPA | GEÇTİ |
| K3_KANAL_YAZARI | GEÇTİ |
| K4_TUKETICI_OKUMA | GEÇTİ |
| K5_TUKETICI_DERLEME | GEÇTİ |
| K6_RUN_TRAINING_ILKELI | DÜŞTÜ |
| K7_GATEWAY_YAZIM | GEÇTİ |
| K8_HTTP_TEYIDI | GEÇTİ |
| K9_DOKUNULMAZLIK | GEÇTİ |

## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen)

| Ölçüm | İLAN'lı | Ölçülen | Durum |
|---|---|---|---|
| epi_kanal_kapisi_similarity_threshold | 0.85 | 0.85 | GEÇTİ |
| epi_record_yazim_sitesi | 1 | 1 | GEÇTİ |
| epi_record_sema_anahtar_sayisi | 14 | 14 | GEÇTİ |
| epi_future_train_path_paramli | True | True | GEÇTİ |
| epi_os_makedirs_satiri | 180 | 180 | GEÇTİ |
| gw_ask_telemetri_anahtar_sayisi | 17 | 17 | GEÇTİ |
| gw_inject_default_koleksiyon | kristal_bellek | kristal_bellek | GEÇTİ |
| gw_target_collection_sahete_secici | False | False | GEÇTİ |
| gw_resolve_target_var | True | True | GEÇTİ |
| gw_inject_reasoning_paylasim_client | False | False | GEÇTİ |
| gw_inject_reasoning_koleksiyon | muhakeme_bellek | muhakeme_bellek | GEÇTİ |
| gw_http_endpoint_kumesi | ['/api/backlog', '/api/check', '/api/inject', '/api/query', '/api/status'] | ['/api/backlog', '/api/check', '/api/inject', '/api/query', '/api/status'] | GEÇTİ |
| gw_localhost_kurulum | 3 | 3 | GEÇTİ |
| retrain_zorunlu_fail_closed | 3 | 3 | GEÇTİ |
| retrain_cmd_flag_kumesi | ['--batch-size', '--data', '--device', '--load-path', '--lr', '--save-path', '--steps', '--vocab'] | ['--batch-size', '--data', '--device', '--load-path', '--lr', '--save-path', '--steps', '--vocab'] | GEÇTİ |
| retrain_cmd_vocab_yok | False | False | GEÇTİ |
| retrain_cmd_load_path_yok | False | False | GEÇTİ |
| retrain_archive_success_gate | True | True | GEÇTİ |
| retrain_archive_sifirlama_modu | w | w | GEÇTİ |
| retrain_replay_buffer_yolu | data/pedagogy/high_school_foundation_dataset.jsonl | data/pedagogy/high_school_foundation_dataset.jsonl | GEÇTİ |
| retrain_replay_samples_default | 25 | 25 | GEÇTİ |
| retrain_output_bin_default | data/train_future_finetune.bin | data/train_future_finetune.bin | GEÇTİ |
| retrain_oversample_factor_default | 20 | 20 | GEÇTİ |
| train_py_default_vocab | data/rebuild/vocab_base_32852.json | data/rebuild/vocab_base_32852.json | GEÇTİ |
| train_py_jeton_guard | True | True | GEÇTİ |
| train_py_vocab_flag_destekli | True | True | GEÇTİ |
| supervisor_retrain_threshold | 5 | 5 | GEÇTİ |
| merak_router_state_dict_yukleme | 0 | 0 | GEÇTİ |
| gercek_kanal_once_bayt | 0 | 0 | GEÇTİ |
| envanter_digest_once | fe7fc7649737ed35 | fe7fc7649737ed35 | GEÇTİ |

## FAZ-B — kanal koşumu

- B1 canlı-döngü: ask **20/20** · istisna 0 · is_high_similarity **3/20** (İLAN'lı 0) · ask-17-anahtar=True
- B1 probe-yazım: tetiklenme **0** (istisna 0) · satır 0→1 · şema-tam 1/1 · append-teyit=True
- B3 tüketici-okuma: probe **1** (beklenen 1) · gerçek-yol **0**
- B4 tüketici-derleme: .bin boyut 72320 (uint16=True) · meta backlog 1 · total 26 · oversample 20 · toplam jeton 36160 · **max-jeton-id 32831** (< 32.852: True) · geri-tokenize=True
- B5 run_training İLKELİ: status **success** · samples 1 · süre 13.67 sn · **canlılık-imzası: Başlangıç Kaybı 3.1524** (T-0148 5A resume-bandı: fresh-imza GÖRÜNMEZ=True · bant-içi=False) · arşiv 8→9 satır (taşıma-teyit=True) · probe-future 0 bayt · save .pt=True · default-bin DOKUNULMADI (mtime/digest-çıpa öncesi==sonrası; önceden-var=False — koşum yazmadı)
- B6 gateway-yazım (probe): inject p5_probe_bellek count 1 · geri-okuma skor 1.3333 · **sahte-seçici KAPANDI (5B): target 'olmayan_ad_p5_x' → istisna=True · probe-count 1 (yazım YOK) · check-sahte=True · reasoning-istisna=True · sahte-ad sunucuda YOK=True** · check top-1 skor 1.3333
- B7 HTTP: port 54542 · 5-endpoint kodlar [200, 200, 200, 200, 200] · backlog-samples 0 · query-telemetri 17-anahtar
- B8 dokunulmazlık: GEÇTİ · kristal 36→36 · digest fe7fc7649737ed35→fe7fc7649737ed35 · kanal 0→0 bayt

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p5_ilan3_2026-09-27.md (İLAN — koşum ÖNCESİ, REVİZYON-1) | `da4a6e6def6e9cef387202c6c17e3887e9c60c9b86cad3b7c557caf53cfabf79` |
| dogrulama_p5_ogrenme_kanali.py (koşulan betik) | `4071e18519b5066c64499516847099a9a4453bb9b18f9112956fc881da40e73d` |
| data/anka_base_v2.pt (DONMUŞ taban) | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` |
| data/rebuild/vocab_anka_r1_33114.json (DONMUŞ sözlük) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| replay-buffer (DONMUŞ) | `59e2f2786d0ec0d1574a77e5b7354e02ae957345a1b4ddce945d0db19527bdf1` |
| data/future_train_vector.jsonl (GERÇEK kanal — 0 bayt) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `fe7fc7649737ed35` |
| data/eval/mimari_dogrulama_p2_onarim_hukum_2026-09-27.json | `4b6d183741e742a506c94fd18dc8e9321c7e30d06f001e23ce1baf619d0a9348` |
| data/eval/mimari_dogrulama_p3_onarim_hukum_2026-09-27.json | `18384263b9895b22e1eb2dd4eb7080b77602060807c42169639fdc2be9de5da1` |
| data/eval/mimari_dogrulama_p4_onarim_hukum_2026-09-27.json | `2a35d6c4cd9746c6fe66a310319b41d4540785ad0d855603f813d32d3810dc14` |
| mimari_dogrulama_p5_onarim_hukum_2026-09-27.json (HÜKÜM JSON — BETİKTEN) | `897bb7c7249d4f534780e0650f4211c0615a089113cd64d3be71f999fea8712a` |

