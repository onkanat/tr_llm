# MİMARİ DOĞRULAMA PAKET-5 SONUÇ — Gateway→Öğrenme Kanalı (T-0145)

**Damga:** 2026-09-27T13:37:23Z · **Hüküm (BETİKTEN):** **P5_GECTİ (rc=0)**

| Kapı | Durum |
|---|---|
| K1_FAZ_A_ENVANTER | GEÇTİ |
| K2_ARZ_CIPA | GEÇTİ |
| K3_KANAL_YAZARI | GEÇTİ |
| K4_TUKETICI_OKUMA | GEÇTİ |
| K5_TUKETICI_DERLEME | GEÇTİ |
| K6_RUN_TRAINING_ILKELI | GEÇTİ |
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
| gw_target_collection_sahete_secici | True | True | GEÇTİ |
| gw_inject_reasoning_koleksiyon | muhakeme_bellek | muhakeme_bellek | GEÇTİ |
| gw_http_endpoint_kumesi | ['/api/backlog', '/api/check', '/api/inject', '/api/query', '/api/status'] | ['/api/backlog', '/api/check', '/api/inject', '/api/query', '/api/status'] | GEÇTİ |
| gw_localhost_kurulum | 2 | 2 | GEÇTİ |
| retrain_zorunlu_fail_closed | 3 | 3 | GEÇTİ |
| retrain_cmd_flag_kumesi | ['--batch-size', '--data', '--device', '--lr', '--save-path', '--steps'] | ['--batch-size', '--data', '--device', '--lr', '--save-path', '--steps'] | GEÇTİ |
| retrain_cmd_vocab_yok | True | True | GEÇTİ |
| retrain_cmd_load_path_yok | True | True | GEÇTİ |
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

- B1 canlı-döngü: ask **20/20** · istisna 0 · is_high_similarity **0/20** (İLAN'lı 0) · ask-17-anahtar=True
- B1 probe-yazım: tetiklenme **1** (istisna 0) · satır 0→2 · şema-tam 2/2 · append-teyit=True
- B3 tüketici-okuma: probe **2** (beklenen 2) · gerçek-yol **0**
- B4 tüketici-derleme: .bin boyut 77720 (uint16=True) · meta backlog 2 · total 27 · oversample 20 · toplam jeton 38860 · **max-jeton-id 32831** (< 32.852: True) · geri-tokenize=True
- B5 run_training İLKELİ: status **success** · samples 2 · süre 12.73 sn · arşiv 4→6 satır (taşıma-teyit=True) · probe-future 0 bayt · save .pt=True · default-bin DOKUNULMADI (mtime/digest-çıpa öncesi==sonrası; önceden-var=True — koşum yazmadı)
- B6 gateway-yazım (probe): inject p5_probe_bellek count 1 · geri-okuma skor 1.0 · **sahte-seçici: target 'olmayan_ad_p5_x' → probe-count 2 · sahte-ad sunucuda YOK=True** · check top-1 skor 1.0
- B7 HTTP: port 52900 · 5-endpoint kodlar [200, 200, 200, 200, 200] · backlog-samples 0 · query-telemetri 17-anahtar
- B8 dokunulmazlık: GEÇTİ · kristal 36→36 · digest fe7fc7649737ed35→fe7fc7649737ed35 · kanal 0→0 bayt

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p5_ilan_2026-09-27.md (İLAN — koşum ÖNCESİ, REVİZYON-1) | `b7af5178b18a36cbea110eb60778a712dbd25479d4634f54d1537c378d5efbbf` |
| dogrulama_p5_ogrenme_kanali.py (koşulan betik) | `b3d98722a374dc4b3bd7317dc46149974319485c2f28a7e5971128f583746a1a` |
| data/anka_base_v2.pt (DONMUŞ taban) | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` |
| data/rebuild/vocab_anka_r1_33114.json (DONMUŞ sözlük) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| replay-buffer (DONMUŞ) | `59e2f2786d0ec0d1574a77e5b7354e02ae957345a1b4ddce945d0db19527bdf1` |
| data/future_train_vector.jsonl (GERÇEK kanal — 0 bayt) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `fe7fc7649737ed35` |
| data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json | `9b217944ce74becbc632e9e9176af64d036025555eb2ae8f72033bf36f9b0189` |
| data/eval/mimari_dogrulama_p3_hukum_2026-09-27.json | `2449294976d0b787dfb3a2bedb7759a271714f14c45d8cfa15dae7a65e88fd42` |
| data/eval/mimari_dogrulama_p4_hukum_2026-09-27.json | `f559ccd3f0f71f8bbc31b61dac331a829e8c3c3b0b00f8e50257f72b741425bb` |
| mimari_dogrulama_p5_hukum_2026-09-27.json (HÜKÜM JSON — BETİKTEN) | `378a686708d295b1f60400dcc608af6faafa8dda966244e7c0c19eec4d6d23d0` |

*(RAPOR'un kendi sha'sı bu §7 eklentisinden sonra değişir — öz-tutarlı olamaz; nihai rapor-sha notes'a yazılır.)*

---

## §7 — Doğrulayıcı kök-neden analizi (koşum-sonrası ek)

### 7.1 Koşum tarihi — 3 koşum (İLAN `b7af5178…` boyunca SABİT; yumuşatma YOK)

| Koşum | rc | Sebep |
|---|---|---|
| 1 | 1 (crash) | `NameError: np` — `import numpy as np` eksik (py_compile + ALL-CAPS-scan yakalamaz) |
| 2 | 2 (DUR) | 4 kapı düştü — dördü de BETİK-ölçüm-kusuru (kanonik davranış temiz; aşağıda) |
| 3 | **0 (P5_GECTİ)** | 9/9 kapı GEÇTİ |

**Koşum-2'nin 4 düşen kapısı ve onarım-ilkesi (İLAN SABİT, ölçüm İLAN'a hizalanır):**
1. **K1 `gw_inject_reasoning_koleksiyon`:** İLAN'lı değer STRING `"muhakeme_bellek"` (:293 sabit); ölçüm boolean (`'collection_name="muhakeme_bellek"' in gw and …`) basıyordu → **ölçüm değer-çıkarmaya hizalandı** (regex `inject_reasoning_trace` fonksiyon-gövdesine çapalı — ilk-site `kristal_bellek` :123'e düşme tuzağı); boolean `gw_inject_reasoning_paylasim_client` rapor-kırılımına.
2. **K1 `supervisor_retrain_threshold`:** İLAN'lı 5 (:622 `retrain_threshold: int = 5` — colon-type'lı); eski regex `retrain_threshold\s*=\s*5` + "2-satır-bütünlüğü" şartı ıskaladı → **ölçüm değer-çıkarmalı** `retrain_threshold(?::\s*int)?\s*=\s*(\d+)`.
3. **K6/K9 `default_bin`:** kapı `not os.path.exists("data/train_future_finetune.bin")` idi — ama dosya **13 Eyl 09:57'den beri duruyormuş** (12M + meta; P5'ten 2 hafta önceki kanal-testi artefaktı; koşum-1'de keşif: `default_bin_once=True`). İLAN'lı iddia "OLUŞMAZ (**default-yol derleme YAPMADI**)" — ölçülebilir karşılığı **koşumun yazıp-yazmadığıdır**: onarım = mtime+digest ÖNCE==SONRA çıpası (bin + meta); dosyanın ön-varlığı RAPOR-kırılımı. İLAN değişmedi.
4. **K8 `query_future_train_path`:** İLAN'lı B7 kriteri "her 5-endpoint yanıt-verir" — betiğin eklediği `path==probe` kriteri İLAN-dışı katıydı; koşum-2'de ölçüldü: `/api/query` yanıtı 17-anahtar ama `future_train_path` alanı **YOK** (null — kanonik HTTP-şekli; soy-beyanı yalnız `/api/status` verir). Kriter kapıdan çıkarıldı, ölçüm RAPOR-kırılımı.

**Betik-öz-kusuru zinciri (ders genişlemesi):** koşum-1 `np` → `_tanimsiz_sabitler` **ALL-CAPS'ten genel tanımsız-ad taramasına genişletildi** (Store-bağlamı adları + `arg` + ExceptHandler + FunctionDef/ClassDef + modül-dunder'leri) — koşum-3 öncesi statik-ön bu geniş taramayla GEÇTİ. `ast.parse` arbiter; görsel paren-sayımı üç kez yanıltıcıydı (SYNTAX-OK'ta birebir geçen satırlar).

### 7.2 Kanonik bulgular (mutasyonla/koşumla KANITLI — onarım YOK, operatör kararı "yalnız doğrulama")

1. **KANAL-TÜKETİCİ VOCAB-KÖRÜ + SOY-AĞACI-KOPMASI (İLAN B5 beyanı canlı-teyit):** `run_training` cmd-flag kümesinde **`--vocab` YOK, `--load-path` YOK** — train.py default `vocab_base_32852.json` (32.852) ile SIFIRDAN model kurar; `model_path` soy-ağacı yalnız :295'e KAYDEDİLİR ama koşuma SÖYLENMEZ. Koşum-3 kanıtı: B5 log-snippet "Başlangıç Kaybı: **10.4746**" ≈ ln(32.852) — fresh-model canlılık-imzası ([[canlilik-imzasi-moda-bagli]]: devam-koşumunda TAM TERSİ olurdu); rc=0 yalnız kanal-max-id 32831 < 32.852 ŞANSIYLA (İLAN'lı tahmin birebir; tail-jetonlar görünseydi guard fail-closed rc=1 → B5 DÜŞERDİ).
2. **SAHTE-SEÇİCİ :240/:340 mutasyon-kanıtı (İLAN'lı):** `inject_knowledge(target_collection="olmayan_ad_p5_x")` → dönüş echo `olmayan_ad_p5_x` AMA yazım **probe-instance'e** (count 1→2) ve koleksiyon **sunucuda OLUŞMADI** (`collection_exists=False`) — verilen-ad yoksayılır, backing-instance `collection_name` kazanır.
3. **DEFAULT-BİN ÖN-VARLIK keşfi:** `data/train_future_finetune.bin` (12M) + meta — **13 Eyl 09:57** damgalı (P5'ten önceki kanal-testi artefaktı; `data/**` salt-okunur-donmuş desende). P5 koşumları (3 koşum) bu dosyaya BİR BAYT yazmadı — koşum-3 mtime/digest-çıpası kanıtı (1789282659.99 == 1789282659.99). İLAN'ın "default-yol derleme YAPMADI" iddiasının gerçek kanıtı: **derleme probe-yol** (`data/eval/p5_probe_future.bin` 77720 bayt + probe-meta).
4. **`/api/query` yanıtı soy-yolu taşımıyor:** 17-anahtar telemetri ama `future_train_path` alanı YOK (null) — kanonik HTTP-şekli; yalnız `/api/status` `future_train_path=data/eval/p5_probe_future_train.jsonl` beyan eder (K8-ölçüm rapor-kırılımı; P5'te hükme girmedi — İLAN'lı kriter 5-endpoint yanıtı).
5. **`inject_reasoning_trace` İLKELİ koşumla KANITLANDI:** FAZ-A beyanı (:293 sabit `muhakeme_bellek` + paylaşımlı-client `self.memory.client` :291 — koşulursa paylaşımlı-remote'ta KURARDI) + koşum-sonu sunucu-envanteri: `muhakeme_bellek` **YOK** (3 koşum hiçbirinde kurulmadı).
6. **append-davranışı kanonik:** `record_to_future_train` open-modu "a" (:181) — probe-yazım satır 0→2 (tetiklenme 1 + şema-kanıtı 1), ezme YOK; `_archive_processed_records` "w"-modu (:313) probe-yolda kanıtlandı (0→0 bayt sıfırlama; arşiv 4→6 satır, taşıma-teyit birebir).
7. **K6 success-gate kanonik-yol:** `_archive_processed_records` yalnız rc==0 dalında (:288) — koşum-3'te success → arşiv koştu (0→4→6 satır kümülatif; İLAN'lı relational-aritmetik birebir).

### 7.3 Sonraki paketlere besme

- **Kanonik kusur-kuyruğu P5 ekleri (onarım operatör kararı + ayrı İLAN'lı tur):** (i) `run_training` `--vocab`/`--load-path` İLETMELİ (vocab-kör + soy-ağacı-kopması — kanal-kapanışı için İLKELİ-engel); (ii) sahte-seçici :240/:340 (target-ad yoksayımı — backing-instance kazanır; sessiz-yazım-çatallaşması sınıfı); (iii) `/api/query` yanıtına soy-alanı (opaklık; T-0089 dersinin HTTP-kardeşi).
- **`data/train_future_finetune.bin` (12M, 13 Eyl) kanaryası:** P5-öncesi artefakt duruyor — operatör kararıyla silinebilir (koşumlar dokunmadı; digest/mtime kanıtlı).
- **Yeni ders:** tanımsız-ad taraması ALL-CAPS'e dar kalmamalı — küçük-ad modül-alias'ları (`np`) py_compile'dan KAÇAR (koşum-1); Store/arg/ExceptHandler/FunctionDef toplayan genel tarama statik-ön-kontrol setinin yeni standart adımı.

