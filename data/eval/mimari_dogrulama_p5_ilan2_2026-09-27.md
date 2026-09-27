# İLAN-2 — MİMARİ DOĞRULAMA PAKET-5 ONARIM DOĞRULAMASI (T-0148 TUR-B)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-27T16:07:35Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Ölçüm İLAN'a hizalanır.
Hüküm BETİKTEN (`scripts/dogrulama_p5_ogrenme_kanali.py`); elle sayı/hüküm YOK.

## Kapsam

P5 onarımları (T-0148 Tur-B, operatör-onaylı plan) SONRASI beklentiler.
Kanonik kod salt-IMPORT (yazım YOK); probe yazımları YALNIZ `data/eval/`
probe-yollarında ve `p5_probe_bellek` probe-koleksiyonunda; koşum sonunda
4D wrapper ile silinir. GERÇEK kanal `data/future_train_vector.jsonl`
koşum boyunca 0 bayt kalır (çıpa `e3b0c442…`).

## Onarım-kontratı (İLAN-1 → İLAN-2 beklenti-değişimi)

| Alan | İLAN-1 (onarım-öncesi) | İLAN-2 (onarım-sonrası beklenti) |
|---|---|---|
| `retrain_cmd_flag_kumesi` | 6 flag (`--vocab`/`--load-path` YOK — koşum-1 kanıtı: train.py default vocab ile SIFIRDAN kurar) | **8 flag** — `--load-path` + `--vocab` İLETİLİR (5A soy-ağacı-iletimi) |
| `retrain_cmd_vocab_yok` / `retrain_cmd_load_path_yok` | True (koşum-1 kanıtı) | **False** (iletim VAR) |
| run_training ön-varlık | checkpoint-yoksa sessiz sıfırdan (train.py:275-276 tuzağı) | model_path MEVCUT-DEĞİLSE **betik-öncesi RuntimeError** (5A fail-closed) |
| canlılık-imzası | koşum-1'de fresh-imza 10,4746 GÖRÜLDÜ (onarım-öncesi) | **devam-rejimi** (T-0077): Başlangıç Kaybı **[3.70, 4.12]** bandında; fresh-imza (≥ 9,0) GÖRÜNMEZ = onarım-kanıtı 5A |
| sahte-seçici (:240/:340) | verilen target-ad YOKSAYILIR; backing-instance'e yazım (koşum-1 mutasyon-kanıtı: 'olmayan_ad_p5_x' → echo + probe-yazım) | `_resolve_target_memory` — bilinmeyen-ad → **ValueError**; yazım YOK (count önce==sonra); sahte-ad sunucuda YOK (5B fail-closed) |
| `inject_reasoning_trace` | İLKELİ (client-yoksa başka-koleksiyonu-kurar; paylaşımlı-client) | **AÇIK-BEYAN**: `collection_name: str = "muhakeme_bellek"` param; paylaşımlı-client YOK; PROBE_DOC çifti → **RuntimeError** (5D fail-closed) |
| `/api/query` yanıtı | 17 anahtar; `future_train_path` null (koşum-1 kanıtı: soy-yolu TAŞIMIYOR) | **18 anahtar** + soy-alanı `data/eval/p5_probe_future_train.jsonl` TAŞIR (5C; koşulsuz — T-0089 dersi HTTP-kardesi) |

## FAZ-A İLAN'lı (koşum-öncesi)

Değer-çapa ILANLI-dict'i betiğin içindedir (K1 kapısı BETİKTEN kıyaslar);
koşum-içi hesaplanan iki anahtar (`gercek_kanal_once_bayt`,
`envanter_digest_once`) koşum-1 deseniyle main'de faz_a'ya enjekte edilir
(satır 827-828 — koşum-1 kalıbı birebir).

Onarım-sürüklemeli FAZ-A ölçümleri: `retrain_cmd_flag_kumesi` 8-flag
(cmd.extend-listesi UNION'la taranır — koşum-1 SAPMA-dersi onarıldı),
`gw_resolve_target_var` (fail-closed-ad-çözümü VAR),
`gw_inject_reasoning_koleksiyon` "muhakeme_bellek",
`gw_inject_reasoning_paylasim_client` False, `gw_localhost_kurulum` 3.
`/api/query` 18-anahtar + canlılık-imzası FAZ-A ölçümü DEĞİLDİR
(sırasıyla B7-K8 ve B5-K6 kapılarında koşum-içi ölçülür; İLAN'lı değerler
modül-sabitidir: `HTTP_QUERY_ANAHTAR_SAYISI=18`,
`RESUME_BASLANGIC_KAYIP_BANT=(3.70, 4.12)`, `FRESH_IMZA_TAVANI=9.0`).

## FAZ-B İLAN'lı (koşum-öncesi)

- **B0 arz-çıpa:** envanter-digest `fe7fc7649737ed35` SABİT
  (P2==P3==P4-PONARIM==P5 zinciri); foreign 9; kristal_bellek 36;
  probe koşum-öncesi YOK; GERÇEK kanal 0 bayt (`e3b0c442…`).
- **B1/B3 kanal-yazarı + tüketici-okuma:** koşum-1 kapı-değerleri KORUNUR
  (kanal-kapısı 0,85; record 14-anahtar; append "a"; probe-yazım yalnız
  probe-yolunda).
- **B4 tüketici-derleme:** probe-bin derlemesi kanonik; max-jeton-id <
  32.852; geri-tokenize birebir. (Kanarya `data/train_future_finetune.bin`
  T-0146 ile SİLİNDİ — default-bin on-varlık mtime+digest-çıpası
  koşum-1 kanıtıyla RAPOR-düzeyinde kayıt altında; koşumlar 1-bayt yazmaz.)
- **B5 run_training İLKELİ (5A kanıtı):** probe-pipeline `run_training`
  (10 adım) resume-bandında Başlangıç Kaybı **[3.70, 4.12]**; fresh-imza
  (≥ 9,0) GÖRÜNMEZ; archive taşıma-teyit + success-gate + `w`-modu
  sıfırlama koşum-1 İLAN'lı değerlerde.
- **B6 gateway-yazım (5B/5D mutasyon-kanıtları):** sahte-ad
  `inject_knowledge` → **ValueError** + yazım-YOK (count önce==sonra) +
  sunucuda sahte-koleksiyon YOK; `check_memory` sahte-ad → mutasyon-YOK;
  `inject_reasoning_trace(PROBE_DOC_1, PROBE_DOC_1)` → **RuntimeError**
  (fail-closed; İLAN-1'de İLKELİ-koşum sunucuda `muhakeme_bellek`
  YOKtu ve sessizdi).
- **B7 HTTP (5C kanıtı):** 5-endpoint 200; `/api/query` yanıtı
  **18-anahtar** ve `query_future_train_path == PROBE_FUTURE_TRAIN`.
- **B8 dokunulmazlık:** sonra-digest == önce-digest == `fe7fc7649737ed35`;
  GERÇEK kanal 0→0 bayt; probe sunucuda YOK; P2/P3/P4-ONARIM hüküm-digest
  koşum boyunca SABİT.

## DOKUNULMAZLAR

- `kristal_bellek` 36 nokta — yalnız-okuma; 9 foreign koleksiyon —
  DOKUNULMAZ.
- GERÇEK kanal `data/future_train_vector.jsonl` — 0 bayt çıpa `e3b0c442…`
  (koşum yazımı yalnız `data/eval/p5_probe_future_train.jsonl` probe-yolunda).
- `data/anka_base_v2.pt` — run_training probe modeli `--base-ckpt`'ten;
  koşum boyunca digest SABİT (yazım YOK).
- `src/**`, `train.py` — koşum İÇİNDE yazım YOK (import-only).
- Kapı-değerleri (0,40 / 0,85) DEĞİŞMEZ (P2/P3 ölçek-çıpası).
- Sunucu-sonu envanter: koşum-öncesi ile birebir (digest çıpası).

## Hüküm kuralı (BETİKTEN)

Tüm kapılar (K1…K9) GEÇTİ → **P5_GECTİ rc=0**; aksi → **DUR rc=2**.
İstisna, kapı-hizasızlığı veya beklenmedik imza → DUR; koşum-sonrası İLAN
yumuşatılmaz. 3-koşum kalıbı AÇIK: İLAN SABİT, ölçüm İLAN'a hizalanır;
betik-öz-kusuru ayrı DUR-sebebi (koşum-2'de ILANLI-dict'ten değil).

## Beyan: damga-yöntemi

Bu İLAN'ın damga-sayısı elle yazılmaz; yukarıdaki damga koşum-öncesi
`date -u` betik-çıkışından işlendi (T-0147 6. ihlal-dersi, madde-8).
Dosyanın varlık-kanıtı ek olarak mtime-çıpasıdır: bu dosya koşum
başlangıcından ÖNCE yazılmıştır (koşum-1 `ilan_sha256_kosumda` alanında
kaydedilir ve koşumlar boyunca SABİT kalır).