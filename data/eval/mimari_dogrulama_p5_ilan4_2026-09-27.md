# İLAN-4 — MİMARİ DOĞRULAMA PAKET-5 ONARIM DOĞRULAMASI (T-0148 TUR-B; K6 fresh-imza-yok formülasyonu)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-27T16:41:30Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Ölçüm İLAN'a hizalanır.
Hüküm BETİKTEN (`scripts/dogrulama_p5_ogrenme_kanali.py`); elle sayı/hüküm YOK.

## Kapsam ve meşruiyet-çıpa

İLAN-3 koşum-3 DUR (hüküm `897bb7c7…`, rc=2): yalnız K6 düştü — band
[3.30, 3.70] n=2 örneklemden kurulmuştu; `Başlangıç Kaybı` TEK-ÖRNEKLEM +
koşum-gürültülüdür (üç koşum 3.6416 / 3.3751 / 3.1524 — T-0078 dersi:
tek-adım örneklemiyle bant-kapı ölçüt-ölü). Operatör kararı (27 Eyl
2026, 16:41Z): **İLAN-4 onaylandı — K6 AYIRT-EDİCİ ölçüt
`fresh_imza_yok` TEK-bileşenli; band RAPOR-kırılımı**. Diğer kapılar
İLAN-3 hüküm-kanıtıyla SABİT (K1…K5, K7…K9 koşum-3'te GEÇTİ; K8
İLAN-3 formülasyonu 17-anahtar + soy-dolu koşum-3'te GEÇTİ).

Kanonik kod salt-IMPORT (yazım YOK); probe yazımları YALNIZ `data/eval/`
probe-yollarında ve `p5_probe_bellek` probe-koleksiyonunda; koşum sonunda
4D wrapper ile silinir. GERÇEK kanal `data/future_train_vector.jsonl`
koşum boyunca 0 bayt kalır (çıpa `e3b0c442…`).

## Onarım-kontratı (İLAN-3 → İLAN-4 formülasyon-değişimi)

| Alan | İLAN-3 (koşum-3 bunu ÇÜRÜTTÜ) | İLAN-4 (operatör formülasyonu) |
|---|---|---|
| canlılık-imzası (B5-K6) | bant [3.30, 3.70] KAPı-ölçütü — n=2 örneklem; üçüncü koşum 3.1524 band-altı | **`fresh_imza_yok` TEK-bileşenli** (tavan 9,0 KALIR — fresh-imza ≈ 10,4 = ln(32.852); İLAN-1 fresh 10,4746 ↔ İLAN-2/3/4 probe 3,6416/3,3751/3,1524 TERS-yön onarım-kanıtı); **band RAPOR-kırılımı** (`b5["resume_bant"]` kapı-koşulu DEĞİL; RAPOR-kırılım bant [3.10, 3.70] üç-koşum aralığı) |
| diğer kapılar | İLAN-3 SABİT | koşum-3 hüküm-kanıtıyla SABİT: K3 `is_high ≥ 1` betik-kriteri; K8 17-anahtar + soy null→dolu; K7 `b6.update`; sahte-seçici ValueError; inject_reasoning fail-closed RuntimeError |

## FAZ-A İLAN'lı (İLAN-3'ten SABİT)

Değer-çapa ILANLI-dict'i betiğin içindedir; koşum-içi enjeksiyonlu iki
anahtar (`gercek_kanal_once_bayt`, `envanter_digest_once`) koşum-1 deseni
(main satır 827-828). Modül-sabitleri İLAN-4 ile hizalanır:
`HTTP_QUERY_ANAHTAR_SAYISI=17`, `RESUME_BASLANGIC_KAYIP_BANT=(3.10, 3.70)`
(RAPOR-kırılımı — KAPı DEĞİL), `FRESH_IMZA_TAVANI=9.0`.

## FAZ-B İLAN'lı (İLAN-3'ten SABİT — koşum-1/2/3 kanıt-değerleriyle)

- **B0 arz-çıpa:** envanter-digest `fe7fc7649737ed35` SABİT; foreign 9;
  kristal_bellek 36; probe koşum-öncesi YOK; GERÇEK kanal 0 bayt.
- **B1/B3:** koşum-3 kapı-değerleri KORUNUR (kanal-kapısı 0,85;
  record 14-anahtar; append "a"; probe-tanı kırılımı beyanlı).
- **B4 tüketici-derleme:** probe-bin derlemesi kanonik; max-jeton-id <
  32.852; geri-tokenize birebir; koşumlar 1-bayt yazmaz.
- **B5 run_training İLKELİ (5A):** `run_training_status == "success"` +
  Başlangıç Kaybı OKUNUR (tek-örneklem — RAPOR-kırılımı) +
  **fresh-imza-yok** (tavan 9,0); archive taşıma-teyit + success-gate +
  `w`-modu sıfırlama + default-bin mtime/digest-çıpa.
- **B6 gateway-yazım (5B/5D):** sahte-ad → ValueError + yazım-YOK +
  sunucuda sahte-koleksiyon YOK; `inject_reasoning_trace` PROBE_DOC
  çifti → RuntimeError (fail-closed).
- **B7 HTTP (5C):** 5-endpoint 200; `/api/query` **17-anahtar** ve
  `query_future_train_path == PROBE_FUTURE_TRAIN` (null→dolu).
- **B8 dokunulmazlık:** sonra-digest == önce-digest == `fe7fc7649737ed35`;
  GERÇEK kanal 0→0 bayt; probe sunucuda YOK; P2/P3/P4-ONARIM hüküm-digest
  SABİT.

## DOKUNULMAZLAR (İLAN-2/3 birebir)

- `kristal_bellek` 36 nokta — yalnız-okuma; 9 foreign koleksiyon — DOKUNULMAZ.
- GERÇEK kanal `data/future_train_vector.jsonl` — 0 bayt çıpa `e3b0c442…`.
- `data/anka_base_v2.pt` — run_training probe modeli; koşum boyunca digest
  SABİT (yazım YOK).
- `src/**`, `train.py` — koşum İÇİNDE yazım YOK (import-only).
- Kapı-değerleri (0,40 / 0,85) DEĞİŞMEZ (P2/P3 ölçek-çıpası).
- Sunucu-sonu envanter: koşum-öncesi ile birebir (digest çıpası).

## Hüküm kuralı (BETİKTEN)

Tüm kapılar (K1…K9) GEÇTİ → **P5_GECTİ rc=0**; aksi → **DUR rc=2**.
İstisna, kapı-hizasızlığı veya beklenmedik imza → DUR; koşum-sonrası İLAN
yumuşatılmaz. 3-koşum kalıbı AÇIK: İLAN SABİT, ölçüm İLAN'a hizalanır;
betik-öz-kusuru ayrı DUR-sebebi (ILANLI-dict'ten değil).

## Beyan: damga-yöntemi

Bu İLAN'ın damga-sayısı elle yazılmaz; yukarıdaki damga koşum-öncesi
`date -u` betik-çıkışından işlendi (T-0147 6. ihlal-dersi, madde-8).
Dosyanın varlık-kanıtı ek olarak mtime-çıpasıdır: bu dosya koşum
başlangıcından ÖNCE yazılmıştır (koşum `ilan_sha256_kosumda` alanında
kaydedilir ve koşumlar boyunca SABİT kalır).