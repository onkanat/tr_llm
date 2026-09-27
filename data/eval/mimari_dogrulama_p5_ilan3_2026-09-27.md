# İLAN-3 — MİMARİ DOĞRULAMA PAKET-5 ONARIM DOĞRULAMASI (T-0148 TUR-B; İLAN-2 formülasyon-onarımı)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-27T16:23:26Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Ölçüm İLAN'a hizalanır.
Hüküm BETİKTEN (`scripts/dogrulama_p5_ogrenme_kanali.py`); elle sayı/hüküm YOK.

## Kapsam ve meşruiyet-çıpa

İLAN-2 iki koşum DUR (hüküm `f794cbd8…`, rc=2): K6+K8 **İLAN-2
formülasyon-kusuru** olarak KARARLI düştü — İLAN-1 koşum-ölçümlerinin
onarım-sonrası beklenti olarak yanlış-devralınması (bellek:
`ilan-formulasyon-kusuru-eski-deger-devralma`). Operatör kararı
(27 Eyl 2026, 16:23Z): **İLAN-3 onaylandı → koşum-3**. İLAN-3 yalnız
K6 ve K8 formülasyonlarını onarım-EKSENİNDEN yeniden formüle eder;
diğer kapılar İLAN-2 hüküm-kanıtıyla SABİT.

Kanonik kod salt-IMPORT (yazım YOK); probe yazımları YALNIZ `data/eval/`
probe-yollarında ve `p5_probe_bellek` probe-koleksiyonunda; koşum sonunda
4D wrapper ile silinir. GERÇEK kanal `data/future_train_vector.jsonl`
koşum boyunca 0 bayt kalır (çıpa `e3b0c442…`).

## Onarım-kontratı (İLAN-2 → İLAN-3 formülasyon-değişimi)

| Alan | İLAN-2 (koşum bunu ÇÜRÜTTÜ) | İLAN-3 (onarım-ekseninden formülasyon) |
|---|---|---|
| canlılık-imzası bandı (B5-K6) | [3.70, 4.12] — T-0077 **A1-r KÜLLİYAT-rejimi** bandı; probe-rejim koşumlarında band-dışı | **[3.30, 3.70] probe-rejim-meydanı** (İLAN-2 çift-koşum ölçümü: 3.6416 / 3.3751) + fresh-imza-tavanı **9.0** KALIR — fresh-imza (≥ 9,0) GÖRÜNMEZ = onarım-kanıtı 5A (İLAN-1 fresh 10,4746 ↔ İLAN-2 3,6416/3,3751 TERS-yön kanıt) |
| `/api/query` yanıtı (B7-K8) | 18 anahtar — `ask` telemetrisi **17**-anahtar (İLAN-1'de `future_train_path` anahtar olarak ZATEN VARDI, null'du); sayım ölçülmeden 17+1 tahmin edilmişti | **17 anahtar** (sayım DEĞİŞMEZ — onarım 5C yeni anahtar EKLEMEZ) + soy-alanı `future_train_path` **null→DOLU** (`data/eval/p5_probe_future_train.jsonl` TAŞIR): ayırt-edici formülasyon |
| K3 kanal-yazarı (betik-kriteri) | İLAN-1'den devralınan `is_high==0` ulaşılamazlık-ölçütü; koşum-1'de betik-düzeyinde onarıldı | **korunur**: `is_high ≥ 1` (Kapı-D canlandı — normalizasyon onarım-kanıtı; default-agent 3/20, probe-agent 9/20). `probe_recorded` kapı-koşulu DEĞİL (İLAN-doc'ta sayım İLAN'lı-değil) — Tur-A etki-yüzeyi tanı-kırılımında beyanlı |
| K7 gateway-yazım | koşum-1 betik-öz-kusuru (`b6 = {...}` ezmesi) → `b6.update` | **korunur** (koşum-2 GEÇTİ) |

## FAZ-A İLAN'lı (İLAN-2'den SABİT)

Değer-çapa ILANLI-dict'i betiğin içindedir; koşum-içi enjeksiyonlu iki
anahtar (`gercek_kanal_once_bayt`, `envanter_digest_once`) koşum-1 deseni
(main satır 827-828). Modül-sabitleri İLAN-3 ile hizalanır:
`HTTP_QUERY_ANAHTAR_SAYISI=17`, `RESUME_BASLANGIC_KAYIP_BANT=(3.30, 3.70)`,
`FRESH_IMZA_TAVANI=9.0` (değişmez).

## FAZ-B İLAN'lı (İLAN-2'den SABİT — koşum-1/2 kanıt-değerleriyle)

- **B0 arz-çıpa:** envanter-digest `fe7fc7649737ed35` SABİT; foreign 9;
  kristal_bellek 36; probe koşum-öncesi YOK; GERÇEK kanal 0 bayt.
- **B1/B3:** koşum-2 kapı-değerleri KORUNUR (kanal-kapısı 0,85;
  record 14-anahtar; append "a"; probe-tanı kırılımı beyanlı).
- **B4 tüketici-derleme:** probe-bin derlemesi kanonik; max-jeton-id <
  32.852; geri-tokenize birebir; koşumlar 1-bayt yazmaz.
- **B5 run_training İLKELİ (5A):** Başlangıç Kaybı probe-rejim bandında
  **[3.30, 3.70]**; fresh-imza (≥ 9,0) GÖRÜNMEZ; archive taşıma-teyit +
  success-gate + `w`-modu sıfırlama.
- **B6 gateway-yazım (5B/5D):** sahte-ad → ValueError + yazım-YOK +
  sunucuda sahte-koleksiyon YOK; `inject_reasoning_trace` PROBE_DOC
  çifti → RuntimeError (fail-closed).
- **B7 HTTP (5C):** 5-endpoint 200; `/api/query` **17-anahtar** ve
  `query_future_train_path == PROBE_FUTURE_TRAIN` (null→dolu).
- **B8 dokunulmazlık:** sonra-digest == önce-digest == `fe7fc7649737ed35`;
  GERÇEK kanal 0→0 bayt; probe sunucuda YOK; P2/P3/P4-ONARIM hüküm-digest
  SABİT.

## DOKUNULMAZLAR (İLAN-2 birebir)

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