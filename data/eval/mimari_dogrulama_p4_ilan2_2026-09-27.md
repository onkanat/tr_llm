# İLAN-2 — MİMARİ DOĞRULAMA PAKET-4 ONARIM DOĞRULAMASI (T-0148 TUR-B)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-27T15:56:11Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Ölçüm İLAN'a hizalanır.
Hüküm BETİKTEN (`scripts/dogrulama_p4_universal_hafiza.py`); elle sayı/hüküm YOK.

## Kapsam

P4 onarımları (T-0148 Tur-B, operatör-onaylı plan) SONRASI beklentiler.
Kanonik kod salt-IMPORT (yazım YOK); mutasyon-kanitları YALNIZ
`p4_probe_bellek` probe-koleksiyonunda; koşum sonunda 4D wrapper ile silinir.

## Onarım-kontratı (İLAN-1 → İLAN-2 beklenti-değişimi)

| Alan | İLAN-1 (onarım-öncesi) | İLAN-2 (onarım-sonrası beklenti) |
|---|---|---|
| Cache-guard | `if not self.is_in_memory:` :72 | `if not self.is_in_memory and self.storage_type.startswith("remote"):` :92 (4A: cache'e yalnız remote yazılır) |
| recreate/delete satırları | [83, 123, 126] | [103, 143, 155, 164, 171] |
| Otomatik-kurulum bandı | 75–77 | 95–97 |
| gateway host="localhost" | 2 site | 3 site (:129/:130 create_default + :322 inject_reasoning — 5D: 3. site AÇIK-BEYANlı storage_path=None) |
| Kurucu default storage_path | `"data/qdrant_db"` | `None` (4B: default KALDIRILDI; host'suz çağrı :memory:'ye düşer) |
| upsert param | YOK | `upsert: bool = False` tek-sitesi (4C) |
| confirm_destroy param | YOK | `confirm_destroy: bool = False` tek-sitesi (4E) |
| delete_collection wrapper | YOK (client-düzeyi :126) | `VectorMemory.delete_collection` VAR (4D) |

## FAZ-B İLAN'lı (koşum-öncesi)

- **B2 arz-çıpa:** envanter-digest `fe7fc7649737ed35` SABİT; foreign 9;
  kristal_bellek 36; simulasyon_bellek YOK; probe koşum-öncesi YOK;
  GERÇEK kanal 0 bayt (çıpa `e3b0c442…`).
- **B3 determinizm:** P2-ONARIM hüküm skorlarıyla birebir 20/20 (4 ondalık;
  normalize-ölçek — P2 onarım koşumuyla aynı zincir); iç-çift-koşum 20/20.
- **B4 add-yolu (probe):** checkpoint-count-sekansı **[0, 3, 8, 9, 10]** —
  3× add_document → 5× add_documents_batch (8) → **upsert=True İLK-yazım
  (9 — hash-türetilmiş deterministik id yaratır)** → upsert TEKRAR
  (9 — **DUBLÖR YOK; 4C mutasyon-kanıtı**) → default upsert=False dublör
  (10 — monoton-sayaç bit-uyumlu). `upsert_count == 9`. Geri-okuma 10
  birebir; pozitif-kontrol dense 8/8 + hibrit 8/8.
- **B5 recreate MUTASYON-kanıtı (4E):** confirm'suz recreate DOLU (10)
  koleksiyonda → **RuntimeError** ve nokta-sayısı KORUNUR (10);
  `confirm_destroy=True` → GÖRÜNÜR yıkım ([UYARI] print) → 0 nokta;
  boyut-uyuşmazlık auto-recreate ENGELLENDİ → 1 nokta KORUNUR, dense-size
  **768** KALIR (İLAN-1'de 384'e recreate ediliyordu); temizlik 4D
  wrapper'ıyla (delete_wrapper_sonuc=True, probe silinir).
- **B6 fallback (K8 kapısı — İLAN-1'de RAPOR-düzeyiydi):** sahte-host
  192.0.2.1 (storage_path VERİLMİŞ olsa bile) → **RuntimeError** (4A
  fail-closed; İLAN-1'de aynı girdi sessizce "local (…)"e düşüyordu) +
  cache-anahtar kümesi DEĞİŞMEZ (mutasyon-kanıtı); host'suz +
  storage_path'siz çağrı → `in-memory (:memory:)`, is_in_memory=True,
  cache-yazım YOK; `vm_varsayilan_storage_path == "None"`.
- **B7 dokunulmazlık:** sonra-digest == önce-digest == `fe7fc7649737ed35`;
  foreign önce==sonra; probe sunucuda YOK; kanal 0→0 bayt; P2-ONARIM
  hüküm-digest koşum boyunca SABİT.

## DOKUNULMAZLAR

- `kristal_bellek` 36 nokta — yalnız-okuma.
- 9 foreign koleksiyon — DOKUNULMAZ.
- GERÇEK kanal `data/future_train_vector.jsonl` — 0 bayt çıpa `e3b0c442…`.
- `src/rag/**`, `src/gateway/**`, `src/llm/**`, `src/compiler/**`,
  `train.py` — koşum İÇİNDE yazım YOK (import-only).
- Kapı-değerleri (0,40 / 0,85) ve `RAG_MATCH_THRESHOLD` tanım-sayısı
  DEĞİŞMEZ (P3 ölçek-çıpası).
- Sunucu-sonu envanter: koşum-öncesi ile birebir (digest çıpası).

## Hüküm kuralı (BETİKTEN)

Tüm kapılar (K1…K8) GEÇTİ → **P4_GECTİ rc=0**; aksi → **DUR rc=2**.
İstisna, kapı-hizasızlığı veya beklenmedik imza → DUR; koşum-sonrası İLAN
yumuşatılmaz.

## Beyan: damga-yöntemi

Bu İLAN'ın damga-sayısı elle yazılmaz; yukarıdaki damga koşum-öncesi
`date -u` betik-çıkışından işlendi (T-0147 6. ihlal-dersi, madde-8).
Dosyanın varlık-kanıtı ek olarak mtime-çıpasıdır: bu dosya koşum
başlangıcından ÖNCE yazılmıştır (koşum-1 sha256 `ilan_sha256_kosumda`
alanında kaydedilir ve üç koşum boyunca SABİT kalır).