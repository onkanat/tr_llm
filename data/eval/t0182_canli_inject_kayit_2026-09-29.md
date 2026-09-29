# T-0182 — Canlı /api/inject yazım + canlı kontrol (operatör-onaylı)

**Damga:** 2026-09-29T03:53:14Z / 03:53:23Z (BETİKTEN, `date -u`) · **Hedef:** `anka_bellek`
**Operatör kararı (AskUserQuestion, 29 Eyl):** "anka_bellek (canlı)" — 36-point
canlı-tanık **bilinçli-ezilir** (T-0151/T-0153 çıpası 36 idi; artık 37'den devam eder).
**Canlı gateway:** `b52gx55g4` (127.0.0.1:8080, MPS, base_v2 `d0f415f3…`, router_sd `d369b3cd…`)

## Yazım

- Metin: "Kırlangıç kuyruğu, masa ve çekmece kasalarında çekme mukavemeti sağlayan
  geleneksel bir köşe birleştirmedir."
- `inject` yanıtı: `status success`, `total_documents: 37`; derlenen
  `crystal_tags` BETİKTEN: `<BOS> kırlangıç kuyruk POSS_3SG , masa ve çekmece kasa
  POSS_3PL CASE_LOC_N çekme mukavemet POSS_3SG sağla PART_An geleneksel bir
  köşe birleştir INF_mA COPULA_AORIST . <EOS>`

## Canlı kontrol (BETİKTEN, /api/check)

- **Önce:** `anka_bellek_docs: 36`; aynı-sorgu (`kırlangıç kuyruğu birleştirme`)
  en-iyi **0,0333** (alakasız — diyatomit belge; alt-OOV bantı) → karşılıksız temel-çıpa.
- **Sonra:** `anka_bellek_docs: 37`; aynı-sorgu en-içi **skor 1,3333**
  (ham iki-kaynak rank-0; `/RRF_SCORE_MAX=0,75` normalizasyonu-sonrası uç değeri),
  metin + crystal_tags **birebir** geri-geldi.
- Nokta-kimlik: `injected_by: "agent_gateway"`, `timestamp:
  "2026-09-29T03:53:19.896148+00:00"`, token_ids BETİKTEN
  `[2, 6039, 5267, 13, 32138, 1339, 46, 341, 1529, 7, 24, 5246, 5192, 13, 1001,
  69, 6790, 40, 945, 7303, 208, 115, 32137, 3]` (24 jeton; 32137='.', 3=<EOS>).

## Uçtan-uca karar-kontrolü (/api/query)

`"Kırlangıç kuyruğu nedir ve neyi sağlar?"` → `rag_score 1,3333` ·
`conditioned: True` (T-0179 eşiği 0,33 + T-0181 alanı canlı) ·
`source_collection: anka_bellek` — yazılan bilgi RAG-koşullamasına girer;
`response_text` carpenter-reçete-kalıptır (LM yetenek-sınırları beklenen davranış).

## Geri-dönüş yolu (beyan)

Koleksiyon-silme sarmalayıcısı yok (T-0144 bulgusu sürdü); bu noktanın
geri alınması **nokta-silme API'si** ile ayrı operatör-emri içindedir
(kimlik: yukarıdaki timestamp + injected_by). canlı-tanık sayacı bundan
böyle **37**'dir; üçüncü-taraflı doğrulamalar buna göre formüle edilir.