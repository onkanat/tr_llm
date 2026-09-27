# MİMARİ DOĞRULAMA PAKET-3 SONUÇ — Epistemik Döngü Kapıları (T-0143)

**Damga:** 2026-09-27T12:02:19Z · **Hüküm (BETİKTEN):** **P3_GECTİ (rc=0)**

| Kapı | Durum |
|---|---|
| K1_FAZ_A_ENVANTER | GEÇTİ |
| K2_ARZ_KORUNDU | GEÇTİ |
| K3_B1_MERAK_KAYIT_TAM | GEÇTİ |
| K4_B3_OOV_SAHTE_KOSULLANMA | GEÇTİ |
| K5_B4_KAPI_D | GEÇTİ |
| K6_DOKUNULMAZLIK | GEÇTİ |

## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen; Assign|AnnAssign sayaç)

| Ölçüm | İLAN'lı | Ölçülen | Durum |
|---|---|---|---|
| esik_rag_match_threshold | 0.4 | 0.4 | GEÇTİ |
| tau_default | 2.5 | 2.5 | GEÇTİ |
| similarity_threshold_default | 0.85 | 0.85 | GEÇTİ |
| esik_tanim_sayisi | 1 | 1 | GEÇTİ |
| esik_import_sayisi | 2 | 2 | GEÇTİ |
| merak_router_state_dict_yukleme | 0 | 0 | GEÇTİ |
- eşik tanım sitesi: ['src/rag/rag_pipeline.py'] · import sitesi: ['src/rag/__init__.py', 'src/rag/epistemic_agent.py']
- recreate-yıkıcılık satırları (vector_memory.py): [83, 123, 126] · sessiz-None (rag_pipeline.py): [53]
- record-şema başlangıç satırı (epistemic_agent.py): 349

## FAZ-B — epistemik döngü koşumu

- B1 Kapı-A: kayıt **20/20** · istisna 0 · tetiklenme: {'retrieval_triggered_true': 16, 'needs_retrieval_true': 16, 'kosullanma_teyitli': 7} · entropy_pre min/max: 1.7858/10.0087
- B2 (RAPOR): P2-skor karşılaştırma triggered alt-kümesi: **16/16** birebir (4 ondalık)
- B3 OOV force'lu: istisna=False · skor=0.7 · koşullanma=True · band-içi=True · bulgu: OOV_SAHTE_KOSULLANMA — Kapı-B boş-roots'ta geçti ve Kapı-C (0,40) OOV sorguyu belgeyle koşullandı
- B4 Kapı-D: varsayılan is_high_similarity **0/20** (İLAN'lı 0) · probe-agent tetiklenme 1 (istisna 0) · probe satır 4→6 · şema 14 anahtar (tam=True) · gerçek kanal 0→0 bayt
- B5 dokunulmazlık: GEÇTİ · kanonik nokta 36→36 · digest fe7fc7649737ed35→fe7fc7649737ed35

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p3_ilan_2026-09-27.md (İLAN — koşum ÖNCESİ) | `4e39610cf9200cd6e8e9eefac09180033b05d1a5971ba9bbc0d521a49fee2d70` |
| data/anka_base_v2.pt (DONMUŞ taban) | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` |
| data/rebuild/vocab_anka_r1_33114.json (DONMUŞ sözlük) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| data/future_train_vector.jsonl (GERÇEK kanal çıpası — 0 bayt) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| scripts/dogrulama_p3_epistemik_kapilar.py (koşulan betik — 3. koşum) | `9eb81adc33157280eef72ff207dfda740ac93839f4bf83b2083eb6f54e1796a3` |
| mimari_dogrulama_p3_hukum_2026-09-27.json (hüküm — BETİKTEN) | `2449294976d0b787dfb3a2bedb7759a271714f14c45d8cfa15dae7a65e88fd42` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `fe7fc7649737ed35` |
| dokunulmazlık | GEÇTİ |

---

## §7 — Doğrulayıcı kök-neden analizi (koşum SONRASI ek; ölçüm sayıları hüküm JSON'undan — elle sayı YOK)

### 7.1 Üç koşum — iki öz-kusur AYNI sınıftan (betikte; İLAN hiç yumuşatılmadı — `4e39610c…` sabit)

1. **1. koşum rc=2 — K3 düştü, kök-neden BETİKTE:** `b1 kayıt 20/20, istisna 0`
   ama kapı-anahtar-kümesi `morpheme_output` bekliyordu; `_kosum_ozet`
   üretmiyordu. Kanonik davranış kusursuzdu — **ölçüt kendi özetini denetliyor**
   (T-0142 §7.1 kardeşi; [[olcut-kendi-payini-yiyor]]).
2. **2. koşum rc=2 — K3 yine, AYNI sınıf ikinci hizasızlık:** kanonik alan-adı
   `retrieved_document` yerine türetilmiş `kosullanma_teyit` ismiyle yazmışım.
3. **Onarım + statik ön-kontrol:** koşumdan AYRI adım olarak AST ile
   `_kosum_ozet`'in ürettiği anahtarlar çıkarıldı ve
   `TELEMETRI_ANAHTARLARI ⊆ özet-anahtarları` denetlendi (EKSIK = YOK) →
   **3. koşum rc=0**. **Ders: kapı bir özet-üreticinin alanlarını denetliyorsa
   hizalama koşum-öncesinde statik konmalı; iki koşum iki öz-kusur yedi.**
   (py_compile koşumdan AYRI adım — P1 dersi; hizalama-önden-kontrolü yeni
   kardeş adım.)

### 7.2 Kapı-D ulaşılmazlık — CANLI döngüde teyit (P5 birincil girdisi)

Varsayılan agent (0,85) ile 20 pozitif sorguda `is_high_similarity` **0/20**
(İLAN'lı beklenti-0). Probe-agent (kanonik kurucu `similarity_threshold=0,5`)
yine de yalnız **1/20** kayıt tetikledi; gerçek kanal
`data/future_train_vector.jsonl` **0→0 bayt** (sha `e3b0c442…b855` boş-dosya
digest'i — değişmedi). **0,85 eşiği RRF skor-bandı (0,29–0,75) içinde
ulaşılamıyor** — P5 gateway→öğrenme kanalının ölçülmüş ölüm-sebebi; probe
şeması 14/14-anahtar birebir kanıtlandı (deterministik yazıcı-probe,
`probe_once 4→6` = koşum-başına +2: 1 gerçek-tetiklenme + 1 şema-kanıtı).

### 7.3 OOV sahte-koşullanma — P2 bulgusunun canlı-döngü teyidi (birebir)

`zzqwxx zqxwv zqqzzq` + `force_rag=True`: istisna YOK, **skor 0,7** (İLAN'lı
bant [0,6–0,8] içi), **koşullanma=True** — Kapı-B boş-roots'ta geçiyor
(`has_root_match=True` varsayılanı) ve Kapı-C (0,40) OOV sorguyu belgeyle
koşullandırıyor. P2 §7.3'ün TERSİ-sınıf bulgusu (fazla-koşullanma) iki pakette
kararlı.

### 7.4 B2 determinizm — P2 skorlarıyla 16/16 BİREBİR (4 ondalık)

Triggered alt-kümesinde (16/20) her skor P2 hüküm JSON'undaki aynı-sorgu
skoruyla birebir (örnek: 0,35 / 0,2933 / 0,6) — aynı sorgu + aynı koleksiyon +
deterministik embedding ⇒ koşum-koşum bit-özesi okuma. Tetiklenme 16/20
model-entropisine bağlı (entropy_pre 1,79–10,01 aralığı; 4 sorgu tau-altı) —
**B2'nin hüküm-dışı bırakılma gerekçesi koşumla doğrulandı** (garanti
edilemeyen olay kapı yapılmaz).

### 7.5 Satır-referansı uzlaşması (rapor-düzeyi beyan)

Betik sessiz-None komşuluk-çiftini **except-satırı (53)** olarak yazdı; P2
raporu **return-None-satırını (54)** yazmıştı. Madde aynıdır
(`rag_pipeline.py:53-54`); kapı-tanımları satır-kaymasına bağlanmaz
(yapı-envanteri, hüküm-dışı). recreate-yıkıcılık [83,123,126] P2 ile birebir
tutarlı.

### 7.6 Eğitimsiz-projeksiyon beyanı — teyit

`merak.py` + `router.py` state_dict/load grep **0** (koşumsuz ölçüm) —
router-ağırlıkları ve q_merak projeksiyonları fresh-init (seed=42); epistemik
döngünün tek gerçek-sinyali Anka entropisidir. Onarım önceliği önerisi P5'e:
`CuriosityEngine`/`TriModalRouter` için eğitilmiş ağırlık kaynağı ya da
kural-tabanlı deterministik merak-skoru.
