# MİMARİ DOĞRULAMA PAKET-3 SONUÇ — Epistemik Döngü Kapıları (T-0143)

**Damga:** 2026-09-27T14:30:08Z · **Hüküm (BETİKTEN):** **P3_GECTİ (rc=0)**

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
- recreate-yıkıcılık satırları (vector_memory.py): [91, 131, 134] · sessiz-None (rag_pipeline.py): [53]
- record-şema başlangıç satırı (epistemic_agent.py): 349

## FAZ-B — epistemik döngü koşumu

- B1 Kapı-A: kayıt **20/20** · istisna 0 · tetiklenme: {'retrieval_triggered_true': 16, 'needs_retrieval_true': 16, 'kosullanma_teyitli': 8} · entropy_pre min/max: 2.2804/9.5308
- B2 (RAPOR): P2-skor karşılaştırma triggered alt-kümesi: **0/16** birebir (4 ondalık)
- B3 OOV force'lu: istisna=False · skor=0.0 · koşullanma=False · band-içi=False · bulgu: OOV_FAIL_CLOSED_ONARIM_KANITI — skor tavan-altı ve koşullanma kapandı
- B4 Kapı-D: varsayılan is_high_similarity **3/20** (İLAN'lı 0) · probe-agent tetiklenme 0 (istisna 0) · probe satır 0→1 · şema 14 anahtar (tam=True) · gerçek kanal 0→0 bayt
- B5 dokunulmazlık: GEÇTİ · kanonik nokta 36→36 · digest fe7fc7649737ed35→fe7fc7649737ed35

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p3_ilan2_2026-09-27.md (İLAN — koşum ÖNCESİ) | `7d9e307eb73911dc2466b0c7ddba2e76731f633ff6a7eaa57c6b0a7ca828714d` |
| data/anka_base_v2.pt (DONMUŞ taban) | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` |
| data/rebuild/vocab_anka_r1_33114.json (DONMUŞ sözlük) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| data/future_train_vector.jsonl (GERÇEK kanal çıpası — 0 bayt) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `fe7fc7649737ed35` |
| dokunulmazlık | GEÇTİ |
