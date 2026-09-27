# MİMARİ DOĞRULAMA PAKET-2 SONUÇ — RAG Gezgini (Qdrant) (T-0142)

**Damga:** 2026-09-27T14:29:30Z · **Hüküm (BETİKTEN):** **P2_GECTİ (rc=0)**

| Kapı | Durum |
|---|---|
| K1_FAZ_A_BIREBIR | GEÇTİ |
| K2_B0_HEDEF_YOK | GEÇTİ |
| K3_B1_KURULUM | GEÇTİ |
| K4_B1_ARZ_EKSİKSİZ | GEÇTİ |
| K5_B2_POZITIF_20_20 | GEÇTİ |
| K6_B3_DAVRANIS | GEÇTİ |
| K7_DOKUNULMAZLIK | GEÇTİ |

## FAZ-A — sunucu envanteri (İLAN'lı ↔ ölçülen)

| Ölçüm | İLAN'lı | Ölçülen | Durum |
|---|---|---|---|
| koleksiyon_sayisi_once | 10 | 10 | GEÇTİ |
| kanonik_ad_mevcut | 1 | 1 | GEÇTİ |
| hibrit_koleksiyon | 1 | 1 | GEÇTİ |
| tek_dense_768 | 9 | 9 | GEÇTİ |
| crystal_tagsli | 1 | 1 | GEÇTİ |
| kristal_bellek_nokta | 36 | 36 | GEÇTİ |
| turk_articles_nokta | 5 | 5 | GEÇTİ |
| elektor_articles_nokta | 90122 | 90122 | GEÇTİ |
| foreign_sema_anahtar | 6 | 6 | GEÇTİ |

## FAZ-B — kendi-veri arzı + döngü

- B1 kurulum: `{'dense_768_cosine': True, 'sparse': True, 'sparse_adlar': ['sparse'], 'storage_type': 'remote (192.168.1.5:6333)'}` · arz: **36/36** (derleme-boş 0)
- B2 pozitif kontrol: **20/20** · skor min/med/max: 0.3555555733333333/0.44444446666666665/1.0 · eşik-üstü (0,40): 10/20
- B3 OOV negatif kontrol: istisna=False · sonuc_uzunluk=1 · skor=0.0467 · has_root_match=False

## FAZ-C — kod-envanteri (koşumsuz)

- `RAG_MATCH_THRESHOLD` tanım sitesi: [] · import sitesi: 2
- root-filtre blok tekrarı: ['scripts/evaluate_rag_baselines.py', 'scripts/evaluate_rag_pilot.py', 'scripts/import_examples.py', 'src/llm/tokenizer.py', 'src/rag/embedding.py', 'src/rag/vector_memory.py']
- recreate-yıkıcılık satırları (vector_memory.py): [91, 131, 134]
- sessiz-None satırları (rag_pipeline.py): [54]

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p2_ilan2_2026-09-27.md (İLAN — koşum ÖNCESİ) | `f9c132a78cbc80a5c070768ea6d3c07971b88d0b15734d47f7120c147552314b` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `f721db0af5133f2e` |
| dokunulmazlık | GEÇTİ |
