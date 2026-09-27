# MİMARİ DOĞRULAMA PAKET-4 SONUÇ — Üniversal Hafıza (VectorMemory yaşam-döngüsü; T-0144 İLAN-1 / T-0148 TUR-B İLAN-2 onarım-doğrulama)

**Damga:** 2026-09-27T16:10:52Z · **Hüküm (BETİKTEN):** **P4_GECTİ (rc=0)**

| Kapı | Durum |
|---|---|
| K1_FAZ_A_ENVANTER | GEÇTİ |
| K2_ARZ_CIPA | GEÇTİ |
| K3_B1_BAGLANMA | GEÇTİ |
| K4_B3_DETERMINIZM | GEÇTİ |
| K5_B4_ADD_YOLU | GEÇTİ |
| K6_B5_MUTASYON_KANITI | GEÇTİ |
| K7_DOKUNULMAZLIK | GEÇTİ |
| K8_B6_FALLBACK_FAIL_CLOSED | GEÇTİ |

## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen)

| Ölçüm | İLAN'lı | Ölçülen | Durum |
|---|---|---|---|
| vm_qdrantclient_kurulum | 3 | 3 | GEÇTİ |
| vm_remote_timeout_sn | 2.0 | 2.0 | GEÇTİ |
| vm_cache_yazim_sitesi | 1 | 1 | GEÇTİ |
| vm_cache_evict_sitesi | 1 | 1 | GEÇTİ |
| vm_cache_guard_satiri | 92 | 92 | GEÇTİ |
| vm_cache_yazim_satiri | 93 | 93 | GEÇTİ |
| vm_recreate_delete_satirlar | [103, 143, 155, 164, 165, 171] | [103, 143, 155, 164, 165, 171] | GEÇTİ |
| vm_otomatik_kurulum_birebir | True | True | GEÇTİ |
| embedding_random_seedli | 1 | 1 | GEÇTİ |
| embedding_random_seedli_siz | 0 | 0 | GEÇTİ |
| gateway_localhost_kurulum | 3 | 3 | GEÇTİ |
| gateway_kanonik_koleksiyonlar | ['kristal_bellek', 'simulasyon_bellek'] | ['kristal_bellek', 'simulasyon_bellek'] | GEÇTİ |
| vm_varsayilan_storage_path | None | None | GEÇTİ |
| vm_upsert_param | 1 | 1 | GEÇTİ |
| vm_confirm_destroy_param | 1 | 1 | GEÇTİ |
| vm_delete_wrapper | 1 | 1 | GEÇTİ |
| vm_cache_guard_takipediyor | True | True | GEÇTİ |
- recreate-yıkıcılık satırları (vector_memory.py): [103, 143, 155, 164, 165, 171] · otomatik-kurulum bandı: ['        if not self.client.collection_exists(self.collection_name):', '            self._create_hybrid_collection(vector_size)', '            self._next_point_id = 1']
- VectorMemory varsayılan storage_path: None (T-0148 4B onarım: default KALDIRILDI — host'suz çağrı :memory:'ye düşer, repo-içi yazmaz)
- embedding deterministik (saf çift-çağrı): True

## FAZ-B — VectorMemory yaşam-döngüsü koşumu

- B1 bağlanma: storage_type **remote (192.168.1.5:6333)** · is_in_memory False · kristal_bellek **36** · cache-reuse True
- B2 arz-çıpa: digest **fe7fc7649737ed35** · foreign **9** · kristal_bellek **True** (None) · simulasyon_bellek YOK-teyit True
- B3 determinizm: P2-birebir **20/20** + iç-çift-koşum **20/20** (4 ondalık) · OOV recall: {'skor': 0.0467, 'has_root_match': False}
  - NOT: P2 hüküm-detay anahtarları [:60]-kesmeli — P2 skor-anahtarı **15** (20 satır) vs benzersiz-sorgu-anahtarı **15** — kesme-çakışan çiftler aynı P2-skoruyla karşılaştırılır; birebir SATIR-düzeyi 4 ondalık (RAPOR §7 beyanı)
- B4 add-yolu (probe): checkpoint-count-sekansı **[0, 3, 8, 9, 10]** (İLAN-2'li [0, 3, 8, 9, 10] — upsert İLK +1 / TEKRAR dublör-YOK + default +1) ·   - upsert_count **9** / TEKRAR **9** (dublor-yok: True; T-0148 4C) · ara-adım-sekansı **[0, 1, 2, 3]** (RAPOR kırılımı) · geri-okuma **10** birebir · pozitif-kontrol dense **8/8** + hibrit **8/8**
- B5 recreate-yıkıcılık MUTASYON-KANITI (T-0148 4E): confirm'suz recreate DOLU'da istisna **True** (nokta korundu=True, önce 10) · confirm_destroy=True → **0** nokta ([UYARI] True) · boyut-uyuşmazlık auto-recreate ENGELLENDİ → **1** nokta (önce 1; dense-size 768) · wrapper-temizlik True; kaldi=False
- B6 fallback FAIL-CLOSED (T-0148 4A kapısı): sahte-host-istisna **True** · cache-yazılmadı True · :memory: **in-memory (:memory:)** (is_in_memory True; cache-yazılmadı True) · varsayılan storage_path **None** · bulgu: FAIL_CLOSED_KANITLI — sahte-host RuntimeError (sessiz-local-fallback kapalı, T-0148 4A); cache-yazım YOK; default storage_path KALDIRILDI (host'suz çağrı :memory:'ye düşer, repo-içi yazmaz)
- B7 dokunulmazlık: GEÇTİ · digest fe7fc7649737ed35→fe7fc7649737ed35 · kristal 36→36 · probe sunucuda YOK: True · kanal 0→0 bayt

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p4_ilan2_2026-09-27.md (İLAN — koşum ÖNCESİ) | `5f78837a0badb1b0bf86a7098c7ac157ba508c053186bdc190882fd27fe78dae` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| data/rebuild/vocab_anka_r1_33114.json (DONMUŞ sözlük) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| data/eval/mimari_dogrulama_p2_onarim_hukum_2026-09-27.json (P2 hüküm — B3 birebir kaynağı) | `4b6d183741e742a506c94fd18dc8e9321c7e30d06f001e23ce1baf619d0a9348` |
| data/future_train_vector.jsonl (GERÇEK kanal çıpası — 0 bayt) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `fe7fc7649737ed35` |
| dokunulmazlık | GEÇTİ |
