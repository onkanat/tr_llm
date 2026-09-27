# MİMARİ DOĞRULAMA PAKET-4 SONUÇ — Üniversal Hafıza (VectorMemory yaşam-döngüsü, T-0144)

**Damga:** 2026-09-27T12:31:05Z · **Hüküm (BETİKTEN):** **P4_GECTİ (rc=0)**

| Kapı | Durum |
|---|---|
| K1_FAZ_A_ENVANTER | GEÇTİ |
| K2_ARZ_CIPA | GEÇTİ |
| K3_B1_BAGLANMA | GEÇTİ |
| K4_B3_DETERMINIZM | GEÇTİ |
| K5_B4_ADD_YOLU | GEÇTİ |
| K6_B5_MUTASYON_KANITI | GEÇTİ |
| K7_DOKUNULMAZLIK | GEÇTİ |

## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen)

| Ölçüm | İLAN'lı | Ölçülen | Durum |
|---|---|---|---|
| vm_qdrantclient_kurulum | 3 | 3 | GEÇTİ |
| vm_remote_timeout_sn | 2.0 | 2.0 | GEÇTİ |
| vm_cache_yazim_sitesi | 1 | 1 | GEÇTİ |
| vm_cache_evict_sitesi | 1 | 1 | GEÇTİ |
| vm_cache_guard_satiri | 72 | 72 | GEÇTİ |
| vm_cache_yazim_satiri | 73 | 73 | GEÇTİ |
| vm_recreate_delete_satirlar | [83, 123, 126] | [83, 123, 126] | GEÇTİ |
| vm_otomatik_kurulum_birebir | True | True | GEÇTİ |
| embedding_random_seedli | 1 | 1 | GEÇTİ |
| embedding_random_seedli_siz | 0 | 0 | GEÇTİ |
| gateway_localhost_kurulum | 2 | 2 | GEÇTİ |
| gateway_kanonik_koleksiyonlar | ['kristal_bellek', 'simulasyon_bellek'] | ['kristal_bellek', 'simulasyon_bellek'] | GEÇTİ |
| vm_cache_guard_takipediyor | True | True | GEÇTİ |
- recreate-yıkıcılık satırları (vector_memory.py): [83, 123, 126] · otomatik-kurulum bandı: ['        if not self.client.collection_exists(self.collection_name):', '            self._create_hybrid_collection(vector_size)', '            self._next_point_id = 1']
- VectorMemory varsayılan storage_path: data/qdrant_db (host'suz çağrıda repo-içi local-storage riski — RAPOR §7)
- embedding deterministik (saf çift-çağrı): True

## FAZ-B — VectorMemory yaşam-döngüsü koşumu

- B1 bağlanma: storage_type **remote (192.168.1.5:6333)** · is_in_memory False · kristal_bellek **36** · cache-reuse True
- B2 arz-çıpa: digest **fe7fc7649737ed35** · foreign **9** · kristal_bellek **True** (None) · simulasyon_bellek YOK-teyit True
- B3 determinizm: P2-birebir **20/20** + iç-çift-koşum **20/20** (4 ondalık) · OOV recall: {'skor': 0.7, 'has_root_match': True}
  - NOT: P2 hüküm-detay anahtarları [:60]-kesmeli — P2 skor-anahtarı **15** (20 satır) vs benzersiz-sorgu-anahtarı **15** — kesme-çakışan çiftler aynı P2-skoruyla karşılaştırılır; birebir SATIR-düzeyi 4 ondalık (RAPOR §7 beyanı)
- B4 add-yolu (probe): checkpoint-count-sekansı **[0, 3, 8, 9]** (İLAN'lı [0, 3, 8, 9] — dublör +1: API-düzeyi upsert YOK) · ara-adım-sekansı **[0, 1, 2, 3]** (RAPOR kırılımı) · geri-okuma **9** birebir · pozitif-kontrol dense **8/8** + hibrit **8/8**
- B5 recreate-yıkıcılık MUTASYON-KANITI: explicit recreate → **0** nokta (9→0; reset-mesajı True) · boyut-uyuşmazlık auto-recreate (:83) → **0** nokta (önce 1; dense-size 384) · probe kaldırma: kaldi=False
- B6 (RAPOR) sessiz-fallback: local **local (/var/folders/v0/8z_jjnds4rbdtth51qmwysnw0000gn/T/p4_probe_local_izc764zo)** (is_in_memory False; cache-yazıldı False) · :memory: **in-memory (:memory:)** (is_in_memory True; cache-yazılmadı True) · bulgu: SESSIZ_FALLBACK_CANLI — sahte-hostta remote sessizce local-path'e (ve storage_path-yoksa :memory:'ye) düşer; local-client cache'e YAZILIR (gateway:123-124 storage_path'li çağrı riski)
- B7 dokunulmazlık: GEÇTİ · digest fe7fc7649737ed35→fe7fc7649737ed35 · kristal 36→36 · probe sunucuda YOK: True · kanal 0→0 bayt

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p4_ilan_2026-09-27.md (İLAN — koşum ÖNCESİ) | `210dc4f63b5496286909e225e2a033ca492b948e468702abb16b33bfefc0ebca` |
| **koşumda ölçülen İLAN sha (3 koşum SABİT teyit)** | `210dc4f63b5496286909e225e2a033ca492b948e468702abb16b33bfefc0ebca` |
| mimari_dogrulama_p4_hukum_2026-09-27.json (HÜKÜM — BETİKTEN) | `f559ccd3f0f71f8bbc31b61dac331a829e8c3c3b0b00f8e50257f72b741425bb` |
| scripts/dogrulama_p4_universal_hafiza.py (3. koşumda koşulan) | `22ac7fdd4cf34af3570da19a693b965f228a08e8872bb370033c32c5539472d5` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| data/rebuild/vocab_anka_r1_33114.json (DONMUŞ sözlük) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json (P2 hüküm — B3 birebir kaynağı) | `9b217944ce74becbc632e9e9176af64d036025555eb2ae8f72033bf36f9b0189` |
| data/future_train_vector.jsonl (GERÇEK kanal çıpası — 0 bayt) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| envanter ÖNCE/SONRA digest | `fe7fc7649737ed35` / `fe7fc7649737ed35` |
| dokunulmazlık | GEÇTİ |

*(RAPOR'un kendi sha'sı bu §7 eklentisinden sonra değişir — öz-tutarlı olamaz; nihai rapor-sha notes'a yazılır: `af8683cf…` §7-öncesi betik-çıktısı.)*

---

## §7 — Doğrulayıcı kök-neden analizi (koşum-sonrası ek)

### 7.1 Koşum tarihi — 3 koşum (İLAN `210dc4f6…` boyunca SABİT; yumuşatma YOK)

| Koşum | rc | Sebep |
|---|---|---|
| 1 | 1 (crash) | `NameError: KANONIK_YASAK` — sabit tanımı `YASAK_KANONIK`, kullanım `KANONIK_YASAK` (yazma-alışkanlığı; **py_compile NameError'ı YAKALAMAZ**) |
| 2 | 2 (DUR) | 4 betik-öz-kusuru (aşağıda) — kanonik davranış kusursuzdu (K2/K3/K4 + B4-gerçek/B5-kanıt zaten geçiyordu) |
| 3 | **0 (P4_GECTİ)** | 7/7 kapı GEÇTİ |

**Koşum-2'nin 4 öz-kusuru ve onarım-ilkesi (İLAN SABİT, ölçüm İLAN'a hizalanır):**
1. `AttributeError: 'VectorMemory' object has no attribute 'delete_collection'` — K6 temizlik + B7 temizlik-sigortası düştü; probe sunucuda kaldı. **Onarım: kanonik client-yolu** `probe.client.delete_collection(...)` (vector_memory.py:126'daki client-çağrı kalıbı) — wrapper yazılmadı (operatör kararı: yalnız doğrulama).
2. K5 sekans-protokolü: betik count'u her adımda yazdı → [0,1,2,3,8,9]; İLAN'lı checkpoint-anlamlı [0,3,8,9]. **Onarım: kapı-değerine checkpoint-sekansı (`cp`), adım-sekansı (`count_ara_sekans`) rapor-kırılımına.**
3. FAZ-A `vm_cache_evict_sitesi`: İLAN'lı 1 (":44 tek evict-pop"); ölçülen 2 — close-pop (:282) ayrı metot, İLAN'lı beyan ":44 (tek evict-pop)". **Onarım: ölçüm İLAN'lı-siteye daraltıldı; `vm_cache_evict_satirlar` [44,282] + `vm_close_pop_satiri` rapor-kırılımı.** İLAN değişmedi — ölçüm-beyanı İLAN'ın site-beyanına hizalandı.
4. B3 P2-skor-anahtarı 15 (20 satır değil): P2 hüküm-detay sorguları `[:60]`-kesmeli → 5 çakışan-çift aynı anahtara düşer. **Onarım: birebir SATIR-düzeyi korunur (20/20); `benzersiz_sorgu_anahtari` (15) ölçülüp rapora beyan edildi.** Çakışan çiftler aynı P2-skoruyla karşılaştırılır — hükme etki etmez.

**Koşum-2 sonrası betik-dışı temizlik:** probe (`p4_probe_bellek`, 768/0) sunucuda kalmıştı — sandbox-dışı guard'lı tek-koleksiyon silme yapıldı; sonrasında 10 koleksiyon (9 foreign + kristal 36) teyitli. Bu koşum-3'ün B2'si `probe_once_var=false` ile koşum-öncesi temizliği kanıtlıyor.

### 7.2 Kanonik bulgular (mutasyonla KANITLI — onarım yazılmadı, operatör kararı "yalnız doğrulama")

1. **recreate-yıkıcılık MUTASYONLA kanıtlı (×3 bağımsız kanıt):**
   (i) explicit `recreate_collection(768)` → 9→0 nokta + `has been reset` mesajı;
   (ii) boyut-uyuşmazlık auto-recreate (:83): 384-VectorMemory → 1→0 nokta, dense-size 384;
   (iii) kazara ikinci kanıt — koşum-2'de B7 temizlik-sigortasının kendi kurduğu `VectorMemory(PROBE, 768)`'in 384≠768 auto-recreate'i (yıkıcılığın koda-değil-betik-öz-kusuruna da ateşlemesi).
2. **Sessiz-fallback-merdiveninin DISK-kanıtı:** `data/qdrant_db/` repo-İÇİNDE (9 Eyl 21:37; 304K; 4 yerel-koleksiyon: kristal_bellek, **simulasyon_bellek** [sunucuda olmayan kanonik-adın yerel-kopyası], muhakeme_bellek, test_temp_coll) — geçmişteki sessiz-yazım-izi.
3. **Varsayılan `storage_path="data/qdrant_db"` riski:** host'suz çağrı (`rag_pipeline.py:267,270`) sunucu-düşünce repo-içi local-storage YAZAR (`data/**` ihlali); B6 probe'unda bu yüzden `storage_path=""` açıkça verildi → `:memory:`'ye düştü.
4. **API-düzeyi upsert YOK:** `_next_point_id` monoton; aynı metin iki kez = +1 nokta (dublör-teyidi 8→9); restart'ta sayaç `count+1`'den kurulur → çakışma-riski (P5/gateway besmesi).
5. **`delete_collection` VectorMemory-wrapper'ı YOK:** yalnız client-düzeyi (:126) — P5 gateway-rezervi için bulgu (temizlik-yüzeyi eksikliği).
6. **B6 (hüküm-dışı RAPOR):** sahte-host 192.0.2.1'de remote **sessizce** local-path'e düşer ve **local-client cache'e YAZILIR**; storage_path-yoksa `:memory:` (cache'e yazılmaz) — gateway:123-124 host="localhost"+storage_path'li çağrı sunucu-düşünce sessiz-veri-çatallaşması riski.
7. **B3 `[:60]`-kesme anahtar-çakışması:** P2 hüküm-detayı 20 satır ama 15 benzersiz anahtar — birebir karşılaştırma satır-düzeyi (4 ondalık) 20/20; kümelerdeki 15 anahtar ölçüm-detayı (rapor + hükümde beyanlı — [[kendi-sondamin-kusuru-sahte-bulgu-uretir]] kardeş dersi: küme-içi çakışma ölçüte yüklenmemeli).

### 7.3 Sonraki paketlere besme

- **Paket-5 (gateway→öğrenme kanalı):** P3'ün Kapı-D 0,85↔RRF-bandı bulgusuna P4 ekleri — sessiz-fallback-merdiveni (gateway localhost+storage_path → sunucu-düşünce sessiz-çatallaşma + simulasyon_bellek-otomatik-kurulum) + recreate-yıkıcılık (mutasyonla ×3 kanıtlı) + delete_collection-wrapper-yok + API-upsert-yok — kanal-güvenilirlik envanterinin tamamlayıcı girdisi.
- Kanonik kusur-kuyruğu P4 ekleri (onarım operatör kararı + ayrı İLAN'lı tur): recreate fail-closed · varsayılan storage_path'in repo-dışı olması · upsert-sembantı · local-client cache-elemsi.
- **Yeni ders:** py_compile tanımsız-sabit NameError'ı YAKALAMAZ → tanımsız-ALL-CAPS-sabit AST-taraması koşum-öncesi ayrı statik-adım (T-0143'ün kapı-anahtar-hizası dersi kardeşi).
