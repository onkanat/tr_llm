# MİMARİ DOĞRULAMA PAKET-2 SONUÇ — RAG Gezgini (Qdrant) (T-0142)

**Damga:** 2026-09-27T11:38:17Z · **Hüküm (BETİKTEN):** **P2_GECTİ (rc=0)**

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
| koleksiyon_sayisi_once | 9 | 9 | GEÇTİ |
| kanonik_ad_mevcut | 0 | 0 | GEÇTİ |
| hibrit_koleksiyon | 0 | 0 | GEÇTİ |
| tek_dense_768 | 9 | 9 | GEÇTİ |
| crystal_tagsli | 0 | 0 | GEÇTİ |
| turk_articles_nokta | 5 | 5 | GEÇTİ |
| elektor_articles_nokta | 90122 | 90122 | GEÇTİ |
| foreign_sema_anahtar | 6 | 6 | GEÇTİ |

## FAZ-B — kendi-veri arzı + döngü

- B1 kurulum: `{'dense_768_cosine': True, 'sparse': True, 'sparse_adlar': ['sparse'], 'storage_type': 'remote (192.168.1.5:6333)'}` · arz: **36/36** (derleme-boş 0)
- B2 pozitif kontrol: **20/20** · skor min/med/max: 0.29333334799999994/0.35/0.75 · eşik-üstü (0,40): 9/20
- B3 OOV negatif kontrol: istisna=False · sonuc_uzunluk=1 · skor=0.7 · has_root_match=True

## FAZ-C — kod-envanteri (koşumsuz)

- `RAG_MATCH_THRESHOLD` tanım sitesi: [] · import sitesi: 2
- root-filtre blok tekrarı: ['scripts/evaluate_rag_baselines.py', 'scripts/evaluate_rag_pilot.py', 'scripts/import_examples.py', 'src/llm/tokenizer.py', 'src/rag/embedding.py', 'src/rag/vector_memory.py']
- recreate-yıkıcılık satırları (vector_memory.py): [83, 123, 126]
- sessiz-None satırları (rag_pipeline.py): [54]

## Digest tablosu

| Dosya | sha256 |
|---|---|
| mimari_dogrulama_p2_ilan_2026-09-27.md (İLAN — koşum ÖNCESİ) | `e20c5ead50e1a84815e67ecd52ddd3867e4850d9c6572630d10c5a8c6c34ffd4` |
| data/realistic_rag/test_natural_150.jsonl (DONMUŞ kaynak) | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| scripts/dogrulama_p2_rag_gezgini.py (koşulan betik) | `f0818913f245174f07b77f0c78afa78db9a2bfa1e6307f803084faab2337d39a` |
| mimari_dogrulama_p2_hukum_2026-09-27.json (hüküm — BETİKTEN) | `9b217944ce74becbc632e9e9176af64d036025555eb2ae8f72033bf36f9b0189` |
| envanter ÖNCE/SONRA digest | `fa794dc6557eb12b` / `fe7fc7649737ed35` |
| dokunulmazlık | GEÇTİ (foreign 9 koleksiyon nokta-sayısı birebir; fark yalnız +`kristal_bellek` — KENDİ koleksiyonumuz) |

---

## §7 — Doğrulayıcı kök-neden analizi (koşum SONRASI ek; ölçüm sayıları betikten/ölçümden — elle sayı YOK)

### 7.1 FAZ-C ölçüt kusuru (doğrulayıcı hatası — beyan edilir, İLAN yumuşatılmaz)

Yukarıdaki FAZ-C satırında `RAG_MATCH_THRESHOLD` tanım sitesi `[]` (0) çıktı.
**Kök neden:** betiğin içi AST taraması yalnız `ast.Assign` düğümlerini saydı;
kanonik tanım **`ast.AnnAssign`** (type-hint'li — `rag_pipeline.py:30`, proje
standardı type-hints ZORUNLU). **Düzeltilmiş ölçüm (koşum sonrası, ayrı ölçüm):**

- `RAG_MATCH_THRESHOLD` **tanım: TEK site** — `src/rag/rag_pipeline.py`
- import sitesi: 2 — `src/rag/__init__.py`, `src/rag/epistemic_agent.py`
- normalizasyon/root-filtre blok tekrarı: 6 dosya (`evaluate_rag_baselines`,
  `evaluate_rag_pilot`, `import_examples`, `tokenizer`, `embedding`, `vector_memory`)

Hükme bağlı DEĞİLDİ (İLAN §4: kopya-sayımı yapı-envanteri, rapora). Ders
([[olcut-kendi-payini-yiyor]] kardeşi): **ölçüt kendi AST taramasıyla kendi
konuşlandığı kodu sayamadı** — AST ölçümlerinde `Assign | AnnAssign` ikisini
kapsayan sayaç yazılmalı. "Üç kopya" belleği eşik düzeyinde BAYAT çıktı.

### 7.2 B2 eşik-üstü 9/20 — RRF skor-ölçeği ↔ `RAG_MATCH_THRESHOLD` 0,40 uyumsuzluğu

20/20 pozitif kontrol GEÇTİ (kapı skor DEĞİL kimlik: top-1 kendi-belge +
`has_root_match=True`). Ama ölçülen skor bandı **0,2933–0,75 (medyan 0,35)** ve
`is_context_usable(RAG_MATCH_THRESHOLD=0,40)` kırılımında eşik-üstü yalnız **9/20**:
**RRF füzyon skor-ölçeği (0..~0,7) ile kanonik eşik 0,40 AYRI ölçeklerdedir.**
Üretim zincirinde (`rag_pipeline.retrieve_context`) bu, kendi-belgelerimizin
~%55'inin eşik-altı kalacağı (→ `retrieve_context` sessiz-None satır 54 ile
birleşince **sessiz-boş koşullanma**) anlamına gelir. **Paket-3'ün birincil
girdisi** — onarım (skor-normalizasyonu veya eşik-ölçek uyumu) kanonik kodda,
operatör kararı.

### 7.3 B3 OOV sahte-yüksek-skor sınıfı (sessiz-None'un tersi)

OOV probe (`<BOS> <UNK> <UNK> <UNK> <EOS>`): istisna YOK, `sonuc_uzunluk=1`,
**skor 0,7 > 0,40**, `has_root_match=True`. **Kök neden:** OOV sorguda
`distinctive_query_roots` BOŞ kalınca root-match ceza filtresi devre-dışıdır
(`has_root_match` varsayılan `True`) → ceza ×0,05 uygulanmaz → OOV sorgu
**sahte-yüksek-skorla koşullanabilir** (sesli-None'un tersi sınıf: fazladan-
koşullanma; sessiz-boş değil). B3 İLAN'lı davranış (istisna-yok) DOĞRULANDI;
ama davranış envanteri bu **sessiz-false-positive** sınıfını kanıtladı.
Paket-3 girdisi (epistemik döngü kapıları: düşük-kanıt sorguda kapı).

### 7.4 Diğer kanıtlanmış bulgular (kod-envanteri — koşumsuz)

- **Recreate-yıkıcılık:** `vector_memory.py` satır [83, 123, 126] —
  mevcut koleksiyonda dense-size uyuşmazlığı `recreate_collection`
  (=`delete_collection`) tetikler; paylaşımlı sunucuda sessiz veri-silme riski.
- **Sessiz-None:** `rag_pipeline.py:54` — `retrieve_context` istisnayı None'a indirger.
- **`kristal_bellek` adı bayat:** kanonik ad kristal→Anka geçişinden önce
  (Anka dersi, T-0081). Kod yazılmadığı için kanonik ad KULLANILMAK ZORUNDAYDI;
  yeniden adlandırma (kod + 3 çağrı-yüzeyi + koleksiyon) operatör kararıdır.

### 7.5 Durum

Kanonik RAG zinciri sunucuda **ilk kez kendi-şemasıyla (hibrit dense+sparse +
`crystal_tags`) doğrulandı**: kurulum → arz 36/36 → okuma 20/20 → dokunulmazlık.
Kanonik kod DOKUNULMADI (import-only); 9 foreign koleksiyon digest ile
birebir korundu.
