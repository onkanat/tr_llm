# MİMARİ DOĞRULAMA PAKET-2 İLANI — RAG Gezgini Dikişi (Qdrant 192.168.1.5:6333) (T-0142)

**Damga (koşum ÖNCESİ, betikten):** 2026-09-27T11:25:29Z (UTC, `date -u`)
**Operatör onayı:** plan modu onayı 27 Eyl 2026 (.claude/plans/enchanted-wiggling-moon.md) + mid-turn yönlendirme: **"Kendi verimizi yükle, mevcutlara dokunma"**
**Yürütücü:** claude (operatör kararı: "Ben doğrudan")
**Kira:** T-0142 — `scripts/dogrulama_p2_rag_gezgini.py`, `data/eval/` (dir), `.agent-bus/state/` (dir), `.agent-bus/notes/T-0142.md` (file) — acquire ok:true (2 çağrı, scope dir+file)

**Girdi (salt-okunur, DONMUŞ):** `data/realistic_rag/test_natural_150.jsonl` — sha256 `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` (150 satır; şema: instruction/input/output/entity/domain/is_counterfactual). Ayrıca P1 betiği kalıp-referansı `ee2251cf…`.

---

## 1. Kapsam ve kural

Operatör kararları (27 Eyl 2026): (i) **YALNIZ DOĞRULAMA** — kanonik kod
(`src/rag/**`, `src/gateway/**`, `src/compiler/**`, `src/llm/tokenizer.py`) salt
IMPORT, YAZIM YOK; (ii) **"KENDİ VERİMİZİ YÜKLE, MEVCUTLARA DOKUNMA"** —

- Kanonik koleksiyon **`kristal_bellek`** kanonik `VectorMemory(collection_name="kristal_bellek", host=192.168.1.5, port=6333)` ile kurulur (betikte `create_collection`/`recreate_collection`/`delete_collection` DOĞRUDAN çağrı YOK — kurulum kanonik `__init__` yoluyladır).
- Arz kaynağı: `test_natural_150.jsonl`'in `input` alanındaki benzersiz `<BELGE>…</BELGE>` metinleri; kanonik `KristalTokenizer.encode/decode` ile `crystal_tags` derlenir; `generate_kristal_vector` + `generate_sparse_vector` ile hibrit vektör; `add_documents_batch` (payload şeması `domain/crystal_tags/token_ids/text` — `rag_pipeline.py:288-292` fallback şemasıyla AYNI).
- **Mevcut 9 foreign koleksiyon DOKUNULMAZ:** yalnız okuma/envanter; foreign koleksiyonlarda `VectorMemory` inşa EDİLMEZ (`recreate`-yıkıcılık `vector_memory.py:81-83` kod-envanteriyle, koşumsuz kaydedilir).

**Hüküm (operatör kararı, koşum ÖNCESİ sabit):** FAZ-A birebirlik + B1 arz-eksiksizlik + B2 pozitif kontrol 20/20 + B3 İLAN'lı-davranış + FAZ-C İLAN'lı sayılar → `P2_GECTİ` (rc=0); aksi her dal → `DUR` (rc=2, stderr+rc kayıtlı). Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK.

## 2. FAZ-A sunucu envanteri — İLAN'lı değerler (koşum-öncesi keşif, 11:14-11:22Z bandı)

| Ölçüm | İLAN'lı değer | Kaynak |
|---|---|---|
| Koleksiyon sayısı (koşum ÖNCESİ) | **9** | GET /collections (canlı ölçüm) |
| Kanonik ad mevcudiyeti (`kristal_bellek`, `simulasyon_bellek`) | **0/2** | aynı |
| Hibrit-config'li koleksiyon (sparse config VAR) | **0/9** | GET /collections/{ad} ×9 |
| Tek-dense 768 Cosine koleksiyon | **9/9** | aynı |
| Payload'da `crystal_tags` içeren | **0/9** | scroll örnekleme (elektor, türk, rendergit_01) |
| `türk_articles` nokta sayısı | **5** | GET /collections/türk_articles |
| `elektor_articles` nokta sayısı | **90.122** | aynı |
| Foreign payload şema-anahtarları | **{article_id, title, year, filename, chunk_index, text}** (6-anahtar) | scroll |

Sapma = fail-closed **DUR** (sunucu-durumu dış değişken; koşum sırasında beklenmedik değişim de DUR üretir — iki ölçüm arasında tutarlılık kapısı).

## 3. FAZ-B kendi-veri arzı + döngü (İLAN'lı protokol)

- **B0 ÖN-koşul:** `kristal_bellek` sunucuda **YOK** teyidi (varsa nokta-sayısı ölçülür, üzerine YAZILMAZ — sayı raporlanır ve o dal İLAN'lı "MEVCUT_KORU" davranışıdır).
- **B1 kurulum + arz:** kanonik `VectorMemory` → hibrit koleksiyon teyidi (dense 768 Cosine + sparse IDF-modifier); benzersiz `<BELGE>` metinleri → kanonik derleme → `add_documents_batch` → `get_document_count()` == benzersiz-BELGE sayısı (**arz-eksiksizlik**: N/N, kesme yok — "sessiz kırpma yok" standardı).
- **B2 pozitif kontrol:** 20 sorgu, seed 42, deterministik seçim → kanonik `build_query_vectors` → `hybrid_recall(top_k=1, query_tags)` → **İLAN'lı beklenti: top-1 text, yüklenen KENDİ belgelerimizden biri VE `has_root_match=True`** → **20/20**. İkincil kayıt (hükme bağlanmaz): `match_score` dağılımı, `is_context_usable(0,40)` kırılımı.
- **B3 negatif kontrol:** OOV sorgu probe'u → **İLAN'lı beklenti: istisna YOK**, davranış envanteri (sessiz-boş mu, düşük-skor mu — kaydedilir; KSUR-4 kardeşi).
- **Dokunulmazlık kapısı (koşum SONU):** FAZ-A envanteri koşum sonunda TEKRAR ölçülür; 9 foreign koleksiyon nokta-sayısı + koleksiyon-listesi digest'i ÖNCE/SONRA birebir → kapı. Sapma = DUR + derhal bildirim.

## 4. FAZ-C kod-envanteri (koşumsuz — kodda kanıt, satır-referanslı)

1. Kanonik-ad ↔ sunucu-ad küme karşılaşması (İKİ YÖNLÜ; özet-bayat dersi: ölç, varsayma).
2. Hibrit-config uyum ölçümü (FAZ-A ile birleşik).
3. **Kopya sayımı:** `RAG_MATCH_THRESHOLD` tanım+import sitesi sayımı; normalizasyon/root-filtre bloklarının (special-tokens kümesi + inflection-prefix demeti) tekrar sayımı — grep+AST ile; İLAN'lı beklenti: eşik **1 tanım** (`rag_pipeline.py:30`), import-yönüyle paylaşım. Sapma = kopya-sayısı bulgusu (rapora; hükme bağlanmaz — yapı-envanteri).
4. **Recreate-yıkıcılık** (`vector_memory.py:81-83`) ve **sessiz-None** (`rag_pipeline.py:52-55`) satır-referanslı envanter kaydı — koşumsuz, kod-incelemesi kanıtı.

## 5. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p2_rag_gezgini.py \
  --host 192.168.1.5 --port 6333 \
  --korpus data/realistic_rag/test_natural_150.jsonl \
  --koleksiyon kristal_bellek \
  --ilan data/eval/mimari_dogrulama_p2_ilan_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p2_sonuc_2026-09-27.md
```

- **Koşum sandbox DIŞI** (Qdrant trafiği sandbox proxy'sinden geçmez — 1 ms proxy-redi 27 Eyl ölçüldü). Saf CPU+ağ; MPS YOK; çift-eğitici YOK.
- Çıktı: rapor + hüküm JSON (`data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json`) betikten; rc ∈ {0, 2}.
- Bu İLAN dosyasının sha256'sı RAPORA işlenecek (İLAN ≠ RAPOR; koşum-öncesi damga).

## 6. DOKUNULMAZ / YAPILMAYANLAR

`src/rag/**` · `src/gateway/**` · `src/compiler/**` · `src/llm/tokenizer.py` ·
`scripts/evaluate_carpenter_anka.py` DOKUNULMAZ · `data/**` yazımı yalnız
`data/eval/`'e (kaynak korpus DONMUŞ salt-okunur) · **9 foreign koleksiyon + içindeki
~91.000 nokta DOKUNULMAZ** (yazım/silme/recreate YOK) · ESİK_ROUGE 0,3221 /
TAVAN_ROUGE_DECOMP 0,9509 ilgilendirmez (bu paket ROUGE ölçmez) ·
`future_train_vector.jsonl` yazımı YOK (Paket-5) · commit ayrı operatör onayıyla ·
`git add -A` YASAK.