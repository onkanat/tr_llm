# MİMARİ DOĞRULAMA PAKET-4 İLANI — Üniversal Hafıza (VectorMemory Yaşam-Döngüsü, T-0144)

**Damga (koşum ÖNCESİ, `date -u`):** 2026-09-27T12:12:54Z (UTC)
**Operatör onayı:** plan modu onayı 27 Eyl 2026 (.claude/plans/enchanted-wiggling-moon.md
— P4 üzerine yazıldı) + AskUserQuestion kararları: **"Yalnız doğrulama"** (kanonik
kod IMPORT-only; recreate-yıkıcılığı ONARMA, mutasyonla KANITLA) + **"Geçici probe
+ sonda sil"** (yazım+recreate+delete YALNIZ kendi `p4_probe_bellek`
koleksiyonunda; koşum sonunda kaldırılır).
**Yürütücü:** claude
**Kira:** T-0144 — `scripts/dogrulama_p4_universal_hafiza.py`, `data/eval/` (dir),
`.agent-bus/state/` (dir), `.agent-bus/notes/T-0144.md` — acquire ok:true (2 çağrı)

**Girdiler (salt-okunur, DONMUŞ):**
- Kanonik modül: `src/rag/vector_memory.py` (305 satır) + `src/rag/embedding.py`
  — IMPORT-only, YAZIM YOK (kopya YASAK)
- Kanonik yardımcı: `build_query_vectors` (`src/rag/rag_pipeline.py:33`) — IMPORT
- Envanter yardımcıları (P2 betik-arası yeniden-kullanım, kanonik DEĞİL):
  `_envanter` / `_envanter_digest` (`scripts/dogrulama_p2_rag_gezgini.py`)
- Çıpa-digest: P2 hüküm `mimari_dogrulama_p2_hukum_2026-09-27.json` (b2_pozitif
  detay skorları — B3 birebir-karşılaştırma kaynağı)
- Kanal çıpası: `data/future_train_vector.jsonl` — **0 bayt** (sha256
  `e3b0c442…b855`; 27 Eyl 12:12 ölçümü: 0 bayt, digest birebir)

**Sunucu (Qdrant 192.168.1.5:6333):** `kristal_bellek` 36 nokta (hibrit dense 768
Cosine + sparse IDF + crystal_tags; P2==P3 digest çıpası `fe7fc7649737ed35`) ·
foreign 9 koleksiyon · `simulasyon_bellek` sunucuda YOK · **P4 yazım-yüzeyi
yalnız `p4_probe_bellek`** (kurulum → yazım → recreate → delete — operatör kararı).

**Model YÜKLEMEZ:** P4 koşumu tokenizer-morfem tabanlı deterministik embedding
kullanır (`generate_kristal_vector` + `generate_sparse_vector` — kanonik,
`random.Random(seed)` seed'li) — KristalLM YÜKLENMEZ; MPS YOK; çift-eğitici YOK;
saf CPU + ağ. seed=42.

---

## 1. Kapsam ve hüküm kuralı

Operatör kararları (27 Eyl 2026): (i) **YALNIZ DOĞRULAMA** — recreate-yıkıcılık
satırları [83,123,126] ONARILMAZ; davranış **mutasyonla kanıtlanır** (yalnız
probe koleksiyonunda); (ii) **GEÇİCİ PROBE + SONDA SİL** — `p4_probe_bellek`
koşum sonunda `delete_collection` ile kaldırılır; sunucu-sonu envanteri
koşum-öncesi listeyle birebir.

**Hüküm (koşum ÖNCESİ sabit):** FAZ-A İLAN'lı birebir + B1 bağlanma-teyidi +
B2 arz-çıpa birebir + B3 determinizm (P2-birebir 20/20 + iç-çift-koşum 20/20)
+ B4 add-yolu (count-sekansı + geri-okuma birebir + pozitif-kontrol) + B5
mutasyon-kanıtı (explicit-recreate 0 nokta + boyut-uyuşmazlık auto-recreate
0 nokta + probe-kaldırma) + B7 dokunulmazlık → `P4_GECTİ` (rc=0); aksi her
dal → `DUR` (rc=2, stderr+rc kayıtlı). Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm
YOK; koşum-sonrası İLAN yumuşatılmaz. **B6 (sessiz-fallback probe) hükme
bağlanmaz — RAPOR kırılımıdır** (sahte-host 192.0.2.1 RFC-5737 + `$TMPDIR`
local-storage; merdiven-davranışı kaydı, yıkıcılık DEĞİL).

## 2. FAZ-A kod-envanteri — İLAN'lı değerler (koşumsuz; grep/AST)

| Ölçüm | İLAN'lı değer | Kaynak |
|---|---|---|
| `vm_qdrantclient_kurulum` | **3** (host :50 / local-path / `:memory:`) | grep `QdrantClient(` |
| `vm_remote_timeout_sn` | **2.0** | `vector_memory.py:50` |
| `vm_cache_yazim_sitesi` | **1** (tek yazım `:73`; evict-pop `:44` ayrı) | grep `VectorMemory._shared_clients[` |
| `vm_cache_guardi` | **"if not self.is_in_memory:"** (`:72` — in-memory cache'e YAZILMAZ) | satır-metni |
| `vm_recreate_delete_satirlar` | **[83, 123, 126]** (P2/P3 çıpası birebir) | grep `recreate_collection\|delete_collection` |
| `vm_otomatik_kurulum_satirlar` | **[75, 76, 77]** (koleksiyon-yoksa sessiz `_create_hybrid_collection`) | satır-bandı |
| `embedding_random_seedli` | **1** (`random.Random(` tek site; seed'siz `random.`-çağrı **0**) | `embedding.py:68` |
| `gateway_localhost_kurulum` | **2** (`agent_gateway.py:123-124` — host="localhost" + storage_path: **sessiz-fallback riski**) | grep |
| `gateway_kanonik_koleksiyonlar` | **["kristal_bellek", "simulasyon_bellek"]** (`:123-124`) | aynı |
| `gateway_yikici_yuzey` | **KOŞULMAZ** (`scripts/sanitize_vector_memory.py:141` — envanter-beyanı) | beyan |
| envanter ÖNCE: foreign / kristal / simulasyon | **9 / 36 / YOK** | sunucu (P2/P3 çıpası) |
| envanter-digest ÖNCE (16-hex) | **`fe7fc7649737ed35`** | P2==P3 koşumu SABİT |
| `data/future_train_vector.jsonl` | **0 bayt** (`e3b0c442…b855`) | disk ölçümü 12:12Z |

## 2. FAZ-B — koşumlu kapı-protokolü (İLAN'lı)

Ortak: kanonik import `VectorMemory`, `generate_kristal_vector`,
`generate_sparse_vector`, `build_query_vectors`; seed=42; sorgu-kümesi P2
seed'li seçim protokolü (random.Random(42).sample + sorted, `POZITIF_N=20`).

- **B1 bağlanma-teyidi (HÜKÜM):** kanonik kurucu
  `VectorMemory("kristal_bellek", 768, host="192.168.1.5", port=6333)` →
  `storage_type == "remote (192.168.1.5:6333)"` + `is_in_memory == False` +
  `get_document_count() == 36` + cache-anahtarı `"remote:192.168.1.5:6333"`
  `_shared_clients`'te teyit + **ikinci-örnek cache-reuse** (kademe-2):
  yeni kurucu → yeni `QdrantClient`-kurulumu DEĞİL, `storage_type` cache'ten
  birebir.
- **B2 arz-envanteri (HÜKÜM, koşum ÖNCESİ):** `_envanter` + `_envanter_digest`
  → digest == `fe7fc7649737ed35` (P2/P3 çıpası) + foreign 9 + `kristal_bellek`
  36 + `simulasyon_bellek` YOK (İLAN'lı).
- **B3 recall-determinizm (HÜKÜM):** 20 pozitif sorgu →
  `build_query_vectors` + `hybrid_recall(top_k=1, query_tags)` → (i) **P2
  hüküm `b2_pozitif.detay` skorlarıyla birebir == 20/20** (4 ondalık; aynı
  sorgu+koleksiyon+deterministik-embedding); (ii) **iç-çift-koşum**: aynı
  20 sorgu ikinci VectorMemory-örneğiyle (cache-reuse yolu) tekrar →
  birebir == 20/20. Ek (RAPOR): OOV sorgu `zzqwxx zqxwv zqqzzq` recall
  teyidi (P2/P3 skor 0,7 bandı).
- **B4 add-yolu probe (HÜKÜM):** `VectorMemory("p4_probe_bellek", 768, host…)`
  → koleksiyon YOK → **sessiz otomatik-kurulum** (:75-77; `hibrit` şema
  dense 768 Cosine + sparse IDF teyit) → `add_document` ×3 → count **3** →
  `add_documents_batch` ×5 → count **8** → **dublör-teyidi**: aynı metin
  tekrar `add_document` → count **9** (İLAN'lı davranış: id-sayaç
  `_next_point_id` monoton; **API-düzeyi upsert YOK** — Qdrant-upsert yalnız
  aynı point-id'de geçerli; sayaç-yeniden-kurulumda `count+1` çakışma-riski
  RAPOR §7) → `list_documents` geri-okuma **9/9 payload+text birebir** →
  **pozitif-kontrol**: 8 probe-belgenin kendi-metniyle recall top-1 kendi-belge
  **8/8** (dense_recall + hybrid_recall İKİSİ; P2 K5 pozitif kalıbı).
- **B5 recreate-yıkıcılık MUTASYON-kanıtı (HÜKÜM — yalnız probe üstünde):**
  (i) `recreate_collection(768)` doğrudan çağrı (9 nokta) →
  `get_document_count() == 0` (İLAN'lı beklenti-0 — çelişki-teyit GEÇTİ dalı)
  + reset-mesajı (`has been reset`) teyidi; (ii) probe'ye 1 belge geri-yaz →
  **boyut-uyuşmazlık auto-recreate** (:83 yolu): fresh
  `VectorMemory("p4_probe_bellek", 384, host…)` → sessiz recreate →
  `count == 0` + dense-size **384** teyidi (**sessiz-silme CANLI-kanıt;
  onarım YAZILMAZ — bulgu**); (iii) **temizlik:** `delete_collection` →
  `collection_exists == False`.
- **B6 sessiz-fallback probe (RAPOR — hükme bağlanmaz):** sahte-host
  `192.0.2.1` (RFC-5737; timeout 2,0 sn) + `$TMPDIR`-içi storage_path →
  merdiven-teyidi: remote-başarısız → local-düşüş
  (`storage_type == "local (…)"`, `is_in_memory == False`) + **local-client
  cache'e YAZILIR** (:72-73 guard teyidi — risk-bulgu); storage_path
  vermeyen ikinci örnek → `:memory:` (`is_in_memory == True`) + cache'e
  YAZILMAZ-teyidi. Her iki örnek `close()` ile kapatılır; `$TMPDIR`
  storage koşum-sonunda silinir (repo-dizine local-storage YOK).
- **B7 dokunulmazlık (HÜKÜM, koşum SONU):** envanter-digest SONRA ==
  ÖNCE (`fe7fc764…`) + `kristal_bellek` 36→36 + foreign 9 birebir +
  sunucuda `p4_probe_bellek` YOK + `simulasyon_bellek` hâlâ YOK +
  `data/future_train_vector.jsonl` **0→0 bayt** + P2/P3-hüküm-dosyaları
  digest'i değişmedi.

## 3. Hüküm kuralı (koşum ÖNCESİ sabit — betik içinde, elle YOK)

`P4_GECTİ` (rc=0): FAZ-A İLAN'lı birebir (9 satır) **VE** B1 (storage_type +
is_in_memory + 36 + cache-anahtarı + cache-reuse) **VE** B2 (digest çıpası +
9/36/YOK) **VE** B3 (P2-birebir 20/20 + iç-çift-koşum 20/20) **VE** B4
(count-sekansı [0,3,8,9] + geri-okuma 9/9 + pozitif 8/8) **VE** B5 (0 nokta ×2
+ dense-size 384 + reset-mesajı + kaldırma-teyidi) **VE** B7 (digest ÖNCE==
SONRA + 36 + 9 + probe-YOK + simulasyon-YOK + kanal-0) → aksi her dal →
`DUR` (rc=2). **Statik hizalama-ön-kontrolü koşumdan AYRI adımdır** (AST ile
hüküm-anahtar-kümeleri ↔ rapor-özet üreticisi; T-0143 dersi) + py_compile
AYRI adım.

## 4. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p4_universal_hafiza.py \
  --host 192.168.1.5 --port 6333 \
  --korpus data/realistic_rag/test_natural_150.jsonl \
  --p2-hukum data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json \
  --ilan data/eval/mimari_dogrulama_p4_ilan_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p4_sonuc_2026-09-27.md \
  --hukum-json data/eval/mimari_dogrulama_p4_hukum_2026-09-27.json
```

- **Koşum sandbox DIŞI** (Qdrant trafiği sandbox proxy'sinden geçmez — P1/P2/P3
  ölçüldü). Embedding saf CPU'da; model YÜKLEMEZ; MPS YOK; seed=42.
- Çıktı: rapor + hüküm JSON betikten; rc ∈ {0, 2}. Bu İLAN'ın sha256
  RAPORA işlenecek (İLAN ≠ RAPOR).

## 5. DOKUNULMAZ / YAPILMAYANLAR

`src/rag/**` · `src/gateway/**` · `src/compiler/**` · `src/llm/**` kanonik
yüzeyi IMPORT-only (YAZIM YOK) · **`kristal_bellek` YALNIZ OKUNUR** (36 nokta;
hibrit_recall salt-okunur arama — yazım YOK) · **9 foreign koleksiyon
DOKUNULMAZ** · **`simulasyon_bellek` KURULMAZ** (sunucuda yok-kalır) ·
yazım+recreate+delete **YALNIZ `p4_probe_bellek`** ·
`scripts/sanitize_vector_memory.py` KOŞULMAZ · `data/**` yazımı yalnız
`data/eval/` · `data/future_train_vector.jsonl` 0-bayt çıpa · ESİK_ROUGE
0,3221 / TAVAN_ROUGE_DECOMP 0,9509 ilgilendirmez (P4 ROUGE ölçmez) ·
commit ayrı operatör onayıyla · `git add -A` YASAK · push YOK.