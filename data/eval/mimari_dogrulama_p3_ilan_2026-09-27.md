# MİMARİ DOĞRULAMA PAKET-3 İLANI — Epistemik Döngü Kapıları (T-0143)

**Damga (koşum ÖNCESİ, `date -u`):** 2026-09-27T11:48:19Z (UTC)
**Operatör onayı:** plan modu onayı 27 Eyl 2026 (.claude/plans/enchanted-wiggling-moon.md — Paket-3) + AskUserQuestion kararları: **"Yalnız doğrulama"** + **"Geçici probe yolu"** (future_train yazımı yalnız `data/eval/p3_future_train_probe.jsonl`'e; GERÇEK kanal 0-satır korunur)
**Yürütücü:** claude (operatör kararı: "Ben doğrudan")
**Kira:** T-0143 — `scripts/dogrulama_p3_epistemik_kapilar.py`, `data/eval/` (dir), `.agent-bus/state/` (dir), `.agent-bus/notes/T-0143.md` — acquire ok:true (2 çağrı)

**Girdiler (salt-okunur, DONMUŞ):**
- Taban: `data/anka_base_v2.pt` — sha256 `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` (mühürlü `2.0-sealed`; meta `data/anka_base_v2.meta.json`: vocab 33.114, RoPE, block 4096, 6/6/768)
- Sözlük (AÇIKÇA verilir — T-0087 dersi): `data/rebuild/vocab_anka_r1_33114.json` — sha256 `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` (33.114 giriş)
- Korpus: `data/realistic_rag/test_natural_150.jsonl` — sha256 `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` (DONMUŞ)
- Kanal çıpası: `data/future_train_vector.jsonl` — **0 bayt** (sha256 `e3b0c442…b855` = boş-dosya digest'i; 26 Eyl ölçümüyle aynı)

**Kanonik koleksiyon (P2'den kalıcı arz):** `kristal_bellek` — 36 nokta (hibrit dense 768 Cosine + sparse IDF + `crystal_tags`); **P3'te YALNIZ OKUNUR — yazım YOK**.

---

## 1. Kapsam ve kural

Operatör kararları (27 Eyl 2026): (i) **YALNIZ DOĞRULAMA** — kanonik kod
(`src/rag/epistemic_agent.py`, `src/rag/merak.py`, `src/llm/router.py`,
`src/rag/rag_pipeline.py`, `src/rag/vector_memory.py`, `src/llm/tokenizer.py`,
`scripts/train_step_demo.py` KristalLM yüzeyi) salt IMPORT, YAZIM YOK;
(ii) **GEÇİCİ PROBE YOLU** — `EpistemicCuriosityAgent(future_train_path=…probe)`
(kanonik parametre, `epistemic_agent.py:54`); `data/future_train_vector.jsonl`
**0-bayt kapısı koşum SONUNDA teyit edilir**.

**Hüküm (operatör kararı, koşum ÖNCESİ sabit):** FAZ-A İLAN'lı + K2 arz-korundu
+ K3 B1 kayıt-tam (20/20) + K4 B3 OOV force'lu-davranış + K5 B4 (0/20 +
probe-şema-tam + gerçek-kanal-0) + K6 birebir → `P3_GECTİ` (rc=0); aksi her
dal → `DUR` (rc=2, stderr+rc kayıtlı). Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm
YOK. **B2 (pozitif-sorgu koşullanma-kırılımı) hükme bağlanmaz — RAPOR
kırılımıdır** (tetiklenme, model-entropisine bağlı olasıl davranıştır; garanti
edilemez olay kapı yapılmaz — fail-closed ilkeyle uyumlu: kapı, garanti-edilen
davranışa bağlanır).

## 2. FAZ-A kod-envanteri — İLAN'lı değerler (koşumsuz; AST Assign|AnnAssign İKİSİNİ sayan düzeltilmiş sayaç — T-0142 §7.1 dersi)

| Ölçüm | İLAN'lı değer | Kaynak |
|---|---|---|
| `RAG_MATCH_THRESHOLD` tanım sitesi | **1** (`src/rag/rag_pipeline.py`) | T-0142 düzeltilmiş ölçüm (AnnAssign) |
| `RAG_MATCH_THRESHOLD` import sitesi | **2** (`src/rag/__init__.py`, `src/rag/epistemic_agent.py`) | aynı |
| Kapı-A satırları | `src/rag/merak.py:39-77` + `epistemic_agent.py:269-272` (tau=2,5; UNK tetikleri) | kod (AST-parse OK 27 Eyl) |
| Kapı-B satırları | `epistemic_agent.py:167,173` (`has_root_match` red) | aynı |
| Kapı-C satırları | `epistemic_agent.py:298` (`is_context_usable` + 0,40) | aynı |
| Kapı-D satırları | `epistemic_agent.py:331-365` (0,85 + future_train append) | aynı |
| `similarity_threshold` | **0,85** (kurucu varsayılan, `:53`) | aynı |
| `tau` | **2,5** (kurucu varsayılan, `:52`) | aynı |
| CuriosityEngine/TriModalRouter eğitilmiş ağırlık | **YOK** (`merak.py` 81 satır + `router.py` 92 satır; state_dict-load grep=0) — fresh-init projeksiyon; seed=42 ile deterministik init | grep ölçümü |
| `data/future_train_vector.jsonl` koşum ÖNCESİ | **0 bayt** | disk ölçümü |
| `kristal_bellek` nokta (koşum ÖNCESİ) | **36** | sunucu (P2 arzı) |
| foreign koleksiyon (koşum ÖNCESİ) | **9** (kanonik-ad kümesiyle kesişmez) | aynı |

## 3. FAZ-B — koşumlu kapı-davranışı (İLAN'lı protokol)

Model yükleme kanonik kalıpla: `KristalLM` (`scripts/train_step_demo.py`) +
`resize_state_dict` + `load_state_dict(…, strict=False)` — **kafa/sözlük uyum
kapısı: checkpoint lm_head kafası == 33.114** (uyumsuzsa DUR — T-0044 sınıfı).
Agent: `EpistemicCuriosityAgent(model, tokenizer, memory=VectorMemory("kristal_bellek", storage_path=""), future_train_path="data/eval/p3_future_train_probe.jsonl")` — CPU, seed=42, MPS YOK, tek model (çift-eğitici YOK).

- **B1 Kapı-A (merak-tetikleme):** 20 pozitif sorgu (P2 seed'li kümesi — seed 42,
  aynen) → `process_query(force_rag=False)` → her sorguda `entropy_pre`,
  `needs_retrieval`, `retrieval_triggered` kaydı **tam 20/20** (istisna = kapı
  düşer). Tetiklenme-kırılımı raporlanır; hükme bağlanan sayı **kayıt-tamlığıdır**
  (20/20), tetiklenme-oranı İLAN'lı tek değer DEĞİLDİR (bant beyanlı).
- **B2 Kapı-B/C koşullanma-kırılımı (RAPOR — hükme bağlanmaz):** 20 pozitif
  sorgu → `process_query(force_rag=False)` telemetrisinden `retrieval_triggered`,
  `match_score` (>0 ise), `retrieved_document` boş-değil (koşullanma-teyidi)
  kırılımı; triggered alt-kümesinde match_score, P2 hüküm JSON'undaki
  (`mimari_dogrulama_p2_hukum_2026-09-27.json` `b2_pozitif.detay`) aynı-sorgu
  skoruyla **birebir (4 ondalık)** karşılaştırılır (determinizm: aynı sorgu,
  aynı koleksiyon, deterministik embedding) — sapma bulgu-raporuna.
- **B3 OOV sahte-koşullanma probe'u (HÜKÜM — force'lu, garanti-edilen dal):**
  OOV sorgu (`zzqwxx zqxwv zqqzzq` — P2 B3'ün aynısı) `process_query(force_rag=True)`
  ile koşulur → İLAN'lı beklenti: **istisna YOK** + **Kapı-B sahte-geçiş**
  (`has_root_match=true` yoluyla retrieved_document BOŞ DEĞİL) + **Kapı-C
  sahte-koşullanma** (skor ~0,7 ≥ 0,40 ⇒ koşullanma) + skor
  **İLAN'lı OOV-bandı [0,6–0,8]** (P2 B3 skoru 0,7) → P2 bulgusunun
  canlı-döngü teyidi.
- **B4 Kapı-D (öğrenme-yazımı):**
  (i) varsayılan agent (0,85) ile `is_high_similarity` **beklentisi 0/20**
  (İLAN'lı çelişki-teyidi: 0,85 > P2 bant-maks 0,75 — beklenen-düşük, GEÇTİ dalı);
  (ii) **probe-agent:** kanonik kurucu argümanıyla (`similarity_threshold=0,5`,
  KOD YAZIMI YOK) failure-dalı tetiklenirse → `future_train_recorded=true` +
  record şeması **14-anahtar** (`instruction/input/output/model_failed_output/
  decompiled_output/rag_document/retrieval_collection/similarity_score/
  similarity_threshold/entropy_pre/entropy_post/tau/reason/timestamp`)
  probe-yoluna append kanıtlanır; tetiklenme-sayısı RAPOR DÜZEYİNDE (hükme
  bağlanmaz); **şema-kanıt kapısı deterministiktir: kanonik yazıcı-metodu
  `record_to_future_train` İLAN'lı-14-anahtarlı kayıtla çağrılır → probe
  dosyası +1 satır + anahtar-kümesi teyidi; şema-uyumu (kanonik
  `process_query:349-364` ile) kod-incelemesi olarak §7'de beyan edilir;
  (iii) **GERÇEK kanal kapısı: koşum-sonunda
  `data/future_train_vector.jsonl` boyutu 0 bayt** — sapma = DUR.
- **B5 dokunulmazlık (koşum SONU):** 9 foreign koleksiyon nokta-sayısı +
  koleksiyon-listesi digest'i ÖNCE==SONRA + `kristal_bellek` == 36 (P2 kalıbı).

## 4. FAZ-C kod-envanteri (koşumsuz — kodda kanıt, satır-referanslı)

1. 4 kapı satır-referanslı envanteri (yukarıda §2) + eşik sabitleri
   (`tau`, `similarity_threshold`, `RAG_MATCH_THRESHOLD` — AST AnnAssign-dahil sayaçla).
2. `is_high_similarity` ulaşılabilirlik beyanı: 0,85 ↔ P2 RRF-bandı 0,29–0,75
   çelişkisi — koşumla TEYİT edilir (B4-i), kanonik kodda ONARILMAZ (bulgu).
3. recreate-yıkıcılık (`vector_memory.py:83,123,126`) + sessiz-None
   (`rag_pipeline.py:54`) tekrar-teyidi (P2 envanteriyle tutarlılık).

## 5. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p3_epistemik_kapilar.py \
  --host 192.168.1.5 --port 6333 \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --korpus data/realistic_rag/test_natural_150.jsonl \
  --koleksiyon kristal_bellek \
  --probe-yol data/eval/p3_future_train_probe.jsonl \
  --ilan data/eval/mimari_dogrulama_p3_ilan_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p3_sonuc_2026-09-27.md
```

- **Koşum sandbox DIŞI** (Qdrant trafiği sandbox proxy'sinden geçmez — 27 Eyl
  ölçüldü). Model saf CPU'da; MPS YOK; çift-eğitici YOK; seed=42 (router/merak
  fresh-init determinizmi).
- Çıktı: rapor + hüküm JSON (`data/eval/mimari_dogrulama_p3_hukum_2026-09-27.json`)
  betikten; rc ∈ {0, 2}. Bu İLAN'ın sha256'sı RAPORA işlenecek (İLAN ≠ RAPOR).

## 6. DOKUNULMAZ / YAPILMAYANLAR

`src/rag/**` · `src/gateway/**` · `src/compiler/**` · `src/llm/**` kanonik
yüzeyi · `scripts/train_step_demo.py` · `scripts/evaluate_carpenter_anka.py`
DOKUNULMAZ (import-only) · `data/**` yazımı yalnız `data/eval/`'e ·
**`data/future_train_vector.jsonl` GERÇEK yazım YOK** (0-bayt çıpası; geçici
probe yolu — operatör kararı) · **`kristal_bellek`'e yazım YOK** (salt-okunur;
36 nokta envanteri) · **9 foreign koleksiyon DOKUNULMAZ** (recreate/delete YOK) ·
ESİK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 ilgilendirmez (P3 ROUGE ölçmez) ·
commit ayrı operatör onayıyla · `git add -A` YASAK.

## 7. Koşum-öncesi revizyon (damgalı — P1 §6 kalıbı)

- §3 B4(ii): record şeması sayımı düzeltildi **11 → 14** (kanonik
  `process_query:349-364` alan-listesi yeniden sayıldı).
- §1 hüküm + §3 B2/B3: hüküm kapıları kesinleştirildi — B2 (model-entropisine
  bağlı olasıl davranış) hüküm-dışı RAPOR kırılımına indirildi; hükmü taşayan
  koşullanma-davranışı **B3 force'lu OOV probe'una** (garanti-edilen dal)
  bağlandı. Bu revizyon İLAN digest'ine koşum-öncesi işlenir — koşum-sonrası
  yumuşatma YOK.