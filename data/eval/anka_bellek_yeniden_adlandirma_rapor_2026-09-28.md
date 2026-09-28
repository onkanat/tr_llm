# T-0151 — anka_bellek yeniden-adlandırma koşumu (sonuç)

**Hüküm:** **DUR** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T06:28:14Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/anka_bellek_yeniden_adlandirma_ilan1_2026-09-28.md` (sha256 `4e2383c3bacbf42d74206e454d4ef79269d3c991bc051f1842c1c9a1499a9cac`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 ESKİ-AD SIFIR (İSTİSNA dogrulama_p*) | `["scripts/dogrulama_t0151_anka_bellek.py"]` | DÜŞTÜ |
| K2 YENİ-AD ENVANTER == İLAN (36 geçiş) | `{"src/rag/vector_memory.py": 1, "src/gateway/agent_gateway.py": 9, "src/gateway/pedagogical_supervisor.py": 11, "scripts/sanitize_vector_memory.py": 9, "scripts/sanitize_all_accumulated_datasets.py": 1, "scripts/run_agent_arena.py": 2, "tests/test_agent_gateway.py": 3}` | GEÇTİ |
| K3 STATİK-ÖN: py_compile + AST tanımsız-ad 0 | `[]` | GEÇTİ |
| K4 DAVRANIŞ: default'lar + resolve + status (:memory:, canlıya yazım YOK) | `{"vector_memory_default": "anka_bellek", "is_in_memory": true, "inject_default": "anka_bellek", "check_default": "anka_bellek", "resolve_anka": "anka_bellek", "resolve_bilinmeyen_istisna": "ValueError", "status_anahtarlar": ["anka_bellek_docs", "device", "epistemic_backlog_samples", "future_train_path", "simulasyon_bellek_docs", "status"], "status_anka_anahtar": true}` | GEÇTİ |
| K5 ÇIPA-SABİT: P2-P5 betik-shaları birebir | `{"scripts/dogrulama_p2_rag_gezgini.py": "d0d0b66e0ad0612a…", "scripts/dogrulama_p3_epistemik_kapilar.py": "1249a055f5162acb…", "scripts/dogrulama_p4_universal_hafiza.py": "4790e79a51d2e3f3…", "scripts/dogrulama_p5_ogrenme_kanali.py": "aee7596f6074bf9e…"}` | GEÇTİ |
| K6 CANLI-DOKUNULMAZLIK: kristal_bellek 36 SABİT | `{"sunucu": "http://192.168.1.9:6333", "status": "green", "points_count": 36, "indexed": 36}` | GEÇTİ |

## Tam SHA-256 digest tablosu

| Dosya | SHA-256 |
|---|---|
| `src/rag/vector_memory.py` | `efc3519faad5e11602894fa018a26297e3449e3ee3ea311ced8bc26ce74b38f5` |
| `src/gateway/agent_gateway.py` | `c313e1600b38e1a516f98af49da91bfbb7ddfb0c47898d4ffe7a61096d2bbda5` |
| `src/gateway/pedagogical_supervisor.py` | `eb271d8fa3f5f19b4f0c485b1ecb8dcb688352f0ba45cfb3852736180c747ea6` |
| `scripts/sanitize_vector_memory.py` | `43de22b7fc5d53a0f9cc3107bb3c741b0186b1d717e3947abebf6fccc510f67f` |
| `scripts/sanitize_all_accumulated_datasets.py` | `f1b6dc41922ee23306599f66a35283c91cfdc1b4c809748f8b1f383ac215fa2f` |
| `scripts/run_agent_arena.py` | `1fd9a260e907d7e327693c969719822504cdecdeba553eeb3ce6f3f7c151072d` |
| `tests/test_agent_gateway.py` | `944ebc26cc88e6572e0e33c7830a3197ba0ad458e6f37d03ffe9278b1182743e` |
| `scripts/dogrulama_p2_rag_gezgini.py` (çıpa) | `d0d0b66e0ad0612a2bb4e4f221d65047541e0c1eca12b90d9696b9d46717a77a` |
| `scripts/dogrulama_p3_epistemik_kapilar.py` (çıpa) | `1249a055f5162acb352ae4a9c86334e3d70803f5cf2b3f24c125b3d45d7c8c1a` |
| `scripts/dogrulama_p4_universal_hafiza.py` (çıpa) | `4790e79a51d2e3f3dfb0778cc9e40b786eeb0a155d045bdbc74eb27a5951428c` |
| `scripts/dogrulama_p5_ogrenme_kanali.py` (çıpa) | `aee7596f6074bf9e16358e98b4da4453aede44ac46960759abea409dbb5eb65a` |
| `data/eval/anka_bellek_yeniden_adlandirma_ilan1_2026-09-28.md` | `4e2383c3bacbf42d74206e454d4ef79269d3c991bc051f1842c1c9a1499a9cac` |
| `data/eval/anka_bellek_yeniden_adlandirma_hukum_2026-09-28.json` | `f70215c5d5bc696cd9ed5627bae366cc0d9f9407cf53624bdaf8aa007cd490d7` |

## §7 — canlı-migrate önerisi (ayrı operatör onayı)

Bu tur sunucuya YAZMADI. Sunucu 192.168.1.9:6333'te `kristal_bellek`
koleksiyonu 36-point olarak duruyor (K6 kanıtı). Kod artık
`anka_bellek` okur → mevcut 36 kanonik-belge erişim-dışı kalır.
Önerilen devam-turu: (a) `anka_bellek` koleksiyonu kurulumu
(aynı şema: dense 768 Cosine + sparse idf + on_disk_payload),
(b) `data/pedagogy_canonical/**` kaynağından re-index, (c) 36/36
birebir kapısı, (d) eski `kristal_bellek` silme — `delete_collection`
wrapper (T-0148 4D) ile, silme-öncesi digest/envanter kaydıyla.
Canlı-sunucu YAZIMI olduğundan AYRI operatör onayı şart.

