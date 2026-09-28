# İLAN-1 — T-0151: kristal_bellek → anka_bellek koleksiyon yeniden-adlandırma (kod-yüzeyi)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-28T06:16:09Z (revizyon-1:
geçiş-envanteri satır-sayısından `grep -o` geçiş-sayısına düzeltildi — 34→36;
betik yok, koşum yok, revizyon koşum-öncesi)
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Hüküm BETİKTEN
(`scripts/dogrulama_t0151_anka_bellek.py`); elle sayı/hüküm YOK.

## Kapsam ve meşruiyet-çıpası

Operatör kararı (28 Eyl 2026): "operatör kararı bekleyen işlere devam
edelim… dört işe başla" + sunucu-beyanı **192.168.1.9:6333** (Qdrant canlı).
P4 kaydındaki "kristal_bellek → anka_bellek sonraki paketlere kalsın" maddesi
bu turda kapanır. Kiralamalar T-0151 (src/, scripts/, tests/, data/eval/,
.agent-bus/notes/ — ÜST-DİZİN).

## Yeniden-adlandırma-yüzeyi (envanter ölçüldü, grep)

Koleksiyon-adı sabiti `"kristal_bellek"` → `"anka_bellek"`, yalnız bu 7 dosya:

| Dosya | geçiş |
|---|---|
| `src/rag/vector_memory.py` | 1 (kurucu default) |
| `src/gateway/agent_gateway.py` | 9 |
| `src/gateway/pedagogical_supervisor.py` | 11 |
| `scripts/sanitize_vector_memory.py` | 9 |
| `scripts/sanitize_all_accumulated_datasets.py` | 1 |
| `scripts/run_agent_arena.py` | 2 (aynı satırda 2) |
| `tests/test_agent_gateway.py` | 3 (davranış-güncelleme; test-silme YOK) |

**DOKUNULMAZ-İSTİSNA:** `scripts/dogrulama_p{2,3,4,5}_*.py` geçmiş-hüküm
çıpa-betikleri (K5 sha-sabit kanıtlar) — bu betiklerde eski-ad KALIR
(betikler geçmiş-koşumların artefaktı; yeniden-koşumlarında ayrı revizyon
kararı gerekir).

## Canlı-sunucu yüzeyi (SALT-OKUMA; canlı-yazım BU TURDA YOK)

- Sunucu **192.168.1.9:6333** ayakta (ölçüldü): 10 koleksiyon — `kristal_bellek`
  + 9 foreign (elektor_articles, octave_articles, rapberry_pi_pico_all_articles,
  rendergit_01/02/03_articles, rp2040_articles, sdr_articles, türk_articles).
- **Çıpa (koşum-öncesi ölçüldü):** `kristal_bellek` status=green,
  points_count=**36**, indexed=36, dense 768 Cosine, sparse(idf),
  on_disk_payload=true.
- Bu tur sunucuya **YAZMAZ** (davranış-testleri `:memory:`). Canlı-migrate
  (yeni-ad koleksiyon kurulumu + `data/pedagogy_canonical/**` re-index +
  eski-koleksiyon silme 4D-delete_collection ile) **RAPOR §7'de ayrı
  operatör-onay maddesidir.**
- Dokunulmazlık-kanıtı: tur-sonunda canlı-envanter YENİDEN ölçülür —
  `kristal_bellek` 36-point SABİT (salt-okuma ihlali = DUR).

## Kapılar (koşum öncesi sabit)

- **K1 ESKİ-AD SIFIR:** grep `kristal_bellek` src/ + canlı-scripts + tests
  == 0 (İSTİSNA beyanlı dogrulama_p*.py hariç).
- **K2 YENİ-AD ENVANTER:** grep `anka_bellek` dosya-kırılımı == İLAN
  tablosundaki geçiş-sayıları (7 dosya, **36 geçiş**; `grep -o` geçiş-sayısı).
- **K3 STATİK-ÖN:** py_compile 7-dosya + AST tanımsız-ad 0.
- **K4 DAVRANIŞ-TESTİ:** `agent_gateway` yeni-ad default ile çalışır
  (`:memory:` — canlıya YAZMAZ; create-default exists-kapısı yeni-ad
  koleksiyonu bellek-içi kurar; `inject_knowledge` target_collection
  default == anka_bellek).
- **K5 ÇIPA-SABİT:** P2/P3/P4/P5 doğrulama-betik shaları değişmez
  (dokunulmazlık; koşum-öncesi ölçüldü — revizyon-2):
  P2 `d0d0b66e…` · P3 `1249a055…` · P4 `4790e79a…` · P5 `aee7596f…`
  (tam-deigest betikte karşılaştırılır); `data/qdrant_db/` disk-artefaktları
  BU TURDA DOKUNULMAZ (T-0153 temizlik-turu ayrı).
- **K6 CANLI-DOKUNULMAZLIK:** tur-sonunda `kristal_bellek` points==36 SABİT
  (salt-okuma kanıtı).
- 6/6 → **T0151_GECTI rc=0**; aksi her dal → **DUR rc=2**.

## pytest (ayrı koşum — hüküm-dışı)

Baseline onarım-öncesi ölçülür; geçen == baseline VE yeni-düşen YOK.
`tests/test_agent_gateway.py` 3-geçiş davranış-güncellemesi İLAN'da beyanlı.

## DOKUNULMAZLAR

- `data/**` salt-okunur (istisna `data/eval/`); `data/qdrant_db/` bu turda
  dokunulmaz; `data/pedagogy_canonical/**` frozen.
- P2-P5 hüküm-json'ları, geçmiş İLAN/RAPOR artefaktları.
- Sunucu 192.168.1.9:6333 — yalnız GET (salt-okuma); yazım YOK.

## Beyan: damga-yöntemi

Damga betikten `date -u` işlendi (mtime-çıpası); betik-sha RAPOR-digest
tablosundadır. Hüküm-JSON: ilan_sha + damga + kapılar.