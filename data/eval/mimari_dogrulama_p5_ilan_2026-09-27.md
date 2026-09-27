# MİMARİ DOĞRULAMA PAKET-5 İLANI — Gateway→Öğrenme Kanalı (T-0145)

**Damga (koşum ÖNCESİ, `date -u`):** 2026-09-27T12:51:30Z (UTC) —
**REVİZYON-1** (önceki damga 12:45:03Z; **koşum YOK**, keşif-ölçümü:
train.py arayüzü + P3-probe/replay max-jeton-id — aşağıda)
**Operatör onayı:** plan modu onayı 27 Eyl 2026 (.claude/plans/
enchanted-wiggling-moon.md — P5 üzerine yazıldı) + AskUserQuestion
kararları: **"Uçtan-uca kanal"** (üç halka da koşumlu; kanonik kod
IMPORT-only, onarım yok — bulgular mutasyonla kanıtlanır, P4 kalıbı) +
**"Probe-yollar"** (`future_train_path` + `output_bin_path` +
`archive_path` + `save_path` hepsi probe; GERÇEK kanal 0-bayt çıpa
koşum boyunca KORUNUR).
**Yürütücü:** claude
**Kira:** T-0145 — `scripts/dogrulama_p5_ogrenme_kanali.py`,
`data/eval/` (dir), `.agent-bus/state/` (dir), `.agent-bus/notes/T-0145.md`
— acquire ok:true (2 çağrı; sunucu-yazımı yalnız `p5_probe_bellek` — İLAN
beyanı; bus kiralama dosya-sistemi tabanlıdır).

> **REVİZYON-1 notu (koşum-ÖNCESİ; P1 §6-revizyon kalıbı):** koşum-öncesi
> keşif-ölçümleri (salt-okunur): (a) `run_training` subprocess cmd-flag
> kümesi `["--data","--steps","--batch-size","--lr","--device",
> "--save-path"]` — **`--vocab` YOK, `--load-path` YOK** (statik,
> retrain_pipeline.py:255-264); (b) `train.py` default sözlük
> `data/rebuild/vocab_base_32852.json` (train.py:153) + fail-closed
> jeton-guard `:203-207` (`max_id >= vocab_size ⇒ RuntimeError, rc≠1
> değil rc=1 traceback); (c) koşum-öncesi max-jeton-id ön-ölçümü:
> **P3-probe 6/6 kayıt max 32195** + **replay-25 max 32831** —
> ikisi de < 32.852 ⇒ kanal-yüzeyinde tail-jetonlar (32852+) GÖRÜNMÜYOR
> ⇒ B5'in İLAN'lı tahmini **rc=0**; (d) FAZ-A düzeltmesi: `gw_ask`
> telemetri **17** anahtar (önceki İLAN'da 14 yazıldı — :210-228 sayım).

**Girdiler (salt-okunur, DONMUŞ):**
- Kanonik modüller: `src/gateway/agent_gateway.py` (441 satır) ·
  `src/gateway/retrain_pipeline.py` (315 satır) ·
  `src/gateway/pedagogical_supervisor.py` (sanitize_teacher_card —
  IMPORT-only) · `src/rag/epistemic_agent.py` (377 satır) ·
  `src/rag/vector_memory.py` · `src/rag/embedding.py` ·
  `src/rag/rag_pipeline.py` · `src/llm/tokenizer.py` · `train.py`
  (statik-envanter salt-okunur) — IMPORT-only, YAZIM YOK (kopya YASAK)
- Kanonik yardımcılar: `build_query_vectors` (rag_pipeline),
  `render_example` (prompt_contract — retrain kanonik yolu),
  `resize_state_dict` (P3 `_model_yukle` kalıbı)
- Model + sözlük (DONMUŞ, salt-okunur yükleme): `data/anka_base_v2.pt`
  (`d0f415f3…`) · `data/rebuild/vocab_anka_r1_33114.json` (`f9940a8d…`)
- Replay-buffer: `data/pedagogy/high_school_foundation_dataset.jsonl`
  (268K; `59e2f2786d0ec0d1574a77e5b7354e02ae957345a1b4ddce945d0db19527bdf1`)
- P3 çıpası: `data/eval/p3_future_train_probe.jsonl` 6 satır (P3-bırak;
  P5 probe DOSYA AYRI: `p5_probe_future_train.jsonl`)
- **GERÇEK kanal çıpası:** `data/future_train_vector.jsonl` — **0 bayt**
  (sha256 `e3b0c442…b855`; P2/P3/P4 koşumları korudu)

**Sunucu (Qdrant 192.168.1.5:6333):** `kristal_bellek` 36 nokta (çıpa
`fe7fc7649737ed35` P2==P3==P4) · foreign 9 · `simulasyon_bellek` YOK ·
`muhakeme_bellek` YOK (inject_reasoning_trace İLKELİ — koşulursa
`self.memory.client` paylaşımlı-remote ile sunucuda kurardı :291-297) ·
**P5 yazım-yüzeyi yalnız `p5_probe_bellek`** — kanonik-yolla:
probe-gateway'in `general_memory`'si `VectorMemory(collection_name=
"p5_probe_bellek")` olarak kurulur (kanonik else-dalı `:240/:340`);
inject+check+temizlik bu probe-instance üzerinden; sondа
`client.delete_collection` ile kaldırılır.

**Model YÜKLENİR (B1/B5 İLKELİ için):** kanonik P3 `_model_yukle` kalıbı —
`data/anka_base_v2.pt` CPU'da (DONMUŞ salt-okunur; **yazım YOK**);
train.py subprocess device="cpu", steps=10 — **MPS YOK;
save_path=$TMPDIR/data/eval/p5_probe_anka.pt (probe; data/*.pt'ye yazım
YOK)**; çift-eğitici YOK (tek train.py subprocess); seed=42.

---

## 1. Kapsam ve hüküm kuralı

Operatör kararları (27 Eyl 2026): (i) **UÇTAN-UCA KANAL** — yazar +
tüketici + gateway-yazım-yüzeyleri + HTTP, üç halka da koşumlu; (ii)
**PROBE-YOLLAR** — `record_to_future_train` + `compile_backlog_to_bin` +
`_archive_processed_records` + `save_path` YALNIZ probe-yollarda;
GERÇEK kanal `data/future_train_vector.jsonl` **0→0 bayt** koşum-sonu
kapısıyla teyit edilir.

**Hüküm (koşum ÖNCESİ sabit):** FAZ-A İLAN'lı birebir + B1 kanal-yazarı
(canlı-döngü 0/20 + probe-yazım ≥1 + şema 14-anahtar + append) + B2
arz-çıpa + B3 tüketici-okuma (relatif/0) + B4 tüketici-derleme (meta +
uint16 + geri-tokenize) + B5 run_training İLKELİ (rc=0 + arşiv-probe +
GERÇEK kanal sabit) + B6 gateway-yazım (probe + sahte-seçici mutasyonu) +
B7 HTTP-teyidi (5-endpoint) + B8 dokunulmazlık → `P5_GECTİ` (rc=0); aksi
her dal → `DUR` (rc=2, stderr+rc kayıtlı). Hüküm BETİK İÇİNDEDİR; elle
sayı/hüküm YOK; koşum-sonrası İLAN yumuşatılmaz. **`inject_reasoning_
trace` İLKELİ + run_training adım-düzeyi detayları + B1 telemetri
kırılımı hükme bağlanmaz — RAPOR kırılımıdır** (sabit koleksiyon-adı
riski; [[kapi-vakum-degil-mutasyonla-kanitlanir]]: koşulmayan yüzey
envanterle beyan edilir).

## 2. FAZ-A kod-envanteri — İLAN'lı değerler (koşumsuz; grep/AST)

| Ölçüm | İLAN'lı değer | Kaynak |
|---|---|---|
| epi_kanal_kapisi_similarity_threshold | **0.85** (`:331` tek-sim-kaşılaştırma; default :53/:65) | epistemic_agent.py |
| epi_record_yazim_sitesi | **1** (`record_to_future_train` çağrısı `:365`; open-modu **"a"** :181 — append, ezme YOK) | grep |
| epi_record_sema_anahtar_sayisi | **14** (instruction…timestamp; P3 kanıtı) | :349-364 |
| epi_future_train_path_paramli | **True** (`:54` kurucu-arg) | :54 |
| epi_os_makedirs_satiri | **180** (dirname — üst-dizin-yaratımı) | :180 |
| gw_ask_telemetri_anahtar_sayisi | **17** (:210-228 — REVİZYON-1 düzeltmesi) | agent_gateway.py |
| gw_inject_default_koleksiyon | **"kristal_bellek"** (:233 def-param — kanonik arza yazım-yüzeyi) | :233 |
| gw_target_collection_sahete_secici | **True** (:240 ve :340 üç-dallı `kristal_bellek \| else general_memory` — verilen ad YOKSAYILIR; backing-instance collection_name kazanır) | :240/:340 |
| gw_inject_reasoning_koleksiyon | **"muhakeme_bellek"** (:293 sabit + paylaşımlı-client :291 — sunucuda-kurulum riski; İLKELİ) | :291-297 |
| gw_http_endpoint_kumesi | **["/api/status", "/api/backlog", "/api/query", "/api/inject", "/api/check"]** | :401-435 |
| gw_localhost_kurulum | **2** (:123-124 — P4 B6 sessiz-fallback riski birebir) | :123-124 |
| retrain_zorunlu_fail_closed | **3** (vocab :151 · model :247 · save :248 — `_zorunlu` :121) | grep |
| retrain_cmd_flag_kumesi | **["--data","--steps","--batch-size","--lr","--device","--save-path"]** (:255-264 — **--vocab YOK, --load-path YOK**; REVİZYON-1) | AST |
| retrain_cmd_vocab_yok | **True** (kanonik BULGU: tüketici vocab-körü — subprocess default 32.852'de açılır) | AST |
| retrain_cmd_load_path_yok | **True** (model_path soy-ağacı-kaydı yalnız :295) | AST |
| retrain_archive_success_gate | **True** (:288 `_archive_processed_records` yalnız rc==0 dalında; rc≠0'da early-return :279-285) | :279-288 |
| retrain_archive_sifirlama_modu | **"w"** (:313 — kanal-sıfırlama yıkıcılığı; koşum yalnız probe-yolda) | :300-314 |
| retrain_replay_buffer_yolu | **data/pedagogy/high_school_foundation_dataset.jsonl** (digest `59e2f278…`) | :189-190 |
| retrain_replay_samples_default | **25** (:141) | :141 |
| retrain_output_bin_default | **data/train_future_finetune.bin** (:98 — `data/**`-yazım riski; P5 koşumunda probe-yol) | :98 |
| retrain_oversample_factor_default | **20** (:139) | :139 |
| train_py_default_vocab | **data/rebuild/vocab_base_32852.json** (train.py:153 — REVİZYON-1) | train.py |
| train_py_jeton_guard | **True** (:203-207: `max_id >= vocab_size ⇒ RuntimeError` — fail-closed, traceback rc=1) | train.py |
| train_py_vocab_flag_destekli | **True** (:155-156) — run_training İLETMİYOR (yukarıda) | train.py |
| supervisor_retrain_threshold | **5** (:622, :877) | grep |
| merak_router_state_dict_yukleme | **0** (eğitimsiz-projeksiyon — P3 çıpası) | grep |
| GERÇEK kanal ÖNCE | **0 bayt** (`e3b0c442…`) | disk ölçümü |
| envanter ÖNCE: foreign / kristal / probe / muhakeme / simulasyon | **9 / 36 / YOK / YOK / YOK** | sunucu |
| envanter-digest ÖNCE (16-hex) | **`fe7fc7649737ed35`** | P2==P3==P4 SABİT |

## 3. FAZ-B — koşumlu kapı-protokolü (İLAN'lı)

Ortak: kanonik import'lar; seed=42; P3 sorgu-kümesi (random.Random(42)
+ sorted, POZITIF_N=20); model CPU'da (`data/anka_base_v2.pt`,
`d0f415f3…`; P3 `_model_yukle` kalıbı — kafa/sözlük uyum-kapısı +
resize_state_dict + strict=False).

- **B1 kanal-yazarı (HÜKÜM):** (i) **canlı-döngü ulaşılmazlık teyidi:**
  `AgentGateway` kanonik-otomatik-kur (:64-75: tau=2,5,
  similarity_threshold=0,85, future_train_path=gateway'in probe-yolu)
  ile `ask()` → `process_query(force_rag=True)` P3 sorgu-kümesinde (20)
  → `is_high_similarity` **0/20** (P3 Kapı-D tekrar-teyit — kanal
  koşum-içi ÖLÜ; 0,85 > RRF bant-maks 0,75) + ask-telemetri **17-anahtar**
  her yanıtta; (ii) probe-agent (kanonik kurucu
  `similarity_threshold=0.5` — P3 kalıbı) → `future_train_recorded`
  kayıtları **`data/eval/p5_probe_future_train.jsonl`'e** (GERÇEK kanal
  DOKUNULMAZ) → **tetiklenme ≥ 1** (P3-pusula: 2/20) + **şema
  14-anahtar** her satır + **append-davranışı:** aynı kayıt ikinci kez →
  satır +1 (ezme YOK — kanonik `"a"`-modu :181).
- **B2 arz-çıpa (HÜKÜM, koşum ÖNCESİ):** `_envanter_digest` ==
  `fe7fc7649737ed35` + foreign 9 + `kristal_bellek` 36 +
  `p5_probe_bellek` YOK + `muhakeme_bellek` YOK (İLKELİ-beyan teyidi) +
  `simulasyon_bellek` YOK + GERÇEK kanal 0 bayt (`e3b0c442…`).
- **B3 tüketici-okuma (HÜKÜM):** `RetrainPipeline(future_train_path=
  probe, vocab_path=kanonik, output_bin_path=probe, save_path=probe)`
  → `get_pending_count()` == probe-satır-sayısı (B1-sonu; > 0); 
  gerçek-yol pipeline'ında **0**.
- **B4 tüketici-derleme (HÜKÜM):** `compile_backlog_to_bin` → rc-sız
  dönüş + meta.json alanları (backlog_samples == probe-satır-sayısı;
  total_samples == backlog + replay 25; oversample_factor 20;
  block_size 64) + .bin boyut > 0 + uint16 (2-bayt hücre) +
  **geri-tokenize teyidi** (probe-kaydının render_example+encode
  dizisi .bin'de birebir bulunur — aynı kanonik tokenizer-instansı;
  [[indeks-yazan-ile-arayan-normalizasyonu-ayni-olmali]] dersi) +
  **max-jeton-id rapor-kırılımı** (B5 dışlanan-dal teyidi).
- **B5 run_training İLKELİ (HÜKÜM — REVİZYON-1):** `run_training(steps=
  10, device="cpu" — device kurucuda; save_path probe $TMPDIR)`
  → betik-öncesi keşif-ölçümü gereği İLAN'lı tahmin: **train.py
  subprocess rc=0** (default-vocab 32.852; kanal-kayıt max-id < 32.852
  — P3-probe 32195 + replay-25 32831; [sinir] guard GEÇER) → status
  "success" → **`_archive_processed_records` kanonik-yolda koşar:**
  probe-future_train içeriği `data/eval/p5_probe_archive.jsonl`'e
  taşınır + probe-future_train SIFIRLANIR (0 bayt; kanonik `w`-modu
  :313 — yıkıcılık probe-üstünde kanıtlandı) + **GERÇEK kanal 0-bayt
  SABİT** + probe-save .pt oluşur ($TMPDIR — frozen-desene uymaz,
  `--allow-frozen-write` YOK) + **`data/train_future_finetune.bin`
  OLUŞMAZ** (default-yol derleme YAPILMADI). **Dışlanan dal (koşum-
  öncesi beyan):** probe-.bin max-id ≥ 32.852 → train.py :203
  RuntimeError → subprocess rc≠0 → status "error" + arşiv KOŞMAZ
  (success-gate :288) → bu dalda B5 DÜŞER (hüküm DUR). RAPOR bulgusu
  (hüküm-dışı): **run_training `--vocab`/`--load-path` İLETMİYOR** —
  train.py default sözlükle SIFIRDAN model kurar (model_path soy-
  ağacı-kaydı yalnız :295); rc=0 yalnız tail-jeton görünmüyorsa
  mümkün — vocab-hizası ŞANS, guard fail-closed yedek: **KANAL-TÜKETİCİ
  VOCAB-KÖRÜ + SOY-AĞACI-KOPMASI** (kanonik bulgu — onarım YOK).
- **B6 gateway-yazım yüzeyleri (HÜKÜM — yalnız probe):** probe-gateway:
  `memory=VectorMemory("kristal_bellek" — yalnız-okuma arzı)` +
  `general_memory=VectorMemory(collection_name="p5_probe_bellek")`
  (kanonik sahte-seçici :240/:340'ın else-dalı probe-instance'e düşer).
  (i) `inject_knowledge(metin, target_collection="p5_probe_bellek")` →
  otomatik-kurulum + count +1 + payload teyidi (crystal_tags/
  token_ids/injected_by="agent_gateway") + geri-okuma; (ii)
  **SAHTE-SEÇİCİ mutasyon-kanıtı (İLAN'lı):** `target_collection=
  "olmayan_ad_p5_x"` (sunucuda YOK) ile ikinci-inject → count **+1
  probe'ta** (ad yoksayılır — :240 else-dalı kanonik) + dublör-teyidi
  (upsert YOK — P4 bulgusu); (iii) `check_memory(sorgu,
  target_collection="p5_probe_bellek")` pozitif-kontrol top-1
  (skor > 0); (iv) `inject_reasoning_trace` İLKELİ (koşulmaz —
  FAZ-A beyanı).
- **B7 HTTP-teyidi (HÜKÜM):** `create_http_server("127.0.0.1", 0)`
  (OS-assigned ephemeral; sandbox DIŞI) → sıra: GET /api/status
  (status="online" + epistemic_backlog_samples == arşiv-sonrası probe
  **0** — kanonik get_epistemic_backlog probe-yoldan) + GET
  /api/backlog (liste) + POST /api/check (probe-koleksiyon) +
  POST /api/inject (`{"text", "collection":"p5_probe_bellek",
  "metadata"}` payload — default'a YOK) + POST /api/query (son —
  probe-agent'lı gateway; kanal-yazımı probe'a olası) → her
  5-endpoint yanıt-verir + `server_close()`.
- **B8 dokunulmazlık (HÜKÜM, koşum SONU):** envanter-digest SONRA ==
  ÖNCE (`fe7fc764…`) + kristal 36→36 + foreign 9 birebir +
  `p5_probe_bellek` sunucuda YOK (temizlik: `client.delete_collection`)
  + `muhakeme_bellek` hâlâ YOK + `simulasyon_bellek` hâlâ YOK +
  GERÇEK kanal **0→0 bayt** (`e3b0c442…`) +
  `data/train_future_finetune.bin` **OLUŞMADI** (default-yol derleme
  YAPILMADI — kapı-teyidi) + P2/P3/P4-hüküm-dosyaları digest sabit.
- **RAPOR kırılımı (hüküm-dışı):** `inject_reasoning_trace` İLKELİ
  beyanı + risk; run_training adım-düzeyi detayları (train.py stdout
  log-snippet: [sinir]/[meta] satırları) + **VOCAB-KÖRÜ bulgusu**
  (cmd-flag kümesi; B5'te beyanlı) + B1 canlı-döngü telemetri
  kırılımı (needs_retrieval/entropy dağılımı) + probe-arşiv içeriği.

## 4. Hüküm kuralı (koşum ÖNCESİ sabit — betik içinde, elle YOK)

`P5_GECTİ` (rc=0): FAZ-A İLAN'lı birebir (26 satır) **VE** B1 (0/20 +
ask-17 + probe-yazım ≥1 + şema 14 + append) **VE** B2 (digest çıpası +
9/36/YOK/YOK/YOK + kanal-0) **VE** B3 (probe-N/0) **VE** B4 (meta +
uint16 + geri-tokenize) **VE** B5 (rc=0 + arşiv-probe + sıfırlama +
GERÇEK kanal sabit + save-probe .pt) **VE** B6 (probe-yazım +
sahte-seçici mutasyonu + pozitif-kontrol) **VE** B7 (5-endpoint) **VE**
B8 (digest ÖNCE==SONRA + 36 + 9 + probe-YOK + muhakeme/simulasyon-YOK +
kanal-0→0 + default-bin oluşmadı + P2/P3/P4-hüküm-digest sabit) →
aksi her dal → `DUR` (rc=2). **Statik hizalama-ön-kontrolü koşumdan
AYRI adımdır** (AST ile kapı-anahtar-kümeleri; T-0143 dersi) +
**tanımsız-sabit AST-taraması AYRI adım** (T-0144 dersi: py_compile
NameError'ı yakalamaz) + import-teyidi AYRI adım.

## 5. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p5_ogrenme_kanali.py \
  --host 192.168.1.5 --port 6333 \
  --korpus data/realistic_rag/test_natural_150.jsonl \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --base-ckpt data/anka_base_v2.pt \
  --ilan data/eval/mimari_dogrulama_p5_ilan_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p5_sonuc_2026-09-27.md \
  --hukum-json data/eval/mimari_dogrulama_p5_hukum_2026-09-27.json
```

- **Koşum sandbox DIŞI** (Qdrant trafiği + localhost HTTP-soket +
  train.py subprocess — proxy'den geçmez; P1-P4'te ölçüldü). Model
  saf CPU'da; save_path probe ($TMPDIR); **MPS YOK; çift-eğitici YOK**
  (tek train.py subprocess; `lsof` ön-kontrolü); seed=42.
- Çıktı: rapor + hüküm JSON betikten; rc ∈ {0, 2}. Bu İLAN'ın sha256
  RAPORA işlenecek (İLAN ≠ RAPOR).

## 6. DOKUNULMAZ / YAPILMAYANLAR

`src/rag/**` · `src/gateway/**` · `src/llm/**` · `src/compiler/**` ·
`train.py` kanonik yüzeyi IMPORT-only (YAZIM YOK) · **`kristal_bellek`
YALNIZ OKUNUR** (36 nokta) · **9 foreign koleksiyon DOKUNULMAZ** ·
**`muhakeme_bellek`/`simulasyon_bellek` KURULMAZ** (inject_reasoning_
trace İLKELİ; B6 yazımları backing-instance `p5_probe_bellek`'e) ·
yazım+arşiv+delete **YALNIZ `p5_probe_*` yüzeylerinde** · **GERÇEK
kanal `data/future_train_vector.jsonl` 0-bayt çıpa `e3b0c442…`** ·
`data/train_future_finetune.bin` default-yolu OLUŞMAZ · `data/*.pt`
yazım YOK (save_path probe) · `scripts/sanitize_vector_memory.py`
KOŞULMAZ · `data/**` yazımı yalnız `data/eval/` · ESİK_ROUGE 0,3221 /
TAVAN_ROUGE_DECOMP 0,9509 ilgilendirmez (P5 ROUGE ölçmez) · commit
ayrı operatör onayıyla · `git add -A` YASAK · push YOK · elle
sayı/hüküm YOK.