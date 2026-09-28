# İLAN — T-0154: TriModalRouter eğitilmiş-ağırlık (3-sınıf uzman-sınıflandırma)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-28T06:52:14Z
(revizyon-1: kaynak-digest'ler kısa-önek yerine TAM SHA-256 yazıldı —
hash-iddiaları-tam-digest dersi; koşum-öncesi, betik-yokken yasal)
(revizyon-2: pedagogy satır-toplamı aritmetik düzeltme — 688+220+125+
19.342+6.507 = **26.882** idi; "25.882" yanlış toplanmıştı; bileşen-değerler
değişmedi, kapılara etkisi YOK — K5 kırılım 500/100×3'tür; koşum-öncesi
ön-ölçümle teyit edildi: satır 26.882, benzersiz 15.073)
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Hüküm BETİKTEN
(`scripts/dogrulama_t0154_router_egitim.py`); elle sayı/hüküm YOK. rc ∈ {0, 2}.

## Meşruiyet

Operatör kararları (28 Eyl 2026, AskUserQuestion): **kapsam SADECE ROUTER**
(q_proj fresh-init kalır) + **legal sınıfı 3-sınıf eğitimi** (kaynak yok;
legal fresh-init kalır). P3 §7.6 eğitimsiz-projeksiyon beyanının kapanışı.
Bus T-0154; kiralamalar `data/` (ÜST-DİZİN — `data/anka_router.pt` donmuş
`data/*.pt` deseni), `scripts/`, `.agent-bus/notes/`; `data/eval/` T-0152
kiralamasıyla örtüşük (owner=claude). **Bu tur kod-değişikliği YOK**
(entegrasyon — gateway state_dict yükleme — AYRI onarım maddesi). Canlı
sunucu 192.168.1.9:6333'a HİÇ erişim yok (anka_bellek DOKUNULMAZ).

## Sabitler (koşum-öncesi; değişmez)

- **Embedding-yüzeyi (mühürlü taban):** `data/anka_base_v2.pt`
  (`d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`) +
  `data/rebuild/vocab_anka_r1_33114.json`
  (`f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`) +
  `data/lexicon/roots.tsv`
  (`fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`)
  — base_v2.meta.json değerleri; betik koşumda yeniden ölçer ve birebir
  kontrol eder.
- **Eğitilen modül:** `TriModalRouter(prompt_dim=768, merak_dim=768,
  rag_dim=768, router_dim=256, num_experts=4, top_k=2, expert_names=
  ["grammar_core","pedagogy","carpenter","legal"])` — kanonik sınıf
  (src/llm/router.py; `epistemic_agent.py:80-88` koşum-zamanı konfigürasyonu).
- **Girdi-konfigürasyonu (koşum-zamanına sadık):** prompt_vec = mean-pool
  embedding (`calculate_prompt_embedding` birebir: `model.embedding(x).mean(1)`)
  + **q_merak = fresh-init seed-42 CuriosityEngine** (768; eval; taban-model
  `return_hidden_states` son-gizli-durum üzerinden; `evaluate_entropy_and_merak`
  birebir) + **rag_vec=None** (koşum-zamanı opsiyonel-dal; eğitim kapsamı
  sorgu-anı, RAG-öncesi yönlendirme).
- **Veri (koşum-öncesi ölçülen satır-sayıları):**
  - `grammar_core` ← `data/pedagogy/lexical_semantics_dataset.jsonl` (25.000)
  - `pedagogy` ← `data/pedagogy_canonical/` 5 dosya: high_school 688 +
    literature 220 + middle_school 125 + parenting 19.342 + turk_tarihi
    6.507 = **26.882** (revizyon-2: toplam aritmetik düzeltme)
  - `carpenter` ← `data/pedagogy_canonical/carpenter_canonical.jsonl` (5.400)
  - **legal ← 0 örnek** (kaynak yok; İLAN'lı karar)
  - Metin alanı: `input` (boşsa `instruction`); SEED-42; sınıf-başı **500
  train + 100 val** ⇒ 1.500 train / 300 val; val kesişimi train ile YOK.
- **Eğitim:** kayıp = CE(w_g(e_route) 4-logit, 3-hedef sınıf); AdamW
  lr=1e-3, batch=32, epoch=3 (≈141 adım); **eğitim cihazı CPU** (router
  ~800K param — determinizm-garantisi: iki koşum state_dict bit-özdeş
  olmalı); **özellik-çıkarımı cihazı ölçülür** (MPS hedef; yoksa DUR —
  sessiz-cihaz-fallback YOK; sandbox-MPS dersi).
- **Çıktı (donmuş-desen yazım; operatör-onayı bu görev):**
  `data/anka_router.pt` — `torch.save(state_dict)`; anahtar-kümesi kanonik
  TriModalRouter `state_dict()` ile birebir (şema-koruma).

## Kapılar (koşum öncesi sabit)

- **K1 SIFIR-KILIÇ:** çıktı dosyası `data/anka_router.pt` koşum-ÖNCESİ YOK
  (üzerine yazım yasak); kaynak dosyaların SHA-256'leri İLAN-değerleriyle
  birebir (base_v2 `d0f415f3…`, vocab `f9940a8d…`, lexicon `fe3005e5…`).
- **K2 BASELINE:** fresh-init seed-42 router aynı-val kümesinde top-1
  doğruluk ölçülür (4-logit argmax — legal dahil; kayıt-değer, ayırt-edici
  taban). Ek olarak **rastgele-taban ölçümü**: 3-sınıf tekdüze ~0,333 +
  legal-dışı kısıtı ~0,25 — raporda karşılaştırma.
- **K3 EĞİTİM-KAPISI (ayırt-edici):** eğitilmiş val top-1 doğruluk ≥
  **0,75** VE ≥ baseline + **0,15** marj; train CE son-epoch < ilk-epoch.
- **K4 DETERMİNİZM:** aynı-seed ikinci eğitim koşumu state_dict SHA-256
  **bit-özdeş** (CPU-determinizm); sapma → DUR.
- **K5 VERİ-ENVANTER:** sınıf-kırılım birebir 500/100 × 3; val∩train metin
  kesişimi 0; legal etiketli örnek 0.
- **K6 ŞEMA + ÇIKTI:** state_dict anahtar-kümesi == kanonik; `data/anka_router.pt`
  yazılır + SHA-256 hüküm-JSON'da; **canlı-dokunulmazlık: tur boyunca
  sunucuya 0 istek** (kod-yolu: VectorMemory kurulmaz).
- **6/6 → T0154_ROUTER_GECTI rc=0**; aksi her dal → **DUR rc=2** (çıktı
  yazılmaz; kısmi eğitim artefaktı `data/eval/` altına kaydedilir).

## pytest (ayrı koşum — hüküm-dışı)

`tests/test_router_and_merak.py` boyut/davranış testleri — kendi instance'ları;
kaynak-kod değişmediğinden (bu tur kod-değişikliği YOK) düşmesi BEKLENMEZ;
yeniden ölçüm raporda.

## DOKUNULMAZLAR

`data/**` (bu tur yazımları: `data/anka_router.pt` + `data/eval/` çıktıları
yalnız); `src/**` kod-değişikliği YOK; canlı sunucu; P2-P5 çıpa-betikleri;
`src/compiler/**`, `src/llm/tokenizer.py` (yalnız OKUMA — tokenizer kurulumu
P2 kanonik-yöntemi).

## Beyan: damga-yöntemi

Damga betik koşulmadan `date -u` işlendi (mtime-çıpası); hüküm-JSON:
ilan_sha + damga + kapılar. Hüküm BETİKTEN; elle sayı YOK.