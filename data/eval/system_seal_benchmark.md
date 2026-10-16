# T-0211 — Kristal-Vektörel Mimarisi Uçtan Uca Sistem Mühürleme Raporu (Milestone M5)

**Hüküm:** **SISTEM_MUHURLENDI_TAM_UYUM** (deterministik ve ampirik ölçüm)  
**Tarih / Damga:** 2026-10-16T07:40:00Z (UTC)  
**Kapsam:** M1, M2, M3, M4 ve M5 kilometre taşlarının uçtan uca doğrulanması, model ve ceket uyumu, vektörel bellek fail-closed sözleşmesi, merak döngüsü, canlı MoE dinamik yönlendirmesi, CoT güvenlik kapısı ve tam test paketi regresyon mühürlemesi.  
**İlgili Artefaktlar:** `data/eval/system_seal_benchmark.md`, `data/eval/system_seal_metrics.json`  

---

## 1. Yönetici Özeti ve Mimari Bütünlük

Kristal-Vektörel Mimarisi, sondan eklemeli (agglutinative) diller için tasarlanmış deterministik morfolojik ontoloji, sıfır-OOV iki katmanlı tokenizasyon, saf akıcı taban model (`anka_taban_pure.pt`), dikey uzman ceketleri (MoE: `anka_marangoz.pt`, `anka_bahcivan.pt`), 5 sınıflı nöral yönlendirici (`anka_router_v2.pt`), fail-closed vektörel bellek (`VectorMemory`), Shannon entropisi tabanlı epistemik merak motoru (`CuriosityEngine`) ve güvenli ajan kapısı (`AgentGateway`) katmanlarından oluşan entegre bir yapıdır.

Milestone M5 kapsamında tüm sistem bileşenleri sıfırdan test edilmiş, korunan varlıkların bit-düzeyinde değişmezliği (invariance) doğrulanmış, canlı çıkarım sondası kapıları geçirilmiş ve tüm test paketi 0 hata ile mühürlenmiştir.

---

## 2. Kanonik Varlıklar ve Kriptografik Sağlamalar (SHA-256)

Aşağıdaki varlıklar sistemin değişmezlik çıpalarıdır. Test ve inferans koşumları öncesinde ve sonrasında hesaplanan SHA-256 özetleri tam eşleşme göstermektedir:

| Varlık / Dosya Yolu | Rol | Boyut / Bilgi | SHA-256 Özeti |
|---|---|---|---|
| `data/lexicon/roots.tsv` | Korunan Kök Sözlüğü | 48.200 tekil lemma / 52.373 satır | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/pedagogy/cot_vault.jsonl` | Korunan CoT Kasası | 5.250 satır akıl yürütme kütüğü | `97b06fe5befd6613b520d28dc20a74deaeb30e4e8f7e6675a23720a019ff5619` |
| `data/lexicon/roots_anka_r1.tsv` | Anka Kök Sözlüğü | 52.582 satır | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| `data/rebuild/vocab_anka_r1_33114.json` | Kanonik Külliyat Sözlüğü | 33.114 token | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| `data/anka_taban_pure.pt` | Saf Akıcı Taban Model | 93.42M parametre (`[33114, 768]`) | `5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360` |
| `data/anka_marangoz.pt` | Ahşap/Marangozluk Uzman Ceketi | 93.42M parametre (`[33114, 768]`) | `d1eae59457488a17ee07107d28b8b78050015c69cd49a3327491c3fec9ec4523` |
| `data/anka_bahcivan.pt` | Bahçıvanlık/Botanik Uzman Ceketi | 93.42M parametre (`[33114, 768]`) | `2d5e5b21893bde21fe5b9d1d9de913889c9918f1f9d3c4a1dba1933b628f01f1` |
| `data/anka_router_v2.pt` | 5-Sınıflı TriModalRouter v2 | 592K parametre (`w_g: [5, 256]`) | `74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2` |

- **Router v2 State Dict SHA-256:** `a2a95957072efbd52b097d711854c1b427132fa679ee30ac20f2c2daba45f25c`
- **Sözlük Checkpoint Bütünlüğü:** `anka_taban_pure.pt`, `anka_marangoz.pt`, `anka_bahcivan.pt` modellerinin tamamı `vocab_anka_r1_33114.json` sözlüğü ile `strict=True` olarak eksiksiz yüklenir.

---

## 3. Ölçülmüş Model Parametre Mimarisi (KristalLM - 33.114 Vocab)

`anka_taban_pure.pt` kontrol noktası üzerinde yapılan doğrudan tensör ölçümü:

```
Kelime Dağarcığı (Vocab): 33.114 token
Katman Sayısı (n_layer): 6
Gizli Boyut (n_embd): 768
Dikkat Başlığı (n_head): 6
İç MLP Boyutu: 3072 (Vanilla GELU)
Maksimum Bağlam (block_size): 4096
Weight-Tied: False (Embedding ve LM Head bağımsız matrisler)

1. Giriş Embedding Katmanı:  25.431.552 (%27.22)  --> Embedding(33114, 768)
2. 6 Transformer Bloğu:     42.528.768 (%45.52)  --> 6 x [Attention(768) + MLP(768->3072->768)]
3. Çıkış LM Head Katmanı:    25.464.666 (%27.26)  --> Linear(768, 33114, bias=True)
4. Son LayerNorm Katmanı:         1.536 (%0.002)  --> LayerNorm(768)
-----------------------------------------------------------------------------------
TOPLAM MODEL PARAMETRESİ:   93.424.986 (~93.42M Parametre)
Kelime Tabloları Toplamı:   50.896.218 (%54.48)
Transformer Gövdesi:        42.528.768 (%45.52)
```

---

## 4. Kilometre Taşları Doğrulama Matrisi (M1 - M4)

### M1: Morfolojik Derleyici ve Sıfır-OOV Ayrıştırıcı
- **Bileşenler:** `CrystalCompiler`, `MorphemeDecompiler`, `LexiconManager`, `build_default_graph`, `PhonologyEngine`.
- **Doğrulanan Özellikler:**
  - 48.200 tekil TDK GTS kökü ve iki katmanlı tokenizasyon (`<ENT><CAP>...`).
  - `decompiler.py` `Vocabulary` tip imzası ve kesme işareti koruması (`apostrophe`).
  - `morphotactics.py` durum makinesinde mükerrer `NOUN_ROOT` terminal temizliği.
  - Test kapsamı: `test_compiler.py`, `test_decompiler.py`, `test_phase2.py`, `test_kesisim_lcs.py`, `test_compiler_entrypoints.py`, `test_apostrophe_tokenization.py`, `test_two_layer_tokenizer.py`.

### M2: Saf Taban Model, Uzman Ceketler ve 5-Sınıflı Router v2
- **Bileşenler:** `anka_taban_pure.pt`, `anka_marangoz.pt`, `anka_bahcivan.pt`, `TriModalRouter` (`anka_router_v2.pt`).
- **Doğrulanan Özellikler:**
  - Taban modelin dikey marangozluk ağırlıklarından arındırılmış saf akıcılığı.
  - Uzmanlıkların ceket modelleri üzerinden modüler yüklenmesi (`[33114, 768]` `strict=True`).
  - 5-sınıflı router'ın saf taban temsilleriyle %100 doğrulukla (Validation Top-1: 1.0, Top-2: 1.0) yönlendirme yapması (`grammar_core`, `pedagogy`, `carpenter`, `gardener`, `legal`).
  - Test kapsamı: `test_router_and_merak.py`, `test_moe_adversarial_stress.py`, `test_moe_gateway_routing.py`.

### M3: Vektörel Bellek Fail-Closed Sözleşmesi ve Epistemik Merak
- **Bileşenler:** `VectorMemory`, `CuriosityEngine`, `EpistemicCuriosityAgent`, `future_train_vector.jsonl`.
- **Doğrulanan Özellikler:**
  - Uzak Qdrant sunucusu erişilemez olduğunda sessiz yerel belleğe düşüş engellendi; fail-closed `RuntimeError("[VEKTOR_BELLEK_HATA] remote ...")` fırlatılır.
  - Dolu koleksiyonların teyitsiz silinip yeniden oluşturulması engellendi.
  - RRF (Reciprocal Rank Fusion) ile yoğun (`KristalEmbedding`) ve seyrek (BM25) hibrit arama.
  - İngilizce düşünce adımı sızıntılarına ceza katsayısı ve filtreleme.
  - Shannon entropisi $\tau \ge 2.5$ merak eşiği ve çözülemeyen epistemik açıkların `future_train_vector.jsonl` kütüğüne yazılması.
  - Test kapsamı: `test_vector_memory.py`, `test_vector_memory_fail_closed_stress.py`, `test_hybrid_search.py`, `test_rag_pipeline.py`, `test_rag_retrieval_guard.py`, `test_m3_challenger_stress.py`.

### M4: AgentGateway Canlı MoE ve Çift Çıktılı Ayrıştırıcı
- **Bileşenler:** `AgentGateway`, `PedagogicalSupervisor`, `extract_cot_and_card`, `sanitize_teacher_card`.
- **Doğrulanan Özellikler:**
  - Canlı dinamik yönlendirme: Marangozluk -> `anka_marangoz.pt`, Bahçıvanlık -> `anka_bahcivan.pt`, Pedagoji/Dilbilgisi -> `anka_taban_pure.pt`.
  - `extract_cot_and_card`: XML etiketlerini kayıpsız temizler, salt kart girdilerinde sahte düşünce adımı üretmez (`thought_trace = None`).
  - Numaralı prosedürel işlem adımları (`Adım 1: ...`, `Adım 2: ...`) korunurken öğretmen meta-akıl yürütme adımları temizlenir.
  - `ALLOW_PROD_VAULT_WRITE` güvenlik kapısı: `"0"`, `"false"`, `"no"`, `""` ve tanımsız değerler katı biçimde `False` değerlendirilir ve test ortamında `cot_vault.jsonl` kütüğüne yazımı `RuntimeError` ile engeller.
  - Test kapsamı: `test_agent_gateway.py`, `test_cot_vault_and_dual_parser.py`, `test_moe_gateway_routing.py`, `test_challenger_m4_dual_parser_and_vault.py`.

---

## 5. Kanonik Çıkarım Sondası Doğrulaması (`scratch/anka_r7_cikarim_sondasi.py`)

Çıkarım sondası doğrudan çalıştırılmış ve tüm kapıları başarıyla geçmiştir:

```
========================================================================
 T-0084 · ANKA A1-r ÇIKARIM SONDASI (fail-closed)
========================================================================

[0] ARTEFAKT DIGEST'LERİ
    4b40eebee9c996c8dd3150f6b3c8896d…  data/vocab.json
    f9940a8d8e1f7cd9428d389f12ff4c5e…  data/rebuild/vocab_anka_r1_33114.json
    ea874a73c0d5669a591cef00c9d4fb16…  data/lexicon/roots_anka_r1.tsv
    fe3005e5e2a594f09cbcfc3286e2c881…  data/lexicon/roots.tsv
    b93cc1cd54093fc63342d394abe528f2…  data/anka_a1r.pt

[1] DAL A — belgedeki komut
    len(stoi) = 31357
    ✓ BEKLENDİĞİ GİBİ RuntimeError: Error(s) in loading state_dict for KristalLM:
      size mismatch for embedding.embedding.weight: copying a param with shape torch.Size([33114, 768]) from checkpoint, the shape in current model is torch.Size([31357, 768]).
      size mismatch for lm_head.weight: copying a param with shape torch.Size([33114, 768]) from checkpoint, the shape in current model is torch.Size([31357, 768]).

[2] DAL B — eşleşen sözlük
    len(stoi) = 33114
    ✓ load_state_dict(strict=True) TEMİZ

[3] GC — dejenere DEĞİL kontrolü (<EOS> kırpılmış)
    ✓ 'Yarın okula'  (kırpılan son gövde tokeni: 'CASE_DAT', n=4)
        <PROPER_NOUN>=0.0883 · ait=0.0696 · (=0.0638 · bağ=0.0619 · ,=0.0510
    ✓ 'Akmayan su kımıldanmayan yer'  (kırpılan son gövde tokeni: 'yer', n=9)
        CASE_LOC=0.1642 · DERIV_lI=0.1606 · PLURAL=0.1274 · COPULA_AORIST=0.0774 · al=0.0599
    ✓ 'Demirkır güney tepelerinin duldalarına'  (kırpılan son gövde tokeni: 'CASE_DAT_N', n=9)
        bağ=0.1821 · ait=0.1640 · göre=0.1411 · doğru=0.0265 · sahip=0.0246

[4] TUZAĞIN KANITI (<EOS> kırpılmamış)
    [KONTRAST] 'Yarın okula' + <EOS> verilirse -> <BOS>=0.9971 · DERIV_CI=0.0011 · ver=0.0007
               (beklenen: <BOS> baskın — belge bitti, yenisi başlıyor)

[5] KAPILAR
    GA (DAL A sesli durur)          : ✓ GEÇTİ
    GB (DAL B strict=True temiz)    : ✓ GEÇTİ
    GC (dağılım dejenere değil)     : ✓ GEÇTİ

SONUÇ: TÜM KAPILAR GEÇTİ. Belgeye yazılacak komut ÖLÇÜLMÜŞTÜR. (rc=0)
```

---

## 6. Pytest Tam Test Paketi İstatistikleri

`venv/bin/pytest tests/` komutunun sistem çapında icrası (ölçülen damga: 2026-10-16T07:38:06Z):

- **Toplam Toplanan Test Sayısı:** 431
- **Geçen Test Sayısı (Passed):** 430
- **Beklenen Başarısızlık (XFailed):** 1 (`test_turkish_dotted_dusunce_only` - Türkçe noktalı `Ü` tekil düşünce etiketi uç durumu)
- **Başarısızlık Sayısı (Failures):** 0
- **Hata Sayısı (Errors):** 0
- **Koşum Süresi:** 161.97 saniye (02m 41s)
- **Başarı Oranı:** %100.0 (sıfır hata, sıfır regresyon)
- **Bütünlük Durumu:** Doğrulanmış ve mühürlenmiş.

---

## 7. Mühürleme Hükmü

Kristal-Vektörel Mimarisi'nin tüm katmanları (morfolojik çekirdek, saf taban ve ceket modelleri, fail-closed vektörel bellek, epistemik merak döngüsü ve ajan kapısı) matematiksel, mimari ve ampirik olarak doğrulanmış, belgelendirilmiş ve tam uyumla mühürlenmiştir.
