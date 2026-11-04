# P2.4 — τ=2,5 ve 0,33 Eşik Kalibrasyon Ölçümü (2026-11-04)

## İLAN

**Kapsam:** Onaylı ana plan P2.4 — "τ=2,5 ve 0,33 eşik kalibrasyonu". Bu belge YALNIZ ÖLÇER ve ÖNERİR; **hiçbir sabit değiştirilmedi** (eşik dondurma denetim sonrası kontrolörde). Eğitim koşulmadı, git commit yapılmadı, izlenen dosya düzenlenmedi.

**Ölçülen sabitler:** `src/rag/merak.py:16` (τ, merak/entropi tetiği) ve `src/rag/rag_pipeline.py:34` (`RAG_MATCH_THRESHOLD`, RAG benzerlik eşiği).

**Güvenlik kanıtı:** `data/future_train_vector.jsonl` sha256 önce/son aynı (`e3b0c44298fc1c14…`, boş dosya — DEĞİŞMEDİ); future-train yazımı ölçümde scratch'e yönlendirildi (0 kayıt düştü). Qdrant `192.168.1.9:6333` (anka_bellek=46 nokta, simulasyon_bellek) erişimi salt-okunurdu (yalnız `query_points`/`count` GET'leri; koleksiyon yaratma dalına girilmedi — iki koleksiyon da mevcut).

**Ölçüm düzeneği (gerçek canlı hat, simülasyon değil):** `AgentGateway.create_default(model_path=data/anka_taban_pure.pt [sha 5479a2f3…], vocab_path=data/rebuild/vocab_anka_r1_33114.json, router_state_path=data/anka_router_v2.pt, expert_model_paths={gardener: anka_bahcivan.pt, carpenter: anka_marangoz.pt}, device=mps, qdrant_url=http://192.168.1.9:6333)` → `gw.ask(mode="RAG", force_rag=False, max_new_tokens=45)` — canlı `/api/ask` yüzeyiyle birebir. 36 sorgu, ölçüm döngüsü 32,8 sn (MPS). Betik: scratch `p24_olcum.py` (repo-dışı).

---

## 1. Sabitlerin anlambilimi ve çağrı yerleri (GÖREV 1)

### τ = 2,5 (`merak.py:16`, `CuriosityEngine(tau=2.5)`)
- **Neyi tetikler:** Shannon entropisi **H(z) = −Σ p·log p** (nat), tam 33.114'lük sözlük softmax'unda, prompt-zarfının son jetonunun next-token logitlerinden (`merak.py:26-37`). `entropy ≥ τ` ⇒ `needs_retrieval=True` (`merak.py:43`) ⇒ RAG modunda vektörel bellek araması açılır (`epistemic_agent.py:398-399`).
- **Yan-tetikler (τ'den bağımsız OR):** (1) girdide `<UNK>` morf varsa entropiye `+τ+1,0` boost — tetik garantili (`merak.py:67-69`); (2) model ön-tahmini `<UNK>` olasılığı `> 0,05` ise tetik (`merak.py:71-77`).
- **İkinci rolü:** üretim-sonrası entropi `entropy_post ≥ τ` epistemik-başarısızlık A-koşuludur (`epistemic_agent.py:505`) — `similarity_threshold=0,85` ile birlikte `future_train_vector.jsonl` kaydını açar (`:500-534`).
- **Canlı hat çağrı grafiği:**
  - `src/gateway/agent_gateway.py:85, :96` → canlı gateway τ=2,5'i AÇIK sabit verir; `:305-306` entropy_pre/post telemetrisi
  - `src/rag/epistemic_agent.py:94/:107/:119` (varsayılan+atama), `:383` (entropy_pre hesabı), `:505` (post ≥ τ), `:530` (kayıtta `tau` alanı)
  - `chat_prompt.py:378` (τ=2,5), `scripts/train_t0208_router_v2.py:244` (router eğitim çapası), `run_agent_arena.py:165` / `run_goal_pipeline.py:269` (tüketici görselleştirme)
  - Tüketim birimi notu: `pedagogical_supervisor.py:800/:883` yüksek-benzerlik (0,85) kapısını `rag_score` üzerinden kullanır; τ yalnız yukarıdaki iki rolde.

### RAG_MATCH_THRESHOLD = 0,33 (`rag_pipeline.py:34`)
- **Neyi süzer:** geri çağrılan belgenin prompt'a `<BELGE>` koşullaması olarak ENJEKTE edilmesine izin verir: `is_context_usable(match_score >= 0.33)` (`rag_pipeline.py:65-67`). Altında kalırsa model çıplak sorguya yanıt verir (`conditioned=False`).
- **Ölçüm birimi:** normalize RRF skoru **[0,1]** — `RRF_SCORE_MAX=1.0` (`hybrid_search.py:22`); dense (768, Cosine) + sparse (IDF) RRF füzyonu, kök-eşleşme cezası filtreli (`vector_memory.py:368-396 → hybrid_search.filter_and_rank_candidates`).
- **Çağrı grafiği:** `rag_pipeline.py:164/:189` (RagPipeline.answer kapısı), `:236` (build_from_disk); `src/rag/__init__.py` (dışa aktarım); **canlı kullanım: `epistemic_agent.py:409`** (conditioned kapısı), `:547-550` (T-0181 telemetri).
- **Tarihçe:** T-0179 (2026-09-29, operatör onaylı): 0,40 → 0,33; kendi-belge dağılımı min 0,3556 / medyan 0,4444 ile 20/20 yakalanmış, OOV'den ~4,4× mesafe.

---

## 2. Sorgu seti (GÖREV 2)

`data/eval/p24_sorgu_seti.json` — **36 sorgu**, dengeli: 10 alan (marangozluk/bahçıvanlık — ceket alanları), 10 nötr ansiklopedik, 10 belirsiz/eksik-bilgili (merak adayı), 6 kenar (tek harf, tek kelime, anlamsız token, 82-jeton uzun sorgu, boş sorgu). Hepsi Türkçe, canlı kullanım profilinde.

---

## 3. Dağılım ölçümü (GÖREV 3) — `data/eval/p24_esik_dagilim_2026-11-04.json`

### 3a. Kategori × tetiklenme ve skor özeti (τ=2,5, eşik 0,33)

| kategori | n | τ tetiklenme | ham H_pre (medyan / [min–maks]) | en-iyi skor (medyan) | 0,33 geçen | conditioned | epistemic_failure |
|---|---|---|---|---|---|---|---|
| alan | 10 | **10/10 (%100)** | 5,22 / [3,34–6,36] | 0,4375 | **8/10** | 8 | 0 |
| nötr | 10 | **10/10 (%100)** | 5,67 / [4,49–6,38] | 0,166 | 5/10 | 5 | 0 |
| belirsiz | 10 | **10/10 (%100)** | 5,41 / [5,00–5,88] | 0,343 | 5/10 | 5 | 0 |
| kenar | 6 | **6/6 (%100)** | 6,24 / [3,28–7,00] | 0,000 | 1/6 | 1 | 0 |
| **TOPLAM** | **36** | **36/36 (%100)** | — | — | 19/36 | 19 | **0** |

**Başlıca ölçülmüş olgular:**
1. **τ=2,5'te tetiklenme %100** — modelin prompt-entropi tabanı ölçülen en düşük değerle **3,28 nat** (#35, 82-jeton alan sorgusu) ve 3,34 (#4); taban τ'nin üzerinde. Eşik ayrım üretmiyor, "her zaman ara" anahtarı.
2. 5 sorguda girdide `<UNK>` (id 3, 15, 30, 34, 35) → boost **tam +3,5** doğrulandı (raporlanan − ham = 3,5000). #30'da ayrıca `unk_p=0,0586 > 0,05` prediktif tetiği.
3. **Yanlış-koşullanma:** 0,33 kapısından 10 aday nötr/belirsiz sorgu geçti ve alan-dışı belgeyle koşullandı (Fomalhaut, Pulsar PSR, Diyatomit, Alacahöyük Andezit, Victoria Amazonica, Piroksenit, Utricularia — tümü anka_bellek'ten; koleksiyonda sentetik alan-dışı belgeler var).
4. **Koşullanma entropiyi düşürüyor:** skor 1,0'la koşullanan #21 "Nasıl yapılır?" entropy_post **0,003**, #33 "zıvana" **0,297** (pre: 5,76 / 6,47). `entropy_post ≥ 2,5` yine de 19/36 sorguda (koşullanan 9 / çıplak 10) — post-τ "hâlâ emin değil" sinyali çalışıyor; ancak 0,85 yüksek-benzerlikle birlikte hiç gerçekleşmedi → **bu sette 0 epistemik-başarısızlık kaydı** ("Model Anlamaz" yolu kapalı kaldı).
5. **FP sınır vakası:** #26 "Onu bana biraz anlat." skoru matematiksel 0,33 ama kayan-nokta `0.32999999999999996 < 0.33` ⇒ kapıdan ELENDİ (cond=False).
6. **simulasyon_bellek (fallback genel bellek) uzak sunucuda BOŞ (0 nokta)** — `search_memory`'nin anka_bellek-miss fallback dalı (`epistemic_agent.py:244-248`) bu sunucuda hiç sonuç üretemez; tüm ölçülen skorlar anka_bellek'ten geldi.

### 3b. Alternatif τ masası (ajanın OR-yapısına sadık: ham H ≥ τ′ veya girdide UNK veya unk_p > 0,05)

| τ′ | alan | nötr | belirsiz | kenar | TOPLAM |
|---|---|---|---|---|---|
| 2,5 (mevcut) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| 3,5 / 4,0 | 0,90 | 1,00 | 1,00 | 1,00 | 0,97 |
| 4,5 / 5,0 | 0,60 | 0,90 | 1,00 | 1,00 | 0,86 |
| 5,5 | 0,30 | 0,70 | **0,40** | 1,00 | 0,56 |
| 6,0 | 0,20 | 0,30 | 0,10 | 1,00 | 0,33 |
| 6,5 / 7,0 | 0,10 | 0,10 | 0,10 | 0,50 | 0,17 |

### 3c. Alternatif RAG eşiği masası (bağımsız en-iyi-skor sondasıyla geçme oranı)

| eşik | alan | nötr | belirsiz | kenar |
|---|---|---|---|---|
| 0,25 / 0,30 | 0,80 | 0,50 | 0,70 / 0,60 | 0,17 |
| **0,33 (mevcut)** | **0,80** | **0,50** | **0,50** | **0,17** |
| 0,35 | 0,80 | 0,40 | 0,50 | 0,17 |
| 0,40 | 0,50 | 0,20 | 0,40 | 0,17 |
| 0,50 | 0,40 | 0,20 | 0,10 | 0,17 |
| 0,60 / 0,70 | 0,20 | 0,00 | 0,10 | 0,17 |

### 3d. Alan sorgularında en-iyi skor dağılımı (0,33 süzgeci)

`[0,000 (✗ #3 Zımpara grit) — 0,110 (✗ #1 Meşe kereste) — 0,356 (#9 Gübreleme) — 0,396 — 0,400 — 0,475 — 0,500 — 0,533 — 0,700 — 0,833 (#2 Kırlangıç)]`

- 0,33 **8/10 alan sorgusunu geçiyor** (%80). Elelenenler: #3 (skor 0,000 — hiç kök-eşleşmesi yok; **hiçbir eşik kurtaramaz**, geri-çağırma/kapsam boşluğu), #1 (0,110 — yalnız ≤0,11'lik eşikle kurtulur).
- Kenar #35 (82-jeton alan sorgusu) 0,226 ile KALDI — RRF skoru sorgu uzunluğuyla seyrelıyor (sistematik eğilim, eşik kalemi değil).

---

## 4. HÜKÜM ve kalibrasyon ÖNERİSİ (GÖREV 4 — yalnız öneri, uygulama YOK)

### τ = 2,5 → **DEĞİŞİKLİK GEREKMEZ** (ölçü hiçbir alternatifi desteklemiyor)
- Mevcut τ ile tetiklenme %100; hedef "normal sorguda sessizlik" karşılanmıyor, ama **ölçülen alternatif masası hiçbir τ′ değerinde ayrımı iyileştirmiyor**: kategorilerin ham-entropi dağılımları neredeyse tam örtüşüyor ve **sıralama ters** (belirsiz medyan 5,41 < nötr medyan 5,67). τ′=5,5'te belirsiz tetiklenmesi (%40) nötrün (%70) altına düşüyor — kalibrasyon hedefini tam tersine çevirir.
- **Kök neden (ölçülmüş):** pre-entropi prompt-zarfının son jetonu üzerinden hesaplanıyor ve sorgunun bilgi-eksikliğini değil, taban modelin genel belirsizliğini yansıtıyor (taban model pencere-CE ≈ 3,8 nat / PPL 45,6 — entropi tabanı ~3,3+). Tek-global-τ bu taban modelde ayrıştırıcı olamaz.
- **Öneri:** τ **2,5 olarak donsun**; eşik artık fiilen "her zaman ara" anahtarıdır ve 0,33 tek etkin süzgeçtir — bu bilinçli kabul edilsin (belgeye yazılsın). "Gerçek belirsizlikte tetik, normalde sessizlik" istenirse çözüm τ DEĞERİ değil **sinyal değişikliğidir** (ör. koşullanma-sonrası entropi düşüşü ΔH = H_pre − H_post veya post-entropi tabanlı ikinci aşama; ölçülen post-davranış — koşullanmada 0,003–0,3'a çökme — bu sinyalin zengin olduğunu gösteriyor). Bu bir tasarım kalemidir, P2.4 sabit-dondurma kapsamının dışında — kontrolör kararı.
- τ'nin ikinci rolü (post ≥ τ, epistemik-başarısızlık A-koşulu) ölçüde çalışır durumda (19/36); kayıt yolu bu sette açılmadı çünkü bağlayıcı kısıt 0,85 benzerlik eşiğiydi (yalnız 2 sorgu ≥0,85, ikisinin de post'u <2,5).

### RAG_MATCH_THRESHOLD = 0,33 → **DEĞİŞİKLİK GEREKMEZ** (ölçü bir değişikliği desteklemiyor)
- **Aşağı çekmek:** 0,33→0,10 yalnız #1'i (0,110) kurtarır ama T-0179'un OOV mesafe gerekçesini (4,4×) yok eder ve gürültü-bant eşleşmelerini içeri alır; #3 (0,000) hiçbir eşikte kurtulmaz.
- **Yukarı çekmek:** yanlış-koşullanmayı kısmen keser ama alan kapsamını daha hızlı düşürür — 0,40'ta alan %80→%50, 0,60'ta %20. Ceket alanları için kabul edilemez.
- **Ölçülen asıl sorun eşik değil, skor fonksiyonunun ayrım gücü:** ilgili ve ilgisiz eşleşmelerin RRF skorları aynı bantta yatıyor (alan [0,356–0,833] ile alan-dışı [0,332–0,533] örtüşüyor; anka_bellek'te sentetik alan-dışı belgeler mevcut). Bu, koleksiyon içeriği/skorlama kalemidir, sabit kalemi değil.
- **Öneri:** 0,33 **korumada**. Ayrıca (a) T-0179'un kendi-belge tabanı (min 0,3556) doğal-parafraz sorgularda tutmuyor (#1: 0,11, #3: 0,0) — bu, o ölçümün belge-benzer prob karakterini taşır, yeni ölçümle çelişmez, ama gelecekteki eşik kararlarında "kendi-belge tabanı" varsayımına dayanılmaması gerektiğini gösterir; (b) FP sınırı: `is_context_usable` kayan-nokta karşılaştırması matematiksel 0,33'ü elebiliyor (#26) — sabit değişikliği olmayan sağlamlık notu (karşılaştırmada ε payı veya yuvarlama düşünülebilir; kontrolör takdiri).

---

## 5. Bilinen bulgu (P2.3'ten taşınır, düzeltme AYRI KALEM)

**supervisor --auto-retrain varsayılan RetrainPipeline() ile KIRIK:** `pedagogical_supervisor.py:644` `retrain_pipeline or RetrainPipeline()` ile yapılandırılmamış varsayılan alır; backlog ≥ 5 kapısında (`:635` eşik, `:891` kapı) çağrılan `run_training` (`src/gateway/retrain_pipeline.py:231`) T-0087 gereği `model_path`/`save_path` AÇIK istemeyince `RuntimeError` fırlatır (kontroller `compile_backlog_to_bin`'DEN ÖNCE çalıştığı için hiçbir yazım olmaz — sessiz hasar YOK, ama otomatik retrain hiç koşamaz). **Canlı kullanım için konfigüre pipeline enjekte edilmelidir** — bu görevin kapsamı dışında, ayrı düzeltme kalemi olarak kayıtlıdır.

---

## 6. Kanıt dosyaları

| dosya | içerik |
|---|---|
| `data/eval/p24_sorgu_seti.json` | 36 sorguluk canlı-varı seti (kalıcı kanıt) |
| `data/eval/p24_esik_dagilim_2026-11-04.json` | tüm dağılım verisi: 36 sorguluk ham kayıtlar (ham/raporlanan entropi, unk_p, skor, conditioned, uzman, süre) + kategori özetleri + alternatif τ/eşik masaları + yanlış-koşullanma envanteri |
| `data/eval/p24_esik_kalibrasyon_2026-11-04.md` | bu rapor |
| scratch: `p24_olcum.py`, `p24_topla.py`, `p24_olcum_kayitlar.ndjson` | ölçüm betiği, toplayıcı, ham kayıtlar (repo-dışı) |

**Değiştirilen izlenen dosya: YOK.** Bu rapor ve iki JSON/veri dosyası yeni, izlenmeyen kanıt dosyalarıdır; kontrolör eşikleri bu ölçüme dayanarak dondururken değişiklikleri ayrı uygular.