# P2.4b — ΔH Kalibrasyon Seti Genişletme: 36 → 107 + Eşik Taramaları + Bootstrap CI (2026-11-04)

## İLAN

**Kapsam (kalem 4):** P2.4'ün 36 sorguluk zemin ölçümünde ΔH/post_h sinyalleri için önerilen eşikler n=36 ile ince kalması nedeniyle setin 100+ sorguya büyütülmesi, gerçek canlı hat üzerinde yeniden ölçümü, J tablolarının yeniden kurulması ve bootstrap güven aralıklı stabil öneri.

**Sert kurallar (hepsi tutuldu):**
- **git commit YOK**; **eğitim YOK**; **Qdrant SALT-OKUNUR** (yalnız `query_points`/koleksiyon `GET` sayımları; koleksiyon yaratma/yazma dalına girilmedi);
- **izlenen dosyalara DOKUNULMADI** — bu görev yalnız YENİ `data/eval/` kanıt dosyaları üretir; ölçüm betikleri scratch'tedir (`p24b_olcum.py`, `p24b_topla.py`, repo-dışı);
- **eşik DON_DURULMADI** — bu belge yalnız ölçer ve önerir; dondurma denetleyici kararıdır (kalem dışı).

**Güvenlik kanıtı:** `data/future_train_vector.jsonl` sha256 önce/son aynı (`e3b0c44298fc1c14…`, boş dosya — DEĞİŞMEDİ); future-train yazımı koşum-öncesi scratch'e yönlendirildi (`p24b_future_train_yonlendirme.jsonl`, **1 kayıt** düştü — bkz. §2). Model `data/anka_taban_pure.pt` sha `5479a2f35a88975c…` (P2.4 ile aynı; değişmedi).

---

## 1. Sorgu seti (GÖREV 1)

`data/eval/p24b_sorgu_seti_100plus.json` — **107 sorgu**, dengeli: **28 alan** (marangozluk/bahçıvanlık; 9+9 yeni), **28 nötr ansiklopedik**, **28 belirsiz/eksik-bilgili** (merak adayı), **23 kenar** (çok kısa/tek kelimelik/uzun/bozuk).

- **Taban korunumu:** ID 1–36, `p24_sorgu_seti.json`'dan **birebir** kopyalandı (programatik doğrulandı: `taban == sorgular[:36]` — karşılaştırılabilirlik garantisi). Yeni sorgular ID 37–107.
- **Üretim disiplini:** p24 ile aynı — gerçekçi kullanıcı soruları (canlı `/api/ask` profili), talimat-kalıplarından arınmış. Yeni kenar sorguları: `?`, `123`, `1+1`, `ev`, `MERMER`, `aaaaaa`, `neden`, `kaç`, `ahşap mı`, yazım hatalı (`nassıl`), emoji'li (`gübre 🌱`), yinelenmiş (`ahşap ahşap ahşap`), bitişik-yazım tek-kelime, ~100-jeton uzun nötr ve ~100-jeton uzun alan ramble'ları.
- Tekrar yok: 107 soru metni benzersiz (programatik kontrol).

## 2. Ölçüm kanıtı (GÖREV 2)

**Düzeneği P2.4 ile birebir:** `AgentGateway.create_default(anka_taban_pure + anka_router_v2 + bahcıvan/marangoz ceketleri + anka_bellek @ http://192.168.1.9:6333, device=mps)` → `gw.ask(mode="RAG", force_rag=False, max_new_tokens=45)` == canlı `/api/ask`. Betik `p24_olcum.py`'nin birebir uyarlaması (`p24b_olcum.py`); yalnız set/çıktı/future-train yolları farklı.

**Koşum:** kurulum 3,7 sn; **107 sorgu 79,8 sn** (ort. 0,75 sn/sorgu, MPS); **0 hata** (ask/tokenize/skor). Ham kayıtlar: scratch `p24b_olcum_kayitlar.ndjson` (107 NDJSON) → toplandığında `data/eval/p24b_esik_dagilim_2026-11-04.json` (her sorgu için: kategori, token sayısı, UNK, ham/raporlanan entropy_pre, entropy_post, ΔH, rag_score, bağımsız en-iyi skor + kaynak, conditioned, uzman, süre).

**τ=2,5 (mevcut donmuş tetik) yine 107/107 (%100) tetikledi** — P2.4 bulgusu 107 sorguda da doğrulandı: taban modelin pre-entropi tabanı (~3,3+ nat) τ'nin üzerinde; mutlak pre-entropi eşiği ayrım üretmiyor.

**Önemli yan bulgu — epistemik-başarısızlık yolu ilk kez AÇILDI:** #49 (alan, "Domates yaprakları neden kıvrılır?") `rag_score=1,0 ≥ 0,85` yüksek-benzerlik + `entropy_post=3,09 ≥ 2,5` ⇒ A-koşulu sağlandı, **1** future-train kaydı üretildi (yönlendirme sayesinde izlenen dosyaya değil scratch'e). P2.4'ün 36 sorgusunda bu yol 0'dı; dolu `simulasyon_bellek` (aşağıda) yüksek skorlu eşleşmeleri çoğalttığı için yol artık fiilen erişilebilir.

## 3. ORTAM DEĞİŞİKLİĞİ BULGUSU — 36↔107 karşılaştırmasının konfandörü (ölçülmüş)

Aynı 36 taban sorgu iki koşumda karşılaştırılınca **4 sorguda** (#23, #24, #32, #34) koşullanma durumu değişti; `entropy_pre` ise **36/36 sorguda bit-seviyesinde özdeş** (maks fark 0,0000) — ölçüm hattı deterministiktir, fark sinyalde değil **bellek ortamındadır**:

| koleksiyon | p24 koşumu (sabah) | p24b koşumu (şimdi) |
|---|---|---|
| anka_bellek | 46 nokta | 46 nokta (değişmedi) |
| simulasyon_bellek | **0 nokta (BOŞ)** | **4.073 nokta (DOLDU)** |

`simulasyon_bellek` (fallback genel bellek, `epistemic_agent.py:244-248`) iki koşum arası dolduruldu: **35/107 sorgu** artık en-iyi belgesini `simulasyon_bellek`'ten alıyor (72'si anka_bellek). Bu sorgularda koşullanma + entropy_post değişir; ΔH = H_post − H_pre bu yüzden **bellek durumuna bağımlıdır**. Sonuç: (a) yeni n=107 ölçümü **güncel canlı durumu** yansıtır (geçerli kalibrasyon zemini budur); (b) eski n=36 J değerleri eski (fallback ölü) ortamı yansıtır — 36↔107 J karşılaştırmasında set büyümesi ile ortam değişikliği iç içedir (§6'da ayrıştırıldı).

## 4. Kategori × ölçüm özeti (n=107, güncel ortam)

| kategori | n | H_post (medyan) | ΔH (medyan / [min–maks]) | conditioned | 0,33 geçen | girdide UNK |
|---|---|---|---|---|---|---|
| alan | 28 | **1,58** | −3,28 / [−8,73 … +1,18] | 19 | 19 | 1 |
| nötr | 28 | 3,70 | −1,58 / [−5,49 … +0,48] | 9 | 9 | 1 |
| belirsiz | 28 | **4,21** | −1,77 / [−6,59 … +1,35] | 17 | 17 | 5 |
| kenar | 23 | 4,02 | −4,09 / [−9,62 … −0,21] | 14 | 14 | 8 |

Ayrımın mekanizması ölçülebilir: koşullanma (alan+kenar belgeleri) entropiyi çökertiyor (alan H_post medyanı 1,58), belirsiz/eksik-bilgili sorgularda çöküş sınırlı (H_post medyanı 4,21) → **H_post ve ΔH kategorileri ayırt ediyor**; mutlak H_pre ayırt etmiyordu (P2.4 §4).

## 5. Eşik taramaları (GÖREV 3) — tetik: belirsiz=pozitif, J = belTPR − restFPR (Youden)

### 5a. delta_h — tetik koşulu: ΔH = H_post − H_pre ≥ t (merak.py opt-in 'delta_h' formu)

| t | belTPR | restFPR | J | tetik (n=107) |
|---|---|---|---|---|
| −2,50 | 0,536 | 0,392 | +0,143 | %43,0 |
| −1,75 | 0,500 | 0,329 | +0,171 | %37,4 |
| **−1,50** (eski öneri) | 0,464 | 0,291 | **+0,173** | %33,6 |
| −1,25 | 0,429 | 0,228 | +0,201 | %28,0 |
| −1,00 | 0,393 | 0,215 | +0,178 | %26,2 |
| **−0,85 (GÜÇLÜ ÖNERİ)** | 0,357 | 0,127 | **+0,231** | **%18,7** |
| −0,75 | 0,286 | 0,114 | +0,172 | %15,9 |
| −0,50 | 0,179 | 0,101 | +0,077 | %12,1 |
| 0,00 | 0,107 | 0,038 | +0,069 | %5,6 |

- Nokta-J maksimumu: **t=−0,87 → J=+0,254** (tetik %19,6) — ancak tek-adaylı dar plato [−0,88; −0,86) gürültüye açık.
- **Güçlü öneri (bootstrap J_p5-argmax): t=−0,85** → nokta J=+0,231; J_p5=+0,073. Robust bant **[−0,95 … −0,80]** (J_p5: −0,95→+0,050, −0,90→+0,068, −0,85→+0,073, −0,80→+0,035).
- Bootstrap argmax dağılımı: medyan **−0,88**, IQR [−1,45; −0,82].
- Tetiklenen belirsizler @−0,85 (10/28): #26, #27, #28, #76, #77, #80, #81, #83, #87, #88 — gerçek bağlam-eksik takip soruları; rest tetiklenenler yalnız 10/79 (alan 3, nötr 5, kenar 2).

### 5b. post_h — tetik koşulu: H_post ≥ t (merak.py opt-in 'post_h' formu)

| t | belTPR | restFPR | J | tetik |
|---|---|---|---|---|
| **+0,65** (J_p5-argmax) | 0,964 | 0,798 | +0,167 | %84,1 ← operasyonel olarak işe yaramaz |
| +2,00 | 0,786 | 0,620 | +0,165 | %66,4 |
| **+2,50** (eski hatırlama-öncelikli öneri) | 0,786 | 0,582 | **+0,203** | %63,6 |
| +3,00 | 0,714 | 0,506 | +0,208 | %56,1 |
| **+3,16** (nokta-J maks) | 0,679 | 0,481 | +0,198 | %53,3 |
| +4,00 | 0,536 | 0,367 | +0,169 | %41,1 |
| +4,50 | 0,357 | 0,278 | +0,079 | %29,9 |

- Nokta-J maksimum bölgesi ~3,0–3,2 (J ≈ +0,21–0,23); bootstrap argmax medyanı **3,12** (IQR [2,45; 3,58]).
- Mekanik J_p5-argmax +0,65'i seçer ama tetik oranını yalnız %100→%84'e indirir — kalibrasyon hedefiyle (normalde sessizlik) çelişir; öneri olarak **REDDEDİLİR**, tabloya dürüstlük için alındı.

## 6. 36 ↔ 107 stabilite ve n-büyümesi etkisi

| eşik | eski 36 (eski ortam) | aynı 36 (güncel ortam) | **107 (güncel ortam)** | yorum |
|---|---|---|---|---|
| delta_h @ −1,50 | J=+0,269 | J=+0,169 | **J=+0,173** | düşüşün tamamı ORTAM değişikliğinden (−0,100); n büyümesi tek başına +0,004 ekledi |
| delta_h @ −0,87 (yeni argmax) | J=+0,285 | J=+0,185 | **J=+0,218** | yeni argmax bölgesi 36'da da işliyor (n=107'de robust komşusu −0,85: J=+0,231) |
| post_h @ +2,50 | J=+0,238 | J=+0,238 | **J=+0,203** | ortam etkisi ~0 (tesadüfi dengeleme); düşüş n-büyümesinden (−0,035) |
| post_h @ +3,16 | J=+0,138 | J=+0,138 | **J=+0,198** | yüksek bölge n ile güçlendi |

**Argmax kayması:** delta_h argmaxı −1,50 → ≈−0,88; post_h argmaxı 2,50 → ≈3,12. **Sinyalin kendisi ve yönü stabil (koşullanma-sonrası entropi bilgisi taşıyor), sabit eşik DEĞERİ değil** — öneri bölge olarak yazılmalı: "ΔH ≥ ~−0,9" (yani koşullanma entropiyi ~0,9 nattan az düşürmüşse hâlâ belirsizsin).

## 7. Bootstrap güven aralıkları (sorgu seviyesinde 1000 yeniden-örnekleme, kategori-bazlı tabakalı, tohum 20261104, yüzde CI 2,5–97,5)

| sinyal @ eşik | nokta J | J CI95 | belTPR CI95 | tetik CI95 |
|---|---|---|---|---|
| **delta_h @ −0,85 (n=107)** | **+0,231** | **[+0,024; +0,445]** (0'ı dışlar) | [0,179; 0,536] | **[%11,2; %26,2]** |
| delta_h @ −0,87 (n=107) | +0,254 | [+0,016; +0,434] (0'ı dışlar) | [0,179; 0,536] | [%12,2; %27,1] |
| delta_h @ −1,50 (n=107) | +0,173 | [−0,043; +0,377] (0'ı İÇERİR) | [0,286; 0,643] | [%25,2; %43,0] |
| delta_h @ −1,50 (n=36 eski set) | +0,269 | [−0,085; +0,608] (0'ı İÇERİR) | — | — |
| post_h @ +2,50 (n=107) | +0,203 | [+0,002; +0,381] (kıl payı dışlar) | [0,607; 0,929] | [%55,1; %72,9] |
| post_h @ +3,16 (n=107) | +0,198 | [−0,011; +0,383] (0'ı İÇERİR) | [0,500; 0,857] | [%44,9; %62,6] |
| post_h @ +2,50 (n=36 eski set) | +0,238 | [−0,131; +0,554] (0'ı İÇERİR) | — | — |

**Kritik okuma:** n=36'da HİÇBİR eşik J'si 0'dan istatistiksel olarak ayrılamıyordu (her iki eski önerinin CI'sı 0'ı içeriyor) — "eşik dondurulmadı" hükmünün sayısal gerekçesi budur. n=107'de delta_h robust bölgesi (+0,85 bant) CI'sıyla 0'ı dışlayan TEK aday; post_h @2,50 kıl payı dışarıda, @3,16 içeride. İki sinyal birbirinden istatistiksel olarak AYIRT EDİLEMEYOR (CI'lar örtüşüyor) — seçim J'den değil operasyonel hedeften (tetik oranı) geliyor.

## 8. HÜKÜM — kalibrasyon ÖNERİSİ (uygulama YOK; dondurma denetleyicide)

**EN İYİ ÖNERİ: `trigger_signal='delta_h'`, τ = −0,85** (merak.py opt-in semantiği: `(H_post − H_pre) ≥ −0,85` ⇒ "koşullanma entropiyi ~0,9 nat düşüremedi — hâlâ belirsizsin"; robust bant [−0,95 … −0,80], bootstrap argmax medyanı −0,88).

- **Beklenen davranış (güncel canlı ortam, n=107 zemin):** tetiklenme **%100 → %18,7** (CI95 %11,2–26,2); belirsiz yakalama belTPR ~0,36, yanlış tetik restFPR ~0,13; ayrım J=+0,23 (CI95 [+0,02; +0,45]).
- **Eski öneriyle stabilite:** sinyal (ΔH) ve yön stabil; eşik değeri −1,50 → −0,85'e kaydı — eski −1,50 n=107'de J CI'sı 0'ı içerdiği için artık savunulamaz; kaymanın nedeni ağırlıklı ortam değişikliği (§6).
- **Hatırlama-öncelikli alternatif:** `post_h @ +2,50` (eski öneri) hâlâ geçerli bir ikincil seçenek: belTPR 0,79 (CI [0,61; 0,93]) ama tetik %63,6'da kalır ve J CI'sı kıl payı pozitif — "sessizlik" hedefiyle zayıf uzlaşır.
- **post_h'ın J_p5-argmax'ı (+0,65) operasyonel hedefle çeliştiği için öneri dışıdır** (tetik %84).
- **Eşik yalnız ÖNERİDİR; dondurma, uygulanabilirlik (ikinci aşama mimarisi — ΔH, H_post'u ancak geri çağırma SONRASI bildiği için tetik ancak ikinci-tur/izleme kapısı olarak anlam kazanır) ve ε-payı gibi konular denetleyici kararına aittir.**
- **Uyarı (ölçülmüş bağımlılık):** ΔH, bellek ortamına duyarlıdır — `simulasyon_bellek` 0→4.073 nokta olunca aynı 36 sorguda J 0,27→0,17'ye oynadı. Eşik dondurulmadan önce bellek envanterinin (koleksiyon + nokta sayısı) eşik belgesine çıpalanması ve büyük indeks değişikliklerinde yeniden ölçüm önerilir.

## 9. Kanıt dosyaları

| dosya | içerik |
|---|---|
| `data/eval/p24b_sorgu_seti_100plus.json` | 107 sorguluk genişletilmiş set (taban 36 birebir; ID 1–107, kategori + alt-kategori etiketli) |
| `data/eval/p24b_esik_dagilim_2026-11-04.json` | tüm ölçüm: 107 sorguluk ham kayıtlar (entropy_pre/post, ΔH, skor+kaynak, conditioned, UNK, süre) + kategori özetleri + delta_h/post_h tarama masaları + J-maks platoları + güçlü öneri (J_p5-argmax) + bootstrap CI (n=107 ve n=36) + J-kantilleri (0,05 adımlı ızgarada p5/p50/p95) + 36↔107 karşılaştırma + determinizm kontrolü |
| `data/eval/p24b_delta_h_kalibrasyon_2026-11-04.md` | bu rapor |
| scratch: `p24b_olcum.py`, `p24b_topla.py`, `p24b_olcum_kayitlar.ndjson`, `p24b_future_train_yonlendirme.jsonl` | ölçüm betiği, toplayıcı, ham NDJSON, yönlendirilen future-train kaydı (repo-dışı) |

**Değiştirilen izlenen dosya: YOK.** Commit yok, eğitim yok, Qdrant'a yazma yok. Kontrolör isterse ΔH opt-in'ini canlıya alır ve eşik dondurma kararını bu ölçüme dayandırır.