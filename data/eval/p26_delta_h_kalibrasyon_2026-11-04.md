# P2.6 Kalem-1 — ΔH / Post-H Tetik Sinyali: OPT-IN Parametre (TDD) + Çevrimdışı Kalibrasyon

**Tarih:** 04.11.2026 · **Kural:** commit YOK, eğitim KOŞMA, model KOŞMAZ (kalibrasyon tamamen kayıtlı P2.4 verisinden) · **Donmuş τ=2,5 DEFAULT'I DEĞİŞMEZ** — yeni sinyaller YALNIZ OPT-IN, eşiği denetleyici dondurur.

**Kaynak ölçüm:** [P2.4 eşik dağılımı](p24_esik_dagilim_2026-11-04.json) (36 canlı-varı sorgusu; her sorgunun `entropy_pre` (canlı + ham) ve `entropy_post` kayıtlı). P2.4 bulgusu: mutlak entropi tetiği τ=2,5'te **%100 tetikleniyor** (taban ham ≥3,28), ayrım üretmiyor.

---

## 1. Değişiklik (minimal diff) — `src/rag/merak.py`

| Konum | Değişiklik |
|---|---|
| `merak.py:15-19` | Sınıf docstring'ine opt-in sözleşmesi (üç sinyal, τ=2,5 default değişmez) |
| `merak.py:21-31` | `__init__(..., trigger_signal: str = 'entropy')` + geçersiz değer → `ValueError` |
| `merak.py:50-69` | `detect_epistemic_gap(entropy, entropy_pre=None)`: `'delta_h'` dalı `(entropy − entropy_pre) >= tau` (entropy_pre yoksa `ValueError`); `'entropy'`/`'post_h'` mutlak `entropy >= tau` |
| `merak.py:71-93` | `forward(..., entropy_pre=None)` — opt-in parametre sinyal hesabına akar |

**Modlar:** `'entropy'` (DEFAULT; mevcut mutlak H(z)≥τ — çağıran pre-retrieval entropisi verir) · `'delta_h'` (ΔH = H_post−H_pre ≥ τ; `entropy_pre` ZORUNLU) · `'post_h'` (mutlak eşik, çağıran sözleşmesi POST-retrieval entropisini `logits`/`entropy` argümanına verir; `entropy_pre` yok sayılır).

**Mevcut davranış birebir:** `trigger_signal` verilmeden kurulan motorun hesaplama yolu bit düzeyinde aynıdır (varsayılan `'entropy'`, `entropy_pre=None`); `has_unk` (1. tetik) ve `unk_p>0,05` (2. tetik) dalları tüm modlarda aynen korunur. Yeni parametreyi kullanan tek çağıran YOK — `epistemic_agent.py:116` kurulumu ve tüm güncel davranış dokunulmadı (kapsam: parametre + test + kalibrasyon; hattın kablolaması bu kalemin DIŞINDA).

## 2. TDD kanıtları (RED önce — param yok)

Test dosyası: `tests/test_merak_trigger_signal.py` (8 test; üretim kodundan ÖNCE yazıldı).

**RED (üretim kodu değişmeden, koşum aynen):**

```
tests/test_merak_trigger_signal.py
  FAILED test_invalid_trigger_signal_rejected
  FAILED test_entropy_mode_is_absolute_and_ignores_entropy_pre
  FAILED test_delta_h_mode_triggers_on_entropy_gain_not_absolute
  FAILED test_delta_h_mode_requires_entropy_pre
  FAILED test_delta_h_mode_via_forward - TypeError: CuriosityEngine.__init__()
          got an unexpected keyword argument 'trigger_signal'
  FAILED test_post_h_mode_absolute_post_entropy
  FAILED test_post_h_mode_via_forward
  7 failed, 1 passed in 1.38s
```

7 test beklenen eksik-özellik hatasıyla düştü (param yok). 1 test — `test_default_engine_unchanged_without_entropy_pre` — mevcut-davranış sabiti olarak değişiklik ÖNCESİ de SONRASI da geçer (2,4999→tetik yok; 2,5→tetik; eşik dahil `>=`): geriye dönük uyumluluk pimi.

**GREEN (minimal diff sonrası):** `8 passed in 2.63s` (üretim kodundan sonra testte yalnız bir inceleme düzeltmesi: üniform-logit kurulumu 2. tetikleyiciyi (`p(<UNK>)=1/vocab>0,05`) ateşlediği için iki forward testine `unk_token_id=None` izolasyonu eklendi — sinyal tetiği ölçülürken yan tetikleyici susturuldu).

**İlgili mevcut süitler (değişiklik sonrası, koşum aynen):** `test_router_and_merak` + `test_epistemic_unk_and_vocab_expansion` + `test_m3_challenger_stress` + `test_epistemic_agent_fallback` + `test_moe_adversarial_stress` + `test_moe_gateway_routing` → **48 passed, 23 subtests passed (0:01:13)** — `CuriosityEngine` kullanan tüm dosyalar.

**Tam süit:** `venv/bin/python -m pytest tests/ -q` → **542 passed, 1 xfailed, 23 subtests passed, EXIT=0** (1389,23 sn; log: scratch `p26_full_suite.log`) — `CuriosityEngine` değişikliği tüm repo süitinde sıfır regresyon.

Test kapsamı: geçersiz sinyal `ValueError`; DEFAULT mutlak eşik + `entropy_pre` yok sayılır; `delta_h` kazanç-tabanlı (mutlak H≥τ olsa bile ΔH<τ tetiklemez; tam eşik `>=` semantiği; `entropy_pre` zorunlu; `forward` entegresi); `post_h` mutlak post-entropi + `entropy_pre` yok sayılır.

## 3. Çevrimdışı kalibrasyon (P2.4 ham verisi — model KOŞMADI)

Betik: `~/.hermes/cache/scratch/trllm_p26_deltah_kalibrasyon.py` · makine-ölçümü çıktı: `~/.hermes/cache/scratch/trllm_p26_deltah_kalibrasyon.json`.
Sinyal tanımları: **entropy** = `entropy_pre` (canlı; mevcut tetiğin gördüğü değer) · **post_h** = `entropy_post` · **delta_h** = `entropy_post − entropy_pre` (canlı pre) · **delta_h_ham** = `entropy_post − ham_entropy_pre` (saf pre; UNK-yükseltmesi olmadan — sağlamlık varyantı).

### 3.1 P2.4 sabit teyidi (yeniden hesaplandı)

- canlı `entropy_pre` min **3,3383**; ham min **3,2846** (taban ≥3,28 ✓)
- τ=2,5 tetiklenme: canlı **36/36**, ham **36/36** (%100 — P2.4 bulgusu birebir)

### 3.2 Sinyal dağılımları — kategori bazında (nat)

| kategori | n | ΔH (canlı pre) min / ort / medyan / maks | ΔH (ham pre) min / ort / medyan / maks | post-H min / ort / medyan / maks |
|---|---|---|---|---|
| alan | 10 | −8,734 / −3,778 / −3,422 / +1,177 | −5,234 / −3,428 / −3,422 / +1,177 | 0,055 / 1,571 / 1,040 / 5,669 |
| nötr | 10 | −5,492 / −2,547 / −2,203 / −0,860 | −5,491 / −2,197 / −1,560 / −0,068 | 0,110 / 3,407 / 3,573 / 5,408 |
| belirsiz | 10 | −5,753 / −2,758 / −2,215 / +0,078 | −5,753 / −2,408 / −1,814 / +0,078 | 0,003 / 3,024 / **4,039** / 5,129 |
| kenar | 6 | −6,173 / −4,242 / −5,418 / −0,210 | −6,173 / −3,075 / −2,365 / −0,210 | 0,108 / 2,760 / 2,504 / 6,237 |

Okuma: **alan** sorgularında retrieval entropiyi derin düşürüyor (ΔH ort −3,78; post-H ort 1,57 — çözülmüş); **belirsiz** sorgularında düşüş sığ (ΔH ort −2,76; post-H medyan 4,04 — çözülmemiş). ΔH ve post-H bu ayrımı taşır; mutlak pre-entropi taşımıyor (tüm kategoriler 3,3-9,3 bandında üst üste).

### 3.3 Sinyal × eşik → kategori tetiklenme oranı (tam tarama çıktısı)

**sinyal: entropy (mutlak pre — MEVCUT):**

| eşik | alan | nötr | belirsiz | kenar | toplam | belTPR | restFPR | J |
|---|---|---|---|---|---|---|---|---|
| 2,50 (donmuş) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 0,00 |
| 4,50 | 0,60 | 0,90 | 1,00 | 1,00 | 0,86 | 1,00 | 0,81 | +0,19 |
| 5,00 | 0,60 | 0,90 | 1,00 | 1,00 | 0,86 | 1,00 | 0,81 | +0,19 |
| 5,50 | 0,30 | 0,70 | 0,40 | 1,00 | 0,56 | 0,40 | 0,62 | −0,22 |
| 6,50 | 0,10 | 0,10 | 0,10 | 0,50 | 0,17 | 0,10 | 0,19 | −0,09 |

(tam ızgara 2,5…9,0; J ≤ +0,19 — eşik ne olursa olsun anlamlı ayrım YOK; en iyi durumda bile %86 tetik)

**sinyal: post_h (mutlak post):**

| eşik | alan | nötr | belirsiz | kenar | toplam | belTPR | restFPR | J |
|---|---|---|---|---|---|---|---|---|
| 0,50 | 0,70 | 0,90 | 0,80 | 0,67 | 0,78 | 0,80 | 0,77 | +0,03 |
| 2,50 | 0,20 | 0,70 | 0,70 | 0,50 | 0,53 | 0,70 | 0,46 | **+0,24** |
| 3,00 | 0,20 | 0,70 | 0,60 | 0,50 | 0,50 | 0,60 | 0,46 | +0,14 |
| 3,50 | 0,20 | 0,50 | 0,60 | 0,50 | 0,44 | 0,60 | 0,38 | +0,22 |
| 4,00 | 0,10 | 0,40 | 0,50 | 0,50 | 0,36 | 0,50 | 0,31 | +0,19 |

**sinyal: delta_h (post − canlı pre):**

| eşik | alan | nötr | belirsiz | kenar | toplam | belTPR | restFPR | J |
|---|---|---|---|---|---|---|---|---|
| −5,00 | 0,80 | 0,90 | 0,70 | 0,50 | 0,75 | 0,70 | 0,77 | −0,07 |
| −3,00 | 0,20 | 0,60 | 0,50 | 0,33 | 0,42 | 0,50 | 0,38 | +0,12 |
| −2,50 | 0,10 | 0,50 | 0,50 | 0,33 | 0,36 | 0,50 | 0,31 | +0,19 |
| −2,00 | 0,10 | 0,50 | 0,50 | 0,17 | 0,33 | 0,50 | 0,27 | +0,23 |
| **−1,50** | 0,10 | 0,40 | 0,50 | 0,17 | 0,31 | 0,50 | 0,23 | **+0,27** |
| −1,00 | 0,10 | 0,20 | 0,40 | 0,17 | 0,22 | 0,40 | 0,15 | +0,25 |
| 0,00 | 0,10 | 0,00 | 0,10 | 0,00 | 0,06 | 0,10 | 0,04 | +0,06 |

**sinyal: delta_h_ham (post − ham pre; sağlamlık):** en iyi **−2,50 → J +0,22** (belTPR 0,60; restFPR 0,38); −1,00 → J +0,21. Canlı-pre varyantıyla aynı yapı — öneri UNK-yükseltmesi olmayan ham ölçüme de dayanıyor.

### 3.4 Belirsiz ayrım sıralaması (J = belTPR − restFPR; ilk 6)

| J | sinyal | eşik | belTPR | restFPR | toplam tetik |
|---|---|---|---|---|---|
| **+0,27** | delta_h | −1,50 | 0,50 | 0,23 | 0,31 |
| +0,25 | delta_h | −1,00 | 0,40 | 0,15 | 0,22 |
| +0,24 | post_h | +2,50 | 0,70 | 0,46 | 0,53 |
| +0,23 | delta_h | −2,00 | 0,50 | 0,27 | 0,33 |
| +0,22 | post_h | +3,50 | 0,60 | 0,38 | 0,44 |
| +0,22 | delta_h_ham | −2,50 | 0,60 | 0,38 | 0,44 |

Kusursuz belirsiz yakalama (belTPR=1,0) istense en düşük restFPR **0,81** (her sinyalde) — %100 hatırlama bu veri setinde ayrım DEĞİL; anlamlı seçicilik ancak belirsizin bir kısmını feda ederek elde ediliyor.

## 4. ÖNERİ (uygulama YOK — eşik denetleyicide donurulur)

1. **Birincil öneri: `trigger_signal='delta_h'`, τ_ΔH = −1,50.** Tetik koşulu ΔH ≥ −1,5, yani "retrieval entropiyi en az 1,5 nat düşüremedi". Ölçülen davranış: belirsizin %50'si, nötrün %40'ı tetiklenir; retrieval'ın çözdüğü **alan %10'a, kenar %17'ye düşer**; toplam tetik oranı %100 → **%31**. Semantik olarak "retrieval çözemedi — epistemik boşluk sürüyor" dedektörü; mevcut %100-tetik davranışının yerine seçici yeniden-tetik sinyali.
2. **Hatırlama-öncelikli alternatif: `trigger_signal='post_h'`, τ = 2,50** (donmuş τ ile aynı sayısal değer, fakat sinyal POST-retrieval entropisi): belirsizin %70'i tetiklenir, restFPR 0,46, toplam tetik %53. J farkı (+0,24'e karşı +0,27) tek-kayıt mertebesinde — belirsiz hatırlaması önemliyse bu makul tercih.
3. **Mutlak entropi sinyali (mevcut) hiçbir eşikte ayrım üretmiyor** (en iyi J +0,19 @ eşik 4,5-5,0, toplam tetik yine %86): P2.4 bulgusu bu kalibrasyonla da doğrulandı. Sinyal değişikliği eşik ayarından ÖNCE gelir.
4. **Öneri bandı** (dondurma öncesi ikinci ölçüm için): delta_h için −2,0…−1,5 (J ≥ +0,23); post_h için 2,5…3,5 (J ≥ +0,22).
5. **Kısıt:** n=36 (belirsiz=10; belirsizde 1 kayıt = %10). J sıralaması tek-kayıt hassasiyetinde; eşik dondurmadan önce P2.4 sorgu setinin genişletilmesi (özellikle belirsiz+kenar) önerilir. Bu rapor yalnız öneridir — **τ=2,5/'entropy' donmuş varsayılan bu kalemden etkilenmedi.**

Opt-in kullanım (hattın kablolaması ayrı kalem; örnektir):

```python
# ΔH modu: çağıran pre-entropiyi de verir
engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=-1.5,
                         trigger_signal='delta_h')
entropy, needs_retrieval, q_merak = engine(hidden, logits_post,
                                           entropy_pre=H_pre)
```

---

**Değişen/yeni dosyalar:** `src/rag/merak.py:15-31,50-73,76-100` (opt-in parametre) · `tests/test_merak_trigger_signal.py` (yeni, 8 test) · bu rapor. Betikler repo-dışı scratch'te: `trllm_p26_deltah_kalibrasyon.py`, `trllm_p26_deltah_kalibrasyon.json`, `trllm_p26_probe_neg2sg.py`.

**Ölçüm oturumu:** 04.11.2026; tüm koşumlar salt-okuma; izlenen dosyalara dokunulmadı; commit YOK.