# P24e — ΔH Sinyalinin Aux Çağına Uyarlanması: Kaynak-Koşullu Eşik (2026-11-05)

## İLAN

**Kapsam:** Kalem-4. p24c hükmünün devamı: çıpalı envanterde `delta_h@−0,85` çürüdü (J +0,113, CI 0 içerir; aux 32/107 flip, belirsiz 28/28 koşullandı — ayrım öldü). Bu kalem, p24c'nin 107 ham kaydı üzerinde **kaynak-koşullu kural ailelerini offline tarar** (J + bootstrap CI, p24c yöntemiyle birebir), 0-dışlayan kural varsa OP-İN `aux_aware` sinyali olarak merak.py'ye uygular; yoksa dürüst 'yok' der.

**Sert kurallar (hepsi tutuldu):** git commit YOK; eğitim YOK; **Qdrant SALT-OKUNUR**; ölçüm offline (p24c ham kaydı; yeni ağ çağrısı YOK); izlenen dosyalar en küçük diff (yalnız `merak.py` + yeni test). Betikler scratch'te (`p24e_kural_tarama.py`, `p24e_ek_analiz.py`).

**Yöntem doğrulaması:** p24c ile aynı bootstrap (sorgu-seviyesi 1000 yeniden-örnekleme, kategori-tabakalı, tohum=20261104, %2,5–97,5 CI) yeniden kurularak yayınlanmış değerlerle karşılaştırıldı: `delta_h@−0,85` → J=+0,113, CI [−0,049;+0,294] (p24c: [−0,045;+0,292]); `post_h@2,50` → +0,145 [−0,061;+0,330] (p24c: [−0,067;+0,336]). Sapmalar bootstrap örnekleme-gürültüsü düzeyinde; yöntem-tutarlı.

---

## HÜKÜM

### 1. 0-DIŞLAYAN KURAL VAR — kaynak-koşullu eşik

3.509 kural nokta-J'ye tarandı (7 aile × eşik ızgaraları); tetik ≤%75 koşuluyla en iyi 12 aday bootstrap edildi:

| aile | kural | nokta J | J CI95 (1000) | J_p5 | tetik | 0'ı dışlar? |
|---|---|---|---|---|---|---|
| **F1 kaynak-böl (ÖNERİLEN)** | **aux→post_h≥2,25 \| birincil→ΔH≥−0,75** | **+0,297** | **[+0,091;+0,516]** | **+0,129** | **%28,0** | **EVET** |
| F2 cond-böl | cond→post_h≥2,75 \| cond-değil→ΔH≥0,0 | +0,322 | [+0,116;+0,515] | +0,162 | %47,7 | EVET (ama ikinci kol VAKUM) |
| F2 cond-böl (yumuşak) | cond→post_h≥2,75 \| cond-değil→ΔH≥−0,75 | +0,297 | [+0,093;+0,495] | +0,128 | %49,5 | EVET (kıl payı) |
| F7 kaynak-böl ΔH | aux→ΔH≥−4,25 \| birincil→ΔH≥−0,75 | +0,260 | top-12 dışı (bootstrap edilm.) | — | %30,8 | bilinmiyor |
| F3 skor-böl | skor<0,50→post_h≥6,25 \| skor≥0,50→ΔH≥−2,75 | +0,170 | top-12 dışı | — | %19,6 | hayır (nokta zayıf) |
| F4 aux-kapılı ΔH | birincil→ΔH≥−0,75; aux asla | +0,069 | — | — | %5,6 | hayır |
| F5 aux-kapılı post_h | birincil→post_h≥3,0; aux asla | +0,015 | — | — | %34,6 | hayır |
| F6 skor-kapılı ΔH | skor<0,6→ΔH≥0,0 | +0,095 | — | — | %3,7 | hayır |
| taban delta_h@−0,85 | koşulsuz | +0,113 | [−0,049;+0,294] | — | %13,1 | HAYIR (p24c hükmü) |
| taban τ=2,5 (yürürlükteki) | entropy(pre)≥2,5 | 0,000 | — | — | **%100** | hayır (sıfır ayrım) |

**Neden F1 önerilir (F2 değil):**
1. **Her iki kol canlı:** F1'de aux kolu 11 belirsiz, birincil kol 3 belirsiz tetikler — ΔH birincil zincirde YAŞAR (kalem-4'ün amacı). F2'nin `cond-değil` kolu vakumdur: koşullanmamış 16 sorgunun ΔH≥0 olanı 0 → kural fiilen "cond ∧ post_h≥2,75"e dejenere; J üstünlüğü tamamen post_h-kolundan gelir, ΔH mekanizması katkı vermez.
2. **Mekanizma tutarlılığı (p24c §6):** aux koşullanması entropiyi tek-yönlü düşürmüyor, YENİDEN DAĞITIYOR (20 düşüş/12 artış) → aux'ta ΔH işaret taşımaz, mutlak post-h taşır. F1 bu ölçümün doğrudan kodlamasıdır; iş-emrinin 'kaynak-koşullu eşik' adı budur.
3. **Plato geniş (robust):** tA∈[2,25; 2,5] × tB∈[−1,0; −0,5] çevresi J +0,26…+0,30 — bıçak-sırtı değil.
4. **Tetik %28:** yürürlükteki τ=2,5 sinyal %100 tetikliyor ve J=0,000 üretiyor; F1 hem seçici hem ilk kez 0-dışlayan.

### 2. DONDURMA ÖNERİSİ — VAR, çıpalama koşuluyla

> **Öneri:** `trigger_signal='aux_aware'` (merak.py OP-İN) ile **tau_aux = 2,25** (aux kaynaklı bağlamda post-entropi eşiği), **tau_primary = −0,75** (birincil zincirde ΔH eşiği). Zemin: p24c çıpalı envanter, 107 sorgu, J=+0,297, CI [+0,091;+0,516] (0'ı DIŞLAR), tetik %28,0. Dondurma DENETÇİ kararıdır (p24c emsalı).

**ÇIPALAMA KURALI (p24c HÜKÜM §2 aynen yürürlükte; merak.py docstring'ine ve bu rapora çıpalıdır):** Eşik, ölçüldüğü Qdrant envanterine (anka_bellek 46 / muhakeme_bellek 91 / simulasyon_bellek 4.073 / gts_bellek 128.327 / kanun_madde 4.068 / yetenek_agi 36) bağlıdır. **Envanter değişince — yeni koleksiyon ekimi, mevcut koleksiyona ekim/silme, aux zinciri değişikliği — eşik YÜRÜRLÜKTEN DÜŞER**; yeniden ölçüm + yeniden çıpalam yapılmadan kullanılmaz.

### 3. Kalem-3 düzeltmesiyle etkileşim (ölçüldü)

Kalem-3 kapısı #91/#104'ü koşullanmamış tabana döndürünce (canlı fark ölçümü) F1'in J'si **+0,297 → +0,310**'a, CI [+0,112;+0,529]'a iyileşiyor (çöp girdiler rest-FPR'den çıkıyor). Kalem-3 ve Kalem-4 aynı yönde çalışır; önerilen sinyal Kalem-3-sonrası envanterde DAHA güçlüdür.

### 4. Denetçiye dürüst notlar

1. **Uygulama OP-İN'dir, canlı yığın DEĞİŞMEDİ:** `aux_aware` merak.py'de uygulanır; gateway/epistemic-agent hâlâ donmuş `entropy`/τ=2,5 ile çalışır. Benimsenme (canlı `needs_retrieval`'ın `aux_aware`'a geçirilmesi + `context_is_aux`/`entropy_pre` kablolaması) denetçi tasarımıdır — bu kalem sinyali ölçüp kapattı, yığına bağlamadı.
2. `context_is_aux` çağıran tarafından ancak RETRIEVAL-SONRASI bilinir; sinyal sözleşmesi buna uygundur (post-decision: future-train A-kapısı, yeniden-arama kararı). Pre-retrieval tetiği olarak kullanılamaz.
3. F2 (cond-böl) alternatif olarak raporlanır: J'si daha yüksek (+0,322) ama ikinci kolu bu envanterde vakum; envanterde koşullanma-oranı değişirse davranışı belirsizdir. Dondurma F1'e önerilir.

---

## RAPOR

### §1 Tarama düzeneği

107 ham kayıt (p24c `olcum_kayitlari`: kategori/kaynak/skor/cond/ΔH/post-H). Pozitif sınıf `belirsiz` (28), rest 79. J = belTPR − restFPR. Aileler: F1 kaynak-böl (aux→post_h | birincil→ΔH), F2 cond-böl, F3 skor-böl (<0,50), F4/F5 aux-kapılı, F6 skor-kapılı ΔH, F7 kaynak-böl ΔH; ızgaralar tA∈[0; 7] ∩ 0,25, tB∈[−5; 2] ∩ 0,25. Tetik-bandı kapısı: ≤%75 (p24b hükmü %78,5+ RED emsali).

### §2 F1 kazananının kol-bazlı ayrışımı (anıt değerler)

- aux kolu (post_h≥2,25): 24 tetik — 11/28 belirsiz (#26,27,28,73,76,77,78,80,83,85,89) + 13 rest.
- birincil kolu (ΔH≥−0,75): 6 tetik — 3/28 belirsiz (#81,87,88) + 3 rest.
- Toplam tetik %28,0 (30/107); belTPR 0,500, restFPR 0,203.

### §3 TDD (RED → GREEN)

- **RED:** `tests/test_p24e_merak_aux_aware.py` 9/10 DÜŞTÜ ('aux_aware' yok — `TypeError`/`ValueError`).
- **GREEN:** `src/rag/merak.py`: (i) `__init__` imzasına `tau_aux=2,25`, `tau_primary=−0,75` parametreleri + `'aux_aware'` mod doğrulaması (satır 21-22, 30, 44-48); (ii) `detect_epistemic_gap`'e `context_is_aux` parametresi + aux_aware dalı (context_is_aux ZORUNLU; aux→mutlak H≥tau_aux; birincil→ΔH≥tau_primary, entropy_pre ZORUNLU; hepsi fail-closed `ValueError`) (satır 71-115); (iii) `forward`'a `context_is_aux` geçişi (satır 117-146).
- **Varsayılan BİT-ÖZDEŞ:** `'entropy'` modu `context_is_aux`/`tau_aux`/`tau_primary`'yi YOK SAYAR (test: `test_varsayilan_mod_yeni_parametreleri_yok_sayar`); mevcut P2.6 testleri değişmeden yeşil.
- Dondurulmuş eşik drift-guard'ı: `test_aux_aware_dondurulmus_esikler` (tau_aux=2,25, tau_primary=−0,75).

### §4 Değişen izlenen dosya

| dosya | satır | değişiklik |
|---|---|---|
| `src/rag/merak.py` | 20-37 | sınıf docstring'ine aux_aware sözleşmesi + ölçüm değerleri + ÇIPALAMA KURALI |
| `src/rag/merak.py` | 38-53 | `__init__`: `'aux_aware'` modu (40-45) + `tau_aux`/`tau_primary` (39, 51-53; dondurulmuş ölçüm varsayılanları) |
| `src/rag/merak.py` | 71-110 | `detect_epistemic_gap`: `context_is_aux` (73) + aux_aware dalı (97-109; fail-closed) |
| `src/rag/merak.py` | 113-146 | `forward`: `context_is_aux` geçişi (118; OP-İN; varsayılan modda yok sayılır) |
| `tests/test_p24e_merak_aux_aware.py` | YENİ | 10 test: varsayılan-birebir, fail-closed kapılar, donmuş eşikler, özel eşik, forward entegresi |

### §5 Kanıt dosyaları

| dosya | içerik |
|---|---|
| `data/eval/p24e_delta_h_aux_cagi_2026-11-05.md` | bu rapor (İLAN→HÜKÜM→RAPOR) |
| scratch: `p24e_kural_tarama.py` + `p24e_kural_tarama.json` | 3.509 kural taraması + doğrulama bootstrap'ları + top-12 CI |
| scratch: `p24e_ek_analiz.py` + `p24e_ek_analiz.json` | F1 plato-komşuları, kol-bazlı ayrışım, F2 vakum kanıtı, Kalem-3 duyarlılığı |

**Değişen izlenen dosya:** `src/rag/merak.py` (+ yeni test dosyası). Commit yok, eğitim yok, Qdrant'a yazma yok, ağ çağrısı yok (tarama tamamen offline).
