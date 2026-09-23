# P2 · AŞAMA 0c — EŞİK KALİBRASYON İLANI (kesişim kapısı ölçülmüş insan tavanına bağlanır)

**Damga:** 23 Eyl 2026 · **Döngü kırıcısı.** Bu ilan, 9 denemenin hepsini "başarısız" yazan
ulaşılamaz eşiği ölçülmüş insan tavanına bağlar. Formul ilanda sabittir; ölçümden sonra
eşik DEĞİŞMEZ — yeni ölçüt ancak yeni bir insan tavanı ölçümüyle değişir (döngü-sonlandırma
tablosu, plan).

---

## 1. Ölçülen zemin (elle sayı yok; iki bağımsız sonda)

| kaynak | ölçüt | insan uzman değeri |
|---|---|---|
| `scratch/t0096_kos/kesisim_sondasi.json` (T-0096, değiştirilemez zemin) | kesişim (kume) | **%13,0** |
| `scratch/anka_p2_tavan_sondasi.json` (P2/0b, yeni) | kesişim (kume) | %13,0 — reproduce ✓ |
| `scratch/anka_p2_tavan_sondasi.json` | kesişim **LCS-F1** | JETON 0,0446 · YUZEY 0,0984 |
| `scratch/t0096_kos/kesisim_sondasi.json` | ROUGE-L | 0,4164 |
| `scratch/t0096_kos/kesisim_sondasi.json` | tutarsızlık | %3,0 |
| `scratch/t0096_kos/kesisim_sondasi.json` | ezber | %0,0 |

## 2. 0b'de ölçülen GERİ-DÖNÜŞ: LCS-F1 ölçütü REDDEDİLDİ

Plan Aşama 0a LCS-F1'i hüküm ölçütü yapmayı hedefledi ve geri-dönüş dalı ilan etmişti:
*"LCS tavanı beklenmedik derecede düşürürse küme ölçütüne dön, yalnız eşik kalibrasyonuyla
devam."* **Dal ateşlendi**:

* İnsan uzman cevabı LCS-F1'de yalnız **0,0446** (JETON — kodun kullandığı yol) alır;
  eşik = 0,0446 × 0,84 ≈ 0,037 — bu boyutta eşik bilgi taşımaz.
* **23/100 kayıtta YUZEY ↔ JETON kararı ayrışır** — ölçüt, temsile bağımlı
  (gösterim-uyuşmazlığı sınıfı; `[[gosterim-uyusmazligi-olcutu-oldurur]]`).
* **Küme ölçütü (≥2 ortak kelime) HÜKÜM ÖLÇÜTÜ OLARAK KORUNUR**; LCS-F1 kanonik kaptan
  silinmeden **tanısal** alan olarak raporlanmaya devam eder (`kesisim_f1_ort`).
* Birim testler (7/7 PASS, `tests/test_kesisim_lcs.py`) LCS-F1'i tanı olarak doğru
  davranışta tutar.

## 3. Kalibrasyon formülü ve İLAN EDİLEN YENİ EŞİKLER

**Formül (sabit):** `eşik = insan_tavani × k` · **k = 0,84** (ROUGE'un mevcut
0,35/0,4164 = 0,8405 oranı — mevcut talep düzeyi korunur, sadece kesişim bu düzeye çekilir).

| eşik | eski | **yeni** | hesap |
|---|---|---|---|
| `ESIK_KESISIM` | 80.0 (ULAŞILAMAZ, 6,15×) | **10.92** (≥ %10,92) | %13,0 × 0,84 = 10,927 |
| `ESIK_ROUGE` | 0.35 | **0.35 (değişmedi)** | zaten 0,4164 × 0,84 |
| `ESIK_TUTARSIZ` | 5.0 | **5.0 (değişmedi)** | %3,0 × 1,67 = 5,01 — mevcut değer zaten bu formülde |
| `ESIK_EZBER` | 10.0 | **10.0 (değişmedi)** | tavan %0 — geniş marj |
| `ESIK_A_ARTIS` / `ESIK_B_DUSUS` | 10.0 / 5.0 | **değişmedi** | gürültü-kalibre (farklı sınıf) |

**Anlam:** model, kesişimde ölçülen insan uzman düzeyinin en az **%84**'ünü üretmelidir
(100 kayıtta ≥2 ortak kelime kuralına göre ≥ %10,92). Bu ulaşılabildir ve anlamlıdır:
taban %0, yetenek aşaması hedefi bu banttır.

## 4. Kilitli zincirin güncellenmesi (0d — tek geçiş)

| dosya | iş |
|---|---|
| `scripts/evaluate_carpenter_anka.py` | `ESIK_KESISIM` 80,0 → **10,92** (+ gerekçe yorumu); LCS-F1 tanısal |
| `scripts/olcum_kabi.py` (ESIK_REFERANSLARI) | kesişim kanaryası 80,0 → 10,92 |
| `tests/test_olcum_kabi_kapilari.py:326` | assert güncellenir |
| `scripts/modul_olcum.py`, `scripts/anka_i1_faz0.py`, `scripts/anka_karsilastirma_tablosu.py` | `ESIK_A_ARTIS`/`ESIK_ROUGE` import'u — **değişiklik yok** |
| `scratch/t0096_kos/kesisim_sondasi.py` | **DEĞİŞMEZ** (kapanmış zemin); kalibrasyon referansı
`scratch/anka_p2_tavan_sondasi.py` |

## 5. Pozitif kontrol şartı (0e — ilan edilir, KOŞULACAK)

* **İnsan tavanı yeni eşikle PASS ÜRETMEK ZORUNDA:** %13,0 ≥ %10,92 ✓ (sonda kaydında
  yazılıdır; kalibrasyon ilanının kendisi bu eşikle insan tavanını GEÇİRTİR).
* **Taban zemini reproduce ZORUNDA:** `anka_a1r.pt` kanonik kaptta bilinen zemin
  (tutarsızlık ~%100, ezber %0, kesişim ~%0) — ölçüt değişikliğinin zemini koruduğunun
  kanıtı. Bu koşum P2/0e'de yapılır ve `data/eval/` altına yazılır.
* Değer 10,92 elle yazılmıştır AMA iki bağımsız kayıttan (t0096 zemin + P2 sonda)
  okunan %13,0'ın ölçülmüş k ile çarpımıdır; ikisi de tablolarda kanıtlıdır.

## 6. Değişimin sınırları

* Ölçüm YÖNTEMİ (n=100, seed 42, ≥2 ortak kelime) **değişmedi** — yalnız eşik değeri.
* Kapların çözünürlük şartları (Wilson yarı-genişlik < eşik payı) değişmedi; kesişim
  Wilson n=100'de ±~6,4 puan genişliğinde — eşik %10,92'nin kararı bu örneklemde
  ±6,4 bantta okunmalı (raporda beyan edilir; dar karar için n artışı ayrı ilan ister).
* Bu ilan **geriye dönük kıyaslanabilirlik** bozmaz: eski `data/eval/*` kayıtları eski
  eşiğiyle kalır (değiştirilmez); yeni koşumlar yeni eşiği `esikler` alanında taşır.