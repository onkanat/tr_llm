# P2 · AŞAMA 0e — POZİTİF KONTROL SONUCU (kalibre eşikler kanonik kaptta doğrulandı)

**Damga:** 23 Eyl 2026 · Aşama 0'ın kapanış kanıtı. İlana bağlı pozitif kontrol
(`data/eval/anka_p2_esik_kalibrasyon_ilani_2026-09-23.md` §5) İKİ dalda koşuldu ve İKİSİ DE
beklendiği gibi üretti.

---

## 1. İnsan tavanı yeni eşikle PASS (ilanda ilan edilen aritmetik)

| kayıt | kesişim (kume, n=100 seed 42, ≥2 ortak kelime) | yeni eşik | karar |
|---|---|---|---|
| `scratch/t0096_kos/kesisim_sondasi.json` (T-0096 zemin) | %13,00 | ≥ 10,92 | **PASS** |
| `scratch/anka_p2_tavan_sondasi.json` (P2/0b, bağımsız reproduce) | %13,00 | ≥ 10,92 | **PASS** |

İki bağımsız sonda aynı tavanı verir; %13,0 ≥ %10,92. Eşik ölçüm yöntemi
(n=100, seed 42, ≥2 ortak kelime) korunarak insan uzman düzeyinin %84'üne bağlandı.

## 2. Taban zemini kanonik kaptta reproduce (ölçüt değişikliği zemini korudu)

Koşum: `venv/bin/python scripts/evaluate_carpenter_anka.py --model data/anka_a1r.pt
--output data/eval/anka_p2_taban_zemin_0e_2026-09-23.json --device mps --ceket-ekseni`
(sandbox DIŞI — MPS sandbox'ta gizli).

| eksen | ölçülen (bu koşum) | bilinen zemin | eşik | karar |
|---|---|---|---|---|
| ezber | %0,00 | %0 | < 10,0 | PASS |
| tutarsızlık | **%100,00** | ~%100 | < 5,0 | **DÜŞER (beklenen)** |
| ROUGE-L | 0,0000 | 0,0000 | ≥ 0,35 | DÜŞER (beklenen) |
| kesişim | **%0,00** (tanı LCS-F1 0,0000) | ~%0 | ≥ 10,92 | **DÜŞER (beklenen)** |

⇒ Saf-Wikipedia tabanı yeni eşiklerle hâlâ düşer — kalibrasyon kapıyı gevşetmiş değil,
**ulaşılabilir banda çekmiş**. Saf tabanın %0 kesişimi ile insan tavanının %13,0'ı arasındaki
aralık, artık yetenek aşamasının gerçek hedefidir.

## 3. Kilitli zincir güncellemesi (0d — tamamlandı, tek geçiş)

| dosya | değişim | sha256 |
|---|---|---|
| `scripts/evaluate_carpenter_anka.py` | `ESIK_KESISIM` 80,0 → 10,92 (satır 71; gerekçe yorumu 62-67); LCS-F1 tanısal | `6c72ca2cb93140eefeba1fb29f32f8e7b141eca1662f95e06c21610fa71f4c64` |
| `scripts/olcum_kabi.py` | `ESIK_REFERANSLARI` satır 68/69/70/**71**/74/75'e güncellendi (kesişim 10,92); K11 kanaryası kalibrasyon-sınıfı hatalarla güçlendirildi | `ac3909852010af12bda680d04c602c08d84461638e51b87a7c3a07ea365cae46` |
| `tests/test_olcum_kabi_kapilari.py` | assert 80,0 → 10,92; ESIK_EZBER referans satırı 62 → 68 | `f90394a72d6ae03c43274c0871515e9bb12e21c56ea734b9c4942fff3a6fa2a7` |
| `scratch/t0096_kos/*` | **DEĞİŞMEZ** (kapanmış zemin) | — |

Değişmeyen tüketenler (beyan): `modul_olcum.py` yalnız `ESIK_A_ARTIS` (değişmedi) ·
`anka_i1_faz0.py` yalnız `ESIK_A_ARTIS` · `anka_karsilastirma_tablosu.py` canlı import
(değişiklik gerekmez, tablo yeni eşikle koşulduğunda kendini günceller).

## 4. Testler

* `tests/test_olcum_kabi_kapilari.py` + `tests/test_kesisim_lcs.py`: **18 passed** (kapı
  testleri: K11 kanaryası güncel ezber sınıfıyla yeşil).
* K11 kanaryası artık kalibrasyon-sınıfı hataları da yakalar: eski eşik (80,0), ham tavan
  (0,4164), eşik karışıklığı (A=5,0) — üçü de `ReferansUyusmazligi` fırlatır.

## 5. Aşama 0 kapandı — döngünün birincil kırıcısı kaldırıldı

Artık kanonik kapıda **ulaşılabildiği ölçülmüş** bir eşik var. 9 denemenin "başarısız"
yazdığı vakum, eşik-aritmetiği kaynaklıydı (80,0 = insan tavanının 6,15× katı, Ç4 VAKUM);
ölçüm yöntemi değişmeden eşik insan tavanının %84'üne bağlandı. Sıradaki: **Aşama 1**
(eğitim kodu onarımı — maske, scheduler, clip, carry).