# G6a (T-0137) İLAN — KÖK-NEDEN ÖLÇÜM GÖREVİ (A9 BAND_ALTINDA + A13 IHLAL) — 2026-09-26 (claude)

Operatör emri (2026-09-26): "kök neden ölçüm görevini aç — marangoz verisi ile
Bahçıvan veri setini kıyasla". ILAN ≠ RAPOR: rapor
`bahcivan_g6a_kokneden_kiyas_sonuc_2026-09-26.md`. Hüküm BETİKTEN (elle sayı
YOK). Bu TANISAL ölçüm görevidir — kabul ayracı DEĞİL; dallar bulgu
etiketleridir, "geçti/kaldı" kararı İÇERMEZ.

## Ölçüm beyanı (koşum ÖNCESİ)

Betik `scratch/t0137/bahcivan_kok_neden_kiyas.py`; üç kısım:

1. **Kısım A — veri kıyas (Bahçıvan arena 2.199 vs marangoz katalog
   `scratch/t0097_kos/katalog_yeni.jsonl` 5.400):** çift sayısı · soru/cevap
   kelime uzunluğu (ort, medyan) · benzersiz soru/cevap oranı · birebir soru
   tekrar oranı · kalıp dağılımı (Bahçıvan `aile` alanı; marangoz instruction
   ilk-kelime üst-6). Kardeş veri `data/pedagogy/carpenter_specialization_dataset.jsonl`
   (4.861) ikinci marangoz kaynağı olarak sayılır (salt-okunur okuma).
2. **Kısım B — ROUGE uzunluk-kesişim ayrıştırması (beyanlı simülasyon):**
   ÖLÇÜM-1/2 sonda JSON `ham_gm` (100 üretim) üretim-uzunluğu + iki heldout
   referans-uzunluğu → mutlak kesişim yaklaşımı
   `K ≈ ROUGE_ort × (ort_len_ref + ort_len_hyp) / 2`. Bu SİMÜLASYONDUR
   (ortalama toplamı yaklaşımı) — ROUGE hükmü DEĞİŞMEZ; yalnız uzunluk
   normalizasyonu etkisini ayırır.
3. **Kısım C — cevap-CE yetenek ölçümü (MPS, sandbox dışı):** Bahçıvan heldout
   220 satır → normalize SFT kaydı `{"instruction": ZARF, "input": <soru>,
   "output": <cevap>}` encode (`tokenizer.encode` — derleme deseni) → tek-kayıt
   pencere (blok 128 PAD + `mask_prompt_targets` + `maske_pad_hedefleri`) →
   maskeli CE taban (`data/anka_base_v2.pt`) vs ft (`seg_3.pt`). Aynı istemler
   iki modelde. **Dallar:** CE farkı (taban − ft) > 0 → **CE_DUSTU** (veri
   öğrenildi — kusur ÜRETİMDE) · ≤ 0 → **CE_DUSMEZ** (öğrenme sinyali yok —
   dozaj/çekicilik). Model kurulum deseni P3 birebir (`KristalLM` +
   `resize_state_dict`, strict=False; RoPE/mask tamponları silinir).

## DOKUNULMAZLAR

`evaluate_carpenter_anka.py` / `anka_p3_surucu.py` / tokenizer / compiler
DOKUNULMAZ (import yalnız) · marangoz heldout DOKUNULMAZ (salt-okunur) ·
data/** salt-okunur · checkpoint yazımı YOK · çıpa/kapı değerleri DOKUNULMAZ ·
ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · hüküm BETİKTEN ·
`git add -A` YASAK.

## Kapanış

Rapor `bahcivan_g6a_kokneden_kiyas_sonuc_2026-09-26.md` (digest tablo nihai
koşum sonrası); notes FAZ-3; bulgular ölçümlerle sınırlı — aday hipotezler
yalnız ölçülen eksenlerde beyan edilir.