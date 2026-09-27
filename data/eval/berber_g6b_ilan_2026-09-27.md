# G6b (T-0138) İLAN — Berber Dikeyi — 2026-09-27 (antigravity)

Bu dosya İLANDIR, RAPOR DEĞİLDİR (ILAN ≠ RAPOR — G4 kusur sınıfı). Ölçümler koşum
ÖNCESİ ilanlanır; hükümler BETİKTEN yazılır, elle sayı yoktur. Taban: `data/anka_base_v2.pt`
(`d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`, MÜHÜRLÜ 2.0-sealed —
T-0136; mimari 6/6/768/RoPE/tying-false DOKUNULMAZ). Şartname: `ceket_tasarim_2026-09-26.md` §8.

## İLAN-1 — TABAN-ÇIPA (koşum başlıyor)

- **Amaç:** Eğitilmemiş mühürlü tabanın kanonik kapta ceket ekseninde referans davranışı.
  T-0137 bulgusu: base_v2 seg_3 devralımı nedeniyle marangoz ceketinde 0,1392 üretmektedir;
  bu nedenle ölçüt canlılığı İLAN-4 (boş model) ile pozitif kontrol edilir.
- **Koşum:** Kanonik kap `scripts/evaluate_carpenter_anka.py` DOKUNULMAZ;
  `./venv/bin/python scripts/evaluate_carpenter_anka.py --model data/anka_base_v2.pt --ceket-ekseni --output scratch/t0138/taban_cipa_sonda.json`
  (kanonik varsayılanlar: vocab `data/rebuild/vocab_anka_r1_33114.json` çıpa zinciri, lexicon,
  heldout, n=100, seed=42, device=mps, max-new=128). Sandbox dışı koşum gereklidir (MPS erişimi).
- **AYRAÇ (koşum-öncesi ilan):** Sonda JSON ROUGE-L alanı betikten okunur;
  `ROUGE-L < 0,005` → **TABAN_CIPA_GECTI** · `≥ 0,005` → **TABAN_CIPA_DUSTU**
  (T-0137 emsali: devralınmış zincir nedeniyle ROUGE-L ~0,1392 bandı beklenir; operatör onayıyla
  LM-bedeli çıpası tabanın kendi ROUGE'sine bağlanır).

## İLAN-4 — BOŞ MODEL POZİTİF KONTROLÜ

- **Amaç:** Ölçütün ayırt ediciliği kanıtı — ROUGE ölçütünün kendisinin boş modelde
  sıfıra yakınsadığı kanıtlanır.
- **Boş model:** `KristalLM(33114, 768, blok 4096, 6 katman, 6 kafa)` rastgele ağırlık,
  tohum **42** (determinizm kuralı), `scratch/t0138/bos_model.pt`
  (`765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9`).
- **Koşum:** Kanonik kap, `--model scratch/t0138/bos_model.pt --ceket-ekseni --output scratch/t0138/bos_pozitif_kontrol.json`,
  aynı varsayılanlar (n=100, seed 42, mps).
- **AYRAÇ (koşum-öncesi ilan):** `ROUGE-L < 0,005` → **POZITIF_KONTROL_GECTI**
  (ölçüt boşu ayırt ediyor) · `≥ 0,005` → **POZITIF_KONTROL_DUSTU** (ölçüt körlemesine geçiyor).

## İLAN-2 — ARZ ÜRETİMİ PİLOT KAPISI

- **Alan:** Berberlik, saç-sakal kesimi ve erkek bakımı uzmanlığı alanı.
- **Örneklem:** Pilot n=20 (D3-istisna kalıbı: soru-kalıp + cevap-sonlandırma düzeni).
- **Kalıp aileleri (5 dengeli aile):**
  1. `nedir`: "`{K}` nedir?"
  2. `nasil`: "`{K}` nasıl yapılır / uygulanır?"
  3. `ne_zaman`: "`{K}` ne zaman kullanılır / yapılır?"
  4. `sorun`: "`{K}` nasıl önlenir / giderilir?"
  5. `islev`: "`{K}` berberlikte ne işe yarar?"
- **AYRAÇ (betik hükmü):**
  1. `kalip_koruma >= 18/20` (%90) — tanımlı soru kalıbı ve tamamlanmış yüklemli cevap sonu.
  2. `kalip_cekimlilik`: Soru-kalıp sayısı ≥ 5 farklı aile.
  3. Türkçe jeton dökümü/yabancı dil karışımı: 0 örnek (mutlak).
- **Hüküm dalları:** **PILOT_GECTI** (arz üretimi açılır) · **PILOT_YETERSIZ** (kayıt yapılmaz).

## İLAN-5 — TAM ARZ ÜRETİMİ (Berber Arena Külliyatı)

- **Hedef boyut:** **2.115 çift** (marangoz 10.575 külliyatının %20'si — T-0137/G6a ile aynı hedef).
- **Aile dağılımı (dengeli, §5 kuralı):**
  - `nedir`: 425
  - `nasil`: 425
  - `ne_zaman`: 425
  - `sorun`: 425
  - `islev`: 415
  Toplam = 2.115 çift.
- **AYRAÇ (betik hükmü):**
  1. Kalıp koruma ≥ %90 (≥ 1.904/2.115), aile sayısı = 5, dil karışımı = 0.
  2. **Birebir soru tekrarı ≤ %5** (≤ 106 çift).
  Hüküm: **BIRLESIM_GECTI** · **BIRLESIM_YETERSIZ**.
- **Nihai dosya:** `data/pedagogy/berber_arena.jsonl` (alanlar: `soru/cevap/kaynak/aile`).

## İLAN-3 — DERLEME (D3-istisna + bin dekod)

- **Adapter şeması:** `{"instruction": "Berber uzmanı olarak cevapla. Soru: {soru}", "input": "", "output": "{cevap}"}`.
- **Sistem zarfı:** `system_prompt = "Berber uzmanı olarak cevapla."`.
- **Kütüphane:** T-0134 `src.llm.dataset_compiler` (`tokenize_specialization_jsonl` + `bin_dekod_dogrula`).
- **Tokenizer:** `data/rebuild/vocab_anka_r1_33114.json` + `data/lexicon/roots.tsv` + `literal_entity_mode=True`.
- **Çıktı:** `scratch/t0138/berber_arena.bin` + `.meta.json`.
- **AYRAÇ:**
  1. Parse/encode/normalize hatası 0.
  2. Beklenen kayıt sayısı: 4.230 (2.115 ham + 2.115 SFT).
  3. `bin_dekod_dogrula` 4/4 OK (0 sapma).
  4. SFT zarf sayısı == 2.115.
  Hüküm: **DERLEME_GECTI** · **DERLEME_SAPMA**.

## İLAN-3b — CARVE & TABAN ÇIPASI & EĞİTİM

- **Heldout carve:** Tohum 42, 211 heldout (D3 şeması `{"instruction", "input", "output"}`) + 1.904 eğitim adapter çifti.
- **Taban Berber heldout çıpası:** Heldout üzerinde `evaluate_carpenter_anka.py` koşumu ile ölçülür.
- **Fine-tune parametreleri:** Taban DONUK, LM bedeli kapısı 0,0230, tutarsızlık çıpa %5.
