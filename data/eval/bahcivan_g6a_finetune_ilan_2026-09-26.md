# G6a (T-0137) İLAN-3b — BAHÇİVAN FİNE-TUNE KOŞUMU — 2026-09-26 (claude)

ILAN ≠ RAPOR: rapor `bahcivan_g6a_finetune_sonuc_2026-09-26.md` (AYRI dosya);
ILAN_YOL ≠ RAPOR_YOL. Hükümler BETİKTEN, elle sayı YOK. Tasarım keşif
ölçümleri koşum ÖNCESİ beyanlıdır (blok-maske oranı, P3 kardeş kanıtları).

## 1. Taban ve checkpoint disiplini

- Taban `data/anka_base_v2.pt` (`d0f415f3…`, mühürlü 2.0-sealed) **DONUK
  okuma** — `data/*.pt` yazım YOK; koşum çıktısı `scratch/t0137_g6a_kos/`
  (kiralanmış); checkpoint silme OPERATÖR KAPISI (doğrulama sonrası).
- Taban-çıpa zinciri: marangoz heldout ROUGE çıpa **0,1392** (operatör onayı
  26 Eyl; `taban_cipa_sonda.json`), kapı **0,0230** (2× ROUGE SE).

## 2. Carve (Bahçıvan heldout — koşum ÖNCESİ damga)

- **Oran %10 (220/2.199 çift), tohum 42, deterministik** (`np.random.default_rng(42).permutation`).
  Kardeş ölçüm: marangoz carve 539/5.400 = %9,98 (`anka_r17_heldout`, `c397eb08…` DOKUNULMAZ).
- `scratch/t0137/bahcivan_carve.py` → `bahcivan_heldout.jsonl` (220 satır,
  **D3 şeması**: `{"instruction": ZARF, "input": <soru>, "output": <cevap>}`)
  + 1.979 satırlık eğitim adapter'ı; **AYRAÇ: heldout ∩ eğitim kesişim 0**
  (tam-satır küme) · maske-oranı ölçümü (400-pencere örneklem, `mask_prompt_targets`
  + PAD→−100 gerçek kodla; blok 128 beklenen ≈ %55,28 — tasarımda ölçüldü, İLAN
  damgası koşum ÖNCESİ). Dallar: **CARVE_GECTİ** / **CARVE_SAPMA** (rc=2).
  - *Ölçüm-sonrası şema düzeltmesi (2026-09-26, dürüst kayıt):* İLK carve
    koşumu heldout'u **ham arena satırı** yazdı; kanonik kap heldout'tan
    `r["output"]` ZORUNLU okur (`evaluate_carpenter_anka.py:463`) ⇒ ÖN-3
    `KeyError: 'output'` ile DURDU. Düzeltme: heldout kap şemasına çevrildi
    (marangoz heldout kardeş şeması `{"instruction","input","output"}` birebir;
    üretim istemi eğitim SFT kaydıyla özdeş — Exp C dersi). **Eğitim adapter'ı
    DEĞİŞMEDİ** (sha `7c2b8578…` birebir korundu) ⇒ `bahcivan_egitim.bin`
    değişmez (sha `f15603d5…`); carve + derle_carve hükümleri yeniden koşumdan
    (heldout sha `edf44331…` → `36416b82…`; kesişim 0 korundu).
- Yeniden derleme `scratch/t0137/bahcivan_derle_carve.py` — İLAN-3 derleme
  sözleşmesiyle AYNI (kanonik `tokenize_specialization_jsonl` IMPORT; zarf
  "Bahçıvan uzmanı olarak cevapla."; tokenizer r1 33114 + roots.tsv +
  literal_entity_mode=True). Çıktı `scratch/t0137/bahcivan_egitim.bin`.
  AYRAÇ: kayıt ≤ 3.958 (2×1.979) · SFT zarf 1.979 · `bin_dekod_dogrula` 0 sapma ·
  sozluk_giris 33114. Dallar: **DERLEME_CARVE_GECTİ** / **_SAPMA** (rc=2).

## 3. Bahçıvan taban-çıpa (pozitif kontrol — ölçüt canlılığı)

Kanonik kap (`scripts/evaluate_carpenter_anka.py` DOKUNULMAZ), `--model
data/anka_base_v2.pt --heldout scratch/t0137/bahcivan_heldout.jsonl
--train-source scratch/t0137/bahcivan_egitim_adapter.jsonl --ceket-ekseni --output
scratch/t0137/bahcivan_taban_cipa.json --device mps` (n=100, seed 42; sandbox DIŞI).
- *`--train-source` düzeltmesi (şema düzeltmesiyle birlikte):* eğitimde
  kullanılan gerçek satırlar carve'lı adapter'dır (D3 şeması — `output` alanlı);
  arena ham satırları `output` alanı taşımadığından kesişim ölçümü şema
  uyuşmazlığıyla yapay 0 üretirdi. Kesişim ölçümü adapter üzerinden BETİKTEN.
- **AYRAÇ:** taban Bahçıvan-heldout `ROUGE-L < 0,005` → **BAHÇIVAN_CIPA_GECTİ**
  (çıpa = bu ölçüm; boş-model kardeş kanıtı 0,0008) · `≥ 0,005` →
  **BAHÇIVAN_CIPA_DUSTU** (kök neden ölçülür; çıpa OPERATÖR onayına gider —
  TABAN_CIPA_DUSTU düzeltme deseni).
- Taban marangoz içeriği görmedi (Bahçıvan kalıpları/cevaplar eğitimsiz).
- **ÖLÇÜM SONUCU (2026-09-26, koşum sonrası — BAHÇIVAN_CIPA_DUSTU):** taban
  ROUGE-L **0,0236** (n=100; medyan **0,0**, DECOMP 0,0325; sonda
  `bahcivan_taban_cipa.json` sha `4960e835…`). Ayraç DÜŞTÜ — çıpa OPERATÖR
  onayı BEKLEMEDE, **eğitimci koşumu BAŞLATILMAZ**. Kök neden (betikten
  `bahcivan_taban_cipa_tani.json`): taban üretimi **marangoz/ceket kalıbı
  taşır** — `ağa DERIV_CI` 99/100, `teknik çözüm` başlangıcı 94/100,
  `<PROPER_NOUN>` yer-tutucu 97/100; Bahçıvan içerik kesişimi ~0 (LCS-F1
  0,0004, kesişim %0,00, ezber %0,00, tutarsızlık %61). ROUGE 0,0236 ortalamayı
  az sayıda tesadüfi kelime kesişimi çekiyor (medyan 0,0) — taban devralınmış
  seg_3/ceket zincirinden marangoz biçimini kopyalıyor, Bahçıvan içeriğini
  GÖRMEDİ (FAZ-0 seg_3-devralma bulgusuyla tutarlı; boş-model kardeş 0,0008
  ile aynı sınıfta: ölçüt ayırt ediyor).
  - **Önerilen çıpa (operatör onayı bekliyor):** A9 ARTİS çıpası = base_v2'nin
    KENDİ Bahçıvan-heldout ölçümü **0,0236** (TABAN_CIPA_DUSTU'da çıpa = base_v2
    kendi ROUGE 0,1392 kararının aynısı); kapı 0,0230 aynı kalır →
    `ROUGE_ft − 0,0236 ≥ +0,0230`. Onay gelmeden koşum başlamaz.
- **OPERATÖR ONAYI GELDİ (2026-09-26T17:01Z):** "çıpa 0,0236 onaylıyorum, koşumu
  başlat" — A9 çıpası **0,0236** SABİTLENDİ (kapı 0,0230 değişmez);
  `bahcivan_taban_cipa_tani.json` (0,0236) çıpa kanıtı. Eğitimci koşumu bu
  damgadan SONRA başlatıldı (koşum öncesi süreç kanıtı: EGITICI_YOK,
  2026-09-26T17:01:10Z sandbox dışı pgrep).
- **Koşum başlangıcı (2026-09-26T17:01Z, task `b2dsukx25`):** sürücü başlangıç
  logu — `[VOCAB-UYUM] 3 bin meta sozluk_sha256 == f9940a8d… (33.114 giriş)` ·
  `[H1] tek eğitici: 0 tutucu` · `[carry] .opt.pt YOK — optimizer SIFIRDAN` ·
  `[veri] wiki 781.250 · ceket 12.250 (1.568.123 jeton) · sft 2.807 pencere` ·
  `[referans] TEPE referansı 0.1392` · lsof kanıtı (PID 22209): üç bin açık
  (`anka_a1r_pretrain.bin` + `train_carpenter_specialization_anka_r18.bin` +
  `scratch/t0137/bahcivan_egitim.bin`), MPS metallib yüklü.
- **L1 dış izleme (canlı):** ilk adım-kayıp satırı log'dan betikle okunur —
  0,0 → DUR alarm · ≥ 4,736 → LNV_IMZASI alarm · (0, 4,736) → sağlıklı (beyanlı
  bant 2,0-3,5). segments.jsonl ilk kayıt ayrıca betikle okunur.
  - **ÖLÇÜM (adım-500, 2026-09-26):** `adım 500/6000 | kayıp 3.3254 | LR
    0.000099 | 207s` — L1 bandı **(0, 4,736) İÇİNDE**, beyanlı bant (2,0-3,5)
    ✓; adım süresi **0,414 sn/adım** (beyanlı 0,506 — koşum hızlı; süre
    beyanı koşum gerçek ölçümüyle güncellenecek). L1 alarm YOK.

## 4. Eğitimci koşumu (koşum ÖNCESİ damga; P3 sürücü DOKUNULMAZ)

- **Mix (P3 üç-kaynaklı desen):** wiki %20 (`data/anka_a1r_pretrain.bin`) +
  ceket %40 (`data/train_carpenter_specialization_anka_r18.bin`) + Bahçıvan
  %40 (`scratch/t0137/bahcivan_egitim.bin`) — üç bin de r1 33114 (`f9940a8d…`,
  VOCAB-UYUM). Kanıt: P3 seg_3 ROUGE 0,0056→0,1014, CE bedel +%1,43;
  yalnız-SFT kolunda tutarsızlık %100 / zarf öğrenilmedi (`anka_i1_lr_sonuc` §4).
- **Eğitimci parametreleri:** blok **128** · batch 8 · lr **1e-4** (2e-4 resume
  varsayılanı EZİLİR — T-0097 kökü) · warmup 50 · cosine (toplam 6.000) ·
  min_lr 1e-6 · clip 1,0 · **3 segment × 2.000 adım** (≈ 6,8 Bahçıvan epoch,
  betikten) · segment başına checkpoint + sonda.
- **Canlılık kapısı (dış izleme — sürücü DOKUNULMAZ):** `segments.jsonl` ilk
  kayıp betikle okunur; **L1_ESİK = ln V × maske-dışı-oran − 1,0 ≈ 4,75**
  (ln V = 10,408; oran koşum-öncesi ölçüm — §2). Dallar: kayıp **0,0 → DUR**
  (MPS no-op) · **≥ 4,75 → LNV_IMZASI DUR** · (0, 4,75) → sağlıklı (beyanlı
  bant 2,0-3,5 — sürücü karma parti: ~2 unmasked + ~6 maskeli pencere).
  - *Ölçüm SONRASI düzeltme (koşum ÖNCESİ, 2026-09-26):* ÖN-2 betik ölçümü
    (400 pencere, tohum 42, blok 128) maske-dışı oranı **%55,11** verdi
    (tasarım keşfi %55,28 — farklı pencere kümesi; İLAN formülü sabittir).
    Betikten: L1_ESİK = 10,4076 × 0,5511 − 1,0 = **4,736** (hükümde bu değer
    BETİKTEN yazılır; boş pencere 0/400 ayracı GEÇTİ).
- **CARRY beyanı:** base_v2 sidecar YOK ⇒ optimizer sıfırdan (Dal-3; ölçülmüş
  1,73× ilk-güncelleme şişmesi, T-0092).
- **Kapılar (sürücü betikten):** TEPE — sonda ROUGE −0,05 düşüş veya ezber
  ≥ %10 ⇒ DUR (referans 0,1392; tepe segment nihai) · CE tavan 3,8887 ·
  H1 tek eğitici + lsof kanıtı · VOCAB-UYUM üç binde fail-closed.
- **Süre beyanı (boyut kontrolü):** 0,506 sn/adım (P3 ölçümü — aynı
  33.114/768/blok128/batch8) × 6.000 ≈ 51 dk + 3 sonda; toplam ~2-3 saat.

## 5. Ölçüm ayracı seti (hüküm BETİKTEN — kanonik kap sonda JSON'ları)

| # | Ayrac | Ölçüm | Dallar |
|---|---|---|---|
| A9 | Bahçıvan ARTİS ≥ +0,0230 (çıpa §3 ölçümü; 2×SE — §3 ölçek mantığı) | ÖLÇÜM-2 | YETENEK_SİNYAL / BAND_ALTINDA (kabul yok) |
| A10 | LM-bedeli: \|f\| ≤ 0,0230 (f = ft ROUGE − 0,1392, marangoz heldout) | ÖLÇÜM-1 | KORUNDU / KALİTE_BEDELİ (kabul yok) |
| A11 | A-ekseni CE artış ≤ +%10 (tavan 3,8887; çözünürlük kapısı fail-closed) | ÖLÇÜM-1 hukum.A_gec | A_GECTİ / A_DUSTU |
| A12 | B top-1 düşüş ≤ 5,0 (Wilson) | ÖLÇÜM-1 hukum.B_gec | B_GECTİ / B_DUSTU |
| A13 | tutarsızlık < %5 · ezber < %10 | ÖLÇÜM-1/2 | GECTİ / İHLAL |

**Kabul birleşimi:** A9 VE A10 VE A11 VE A12 VE A13. TEPE'de durursa tepe
segment ölçülür; hüküm o checkpoint'e aittir.

- **ÖLÇÜM-1:** `--model seg_3.pt --baseline data/anka_base_v2.pt --ceket-ekseni
  --output scratch/t0137/bahcivan_ft_marangoz_sonda.json --device mps` (nötr —
  `--rol-zarf` VERİLMEZ; çıpa karıştırma yasağı ceket §4).
- **ÖLÇÜM-2:** aynı kap, `--heldout bahcivan_heldout.jsonl --train-source
  scratch/t0137/bahcivan_egitim_adapter.jsonl` (§3 şema/`--train-source`
  düzeltmesi birebir).
- **Beyanlı hükümsüz bant:** Bahçıvan heldout ROUGE beklenen 0,10-0,30
  (rampa kanıtı: 1,6 epoch → 0,1014; ~16 epoch → 0,30 bandı). ESIK_ROUGE
  0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ; Bahçıvan hedefi 0,3221'e
  BAĞLANMAZ (insan-tavan kalibrasyonu marangoz içindir; ayrac çıpa-farkıyla).

## 6. Kapanış

Checkpoint `scratch/t0137_g6a_kos/` doğrulama sonrası OPERATÖR kapısıyla silinir;
digest tablo rapor dondurulur. Commit `git add -A` YASAK — dosya listesiyle
operatör kapısı. Rapor ayrı dosya; kiralamalar T-0137.

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · kanonik kod IMPORT
(kopya YASAK) · `anka_p3_surucu.py` / `evaluate_carpenter_anka.py` / tokenizer /
compiler DOKUNULMAZ · data/** salt-okunur (tek istisna data/eval/) · elle sayı YOK.