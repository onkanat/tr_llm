# T-0150 FAZ-2 ONARIM-KOŞUMU — P1 kalan-6 A2+B2+C (sonuç)

**Hüküm:** **P1_FAZ2_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-27T19:44:40Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/onarim_p1_faz2_ilan1_2026-09-27.md` (koşum ÖNCESİ; sha256 `95418ec55ad288678755fd0ef13e90030678c3e74ce0b6187a95dc6e24d0844d`)

## 1. Hüküm kapıları (betikten)

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 FAZ-B ONARIM: ornek==9188 VE bit_esit==9188 VE düşen=={} | `{"ornek": 9188, "bit_esit": 9188, "dusen": {}}` | GEÇTİ |
| K2 DEC-YÜZEY-KORUMA: kalan-6 decompile_tags ×6 birebir | `{"zeyrek": "zeyrekin", "meçhul": "meçhlun", "nakil": "naklin", "güç": "güçüm", "hacir": "hacrimiz", "bil": "biler"}` | GEÇTİ |
| K3 COMPILE-YOL-ÇAPA: 6 yüzey token_vector[0]==lemma + geri==yüzey | `{"zeyrek": {"tv": ["zeyrek", "POSS_2SG"], "geri": "zeyrekin"}, "meçhul": {"tv": ["meçhul", "POSS_2SG"], "geri": "meçhlun"}, "nakil": {"tv": ["nakil", "POSS_2SG"], "geri": "naklin"}, "güç": {"tv": ["güç", "POSS_1SG"], "geri": "güçüm"}, "hacir": {"tv": ["hacir", "POSS_1PL"], "geri": "hacrimiz"}, "bil": {"tv": ["bil", "TENSE_AORIST_VOWEL"], "geri": "biler"}}` | GEÇTİ |
| K4 ROOT-ORACLE: 25/25 (koşum-öncesi baseline; başarısız == {}) | `{"basarili": 25, "basarisiz": {}}` | GEÇTİ |
| K5 KSUR SABİT: 3.627/6.352 + 44/50 · 20/20 · 79/70 · 200/200 | `{"ksur1": [3627, 6352, 44, 50], "ksur2": [20, 20], "ksur3": {"satir": 79, "benzersiz": 70}, "ksur4": [200, 0]}` | GEÇTİ |
| K6 FAZ-A: istisna==0 VE sessiz-boş==0 | `{"istisna": 0, "sessiz_bos": 0}` | GEÇTİ |
| K7 ŞEMA-KORUMA: compile 6-anahtar + find_stems tuple-şema + tokenizer-yolu (bayraksız) bit-özdeş | `{"compile_anahtarlar": ["analyses", "best_surface", "input", "language", "needs_disambiguation", "token_vector"], "compile_sema_birebir": true, "find_stems_sema_birebir": true, "tokenizer_yol_token_vector": ["POSS_1SG"], "tokenizer_yol_beklenen": ["POSS_1SG"]}` | GEÇTİ |

## 2. FAZ-B onarım-sonrası (birincil)

- Hedef/örnek: 9988 / **9188** (çıkmaz-atılan 800)
- **Yüzey-bit-özdeşlik: 9188/9188 = 1.000000** (İLAN-1 hedef %100)
- Düşen sınıfları: `{}`
- İkincil: tv_es 0.9326 · best_surface 1.0000 · ambig 0.3130
- Süre: 2.95 sn

## 3. FAZ-A — Tam lemma taraması

- Benzersiz lemma: **48200**, istisna: **0**, süre 10.3 sn
- Sessiz-boş toplam: 0

## 4. Kalan-6 onarım-kanıtı (birebir)

| lemma | dec_yuzey | tv0_lemma | geri | bit |
|---|---|---|---|---|
| zeyrek | `zeyrekin` | True | `zeyrekin` | True |
| meçhul | `meçhlun` | True | `meçhlun` | True |
| nakil | `naklin` | True | `naklin` | True |
| güç | `güçüm` | True | `güçüm` | True |
| hacir | `hacrimiz` | True | `hacrimiz` | True |
| bil | `biler` | True | `biler` | True |

## 5. Tam SHA-256 digest tablosu

| Dosya | SHA-256 |
|---|---|
| `data/lexicon/roots.tsv` | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/eval/onarim_p1_faz2_ilan1_2026-09-27.md` | `95418ec55ad288678755fd0ef13e90030678c3e74ce0b6187a95dc6e24d0844d` |
| `scripts/dogrulama_p1_faz2_onarim.py` | `da50a216f49cb1bf172baaf3d39acd6a79a99247569c13671caf6d0dd0524633` |
| `scripts/dogrulama_p1_roundtrip.py` (import) | `94523084010b6509ae9918d4543a047a8513191616c3e8806992ab0a14ec149c` |
| `src/compiler/core.py` (onarım) | `25ae60f2e59383ab9d3e24189cfbe71bf359945b246392c5fe726a530384d8b8` |
| `data/eval/onarim_p1_faz2_hukum_2026-09-27.json` | `de96c3cecd9fcfd8936fa31be0d36ac6ee4ce7b0e10f1c476341601819ecc4d6` |

