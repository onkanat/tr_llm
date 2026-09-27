# MİMARİ DOĞRULAMA PAKET-1 SONUÇ — Gidiş-Dönüş Doğrulaması (T-0141)

**Hüküm:** **P1_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-27T16:58:30Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/mimari_dogrulama_p1_ilan3_2026-09-27.md` (koşum ÖNCESİ; sha256 `28e7376b67a0d1919ef5018177a69b201596c2074ff20cb93745146e1d5e4199`)

## 1. Hüküm kapıları (betikten)

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| FAZ-B: İLAN-3 birebir-çıpa — bit_esit==9182 / ornek==9188 + kalan-6 (YOL_0 5 + BIT_UYUMSUZ 1) birebir-dislamalı | `{"ornek": 9188, "bit_esit": 9182, "dusen": {"YOL_0": 5, "BIT_UYUMSUZ": 1}, "dislanali_disi": {}, "kalan_6_uyum": true}` | GEÇTİ |
| FAZ-A: compile istisna FIRLATMAZ (istisna == 0; ilanlı davranış) | `0` | GEÇTİ |
| KSUR-1: NOUN-satırı-yok ADJ == 3.627 / 6.352 ADJ + probe 44/50 0-yol (İLAN-2: 6 yol-bulunan meşru-alternatif-kok dislamalı) | `{"adj_noun_yok": 3627, "adj_toplam": 6352, "probe_yol0": 44, "probe_n": 50}` | GEÇTİ |
| KSUR-2: kesme-apostrof davranışı (probe %100 kesmeli + placeholder) | `{"kesmeli": 20, "n": 20, "placeholder": "[Özel İsim]'e"}` | GEÇTİ |
| KSUR-3: İ/I düz-lower — satır == 79 (72 İ+7 I) VE benzersiz == 70 (63+7) | `{"satir": 79, "benzersiz": 70}` | GEÇTİ |
| KSUR-4: OOV → 200/200 sessiz-boş (istisna/yol YOK) | `{"sessiz_bos": 200, "beklenmedik": []}` | GEÇTİ |

## 2. FAZ-B — Gidiş-dönüş örneklemi (birincil)

- Hedef/örnek: 9988 / **9188** (çıkmaz-atılan 800)
- **Yüzey-bit-özdeşlik: 9182/9188 = 0.999347** (İLAN-3 birebir-çıpa: bit_esit==9182, kalan-6 dislamalı)
- Düşen sınıfları: `{"YOL_0": 5, "BIT_UYUMSUZ": 1}`
- Düşen örneklerin son-ek kırılımı (kök-neden kaydı): `{"CASE_GEN": 3, "POSS_1SG": 1, "POSS_1PL": 1, "TENSE_AORIST_VOWEL": 1}`
- BIT_UYUMSUZ örneklerinde geri-yüzeyde apostrof: **1** (özel-ad ikiz-satır deseni; kök neden §7)
- İkincil kayıt (hükümde DEĞİL): `token_vector==tags` 0.9329 · `best_surface` eşliği 1.0000 · ambigüite payı 0.3128
- Süre: 2.71 sn (saf CPU)
- Düşen ilk örnekler:

| lemma · tags · yüzey · geri | sınıf |
|---|---|
| `{"lemma": "zeyrek", "tags": ["zeyrek", "CASE_GEN"], "yuzey": "zeyrekin", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "meçhul", "tags": ["meçhul", "CASE_GEN"], "yuzey": "meçhlun", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "nakil", "tags": ["nakil", "CASE_GEN"], "yuzey": "naklin", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "güç", "tags": ["güç", "POSS_1SG"], "yuzey": "güçüm", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "hacir", "tags": ["hacir", "POSS_1PL"], "yuzey": "hacrimiz", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "bil", "tags": ["bil", "TENSE_AORIST_VOWEL"], "yuzey": "biler", "geri": "Bi'ler", "hata": null}` | `BIT_UYUMSUZ` |

## 3. FAZ-A — Tam lemma taraması (envanter)

- Benzersiz lemma: **48200**, istisna: **0**, süre 9.61 sn

| POS | toplam | sessiz-boş (analyses==[]) | istisna |
|---|---|---|---|
| ADJ | 6352 | 0 | 0 |
| ADV | 1557 | 0 | 0 |
| CONJ | 39 | 0 | 0 |
| INTERJ | 231 | 0 | 0 |
| NOUN | 30028 | 0 | 0 |
| POSTP | 19 | 0 | 0 |
| PRON | 47 | 0 | 0 |
| VERB | 9927 | 0 | 0 |

## 4. FAZ-C — Bilinen-ksur envanteri (İLAN'lı sayılar)

| Sınıf | Ölçülen | İLAN'lı |
|---|---|---|
| KSUR-1 NOUN-satırı-yok ADJ | **3627** / ADJ toplam 6352 | 3.627 / 6.352 |
| KSUR-1 yalnız-ADJ (rapor) | 3282 (küçük-harf havuz 3249) | 3.282 / 3.249 |
| KSUR-1 probe 0-yol | 44/50 | 44/50 (İLAN-2: 6 meşru-alternatif-kok) |
| KSUR-2 kesmeli probe | 20/20 (havuz 1711) | %100 |
| KSUR-3 İ/I satır | **79** | 79 (72 İ+7 I) |
| KSUR-3 İ/I benzersiz | **70** | 70 (63 İ+7 I) |
| KSUR-4 OOV sessiz-boş | 200 (beklenmedik 0) | 200/200 |

## 5. Kapsam-dışı beyanı (İLAN §2)

FAZ-B örneklem havuzu dışı lemma'lar (büyük-harfli/apostroflu/çok-sözcüklü) bit-özdeşlik ölçütüne SOKULMAMIŞTIR; bu sınıflar KSUR-2/3 probe'larında ve FAZ-A sessiz-boş kırılımında envanterlenir. KSUR-1 probe havuzu: küçük-harf yalnız-ADJ (İLAN §6 revizyonu).

## 6. Tam SHA-256 digest tablosu

| Dosya | SHA-256 |
|---|---|
| `data/lexicon/roots.tsv` | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/eval/mimari_dogrulama_p1_ilan3_2026-09-27.md` | `28e7376b67a0d1919ef5018177a69b201596c2074ff20cb93745146e1d5e4199` |
| `scripts/dogrulama_p1_roundtrip.py` | `94523084010b6509ae9918d4543a047a8513191616c3e8806992ab0a14ec149c` |
| `data/eval/mimari_dogrulama_p1_ilan3_kapanis_hukum_2026-09-27.json` | `6428b1802590ab460916f0cec21d7de6513f4b5a82451ac375e5f5ec1015aaca` |

