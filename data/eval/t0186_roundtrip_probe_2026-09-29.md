# MİMARİ DOĞRULAMA PAKET-1 SONUÇ — Gidiş-Dönüş Doğrulaması (T-0141)

**Hüküm:** **DUR** (betikten; elle sayı YOK)
**Damga:** 2026-09-29T05:05:09Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md` (koşum ÖNCESİ; sha256 `cfa4134011413f79f8b95d4e55fe9cf938b1d9fc488865ed26be9c6859958bab`)

## 1. Hüküm kapıları (betikten)

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| FAZ-B: İLAN-3 birebir-çıpa — bit_esit==9182 / ornek==9188 + kalan-6 (YOL_0 5 + BIT_UYUMSUZ 1) birebir-dislamalı | `{"ornek": 9188, "bit_esit": 9186, "dusen": {"BIT_UYUMSUZ": 1, "YOL_0": 1}, "dislanali_disi": {}, "kalan_6_uyum": false}` | DÜŞTÜ |
| FAZ-A: compile istisna FIRLATMAZ (istisna == 0; ilanlı davranış) | `0` | GEÇTİ |
| KSUR-1: NOUN-satırı-yok ADJ == 3.627 / 6.352 ADJ + probe 44/50 0-yol (İLAN-2: 6 yol-bulunan meşru-alternatif-kok dislamalı) | `{"adj_noun_yok": 3626, "adj_toplam": 6352, "probe_yol0": 44, "probe_n": 50}` | DÜŞTÜ |
| KSUR-2: kesme-apostrof davranışı (probe %100 kesmeli + placeholder) | `{"kesmeli": 20, "n": 20, "placeholder": "[Özel İsim]'e"}` | GEÇTİ |
| KSUR-3: İ/I düz-lower — satır == 79 (72 İ+7 I) VE benzersiz == 70 (63+7) | `{"satir": 90, "benzersiz": 81}` | DÜŞTÜ |
| KSUR-4: OOV → 200/200 sessiz-boş (istisna/yol YOK) | `{"sessiz_bos": 200, "beklenmedik": []}` | GEÇTİ |

## 2. FAZ-B — Gidiş-dönüş örneklemi (birincil)

- Hedef/örnek: 9988 / **9188** (çıkmaz-atılan 800)
- **Yüzey-bit-özdeşlik: 9186/9188 = 0.999782** (İLAN-3 birebir-çıpa: bit_esit==9182, kalan-6 dislamalı)
- Düşen sınıfları: `{"BIT_UYUMSUZ": 1, "YOL_0": 1}`
- Düşen örneklerin son-ek kırılımı (kök-neden kaydı): `{"COPULA_COND": 1, "VOICE_PASS_Il": 1}`
- BIT_UYUMSUZ örneklerinde geri-yüzeyde apostrof: **0** (özel-ad ikiz-satır deseni; kök neden §7)
- İkincil kayıt (hükümde DEĞİL): `token_vector==tags` 0.9348 · `best_surface` eşliği 1.0000 · ambigüite payı 0.3125
- Süre: 2.9 sn (saf CPU)
- Düşen ilk örnekler:

| lemma · tags · yüzey · geri | sınıf |
|---|---|
| `{"lemma": "çırağ", "tags": ["çırağ", "COPULA_COND"], "yuzey": "çırağsa", "geri": "çıraksa", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "uç", "tags": ["uç", "VOICE_PASS_Il"], "yuzey": "ucul", "geri": null, "hata": null}` | `YOL_0` |

## 3. FAZ-A — Tam lemma taraması (envanter)

- Benzersiz lemma: **48407**, istisna: **0**, süre 10.23 sn

| POS | toplam | sessiz-boş (analyses==[]) | istisna |
|---|---|---|---|
| ADJ | 6352 | 0 | 0 |
| ADV | 1557 | 0 | 0 |
| CONJ | 39 | 0 | 0 |
| INTERJ | 231 | 0 | 0 |
| NOUN | 30235 | 0 | 0 |
| POSTP | 19 | 0 | 0 |
| PRON | 47 | 0 | 0 |
| VERB | 9927 | 0 | 0 |

## 4. FAZ-C — Bilinen-ksur envanteri (İLAN'lı sayılar)

| Sınıf | Ölçülen | İLAN'lı |
|---|---|---|
| KSUR-1 NOUN-satırı-yok ADJ | **3626** / ADJ toplam 6352 | 3.627 / 6.352 |
| KSUR-1 yalnız-ADJ (rapor) | 3281 (küçük-harf havuz 3249) | 3.282 / 3.249 |
| KSUR-1 probe 0-yol | 44/50 | 44/50 (İLAN-2: 6 meşru-alternatif-kok) |
| KSUR-2 kesmeli probe | 20/20 (havuz 1907) | %100 |
| KSUR-3 İ/I satır | **90** | 79 (72 İ+7 I) |
| KSUR-3 İ/I benzersiz | **81** | 70 (63 İ+7 I) |
| KSUR-4 OOV sessiz-boş | 200 (beklenmedik 0) | 200/200 |

## 5. Kapsam-dışı beyanı (İLAN §2)

FAZ-B örneklem havuzu dışı lemma'lar (büyük-harfli/apostroflu/çok-sözcüklü) bit-özdeşlik ölçütüne SOKULMAMIŞTIR; bu sınıflar KSUR-2/3 probe'larında ve FAZ-A sessiz-boş kırılımında envanterlenir. KSUR-1 probe havuzu: küçük-harf yalnız-ADJ (İLAN §6 revizyonu).

## 6. Tam SHA-256 digest tablosu

| Dosya | SHA-256 |
|---|---|
| `data/lexicon/roots_anka_r1.tsv` | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| `data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md` | `cfa4134011413f79f8b95d4e55fe9cf938b1d9fc488865ed26be9c6859958bab` |
| `scripts/dogrulama_p1_roundtrip.py` | `94523084010b6509ae9918d4543a047a8513191616c3e8806992ab0a14ec149c` |
| `data/eval/t0186_roundtrip_probe_hukum_2026-09-29.json` | `d8a433399207819c276544632b794bd1749a3f4ddc1330998e542378268fcd0b` |

