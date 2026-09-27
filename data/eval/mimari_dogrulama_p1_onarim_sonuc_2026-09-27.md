# MİMARİ DOĞRULAMA PAKET-1 SONUÇ — Gidiş-Dönüş Doğrulaması (T-0141)

**Hüküm:** **DUR** (betikten; elle sayı YOK)
**Damga:** 2026-09-27T14:17:04Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/mimari_dogrulama_p1_ilan2_2026-09-27.md` (koşum ÖNCESİ; sha256 `cf3bbdceeae6f949f5847b31e9e4567900e3c9c5d1ff7779d1ad99d68e04bd81`)

## 1. Hüküm kapıları (betikten)

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| FAZ-B: yüzey-bit-özdeşlik %100 (0 düşen örnek, çekimlenebilir sınıf) | `{"ornek": 9188, "bit_esit": 9182, "dusen": {"YOL_0": 5, "BIT_UYUMSUZ": 1}}` | DÜŞTÜ |
| FAZ-A: compile istisna FIRLATMAZ (istisna == 0; ilanlı davranış) | `0` | GEÇTİ |
| KSUR-1: NOUN-satırı-yok ADJ == 3.627 / 6.352 ADJ + probe 44/50 0-yol (İLAN-2: 6 yol-bulunan meşru-alternatif-kok dislamalı) | `{"adj_noun_yok": 3627, "adj_toplam": 6352, "probe_yol0": 44, "probe_n": 50}` | GEÇTİ |
| KSUR-2: kesme-apostrof davranışı (probe %100 kesmeli + placeholder) | `{"kesmeli": 20, "n": 20, "placeholder": "[Özel İsim]'e"}` | GEÇTİ |
| KSUR-3: İ/I düz-lower — satır == 79 (72 İ+7 I) VE benzersiz == 70 (63+7) | `{"satir": 79, "benzersiz": 70}` | GEÇTİ |
| KSUR-4: OOV → 200/200 sessiz-boş (istisna/yol YOK) | `{"sessiz_bos": 200, "beklenmedik": []}` | GEÇTİ |

## 2. FAZ-B — Gidiş-dönüş örneklemi (birincil)

- Hedef/örnek: 9988 / **9188** (çıkmaz-atılan 800)
- **Yüzey-bit-özdeşlik: 9182/9188 = 0.999347** (birincil hüküm: %100)
- Düşen sınıfları: `{"YOL_0": 5, "BIT_UYUMSUZ": 1}`
- Düşen örneklerin son-ek kırılımı (kök-neden kaydı): `{"CASE_GEN": 3, "POSS_1SG": 1, "POSS_1PL": 1, "TENSE_AORIST_VOWEL": 1}`
- BIT_UYUMSUZ örneklerinde geri-yüzeyde apostrof: **1** (özel-ad ikiz-satır deseni; kök neden §7)
- İkincil kayıt (hükümde DEĞİL): `token_vector==tags` 0.9329 · `best_surface` eşliği 1.0000 · ambigüite payı 0.3128
- Süre: 2.69 sn (saf CPU)
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

- Benzersiz lemma: **48200**, istisna: **0**, süre 9.5 sn

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
| `data/eval/mimari_dogrulama_p1_ilan2_2026-09-27.md` | `cf3bbdceeae6f949f5847b31e9e4567900e3c9c5d1ff7779d1ad99d68e04bd81` |
| `scripts/dogrulama_p1_roundtrip.py` | `90cb82fcb4c1471895c4913c95d6ee72aca047236a85918e7198bbf8c7c852ab` |
| `data/eval/mimari_dogrulama_p1_onarim_hukum_2026-09-27.json` | `6f471862154b9c21a6e0a621f22f0f1cfc439b7d60534b12f1bdfad072977aef` |

## 7. Koşum-sonrası kök-neden: kalan-6 örneğin sınıf-beyanı (hüküm-disi)

**Onarım-etkisi (koşum-1 → koşum-2 kırılım-karşılaştırması):**

| Sınıf | Koşum-1 | Koşum-2 | Yorum |
|---|---|---|---|
| GERUND_KEN | 154 | **0** | A-1 çok-şablonlu affix_info — sınıf %100 boşaldı |
| GERUND_IncA | 1 | 0 | (A-1) |
| BIT_UYUMSUZ (apostrof) | 58 | **0** | A-2 ikiz-satır `is_case_alias` — apostrof-sınıfı boşaldı |
| CASE_GEN | 3 | 3 | üçüncü-sınıf (aşağıda) |
| POSS_1SG | 3 | 1 | 2 onarıldı; 1 üçüncü-sınıf |
| POSS_1PL | 2 | 1 | 1 onarıldı; 1 üçüncü-sınıf |
| TENSE_AORIST_VOWEL | 1 | 1 | üçüncü-sınıf (aynı örnek) |
| **toplam düşen** | **215** | **6** | **209 onarıldı** |

**Kalan-6 iki alt-sınıf — koşum-1 kırılım-satırında zaten düşen sınıflar; YENİ-kusur DEĞİL
("iki sayı çelişiyor sanma önce küme" dersi):**

1. **`bil` → `Bi'ler` (TENSE_AORIST_VOWEL 1):** TSV'de `Bi` (büyük-harfli lemma) ve
   `bil` (VERB) **iki ayrı GERÇEK lemma** — düz-lower ikiz-KOPYASI DEĞİL; A-2
   `is_case_alias` bayrağı yalnız aynı-lemma'nın büyük-harfli-ikiz-kopyasını
   kapsar, bu tipi kapsamaz (kapsaması = farklı lemmaların birleştirilmesi →
   kapsam-genişletme sorusu). Compile `biler` best-path'i büyük-harfli `Bi`+AORIST
   satırından seçer → decompile apostrof-dalı gerçek-öz-ad davranışını korur.
2. **zeyrek/meçhul/nakil (CASE_GEN 3) + güç (POSS_1SG) + hacir (POSS_1PL) → geri
   null (YOL_0 5):** decompile yüzeyi DOĞRU üretir (`naklin` ✓); compile geri-yolda
   0-yol — VOWEL_DROP / kök-seçim derin-mekanizması, affix_info-şablon katmanının
   ÖTESİNDE. Bu sınıflardan koşum-1'de 9 örnek zaten düşüyordu (kırılım-satırı
   birebir); A-1 onarımı bu sınıflara DOKUNMADI (İLAN-2 kapsamı: GERUND_KEN +
   ikiz-satır) — 6 kalıntı **üçüncü-sınıf, onarım-kapsamı-dışı** beyanı.

**Hüküm-süreç notu:** İLAN-2 hedefi FAZ-B %100 idi; ölçülen 0,999347 (9182/9188) →
betik DUR rc=2 verdi (hüküm betikten; İLAN-2 yumuşatılmaz). Sonraki-ad seçenekleri
operatör kararıdır: (a) İLAN-3 birebir-çıpa 9182/9188 (üçüncü-sınıf dislamalı) +
P2/P3'teki GECTİ-kalıbıyla Tur-A'yı kapatmak; (b) onarım-genişletme (derin-mekanizma
— ayrı İLAN'lı görev); (c) üçüncü-sınıf kapsam-dışı beyanıyla İLAN revizyonu.
Elle sayı/hüküm YOK; karar kaydı operatördendir.

