# MİMARİ DOĞRULAMA PAKET-1 SONUÇ — Gidiş-Dönüş Doğrulaması (T-0141)

**Hüküm:** **DUR** (betikten; elle sayı YOK)
**Damga:** 2026-09-27T11:07:24Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md` (koşum ÖNCESİ; sha256 `cfa4134011413f79f8b95d4e55fe9cf938b1d9fc488865ed26be9c6859958bab`)

## 1. Hüküm kapıları (betikten)

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| FAZ-B: yüzey-bit-özdeşlik %100 (0 düşen örnek, çekimlenebilir sınıf) | `{"ornek": 9188, "bit_esit": 8973, "dusen": {"YOL_0": 156, "BIT_UYUMSUZ": 59}}` | DÜŞTÜ |
| FAZ-A: compile istisna FIRLATMAZ (istisna == 0; ilanlı davranış) | `0` | GEÇTİ |
| KSUR-1: NOUN-satırı-yok ADJ == 3.627 / 6.352 ADJ + probe 50/50 0-yol | `{"adj_noun_yok": 3627, "adj_toplam": 6352, "probe_yol0": 44, "probe_n": 50}` | DÜŞTÜ |
| KSUR-2: kesme-apostrof davranışı (probe %100 kesmeli + placeholder) | `{"kesmeli": 20, "n": 20, "placeholder": "[Özel İsim]'e"}` | GEÇTİ |
| KSUR-3: İ/I düz-lower — satır == 79 (72 İ+7 I) VE benzersiz == 70 (63+7) | `{"satir": 79, "benzersiz": 70}` | GEÇTİ |
| KSUR-4: OOV → 200/200 sessiz-boş (istisna/yol YOK) | `{"sessiz_bos": 200, "beklenmedik": []}` | GEÇTİ |

## 2. FAZ-B — Gidiş-dönüş örneklemi (birincil)

- Hedef/örnek: 9988 / **9188** (çıkmaz-atılan 800)
- **Yüzey-bit-özdeşlik: 8973/9188 = 0.976600** (birincil hüküm: %100)
- Düşen sınıfları: `{"YOL_0": 156, "BIT_UYUMSUZ": 59}`
- Düşen örneklerin son-ek kırılımı (kök-neden kaydı): `{"GERUND_KEN": 154, "POSS_1SG": 3, "DERIV_lA": 3, "DERIV_lIk": 2, "CASE_ABL": 2, "COPULA_COND": 7, "DERIV_lAn": 5, "DERIV_lI": 2, "DERIV_CI": 2, "POSS_3SG": 4, "POSS_2SG": 2, "CASE_ACC": 1, "REL_ki": 2, "POSS_1PL": 2, "CASE_DAT": 1, "CASE_GEN": 3, "DERIV_sIz": 2, "COPULA_EVIDENTIAL": 7, "POSS_2PL": 1, "TENSE_EVIDENTIAL": 1, "TENSE_OPTATIVE": 1, "TENSE_AORIST": 1, "TENSE_AORIST_VOWEL": 1, "GERUND_IncA": 1, "COPULA_AORIST": 3, "DERIV_lAş": 2}`
- BIT_UYUMSUZ örneklerinde geri-yüzeyde apostrof: **59** (özel-ad ikiz-satır deseni; kök neden §7)
- İkincil kayıt (hükümde DEĞİL): `token_vector==tags` 0.9318 · `best_surface` eşliği 1.0000 · ambigüite payı 0.3110
- Süre: 2.58 sn (saf CPU)
- Düşen ilk örnekler:

| lemma · tags · yüzey · geri | sınıf |
|---|---|
| `{"lemma": "rica", "tags": ["rica", "GERUND_KEN"], "yuzey": "ricaken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "radyofizyoloji", "tags": ["radyofizyoloji", "GERUND_KEN"], "yuzey": "radyofizyolojiken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "doğrultma", "tags": ["doğrultma", "GERUND_KEN"], "yuzey": "doğrultmaken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "mahvetme", "tags": ["mahvetme", "GERUND_KEN"], "yuzey": "mahvetmeken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "düşürtme", "tags": ["düşürtme", "GERUND_KEN"], "yuzey": "düşürtmeken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "meşrutiyet", "tags": ["meşrutiyet", "POSS_1SG"], "yuzey": "meşrutiyetim", "geri": "Meşrutiyet'im", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "eşme", "tags": ["eşme", "DERIV_lA"], "yuzey": "eşmele", "geri": "Eşme'le", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "belen", "tags": ["belen", "DERIV_lIk"], "yuzey": "belenlik", "geri": "Belen'lik", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "köprübaşı", "tags": ["köprübaşı", "CASE_ABL"], "yuzey": "köprübaşıdan", "geri": "Köprübaşı'dan", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "haymana", "tags": ["haymana", "COPULA_COND"], "yuzey": "haymanaysa", "geri": "Haymana'ysa", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "kayıkçı", "tags": ["kayıkçı", "GERUND_KEN"], "yuzey": "kayıkçıken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "seçki", "tags": ["seçki", "GERUND_KEN"], "yuzey": "seçkiken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "gırgırlama", "tags": ["gırgırlama", "GERUND_KEN"], "yuzey": "gırgırlamaken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "menakıpname", "tags": ["menakıpname", "GERUND_KEN"], "yuzey": "menakıpnameken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "as", "tags": ["as", "CASE_ABL"], "yuzey": "astan", "geri": "As'tan", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "üslupçu", "tags": ["üslupçu", "GERUND_KEN"], "yuzey": "üslupçuken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "sorgun", "tags": ["sorgun", "DERIV_lA"], "yuzey": "sorgunla", "geri": "Sorgun'la", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "amme", "tags": ["amme", "DERIV_lAn"], "yuzey": "ammelen", "geri": "Amme'len", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "mevla", "tags": ["mevla", "DERIV_lAn"], "yuzey": "mevlalan", "geri": "Mevla'lan", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "kaykıltma", "tags": ["kaykıltma", "GERUND_KEN"], "yuzey": "kaykıltmaken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "yarışabilme", "tags": ["yarışabilme", "GERUND_KEN"], "yuzey": "yarışabilmeken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "tutak", "tags": ["tutak", "DERIV_lI"], "yuzey": "tutaklı", "geri": "Tutak'lı", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "manzara", "tags": ["manzara", "GERUND_KEN"], "yuzey": "manzaraken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "kungfu", "tags": ["kungfu", "GERUND_KEN"], "yuzey": "kungfuken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "kutlulama", "tags": ["kutlulama", "GERUND_KEN"], "yuzey": "kutlulamaken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "sse", "tags": ["sse", "GERUND_KEN"], "yuzey": "sseken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "barda", "tags": ["barda", "GERUND_KEN"], "yuzey": "bardaken", "geri": null, "hata": null}` | `YOL_0` |
| `{"lemma": "uşak", "tags": ["uşak", "DERIV_CI"], "yuzey": "uşakçı", "geri": "Uşak'çı", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "çeşme", "tags": ["çeşme", "DERIV_lAn"], "yuzey": "çeşmelen", "geri": "Çeşme'len", "hata": null}` | `BIT_UYUMSUZ` |
| `{"lemma": "küre", "tags": ["küre", "POSS_3SG"], "yuzey": "küresi", "geri": "Küre'si", "hata": null}` | `BIT_UYUMSUZ` |

## 3. FAZ-A — Tam lemma taraması (envanter)

- Benzersiz lemma: **48200**, istisna: **0**, süre 9.7 sn

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
| KSUR-1 probe 0-yol | 44/50 | 50/50 |
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
| `data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md` | `cfa4134011413f79f8b95d4e55fe9cf938b1d9fc488865ed26be9c6859958bab` |
| `scripts/dogrulama_p1_roundtrip.py` | `ee2251cf78f288b565e24c447f8a1642b20e6fda587b6eeab7cda7f6997ee618` |
| `data/eval/mimari_dogrulama_p1_hukum_2026-09-27.json` | `7887699bb3f9e2b7b0741940b396ce0aea2da90c6dcb56b9014147faff9d05d5` |


## 7. Doğrulayıcı (claude) kök-neden analizi — koşum SONRASI ek bölüm

*Ölçüm sayıları yukarıdaki bölümlerde BETİKTEN; bu bölüm koşum-sonrası probe'lu kök-neden
analyzeridir (`src/compiler/**` DOKUNULMADI — donmuş; onarım operatör kararıdır).*

- **YOL_0 (156 örnek; kırılımda GERUND_KEN 154 + GERUND_IncA 1): GERÇEK SİMETRİ KIRIĞI.**
  Graph'ta `GERUND_KEN` İKİ farklı şablonla geçer: `VERB_POST_TENSE→"ken"` ile
  isim-kopula state'leri→`"(y)ken"` (morphotactics.py:221-223). `MorphemeDecompiler`
  `affix_info`'yu **ilk görülen şablonla** kurar (decompiler.py:53-60) → decompiler
  isim-kökünde sabit `"ken"` kullanır: `rica → "ricaken"` (ünlü-sonlu gövdede `y`
  DÜŞÜLMÜŞ — kusurlu yüzey). `resolve_affix("rica","(y)ken")` → `"yken"` (phonology
  probe: birebir ölçüldü) → compile DFS `ricayken` yolunu arar, `ricaken`'i çözemez
  → `YOL_0` (İLAN'lı yeni sınıf → fail-closed DUR). **Doğru yüzey `ricayken`'dir;
  kırık tek affix_id'nin İKİ şablonlu olmasından doğar.** `GERUND_IncA` (1 örnek)
  aynı id-çakışması sınıfının kardeşidir.
- **BIT_UYUMSUZ (59; %100 apostrof-deseni): ÖZEL-AD İKİZ-SATIRI SIZMASI.** Lemma'nın
  TSV'deki özel-ad ikizi (örn. `Meşrutiyet` + `meşrutiyet`) trie'nin AYNI düğümünde
  birleşir (lexicon.py `load_from_tsv` — düz-lower anahtar YORUNDUĞU için); compile
  best-path kökünü orijinal-büyük-harfli satırdan seçer → `decompile_tags` `is_proper_noun`
  dalına düşer → `Meşrutiyet'im` ≠ `meşrutiyetim`. KSUR-2 (özel-ad + kesme) sınıfının
  İKİZ-SATIR yansımasıdır; İLAN §2 kapsam-dışı kuralım lemma-string bazlıydı, satır-bazlı
  ikizliği kapsamadı → İLAN'lı fail-closed DUR doğru çalıştı.
- **KSUR-1 probe 44/50 (İLAN'lı 50/50): probe BEKLENTİM dar çıktı** — yol bulan 6 yalnız-ADJ
  lemma'nın hepsi MEŞRU alternatif-kök çözümlemesidir (`abuklar` = VERB kökü `abukla-`
  + AORIST; `ademimerkeziyetçiler` = +DERIV_CI; `akışkanlar` = ayrı lemma; … — probe
  kayıtlı). ADJ-öz çözümlemede sessiz-boş davranışı **44/50'de KORUNDU**; derleyici
  kusuru değildir, İLAN §4'ün dar formülasyonudur. Hüküm İLAN'lıdır — bu sapma da
  DUR'da kalır; İLAN yenileme ayrı operatör kararıdır (koşum-sonrası İLAN yumuşatılmaz).
- **FAZ-A: 48.200 benzersiz lemma'da istisna 0** — `compile` istisna-fırlatmaz ilanlı
  davranışı birebir korundu; sessiz-boş kırılımı rapor §3'tedir.
- **Determinizm teyidi:** ilk tam koşumla (hüküm `0441b07b…`, rapor-1 üzerine yazıldı)
  nihai koşum sayıları birebir aynı (9.188 örnek / 0,976600) — seed 42 örneklemi
  koşumlar-arası kararlı.
