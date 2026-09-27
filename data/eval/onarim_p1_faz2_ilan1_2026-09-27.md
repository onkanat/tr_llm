# İLAN-1 — T-0150 FAZ-2 ONARIM-KOŞUMU: P1 kalan-6 A2+B2+C compiler-onarımı

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-27T19:41:08Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Ölçüm İLAN'a hizalanır.
Hüküm BETİKTEN (`scripts/dogrulama_p1_faz2_onarim.py`); elle sayı/hüküm YOK.

## Kapsam ve meşruiyet-çıpa

Operatör kararı (27 Eyl 2026; T-0149 koşum-2 KEŞIF_TAMAM `75277676…` üzerinde
AskUserQuestion: **"A2+B2+C (Recommended)"** — kayıt `79a8584` commit-i
mesajında). Bu koşum **ONARIM koşumudur**: `src/compiler/core.py` DONMUŞ-desene
kiralamalı (T-0150 kiralamaları: `src/`, `scripts/`, `data/eval/`,
`.agent-bus/notes/` — ÜST-DİZİN; frozen.json `src/compiler/**` kalıbı
DEĞİŞMEZ). Onarım yüzeyi yalnız `src/compiler/core.py` — **morphotactics.py /
phonology.py / decompiler.py / lexicon.py / tokenizer.py YAZIM YOK**.

## Onarım-kontratı (üç sınıf; T-0149 koşum-2 ölçümlü-tanısından)

- **[A2]** Sınıf-A (nakil/meçhul/hacir — VOWEL_DROP-kırpılmış-gövde):
  `compile` gövde-döngüsünde, VOWEL_DROP-lookahead adayının (kırpılmış
  matched_prefix, ör. "hacr") çözümü **lemma-gövdesinden** yapılır
  (`mutate_stem(lemma) == matched_prefix` guard'ı; decompiler-yüzeyi ve
  `find_stems` dönüş-şekli DEĞİŞMEZ). Uyum-harmonisi lemma'nın son-ünlüsünden
  ("hacir"→'i' front → "hacrimiz" ✓; "hacr"→'ı' → "hacrımız" ✗).
- **[B2]** Sınıf-B (zeyrek/güç — VOICING tam-eşleşme): beam-pruning
  düşmesinde gövde-yanı VOICING'i nötralize edip yeniden çözüm (ÇÖZÜMLE-SEÇ,
  yalnız düşen-dalda; mevcut voicing-izni dalı KORUNUR).
- **[C]** Sınıf-C (bil/Bi — iki-farklı-lemma): `_score_paths` skor-tie
  kırılımına üçüncü ölçüt: **kanonik-gövde-önce** (kök `is_case_alias`'sız
  yol, eşit skorla öncelikli; aynı-bayrak içinde mevcut stable-sıra birebir
  korunur; tokenizer'ın kendi yolları bayraksız → davranış bit-özdeş).
  TSV-sembolik karar beyanı: kanonik lemma önceliklidir (is_case_alias ikiz
  gölgelemez — T-0147 A-2 ile aynı yön).

## Hedef-değerler (onarım-ETKİSİNDEN formüle — İLAN-formülasyon-kusuru dersi)

Onarım-öncesi kanıt (İLAN-3 koşum-2, betik deterministik SEED-42): bit_esit
9182/9188, kalan-6 dislamalı (YOL_0 5 + BIT_UYUMSUZ 1). Üç sınıfın onarımı
kalan-6'yı düzeltir → **beklenti bit_esit==9188 (%100), düşen boş**.
Decompiler-yüzeyi DEĞİŞMEZ (A2 core-pathfinding, B2 core-beam, C core-scoring
— decompiler/lexicon/phonology/morphotactics/tokenizer değişmez).

## Kapılar (İLAN-1 §1 — koşum öncesi sabit)

- **K1 FAZ-B ONARIM:** ornek==9188 (SEED-42 örnekleme birebir korunur;
  `_yuruyus` graph-only, decompiler-yüzeyi değişmez) VE bit_esit==9188 VE
  düşen == {} (İLAN-3 kalan-6 üç-sınıfı boşalır).
- **K2 DEC-YÜZEY-KORUMA:** İLAN-3 kalan-6 çiftleri `decompile_tags` ×6
  birebir: zeyrekin/meçhlun/naklin/güçüm/hacrimiz + [bil,TENSE_AORIST_VOWEL]→
  "biler" (decompiler-yüzeyi çıpa `6428b180…` değişmez-kanıtı).
- **K3 COMPILE-YOL-ÇAPA:** 6 yüzey `compile()` → `token_vector[0] == lemma`
  ×6 + geri==yüzey ×6 (naklin→nakil · meçhlun→meçhul · zeyrekin→zeyrek ·
  güçüm→güç · hacrimiz→hacir · biler→bil).
- **K4 ROOT-ORACLE:** koşum-ÖNCESİ baseline (onarım-öncesi kod, bu İLAN'ın
  damgasından önce ölçüldü): **25/25, başarısız {}** → onarım-sonrası
  birebir 25/25 (yeni-başarısızlık = DUR; iyileşme yok-beklentili, ROOT
  girdileri tek-gövde — A2/B2/C bu küçük-oran etkisini KOŞUM ölçer).
- **K5 KSUR-1/2/3/4 SABİT:** 3.627/6.352 + probe 44/50 · 20/20 kesmeli ·
  79/70 · 200/200 sessiz-boş (davranış-onarımı, veri-değişikliği YOK).
- **K6 FAZ-A:** istisna==0 VE sessiz-boş==0 (İLAN-3 envanteri; onarım yalnız
  YOL AÇAR, kapatamaz — sessiz-boş artamaz).
- **K7 ŞEMA-KORUMA (tokenizer imza):** `compile()` dönüş anahtar-kümesi ==
  {input, language, analyses, best_surface, token_vector,
  needs_disambiguation}; `find_stems` dönüş list[tuple(str, dict)] ve
  stem-dict anahtarları {lemma, pos, attributes} içerir; `_score_paths`
  tokenizer-yolu (bayraksız init_path) bit-özdeş.
- 7/7 → **P1_FAZ2_GECTİ rc=0**; aksi her dal → **DUR rc=2**.

## pytest (ayrı koşum — hüküm-dışı, RAPOR'da kayıt)

Onarım-öncesi baseline: **307/307** (bu İLAN'dan önce koşuldu, 19:4xZ).
Onarım-sonrası: geçen ≥ 307 VE yeni-düşen YOK; düşen test → onarım veya
İLAN'da beyanlı (test-silme YOK; değişen davranış testi güncellenir).

## DOKUNULMAZLAR

- `data/lexicon/roots.tsv` digest `fe3005e5…` SABİT (salt-okuma).
- `src/compiler/morphotactics.py` (graph), `phonology.py` (golden-tablo),
  `lexicon.py`, `decompiler.py`, `src/llm/tokenizer.py` — YAZIM YOK.
- İLAN-3 hüküm `6428b180…` — onarım-öncesi kanıt; bu tur decompiler-yüzeyini
  değiştirmez (K2 kanıtlar), FAZ-B ölçütü compile-yolu açar.
- T-0149 koşum-artefaktları DOKUNULMAZ.
- `data/**` yazımları yalnız `data/eval/` rapor/hüküm-yollarında.

## Beyan: damga-yöntemi

Damga koşum-öncesi `date -u` betik-çıkışından işlendi; dosyanın varlık-kanıtı
mtime-çıpasıdır (koşum `ilan_sha256` alanında kaydedilir). Betik-sha
RAPOR-digest tablosundadır (T-0147 dersi: İLAN'da betik-sha beyanı genel
kalır).