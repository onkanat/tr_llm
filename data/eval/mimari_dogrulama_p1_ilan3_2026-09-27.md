# İLAN-3 — MİMARİ DOĞRULAMA PAKET-1 KAPANIŞ-ÇIPASI (T-0147; P1 kalan-6)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-27T16:57:49Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Ölçüm İLAN'a hizalanır.
Hüküm BETİKTEN (`scripts/dogrulama_p1_roundtrip.py`); elle sayı/hüküm YOK.

## Kapsam ve meşruiyet-çıpa

Bu koşum **ONARIM KOŞUMU DEĞİLDİR — KAPANIŞ-ÇIPASI**. A-1/A-2 onarımı
(commit `5db5062`) koşum-2'de ölçüldü: FAZ-B 215 düşen → **6** (hüküm
`6f471862…`, rc=2; İLAN-2 %100-hedefi DUR). Operatör kararı (27 Eyl 2026,
16:5xZ): İLAN-3 çıpa — kalan-6 **iki alt-sınıf AÇIKÇA dislamalı** ve betik
deterministik (SEED 42 + TSV donmuş; koşum-2/3 bit-özdeş ölçüldü) olduğu
için **kalan-6 örnek-beyanı birebir çıpa olarak İLAN'lıdır**. Çıpa
koşum-2 KANIT-ölçümünden formüle edilir (İLAN-formülasyon-kusuru dersi:
beklenti etki-ekseninden kurulur — bu turda onarım-etkisi SIFIR, decompiler-
yüzeyi değişmedi; çıpa meaningful). Betik sha256:
`94523084010b6509ae9918d4543a047a8513191616c3e8806992ab0a14ec149c`
(koşum boyunca SABİT — koşum betiği kendi sha'sını hüküm-JSON'a yazar ve
bu değerle birebir denetlenir).

## FAZ-B birebir-çıpa (birincil kapı — İLAN-3)

| Alan | İLAN-2 (koşum-2 bunu ÇÜRÜTTÜ: kalan-6) | İLAN-3 (operatör formülasyonu) |
|---|---|---|
| birincil ölçüt | bit-özdeşlik %100 (0 düşen) | **birebir-çıpa**: `ornek_sayisi == 9188` VE `bit_esit == 9182` VE toplam düşen `== 6` VE düşen-sınıfları ⊆ `{YOL_0, BIT_UYUMSUZ}` (dislanalı-dışı sınıf YOK) VE kalan-6 örnek-beyanı `{lemma, tags, yuzey, geri}` sıra-duyarlı **birebir** |
| kalan-6 sınıf-beyanı | — | YOL_0 (5): **VOWEL_DROP/kök-seçim derin-mekanizması** — zeyrek/meçhul/nakil CASE_GEN + güç POSS_1SG + hacir POSS_1PL (affix_info-şablon katmanı ötesi); BIT_UYUMSUZ (1): **iki-farklı-lemma** — `bil` VERB ↔ `Bi` büyük-harfli (iki-ayrı-lemma TSV-sembolik kararı ayrı turdadir) |
| son-ek kırılımı | — | beyanlı birebir: `{"CASE_GEN": 3, "POSS_1SG": 1, "POSS_1PL": 1, "TENSE_AORIST_VOWEL": 1}` (RAPOR-kırılımı — kapı ölçütü bit_esit/örnek/kalan-6'dır) |

## kalan-6 örnek-beyanı (birebir çıpa — betikte `FAZ_B_DISLANALI_ORNEKLER`)

| # | lemma | tags | yüzey | geri | sınıf |
|---|---|---|---|---|---|
| 1 | zeyrek | `[zeyrek, CASE_GEN]` | zeyrekin | — | YOL_0 |
| 2 | meçhul | `[meçhul, CASE_GEN]` | meçhlun | — | YOL_0 |
| 3 | nakil | `[nakil, CASE_GEN]` | naklin | — | YOL_0 |
| 4 | güç | `[güç, POSS_1SG]` | güçüm | — | YOL_0 |
| 5 | hacir | `[hacir, POSS_1PL]` | hacrimiz | — | YOL_0 |
| 6 | bil | `[bil, TENSE_AORIST_VOWEL]` | biler | `Bi'ler` | BIT_UYUMSUZ |

Sıra koşum-2/3 ile birebir (deterministik: secilenler SEED-42 örneklemi +
yürüyüş rng SEED-42 — T-0147 koşum-2/3 bit-özdeş teyitli). YOL_0'da geri
değeri ÜRETİLMEDİ (compile 0-yol erken-dönüş) → projeksiyonda `None`.

## FAZ-A + FAZ-C İLAN'lı (koşum-2 SABİT — değişmez)

- FAZ-A: compile **istisna FIRLATMAZ** (istisna_toplam == 0).
- KSUR-1: NOUN-satırı-yok ADJ == **3.627/6.352** + probe **44/50** 0-yol
  (İLAN-2 dislaması KORUNUR: 6 yol-bulunan meşru-alternatif-kok).
- KSUR-2: kesme-apostrof probe **%100** kesmeli + placeholder apostroflu.
- KSUR-3: İ/I düz-lower satır == **79** (72 İ+7 I) VE benzersiz == **70**
  (63 İ+7 I) — ikiz-satır envanteri korunur (veri değişmez).
- KSUR-4: OOV → **200/200** sessiz-boş, istisna/yol YOK.

## DOKUNULMAZLAR

- `data/lexicon/roots.tsv` — koşum boyunca SABİT (digest raporda).
- Kanonik kod (`src/compiler/**`, `src/llm/tokenizer.py`) — koşum İÇİNDE
  yazım YOK (import-only; DONMUŞ-desen).
- `data/**` — koşum yazımları yalnız `data/eval/` rapor/hüküm-yollarında.
- Ağ YOK, saf CPU, tek süreç, seed 42.

## Hüküm kuralı (BETİKTEN)

Tüm kapılar (FAZ-B çıpa + FAZ-A + KSUR-1/2/3/4) GEÇTİ → **P1_GECTİ (kalan-6
dislamalı çıpa ile kapandı)** rc=0; aksi → **DUR rc=2**. İstisna veya
beklenmedik imza → DUR; koşum-sonrası İLAN yumuşatılmaz.

## Sonraki plan (operatör beyanı, kayıt-altında)

**Onarım-genişletme** sonraki ilk plandır: VOWEL_DROP/kök-seçim
derin-mekanizması (CASE_GEN/POSS_* YOL_0-3) + iki-farklı-lemma yol-skor
kuralı (`bil`/`Bi`) — ayrı İLAN + keşif-koşumu; `src/compiler/**`
DONMUŞ-desene yazım (kiralamalı, operatör-onaylı).

## Beyan: damga-yöntemi

Bu İLAN'ın damga-sayısı elle yazılmaz; yukarıdaki damga koşum-öncesi
`date -u` betik-çıkışından işlendi (T-0147 6. ihlal-dersi, madde-8).
Dosyanın varlık-kanıtı ek olarak mtime-çıpasıdır: bu dosya koşum
başlangıcından ÖNCE yazılmıştır (koşum `ilan_sha256` alanında kaydedilir).