# T-0105 · A2 YETENEK/UNUTMA ÖLÇÜMÜ (İLÂN)

**Damga:** 24 Eyl 2026 · **Görev:** T-0105 · **Yürütücü:** claude ·
Üst: T-0104 (KAPANDI). Operatör: *"a2 modelinin yetenek/unutma ölçümü (a1r
zeminiyle kıyas)"*. İlan koşumdan ÖNCE; beklenti çıpaları önceki koşum
JSON'larından okunur (3-kez-yanlış-yazıldı dersi).

## 1. Kanonik kabin

`scripts/evaluate_carpenter_anka.py` (T-0094/Faz 2) — taban-eşli
karşılaştırma: `--model data/anka_a2.pt --baseline data/anka_a1r.pt
--vocab data/rebuild/vocab_anka_r2.json --lexicon data/lexicon/roots_anka_r2.tsv
--heldout data/eval/anka_r17_heldout_2026-09-20.jsonl
--train-source data/pedagogy/carpenter_specialization_dataset.jsonl
--output data/eval/anka_t0105_a2_yetenek_2026-09-24.json --ceket-ekseni
--n 100 --seed 42 --device mps`. `--output` ZORUNLU (marangoz-eval
kırpılmış-kafa dersi). Koşum sandbox DIŞINDA (MPS).

## 2. Beklenti çıpaları (betik JSON'larından okunan)

| eksen | a1r zemin | kaynak | a2 beklentisi |
|---|---|---|---|
| A-ekseni CE (taban-eşli, aynı dilim) | 3,535 ± 0,803 (maskesiz, a1r bin) | `anka_p2_taban_zemin_0e_2026-09-23.json` | taban bandında (dilim a2'de tabanla AYNI ölçülür — betik eşli) |
| ceket ROUGE | a1r bant ~0,13-0,15 (P4 zemin 0,1346) | P4 | AYNI bant — a2 yalnız ön-eğitim; SFT/ceket DOKUNULMADI |
| taban ROUGE çıpası | 0,0000 | eğitilmemiş model | korur |
| ezber | ~0 (a1r taban 0,0) | i1/p2 JSON | ~0 |
| tutarsızlık | %100 taban-çapa | p2 | a1r bantı |

Beyan: a2'nin SFT/ceket arzı DEĞİŞMEDİ (yalnız A-ekseni külliyatı yenilendi);
beklenti "ceket yeteneği taban-bantla aynı, A-ekseni CE a1r ile bant-içi" —
ceket-ekseninde İYİLEŞME bekletisi YOK (ceyet arzı aynı: P4 dersi, epoch arzı
doydu). ROUGE 0,3221 eşiği (P5 Dal-K) hüküm eşiği olarak KALIR; a2 bu
koşumda ceket-eğitim ALMADIĞI için bu eşik a1r-zemin kıyası içindir.

## 3. Sınırlar

* A-ekseni CE maskesiz ölçüm (p2 kabininde beyanlı); dilim taban-eşli.
* Üretim RNG cihaza bağlı — sayılar YALNIZ aynı cihaz (MPS) koşumuyla kıyaslanır.
* `data/eval/` tek yazma yüzeyi; donmuş yollara dokunma yok.

## Revizyon 1 (ilk koşum fail-closed rc=2)

Betik kafa/sözlük uyum kapısı taban-eşli koşumu DURDURDU: a1r checkpoint
kafası 33.114 ≠ r2 sözlük 33.911 — betik tabanı r2-vocab ile kırpacaktı
(T-0044 sınıfı kapı, doğru davrandı). **Taban-eşli koşum yerine taban-çıpa
deseni:** koşum taban-sız (`--baseline` yok), taban değerleri kanonik
JSON'lardan beyanlı okunur:
* A-ekseni CE (a1r, maskesiz): **3,535 ± 0,803** (`anka_p2_taban_zemin_0e_2026-09-23.json`)
* ceket-ekseni a1r zemin: **ROUGE 0,1346** (P4 zemin; band altı, BİTİŞ dali)
* ezber taban-çapa: 0,0 · tutarsızlık taban-çapa: %100

Hüküm: a2 değerleri bu ilanlı çıpalarla kıyaslanır; taban-çipa AYNI kabin
(aynı betik, aynı cihaz MPS, aynı seed/n).

## Revizyon 2 (çıpa kabin-uyuşmazlığı — Rev1'de yazdığım kusur)

Rev1'de ceket-ekseni a1r zemini "P4 ROUGE 0,1346" olarak çıpaladım —
**kabin-uyuşmaz:** P4 ölçümü ceket-eğitimli model + farklı betik; AYNI kabin
a1r zemini `anka_p2_taban_zemin_0e_2026-09-23.json`'da **ROUGE 0,0 ·
tutarsızlık %100 · kesişim %0 · ezber 0,0**. Düzeltilmiş çıpa (p2 taban,
a1r, aynı betik + r1 vocab + MPS + seed 42/n 100/max_new 128):

| eksen | a1r çıpa (AYNI kabin) | a2 ölçüm | hüküm |
|---|---|---|---|
| A-ekseni CE (maskesiz, a1r_pretrain.bin, seed 7) | 3,5352 ± 0,8032 | **3,3279 ± 0,7742** | İYİ yön (−0,207), taban bandı |
| B-ekseni top-1 (2522 konum, seed 7) | 49,5638 (Wilson 47,61-51,51) | **52,6963** (50,75-54,64) | İYİ yön (+3,13 pp; Wilson bantları 50,75-51,51'de hafif örtüşür) |
| çekirdek top-1 | 56,3337 | 56,8819 | İYİ yön |
| ceket ROUGE-L | 0,0000 | 0,0049 (LCS 0,0035) | taban-çapa bandı |
| ezber | 0,0 | 0,0 | korur |
| tutarsızlık | %100 | %100 | korur (p2 a1r da %100 — P4 %14 farklı kabindi) |
| kesişim | %0 | %0 | korur |

Beklenti (ilan §2) doğrulandı: a2 SFT/ceket arzı değişmedi; ceket-ekseni
taban-çapa bandında, A/B eksenlerinde a2 eğitiminin dilim-iyileşmesi
görünüyor (kabin-eşli her üç ölçümde İYİ yön).