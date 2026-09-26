# G6a (T-0137) İLAN — KÖK-NEDEN TANISI: ÜRETİM-FAZI EKSENLERİ — 2026-09-26 (claude)

Operatör emri (2026-09-26T19:12Z): "üretim eksenlerini ölç: sıcaklık, decoding,
OUTPUT-zarf, sonlandırma". Kök-neden raporu
(`bahcivan_g6a_kokneden_kiyas_sonuc_2026-09-26.md` `2741e4d5…`) üç ekseni
"ÖLÇÜLMEMİŞTİR" beyanıyla ilan etmişti; bu İLAN onları ölçüme açar. ILAN ≠
RAPOR: rapor `bahcivan_g6a_uretim_eksenleri_sonuc_2026-09-26.md`. Bu TANISAL
görevdir — **kabul ayracı YOK; KABUL_YOK (A9 BAND_ALTINDA + A13 IHLAL)
nötr ölçümden DEĞİŞMEZ.** Hüküm BETİKTEN (elle sayı YOK).

## Koşum ÖNCESİ damga: 2026-09-26T19:12:58Z (date -u)

## Deney beyanı (koşum ÖNCESİ)

- **Girdi (salt-okunur):** ft `scratch/t0137_g6a_kos/seg_3.pt` (`bd68c450…`) ·
  heldout `scratch/t0137/bahcivan_heldout.jsonl` (`36416b82…`) · sözlük
  `vocab_anka_r1_33114` · lexicon `roots.tsv` (literal_entity_mode=True —
  kök-neden betiği kurulum deseni birebir).
- **Örneklem:** kap birebir — `held` dosya sırası + `random.Random(42).sample(held, 100)`
  (`evaluate_carpenter_anka.py:460-461`) — ÖLÇÜM-2 nötr koşumuyla bit-uyumlu.
- **Dallar (5):**
  - **D0 GREEDY-REFERANS (tavan 128):** sarmal üretici T=0 — kanonik `uret`
    (import) ile İLK 3 örnekte bit-özdeş DOĞRULANIR (pozitif kontrol;
    sarmal-doğrulaması — fail-closed). D0 tam dal sarmal T=0 ile koşar.
  - **E1 SICAKLIK:** T=0,7 (multinomial, top-k YOK; CPU'da `torch.multinomial`,
    `torch.manual_seed(42)` koşum başı — beyanlı; T=1,0 bu koşumda ölçülmez,
    0,7 bulgusuna bağlı izleyen dal).
  - **E2 DECODING:** BEAM_W=3 (log-softmax toplam puan; **uzunluk
    normalizasyonu YOK** — beyanlı; bu seçimin etkisi rapora not edilir).
  - **E3 OUTPUT-ZARF:** zarsız istem — kanonik prompt'un sondaki `<OUTPUT>`
    jetonu ATILIR (jeton düzeyi; string düzenleme YOK; atılınca son jeton
    out_id OLMAMALI — fail-closed). Ölçüt: ilk 10 üretilen jetonda
    `<OUTPUT>` görülme payı (model kendisi OUTPUT'a geçiyor mu).
  - **E4 SONLANDIRMA:** greedy, tavan **256** (D0 tavan 128 kıyası) + TÜM
    dallarda sonlanma-istatistiği: sonlanma_tipi ∈ {eos, out_end, kesme}.
- **Sarmal üretici beyanı:** `uret` gövdesinin parametreli uzantısıdır
  (sıcaklık/beam/istem-atla); kanonik `uret` D0-ilk-3 pozitif kontrolde
  import edilir. Tutarsızlık tanımı kap birebir: son 5 jetonda TENSE_/COPULA_
  yok VEYA 2-jeton deseni ×3 (6 jeton penceresi) —
  `evaluate_carpenter_anka.py:511-514`.
- **Ölçümler her dalda:** ham ROUGE-L (`rouge_l_score` import; kapın `rouges`
  dalı — HAM gm, DECOMP değil) · tutarsızlık · üretim-ort kelime ·
  sonlanma payları · ort jeton sayısı. **Artımlı yazım:** her dal bitince
  sonuç JSON disk'e yazılır (uzun-ölçüm kuralı).
- **E3 ROUGE beyanı:** zarsız dalda ROUGE, `<OUTPUT>`'tan sonraki cevap-kısmı
  üzerinden hesaplanır (model geçmediyse tüm üretim) — D0 ile koşul farkı
  var; E3'ün asıl ölçütü out_id-ilk-10 PAYIDIR; ROUGE indirekt bilgi.
- **Maliyet beyanı:** 5 dal (beam 3×, E4 ~1,5×) ≈ 7,5 üretim-birim; ÖLÇÜM-2
  koşumu ~2,5 saat çıktı — tahmin 3-4 saat; artımlı kayıt + izleyici.

## Beyanlı hüküm dalları (TANISAL etiket — kabul kararı DEĞİL)

| Eksen | kilit dalı | ROUGE dalı (eşik ±0,005 — İLAN-3b/zarf-denemesi ile aynı) |
|---|---|---|
| E1 SICAKLIK | tut < D0 → **SICAKLIK_KILIT_AZALDI** / değilse **_AZALMADI** | ARTTI ≥ +0,005 · KORUNDU · DUSTU ≤ −0,005 |
| E2 BEAM | tut < D0 → **BEAM_KILIT_AZALDI** / **_AZALMADI** | aynı üç dal |
| E3 ZARF | out_id-ilk-10 payı ≥ %80 → **ZARF_GEREKSIZ**; < %80 → **ZARF_GEREKLI** | (indirekt — ayrı etiket YOK) |
| E4 TAVAN | kesme E4 < D0 VE tut ≤ D0 − 10 pp → **TAVAN_KILIT_TASIYOR**; kesme azalır VE ROUGE ≥ +0,005 → **TAVAN_KIRILDI_ROUGE**; kesme azalmaz → **TAVAN_DOKUNMAZ** | aynı üç dal |

- ROUGE_KORUNDU eşik dalı: kayan-nokta sınırlaması — fark **round4** alınır
  (kap 4 ondalık yazdığından matematiksel eşdeğerlik; zarf-denemesi
  sınır-karar dersi uygulandı).

## Dürüst kayıt (İLK koşum pozitif-kontrol DUSTU — rc düşüşü)

- İLK koşum D0 ilk-3 pozitif kontrolde RuntimeError ile DURDU: "sarmal T=0
  uret ile bit-ozdes degil (gen 36 vs 36)". **Kök neden ölçüldü (izole koşum):**
  kıyas TÜR-UYUŞMAZLIĞIYDI — kanonik `uret` DECODE edilmiş string listesi
  döndürür (`evaluate_carpenter_anka.py:314`), sarmal id listesi; içerik
  birebir özdeşti (argmax id 32816 `<ENT>` iki yolda; max|Δlogits| 0,0;
  `uret1==uret2` True — MPS determinizmi doğrulandı). Kusur üretimde değil
  KONTROLDEYDİ; ayrac DEĞİŞTİRİLMEDEN kontrol id-normalize edildi
  (`vocab.stoi.get(s)` — None olursa RuntimeError, fail-closed korunur).
  Hüküm İKİNCİ koşumdan gelir (ÖN-3 şema-düzeltme deseni).

## DOKUNULMAZLAR

`uret`/`render_prompt`/`evaluate_carpenter_anka.py`/tokenizer/compiler
DOKUNULMAZ · ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ ·
ROL_ZARFI kullanılmaz (nötr istem; zarf ekseni FAZ-3b'de kapandı) ·
data/** salt-okunur · checkpoint yazımı YOK · `git add -A` YASAK ·
MPS koşum sandbox DIŞI · commit operatör kapısıdır.