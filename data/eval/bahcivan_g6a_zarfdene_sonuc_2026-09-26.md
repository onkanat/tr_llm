# G6a (T-0137) RAPOR — ÜRETİM TANISI: ROL-ZARF YÖLLENDİRME DENEMESİ — 2026-09-26 (claude)

Operatör emri (2026-09-26): "system_message ile yönlendirme denensin — Marangozda
işe yaramıştı". ILAN ≠ RAPOR: ilan `bahcivan_g6a_zarfdene_ilan_2026-09-26.md`
(`6fc23835…`, koşum ÖNCESİ damga). Hüküm BETİKTEN
(`scratch/t0137/bahcivan_zarfdene_hukum.py` → `bahcivan_zarfdene_hukum.json`,
rc=0; elle sayı YOK). Bu TANISAL deneydir — **kabul ayracı DEĞİŞMEZ: A9
BAND_ALTINDA + A13 IHLAL → KABUL_YOK nötr ölçümden kalır**.

## Hüküm (betikten, iki sonda JSON kıyası)

| Eksen | nötr ikiz (`be84f93e…`) | zarflı ikiz (`--rol-zarf`, `524b6806…`) | fark |
|---|---|---|---|
| ROUGE-L ort | 0,0365 | 0,0315 | **−0,0050** |
| ROUGE-L medyan | 0,0323 | 0,0290 | −0,0033 |
| tutarsızlık (yüklemsiz-son/döngü) | %43,0 | %36,0 | **−7 pp** |
| üretim ort (kelime) | 61,32 | 52,25 | **−9,07** |
| ezber | 0,00 | 0,00 | 0 |
| yüzey unk sayısı | 40 | **273** | +233 |
| decompile istisna | 0 | 0 | 0 |

**Dallar (İLAN beyanı, betikten):** tutarsızlık zarflı < nötr →
**KILIT_AZALDI** · ROUGE fark ≤ −0,005 → **ROUGE_DUSTU** · üretim-uzunluk
zarflı < nötr → **UZUNLUK_KILANDI**.

## Okuma (ölçülenler)

1. **Kilit azaldı ama kaldırılmadı:** tutarsızlık %43 → %36 (−7 pp). Kardeş
   marangoz kanıtı T-0114'te zarf aynı eksen −15 pp vermişti — Bahçıvan
   istemlerinde etki yarıdan az.
2. **ROUGE eşiğe TAM oturarak düştü (−0,0050):** zarf Bahçıvan yetenek
   ekseninde nötr değil, sayılabilir düşüş. Medyan da aynı yönde (−0,0033).
3. **Yüzey unk 40 → 273 (6,8×):** marangoz zarfı (ROL_ZARFI, DOKUNULMAZ
   sabit) Bahçıvan kısa-kalıp istemle birleşince üretimde dağılım-dışı jeton
   patlaması — marangoz zarfının Bahçıvan dikeyinde uyumsuzluğunun yüzey
   izi. Bu, İLAN-3b §5'teki "Bahçıvan zarfı ROL_ZARFI'ya verilemez" kısıtını
   veriyle destekler.
4. **Uzunluk kayması kısmen düzeldi:** üretim 61,32 → 52,25 kelime (referans
   33,9) — T-0114 desenindeki zarf-kısaltma etkisi burada da VAR ama hâlâ
   referansın 1,54×'i.

## Bahçıvan vs marangoz zarf-etkisi (kardeş kıyas, ölçülenler)

| Eksen | marangoz (T-0114/T-0115) | Bahçıvan (bu deney) |
|---|---|---|
| tutarsızlık | −15 pp (S3) | **−7 pp** (%43→%36) |
| ROUGE | KORUNDU (f −0,0012) | **DUSTU** (−0,0050, eşik-tam) |
| unk | — | **+233 sayı (40→273)** |

Sonuç: marangoz zarfı Bahçıvan dikeyine taşınmaz — kilit kısmen azalır ama
yetenek ekseninde bedeli ROUGE-DUSTU + unk-patlamasıdır. **Bulgu: kusurun
kaynağı istem-zarfı değil; kök-neden raporundaki üretim-fazı eksenleri
(sıcaklık/decoding/OUTPUT-zarf uyumu/sonlandırma) bu deneyin dışında
kalmaya devam eder.**

## Dürüst kayıtlar

1. **İlk hüküm koşumu kayan-nokta artefaktıyla sınır-dalda kaldı:** ham fark
   `−0,0049999999999999975` (matematiksel TAM −0,0050) yüzünden `<= -0.005`
   False oldu → ilk koşum `ROUGE_KORUNDU` yazdı. Ayrac DEĞİŞTİRİLMEDEN betik
   kap-hassasiyetiyle (4 ondalık — ayracın kaynağı) hizalandı; **hüküm ikinci
   koşumdan: ROUGE_DUSTU**. İki koşum da kanıt; hüküm JSON ikinci koşumdur
   (`9e221689…`).
2. ROUGE eşiği ±0,005 İLAN'da beyanlıydı; fark eşiğe TAM oturdu — sınır-değer
   kararına iki bağımsız destek: medyan aynı yönde ve unk 6,8×.
3. Zarflı koşum ÖLÇÜM-2 parametreleri birebir (n=100, seed 42; tek fark
   `--rol-zarf`); EGITICI_YOK kanıtlı; MPS sandbox dışı; rc=0.
4. Bu deney A9/A10 kabul ayracına girmez — nötr ölçüm (`be84f93e…`)
   kabul birleşiminin kaynağıdır (KABUL_YOK değişmez).

## Kanıt digest tablosu (tam sha256)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/bahcivan_zarfdene_hukum.json` (hüküm) | `9e2216890c6ec4596024ad18c60bc86e6d96855b1e9715beba1ab522648178bd` |
| `scratch/t0137/bahcivan_zarfdene_hukum.py` (betik, 2. sürüm) | `41abc42e1675d59ba752d3321a29f004676f2493ebe86481fb8025a3bb84f367` |
| `scratch/t0137/bahcivan_ft_bahcivan_zarfli.json` (zarflı sonda) | `524b6806ae708c622d75934a963f8f55f5da99110e114ebcc877422f6c32f33e` |
| `scratch/t0137/zarfdene.log` (koşum log) | `33a6e5f50f91addf46815abd0c5ea4fa2491d1cf3b482ac939296c28717b3e12` |
| `data/eval/bahcivan_g6a_zarfdene_ilan_2026-09-26.md` (İLAN) | `6fc23835a586b807f61cfe5cd12ae337f3e0c87149206494dd4eb8f47533d893` |
| girdi (salt-okunur): nötr ikiz `be84f93e…` · ft ckpt `bd68c450…` | (önceki faz kanıtlarıyla birebir) |

## Sonuç dalları

- **KILIT_AZALDI + ROUGE_DUSTU + UZUNLUK_KILANDI** — marangoz zarfı Bahçıvan
  dikeyinde marangozdaki kadar işe yaramıyor; taşıma kararı veriyle reddedildi.
- Kabul hükmü değişmez: **KABUL_YOK** (A9 BAND_ALTINDA + A13 IHLAL).
- İzleyen eksenler (sıcaklık/decoding/OUTPUT-zarf/sonlandırma) AYRI görev —
  ölçülmeden ilan edilmez.
- Checkpoint silme + commit OPERATÖR kapıları bekliyor. G6b (T-0138)
  antigravity'ye açık.

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · ROL_ZARFI ve
`evaluate_carpenter_anka.py` DOKUNULMAZ · tokenizer/compiler DOKUNULMAZ ·
data/** salt-okunur · elle sayı YOK · `git add -A` YASAK · commit operatör
kapısıdır.