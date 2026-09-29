# T-0197 — Üretim-saf veri onarımı SONUÇ (Faz-2B + Faz-3)

Damga: 2026-09-29T16:21:52Z (BETİKTEN)

## Zincir (İLAN-beyanlı 3 adım)
1. **Onarım** `scripts/t0197_uretim_saf_onarim.py` GECTİ rc=0: T-0196 curated
   (n=6.680, sha-çıpalı) ÜSTÜNE 3-eylem → **n=5491**
   (P1 dil-dışlama 1188 · P3 meta-pano 1 ·
   P2 zarf-canon 235+70=305).
   Onarım-çıktı sha `data/eval/t0197_onarim_meta.json` cikti_sha256 (disk-teyit PASS).
2. **Eğitim** `scripts/train_t0197_uretim_saf.py` **T0197_EGITIM_GECTI** rc=0:
   init val 2.4334 → best **2.4193**
   (epoch-2); load=halef anka_b1_5_best (e5eb114e…,
   disk-teyit KORU); save `data/anka_b1_5_usaf.pt` sha `8f846023f2fb2374dd411f674b199f373198c108cd961c836c44e094948b2c10`
   (disk-teyit birebir). T-0196 reçete birebir (2-epoch, peak_lr 1e-4).
3. **Faz-3** `scripts/t0197_faz3_kabul.py` **T0197_ONARIM_DUR** rc=2 — disk RC=2.

## Faz-3 ölçüm (eşli-100/seed-42, arm-A birebir; T-0192 kalıbı)
| metrik | T-0197 | T-0196 çıpa | T-0194 kol-A | kapı |
|---|---|---|---|---|
| Ezber | 0.00 | 0,00 | 0,00 | PASS (≤5) |
| Tutarsızlık | 28.00 | 26,00 | 22,00 | DUS (<5) |
| decomp-ROUGE | 0.1165 | 0,1368 | 0,1325 | DUS (≥0,35) |
| RAW ROUGE | 0.0981 | 0,1078 | 0,1058 | — |
| şartlanma | 13,00 | 13,00 | 6,00 | — |

## Dil-hizalı alt-küme (İLAN-birincil gösterge; BETİKTEN)
- **İNG-hedef alt-küme** n=16: RAW 0.0467 · decomp 0.0581
- **Türkçe-ref alt-küme** n=84: RAW 0.1078 · decomp 0.1276
→ GAP: RAW 0.0612 — İngilizce-hedef örnekler Türkçe karşılığın %57'i düzeyinde; dil-dışlama (P1) eğitim-tarafı, ama test arzı karışık kaldığından üretim-gap KAPANIYOR değil.

## Hüküm ve ders
**T0197_ONARIM_DUR.** Üretim-saf veri onarımı (dil-hiza + zarf-canon + meta-pano)
üretim-metriğini TAŞIMIYOR — tutarsızlık %26→%28 (KÖTÜLEŞTİ, gürültü-bandı ±2),
decomp 0,1368→0,1165 (DÜŞÜŞ), RAW 0,1078→0,0981 (DÜŞÜŞ). Eğitim-kapısı
(EK3) ikinci kez PASS olmasına rağmen üretim eşiklerine taşınma YOK —
**5. bağımsız kanıt** (T-0193 hiza-yok · T-0194 şartlanma-nötr · T-0195 arz-kanıt ·
T-0196 arz-downsample-taşımıyor · T-0197 üretim-saf-onarım-taşımıyor):
kısa-şablon kestirme + soru→içerik hiza açığı VERİ-SÜZME ile çözülmez;
yapısal-eğitim hedefi gerekir (daha uzun koşum / yüksek-lr / üretim-hazırlı külliyat-
mimari). Risk-beyan İLAN'da: 0,35 kanıtlı-değil; teyit-edildi — UZAKTA.
