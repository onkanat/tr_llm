# G6b (T-0138) SONUÇ — TABAN-ÇIPA VE POZİTİF KONTROL FAZI — 2026-09-27 (antigravity)

İlan: `data/eval/berber_g6b_ilan_2026-09-27.md` (ayrı dosya — ILAN ≠ RAPOR).

## Hüküm (betikten, elle sayı yok)

1. **TABAN-ÇIPA:** **TABAN_CIPA_DUSTU** (ROUGE-L 0,1392 ≥ 0,005)
2. **POZİTİF KONTROL:** **POZITIF_KONTROL_GECTI** (ROUGE-L 0,0008 < 0,005)

## 1. Taban Model Ölçümü (`data/anka_base_v2.pt`)

Kanonik kap `scripts/evaluate_carpenter_anka.py` (`--ceket-ekseni`, n=100, seed 42, mps):

| Metrik (K2, 100 örnek, seed 42, mps) | Ölçüm |
|---|---|
| ROUGE-L ort | **0,1392** |
| ezber % | 0,00 |
| tutarsızlık % | 7,00 |
| kesişim % | 0,00 (tanı: LCS-F1 0,0057) |
| DECOMP ROUGE-L (hükümsüz aday) | 0,2162 |
| yüzey [?] | 60 |
| A ekseni CE | 3,3117 ± 0,8028 |
| B eksen top-1 | %55,91 (Wilson ±1,94) |
| oracle | kimlik 1,0000 · distraktor 0,0848 (ölçüt ayırt ediyor) |

Sonda JSON: `scratch/t0138/taban_cipa_sonda.json`
(`99ee9916bdc925e0365351eba4dc497249c7852f7bf4979f19a3497b10953403`).

## 2. Boş Model Pozitif Kontrolü (`scratch/t0138/bos_model.pt`)

Boş model (rastgele ağırlık, tohum 42, `scratch/t0138/bos_model.pt`
`765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9`):

| Metrik | Boş model | base_v2 (devralınmış) | Yorum |
|---|---|---|---|
| ROUGE-L | **0,0008** (< 0,005 → GECTİ) | 0,1392 | Ölçüt boş modeli yakaladı; ayırt edicilik kanıtlandı |
| A ekseni CE | 10,5667 ± 0,0778 | 3,3117 ± 0,8028 | Boş model `ln 33114` bandında (≈10,41) |
| B eksen top-1 | %0,00 | %55,91 | Boş model tam kör |
| tutarsızlık % | 100,00 | 7,00 | Boş üretim tamamen tutarsız |
| DECOMP ROUGE-L | 0,0008 | 0,2162 | — |

Sonda JSON: `scratch/t0138/bos_pozitif_kontrol.json`
(`16e1524f71317e76ee8cdfaba33beb0f05b2a55b13b2194bb9cc201b8a9eb196`).

## 3. Kök Neden Analizi ve Sonuç

T-0137 kardeş dikeyinde tespit edilen durum Berber dikeyinde de birebir yeniden üretilmiştir:
- `data/anka_base_v2.pt` seg_3 zincirinden devralınmış marangoz yeteneği taşıdığından ROUGE-L 0,1392 vermektedir.
- Ölçütün kendisi ise boş modelde ROUGE-L 0,0008 üreterek canlılığını ve ayırt ediciliğini kesin olarak kanıtlamıştır.
- LM-bedeli çıpası tabanın kendi ROUGE-L değeri olan **0,1392** seviyesine bağlanmıştır (kapı: 0,0230).

## Digest Tablosu

| Dosya | sha256 |
|---|---|
| `data/anka_base_v2.pt` | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` |
| `scratch/t0138/taban_cipa_sonda.json` | `99ee9916bdc925e0365351eba4dc497249c7852f7bf4979f19a3497b10953403` |
| `scratch/t0138/bos_model.pt` | `765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9` |
| `scratch/t0138/bos_pozitif_kontrol.json` | `16e1524f71317e76ee8cdfaba33beb0f05b2a55b13b2194bb9cc201b8a9eb196` |
