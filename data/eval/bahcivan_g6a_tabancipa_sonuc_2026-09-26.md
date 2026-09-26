# G6a (T-0137) SONUÇ — TABAN-ÇIPA FAZI — 2026-09-26 (claude)

İlan: `data/eval/bahcivan_g6a_ilan_2026-09-26.md` (ayrı dosya — ILAN ≠ RAPOR).

## Hüküm (betikten, elle sayı yok)

**TABAN_CIPA_DUSTU** — İLAN-1 ayracı koşum-öncesi ilanlıydı: ROUGE-L < 0,005 → GECTI.

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

Sonda JSON: `scratch/t0137/taban_cipa_sonda.json`
(`dfddbd509bc820113c4c86dc76f8b960ca2a43a20043be3c448e6e649f268d98`). ROUGE-siz ilk koşum
kanıtı: `scratch/t0137/taban_cipa_sonda_rougesiz.json`
(`b891afc02a1ff55e29cd8ea3bcc690676f31abd255e6f3d3fc985f5e7d547820`).

## Kök neden (ölçülmüş): taban-çıpa TASARIM VARSAYIMI YANLIŞTI

İlan-öncesi varsayım "anka_base_v2 = eğitilmemiş/kör taban ⇒ ROUGE 0,0000" ÇÜRÜTÜLDÜ:

1. **base_v2 zincir-içi devralınmış:** seg_3 zincirinden kanonik resize ile kuruldu
   (T-0133: sıfır-kırpma; T-0113: seg_3 faz-C koşumu).
2. **Karşılaştırma:** aynı kanonik kap, aynı heldout (anka_r17, 539 satır) — seg_3
   ROUGE **0,1354** (`scratch/t0113/sonda_seg_3.json`); base_v2 **0,1392**. Fark
   **+0,0038**, ROUGE SE ~0,0115 ⇒ **gürültü içinde** — base_v2, seg_3 yeteneğini
   resize ile KORUDU (beklenen davranış; "düşen ROUGE" değil).
3. Yani 0,1392 KUSUR değil, çıpanın **yanlış modele** bağlanmasıdır: "0,0000" çıpası
   yalnız gerçekten boş (sıfırdan) model için ayırt edicidir — base_v2'de zaten
   marangoz yeteneği var (kaynak zincir marangoz ince ayarlı).

## Etki (fail-closed)

- İLAN-1 hükmü: **fine-tune BAŞLAMAZ** (TABAN_CIPA_DUSTU dalında kabul yok).
- G6a şartnamedeki kabul maddesi "eğitilmemiş anka_base_v2 ROUGE 0,0000" bu tabanda
  **doğrulanamaz** — şartname varsayımı düzeltme ister (operatör kararı).
- Öneri (operatör onayına sunuldu): (a) LM-bedeli çıpası (0,0230 kapısı) base_v2'nin
  **kendi ROUGE'sine** bağlansın (0,1392 = ceket-çıpa nötr noktası); (b) "0,0000"
  pozitif kontrolü gerçekten BOŞ modelle ayrı koşumda yapılır (ölçütün ayırt
  ediciliği kanıtı — taban değil ölçüt için); (c) tutarsızlık %7,00 > %5 base için
  hüküm değildir (esikler ince ayarlı model içindir — rapor beyanlı).
- Arz pilot kapısı (İLAN-2) etkilenmez; OLLAMA meşkul — beklemede.

## İLAN-4 SONUCU (operatör onayıyla; koşum 2026-09-26 11:15Z) — **POZITIF_KONTROL_GECTI**

Boş model (rastgele ağırlık, tohum 42, `scratch/t0137/bos_model.pt`
`765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9`) kanonik kapta
(`--ceket-ekseni`, n=100, seed 42, mps):

| Metrik | Boş model | base_v2 (devralınmış) | Yorum |
|---|---|---|---|
| ROUGE-L | **0,0008** (< 0,005 → GECTİ) | 0,1392 | ölçüt BOŞU yakaladı; ayırt edicilik kanıtlandı |
| A ekseni CE | 10,5667 ± 0,0778 | 3,3117 ± 0,8028 | boş model `ln 33114` bantta (≈10,41) — canlılık imzası |
| B eksen top-1 | %0,00 | %55,91 | tam kör |
| tutarsızlık % | 100,00 | 7,00 | boş üretim tutarsız |
| DECOMP ROUGE-L | 0,0008 | 0,2162 | — |

Sonda JSON: `scratch/t0137/bos_pozitif_kontrol.json`
(`8bfc104f0e39d5e11649df56e610ac9eb708c9f8b8f7b051a18923dba92e2d73`).

**Kapanış halkası:** taban-çıpa deseninin iki kanadı birebir ölçüldü — ölçüt boş
modeli 0,0008 ile yakalar (0,0000 çıpası ölçüt için DOĞRU); base_v2'nin 0,1392'si ise
devralınmış zincir yeteneğidir (kaynak: seg_3 0,1354; fark gürültü içinde). Fine-tune
tasarımı düzeltilmiş çıpayla (LM-bedeli kapısı 0,0230, çıpa = 0,1392 — operatör onayı
2026-09-26, İLAN-3 DÜZELTME) arz pilotunun sonucunu bekliyor.

## Digest tablosu (İLAN-4 eki)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/bos_pozitif_kontrol.json` | `8bfc104f0e39d5e11649df56e610ac9eb708c9f8b8f7b051a18923dba92e2d73` |
| `scratch/t0137/bos_model.pt` (kanıt; kapanışta silinir) | `765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9` |

| Dosya | sha256 |
|---|---|
| `scratch/t0137/taban_cipa_sonda.json` (2. koşum, ROUGE'lu) | `dfddbd509bc820113c4c86dc76f8b960ca2a43a20043be3c448e6e649f268d98` |
| `scratch/t0137/taban_cipa_sonda_rougesiz.json` (1. koşum) | `b891afc02a1ff55e29cd8ea3bcc690676f31abd255e6f3d3fc985f5e7d547820` |
| `data/eval/bahcivan_g6a_ilan_2026-09-26.md` | (koşum-öncesi ilan; digest koşum logunda) |
| `data/anka_base_v2.pt` | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` (DEĞİŞMEDİ) |
| kanonik kap + vocab + lexicon + heldout | sonda JSON'daki sha256 alanlarından birebir |