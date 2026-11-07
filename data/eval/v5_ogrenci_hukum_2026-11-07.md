# HÜKÜM — V5 öğrenci (pure'dan lr 1,5e-4): HÜKÜM 1/3 — K1 seri rekoru, K2 0,0336 nat ile KALDI (2026-11-07)

**İlan:** `data/eval/v5_ogrenci_ilan_2026-11-07.md` (kapılar koşumdan önce dondu; hiçbiri kaydırılmadı)
**Koşum:** 11.380 adım / **36.451,9 sn (10,1 sa)** / 3,20 s-adım · `scratch/v5_ogrenci_run.log`
**Çıktı:** `data/anka_ogrenci_v5.pt` sha `1eded7fa2c4a5337496a890c01b683b74133191433f69b23853a70ee1f7e3904`
**Provenans:** init = öğretmen = pure `5479a2f3…`, sft `f90cf88b…`, külliyat `7f91317a…` — İLAN çıpalarıyla birebir
**Tarif:** v4c ile AYNI, tek değişken `--peak-lr 1e-4 → 1,5e-4`

## 1. Üç kapı

| Kapı | Ölçüm | Eşik | Hüküm |
|---|---|---|---|
| **K1 okuryazarlık** | SFT val CE **2,221915** (PPL 9,2249) | ≤ 2,45 | **GEÇTİ — seri rekoru** (pure 2,3043, v4c 2,2345) |
| **K2 bilgi** | resmi VAL pencere-CE **3,2636** / PPL 26,1424 (`eval_ce_val.py`, 500 pencere, tohum 1337) | ≤ 3,23 | **KALDI — 0,0336 nat** (pure 3,8205 → −0,5569; bar 0,59 ister) |
| **K3 nöbet** | `test_model.py`: root 1,0000 · TENSE 0,9997 · case 0,9986 · **PLURAL 0,9696** | 4/4 ve ≥0,99 | **KALDI — PLURAL 0,0219 eksik** (v4c 0,9721) |

**Hüküm: 1/3.** v4c'den iki eksende de ileri: K2 0,0651 → **0,0336** (yarıya indi), K1 2,2345 → 2,2219.

## 2. Nöbet eğrisi (v4c ile karşılaştırma)

| adım | v5 pencere-CE (TRAIN avlusu) | v4c aynı nokta |
|---|---|---|
| 0 | 3,9633 | 3,9633 (birebir) |
| 2000 | 3,4297 | 3,4348 |
| 6000 | 3,3410 | 3,3688 |
| 10000 | 3,3218 | 3,3479 |
| 11380 (final) | **3,3197** | 3,3495 |

Kuyruk: son 3.000 adım eğimi **−0,00146 nat/1000** → yakınsamış (v4d sonda dersi aynen: devam ekseni kapalı).
K1 yolu: 2,3043 → 2,3329 (@500) → 2,2219 — nöbet hiç tetiklenmedi, pay 0,228 nat kullanılmadı.

## 3. Ölçülen LR yanıtı (bu koşumun asıl çıktısı)

| lr | resmi VAL | not |
|---|---|---|
| 1e-4 | 3,2951 | v4c (çıpalı) |
| **1,5e-4** | **3,2636** | **v5 (çıpalı)** → +0,5e-4 için **−0,0315** |
| 2e-4 | 3,1936 | T3.4 (**çıpasız**, okuryazarlık öldü) |

Doğrusal ekstrapolasyon, çıpalı 2e-4: **3,2321** — barın **0,0021 üstünde**. Yani LR ekseni de barın hemen dibinde doyuyor; çıpa↔külliyat faturası (~0,10 nat) K2 barının (T3.4'ün çıpasız 3,1936'sından türetilmiş) içine sığmıyor.

## 4. Teşhis

1. **Çıpa K1'i korudu, bilgiyi ~0,07 nat pahalıya aldı** (T3.4 çıpasız 3,1936 ↔ v5 çıpalı 3,2636; lr farkı 2e-4↔1,5e-4).
2. **K2 barı çıpalı rejimde ulaşılabilir bandın hemen dışında:** üç lr noktası + bir 2. epok sondası ölçüldü; kalan tek hamle ya lr 2e-4 (ekstrapolasyon barın 0,002 üstü) ya karışım ağırlığını K1 payından harcamak.
3. **K3 PLURAL ceza ekseni ayrı:** K1 iyileşirken PLURAL 0,9721 → 0,9696 geriledi — nöbet güveni okuryazarlık CE'siyle birlikte hareket etmiyor; ayrı bir mekanizma (muhtemelen PLURAL karar sınırının çok-şablonlu eğitim dağılımındaki marjinalliği).
4. **Ders (kalıcı):** "N adım daha" değil, "bir lr kademesi = ~0,03 nat"; ölçülen yanıt eğrisi olmadan tam koşum başlatma.

## 5. Açık seçenekler

- **v6:** lr 2e-4, çıpa ve karışım AYNEN (~10 sa). Beklenti 3,2321 ± 0,005 → **yazı-tura, muhtemelen kıl payı KALDI**.
- **Karışım ekseni:** lr 1,5e-4 + 12P+4S (çapa payı %19,9→%13,3) — K1 payının 0,05'ini harcar, K2'de 0,03–0,05 getirebilir; iki değişken birden.
- **B:** v5'i iki kapı istisnasıyla terfi (K2 0,0336 + K3 0,0219). Dayanak: K2 barın %94,3'ü, K1 seri rekoru ve iyileşerek geçti, ağırlıklar semantik bilgi taşımıyor → bilgi vektör katmanında.
