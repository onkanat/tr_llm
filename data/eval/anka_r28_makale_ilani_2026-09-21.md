# ANKA · ÖN-KAYITLI TEK DENEME İLANI — makale komşuluğu (T-0067, r27'nin devamı)

**Damga:** 21 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 1. Sorulan soru — r27'nin GEÇERLİLİĞİ

`anka_r27` (bkz. `anka_r27_wikireplay_yetenek_2026-09-21.md`) unutma kapısını **ilk kez**
geçti: A CE 3,5352 → **3,7658** = **+%6,52** (eşik ≤ +%10). Replay alanı chat → Wikipedia
yapılmıştı.

Ama r27'nin sızıntı kapısı **yalnız birebir blok** çakışmasını dışladı (`eval_blok_haric`
kuralı, 256 pencere bloğu atıldı). Geriye kalan tehlike: eğitime giren bir Wikipedia bloğu,
bir eval penceresiyle **aynı makaleden** olabilir ⇒ o zaman A CE'deki düşüş **koruma** değil
**aynı makaleyi hatırlama** olur.

### ÖLÇÜLEN TEMAS (bu ilandan önce, tek seferlik yeniden kurma ile)

`build_replay_mix.py:74`'ün seçimi birebir yeniden üretilerek r27'nin karışımındaki 2.044
replay bloğunun kaynağı bulundu:

| | blok | oran |
|---|---|---|
| eval penceresini İÇEREN makaleden gelen | **19** | **%0,93** |
| farklı makaleden gelen | 2.025 | %99,07 |

⇒ Temas **sıfır değil, küçük.** Bu bir **yargı** — "0,93% bir şey ifade etmez" demek ölçüm
değildir. Bu ilan onu ölçüme çevirir.

## 2. Tek değişken

**Dışlama kuralı.**

| | r27 (referans) | **r28 (bu deneme)** |
|---|---|---|
| kural | `eval_blok_haric` | **`eval_makale_haric`** |
| atılan blok | 256 | **7.521** (306 makale + pencere blokları) |
| dilim jetonu | 99.967.232 | 99.037.312 |
| **makale teması** | **19 / 2.044 (%0,93)** | **0 / 2.044 (%0,00)** — denetimle doğrulandı |
| ceket kaynağı · desen · oran · blok | `…_anka.bin` · `[ceket×3, replay]` · %25 · 128 | **aynı** |
| karışım jetonu · ceket · replay | 1.046.429 · 784.797 · 261.632 | **aynı** |
| mix tohumu · eğitim tohumu · lr · r · adım | 42 · 43 · 2e-4 · 16 · 1000 | **aynı** |

Yalnız **hangi Wikipedia bloklarının replay olabileceği** değişiyor.

**Doğrulanmış (koşumdan önce):**
* Yeni kod, r27 dilimini **bit-özdeş** yeniden üretti (sha256 `49f88cb8d13f42cd…` = aynı) ⇒
  refactor donmuş artefakta karşı sınandı.
* r28 karışımının denetimi: **birebir blok kesişimi 0** · **makale teması 0 / 2.044**.
* r27'nin dilim meta'sında `kural` alanı **yok** (o zaman kapı yoktu) ⇒ r27 karışımı yeni
  denetimle **yeniden denetlenemez**; betik bu durumda `rc=2` ile durur (fail-closed,
  ölçüldü). r27'nin 19 bloğı yukarıdaki tek seferlik yeniden kurmayla ölçüldü.
  **r27 meta'sı sonradan DOLDURULMADI** — kanıtı elle düzenlemek olurdu.

## 3. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

Referanslar (hepsi aynı kanonik kap, n=256 pencere): taban **3,5352** · r27 (blok kuralı)
**3,7658** · r24 (chat replay) **3,9953**.

İki örneklem SE'si (ölçülen std'lerden): ≈ **0,0703**.

| sonuç | hüküm |
|---|---|
| A CE **≤ 3,8361** (r27'den fark < 1 SE) | **KOMŞULUK SIZINTISI YOK** — +%6,52 gerçek koruma |
| A CE **≥ 3,9064** (fark ≥ 2 SE) | **SIZINTI VAR** — düşüşün bir kısmı aynı-makale hatırlaması |
| 3,8361 < A CE < 3,9064 | **AYIRT EDİLEMEDİ** |
| A CE **≥ 3,9600** | tüm kazanç komşuluktan; **kapı yeniden düşer** (artış ≥ +%12,0) |

Her dal **mümkün ve anlamlıdır** ⇒ bu bir ayar kovalaması değil, **geçerlilik sınamasıdır**.

## 4. Ne okunur

* **Hüküm A ekseni üzerinedir.** B ve yetenek eksenleri kaydedilir, hükme girmez.
* **Vakum kapısı** ve **taban dokunulmazlığı** zorunlu (r27'de ateşlendi).
* Beklenen yön ilan **edilmez** — hangi dal çıkarsa o yazılır.

## 5. Uygulama

1. ✅ ~~`--kural eval_makale_haric` ile dilim~~ → `scratch/t0099_wiki_replay_makale.bin` (773.729 blok)
2. ✅ ~~karışım~~ → `scratch/t0099_wiki_mix_makale.bin` (1.046.429 jeton)
3. ✅ ~~denetim~~ → blok 0 · makale 0
4. ⏳ `train_module.py --data scratch/t0099_wiki_mix_makale.bin --steps 1000 --lr 2e-4 --seed 43 --block-size 128 --device mps`
   → `modules/marangoz_makale_1000.mod.pt`
5. ⏳ `scripts/modul_ile_olcum.py` → `data/eval/anka_r28_makale_yetenek_2026-09-21.json`

**DOKUNULMAZ:** `data/**` (donmuş `data/*.bin` dahil) · `src/**` · `CLAUDE.md` · kapanmış
`data/eval/anka_r17…r27*` · `scratch/t0096_*` · `scratch/t0097_*` · r27'nin dilim meta'sı.
Karışım **scratch/**'e yazılır (`data/train_f4_replay_mix.bin` donmuş — ölçüldü).
