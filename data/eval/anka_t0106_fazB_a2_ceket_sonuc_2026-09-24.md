# T-0106 FAZ B+C SONUÇ — a2-tabanlı ceket koşumu (TEK DEĞİŞKEN: r1→r2 derleme)

**Tarih:** 2026-09-24 · **Görev:** T-0106 · **İlan:** `anka_t0106_fazB_a2_ceket_ilan_2026-09-24.md`
**Koşum:** `scratch/anka_t0106_kos/` · rc=0 · 3.059,7 sn · HELD-OUT çıpası birebir korundu.

---

## 1. Koşum sonuçları (sonda JSON'dan, elle sayı yok — çift-hüküm deseni)

| Eksen | P3 çıpa (a1r/r1, seg 3) | **T-0106 (a2/r2, seg 3)** | Eşik | Hüküm |
|---|---|---|---|---|
| ROUGE-L (RAW, hüküm) | 0,1014 | **0,1050** | ≥ 0,3221 | **ALTINDA** (beklenti ilanı: band-üstü beklenir, VAAT EDİLMEZ) |
| ROUGE-L (DECOMP, tanısal) | yok | 0,1704 (aday 0,7988 bağlanmaz) | — | tanısal |
| Wikipedia CE (ppl) | 3,4931 (32,89) | **3,3622 (28,85)** | ≤ 3,8887 | GEÇTİ (tavan payı %13,6) |
| LM bedeli A artış | +1,43% | **+1,03%** | ≤ +10% | GEÇ · unutma GEÇ |
| B top-1 düşüş | +0,04 | **−2,82** | ≤ 5,0 | GEÇ (taban +3,13 ile birleşince taban-üstü) |
| ezber | 2,0 | **1,0** | < 10 | GEÇ |
| tutarsızlık | 40,0 | **31,0** | < 5 | ALTINDA (P3 deseni korunuyor; P5-A tanısal) |
| kesişim | 2,0 | **3,0** | ≥ 11 | ALTINDA (kesişim eşiği insan tavanı üstü) |
| seg 1/2/3 ROUGE | 0,0875 → 0,0968 → 0,1050 | — | — | TEPE 3/3 geçti (gerileme YOK) |

**Hüküm (f(metrik, ilanlı eşik) — betik yazdı, elle YOK):** ezber GEÇ · ROUGE FAIL ·
tutarsızlık FAIL · kesişim FAIL · A/B/unutma GEÇ. **Eski kayıtlar düzenlenmedi.**
YORUM: P3 hükmü aynen durur — 6.000 adımda ROUGE 0,105 ≈ P3 çıpası 0,1014 (+0,0036,
gürültü-bandı; "değişti ≠ iyileşti" dersi). Boşluk külliyat arzındadır (P4 hükmü);
bu koşum külliyat vaat ETMİYORDU (ilan §6). CE a2 tabandan +1,0% (sağlık korunuyor).

## 2. Kapı kanıtları (fail-closed)

* **VOCAB-UYUM KAPISI** iki negatif dal ÖLÇÜLDÜ (beyan: `scratch/anka_t0106_vocab_uyum_negatif_beyan.json`):
  çatlak dalı (r1 wiki + r2 sürücü) → DUR rc=2; alan-yok dalı (eski meta şeması) → DUR rc=2.
  Pozitif: koşum başlangıcı `[VOCAB-UYUM] 3 bin meta sozluk_sha256 == sürücü 14ce9f4e… (33.911)`.
* Ceket r2 derleme: 13/13 V kapısı PASS, rc=0 (V5 carve T-0094 digest'leri birebir,
  V6 sızıntı 0, V10 korunanlar dokunulmadı). Kayıt 22.283 birebir; jeton 1.568.123 → 1.569.200 (+1.077).
* SFT r2 derleme: 125.814 kayıt / 16.104.192 jeton **BİREBİR** (r1 meta çıpasıyla);
  kırpma 1.041 (%0,83), zarf kapısı geçti; sızıntı 0.
* Carry: `anka_a2.pt.opt.pt` YOK ⇒ optimizer SIFIRDAN (T-0092/G4a, ilan §1 beyanlı; L1/CE/TEPE tümü geçti).
* Koşum kaynağı dağılımı: wiki 9.601 (%20) · ceket 19.200 (%40) · SFT 19.200 (%40) — patern birebir.

## 3. DECOMP ikinci temsil (Faz A geçişi — tanısal, hüküm dışı)

Nihai sonda: DECOMP ROUGE 0,1704 (RAW 0,1050) · decompile istisna 0 · yuzey [?] 7 ·
`ham_gm` 100/100 dolu (kalıcı yeniden-ölçülebilirlik). Aday eşik 0,7988 hüküme BAĞLANMADI.

## 4. Beyanlı sınırlar ve kusurlar

* **İlan dosyası adı:** koşum kaydı ve ceket CKPT `…_ilan_…` referansı taşıyordu; diskte `…_ilani_…`
  yazılmıştı. İlan dosyası kayıtlı referans adına TAŞINDI (tek değişiklik; içerik sha aşağıda, yeniden adlandırma sonrası ölçüldü).
* Ceket r2 meta `ureten_betik` alanı r18 taban betiğini gösterir (betik içi literal — ilan §6 beyanlı);
  gerçek üretici CKPT `_beyan.json`'da kayıtlı.
* ROUGE farkı küçük ve işareti gürültüden doğrulanamaz — iyileşme İDDİA EDİLMEZ.
* 12.000 adım tekrarı YAPILMADI (P4 hükmü: epoch arzı doydu).

## 5. Artefakt digest tablosu (tam — önek karşılaştırması YOK)

| sha256 (ilk 16) | Artefakt |
|---|---|
| 8e658db8fd32a58c | data/eval/anka_t0106_fazB_a2_ceket_ilan_2026-09-24.md |
| 26ac1dfd960d2527 | data/eval/anka_t0106_fazA_decomp_rouge_sonuc_2026-09-24.md |
| e1b81b064b32f66c | data/eval/anka_t0106_decomp_tavan_r2_2026-09-24.json |
| 9f02e1e7b86759b3 | data/eval/anka_t0106_duman_a1r_n20.json |
| 3e080cb20b3d9ee1 | data/eval/anka_t0106_ceket_r2_build_2026-09-24.json |
| 5e06dcf97424ea9e | data/eval/anka_t0106_ceket_r2_build_2026-09-24_beyan.json |
| 5590a9abb207a49b | data/train_carpenter_specialization_anka_r18_v2.bin |
| c133a9e55da977fc | data/train_carpenter_specialization_anka_r18_v2.bin.meta.json |
| d86681574ddee32b | data/rebuild/anka_t0106_sft_mix_r2.bin |
| 638a2dab3942cd7b | data/rebuild/anka_t0106_sft_mix_r2.bin.meta.json |
| ec5b31f6614d471c | scratch/anka_t0106_ceket_r2_build.py |
| 043d9a56d8fe6256 | scratch/anka_t0106_sft_r2_build.py |
| 78ac3dec9edf7ef0 | scratch/anka_p3_surucu.py (genişletildi) |
| 7ff21b0e6f95dccc | scratch/anka_t0106_vocab_uyum_negatif_beyan.json |
| 72581ba4a718dc8d | scratch/anka_t0106_kos/sonuc.json |
| df810970902ea677 | scratch/anka_t0106_kos/sonda_seg_3.json |
| 7b001451c1f86c54 | scratch/anka_t0106_kos/seg_3.pt (nihai checkpoint) |
| e134cb84c8c0943e | scripts/anka_karsilastirma_tablosu.py (t0106 satırı) |

**Durum: FAZ B+C KAPANDI.** T-0106 tamamı: Faz A (DECOMP ölçüt geçişi) + Faz B (a2/r2 koşum)
+ Faz C (bu rapor). Açık iş (operatör kararı): yeni ceket külliyat arzı üretimi — mevcut
havuzlar tükendi; ROUGE boşluğunun 0,3221'e kapatılması bunun dışında ÖLÇÜLDÜ (band 0,10).