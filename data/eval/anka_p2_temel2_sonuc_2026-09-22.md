# P2 · AŞAMA 2b — TEMEL 2.0 KOŞUM SONUCU (22 Eyl 2026)

**Damga:** 22 Eyl 2026 20:08 UTC bitiş · rc=0 · İlân:
`anka_p2_temel2_ilani_2026-09-22.md` (koşumdan önce yazıldı, sha256 `4204ae7c…`).

## 1. Koşum kanıtı (üç kapanış sinyali + kapılar)

* `SONUC_BITTI` sentinel sha256 = `seg_4.pt` digest'i **birebir** ✓ · arka plan koşumu
  **rc=0** ✓ · `sonuc.json` `durum: BITTI` ✓ (kısmi-sonuç kanıtı DEĞİL).
* **4/4 segment CE kapısı GEÇTİ** (tavan 3,8887; taban zemin 3,5352) — **monoton
  iyileşme**, LM hiç bozulmadı:

| segment | adım | CE | ppl | kayıp (son) | LR (son) |
|---|---|---|---|---|---|
| taban zemin | — | 3,5352 | 34,4 | — | — |
| seg_1 | 5.000 | 3,4779 | 32,39 | 3,9862 | 1,00e-4 |
| seg_2 | 10.000 | 3,4532 | 31,60 | 3,7938 | 8,6e-5 |
| seg_3 | 15.000 | 3,4449 | 31,34 | 3,6057 | 5,1e-5 |
| seg_4 | 20.000 | **3,4427** | **31,27** | 3,4742 | 1,0e-6 |

* Toplam 9.510 sn (2,64 saat) · 0,42-0,46 sn/adım sabit band · H1 (0 tutucu) ·
  L1 geçti (ilk kayıp 0,0/≈ln V değil) · carry: taban momenti YOK ⇒ açık uyarıyla
  sıfırdan optimizer (ilan §3 beyanıyla uyumlu) · boş-hedef pencere 2→6 (AÇIK sayım).
* CE seg_1'de zaten tabanın ALTINA indi — talimat karışımı wiki dilini korudu.

## 2. Kalitatif zarf sondası (ilan §5) — DÜRÜST HÜKÜM: AYIRT EDİCİ DEĞİL

Held-out 3 girdi, kanonik üretim yolu (`render_prompt` + `ECA.uret`, T-0094/K7 kırpma
kapısı; lexicon yüklü — ölçüm-kabı dersi):

| istem | taban | seg_4 |
|---|---|---|
| şerit testere tanımı | 24× `<UNK>` zinciri | tek morfem `<PROPER_NOUN>:POSS_3PL` |
| iskarpela parlatma | `,` + 24× `<UNK>` | `bir` + `<UNK>` zinciri |
| yarı-goyuntulu bindirme | `,` + 23× `<PROPER_NOUN>` | morfolojik dizilim (`ve … gibi çeşit DERIV_lI bir yer CASE_DAT sahip COPULA_AORIST.`) |

* **`[?]` hiçbir modelde çıkmadı** ⇒ ilanlı kontrolün taban tarafındaki pozitif referansı
  bu kabinde ölçülemedi; sonda "**düzeldi**" hükmünü TAŞIMAZ (gösterim-uyuşmazlığı
  ölçütü öldürür + kapı-örnelemi-zayıfsa dersleri). Beyan: taban ayraç+yer-tutucu
  tekrarını, seg_4 farklı yapıları basıyor — fark VAR, içerik YOK.
* **Karar:** hüküm Aşama 3'ün kanonik kaba ertelendi (LCS-kesişim 10,92 · ROUGE 0,35 ·
  tutarsızlık 5,0 · ezber 10,0 — Aşama 0 kalibrasyonu). Koşum kabulü CE kapısına
  bağlanmıştı; CE kabulü SAĞLANDI.

## 3. Digest tablosu

| artefakt | sha256 |
|---|---|
| `scratch/anka_p2_temel2_kos/seg_4.pt` (nihai Temel 2.0) | `c786f118d91a5c30b6d23dce210602e90b570b48cdec19ebc7401a2c8e6fad86` |
| `…/seg_4.pt.opt.pt` (optimizer+scheduler carry, DAL-1 girişi) | `f8e844f68d8fc751e85b289f648416fe48579d2161c39d8952e1eb714a74fb67` |
| `…/sonuc.json` | `9b83a98cec38423495d2593b71f52a1203f90d4f316e109c0c9c6f6b1a119472` |
| `scratch/anka_p2_temel2_surucu.py` | `d864e156799c8b43f5cfd22194e1cadae33d3acc5ffdb8bd35f1fdaecd26d283` |
| `data/eval/anka_p2_temel2_ilani_2026-09-22.md` | `4204ae7c942406d059ffc4151877c5d26baa36bf7c39df0c2107e380b16ad7b1` |

## 4. Aşama 2 durumu ve sonraki

**Aşama 2 KAPANDI (kabul sağlandı).** Temel 2.0 = `seg_4.pt`; soyağacı: `anka_a1r.pt`
(`b93cc1cd…`) + P2 karışım 20K adım. DAL-1 (wiki payı artışı) ilanlıydu ama koşul OLMADI —
CE hiç düşmedi.

**Aşama 3 — Yetenek aşaması (sıradaki):** Temel 2.0 üstünde kısa tam ince ayar (lr 1e-4,
scheduler'lı, ~2.000 adım — ilanla kesinleşir) + onarıcı modül (r16 LoRA deseni, rol:
biçim/onarım); ölçüm kanonik kapta **kalibre eşiklerle** (kesişim 10,92 · ROUGE 0,35 ·
tutarsızlık 5,0) + `modul_ile_olcum` vakum kapısı. Önce kanonik kapta Temel 2.0 zemini
ölçülmelidir (Aşama 3'ün sıfır noktası).

**Kalan borç:** i1 eski seg checkpoint'leri ~2,1G (disk); Temel 2.0 ara yan dosyaları
(seg_1-3 `.opt.pt`) DAL-1 iptaliyle taşıyıcısız — temizlik operatör kararına.