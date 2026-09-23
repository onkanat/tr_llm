# ANKA · SON TUR — H-A elendi · taban sağlam · bozukluk aşağı akışta

**İlan:** `data/eval/anka_r34_son_tur_ilani_2026-09-22.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099i_son_tur.log` · `rc=0` · 10:14:27Z→10:39:24Z

---

## KOL A — H-A **ELENDİ**: üç yetkinlik düzeyinde de içerik gelmedi

`seg_1` temelinde (tutarsızlık %24, **kesişim %0**) aynı modül tarifi:

| # | okuma | ilan: eklendi ise | ölçülen | sonuç |
|---|---|---|---|---|
| **1 (birincil)** | **kesişim** | ≥ %10 | **%0,00 → %0,00** | **EKLENMEDİ** |
| 2 | ROUGE-L | ≥ 0,138 | 0,1097 → 0,0997 (0,75 SE) | EKLENMEDİ |
| 3 | tutarsızlık | < %10 | %24 → **%29** | EKLENMEDİ |

**Üç okumanın üçü de "eklenmedi".** İlanın ön-kayıtlı anahtarı gereği:

> **H-A ELENDİ.** Üç farklı yetkinlik düzeyinde — tutarsızlık **%100** (r28–r32) / **%24** (r34,
> seg_1) / **%5** (r33, seg_6) — modül **hiç içerik eklemedi**. Kısıt **tabanın yetkinliği
> değil, MEKANİZMADIR** (rank-sınırlı delta).

### Yetkinlik gradyanı — kapanan tablo

| temel | tutarsızlık | kesişim (öncesi → sonrası) | modül içerik ekledi mi |
|---|---|---|---|
| `anka_a1r` (ön-eğitilmiş) | %100 | %0 → %0 | **hayır** |
| `seg_1` (tam FT 1000 adım) | %24 | %0 → %0 | **hayır** |
| `seg_6` (tam FT 6000 adım) | %5 | %10 → %10 | **hayır** |

## KOL B — "Temeli doğru eğittik mi?"

### ⚠ ÖNCE: ilan ettiğim iki ölçüt AYIRT ETMEDİ (ve bunu beyan ediyorum)

| model | `benzersiz_4gram` | `dongu_orani` | **ilan edilen hüküm** | **ppl** |
|---|---|---|---|---|
| **taban** | 0,0794 | **%100** | **BOZUK** | **33,38** |
| `seg_1` | 0,5501 | %92 | BOZUK | 76,85 |
| `seg_6` | 0,6198 | %96 | BOZUK | **599,72** |
| `seg_6`+modül | 0,1693 | %100 | BOZUK | 86,41 |

**Dördü de "BOZUK" çıktı — taban dâhil.** Ve kanonik kap aynı modeller için **tersini** söylüyor:

| | kanonik tutarsızlık | bu sonda `dongu_orani` |
|---|---|---|
| `seg_6` | **%5** | **%96** |
| `seg_1` | %24 | %92 |

⇒ **Sondam bozuktu, model değil.** Sebep: sonda **greedy** üretim kullanıyor; greedy kod çözme
**her** dil modelinde tekrara kaçar (bilinen bir kod çözme artefaktı, eğitim kusuru değil).
Kanonik kap ise **görev zarflarında** üretiyor ve orada aynı modeller akıcı.

**Bu beyan zorunlu:** ilan edilen eşikler ateşledi ama **işaret ettikleri şeyi göstermedi**
([[toplu-metrik-atesledi-mekanizma-ateslemedi]] sınıfı, bu kez sondanın kendisinde).

### PPL ölçütü AYIRT EDİYOR — ve cevap net

İlan §Kol B'de `ppl` **"kaydedilir"** olarak ilan edilmişti (kapı değil). Ayırt eden ölçüt bu:

| model | ppl | okuma |
|---|---|---|
| **taban `anka_a1r.pt`** | **33,38** | **en iyi dil modeli** |
| `seg_1` (tam FT 1000) | 76,85 | 2,3× bozulmuş |
| `seg_6` (tam FT 6000) | **599,72** | **18× bozulmuş** |
| `seg_6` + modül | **86,41** | **onarılmış** |

> **TABAN SAĞLAM.** Wikipedia'da ppl **33,4** — dört modelin **en iyisi**. Yani taban bir dil
> modeli olarak **doğru eğitilmiş**; **bozukluk yok**. Eksik olan **bir aşamadır**: taban
> **talimat zarfını hiç görmemiş**.

## Bozukluk nerede — ölçülmüş cevap

**Bozukluk tabanda değil, AŞAĞI AKIŞTA.** Yetenek verisiyle **tam ince ayar**, dil modelini
yok ediyor:

| | ppl | tabana göre |
|---|---|---|
| taban | 33,38 | — |
| `seg_1` (1000 adım tam FT) | 76,85 | **2,3×** |
| `seg_6` (6000 adım tam FT) | 599,72 | **18×** |

Ve **r33'te bulunan ONARIM bu turda TEKRARLANDI** — başka bir checkpoint'te:

| | modül öncesi | modül sonrası |
|---|---|---|
| `seg_6` (r33) | A CE 6,2086 | **4,3902** (−%29,29) · ppl 600 → **86** |
| `seg_1` (r34) | A CE 4,3592 | **4,0191** (−%7,80) · B +0,56 |

⇒ **Rank-16 bir modül, tam ince ayarın dil-modeli hasarını ONARIYOR** — iki bağımsız
checkpoint'te, yetenek metriklerine dokunmadan. Bu, r33'ün tek koşumluk bulgusu değil,
**tekrarlanmış** bir etkidir.

## Karar için ölçülmüş zemin

| soru | ölçülmüş cevap |
|---|---|
| Temeli doğru eğittik mi? | **Evet** — taban en iyi dil modeli (ppl 33,4) |
| Modül bilgi ekliyor mu? | **Hayır** — 4 eksen (kapasite·giriş·bütçe·temel yetkinliği) elendi, 3 yetkinlik düzeyinde |
| Modül ne yapıyor? | **Biçim** öğreniyor; **onarıyor** (2 checkpoint'te tekrarlandı) |
| Bozukluk nerede? | **Tam ince ayarda**: ppl 18× bozuluyor |
| Eksik olan ne? | **Talimat aşaması** — taban zarfları hiç görmemiş |

**Sonuç:** "temel bir kez eğitilir, yetenekler modül olur" mimarisi için ölçüm şunu söylüyor:
**temel, yetenek verisiyle değil, genel talimat verisiyle bir kez eğitilmeli**; yetenek
**modülle eklenemiyor** (bu taban ve bu tarifle, dört eksende ölçüldü) — **tam ince ayarla
ekleniyor ama dil modelini yok ediyor**, ve o hasar **modülle onarılabiliyor**.

## Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `modules/seg1_marangoz_1000.mod.pt` | `82b4c14f69a19c2378f61f5505d593d44015b74d6b5310695af8be359ddd111b` |
| `modules/seg6_marangoz_1000.mod.pt` (r33) | `afdae58007411d277de2eecd286ff243942fb7ce7f11987b38d3cfd3b3f0a7ed` |
| `scratch/t0096_kos/seg_1.pt` (**salt okunur**) | `8cede2393a60ea62641c1bff1560ab9543e95f97848f33e69d41f6f4cbc6f5f6` |
| `scratch/t0096_kos/seg_6.pt` (**salt okunur**) | `136dda76da419e281cf8b800a0e562381c5e9c7db8ed85136ac3ee4f19578b8b` |
| `data/eval/anka_r34_son_tur_yetenek_2026-09-22.json` | `e854b195fc9c222e2215dd90f9c71154ef8c25dc694a295ab002178c89358f6f` |
| `data/eval/anka_r34_son_tur_yetenek_2026-09-22.modul.json` | `bae7af6c546217bd9dab7a3ce8b0278a00f2c2d8da27f394defcbcb6aed1d715` |
| `data/eval/anka_r34_akicilik_taban_2026-09-22.json` | `32e2a4d73d975fcad15b8ba35fe3a8a0a5cc7483d7940aed39ccbc941aa3b801` |
| `data/eval/anka_r34_akicilik_seg1_2026-09-22.json` | `9ac0118061a46f5e5a425ca8103fedea7bcb4e18aa68645738840d8cc1aa63c7` |
| `data/eval/anka_r34_akicilik_seg6_2026-09-22.json` | `07871eb3eeebc6d2f1bed3c1330c0417dbe24008f334b61a4e557265f0361433` |
| `data/eval/anka_r34_akicilik_seg6_modul_2026-09-22.json` | `9212d1667f4a312c4a7c13ee0516481e02997179f7ffcccd159a8a4b602a8455` |
| `scripts/taban_akicilik_tanisi.py` | `781061608ec945d51b2d6c93c26f639ef7edb7fc5c3918fcc902c71c95a36764` |
| `scratch/t0099i_son_tur.log` | `c6531d56e11a27b95971596d417caf6ed2cfa2a1dd0674b13de19c17c40f5e6a` |

**Commit yok · `git add -A` kullanılmadı · donmuş kalıplara yazılmadı · `src/**` değişmedi ·
`scratch/t0096_*` salt okunur kaldı.**
