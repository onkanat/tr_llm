# ANKA · ÖN-KAYITLI SON TUR — yetkin-ama-yeteneksiz temel + eğitim bozukluğu tanısı

**Damga:** 22 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Operatör: *"son tur; bundan sonra
köklü eğitim ya da proje kararları verilecek."* Bu tur **karar üretecek iki ölçüm** yapar.

---

## KOL A — H-A'yı kapatmak: `seg_1` (yetkin ama yeteneksiz) temelinde modül

Neden bu temel: bugüne kadarki bütün "içerik gelmedi" sonuçları **iki uçta** alındı —
taban **hiç yetkin değildi** (r28–r32: tutarsızlık %88–100) ya da taban **zaten yetenekliydi**
(r33: `seg_6`, kesişim %10). **Ortası hiç sınanmadı.**

`seg_1` tam o orta noktadır (T-0096, kanonik kap, ölçüldü):

| | seg_1 |
|---|---|
| ROUGE-L | 0,1097 ± 0,0971 |
| tutarsızlık | **%24** (taban %100'e karşı ⇒ kısmen talimat izliyor) |
| **kesişim** | **%0** (Wilson [0 ; 3,7]) ⇒ **içerik YOK** |
| ezber | %20 · A CE 4,3592 · sha256 `8cede239a60a…` |

Yükleme ön-ölçüldü: `KristalLM`'e **0 eksik / 0 fazla**.

**Tek değişken: temel.** Modül tarifi r30/r33 ile birebir aynı (37 katman, `r`=16, 1000 adım,
lr 2e-4, tohum 43, aynı karışım).

### Kol A'nın ilan edilen okumaları

| # | okuma | EKLENDİ ise | EKLENMEDİ ise |
|---|---|---|---|
| **1 (birincil)** | **kesişim** | **≥ %10** (Wilson [5,5 ; 17,4] ⇒ seg_1'in [0 ; 3,7]'siyle **AYRIK**) | **≤ %3** |
| 2 | ROUGE-L | ≥ **0,138** (seg_1 + 2 SE) | ≤ 0,110 |
| 3 | tutarsızlık | < **%10** | ≥ %24 |

**Yorum anahtarı (önceden):**
* **EKLENDİ** ⇒ **H-A desteklendi**: modül başarısızlığının nedeni **tabanın yetkinsizliğiydi**;
  yol "**önce çalışan temel, sonra modül**" ve köklü karar **o yönde** verilir.
* **EKLENMEDİ** ⇒ **H-A ELENDİ**: üç farklı yetkinlik düzeyinde (tutarsızlık %100 / %24 / %5)
  modül **hiç içerik eklemedi** ⇒ kısıt **mekanizmadır** (rank-sınırlı delta), köklü karar
  **modül mimarisini bilgi için terk etmek** yönünde verilir.

## KOL B — "Temeli doğru eğittik mi?" tanısı (eğitimsiz, saf çıkarım)

**Ayırt edilecek iki şey** (operatörün sorusu):

* Taban **kendi dağılımında** (Wikipedia) akıcı üretiyor ama **talimat zarfında** çöküyorsa
  ⇒ taban bir **dil modeli olarak SAĞLAM**; eksik olan **talimat aşamasıdır** → *bozukluk değil,
  eksik aşama*.
* Taban **kendi dağılımında da** döngüye giriyorsa ⇒ taban eğitimi **BOZUK**tur ve üstüne
  kurulan her şey boşa gider.

`scripts/taban_akicilik_tanisi.py` (bu turda yazıldı, saf çıkarım) dört modelde koşar:
**taban** · **seg_1** · **seg_6** · **seg_6 + modül** (r33'ün onarıcı modülü).

Ölçütler ve sınırlar (ÖLÇÜMDEN ÖNCE):

| ölçüt | SAĞLAM | BOZUK |
|---|---|---|
| `benzersiz_4gram` (üretilen 4-gram benzersizlik oranı) | **≥ 0,90** | < 0,70 |
| `dongu_orani` (bir 4-gramı ≥3 kez tekrarlayan örneklerin %'si) | **≤ %20** | ≥ %60 |
| `ppl` (A ekseni pencerelerinde) | kaydedilir | — |

İkisi çelişirse (biri sağlam, öbürü bozuk) hüküm **KISMİ** olur ve **örnek üretim metni** yazılır.

## Kaydedilir, hükme girmez

* Kol A'nın **A/B unutma eksenleri** (baseline = `seg_1`'in kendisi).
* `ezber` (seg_1'de %20; artış tek başına başarı değil).

## Dokunulmazlar

`data/**` (donmuş) · `CLAUDE.md` · kapanmış `data/eval/anka_r17…r33*` ·
**`scratch/t0096_*` (seg_1/seg_6 SALT OKUNUR — üzerine YAZILMAZ)**. `src/**` değişmez.
`git add -A` yasak; commit yok.

## Uygulama sırası

1. Kol B × 4 (saf çıkarım, ≈6 dk)
2. Kol A: `train_module.py --base scratch/t0096_kos/seg_1.pt …` → `modul_ile_olcum.py`
   (`--model` ve `--baseline` = `seg_1`) → `data/eval/anka_r34_son_tur_*.json`
