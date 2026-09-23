# P4 · CEKET EPOCH ARZI KOŞUMU SONUCU (23 Eyl 2026)

**Damga:** 23 Eyl 2026 10:12 (+03, betikten) · **rc=0** · Sentinel:
`BITTI 2026-09-23T07:11:49Z nihai=scratch/anka_p4_kos/seg_2.pt
sha256=b8662629…` · sonuc.json `durum=BITTI` — kapanış üç sinyali BİREBİR.
İlân: `anka_p4_ilani_2026-09-23.md` (koşumdan ÖNCE yazıldı, sha256
`70d8edf3…`). Nihai model: **Anka P4** (`scratch/anka_p4_kos/seg_2.pt`,
P3 nihai → 12.000 adım).

## 1. Koşum kanıtı

* Süre **5.974 sn (~100 dk)** · 12.000 adım + 2 sonda · ~0,50 sn/adım ·
  sandbox DIŞI MPS · H1 tek eğitici.
* Kaynak dağılımı (ölçülmüş): wiki **19.204** / ceket **38.407** / SFT
  **38.406** = **%20,0/%40,0/%40,0** — ilanlı patern birebir.
* Carry: taban sidecar digest'i `7296aab1…` ile birebir (CARRY kapısı;
  101 parametre, adim=6000).
* L1: devam koşumu ilk kayıp ~3,39-bandı (ln V imzası YOK — model sağlam devraldı).
* **Held-out çıpası BİREBİR:** `c397eb08…` baş = son.
* **Beyanlı scheduler restart:** P3 nihai lr 1e-6 → peak 1e-4 (warmup 50 +
  cosine 12.000; ilan §2 beyanı birebir uygulandı).

## 2. Ölçüm — segment/sonda eğrisi (kanonik kap n=100, seed 42, baseline = P3 nihai)

| segment | CE (ppl) | ROUGE-L | tutarsızlık | kesişim (F1) | ezber | LM bedeli A | TEPE |
|---|---|---|---|---|---|---|---|
| P3 nihai (referans) | 3,4931 | 0,1014 | %40 | %2 | %2 | — | — |
| P4 seg_1 | 3,5104 (33,46) | **0,1332** | **%18** | %6 (0,0078) | %0 | +%0,50 | geç |
| **P4 seg_2 (nihai)** | **3,5032 (33,22)** | **0,1346** | **%14** | %2 | %0 | **+%0,29** | geç |

## 3. Hüküm (ilân §4 dallarıyla)

**P4 KAPANDI — ilanlı "ROUGE 0,15'e ulaşmazsa" dali UYGULANDI:** nihai ROUGE
**0,1346 < 0,15** ⇒ beklenti bandının (0,15-0,30) ALTINDA kaldı; program
BİTİŞ ilanına göre biter ve teşhis **"ceket epoch arzı bu tabanda
yetersiz — kalan boşluk külliyatın KENDİSİNDE"** olarak yazılır. Dallardan
hiçbiri koşulmadı (TEPE düşmedi, CE aşmadı).

**Ama ölçüm tek eksenli DEĞİL — eksen bazında ayrışma ölçüldü:**

| eksen | P3 nihai → P4 nihai | yön | hüküm |
|---|---|---|---|
| ROUGE-L | 0,1014 → 0,1346 (**+33% relatif**) | yukarı | 0,35 eşik ALTINDA; beklenti bandı altında |
| tutarsızlık | %40 → **%14** (−65% relatif) | aşağı | 5,0 eşik ALTINDA — **en güçlü hareket** |
| kesişim | %2 → %6 → %2 | n=100'de kayıt düzeyi (±4 kayıt = ±%4) | **GÜRÜLTÜ — işaret hükmü YOK** |
| ezber | %2 → %0 → %0 | sabit | < %10 GEÇTİ |
| CE (tabana birikimli) | 3,4931 → 3,5032 (**+%0,29**) | sabit | ≤ 3,8887 GEÇTİ |

* **İkinci epoch (12.000 adım) ROUGE'u kaldırmadı:** 6.000 → 12.000 adım
  ROUGE +0,0014 (gürültü altı düzlük). P3'te 1,57 epoch'ta 0,10; P4'te
  ~4,7 epoch'ta 0,1346 — **epoch arzı bu tabanda doymuş**; T-0096'nın 0,30
  bandı bu külliyatla bu eğitici mimaride yeniden üretilmedi.
* **Tutarsızlık ise eşik yolunda:** %87 (Temel 2.0) → %40 (P3) → **%14 (P4)**.
  Model artık aynı soruya tutarlı cevap veriyor; kalan tutarsızlık 5,0 eşiğinin
  ~3 katı (n=100'de 14 kayıt — gürültü payı ~±4 kayıt; gerçek hareket).
* **LM bedeli P4'te neredeyse yok:** +%0,29 (tavan %10; P3 +1,43%). Veri
  tekrarının LM'e maliyeti ölçülebilir biçimde sıfıra iniyor.

## 4. Kıyas tablosu (kalibre eşikler: kesişim 10,92 · ROUGE 0,35 · tutarsızlık 5,0 · ezber 10,0)

| eksen | Temel 2.0 | P3 tepe | **P4 tepe** | T-0096 | eşik hükmü |
|---|---|---|---|---|---|
| ROUGE-L | 0,0056 | 0,1014 | **0,1346** | 0,3035 | ALTINDA (0,35) |
| tutarsızlık | %87 | %40 | **%14** | %5 | ALTINDA (5,0) |
| kesişim | %0 | %2 | %2 (seg_1'de %6) | %10 | ALTINDA (10,92) |
| ezber | %0 | %2 | %0 | %0 | < %10 GEÇTİ |
| Wikipedia CE | 3,4438 | 3,4931 | **3,5032** | 6,2086 | +%0,29 GEÇTİ |

## 5. Sıradaki teşhis (yeni ilan konusu)

Epoch arzı ölçüldü ve doydu; LM bedeli sıfıra yakınsa da kesişim %2-6'da
kalıyor ⇒ boşluk artık **maruziyette değil, ceket külliyatının kendisinde**:
(a) külliyat çeşitliliği (22.283 kayıt, 125 olgu ×450 perspektif — perspektif
büyütme biçim öğretiyor, yeni cevapsız), (b) kesişim eşiğinin insan tavanı
ilişkisi ([[kesisim-esigi-insan-tavaninin-kati]] — Ç4 VAKUM kanıtıyla uyumlu),
(c) yeni insan verisi / yeni ceket kaynak arzı. Bir sonraki koşum ancak
**yeni külliyat arzıyla** açılır; aynı külliyatta adım/epoch/pay eksenleri
üçü de kapandı (T-0097, P3, P4).

## 6. Digest tablosu (betikle hesaplandı)

| artefakt | sha256 |
|---|---|
| `scratch/anka_p4_kos/seg_2.pt` (nihai Anka P4) | `b86626291176adb1118e79594a521eedd8b1cbd999a77321074c6cc740b18c9a` |
| `scratch/anka_p4_kos/sonda_seg_2.json` (kanonik kap) | `82a7d54a9644f576a4c2e9683561aee49cd2c5dd53f12fb25f26f75530bf1e1f` |
| `scratch/anka_p4_kos/sonuc.json` | `1a4d696a4007cf337b1bd92882ab9bcb8bd9431dbba347f2436ff4f328d19f83` |
| `scratch/anka_p4_kos/segments.jsonl` | `098b9b278d75b4377851ced435f73612ce03f10351664eb712be95dc4f9639fa` |
| `data/eval/anka_p4_ilani_2026-09-23.md` (ilân) | `70d8edf383d3396af0dc66e4a4b54d402572ff580fded8989ef847fd189a3e76` |
| taban `scratch/anka_p3_kos/seg_3.pt` | `7296aab14109cb47ddb0b08b7c32dba6da86e5ad3e16d50d037131182e0bdd3a` |
| held-out 539 (baş = son) | `c397eb08218937ea000cc6120f42a8783e1dcab7948c70b33537af74b50d096e` |

*Nihai digest iki kanaldan çıpalı: sentinel + sonuc.json aynı `b8662629…`'i yazıyor.*

## 7. Kalan (operatör kararı bekliyor)

* P4 kayıtlarının commit/push onayı (sürücü eki `--referans-sonda` + ilan + bu
  rapor; kayıtlar gitignore'da, kanıt kanalı digest tablosu).
* Disk (~1,4G): `anka_p4_kos/seg_1.pt` (ara ckpt, sonraki koşum tabanı DEĞİL)
  + `anka_p4_kos/seg_1.pt.opt.pt` — temizlik onayı gerekir.
* Sıradaki koşum yeni külliyat arzı ister — operatör kararı.