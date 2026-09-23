# P4 · CEKET EPOCH ARZI KOŞUMU İLANI (ön-kayıtlı)

**Damga:** 23 Eyl 2026 (betikten koşum başında) · **Yazım:** koşumdan ÖNCE ·
P3 teşhisinden (`data/eval/anka_p3_sonuc_2026-09-23.md` §5): T-0096'nın 0,30
ROUGE bandı ceket külliyatının ~16 epoch'unu gördü; P3'te ceket r18 yalnız
**1,57 epoch** gördü ve ROUGE 0,1014'te (seg_2→seg_3 düzleşme) tepe yaptı.
**Hipotez: kalan boşluğun en olası kaldıracı ceket epoch arzıdır** — aynı
külliyat, daha çok maruziyet (daha çok adım TEK BAŞINA T-0097'de kapandı; bu
koşum daha çok VERİ-TEKRARI ölçer, külliyat sabit).

## 1. Girdiler

| girdi | değer |
|---|---|
| taban | `scratch/anka_p3_kos/seg_3.pt` — sha256 `7296aab14109cb47ddb0b08b7c32dba6da86e5ad3e16d50d037131182e0bdd3a` (P3 nihai) |
| carry | `scratch/anka_p3_kos/seg_3.pt.opt.pt` (adim=6000; CARRY digest kapısı) |
| wiki / ceket / SFT | P3 ile BİREBİR AYNI (`anka_a1r_pretrain.bin` / `train_carpenter_specialization_anka_r18.bin` `c9d964d9…` / `anka_p2_sft_mix.bin` `233ca2e3…`) |
| held-out çıpası | `c397eb08…` baş/son birebir (V7 carve sızıntı kanıtı) |
| TEPE referansı | `scratch/anka_p3_kos/sonda_seg_3.json` — ROUGE **0,1014** · ezber %2 (yeni: `--referans-sonda`) |

## 2. Tarif (sürücü: `scratch/anka_p3_surucu.py`, `--referans-sonda` ekiyle)

* **Patern BİREBİR AYNI:** pencere idx % 5 — wiki %20 / ceket %40 / SFT %40 ·
  seed 42 deterministik ⇒ pencere dizisi P3'ün 48.000 penceresiyle birebir
  AYNI (2. maruziyet = 2. epoch etkisi) + 12.000 adım daha.
* **12.000 adım = 2 × 6.000 segment** ⇒ ceket maruziyeti: 38.400 pencere =
  3,13 epoch (bu koşum) ⇒ **P3 ile birlikte toplam ~4,7 epoch**.
* **Scheduler BEYANLI RESTART:** P3 nihai lr 1e-6'ya inmişti; bu koşumda
  warmup 50 + cosine (toplam 12.000), peak lr **1e-4** (P3 ile aynı — P3
  kanıtı: bu zarf CE tavanını korudu). Optimizer momentleri carry (T-0092).
* b8 · blok 128 · clip 1,0 · weight_decay 0,01 · `scratch/anka_p4_kos/` ·
  ara taşıyıcı silme kuralı (yalnız son segmentin opt'ı kalır).
* **Sonda:** her segment sonunda kanonik kap n=100 seed 42; **baseline = P3
  nihai** — A ekseninde P4'ün KENDİ LM bedeli ölçülür (birikim P3'e karşı).
* **TEPE kapısı:** ilk segmentte P3 seg_3 sonda referansına karşı (ROUGE
  0,1014'ten > 0,05 düşüş ⇒ DUR; ezber ≥ %10 ⇒ DUR — alan yüzde, ECA:513).
* Süre: ~12.000 × 0,43-0,51 sn ≈ 1,7-2,0 saat + 2 sonda · sandbox DIŞI MPS.

## 3. Beklenti (ölçümden, tahmin değil)

* ROUGE 0,1014 → **0,15-0,30 bandı** (T-0096 referans 0,3035 ~16 epoch'ta;
  P3+P4 = 4,7 epoch — kısmi artış beklenir; bant altında kalırsa epoch arzı
  hipotezi ZAYIF yazılır).
* tutarsızlık %40 → %20-30 bandı · kesişim %2 → artış (n=100'de kayıt düzeyi).
* **CE ≤ 3,8887** her segmentte (wiki %20 koruma payı değişmeden).
* ezber < %10 (4-gram kapısı).

## 4. Dallar (önceden ilanlı; en fazla 1)

* **TEPE düşer:** tepe segment nihai — geriye koşum yok.
* **CE aşarsa:** ceket %40→%25 + wiki %20→%35 carry ile TEK tekrar.
* **ROUGE 0,15'e ulaşmazsa:** program BİTER; teşhis "epoch arzı bu tabanda
  yetersiz — boşluk külliyatın kendisinde" yazılır.

## 5. Çıktılar

* `scratch/anka_p4_kos/` (segments.jsonl · sonuc.json atomik · sentinel rc==0).
* Kapanış: `data/eval/anka_p4_sonuc_2026-09-23.md` — digest'ler betikle.
* Beyanlar: P3 §8'deki beyanlar aynen geçerli (modül koşulmaz, SFT karışımı
  değişmez, ölçüt eşikleri değişmez, held-out eğitimde yok).