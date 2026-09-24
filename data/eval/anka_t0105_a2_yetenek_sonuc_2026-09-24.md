# T-0105 · A2 YETENEK/UNUTMA ÖLÇÜMÜ (SONUÇ)

**Damga:** 24 Eyl 2026 · **Görev:** T-0105 · **Yürütücü:** claude ·
İlan: `anka_t0105_a2_yetenek_ilani_2026-09-24.md` (Revizyon 1 → 2).

## 1. Koşum

Kabin: `scripts/evaluate_carpenter_anka.py` · cihaz **mps** (açıkça istenen;
sessiz CPU düşüşü yok) · taban-sız (Rev1 onarımı — kafa/sözlük kapısı
taban-eşli koşumu doğru durdurdu, taban çıpa JSON'dan beyanlı okunur).

| kaynak | sha256 |
|---|---|
| model `data/anka_a2.pt` | `e2352cd3bbf3e88f…` |
| vocab `data/rebuild/vocab_anka_r2.json` (33.911) | `14ce9f4e5d283c85…` |
| lexicon `data/lexicon/roots_anka_r2.tsv` (50.923 düğüm / 55.526 girdi) | `c355a372ed247271…` |
| sonuc JSON | `e5bf66e2bac64a1d…` |
| a1r çıpa JSON (p2 taban-zemin) | `data/eval/anka_p2_taban_zemin_0e_2026-09-23.json` |
| a1r model (çıpa tarafı) | `data/anka_a1r.pt` `f32d492d9361c668…` |

## 2. Hüküm (AYNI kabin a1r çıpalarıyla — ilan Revizyon 2 tablosu)

**7/7 eksen beklenti yönünde; fail-closed koşum rc=0.**

| eksen | a1r çıpa | a2 ölçüm | yön |
|---|---|---|---|
| A-ekseni CE (maskesiz, `anka_a1r_pretrain.bin`, seed 7) | 3,5352 ± 0,8032 | **3,3279 ± 0,7742** | İYİ (−0,207) |
| B-ekseni top-1 (2522 konum, seed 7) | 49,5638 | **52,6963** | İYİ (+3,13 pp) |
| çekirdek top-1 (1642 konum) | 56,3337 | 56,8819 | İYİ (+0,55) |
| ceket ROUGE-L | 0,0000 | 0,0049 | taban-çapa |
| ezber | 0,0 | 0,0 | korur |
| tutarsızlık | %100 | %100 | korur |
| kesişim | %0 | %0 | korur |

Beklenti doğrulandı: **a2'nin SFT/ceket arzı değişmedi** (ceket-eğitim
almadı) — ceket-ekseni taban-çapa bandında; A/B eksenlerinde a2 devam
eğitiminin dilim-iyileşmesi görünür. B farkı Wilson bantlarıyla:
çıpa 47,61-51,51 · a2 50,75-54,64 — 50,75-51,51'de hafif örtüşme; beyanlı
okuma "merkezi iyileşme, mutlak ayrışım kanıtı DEĞİL" (gürültü-dersi).

## 3. Ceket-ekseni tanısal

Ham üretimde `[?]` ve `[Özel İsim]` bolluğu (örnek idx 1-3) —
**ham-decode tavanı = decompiler artefaktı** dersinin a2 kopyası:
ROUGE 0,0049'ın çoğu ham kayıp; DECOMP aşamalı üretim ayrı iş (T-0104
sonuç §4'te aynı beyan). Üretim zarfı: `<OUTPUT>` 1706 / `</OUTPUT>` 1705
· `encode→<EOS>` kırpılır ✓ · istem 19 jeton.

## 4. Beyanlı sınırlar

* A/B eksenleri dilim `data/anka_a1r_pretrain.bin` (a1r çıpasıyla AYNI
  dilim — eşli kıyas; a2 kendi bin'inde ölçülmedi).
* Ceket-ekseni ROUGE 0,1346 (P4) bu kabinde HÜKÜM çıpası DEĞİL — kabin
  uyuşmazlığı Rev2'de beyanlandı (Rev1 kusuru, benim çıpa yazma hatam:
  4. kez ilan-çıpa yazımında kabın denetimi).
* a2, ceket eğitimi almadığından P5 Dal-K eşiği (0,3221) bu koşumda
  hedef değildir; o eşik ceket-eğitimli modellere aittir.
* Üretim RNG cihaza bağlı — sayılar yalnız MPS koşumlarıyla kıyaslanır.

## 5. Kapanış

* İlan Rev2 + sonuc JSON sha256'ları yukarıda.
* commit `a3a5c10` (T-0101/02/03/04 kaydı + core.py bypass + r2 vocab/lexicon)
  T-0105'ten ÖNCE atıldı; T-0105 dosyaları bu raporla ayrı commit teklifine
  kalır (operatör onayı ile).
* Silme kanıtı: `scratch/anka_silme_adyarlari.sha256`
  (`gts.json 41add9a0…`, `dpo_all_tokenized 02004c39…`) — iki dosya silindi,
  `data/poems/v12.gts.json.tar.gz` arşivi korundu.