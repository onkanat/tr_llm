# T-0104 · UNK TEMİZLİK + R2 KÖK/VOCAB + YENİDEN DERLEME (SONUÇ)

**Damga:** 23 Eyl 2026 · **Görev:** T-0104 · **Yürütücü:** claude ·
İlan: `anka_t0104_unk_temizlik_ilani_2026-09-23.md` (Revizyon 3+4+5).

## 1. Artefaktlar (digest tablosu)

| artefakt | sha256 |
|---|---|
| `data/anka_a2_pretrain.bin` (100.000.000 jeton) | `da724af570636c55…` |
| `data/anka_a2_pretrain_val.bin` (2.000.000 jeton) | `96ece8010ca5f611…` |
| `data/lexicon/roots_anka_r2.tsv` (49.205 satır; +797 kök) | `c355a372ed247271…` |
| `data/rebuild/vocab_anka_r2.json` (N=33.911) | `14ce9f4e5d283c85…` |
| koşum ckpt | `data/eval/anka_t0104_build_2026-09-23.json` |
| hüküm JSON | `scratch/anka_t0104_hukum.json` (`bd897048…`) |
| ONCE çapa | `scratch/anka_t0104_once_capasi.json` |
| ilan (Revizyon 5) | sha256 koşum sırasında güncel sürüm |

Korunanlar (digest DEĞİŞMEDİ): `anka_a1r.pt` · `anka_a1r_pretrain{,_val}.bin` ·
`train_balanced_sft.bin` — koşum ckpt'inde `dokunulmadi=true` dörtlü.

## 2. Kapı hükümleri (16/16 PASS, rc=0 — `scratch/anka_t0104_hukum.py`)

| kapı | değer | bant | kaynak |
|---|---|---|---|
| K1 noktalama | max_id 33.910 · %10,75 | ≥32.145 · ≥5,0 | değişmez |
| K2 unk | **1,2246** | ≤2,0 | değişmez |
| K3 pad | 0,0 | ≤1,0 | değişmez |
| K4 ölçek | 100.000.000 | ≥90M | değişmez |
| K5 tekrarsızlık | kesişim 0 | 0 | değişmez |
| K6 izlenebilirlik | 2 kaynak sha256'lı | hepsi | değişmez |
| K7 pozitif (K7a-c+K9) | 4/4 True | — | K7c bypass: `İstanbul`→literal id; K9 sayı-apostrof unk 1→0 |
| K8 temsil | ENT=0 · CAP=0 · PN≥0,5 | — | değişmez |
| K10 unk bant | 1,2246 | 1,10-1,54 (Rev5) | ONCE-çapa tazeleme |
| G3_PN | 6,761 | ≤7,50 (Rev5) | a1r ONCE 7,1363 + %5 pay |
| G5 kapsam | train 86,68 · val 48,70 | ≥85,0/≥45,0 (Rev5) | a1r ONCE 86,37/47,57 − pay |
| yön (çapa-gölge) | unk 2,3765→1,2246 · PN 7,1363→6,761 · kapsam 86,37→86,68 | SONRA her eksen İYİ yönde | a1r ONCE |

**İlk koşum rc=2 fail-closed doğru davrandı:** üç bayat eşik (K10 1,40-1,54
kumanda-tahmini · G3 3,0 · G5 95) yakalandı; ONCE-çapa ölçümüyle (AYNI
kabin bin_olc) Revizyon 5'te beyanlı tazelendi. Kusur beyanları ilan §4
Revizyon 5'te (census-örneklem çifte-kullanım + a1r-uyumsuz bayat eşik).

## 3. Ana ölçüm: UNK oranı

**2,3765 → 1,2246 (−%48,5)** · val 3,1246 → 1,9554 (−%37,4).

Bileşenler (kumanda ölçümleri): satır-eleyici+sembol −%10,5 · sayı-apostrof
−%28,4 · kök-ekleme+bypass (797 kök, N 33.114→33.911) · Rev3 temizlik
(entity/LaTeX/dosya-parça) ek −%9,4 relatif · jeton bedel −%0,71.

## 4. Beyanlı sınırlar

* Kalan UNK %1,22: İngilizce/yabancı kelimeler (sözlüğe eklenmedil — ret
  beyanlı), ek-çözümlemeden geçen çekimli ad parçaları (`Ji-hoon'un`),
  census-frekans <5 türler.
* Sayı-ekleri (1953'te → 1953) temizlikte düşer — ek-bilgisi kaybı beyanlı;
  tokenizer-donuş kök-çözümü ayrı iş.
* PN oranı %6,761 a1r'ın %7,1363'ünün altında — bypass PN-kırılma onarımının
  gölge etkisi (istenen yön).

## 5. Faz D: eğitim (24 Eyl, TAMAMLANDI rc=0)

Komut (A1-r kanonik kaydından okunan parametreler): `--pretrain --device mps
--data data/anka_a2_pretrain.bin --vocab data/rebuild/vocab_anka_r2.json
--save-path data/anka_a2.pt --allow-frozen-write --save-every 500
--batch-size 8 --block-size 256 --steps 48828 --lr 0.001 --load-path
data/anka_a1r.pt --loss-report`. Not: ilk `!`-başlatma 80. adımda kesildi
(ckpt yazılmadan — kayıp yok; yeniden başlatma adım 1'den, aynı devam
kaynağından).

| ölçüm | değer | beklenti (ilan §4) |
|---|---|---|
| adım / süre | 48.828 · 41.344 sn (11,49 sa) | ~11,7 saat |
| sn/adım (medyan) | 0,82 | 0,8-1,0 ✓ (azami tavan 3,12 altında) |
| resize | 33.114→33.911 · 797 satır ×3 (emb+head+bias) | `[SOZLESME_UYARI]` görünür — sessiz kırpma yok |
| sınır kapısı | 33.910 < 33.911 | fail-closed devrede |
| **canlılık (devam-modu)** | başlangıç 3,9973 (a1r son-düzey 3,91±0,21 bandında) · son60 **3,3376 ± 0,2804** (n=60) · son200 3,3410 ± 0,2723 | sıfırdan-lnV bandı GEÇERSİZ ✓; gerçek düşüş |
| Bitiş Kaybı | 2,8757 (tek-adım örneklem — dağılım ile kıyasla) | bilgi |
| optimizer | AdamW sıfırdan (moment dosyası yok; ilk güncelleme ~1,73× büyük — T-0092, beyanlı) | a1r ile aynı |

**Faz D artefaktları:** `data/anka_a2.pt` sha256 `e2352cd3bbf3e88f…`
· log `scratch/anka_t0104_a2_devam.log` · ckpt her 500 adımda atomik.

**Post-doğrulama:** pytest 290/291 (bilinen gateway-sandbox istisnası) ·
`scripts/olcum_kabi.py` rc=0 (27 PASS · 0 HATA).