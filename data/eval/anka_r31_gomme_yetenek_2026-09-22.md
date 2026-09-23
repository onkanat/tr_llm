# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — YETENEK EKSENİ: gömme katmanı

**İlan:** `data/eval/anka_r31_gomme_ilani_2026-09-22.md` (**ölçümden önce** yazıldı)
**Koşum:** eğitim `scratch/t0099f_zincir.log` (1000 adım, 309 sn) · ölçüm `scratch/t0099f2_olcum.log` · ikisi de `rc=0`

---

## 1. Hüküm: üç okumanın ÜÇÜ de "açmıyor" ⇒ giriş tarafı ekseni KAPANIR

İlan §3'ün ön-kayıtlı sonucu: *"üç okuma birden 'açmıyor' derse **giriş tarafı ekseni kapanır**."*

| # | okuma | ilan: açar ise | ölçülen (r30 → r31) | sonuç |
|---|---|---|---|---|
| **1 (birincil)** | **kesişim** | ≥ %10 | **%1,00 → %0,00** | **AÇMIYOR** |
| 2 | tutarsızlık | < %60 | **%98,00 → %98,00** | **AÇMIYOR** |
| 3 | ROUGE-L | ≥ 0,096 | **0,0696 → 0,0595** (fark −0,0101 · 0,78 SE) | **AÇMIYOR** |

**Gömme katmanı içeriği açmıyor.** Çıktı hâlâ zarf: ham üretimler **r30'dan daha dejenere** —

```
idx 1  'teknik çözüm: teknik çözüm:'
idx 3  'teknik çözüm: teknik çözüm: teknik çözüm: … (8+ kez)'
```

⇒ Model çıkış tarafı açıldıktan sonra (r30) **zarfı tekrarlamayı öğrenmiş**; giriş tarafını
açmak (r31) bunu **değiştirmiyor**. İçerik kısıtı **giriş temsilinde değil**.

## 2. Tablo

| koşum | katman | ROUGE-L | tutarsızlık | **kesişim** | ezber | A artış | B düşüş |
|---|---|---|---|---|---|---|---|
| r28 · varsayılan | 36 | 0,0191 | %100 | %0 | %0 | +%6,08 ✓ | +0,28 ✓ |
| r29 · `r`=64 (kapasite) | 36 | 0,0248 | %100 | %0 | %0 | +%6,19 ✓ | −1,78 ✓ |
| r30 · +`lm_head` | 37 | **0,0696** | %98 | **%1** | %28 | +%9,98 ✓ | +1,51 ✓ |
| **r31 · +`lm_head`+`embedding`** | **38** | 0,0595 | %98 | **%0** | %20 | **+%9,45 ✓** | +1,59 ✓ |
| *tam ince ayar seg_1 (1000 adım)* | *tümü* | *0,1097* | *%24* | *%0* | *%20* | *+%23,31 ✗* | *−0,08 ✓* |

Çıpalar: sabit-tahmin 0,0780 · distraktör 0,0848 · eşik 0,35. r31 (0,0595) **sabit-tahmin
tabanının altında**.

## 3. Yan bulgu: r30'un unutma "geçişi" kıl payıydı, r31'inki DEĞİL

| | A artışı | pay | pay / yarı-genişlik |
|---|---|---|---|
| r30 | +%9,9821 | 0,0179 puan | **0,147 katı** (gürültünün içinde) |
| **r31** | +%9,4512 | **0,5488 puan** | **4,55 katı** (gerçek geçiş) |

Gömme eklemek unutmayı **artırmadı**; r31'in A geçişi r30'unkinden **sağlam**. Yani
"biçim öğrenen modül unutmayı da artırıyor" işareti r30'un kıl payı geçişinin artefaktı
olabilirdi — r31 bunu **desteklemiyor**.

## 4. Bu koşumdan ÖNCE bulunan iki kusur (ikisi de kapıya çevrildi)

**(a) İstenen hedef İFADE EDİLEMİYORDU — sessiz düşme.** `embedding` = `KristalEmbedding`;
içindeki `embedding.embedding` = `nn.Embedding`, `nn.Linear` **değil**. `modul_ekle` yalnız
`nn.Linear` sarıyordu ⇒ hedef **sessizce düşüyordu**, hata çıkmıyordu (diğer hedefler
eşleştiği için "hiçbir katman uymadı" kapısı ateşlenmiyordu). Bu hedefle koşsaydım r30'un
kopyasını üretir ve **sahte bir null** yazardım.

**Kapatıldı:** `LoRAEmbedding(nn.Embedding)` yazıldı **ve** `modul_ekle`'ye **kısmi-uyum
kapısı** eklendi — her hedef ≥1 katmana uymalı, yoksa DURUR.

**(b) Sarma betiğinin katman sayacı koptu.** `scripts/modul_ile_olcum.py:114` katmanları
**sınıf adı dizesiyle** sayıyordu (`"LoRAKatmani"`) ⇒ `LoRAEmbedding` eklenince 38 → 37.
**Fail-closed kapı ateşledi ve sayı ÜRETMEDİ** (`rc=2`), yani sessizce yanlış bir "37 katman"
yazılmadı. Kök neden modülde değil **yüklemin kopyasında**: kanonik `modul_katmanlari` varken
sınıf adı dizeyle yeniden yazılmıştı. Düzeltildi; depoda başka örneği **yok** (tarandı).

## 5. Mekanik doğrulama (bu turda `src/llm/modules.py` DEĞİŞTİ)

| iddia | ölçüm |
|---|---|
| takma anında bit-özdeş (sözleşme 2) | logit farkı **0,000e+00** |
| parametre formülü iki sınıfta aynı | 2.411.328 = formül ✓ (%2,5161) |
| **kristal yapı korunur** | `çıktı == maske·(W[x]+delta[x])` ⇒ imza maskesi delta'nın **ÜSTÜNDE**, baypas edilmiyor |
| kısmi-uyum kapısı | her hedef uymalı; ölçüldü, ateşliyor |
| yeni testler | `tests/test_modul_embedding_kapisi.py` · 6 iki-dallı · **mutasyon 3/3 yakalandı** |
| tam takım | **1 failed, 273 passed** (düşen tek test bilinen sandbox soket yasağı) |

> **T-0086/T-0087'nin "KOD DEĞİŞMEDİ: `src/**`, `scripts/**`, `tests/**`" beyanı bu turda
> TERSİNE DÖNMÜŞTÜR:** `src/llm/modules.py`, `scripts/modul_ile_olcum.py` değişti ve
> `tests/test_modul_embedding_kapisi.py` eklendi. Donmuş kalıplara **dokunulmadı**.

## 6. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `modules/marangoz_lmhead_gomme_1000.mod.pt` | `7c1b789d835d5af279160e2c23f8bd9e0196cfb29ea3d627e18db874364ade79` |
| `data/eval/anka_r31_gomme_yetenek_2026-09-22.json` | `7ce5296c262522f3be86139fdac54507efe1c0eee374ae5aa5d0d3e102b5a2e8` |
| `data/eval/anka_r31_gomme_yetenek_2026-09-22.modul.json` | `ca2cdb10d852250522c15c519535eea8916a146d13867cccb9ae5b21fedc1804` |
| `src/llm/modules.py` | `8bea9058047f5da016cd3b4a372b87653c2e54f7eb2f5a0c03da3ce83e1c5b4d` |
| `scripts/modul_ile_olcum.py` | `4e35f2b77511c536a898f5c74ba6a40f0c05a1748b296a3a3731bed7a0b97bc0` |
| `data/anka_a1r.pt` (taban, değişmedi) | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |

Modül meta: `38 katman · 2.411.328 parametre · hedefler …,lm_head,embedding · kayıp ilk 6,6248
→ son60 3,8083 ± 0,2205 · 9,7 MB`. Vakum **14,944** ≠ 0 · taban tensörü değişen **0**.

## 7. Elenenler ve kalan

**ELENEN (kalıcı):** kapasite (r29, `r` ×4) · **giriş tarafı / gömme (r31, üç okuma birden)**.
**AÇILAN:** çıkış kafası **biçim** için bağlayıcı (r30).

**KALAN — ve artık mimari DIŞINDA:** içerik hâlâ yok. Ölçülmüş iki aday:

1. **Adım sayısı.** 1000 adım ≈ **1 epoch**; ceket meta'sının kendi formülü
   (`steps_formulu`) **2301** adım diyor. Tam ince ayar 1000 → 6000 adımda ROUGE
   0,1097 → 0,3035 yaptı ⇒ **adım ekseni bu külliyatta gerçek bir kaldıraç** (ölçülmüş).
2. **Veri.** 11.708 kayıt; içerik üretmeyi öğretmek için yeterli olup olmadığı ölçülmedi.

**Commit yok · `git add -A` kullanılmadı · donmuş kalıplara yazılmadı.**
