# ANKA · ÖN-KAYITLI TEK DENEME İLANI — YETENEK EKSENİ: gömme katmanı (T-0067)

**Damga:** 22 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 0. ÖN-ÖLÇÜM: istenen hedef ifade EDİLEMİYORDU (ve bu, ilanı değiştirdi)

`--hedefler …,embedding` **yazıldığı gibi koşulamazdı.** Ölçüldü:

* `model.embedding` = `KristalEmbedding`; içindeki `embedding.embedding` = **`nn.Embedding`**
  (33114 × 768) — `nn.Linear` **değil**.
* `modul_ekle` yalnız `nn.Linear` sarıyordu ⇒ `embedding` hedefi **sessizce düşüyordu**.
* Hata da çıkmıyordu: diğer hedefler eşleştiği için "hiçbir katman uymadı" kapısı ateşlenmiyor.

⇒ Bu hedefle koşsaydım, r30'un **kopyasını** üretir ve "gömme hiçbir şey değiştirmedi" diye
**sahte bir null** yazardım. Mekanizma **kuruldu** (bu ilandan önce, ölçülerek):

| | |
|---|---|
| `LoRAEmbedding(nn.Embedding)` | `E = W + olcek·(B@A)`; `B` sıfır başlar ⇒ delta **tam 0** |
| takma anı bit-özdeşlik | **logit farkı 0,000e+00** (ölçüldü — sözleşme maddesi 2) |
| **kristal yapı korunur** | `KristalEmbedding`'in KENDİSİ **sarılmaz**; hedef içteki `embedding.embedding`'dir ⇒ imza maskesi delta'nın **ÜSTÜNDE** kalır (ölçüldü: `çıktı == mask·(W[x]+delta[x])`), baypas edilmez |
| **kısmi-uyum kapısı** | Her hedef ≥1 katmana uymalı, yoksa **DURUR** — sessiz düşme bir daha olamaz |
| testler | `tests/test_modul_embedding_kapisi.py` · 6 iki-dallı test · **mutasyon 3/3 yakalandı** |
| tam takım | **1 failed, 273 passed** (düşen tek test bilinen sandbox soket yasağı) |

## 1. Soru

**Gömme (giriş tarafı) içeriği açıyor mu?** r30 `lm_head` ile **biçimi** açtı (ROUGE 0,0191 →
0,0696) ama **içerik gelmedi**: çıktı `'teknik çözüm:'` + durma/tekrar, `kesişim %1` (eşik %80).

Gömme donuk kaldığı sürece model **soruyu temsil edemiyor** olabilir — çıkış tarafı açıldı,
giriş tarafı kapalı.

## 2. Tek değişken

**r30'un hedef kümesine `embedding` eklenir.** Başka hiçbir şey değişmez.

| eksen | r30 (referans) | **r31 (bu deneme)** |
|---|---|---|
| hedefler | varsayılan + `lm_head` | **varsayılan + `lm_head` + `embedding`** |
| katman · eğitilen | 37 · 1.869.216 (%1,9615) | **38 · 2.411.328 (%2,5161)** |
| `r` · `alpha` · dropout | 16 · 32 · 0,0 | **aynı** |
| veri · adım · lr · batch · tohum · blok | r30 mix · 1000 · 2e-4 · 8 · 43 · 128 | **aynı** |

**Referans (r30) yeniden KOŞULMAZ** — aynı ayarlarla ölçüldü ⇒ tek değişken gerçekten tek.
`lm_head`+`embedding`'i birlikte eklemek, ölçülmüş bir kazanımın **üstüne** inşa eder; "varsayılan
+ yalnız gömme" kolu **koşulmuyor** çünkü r30 gösterdi ki `lm_head` olmadan biçim bile kazanılmıyor
(o kolda içerik etkisi ölçülemezdi). Etkisiz replay **iki kolda da aynı** ⇒ karıştırmaz.

## 3. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

Ölçülmüş referans, r30: **kesişim %1,00** (Wilson **[0,18 ; 5,45]**) · ROUGE 0,0696 ± 0,0934
(SE 0,0093) · tutarsızlık %98.

| # | okuma | açıyor ise | açmıyor ise |
|---|---|---|---|
| **1 (birincil)** | **kesişim** | **≥ %10** (Wilson [5,5 ; 17,4] ⇒ r30'un üst sınırıyla **AYRIK**) | **≤ %5** (ayırt edilemez) |
| 2 | tutarsızlık | **< %60** | ~%98 |
| 3 | ROUGE-L | ≥ **0,096** (r30 + 2 SE) | ≤ 0,070 |

**Neden birincil okuma `kesişim`:** r30'un açtığı şey biçimdi; `kesişim`, **çıktının sorunun
içeriğine değip değmediğini** ölçen tek metrik ve r30'da **kımıldamadı** (%0 → %1). Zarfı
paylaşmayan metrik budur.

## 4. Kaydedilir, hükme GİRMEZ

* **A/B unutma eksenleri.** r30 kapıyı **0,0179 puanla** (yarı-genişliğin 0,147 katı) geçmişti;
  gömme eklemek A'yı nereye taşır **ölçülür ve yazılır** — bu ayrı bir sorudur.
* **ezber.** r30'da %28'di ve **zarfın** 4-gramlarından geliyordu; yine okunur, yorum bu ilanın
  hükmüne girmez (bkz. toplu-metrik-ateşledi-mekanizma-ateşlemedi dersi).

## 5. Bu bir "ayar kovalama" DEĞİLDİR

Öngörü **iki yönlü**; "gömme içeriği açmıyor" dalı gerçekten mümkündür ve desteklenirse **giriş
tarafı ekseni kapanır**. Hiçbir eşik sonuç görülerek seçilmedi: birincil eşik, r30'un **ölçülmüş
Wilson aralığından** türetildi (ayrıklık ölçütü). Amaç kapıyı (0,35) geçirmek değil, kısıtı atfetmek.

## 6. Uygulama

```
train_module.py --base data/anka_a1r.pt --module modules/marangoz_lmhead_gomme_1000.mod.pt \
  --data scratch/t0099_wiki_mix_makale.bin --vocab data/rebuild/vocab_anka_r1_33114.json \
  --hedefler attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2,lm_head,embedding \
  --r 16 --alpha 32 --block-size 128 --steps 1000 --lr 2e-4 --batch-size 8 --seed 43 --device mps
→ modul_ile_olcum.py --ceket-ekseni → data/eval/anka_r31_gomme_yetenek_2026-09-22.json
```

**DOKUNULMAZ:** `data/**` (donmuş) · `CLAUDE.md` · kapanmış `data/eval/anka_r17…r30*` ·
`scratch/t0096_*` · `scratch/t0097_*`. **Bu turda `src/llm/modules.py` DEĞİŞTİ** (yeni sınıf +
kısmi-uyum kapısı) — bu, T-0086/T-0087'nin "KOD DEĞİŞMEDİ" beyanını **tersine çevirir** ve
raporda açıkça yazılır. `git add -A` yasak; commit yok.
