# ANKA · ÖN-KAYITLI TEK DENEME İLANI — YETENEK EKSENİ: adım bütçesi (T-0067)

**Damga:** 22 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 1. Nerede kaldık

Unutma tarafı kapandı. Yetenek tarafında **mimari eksenlerin hepsi elendi**:

| # | deneme | hüküm |
|---|---|---|
| r28 | referans (varsayılan 36 katman) | ROUGE 0,0191 · kesişim %0 |
| r29 | **kapasite** (`r` ×4) | **ELENDİ** — 1,70 SE |
| r30 | **çıkış kafası** (`+lm_head`) | **BİÇİM açıldı** — ROUGE 0,0696 · ezber %28 · **kesişim %1** |
| r31 | **giriş tarafı** (`+embedding`) | **ELENDİ** — üç okuma da "açmıyor" |

**Açık kalan tek soru:** içerik. Ve kalan iki aday **mimari dışında**: adım ve veri.

## 2. Ölçülmüş gerekçe — adım GERÇEK bir kaldıraç

| | 1000 adım | 6000 adım | oran |
|---|---|---|---|
| tam ince ayar (T-0096, aynı külliyat) | ROUGE 0,1097 | **0,3035** | **2,8×** |
| modül (r30) | 0,0696 | — | ? |

Tam ince ayar **aynı külliyatta** adımla 2,8 kat kazanıyor ⇒ adım bu veride **ölçülmüş** bir
kaldıraçtır. Buna karşılık **bütün modül koşumlarım 1000 adım = ~1 epoch**:

* ceket meta'sının **kendi** formülü: `steps_formulu = ceil(784797/(128·8))·3 = 2301` (3 epoch).
* Karışımda slotların %25'i etkisiz (ölçülmüş) ⇒ adım başına etkili ceket jetonu **768**
  ⇒ 3 epoch için **≈ 3065 adım**.

⇒ Modül bugüne kadar külliyatın **üçte biri** kadar eğitildi.

## 3. Tek değişken

**`--steps`: 1000 → 3000.** Başka hiçbir şey değişmez.

| eksen | r30 (referans) | **r32 (bu deneme)** |
|---|---|---|
| `--steps` | 1000 (≈1 epoch) | **3000 (≈3 epoch)** |
| hedefler | varsayılan + `lm_head` (37 katman) | **aynı** |
| `r`/`alpha`/dropout | 16 / 32 / 0,0 | **aynı** |
| veri · lr · batch · tohum · blok | r30 mix · 2e-4 · 8 · 43 · 128 | **aynı** |

**Referans (r30) yeniden KOŞULMAZ.** Etkisiz replay **iki kolda da aynı** (aynı karışım) ⇒
kontrollü; r31'in `embedding` kolu **kapandığı için** kümeye girmez.

## 4. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

Ölçülmüş referans, r30: **kesişim %1,00** (Wilson **[0,18 ; 5,45]**) · ROUGE 0,0696 ± 0,0934
(SE 0,0093) · tutarsızlık %98 · ezber %28. Eşikler r31 ile **aynı** (bilerek değiştirilmedi).

| # | okuma | içerik geldi ise | gelmedi ise |
|---|---|---|---|
| **1 (birincil)** | **kesişim** | **≥ %10** (Wilson [5,5 ; 17,4] ⇒ r30'la **AYRIK**) | **≤ %5** |
| 2 | tutarsızlık | **< %60** | ~%98 |
| 3 | ROUGE-L | ≥ **0,096** (r30 + 2 SE) | ≤ 0,070 |

**Ayırt edici ek okuma (kaydedilir, hükme girmez):** `ezber`. r30'da **%28** ve **zarftan**
geliyordu. Adım artışı **yalnız biçimi pekiştirirse**, ezber yükselirken `kesişim` sabit kalır;
**içerik gelirse** ezberle birlikte `kesişim` de yükselir. İki metriğin **birlikte** hareketi
bu denemenin ayırt edici imzasıdır.

## 5. İki dalın anlamı — ÖNCEDEN yazılır

* **İçerik gelirse:** bugüne kadarki tavan bir **bütçe** artefaktıydı; modül mimarisi
  (donuk taban + rank-16 delta) **yetenek için de** yeterli. Sonraki iş: bütçeyi/adımı ölçekle.
* **İçerik gelmezse:** kısıt **bütçe değil, mekanizmadır** ⇒ "rank-sınırlı delta donuk tabana
  **biçim** enjekte edebilir, **bilgi** edemez" hipotezi güçlenir ve **mimari ekseni kapanır**:
  o zaman bilgi için tam ince ayar (ya da farklı bir modül tasarımı) gerekir. Bu, olumsuz sonuç
  olsa da **yol göstericidir** ve "çalışan bir temel+modül" arayışında **elenmesi gereken** dalı eler.

## 6. Ölçümün sınırı (ilan edilir)

* Slotların %25'i etkisiz kalır ⇒ 3000 adım ≈ **2250 etkili** adım. Değişken yine tektir (r30'da
  da aynı) ama bütçe 3× değil, **2,25×**'tir.
* **Veri ekseni ölçülmüyor** — bu koşum veriyi sabit tutar.
* A/B unutma eksenleri kaydedilir; **hükme girmez**.

## 7. Uygulama

```
train_module.py --base data/anka_a1r.pt --module modules/marangoz_lmhead_3000.mod.pt \
  --data scratch/t0099_wiki_mix_makale.bin --vocab data/rebuild/vocab_anka_r1_33114.json \
  --hedefler attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2,lm_head \
  --r 16 --alpha 32 --block-size 128 --steps 3000 --lr 2e-4 --batch-size 8 --seed 43 --device mps
→ modul_ile_olcum.py --ceket-ekseni → data/eval/anka_r32_adim_yetenek_2026-09-22.json
```

**DOKUNULMAZ:** `data/**` (donmuş) · `CLAUDE.md` · kapanmış `data/eval/anka_r17…r31*` ·
`scratch/t0096_*` · `scratch/t0097_*`. `src/**` bu koşumda **değişmez** (mekanizma r31'de kuruldu).
`git add -A` yasak; commit yok.
