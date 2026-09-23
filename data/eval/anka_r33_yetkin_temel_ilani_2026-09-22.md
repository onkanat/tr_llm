# ANKA · ÖN-KAYITLI TEK DENEME İLANI — YETKİN TEMEL üzerine modül (T-0067)

**Damga:** 22 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 1. Nerede kaldık — ve neden bu deneme

Mimari ekseninin üç kolu elendi (r29 kapasite · r31 giriş tarafı · **r32 bütçe**). r32'nin
hükmü: rank-sınırlı delta **donuk** bir tabana **biçim** enjekte edebiliyor, **bilgi** edemiyor.

Ama bu hükmün **ayırt edilmemiş bir alternatifi** var: modül bugüne kadar **iki işi birden**
yapmaya çalıştı. Ölçüldü — taban `anka_a1r.pt` talimat izleyemiyor (ROUGE **0,0000**,
tutarsızlık **%100**, çıktı `, [?], [?], …`). Yani modül hem **talimat izlemeyi** hem **bilgiyi**
öğretmek zorundaydı.

**İki hipotez:**
* **H-A (temel yetkinliği):** modül başarısız çünkü taban talimat izleyemiyor.
* **H-B (delta sınırı):** rank-sınırlı delta bilgi enjekte **edemez** — taban ne olursa olsun.

**Bu deneme ikisini ayırır:** modül, talimat izleyebilen bir temel üzerine kurulur.

## 2. Temel: `scratch/t0096_kos/seg_6.pt` — T-0096'nın 6. segmenti

| | değer (kanonik kap, T-0096) |
|---|---|
| ROUGE-L | **0,3035 ± 0,1541** (SE 0,0154) · medyan 0,3254 |
| tutarsızlık | **%5** (eşik <%5 sınırında) |
| ezber | %0 |
| **kesişim** | **%10** (Wilson [5,52 ; 17,44]) |
| A CE | 6,2086 (taban 3,5352 ⇒ **+%75,62**) |
| sha256 | `136dda76da419e281cf8b800a0e562381c5e9c7db8ed85136ac3ee4f19578b8b` |

**Yükleme ön-ölçüldü:** `KristalLM`'e **0 eksik / 0 fazla** anahtarla giriyor, `lm_head` şekli
sözlükle uyuyor, forward çalışıyor.

## 3. ⚠ ÖLÇÜMÜN SINIRI — ilan edilir, sonucu yorumlarken bağlayıcıdır

**seg_6 yeteneğe ZATEN SAHİP** (kesişim %10, ROUGE 0,3035). Modül aynı yetenek verisiyle
eğitilecek ⇒ bu deneme **"delta, yetkin bir modeli iyileştirebilir mi"**yi ölçer;
**"yetkin bir tabana sıfırdan bilgi enjekte edebilir mi"**yi **değil**.

Sonuç ne olursa olsun bu sınır yazılır:
* **Olumlu sonuç** (içerik eklenirse): H-A **desteklenir**, modül mimarisi bilgi için **açık kalır**.
* **Olumsuz sonuç:** H-B **güçlenir** ama **H-A tamamen elenmez** — çünkü test edilen şey
  "yetkin modele ekleme"dir, "yetkin tabana enjeksiyon" değil. H-A'yı tam elemek için
  **yetkin ama yeteneksiz** bir temel gerekir (aday: `seg_1` — tutarsızlık %24, kesişim %0).
  **O kol bu ilanda KOŞULMAZ**; sonuç olumsuzsa sıradaki ilanın konusudur.

## 4. Tek değişken

**Temel checkpoint'i.** Modül tarifi r30 ile **birebir aynı**.

| eksen | r30 (referans) | **r33 (bu deneme)** |
|---|---|---|
| `--base` | `data/anka_a1r.pt` | **`scratch/t0096_kos/seg_6.pt`** |
| hedefler | varsayılan + `lm_head` (37 katman) | **aynı** |
| adım · lr · batch · tohum · blok · `r`/`alpha` | 1000 · 2e-4 · 8 · 43 · 128 · 16/32 | **aynı** |
| veri | `scratch/t0099_wiki_mix_makale.bin` | **aynı** |

**Referans (seg_6 tek başına) yeniden KOŞULMAZ** — T-0096'da **aynı kanonik kap, aynı tutulan
küme, aynı seed** ile ölçüldü.

## 5. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

| # | okuma | EKLENDİ ise | EKLENMEDİ ise |
|---|---|---|---|
| **1 (birincil)** | **kesişim** | **≥ %18** (seg_6 %10 + 2 SE, iki örneklem SE 0,0424) | **≤ %10** |
| 2 | ROUGE-L | ≥ **0,348** (seg_6 + 2 SE) | ≤ 0,304 |
| 3 | tutarsızlık | **< %10** (yetkin model BOZULMADI) | ≥ %10 (bozuldu) |

**Birincil okuma hükmü verir.** 10 < kesişim < 18 ⇒ **KISMİ**.
Okuma 3 ayrıca **koruma** ölçer: seg_6'nın %5 tutarsızlığı korunuyor mu.

## 6. Kaydedilir, hükme girmez

* **A/B unutma eksenleri** — bu kez `--baseline` **seg_6'nın kendisi** olur ⇒ ölçülen şey
  modülün **seg_6'ya eklediği** unutmadır (tabana göre değil).
* **ezber.** seg_6'da %0. Aynı veriyle daha çok eğitim ezberi **artırabilir**; artış tek başına
  başarı sayılmaz ([[toplu-metrik-atesledi-mekanizma-ateslemedi]]).

## 7. Bu bir "ayar kovalama" DEĞİLDİR

Öngörü **iki yönlü**; "eklenmedi" dalı gerçekten mümkündür ve desteklenirse H-B güçlenir.
Değişen tek şey temeldir; tarif, veri ve eşikler önceki ilanlardan **aynen** devralındı.
Eşikler seg_6'nın **ölçülmüş** istatistiklerinden türetildi, sonuç görülerek seçilmedi.

## 8. Uygulama

```
train_module.py --base scratch/t0096_kos/seg_6.pt --module modules/seg6_marangoz_1000.mod.pt \
  --data scratch/t0099_wiki_mix_makale.bin --vocab data/rebuild/vocab_anka_r1_33114.json \
  --hedefler attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2,lm_head \
  --r 16 --alpha 32 --block-size 128 --steps 1000 --lr 2e-4 --batch-size 8 --seed 43 --device mps
→ modul_ile_olcum.py --model scratch/t0096_kos/seg_6.pt --baseline scratch/t0096_kos/seg_6.pt \
    --ceket-ekseni → data/eval/anka_r33_yetkin_temel_yetenek_2026-09-22.json
```

**DOKUNULMAZ:** `data/**` (donmuş) · `CLAUDE.md` · kapanmış `data/eval/anka_r17…r32*` ·
**`scratch/t0096_*` (seg_6 SALT OKUNUR — üzerine YAZILMAZ)**. `src/**` değişmez.
`git add -A` yasak; commit yok.
