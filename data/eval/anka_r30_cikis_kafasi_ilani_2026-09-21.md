# ANKA · ÖN-KAYITLI TEK DENEME İLANI — YETENEK EKSENİ: çıkış kafası (T-0067)

**Damga:** 21 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 1. Soru ve zincirdeki yeri

Yetenek ekseninde **ikinci** ve son ilan edilmiş adım. Zincir:

| # | deneme | hüküm |
|---|---|---|
| r28 | referans: r=16, varsayılan hedefler | ROUGE **0,0191** · tutarsızlık %100 · ezber %0 |
| r29 | **kapasite** (`r` 16→64, ×4 parametre) | **ELENDİ** — ROUGE 0,0248 (1,70 SE, ayırt edilemez) · tutarsızlık %100→%100 |
| **r30** | **çıkış kafası** (`lm_head` hedefe eklenir) | bu ilan |

**Soru:** yeteneğin kazanılamamasının kısıtı **çıkış projeksiyonu mu?**

## 2. Ölçülmüş gerekçe

Modülün hedef kümesi `attn.{q,k,v,out}_proj`, `mlp.0`, `mlp.2` — **36 katman**. Modelde **37**
`nn.Linear` var: kalan tek katman **`lm_head`** (`33114 × 768`), yani **hangi jetonun olası
olduğunu belirleyen son projeksiyon**. Ölçüldü (sarak, elle hesaplamadan):

| | katman | eğitilen | tabanın |
|---|---|---|---|
| r28 (varsayılan) | 36 | 1.327.104 | %1,4006 |
| **r30 (+`lm_head`)** | **37** | **1.869.216** | **%1,9615** |
| `lm_head` deltası | — | **542.112** | — |

`modul_ekle` bu hedefi yakalıyor (`_hedef_mi`: `ad == "lm_head"`), formül kapısı geçiyor.

**Neden bu aday:** tam ince ayar `lm_head`'i **değiştirir** ve 1000 adımda ROUGE 0,1097'ye
çıkarırken eğitim satırlarının **%20'sini ezberler**; modül ezberi **%0**'da tutar. İkisi
**nitelik olarak** farklı davranıyor. Bir yeteneği kazanmak, hangi jetonların olası olduğunun
**yeniden eşlenmesini** gerektiriyorsa, donuk bir çıkış projeksiyonuyla bu yapılamaz.

⇒ **r29 kapasiteyi eledi; kalan yapısal fark budur.**

## 3. Tek değişken

**`--hedefler` listesine `lm_head` eklenir.** Başka hiçbir şey değişmez.

| eksen | r28 (referans) | **r30 (bu deneme)** |
|---|---|---|
| `--hedefler` | `attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2` | **…,lm_head** |
| katman / eğitilen | 36 / 1.327.104 | **37 / 1.869.216** |
| `--r` / `--alpha` | 16 / 32 | **aynı** |
| veri · adım · lr · batch · tohum · blok | r28 mix · 1000 · 2e-4 · 8 · 43 · 128 | **aynı** |
| dropout | 0,0 | **aynı** |

**Referans yeniden KOŞULMAZ:** r28 tam bu ayarlarla ölçüldü (ROUGE 0,0191) ⇒ karşılaştırma birebir.
**Etkisiz replay iki kolda da aynı** (aynı karışım) ⇒ karşılaştırmayı bozmaz.

## 4. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

r28 ölçümü: ROUGE-L **0,0191 ± 0,0216** (n=100, SE **0,0022**) · tutarsızlık **%100** · ezber **%0**.
Eşikler **r29 ile aynı** tutuldu (kıyaslanabilirlik için bilerek değiştirilmedi).

| # | okuma | bağlayıcı ise | bağlayıcı DEĞİL ise |
|---|---|---|---|
| **1 (birincil)** | ROUGE-L | **≥ 0,060** | **≤ 0,024** (r28 + 2 SE) |
| 2 | tutarsızlık | **< %60** | ~%100 |
| 3 | ezber | **≥ %5** (görev öğreniliyor) | %0 |

**Birincil okuma hükmü verir.** 0,024 < ROUGE < 0,060 ⇒ **KISMİ**.
**Üç okuma birden** "bağlayıcı değil" derse kısıt **başka yerdedir** ve o eksen bu ilanla
**kapanır** — bir daha `r`/`lm_head` denenmez.

**Hüküm A/B eksenlerine girmez** (kaydedilir; unutma ayrı sorudur).

## 5. Bu bir "ayar kovalama" DEĞİLDİR

* Öngörü **iki yönlü**; "bağlayıcı değil" dalı **gerçekten mümkündür** ve desteklenirse tüm
  çıkış-kafası ekseni **kapanır**.
* Aday, ölçülmüş bir **nitelik farkından** türetildi (tam ince ayar %20 ezberler / modül %0),
  kapıyı geçirmekten değil. 0,060 bile eşiğin (0,35) **altındadır** — bu deneme kapıyı
  **hedeflemiyor**, kısıtı **atfediyor**.
* Hiçbir parametre sonuç görülerek seçilmez; eşikler önceki ilandan **aynen** devralındı.

## 6. Uygulama

```
train_module.py --base data/anka_a1r.pt --module modules/marangoz_lmhead_1000.mod.pt \
  --data scratch/t0099_wiki_mix_makale.bin \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --hedefler attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2,lm_head \
  --r 16 --alpha 32 --block-size 128 --steps 1000 --lr 2e-4 --batch-size 8 --seed 43 --device mps
→ modul_ile_olcum.py --ceket-ekseni → data/eval/anka_r30_cikis_kafasi_yetenek_2026-09-21.json
```

**DOKUNULMAZ:** `data/**` (donmuş) · `src/**` · `CLAUDE.md` · kapanmış `data/eval/anka_r17…r29*` ·
`scratch/t0096_*` · `scratch/t0097_*`. Modül `modules/` altına yazılır.
`git add -A` yasak; commit yok.
