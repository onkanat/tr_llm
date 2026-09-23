# ANKA · ÖN-KAYITLI TEK DENEME İLANI — YETENEK EKSENİ: kapasite (T-0067)

**Damga:** 21 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 1. Soru

Unutma tarafı kapandı (r27/r28, iki bağımsız koşum). **Yetenek tarafı hiç açılmadı.**
Modül, tam ince ayarın **altıda birini** alıyor:

| adım | modül ROUGE-L | tam ince ayar ROUGE-L | modülün payı |
|---|---|---|---|
| 1000 | 0,0258 | 0,1097 | %23,5 |
| 6000 | 0,0505 | 0,3035 | %16,6 |

Ve modülün çıktısı **sabit-tahmin tabanının (0,0780) altında**; tutarsızlık **%72–100**.
Dört yetenek kapısı (ezber < %10 · tutarsızlık < %5 · ROUGE ≥ 0,35 · kesişim ≥ %80)
**hiçbir koşumda** geçilmedi.

**Soru:** yeteneğin kazanılamamasının bağlayıcı kısıtı **kapasite mi?**

## 2. Ölçülmüş gerekçe — kapasite aday

Modül **1.327.104** parametre eğitiyor = tabanın **%1,401**'i, rank **16**, **36** katman
(6 hedef × 6 blok). Tam ince ayar 93,4 M parametrenin tamamını eğitiyor.

| konfigürasyon | eğitilen | tabana oran | ROUGE-L (1000 adım) |
|---|---|---|---|
| modül r=16 (r28) | 1.327.104 | %1,401 | **0,0191** |
| *tam ince ayar (T-0096 seg_1)* | *93.424.986* | *%100* | **0,1097** |

⇒ Aradaki fark **5,7 kat**. "Kapasite yetmiyor" bu tabloda **mümkün** bir açıklamadır —
ama **ölçülmedi**. Bu ilan ölçer.

## 3. Tek değişken

**Rank (`r`), dolayısıyla eğitilen parametre sayısı.** Başka hiçbir şey değişmez.

| eksen | r28 (referans) | **r29 (bu deneme)** |
|---|---|---|
| `--r` / `--alpha` | 16 / 32 | **64 / 128** |
| eğitilen parametre | 1.327.104 (%1,401) | **5.308.416 (%5,376)** |
| veri | `scratch/t0099_wiki_mix_makale.bin` | **aynı** |
| adım · lr · batch · tohum · blok | 1000 · 2e-4 · 8 · 43 · 128 | **aynı** |
| hedef katmanlar | 6 hedef × 6 blok | **aynı** |
| dropout | 0,0 | **aynı** |

**Referans koşum yeniden KOŞULMAZ:** r28 zaten bu veriyle, bu ayarlarla, r=16'da koşuldu
(ROUGE 0,0191) ⇒ tek değişken gerçekten tek ve karşılaştırma birebir.

## 4. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

r28 ölçümü: ROUGE-L **0,0191 ± 0,0234** (n=100, SE **0,0023**) · tutarsızlık **%100**.

| sonuç | hüküm |
|---|---|
| ROUGE-L **≥ 0,060** (r28'in 3,1 katı; ≈17 SE) | **KAPASİTE BAĞLAYICI** — kısıt kapasiteydi |
| ROUGE-L **≤ 0,024** (r28 + 2 SE) | **KAPASİTE BAĞLAYICI DEĞİL** — 4× parametre hiçbir şey değiştirmedi |
| 0,024 < ROUGE-L < 0,060 | **KISMİ** — kapasite katkılı ama tek başına yeterli değil |

**İkinci, bağımsız okuma:** tutarsızlık. r28'de **%100**. Kapasite bağlayıcıysa çıktı
derlenebilir hâle gelmeli ⇒ tutarsızlık **< %60** olmalı. Hâlâ ~%100 ise modül hâlâ düzgün
çıktı üretemiyor.

**Üçüncü okuma (kaydedilir, hükme girmez):** unutma eksenleri. r=64 ile kapasite artarken
A ekseninin bozulup bozulmadığı — bu **ayrı** bir sorudur.

## 5. Bu bir "ayar kovalama" DEĞİLDİR

* Öngörü **iki yönlüdür**; **"kapasite bağlayıcı değil" dalı gerçekten mümkündür** ve
  desteklenirse tüm kapasite ekseni **elenir** (o zaman sorun mimaridedir, boyutta değil).
* Tek değişken vardır, ölçüm öncesi ilan edilmiştir, hiçbir parametre sonuç görülerek seçilmez.
* Amaç kapıyı geçirmek **değil**, kısıtı **atfetmektir**. 0,060 bile eşiğin (0,35) altındadır —
  yani bu deneme kapıyı geçmeyi **hedeflemiyor**, mekanizmayı ölçüyor.

## 6. Ön-ölçüm: bu koşumdan ÖNCE bulunan ve ilanı ETKİLEYEN olgu

**Modülün replay'i o güne kadar hiç gradyan üretmiyordu** (ölçüldü, bu oturum):

`mask_prompt_targets` (`train_step_demo.py:87-88`), `<OUTPUT>` içermeyen pencerede **tüm
hedefleri −100** yapar. Wikipedia bloklarında `<OUTPUT>` yok ⇒ replay blokları **tamamen
maskeleniyordu**:

| külliyat | `<OUTPUT>` içeren blok | korunan hedef jetonu |
|---|---|---|
| ceket | %100,00 | %43,43 |
| chat (eski replay) | %99,98 | %25,08 → **aktif** |
| **wiki (r27/r28 replay)** | **%75,15** | **%0,00 → ETKİSİZ** |

⇒ **r27/r28'de replay işlevsizdi**; o koşumlar fiilen "ceket tek başına + slotların %25'i
boşa"dır. Bu, r27/r28'in **"tek değişken = replay'in alanı"** iddiasını **çürütür** (iki şey
değişmişti) ve ayrı bir düzeltme yazısını gerektirir.

**Bu ilan için sonucu:** r28 ve r29 **aynı** karışımı kullanır ⇒ etkisizlik **iki kolda da
aynı**dır ⇒ kapasite karşılaştırması **bozulmaz**. Değişken yine tek.

## 7. Uygulama

```
train_module.py --base data/anka_a1r.pt --module modules/marangoz_r64_1000.mod.pt \
  --data scratch/t0099_wiki_mix_makale.bin \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --r 64 --alpha 128 --block-size 128 --steps 1000 --lr 2e-4 --batch-size 8 --seed 43 --device mps
→ modul_ile_olcum.py --ceket-ekseni → data/eval/anka_r29_kapasite_yetenek_2026-09-21.json
```

**DOKUNULMAZ:** `data/**` (donmuş) · `src/**` · `CLAUDE.md` · kapanmış `data/eval/anka_r17…r28*` ·
`scratch/t0096_*` · `scratch/t0097_*`. Modül `modules/` altına yazılır (donmuş değil).
`git add -A` yasak; commit yok.
