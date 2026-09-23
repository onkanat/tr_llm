# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — YETENEK EKSENİ: kapasite

**İlan:** `data/eval/anka_r29_kapasite_ilani_2026-09-21.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099d_zincir.log` · 1000 adım · `rc=0` · 19:07:04Z→19:17:40Z

---

## 1. Hüküm: KISMİ (ilan edilen bant) — ama iki bağımsız okuma "bağlayıcı DEĞİL" diyor

İlan edilen bantlar: **≥ 0,060** kapasite bağlayıcı · **≤ 0,024** bağlayıcı değil · arası **KISMİ**.
Ölçülen **0,0248** ⇒ **ilan edilen kurala göre KISMİ**. Kural, sonuç görülerek kaydırılmaz.

**Ama "kısmi" bandının nasıl okunacağını iki bağımsız okuma belirliyor:**

| okuma | ölçülen | ilan edilen işaret | sonuç |
|---|---|---|---|
| ROUGE-L farkı | +0,0057 · iki örneklem SE 0,0034 · **\|fark\|/SE = 1,70** | — | **AYIRT EDİLEMEDİ** (2 SE'nin altı) |
| tutarsızlık | **%100 → %100** | *"kapasite bağlayıcıysa < %60 olmalı"* | **ateşlenmedi** |

⇒ Kapasite, 5,7 katlık açığı **açıklamıyor**. En fazla küçük bir katkı; ve o katkı bile
gürültüden **ayırt edilemiyor**.

## 2. Asıl bulgu: 4× parametre UYUMU artırdı, YETENEĞİ artırmadı

| | r=16 (r28) | **r=64 (r29)** | değişim |
|---|---|---|---|
| eğitilen parametre | 1.327.104 (%1,401) | **5.308.416 (%5,376)** | **×4,0** |
| son60 eğitim kaybı | 4,6999 ± 0,2249 | **4,3971 ± 0,2168** | **−%6,4 (ayırt edilebilir)** |
| **ROUGE-L** | 0,0191 | **0,0248** | +0,0057 · **1,70 SE ⇒ ayırt edilemez** |
| **tutarsızlık** | %100,00 | **%100,00** | **değişmedi** |
| ezber | %0,00 | %0,00 | — |
| kesişim | %0,00 | %0,00 | — |
| A artışı | +%6,08 ✓ | **+%6,19 ✓** | unutma **sabit** |
| B düşüş | +0,28 ✓ | −1,78 ✓ | — |

**Ekstra kapasitenin tamamı uyuma gitti, yeteneğe gitmedi.** Eğitim kaybı ölçülebilir biçimde
düştü (−%6,4) ama çıktı hâlâ **derlenemiyor** (tutarsızlık %100) ve örtüşme kımıldamadı.
Bu, "kapasite yetmiyor" açıklamasını **zayıflatır**: modül veriyi daha iyi sığdırabiliyor,
sığdırdığı şeyi yeteneğe çeviremiyor.

## 3. Elenen ve kalan

**ELENEN:** kapasite ekseni — `r` 16 → 64 (4×) yeteneği getirmiyor.
Unutma ekseni de sağlam: A +%6,19 ile kapıyı **üçüncü kez** geçti, B −1,78 ile geçti.

**KALAN (ilan edilmemiş, ölçülmemiş — hipotez):** modül **gömme katmanına ve çıkış kafasına
dokunmuyor.** Hedef kümesi `attn.{q,k,v,out}_proj, mlp.0, mlp.2` — 36 katman; `embedding` ve
`lm_head` **hiç değişmiyor**. Bir yeteneği kazanmak, hangi jetonların olası olduğunu
**yeniden eşlemeyi** gerektiriyorsa, çıkış projeksiyonu donuk kalırken bu yapılamaz.
Tam ince ayar bu katmanları **değiştirir** — ve 1000 adımda ROUGE 0,1097'ye ulaşırken
**ezber %20** yapar; modül ise ezberi **%0**'da tutar. İkisi nitelik olarak farklı davranıyor.

**Bu bir iddia değil, sıradaki ilanın adayıdır** — ölçülmedi.

## 4. Kapılar

| kapı | sonuç |
|---|---|
| VAKUM | logit farkı **7,962** ≠ 0 |
| TABAN DOKUNULMAZLIĞI | `b93cc1cd…` başta=sonda · değişen taban tensörü **0** |
| PARAMETRE FORMÜLÜ | eğitilen **5.308.416** = formül; tabanın %5,376'si |
| KARARLILIK | r28 ve r29 **aynı tohumda ilk kayıp 6,1510 / 6,2101** (aynı veri, farklı r) |

## 5. Bu koşumun İLANA yazılmış sınırı

r27/r28'de replay blokları **etkisizdi** (`<OUTPUT>` yok ⇒ tüm hedefler −100; bkz. ilan §6).
r29 **aynı karışımı** kullandığı için etkisizlik **iki kolda da aynı** ⇒ kapasite karşılaştırması
bozulmadı. Ama bu, ölçümün genel verimini düşürür: **slotların %25'i boşa gitti.**

## 6. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `modules/marangoz_r64_1000.mod.pt` | `3d5cafdc431ff0c70d5a4b4f0dc4cb8012aaacc25d7aef4c45e444179a6c84da` |
| `data/eval/anka_r29_kapasite_yetenek_2026-09-21.json` | `bf22e742d89c2854d59f9212780ee5d33ce5c725d136177739b4ca33e799a343` |
| `data/eval/anka_r29_kapasite_yetenek_2026-09-21.modul.json` | `3b678640d9660a090bb82a8dd571f573419a642e3786968271a7b324a654a671` |
| `scratch/t0099d_zincir.log` | `da2cd9963cf38331b18fe56276ae19119c8c4d7bbdacee1623b717be660653e9` |

Modül meta: `adim=1000 · r=64 · alpha=128 · lr=2e-4 · blok=128 · veri=t0099_wiki_mix_makale.bin ·
kayıp ilk 6,2101 → son60 4,3971 ± 0,2168 · 21,3 MB`.

**Commit yok · `git add -A` kullanılmadı · donmuş yollara yazılmadı.**
