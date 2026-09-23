# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — replay'in ALANI

**İlan:** `data/eval/anka_r27_wikireplay_ilani_2026-09-21.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099b_zincir.log` · eğitim 1000 adım 270 sn (0,27 sn/adım) · `rc=0` · 15:22:16Z→15:32:30Z

---

> ## ⚠ DÜZELTME 2 (21 Eyl 2026, r29 denetiminden SONRA — sessiz düzenleme değil)
>
> **"Tek değişken = replay'in ALANI" iddiası GERİ ÇEKİLMİŞTİR.** Ölçüldü: Wikipedia
> bloklarında `<OUTPUT>` yoktur ⇒ `mask_prompt_targets:87-88` **tüm hedeflerini −100 yapar**
> ⇒ **wiki replay'i hiç gradyan üretmedi** (korunan hedef %0,00). Chat replay ise aktifti
> (%25,08). Yani r27'de **iki** şey değişti: alan **ve** replay'in etkisizleşmesi.
>
> Sonuç: +%13,02 → +%6,52 iyileşmesinin açıklaması **"alan uyumu" değil, "replay'i kapatmak"**
> olabilir. §1'in "H3 DESTEKLENDİ — replay alanı bağlayıcı kısıttı" hükmü **geri çekilmiştir**;
> `unutma_gec: True` olgusu durur, **açıklaması** değişir.
> Ayrıntı ve ölçüm tablosu: `anka_r28_makale_yetenek_2026-09-21.md` başındaki DÜZELTME.

---

## 1. Hüküm: H3 DESTEKLENDİ — ve unutma kapısı İLK KEZ geçildi

| | ilan edilen sınır | ölçülen |
|---|---|---|
| **H3** replay alanı bağlayıcı | A CE ≤ **3,8888** (artış ≤ +%10,0) | **A CE = 3,7658 = +%6,52** ✔ |
| H1 kapasite bağlayıcı | A CE ≥ 3,9600 (artış ≥ +%12,0) | elendi |

**Ölçülen, ilan edilen sınırı karşıladı** ⇒ unutmanın bağlayıcı kısıtı **kapasite değil, replay'in
alanıydı.** `A_gec: True` · `B_gec: True` · **`unutma_gec: True`**.

**"İlk kez" iddiası taramayla doğrulandı** (ezberden yazılmadı): kayıtlı **15 kanonik koşumun
15'inde** `unutma_gec` **False**; r27 tek **True**.

## 2. Tek değişken — geri kalan her şey sabit

| eksen | chat replay (r24) | **WIKI replay (r27)** |
|---|---|---|
| replay kaynağı | `data/train_chat_balanced.bin` | **`data/anka_a1r_pretrain.bin`** |
| ceket kaynağı · desen · oran | `…_anka.bin` · `[ceket×3, replay]` · %25,00 | **aynı** |
| toplam / ceket / replay jeton | 1.046.429 · 784.797 · 261.632 | **aynı** |
| blok · mix tohumu · eğitim tohumu · lr · r · adım | 128 · 42 · 43 · 2e-4 · 16 · 1000 | **aynı** |
| **A CE** | 3,9953 → **+%13,02** | 3,7658 → **+%6,52** |
| **B düşüş** | +2,42 | **−0,28** (B **iyileşti**) |
| ROUGE-L | 0,0258 | 0,0228 |

**Unutma yarıya indi** (+%13,02 → +%6,52) ve B ekseni düşmek yerine **iyileşti**. Karışım
birebir aynı boyutta üretildi (1.046.429 jeton) ⇒ tek değişken gerçekten tek.

> **DÜZELTME (21 Eyl 2026, r28 sonrası eklenmiştir — ölçüm sonrası sessiz düzenleme değil).**
> Yukarıdaki *"B ekseni … iyileşti"* ifadesi **fazla okumaydı**. r28 (aynı kurulum, makale
> kuralı) B düşüşünü **+0,28** verdi; işaret döndü. İkisi de Wilson yarı-genişliğinin (±1,95)
> çok altında ⇒ doğru okuma **"B DEĞİŞMEDİ"**dir. **A ekseni hükmü etkilenmez.**
> Ayrıca §5.1'deki "makale komşuluğu dışlanmadı" **açığı r28 ile KAPANDI**: A CE 3,7658 → 3,7502
> (fark 0,21 SE) ⇒ komşuluk sızıntısı **yok**. Ayrıntı: `anka_r28_makale_yetenek_2026-09-21.md`.

## 3. Yetenek ekseni DEĞİŞMEDİ — ölçüldü, "düştü" denemez

| | ROUGE-L ort ± std (n=100) | SE |
|---|---|---|
| chat replay @1000 | 0,0258 ± 0,0246 | 0,0025 |
| WIKI replay @1000 | 0,0228 ± 0,0244 | 0,0024 |

fark = **−0,0030** · iki örneklem SE = 0,0035 · **\|fark\|/SE = 0,87 ⇒ AYIRT EDİLEMEDİ**.

Kıyas için A farkı: −0,2295 · SE 0,0703 · **\|fark\|/SE = 3,26 ⇒ AYIRT EDİLEBİLİR**.

⇒ **Unutma gerçekten azaldı; yetenek kımıldamadı.** Yani "daha az unutmak" burada
"daha çok yetenek" ile gelmedi. Dört yetenek kapısı hâlâ düşük:
ezber %0,00 ✓ · tutarsızlık **%99,00** ✗ · ROUGE **0,0228** ✗ · kesişim **%3,00** ✗.

## 4. Kapılar — hepsi ateşlendi ve geçti

| kapı | sonuç |
|---|---|
| **SIZINTI (içerik düzeyi)** | üretimde A ekseninin 256 penceresi dışlandı (781.250→780.994 blok, 257 koşu); nihai karışımın 128 hizalı **8.175 bloğu** hash'lendi, 256 pencere hash'iyle **KESİŞİM 0** |
| **VAKUM** | takma öncesi/sonrası logit farkı **8,057** ≠ 0 ⇒ modül canlı, no-op değil |
| **TABAN DOKUNULMAZLIĞI** | `b93cc1cd…` başta = sonda · değişen taban tensörü **0** |
| **BLOK BOYUTU** | `--block-size 128` açıkça verildi; kayıt `blok=128` yazdı (T-0099'un fail-closed kapısı) |
| **TEK EĞİTİCİ** | koşum öncesi `lsof scratch/t0099_wiki_mix_r4.bin` boş |

## 5. Ölçümün SINIRI — ilan edilenden dar olan kısım

1. **Bu bir "aynı külliyat içi replay"tir, yalnız "aynı alan" değil.** Replay blokları A
   ekseninin okunduğu **dosyanın kendisinden** geliyor (ayrık bloklar). Kapı *birebir blok*
   çakışmasını **0**'a indirdi, ama **makale komşuluğunu dışlamadı**: bir eval penceresiyle aynı
   makalede bulunan komşu bloklar hâlâ eğitime girebilir. Ölçülmedi — bu bir **açık**.
2. **Yetenek ekseninde hüküm kurulmadı** — ilan §6 gereği hüküm A üzerinedir; yetenek sayıları
   kaydedildi, yorumu ayrı sorunun konusudur.
3. **Adım ekseni bu koşumda sabit** (1000). Wiki replay ile 6000 adım **ölçülmedi** ⇒
   "wiki replay 6000'de de tutar mı" bilinmiyor.

## 6. Bu neyi değiştirir

* **Kapasite hipotezi (H1) geriledi.** `anka_r26`'nın "kapasite yetersiz" aday ekseni,
  ölçülen tek kaldıraçla (replay alanı) **açıklanmadan kaldı**: sorun r=16 değildi.
* **Replay gerçekten kaldıraç — ama yalnız doğru alandan.** `anka_r20`'nin "replay iki ekseni
  birden iyileştirir" bulgusu doğruydu; eksik olan, replay'in **neyi** koruduğuydu: chat replay
  chat'i korur, A eksenini (Wikipedia) korumaz.
* **Unutma ile yetenek ayrıştı:** bu koşumda unutma yarıya inerken yetenek *ayırt edilebilir
  biçimde* değişmedi ⇒ ikisi aynı kaldıraçla birlikte hareket etmiyor. Bu, T-0096'nın
  "kayıp ↔ unutma birlikte hareket eder" desenine **karşı ilk ölçülmüş kanıttır**.

## 7. Artefakt digestleri (TAM sha256)

| dosya | boyut | sha256 |
|---|---|---|
| `data/anka_a1r.pt` (taban) | 373,7 MB | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `modules/marangoz_wikireplay_1000.mod.pt` | 5,1 MB | `835d0db9773ac4facc96ffc9c1d6cdb7104438051eeb85155956319d86db4b8c` |
| `scratch/t0099_wiki_replay.bin` (çakışmasız dilim) | 190,9 MB | `49f88cb8d13f42cd…` (tam digest `scratch/t0099_wiki_replay.bin.meta.json`) |
| `scratch/t0099_wiki_mix_r4.bin` (karışım) | — | `0a16dd54981f0d175afb8f24375bed748ecd0facf1df8d4d730c9d5b886a6f7a` |
| `data/eval/anka_r27_wikireplay_yetenek_2026-09-21.json` | — | `3cc1708d51835832e453dd1d601b8f11c49b257cff6527cd28881a05c4b5804f` |
| `data/eval/anka_r27_wikireplay_yetenek_2026-09-21.modul.json` | — | `c0bd2656bc0d42fed7b7e88317f533a3b6a7686891b75588922ac4ec2db39f13` |
| `scratch/t0099b_zincir.log` | — | `6911cfb5a85122186e22fc699677ca4cb4ca3ca2542d1d76835acedd3d407174` |

Modül meta: `adim=1000 · lr=2e-4 · blok=128 · veri=t0099_wiki_mix_r4.bin ·
kayıp ilk 6,1518 → son60 4,7143 ± 0,2218`.

> **Kayıp karşılaştırması YAPILMADI:** chat replay ve wiki replay farklı **veri** üzerinde
> eğitildi ⇒ kayıp ölçekleri kıyaslanamaz. Yalnız A/B eksenleri (aynı değerlendirme) kıyaslandı.

## 8. Açık

* ~~**Makale komşuluğu** dışlanmadı (§5.1)~~ → **r28 ile KAPANDI** (21 Eyl 2026): 306 makale
  dışlandı, A CE 3,7658 → 3,7502 (fark 0,21 SE) ⇒ sızıntı **yok**. Kalan ve ölçülmemiş kanal
  **konu yakınlığı**dır (aynı dökümden komşu olmayan ama konu bakımından yakın makaleler).
* **Wiki replay ile 6000 adım** koşulmadı.
* **Yetenek neden kımıldamıyor** — unutma düzeldi, yetenek düzelmedi. Bu artık ayrı ve
  bağımsız bir soru; `r`/hedef katman/adım eksenleri burada hâlâ denenmemiş.
* **Commit yok. `git add -A` kullanılmadı.** Donmuş yollara yazılmadı; `data/*.bin` donmuş
  olduğu için karışım `scratch/`'e yazıldı.
