# ANKA · ÖN-KAYITLI TEK DENEME SONUCU — makale komşuluğu

**İlan:** `data/eval/anka_r28_makale_ilani_2026-09-21.md` (**ölçümden önce** yazıldı)
**Koşum:** `scratch/t0099c_zincir.log` · 1000 adım · 283 sn · `rc=0` · 15:49:45Z→16:00:35Z

---

> ## ⚠ DÜZELTME (21 Eyl 2026, r29 denetiminden SONRA eklenmiştir — sessiz düzenleme değil)
>
> **Bu raporun temel iddiası yanlıştır.** "Tek değişken = replay'in **alanı**" denmişti;
> gerçekte **iki** şey değişti.
>
> `mask_prompt_targets` (`train_step_demo.py:87-88`), `<OUTPUT>` içermeyen pencerede **tüm
> hedefleri −100** yapar. Wikipedia bloklarında `<OUTPUT>` yoktur. Ölçüldü:
>
> | külliyat | `<OUTPUT>`'lu blok | korunan hedef jetonu |
> |---|---|---|
> | ceket | %100,00 | %43,43 |
> | chat (r24 replay) | %99,98 | %25,08 → **aktif** |
> | **wiki (r27/r28 replay)** | **%75,15** | **%0,00 → ETKİSİZ** |
>
> Beklenen 0,75 × 43,43 = %32,57; ölçülen %32,36 ⇒ **wiki replay'i hiç gradyan üretmedi.**
> Parti başına kapı vardı ama P(8/8 pencere etkisiz) = 0,25⁸ ≈ 1,5e−5 ⇒ **hiç ateşlenmedi.**
>
> ⇒ r27/r28 fiilen **"ceket tek başına + slotların %25'i boşa"**dır. **+%13,02 → +%6,08
> iyileşmesi "alan uyumu"na değil, "chat replay'i kapatmak"a atfedilebilir** — ve bu daha
> yalın bir açıklamadır (chat verisi Wikipedia'dan uzaklaştırır).
>
> **Etkilenmeyenler:** sızıntı denetimi (§4), makale kuralının doğrulanması, taban
> dokunulmazlığı, kapı geçişi olgusu (`unutma_gec: True` **ölçülmüş bir olgudur**, yalnız
> **açıklaması** değişir). **Etkilenen:** "replay'in alanı bağlayıcı kısıttı" cümlesi —
> bu cümle **geri çekilmiştir**. Ayrıntı: `anka_r29_kapasite_yetenek_2026-09-21.md` §5.

## 1. Hüküm: KOMŞULUK SIZINTISI YOK — r27'nin +%6,52'si GERÇEK

| | ilan edilen sınır | ölçülen |
|---|---|---|
| **sızıntı YOK** | A CE ≤ **3,8361** (r27'den fark < 1 SE) | **A CE = 3,7502** ✔ |
| sızıntı VAR | A CE ≥ 3,9064 (fark ≥ 2 SE) | — |

r27 → r28 farkı: **−0,0156** · iki örneklem SE = 0,0756 · **\|fark\|/SE = 0,21** ⇒ **ayırt edilemez
derecede aynı.** 306 makale (7.521 blok, 928.460 jeton) dışarı atıldı ve A CE **kımıldamadı.**

⇒ r27'de ölçülen unutma kazancı **aynı-makale hatırlaması değil, gerçek korumadır.**
Hüküm sınırı ilan edildiği gibi karşılandı; dört daldan biri (sızıntı yok) çıktı.

**`unutma_gec: True`** — ikinci kez ve **farklı bir veri dilimiyle**. İki bağımsız koşum
(r27 blok kuralı, r28 makale kuralı) A CE'de 0,21 SE aralığında buluştu ⇒ bulgu **şanslı bir
çekiliş değil**.

## 2. Tam tablo — aynı kanonik kap, mps, n=100 / n=256 pencere

| koşum | replay kaynağı | A CE | **A artış** (≤+%10) | B düşüş (≤5,0) | ROUGE-L (≥0,35) | `unutma_gec` |
|---|---|---|---|---|---|---|
| taban `anka_a1r.pt` | — | 3,5352 | — | — | 0,0000 | — |
| r24 modül @1000 | chat | 3,9953 | +%13,02 ✗ | +2,42 ✓ | 0,0258 | ✗ |
| r25 modül @6000 | chat | 4,2858 | +%21,23 ✗ | +7,45 ✗ | 0,0505 | ✗ |
| **r27 modül @1000** | **WIKI** · blok kuralı | 3,7658 | **+%6,52 ✓** | −0,28 ✓ | 0,0228 | **✓** |
| **r28 modül @1000** | **WIKI** · **makale kuralı** | **3,7502** | **+%6,08 ✓** | +0,28 ✓ | 0,0191 | **✓** |
| T-0096 seg_1 · tam FT | — | 4,3592 | +%23,31 ✗ | −0,08 ✓ | 0,1097 | ✗ |
| T-0096 seg_6 · tam FT | — | 6,2086 | +%75,62 ✗ | −8,29 ✗ | 0,3035 | ✗ |

## 3. DÜZELTME — r27 raporundaki bir fazla okuma

`anka_r27_…md` §2 *"B ekseni düşmek yerine **iyileşti**"* diyor (r27 B düşüş = −0,28).
r28'de B düşüş = **+0,28** — işaret **döndü**. İkisi de Wilson yarı-genişliğinin (±1,95) çok
altında ⇒ doğru okuma **"B DEĞİŞMEDİ"**dir, "iyileşti" değil. r27'nin tek koşumdan işaret
okuması **fazla okumaydı**; burada düzeltiliyor. A ekseni hükmü **etkilenmez** (o iki koşumda da
aynı yönde ve büyüklükte).

## 4. Kapılar — üçü de ateşlendi

| kapı | sonuç |
|---|---|
| **SIZINTI (iki katman)** | birebir blok kesişimi **0** (8.175 blok) · **makale teması 0 / 2.044** |
| **VAKUM** | logit farkı **7,419** ≠ 0 ⇒ modül canlı |
| **TABAN DOKUNULMAZLIĞI** | `b93cc1cd…` başta=sonda · değişen taban tensörü **0** |
| **TEK EĞİTİCİ** | koşum öncesi `lsof` boş |

**Kural artık KAPIYA bağlı, sınanmış:**
`tests/test_wiki_replay_kapilari.py` — **12/12 geçti** (0,62 sn), her test iki dallı.
**Mutasyonla kanıtlandı, 3/3 yakalandı:** (A) makale kuralı sessizce blok kuralına dönerse
→ 3 test düşer · (B) `replay_every` sabit 4 yazılırsa → 2 test düşer · (C) pencere makale
sınırını keserse görülmezse → 2 test düşer.

**Yeni kod donmuş artefakta karşı sınandı:** r27 dilimini **bit-özdeş** yeniden üretti
(sha256 `49f88cb8d13f42cd…` = aynı).

## 5. Yetenek ekseni yine DEĞİŞMEDİ (ölçüldü)

| | ROUGE-L ort ± std (n=100) | r27'den fark / SE |
|---|---|---|
| r27 | 0,0228 ± 0,0244 | — |
| r28 | 0,0191 ± 0,0234 | −0,0037 / 0,0033 = **1,14 ⇒ AYIRT EDİLEMEDİ** |

Üç wiki-replay/chat-replay koşumunun hiçbirinde yetenek kapıları geçilmedi
(tutarsızlık **%100**, ROUGE **0,0191**, kesişim **%0**). ⇒ **Unutmayı düzeltmek yeteneği
getirmiyor** — bu artık iki bağımsız koşumla ölçülmüş bir ayrışma.

## 6. Ölçümün sınırı

1. **Külliyat içi replay'in tamamı dışlanmadı.** Aynı Wikipedia dökümünden komşu-olmayan ama
   *konu bakımından yakın* makaleler hâlâ eğitime girebilir. Yalnız **makale komşuluğu**
   dışlandı, **konu yakınlığı** dışlanmadı — bu ayrı ve ölçülmemiş bir kanaldır.
2. **Adım ekseni sabit** (1000). Wiki replay ile 6000 adım hâlâ koşulmadı.
3. **r27'nin dilim meta'sında `kural` alanı yok** (kapı o zaman yoktu) ⇒ r27 karışımı yeni
   denetimle yeniden denetlenemez; betik bu durumda `rc=2` ile durur (ölçüldü). 19 bloğun
   ölçümü tek seferlik yeniden kurmayla yapıldı. **r27 meta'sı sonradan DOLDURULMADI.**

## 7. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `data/anka_a1r.pt` (taban) | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `modules/marangoz_makale_1000.mod.pt` | `d70da35a32db9d159c5518cd40ce01d6f0ff14f7de4de1967639c41e186bda84` |
| `scratch/t0099_wiki_replay_makale.bin` | `024a10f58f4c7c86…` (tam digest `.meta.json`) |
| `scratch/t0099_wiki_mix_makale.bin` | `414dcd30f66ce86953db4d5f538eb6cce43df3d165832954d0fd7ef089037dcf` |
| `data/eval/anka_r28_makale_yetenek_2026-09-21.json` | `57717d016c64ef8bde52f8beab37b54c02a6a0cdb1cb3e43e2bd256b55e63a4d` |
| `data/eval/anka_r28_makale_yetenek_2026-09-21.modul.json` | `6a86d41f539df5384ccf6b5887fd29b2265a461ee60ce39bf23f93e76266e330` |
| `scripts/wiki_replay_disjoint.py` | `af5f1fbdf73f55c351e48cc91d87580708cab8ea9dca6b10cff0bd911d0d2e7e` |
| `tests/test_wiki_replay_kapilari.py` | `845b5197e880b833bb21a17e1409e8e148cb9e1f3c7fa05f9ce4f367107cec46` |
| `scratch/t0099c_zincir.log` | `c41f8b997a860c5faa796d9b6838ba6613f3d1ef378103b87234963717e08fdf` |

Modül meta: `adim=1000 · lr=2e-4 · blok=128 · veri=t0099_wiki_mix_makale.bin ·
kayıp ilk 6,1510 → son60 4,6999 ± 0,2249`.

## 8. Ne kalıyor

**Unutma tarafı kapandı** (iki bağımsız koşum, iki farklı dilim, ikisi de kapıyı geçti, sızıntı
ölçülüp dışlandı). **Yetenek tarafı hiç açılmadı**: modül hâlâ ezberlemiyor ama öğrenmiyor da.

Sıradaki soru artık net ve tek: **yetenek neden kazanılmıyor?** Ölçülmüş aday kaldıraçlar
(sırayla, ön-kayıtlı tek deneme olarak): `r` (kapasite), hedef katman kümesi, adım sayısı,
öğrenme oranı. Bunların hiçbiri **yetenek ekseninde** denenmedi.

**Commit yok · `git add -A` kullanılmadı · donmuş yollara yazılmadı.**
