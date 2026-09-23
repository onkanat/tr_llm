# T-0096 — Anka marangoz (ceket) tam giydirme koşumu · sonuç raporu

> Bu raporun **her sayısı** `scratch/t0096_kos/sonuc.json`'dan betikle okunmuştur
> (`scratch/t0096_rapor.py`). Elle yazılmış ölçüm yoktur.

| kalem | değer |
|---|---|
| görev | **T-0096** (yürütücü: claude) |
| operatör talimatı | *"Marangoz eğitilmeli olası en yüksek hassasiyete 3-5 saat eğitim kabul edilebilir."* |
| operatör seçimi | replay oranı **%25** (`--replay-every 4`) |
| ilan (koşumdan ÖNCE) | `data/eval/anka_r17_marangoz_ilani_2026-09-20.md` |
| başladı (UTC) | 2026-09-20T17:45:24Z |
| bitti (UTC) | 2026-09-20T20:50:38Z |
| toplam duvar süresi | 11115.4 sn (3.09 saat) |
| taban | `data/anka_a1r.pt` |
| karışım | `scratch/t0096_kos/mix_r4.bin` (replay %25.0) |

---

## §1 — Ölçülen düzeltme: T-0094/K8 **ÇÜRÜDÜ**

T-0094 *"ilanın epoch sayısı ~8× yanlış"* demişti; bu bulgu T-0095 raporuna da geçmişti.
Doğrudan çalıştırılan ölçüm:

```
KristalDataset(p, block_size=128); x, y = ds.get_batch(batch_size=8)
  -> x.shape=(8, 128)  => adim basina 1,024 jeton, 1 blok DEGIL
  -> ceket 784.797 jeton => 1 tam gecis = 766,4 adim
  -> 1.000 adim => R0 1,30 · R4 0,98 ceket epoch   <-- ILANIN sayilari DOGRUYDU
```

**Sonuç:** ilan doğruydu; *"ölçülen 0,16"* ölçüm kabının **örtük** varsayımından
(adım başına 1 blok) geliyordu. Ceket metasının `steps_formulu`'su da **doğrudur**
(`ceil(784797/(128*8))*3 = 2301`, 3 epoch). Ayrıntı: ilan §1.

---

## §2 — Sabit yapılandırma (sonda kollarıyla birebir aynı)

| kalem | değer |
|---|---|
| sözlük | `data/rebuild/vocab_anka_r1_33114.json` |
| block / batch | 128 / 8 |
| adım başına jeton | **1,024** |
| öğrenme oranı | 0.0002 (sabit; **program YOK**) |
| rejim | SFT (prompt maskeli) — `--pretrain` YOK |
| cihaz | mps (her segment log'unda doğrulandı) |

---

## §3 — Segment tablosu (her segmentten sonra EŞLİ değerlendirme)

| S | +adım | küm | süre | ilk kayıp | son60 | A (CE artışı) | B (puan) | UNUTMA | ROUGE-L | ezber | tutarsızlık | kesişim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | +1,000 | 1,000 | 404.0 sn | 3.1864 | 1.0256 ± 0.2404 (n=60) | %+23.31 | +0.08p | KALDI | 0.1097 | %20.00 | %24.00 | %0.00 |
| S2 | +2,000 | 3,000 | 896.2 sn | 0.7112 | 0.5132 ± 0.1894 (n=60) | %+39.28 | +4.88p | KALDI | 0.1173 | %0.00 | %5.00 | %0.00 |
| S3 | +3,000 | 6,000 | 1395.5 sn | 0.6498 | 0.3295 ± 0.1507 (n=60) | %+46.78 | +8.01p | KALDI | 0.1313 | %0.00 | %12.00 | %10.00 |
| S4 | +4,000 | 10,000 | 1788.2 sn | 0.3721 | 0.2912 ± 0.1618 (n=60) | %+56.42 | +10.63p | KALDI | 0.2065 | %0.00 | %3.00 | %7.00 |
| S5 | +6,000 | 16,000 | 2636.9 sn | 0.1932 | 0.2133 ± 0.1432 (n=60) | %+65.01 | +10.19p | KALDI | 0.2777 | %0.00 | %4.00 | %8.00 |
| S6 | +8,000 | 24,000 | 3504.2 sn | 0.0938 | 0.1578 ± 0.1078 (n=60) | %+75.62 | +8.29p | KALDI | 0.3035 | %0.00 | %5.00 | %10.00 |

Eşikler (ilan §4, T-0059): **A ≤ +%10** · **B ≥ −5,0 puan** · ezber < %10 · tutarsızlık < %5 · ROUGE-L ≥ 0,35 · kesişim ≥ %80.

| S | ROUGE medyan | n | A taban CE | A kol CE | B taban top1 | B kol top1 | checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| S1 | 0.0952 | 100 | 3.5351739511825144 | 4.359172218944877 | 49.5638 | 49.4845 | `8cede2393a60ea62641c1bff1560ab9543e95f97848f33e69d41f6f4cbc6f5f6` |
| S2 | 0.0834 | 100 | 3.5351739511825144 | 4.92374117532745 | 49.5638 | 44.6868 | `ce4c76b69f3e8aead67b2508b52405c484ccc8995bdf8c0f72a583deb72a70ce` |
| S3 | 0.0952 | 100 | 3.5351739511825144 | 5.188938017934561 | 49.5638 | 41.5543 | `3045b420ed37e89cdbec5f235886914e583849603c81943b760a1627559eca70` |
| S4 | 0.1647 | 100 | 3.5351739511825144 | 5.529599555768073 | 49.5638 | 38.9374 | `4c12d233e9d834dfa5b5da5cee2666c6312e3b20712c3dd35ce1e5b98b2af6fe` |
| S5 | 0.2543 | 100 | 3.5351739511825144 | 5.833526654168963 | 49.5638 | 39.3735 | `2212582fd8a5ce6723fce682d68babdfb2c34d2bedfbd358c062e0d79f84fe92` |
| S6 | 0.3254 | 100 | 3.5351739511825144 | 6.20863498840481 | 49.5638 | 41.2768 | `136dda76da419e281cf8b800a0e562381c5e9c7db8ed85136ac3ee4f19578b8b` |

---

## §4 — İlan edilen seçim kuralının UYGULANMASI

Kural (ilan §4, koşumdan ÖNCE yazıldı, değiştirilmedi): sert kapı `unutma_gec` → dışlama `ezber ≥ %10` veya `tutarsızlık ≥ %5` → en yüksek `rouge_l_ort`.

**Karar: `KAPIYI GECEN YOK`**

* Gerekçe: ilan §4/5 — PASS yazilmaz

| dışlanan S | neden |
|---|---|
| S1 | ezber %20.00 >= %10.0 veya tutarsizlik %24.00 >= %5.0 |
| S2 | ezber %0.00 >= %10.0 veya tutarsizlik %5.00 >= %5.0 |
| S3 | ezber %0.00 >= %10.0 veya tutarsizlik %12.00 >= %5.0 |
| S4 | sert kapi: A %+56.42 (<=+%10.0) B +10.63p (>=-5.0) |
| S5 | sert kapi: A %+65.01 (<=+%10.0) B +10.19p (>=-5.0) |
| S6 | ezber %0.00 >= %10.0 veya tutarsizlik %5.00 >= %5.0 |


**Erken durma ateşlemedi** (ilan §4/4 kuralı koşum boyunca tetiklenmedi).

---

## §4b — İlan edilen çürütme maddeleri (Ç1…Ç5) — hüküm ÖLÇÜMDEN

İlan §5 maddelerinin her biri, koşumun kendi sayılarıyla hükme bağlanır. Hücreler `sonuc.json` ve `kesisim_sondasi.json`'dan okunur.

| # | öncül | ölçülen | hüküm |
|---|---|---|---|
| **Ç1** | %25 replay unutmayı taşınabilir tutar | `unutma_gec` segmentler: YOK · A artışları: [23.31, 39.28, 46.78, 56.42, 65.01, 75.62] | **ÇÜRÜDÜ — %25 yetersiz, marangoz iddia edilemez** (Ç1 çürütücüsü: *her* segmentte `unutma_gec == false`) |
| **Ç2** | kısıt takvimdir, daha çok epoch çözer | kümülatif < 6.000'de `ezber ≥ %10` olan segmentler: **[1]** | **ÇÜRÜDÜ — kısıt KÜLLİYATTIR; takvim değil** |
| **Ç3** | ceket bu boyutta yakınsar | en yüksek `rouge_l_ort` = **0.3035** (çürütücü eşiği 0,15) | **AYAKTA** |
| **Ç4** | ceket soru-içeriğine bağlanır | **pozitif kontrol:** insan uzman cevabı `kesisim` **%13.00** · eşik **%80.0** ⇒ **6.15× uzak** | **VAKUM — kapı ayırt edici DEĞİL** (kusursuz model de düşer; hüküm model hakkında bilgi taşımaz) |
| **Ç5** | erken durma kuralı ateşler | `erken_durma` = **ATEŞLEMEDİ** | **ÇÜRÜDÜ — kural ayırt edici değil (rapora yazılır, 'sorun yok' sayılmaz)** |

Ç4'ün VAKUM olmasının mekanizması (ölçüldü, tahmin değil): kesişim dağılımı **iki kütleli** — 100 held-out kaydının 80'i 0–1 ortak kelime, 20'si 7–18 ortak kelime, **2–6 arası hiç kayıt yok**. Soru ortalama 4,8 kelime, cevap 21,4; Türkçe çekim yüzey eşleşmesini kesiyor (soru `kalınlığa` ↔ cevap `kalınlık`). Üretici: `scratch/t0096_kos/kesisim_sondasi.py` (CPU; GPU koşumuna dokunmaz).

---

## §4c — Eşik sayımı (hükmün NEDENİ; her hücre ölçümden)

`KAPIYI GECEN YOK` hükmü tek bir kusurdan değil, **iki eksenin birden** düşmesinden gelir. Sürücünün `neden` dizgisi **yalnız ilk düşen kovayı** yazar; tam sayım aşağıdadır:

| eşik | istenen | geçen segmentler | kaç/6 |
|---|---|---|---|
| **sert kapı — unutma (A ekseni)** | A ≤ **+%10** | **YOK** | **0/6** |
| ezber | < %10 | S2, S3, S4, S5, S6 | 5/6 |
| tutarsızlık | < %5 | S4, S5 | 2/6 |
| **ceket — ROUGE-L** | ≥ **0,35** | **YOK** | **0/6** |
| kesişim | ≥ %80 | **YOK** | 0/6 **← ULAŞILAMAZ eşik, §4b/Ç4** |

* **En iyi ceket noktası:** ROUGE-L **0.3035** (eşiğe oran **0.867×**, yani eşiğin **%86.7**'i) — S6'da, 24.000 adımda.
* **En iyi unutma noktası:** A artışı **%23.31** (izin verilen +%10'un **2.33×**'i) — **S1'de**, yani en kısa koşumda.

**İki eksen AYNI YÖNDE hareket etmiyor:** süre arttıkça ceket **düzeliyor** (ROUGE 0,1097 → 0,3035; monoton) ve taban **bozuluyor** (A +%23,31 → +%75,62; monoton). Ayrım noktası yok: en iyi unutma noktası en kötü ceket noktasıdır. Süreyi uzatmak ceketi eşiğe **%87**'ye kadar taşıdı ama **geçirmedi** (0,3035 < 0,35); kısaltmak unutmayı **+%23**'e indirdi ama yine **geçirmedi** (23,31 > 10).

**Sürücünün gerekçe dizgisinde İKİ kusur (beyan edilir, düzeltilmedi):**

1. **Sıra sapması.** İlan §4 *"sert kapı `unutma_gec` → dışlama `ezber ≥ %10` veya `tutarsızlık ≥ %5`"* diyor; kod ise **önce kalite filtrelerini** (`t0096_marangoz.py:188`), **sonra sert kapıyı** (`:195`) uygular. Nihai **küme** bu sıradan **etkilenmez** (kesişim değişmeli: hepsini geçmeyen dışarıda kalır) ama **kaydedilen gerekçe etkilenir** — S2/S3/S6 için kayıt *kalite filtresini* adlandırır, oysa o segmentler sert kapıyı **da** düşürmüştür (A +%39,28 / +%46,78 / +%75,62 ≫ +%10). Üçü de aynı kayıtla **iki farklı gerçeği** anlatır.
2. **Gerekçe dizgisi SABİT ŞABLON** (`:191-192`): iki karşılaştırmayı `veya` ile birleştirip **ikisini de** basar; hangi maddenin ateşlediğine bakmaz. Bu yüzden S2/S3/S6 kayıtlarında **`ezber %0.00 >= %10.0`** yazar — **YANLIŞ bir önerme**. Sayı doğrudur, **yüklem yanlıştır** ([[denetim-kapsami-iddiadan-dar]]). Denetim kaydını okuyan biri, tetiklenmemiş bir eşiği tetiklenmiş sanır. **6 gerekçenin 3'ü** bu kusuru taşır.

---

## §5 — Teslim

* **YAZILMADI** — ilan §4/5 — kapiyi gecen segment YOK; 'PASS' yazilmaz

---

## §6 — Beyan edilen sınırlar ve kusurlar

1. **LR programı yok** (sabit 2e-4). Uzun koşuda programlı düşüş daha iyi son checkpoint
   verebilir; bu koşum onu **ölçmedi**. Karşılaştırılabilirlik bilinçli olarak öne alındı.
2. **AdamW momentleri `anka_a1r.pt` için yok** ⇒ S1 momentsız başlar (T-0092: ilk güncelleme
   1,7306× büyük). S2'den itibaren zincir momentsı taşır (`--save-optimizer`/`--load-optimizer`).
3. **Ceket külliyatı küçük** (784.797 jeton / 11.708 kayıt ≈ 67 jeton/kayıt) ⇒ ezber riski
   yapısal; ilan §4/2 ve §4/4 bunu ölçüp dışlamak için var.
4. **Değerlendirme `--max-new 128`** ile yapıldı; arka kapı 45 jeton üretir
   (`src/rag/epistemic_agent.py:311`) ⇒ bu raporun üretimleri **kapının üretimi değildir**.
5. **Seçim tek tohumla** (değerlendirme `--seed 42`, n=100). Wilson aralığı raporda değil;
   `rouge_l_ort` farkları **küçükse** segmentler arası sıralama gürültülü olabilir.
6. `src/llm/frozen_guard.py:59` mutlak-yol FAIL-OPEN açığı **kapatılmadı** (yalnız beyan).
7. D1 (en-uzun-kök) ve D4 (kesme/rakam) **kapsam dışı**.
8. **`son60` AYRIŞTIRICI KUSURU (bu raporun kendi aracında).** Koşum sürücüsü `scratch/t0096_marangoz.py:126` satırı `satir.strip().startswith("son60")` diye arıyor; log satırı ise `Kayıp Dağılımı:    son60 1.0256 ± 0.2404 (n=60)` biçiminde ⇒ `sonuc.json`'daki `son60` alanı **boş** kaldı. Sürücü **koşarken düzeltilmedi** (düzeltmek, segmentleri üreten kodu kendi artefaktından koparırdı — [[kabul-kosusu-olctugu-artefakti-degistirir]]). Değerler bu raporda **koşumun kendi `seg_i.log`'undan** kurtarıldı (`son60_kurtar`); elle yazılmadı.
9. **Değerlendirme JSON'larının provenance alanı BAYAT.** Her `eval_seg_*.json` içindeki `gorev` alanı **`T-0094`** yazar (betiğe sabit-kodlu) — bu koşumun değerlendirmeleri T-0096'ya aittir. Alan **düzeltilmedi** (donmuş/geçmiş ölçüm kayıtları ve `scripts/**` değiştirilmez); **beyan edilir**. `model_sha256` alanı doğrudur ve kimlik bu raporda ondan doğrulanır.
10. **`kesisim_orani` eşiği ULAŞILAMAZ** (§4b/Ç4): insan uzman cevabı %13 alır, eşik %80'dir. Bu ölçüt **karar veremez**; raporda iyileşme/yetersizlik kanıtı olarak **kullanılmamıştır**. `tutarsizlik_orani` eşiğinin payı da dar (insan %3,00 ↔ eşik %5 ⇒ 1,67×) — beyan edilir.
11. **KAPSAM GENİŞLETMESİ — BEYAN EDİLEN KURAL İHLALİ (K1).** `scratch/t0096_rapor.py` ve `scratch/t0096_kapanis.py` **`writes[]`'te yokken yazıldı**; beyan **sonradan** genişletildi (8 → 10 girdi). Bu, SPEC Kural 1'in ihlalidir ve **gizlenmez**. Yazılı kayıt `.agent-bus/notes/T-0096.md` §K1'de, gerekçe ve dosya digest'iyle birlikte durur (SPEC §209 bunu zorunlu kılar: şartname değişikliği **olay üretmez**). Kural 2 ihlali **yoktur**: iki yol da `scratch` kiralaması kapsamındadır ve donmuş 10 desenin hiçbirine uymaz.
12. **TESLİM KAYDI HEDEFİN ADINI TAŞIMIYOR — dallar arası asimetri (ölçüldü).** `sonuc.json[teslim]` sözlüğü **`hedef` alanı içermiyor**. Sürücü bu alanı **yalnız yazan dalda** kuruyor (`scratch/t0096_marangoz.py:351`); **yazmayan iki dalda** (`:338`, `:360`) yalnız `neden` yazıyor. Üstelik kardeş yazmayan dal (`:338-341`) `stderr`'e basarken **ateşleyen dal** (`:359-361`) **log'a hiçbir şey basmıyor** ⇒ bu koşumda teslim kararı **log'da iz bırakmadı** ([[duran-dal-iz-birakmaz]], T-0095 sınıfı). Sonuç: teslim hedefinin **adı** koşumun kendi kaydından okunamadı; kapanış denetimi adı sürücünün kendi sabitinden (`CEKET_CIKTI`, `:46`) **ölçtü** (elle yazılmadı). Kayıt düzeltilmedi: sürücü, ölçtüğü artefaktı üreten betiktir ([[kabul-kosusu-olctugu-artefakti-degistirir]]).
13. **Seçim gerekçesinde SIRA SAPMASI ve SABİT ŞABLON** (ayrıntı §4c): kod sert kapıdan önce kalite filtresini uygular (ilan tersini der) ve gerekçe dizgisi hangi maddenin ateşlediğine bakmadan iki karşılaştırmayı birlikte basar ⇒ 6 gerekçenin **3'ü** tetiklenmemiş bir eşiği tetiklenmiş gösterir. **Nihai hüküm etkilenmez** (dışlama kümesi değişmeli), ama **kayıt etkilenir**. Düzeltilmedi — sürücü ölçülen artefaktı üretir; kusur beyan edilir.

---

## §7 — Digest tablosu (tam sha256; önek DEĞİL)

Bu raporun kendi çıktı yolları (`data/eval/anka_r17_marangoz_2026-09-20.md`, `data/eval/anka_r17_marangoz_2026-09-20.json`) tabloya **alınmadı**:
kendini listeleyen bir digest tablosu yazıldığı anda bayatlar (T-0095 §7).

| yol | önce | sonra | dokunulmadı |
|---|---|---|---|
| `data/anka_a1r.pt` | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | **True** |
| `data/train_carpenter_specialization_anka.bin` | `8cd0696d90ef3816613cdc1c452580a6e9ab381085462e1d3f3e0e19e4f9c25a` | `8cd0696d90ef3816613cdc1c452580a6e9ab381085462e1d3f3e0e19e4f9c25a` | **True** |
| `data/train_chat_balanced.bin` | `443f93d352b07defca91b2a620f2e4154c3ed0b14f7c695535c959209e7608fa` | `443f93d352b07defca91b2a620f2e4154c3ed0b14f7c695535c959209e7608fa` | **True** |
| `data/rebuild/vocab_anka_r1_33114.json` | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` | **True** |
| `data/lexicon/roots_anka_r1.tsv` | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` | **True** |
| `data/eval/anka_r17_heldout_2026-09-20.jsonl` | `c397eb08218937ea000cc6120f42a8783e1dcab7948c70b33537af74b50d096e` | `c397eb08218937ea000cc6120f42a8783e1dcab7948c70b33537af74b50d096e` | **True** |

| artefakt | sha256 |
|---|---|
| `scratch/t0096_kos/mix_r4.bin` | `a31a58b0aaa6903ba4b05fa5a3a24d2ea4eadea14b148d1e726da21062b3f810` |
| `data/eval/anka_r17_marangoz_ilani_2026-09-20.md` | `3c60ac940d4a23ac14d77205a73e8ca41806db1571e480516d34a27eca387c77` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_1.pt` | `8cede2393a60ea62641c1bff1560ab9543e95f97848f33e69d41f6f4cbc6f5f6` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_1.pt.opt.pt` | `35b214e55562667832e548ea74c327c29d9c63b00524317cdc1258571319de9d` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_2.pt` | `ce4c76b69f3e8aead67b2508b52405c484ccc8995bdf8c0f72a583deb72a70ce` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_2.pt.opt.pt` | `e24fb988dc8ebf536dd7292bf6bb266dbd620795e15f404229a3675cee6c27dd` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_3.pt` | `3045b420ed37e89cdbec5f235886914e583849603c81943b760a1627559eca70` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_3.pt.opt.pt` | `67471f4d6d5df7ead2bc00d4556be360a1c7c1e31ae74946d60072d374c4bc01` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_4.pt` | `4c12d233e9d834dfa5b5da5cee2666c6312e3b20712c3dd35ce1e5b98b2af6fe` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_4.pt.opt.pt` | `f91dfae425f8d0f99c68865cbb1869eb2e1d90f7e7ffd6d9cb4f819adf1ab9f8` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_5.pt` | `2212582fd8a5ce6723fce682d68babdfb2c34d2bedfbd358c062e0d79f84fe92` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_5.pt.opt.pt` | `6b77a712d9db6b7804ce39755f0c80376f382eec9391e22dac3b128def7a7c1d` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_6.pt` | `136dda76da419e281cf8b800a0e562381c5e9c7db8ed85136ac3ee4f19578b8b` |
| `/Users/hakankilicaslan/Git/tr_llm/scratch/t0096_kos/seg_6.pt.opt.pt` | `56352980749fbeb1ccf31fc5cd8f3983f0dca5c92a6aded975da7be30a15b468` |

## §8 — Kapanış kanıtları (betikle ölçüldü)

| kanıt | sonuç |
|---|---|
| **Ön-kayıt sırası** (ilan mtime < tüm çıktılar) | TEMIZ — ilan, olculen TUM ciktilaridan once yazildi |
| Donmuş yüzey taraması (T-0096 `claimed_at=2026-09-20T17:38:44Z` sonrası) | taranan **1532** · donmuş desene eşleşen **69** · sonrası yazım **0** · beyan dışı **0** ⇒ **TEMIZ** |
| **Beyan denetimi** (`writes[]` ↔ yazılanlar, iki yönlü, `mtime >= claimed_at`) | beyan **10** · taranan **784** · yazılıp **beyan DIŞI** **0** · beyan edilip **YOK** **2** (gerekçeli **2** · beklenmeyen **0**) ⇒ **TEMIZ (iki yon kapali; 2 yol ON-KAYITLI kural geregi YOK)** |
| `scratch/*.py` py_compile kanaryası | 166 betik · bozuk **0** ⇒ **TEMIZ** |
| Rapor ↔ JSON sayı denetimi | 54 kontrol · uyuşmazlık **0** ⇒ **TEMIZ** |
| Teslim kimliği (bayt kopyası) | TESLIM YOK |
| `venv/bin/pytest -q` | `1 failed, 237 passed in 81.51s (0:01:21)` |

**Beyan edilen ama var olmayan** yollar (üç değerli sınıflandırma):
  * `data/anka_a1r_ceket.pt` ⇒ **GEREKÇELİ** — ön-kayıtlı kural gereği üretilmemesi DOĞRU
  * `data/anka_a1r_ceket.pt.opt.pt` ⇒ **GEREKÇELİ** — ön-kayıtlı kural gereği üretilmemesi DOĞRU

*Gerekçe ölçümden türetildi:* `sonuc.json[teslim].yazildi == False` ⇒ neden *"ilan §4/5 — kapiyi gecen segment YOK; 'PASS' yazilmaz"*. **Kusur (beyan edilir):** koşumun kendi kaydı `teslim` sözlüğünde **`hedef` alanını taşımıyor** — sürücü bu alanı yalnız *yazan* dalda yazıyor (`scratch/t0096_marangoz.py:351`), *yazmayan* iki dalda yazmıyor (`:338`, `:360`). Bu yüzden teslim hedefinin **adı**, koşumu üreten betiğin kendi sabitinden (`CEKET_CIKTI`, `:46`) okundu. Aynı `if/else` içinde kardeş dal (`:338-341`) `stderr`'e basarken **ateşleyen dal** (`:359-361`) **log'a hiçbir şey basmaz** ⇒ [[duran-dal-iz-birakmaz]] sınıfı.

Düşen test(ler): `tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints`

**Not — ÖZ-REFERANS YASAĞI ve İKİ AYRI DIGEST ÖNERMESİ.** Bu raporun kendi MD
digest'i **buraya yazılmaz**: kendini tarif eden bir digest satırı yazıldığı anda
bayatlar (T-0095 §7). Bunun yerine iki **ayrı** önerme `…json →
kapanis.rapor_md_digest` altında ölçülür ve karıştırılmaz:

1. `uretici_iddiasi_kendi_ciktisiyla_tutarli` — rapor üreticisinin kaydettiği
   digest, **kendi yazdığı gövdeyi** doğru tarif ediyor mu?
2. `iddia_nihai_dosyayi_tarif_ediyor` — o iddia, §8 **eklenmiş nihai** dosyayı
   tarif ediyor mu?

Bu koşumda (1) **doğru**, (2) **YANLIŞ**'tır: kapanış, ölçtüğü artefaktı
(MD'yi) §8'i ekleyerek **değiştirdi** ⇒ üreticinin digest'i nihai dosyayı artık
tarif etmez ([[kabul-kosusu-olctugu-artefakti-degistirir]]). Okuyucu genelde
(2)'yi sınar ve **başarısız** bulur. Nihai digest
`rapor_md_sha256_kapanis_sonrasi` alanındadır; üreticinin iddiası **ezilmedi**,
olduğu gibi durur — ikisi karşılaştırılabilsin.
