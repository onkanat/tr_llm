# T-0068 ÖLÇÜM RAPORU — BELGE zarf/içerik ayrıştırması

**Sonuç (tek cümle):** Ceketin genel alandaki bozulması **belgeye bağlı DEĞİLDİR** — belge
tamamen çıkarıldığında kayıp **%84.6 oranında** sürüyor ve üç koşulda da
**150/150** öğede ceket kötü.

| | |
|---|---|
| görev | T-0068 (betimleyici tanı ölçümü — **kapı değil**, hiçbir checkpoint seçmez) |
| tasarım (ölçümden önce ilan) | `data/eval/t0068_belge_ayristirma_design_2026-09-18.md` · sha256 `587202f1871889eb…` · ölçüm betiği tarafından **doğrulandı** |
| ölçüm | `data/eval/t0068_belge_ayristirma_2026-09-18.json` · `data/eval/t0068_ayristirma_2026-09-18.json` |
| damga | 2026-09-18T10:34:17Z · cihaz **mps** (mps) |
| koşu | 31.2 sn · 900 ileri geçiş |

## 1. Soru

T-0067 eksen C, genel alanda (`test_natural_150`, belge-koşullu RAG istemleri) ceketin ebeveyne göre
cevap-bölgesi CE'sini **bozduğunu** ölçtü: fark(f3−f4) = **-0.6254858080546061**.
Mekanizma açıklanmamıştı. Aynı 150 satırdan **üç koşul** türetildi (yalnız `input` içindeki `<BELGE>`
bloğu değişir; `instruction`/`output` sabit ⇒ **cevap bölgesi jetonları özdeş**):

| koşul | input | ort. jeton |
|---|---|---|
| K1 `belge_var` | `<BELGE> {belge} </BELGE> {soru}` | 166.7 |
| K2 `belge_yok` | `{soru}` (blok çıkarıldı) | 92.8 |
| K3 `bos_belge` | `<BELGE> </BELGE> {soru}` (etiketler kalır, metin silinir) | 94.8 |

Yön kuralı: **fark = CE(f3_clean) − CE(f4_r05)** · POZİTİF = ceket daha iyi · NEGATİF = ceket bozmuş.

## 2. Sonuçlar

| koşul | CE f3 | CE f4 | **fark** | eşleştirilmiş GA | iyi/kötü/sıfır |
|---|---|---|---|---|---|
| `belge_var` | 5.2097 | 5.8352 | **-0.6255** | [-0.6679; -0.5828] | 0/150/0 |
| `belge_yok` | 5.4435 | 5.9727 | **-0.5291** | [-0.5719; -0.4874] | 0/150/0 |
| `bos_belge` | 5.4499 | 5.9725 | **-0.5226** | [-0.5664; -0.4793] | 0/150/0 |

### Araç pozitif kontrolü (A kuralı) — **GEÇTİ**

K1, T-0067 eksen C'yi **bit düzeyinde** yeniden üretti: ortalama sapma **3.33e-16**
(tolerans 0.005), **kayıt bazında en büyük f4 sapması 0.0**
(tolerans 0.01). Yani araç bilinen etkiyi üretiyor; üç koşulun okumaları geçerlidir.

### Tavan / ayırt edicilik

* Sıkışma **yok**: f4 CE'si düzgün dağılım sınırının `ln(32852) = 10.3998`
  yalnız **%57.4** düzeyinde (belge çıkınca bile cevaplar
  bilinemez hâle gelmiyor, farklar okunabilir kalıyor).
* Ayrık çift sayısı her koşulda **150** (iyi/kötü/sıfır = 0/150/0)
  ⇒ tavan artefaktı yok; araç ayırt ediyor.
* Yük kontrolü (T-0052): son/ilk pencere süre oranı 0.37–0.54× (<1 = koşum hızlandı, GPU'ya dış yük binmedi).

## 3. Ayrıştırma (post-hoc, betimleyici — `data/eval/t0068_ayristirma_2026-09-18.json`)

| pay | tanım | **değer** | GA | neg/poz/sıfır |
|---|---|---|---|---|
| `S_belge_tam_(K1-K2)` | Delta(belge_var) - Delta(belge_yok) | **-0.0963** | [-0.1253; -0.0688] | 116/34/0 |
| `S_etiket_(K3-K2)` | Delta(bos_belge) - Delta(belge_yok) | **+0.0065** | [+0.0004; +0.0127] | 64/86/0 |
| `S_icerik_(K1-K3)` | Delta(belge_var) - Delta(bos_belge) | **-0.1029** | [-0.1303; -0.0764] | 125/25/0 |

1. S_etiket = +0.0065 (|K1|'in %1.0'i) ve ISARETI POZITIF — yani bos <BELGE> etiketi, etiketsiz duruma gore ceketi biraz daha IYI yapiyor. GA sifiri icermese de (>0) BUYUKLUK ONEMSIZ ve yon 'sok' beklentisinin TERSI: B1'in MEKANIZMA iddiasi (zarf/syntax soku) CURUDU. NOT: GA-sifir-icerme testi buyukluge KORDUR; tek basina 'olculebilir katki' DEMEZ.
2. Belgesiz kosulda kaybin |K2|/|K1| = 0.8460 (84.6%) suruyor; her iki kosulda da 150/150 kayitta ceket kotu.
3. Belgenin TAM katkisi S_belge = -0.0963 (15.4% of |K1|), icerigin etiket-ustu katkisi S_icerik = -0.1029.
4. B1 ve B3 MANTIKSAL OLARAK CELISMIYOR: ikisi de 'girdi icerigi belirleyici degil' okumasidir. B1 YANLIS KURULMUSTUR (ayirt edici degil: girdiden bagimsiz bozulmada da atesler); B3'un iddiasi ise ayirt edici ve DOGRULANDI. Net hüküm: BELGEDEN BAGIMSIZ.

**Okuma:** belgenin *tamamının* katkısı |K1|'in yalnız **%15.4** kadarı;
kalan **%84.6** belge olmadan da var. Boş etiketin katkısı **%1.0**
ve **işareti pozitif** ⇒ zarf/syntax şoku **yok**.

## 4. İlan edilen hüküm ve düzeltilmesi

Tasarımın karar kuralları **B1 ve B3'ü aynı anda** ateşledi; tasarım bu durumu öngörmüştü ve
hüküm **`CELISKILI (B1_ZARF_SOKU+B3_BELGEDEN_BAGIMSIZ) — kurallar ayni anda atesledi, rapor arastirmali`** oldu. Araştırma (bkz. §3) şunu gösterdi:

* **B1 (zarf-şoku) YANLIŞ KURULMUŞ bir kuraldır — ayırt edici değil.** Eşiği `|K3| ≥ 0,75·|K1|`'dir;
  bu eşik "boş etiket kaybı üretiyor" durumunda da, "kayıp girdiden bağımsız" durumunda da ateşler.
  Ayırt edici istatistik `S_etiket = K3 − K2`'dir ve o **+0.0065**'tir.
* **B3 (belgeden bağımsız) ayırt edici ve DOĞRULANDI:** `|K2|/|K1| = 0.8460`,
  `K2 = -0.5291 < −0,30`, üç koşulda da 150/150 öğede bozulma.

**Net hüküm: `BELGEDEN BAĞIMSIZ`.** İki kural mantıksal olarak çelişmiyordu; ikisi de "girdi içeriği
belirleyici değil" okumasıdır — biri (B1) bunu yanlış bir eşikle söylüyordu.

## 5. NE ÇÜRÜDÜ

1. **Danışmanın "%95 boş `<INPUT>`" premisi** — ceketin gerçek eğitim akışında boş `<INPUT> </INPUT>`
   **3300/12763 = %25,86** (`scratch/t0068_bin_scan.py`, §EK: en yüksek oran hiçbir dosyada %26'yı geçmiyor).
2. **`<BELGE>` zarf hipotezi (danışmanın Q1 mekanizması + T-0067'deki bizim çıkarımımız)** — taranan 17
   eğitim `.bin`'inin 14'ünde `<BELGE>` x0 (`f3_clean`'in kendi verisi dâhil) ⇒ iki kol *eşit* yoksul,
   ve ölçüm şoku göstermedi (`S_etiket` = %1.0, ters işaretli).
3. **★ T-0067'nin kendi daraltma hükmü** — raporda *"C1 **belge-koşulludur** ⇒ 'genel dil bozuldu'
   diye genişletilemez"* yazıyordu. Ölçüm bunu **çürüttü**: belge tamamen çıkarıldığında kayıp
   **%84.6 oranında sürüyor**. Daraltma geçersizdir; T-0067 raporu ve
   `.agent-bus/notes/T-0067.md` bu ek ile birlikte okunmalıdır.

## 6. Bu ölçümün İDDİA ETMEDİĞİ şeyler

1. **"Genel dil bozuldu" DEĞİL.** Ölçülen şey *bu* istem ailesinde (genel alan RAG soruları),
   belge olsun olmasın, sistematik bir CE artışıdır. Düz metin (RAG'sız) genel dil CE'si **ölçülmedi**;
   "genel dil modellemesi bozuldu" hükmü için **ayrı bir ölçüm** gerekir.
2. **Açık gerilim (çelişki değil):** A′ unutma ekseni (T-0062) ceketin genel alan unutmasını
   **eşik içinde** bulmuştu (+%7,98 ≤ +%10, maskeli CE, tutulmuş eğitim dilimi). Bu ölçüm aynı ceket için
   genel alanda **-0.5291…-0.6255 CE artışı** buluyor. Eksenler farklıdır — **kapsam** (tutulmuş
   eğitim dilimi vs genel alan RAG kümesi), **tanım** (unutma oranı vs mutlak CE farkı), **yöntem**
   (maskeli CE vs cevap-bölgesi CE), **yön** (oran vs fark) — bu yüzden **iki sayı çelişmiyor**;
   hangi eksenin karar için bağlayıcı olduğu **açık bir sorudur** ([[iki-sayi-celisiyor-sanma-once-kume]]).
3. **Mekanizma değil, ayrıştırma:** "belgeden bağımsız" bir *bileşen atfıdır*; hangi katmanın
   (dikkat, gömme, çıkış) bozulduğunu söylemez.
4. **Kapı değil, tek korpus:** hiçbir checkpoint seçmez; sonuç ceketin kendi alanına (marangozluk)
   taşınamaz.

## 7. Kusurlar (ölçümden sonra bulunanlar — tasarım dosyası DEĞİŞTİRİLMEDİ)

1. **İlan edilen B1 kuralı ayırt edici değildi** (bkz. §4). Tasarımın kural kümesi karşılıklı dışlayıcı
   değildi; `ÇELİŞKİLİ` hükmü verinin değil **kural kümesinin** kusurudur. Düzeltilmiş okuma rapora
   yazıldı; tasarım sha256'sı bütünlük gereği korundu.
2. **Yardımcı betiğin ilk yorumu büyüklüğe kördü:** `S_etiket` için yalnız "GA sıfırı içeriyor mu"
   testi kullanılıyordu ve GA `[+0,0004; +0,0127]` sıfırı içermediği için *"etiket ölçülebilir katkı
   yapıyor"* yazdı — oysa büyüklük |K1|'in %1,0'i ve işaret ters. GA-sıfır-içerme testi **tek başına
   büyüklük hakkında bir şey söylemez**; betik ölçek-farkında okumayla düzeltildi ve yeniden koşuldu
   (ek betik post-hoc analizdir, ilan edilmiş eşik taşımaz). Düzeltme raporda kayıtlıdır.
3. **Tasarım üretimi sırasında yakalandı (ölçümden ÖNCE, bu yüzden tasarımda düzeltildi):** ilk taslak
   §1'deki iki sayıyı (`%25,9` boş-input ve `<BELGE>` taraması) **elle** taşıyordu; `scratch/t0068_bin_scan.py`
   yazılıp sayılar betikten hesaplatıldı ([[tasarim-sayisi-betikle-hesaplanmali]]).

## 8. Artefaktlar (sha256 TAM digest olarak betikle hesaplandı — raporun kendisi hariç)

| yol | bayt | sha256 |
|---|---|---|
| `data/eval/t0068_belge_ayristirma_design_2026-09-18.md` | 10387 | `587202f1871889ebfaad3cc4f3ae9726e68e9ddc98e4871b49355e5bf218a1c8` |
| `scratch/t0068_conditions.py` | 3147 | `e99d63b8ab5ffa8a4b3b7a5b0c72fbf3a2858b6ad82938e66643f394ae26603c` |
| `scratch/t0068_bin_scan.py` | 2954 | `b8da3fb6ddcbec3700808a39ed1c25dbd8d7f898320abf7c0c10d9701eab4a1b` |
| `scratch/t0068_write_design.py` | 13995 | `f5f35a4c2faf999f6f94e5cd2f834fd43d87d75a1da52b9ca7c62aa057fd4529` |
| `data/eval/t0068_belge_ayristirma_2026-09-18.json` | 63417 | `a81c028d8e095526c97b5c81653aba2f1c7ac4af4578ff16341f47060f9e5b9b` |
| `scratch/t0068_measure.py` | 12206 | `c2dc1adcfc4eba8206e7b340facf8d1e111162d7297368cf5b27897922fd1200` |
| `data/eval/t0068_ayristirma_2026-09-18.json` | 3172 | `502285c94801c7e4eb36385655836b7187d868d31a2999061e35a53c2df289ce` |
| `scratch/t0068_ayristirma.py` | 6016 | `7593f50d7fd0c645134af930f5ce0876d862f34f4a3fd14c1443b309d7db0b96` |
| `scratch/t0068_run.log` | 699 | `657766a22f86889ddf55ce0e068280d0be9047b2dd937c33975eaf57882735e3` |
| `scratch/t0068_run.rc0` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `scratch/t0068_make_report.py` | 12156 | `d5a420f946f539c93d6851d8e816e38172683a6bd9da6d8e7c6376b1b0241c78` |

*Üretici betikler de listelenmiştir* ([[kismi-sonuc-dosyasi-bitis-kanitlamaz]]).
Digest'ler **tam 64 karakter** yazılır: önek karşılaştırması zayıf denetimdir
([[hash-iddialari-tam-digest-ile-denetlenir]] — 9/9 önek doğruyken 0/9 kuyruk uydurma çıkmıştı).
Tasarım dosyası ölçümden sonra **değiştirilmedi** (sha256 hâlâ `587202f1871889eb…`).

**Bus'a ait durum dosyaları (`.agent-bus/state/tasks/*.json`) bu tabloya KONMAZ:** `bus_report_result`
kapanışta onları **kendisi** günceller, dolayısıyla donmuş bir tabloya yazılan digest'i kapanış çağrısı
bayatlatır. (Bu kusur T-0068'de iki kez gerçekleşti — raporun ve notun ilk sürümü görev kaydını
hash'lemişti; kapanış denetimi "beyandan sonra bozulan artefakt: `tasks/T-0068.json`" verdi. Beklenen
bir değişimdir ama okuyucu için **yanlış bir ihlal gibi görünür**; ikisinden de çıkarıldı.)

## EK — `.bin` zarf taraması

Tam tablo tasarım dosyasının EK'indedir (`data/eval/t0068_belge_ayristirma_design_2026-09-18.md`); üretici `scratch/t0068_bin_scan.py`,
hiçbir dosya atlanmadı (parça parça okuma). Özet: `<BELGE>` yalnız 3 dosyada
(`train_deep_sft.bin` 1545 · `train_chat_balanced.bin` 614 · `train_f4_replay_mix.bin` 63);
ceketin kendi iki kaynağında (`train_carpenter_specialization.bin`, `train_chat_balanced_clean.bin`) **x0**.
