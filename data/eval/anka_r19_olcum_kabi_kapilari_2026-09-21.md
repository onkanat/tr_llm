# Anka r19 — Ölçüm kabı ve koşum işletmesi kusurlarını KALICI KAPIYA çevirme (T-0098)

**Görev:** T-0098 · **damga_utc:** 2026-09-21 · **operatör talebi:** *"Bu onyedi kusur için plan yap"*
**Öncül:** T-0097 (kapandı) kendi raporuna **17 kusur** yazdı, hiçbirini düzeltmedi — T-0067 gereği
ölçüm sonrası kusur tasarıma göre onarılmaz, **rapor**a yazılır. Bu görev o kaydı **yürürlüğe koyar.**

---

## §1 — Neden düzeltme değil, KAPI

T-0097/K15'in kendi dersi bunu zorunlu kılıyor:

> *"Doğru kuralı bu notun kendisi S1'de zaten yazmıştı (`^Adım\s+\d+/`) ve S6'da çiğnendi ⇒
> kuralın YAZILI olması uygulanmasını sağlamıyor, **KAPIYA BAĞLANMASI** gerekiyor."*

Doğru kural zaten yazılıydı ve yine ihlal edildi. Yazılı kural kendini dayatmaz; **düşen kapı** yürütür.
Bu görev 17 kusurun **12'sini** kalıcı kapıya çevirdi. Kapılar **`tests/` + `scripts/`** (operatör kararı 1):
depoda yerleşik `scratch/` konvansiyonu **değil** — çünkü "kalıcı" denen T-0086 aracı
`scratch/anka_r9_olu_atif_denetimi.py`'de duruyor ve `.gitignore:18 = scratch/` ⇒ **sürümlenmiyor**.

> **Bu görev bir beyanı TERSİNE ÇEVİRİR:** T-0086/T-0087 "KOD DEĞİŞMEDİ: `src/**`, `scripts/**`,
> `tests/**`" demişti. T-0098 `scripts/` ve `tests/` yüzeyini **değiştirir** (yalnız YENİ dosya ekler).
> Sessizce değil, burada açıkça.

## §2 — Kurulan kapılar (ölçülen)

| # | kapı | dosya | kusurlar | sınama |
|---|---|---|---|---|
| **G1** | `say_adim_satirlari` — İKİ bağımsız sayaç (çapalı + alan-ayrıştırma), uyuşmazsa **hata**; naif sayaçlar kanıt olarak **reddedilir** | `scripts/olcum_kabi.py` | K6 + K15 | `test_sayac_iki_yonlu`, `test_sayac_fazla_saymaz` |
| **G2** | `tuple_alan_denetimi` — indeks→ad haritası bir kez basılır | aynı | K16 | `test_tuple_alan_kaymasi` |
| **G3** | `yon_adli` + `karsilastir` — yön, eksenden bağımsız **ADIYLA** | aynı | K7 | `test_yon_eksenden_okunur`, `test_kopya_donmus_kaynagiyla_ayni` |
| **G4** | `sayi_kaynagi` — her sayı kaynağını taşır | aynı | K14 | `test_sayi_kaynagi_zorunlu` |
| **G5** | `yokluk_beyani` — eksik anahtar `ÖLÇÜLEMEDİ`, **asla "değişti" değil** | aynı | K10 | `test_yokluk_degisti_degildir` |
| **G6** | `referans_dogrula` — kaynak satır no. **canlı dosyadan** | aynı | K11 | `test_esik_referanslari` |
| **H1** | `tek_egitici_onkontrol` — `lsof -t`, >1 tutucu ⇒ DUR | `scripts/kosum_kapisi.py` | K5 | `test_tek_egitici_kapisi` |
| **H2** | `kosum_canli_mi` — koşum canlıyken kapanış/kabul aracı DURUR | aynı | K17 | `test_kosum_canli_kapisi` |
| **H3** | `timeout_denetimi` — her `subprocess` çağrısı `timeout=` taşır | aynı | K4 | `test_timeout_kapisi` |

**K12'nin dersi teste gömüldü:** G3'ün kural sözlüğü ve eksen eşlemesi **ilan §5'ten ayrıştırılarak**
okunur (`_ilan_kurallari()`), elle yazılmaz; iki yönlü hizalanır (kodda ilanın attest etmediği kural
olmamalı **ve** ilanın ilan ettiği her kural kodda bulunmalı). T-0097/K12'de yanlış beklenti **doğru
kodu suçlu ilan etmişti**.

## §3 — On yedinin sınıflandırması ve bu turdaki durumu

| id | kusur (kısa) | sınıf | bu turda |
|---|---|---|---|
| K4 | `subprocess.run`'da `timeout=` yok | ardıl sürücü | **KAPI (H3)** |
| K5 | iki eğitici MPS'te kilitlendi | koşum işletmesi | **KAPI (H1)** |
| K6 | sayaç FAZLA sayar (alt dizgi) | ölçüm kabı | **KAPI (G1)** |
| K7 | eksen yönü ters okundu | ölçüm kabı | **KAPI (G3)** |
| K9 | `.pyc` beyan boşluğu | beyan | **KAPATILDI** (`writes[]`'e `tests/__pycache__/` beyan edildi) |
| K10 | "yokluk" ≠ "değişti" | ölçüm kabı | **KAPI (G5)** |
| K11 | eşik referans satırları ezber | ölçüm kabı | **KAPI (G6)** |
| K12 | pozitif kontrol beklentisi yanlış | ölçüm kabı | **KAPATILDI** (beklenti ilandan ayrıştırılıyor) |
| K14 | T-0096 serisi ölçülmeden yazıldı | ölçüm kabı | **KAPI (G4)** |
| K15 | sayaç KÖR (sağa yaslı sayı) | ölçüm kabı | **KAPI (G1)** |
| K16 | tuple sütun kayması | ölçüm kabı | **KAPI (G2)** |
| K17 | kapanış aracı koşum SIRASINDA koştu | koşum işletmesi | **KAPI (H2)** |
| K1 | ilan "her olgu 12 satır" yanlış | ilan (DONMUŞ) | ertelendi (sonraki tur) |
| K2 | ilan muhasebe formülü bayat (+9.283) | ilan (DONMUŞ) | ertelendi |
| K3 | ilan `--steps` 4.596 ↔ tablo 24.000 | ilan (DONMUŞ) | ertelendi |
| K8 | digest tarifi kayıtlı değil | ön-kayıt çıpası | ertelendi |
| K13 | sıfır genişlikte çürütücü eşiği | ilan (DONMUŞ) | ertelendi |

**12/17 kapıya bağlandı · 5/17 ertelendi.** Ertelenenler **donmuş ön-kayıt** sınıfıdır: ölçüm sonrası
onarımları T-0067'ye aykırı olurdu (bkz. §9).

## §4 — Ölçülen kanıtlar (elle yazılmadı)

| kanıt | ölçüm |
|---|---|
| **G1 gerçek log kalibrasyonu** (`scratch/t0097_kos/seg_2.log`) | doğru **201** (adım 1→2000) · kör sayaç **101** · şişen sayaç **403** ⇒ **ikisi de reddedildi** |
| **H3 gerçek sürücü** (`scratch/t0097_marangoz.py`) | eksik `timeout=` siteleri = **[179, 228]** — K4 yalnız `:179`'u kaydetmişti ⇒ kapı, kayıtlı olandan **FAZLASINI** görüyor |
| **Mutasyon sınaması (vakum kontrolü)** | 4/4 mutasyon **yakalandı**: K15 kör sayaç · K11 referans yutması · K4 timeout yutması · K7 işaret ters ⇒ testler **vakum değil**; iki kapı dosyası bit-ödeş geri alındı |
| **H1 iki dal** | tek tutucu ⇒ geçti (PID 1) · iki tutucu ⇒ **DURDU** (PID 2, gerçek alt süreçle) |
| **H2 iki dal** | tutucu varken canlı ⇒ **rc=2** · tutucu yokken ⇒ rc=0 (sonuç.json **taze olmasa bile** durur) |
| **Fail-closed** | `lsof` yokken **GEÇMEZ**, `OrtamHatasi` verir |
| **Korunan yüzey** | **15/15 artefakt bit-ödeş** (önce `10:25:20Z` → sonra) |
| **pytest tabanı** | **1 failed, 248 passed** (82,39 sn) · düşen tek test `test_agent_gateway` sandbox soket yasağı — **değişmedi** ⇒ regresyon yok (T-0097: 1 failed / 237 passed) |
| **Kapı betikleri** | `olcum_kabi.py` **26 PASS · 0 HATA rc=0** · `kosum_kapisi.py` **9 PASS · 0 HATA rc=0** |
| **K30 üç dal ölçümü** | `scripts/*.pyc` **kim yazdı?** DAL 1 `pytest` (guard'lı `_yukle`) ⇒ **YAZMADI** · DAL 3 guard'lı ad-hoc import ⇒ **YAZMADI** · DAL 2 **guard'sız** ad-hoc import (**pozitif kontrol**) ⇒ **YAZDI** ⇒ `.pyc`'ler **benim geçici harness'ımdan**; **K23'ün guard'ı çalışıyor** (iddia çürümedi, doğrulandı) |
| **`scripts/__pycache__/` zemini** | **20 dosya**, en yeni damga `03:42:47Z` — yani **20'si de benim görevimden ÖNCE**; bu bir **depo genelinde beyan boşluğu**dur (her import edilen betik bir `.pyc` bırakmış, hiçbiri beyan edilmemiş), benim ürettiğim bir sınıf değil |

## §5 — Bu turda ÖLÇÜLEN YENİ kusurlar (kendi kapımın kusurları)

Kapıyı kurarken **kapının kendisi** kusurlu çıktı; her biri ölçümle yakalandı ve düzeltildi.
Bu, kapatılan kusurların **kardeşleri**dir — ve kapının gerçek olduğunun kanıtıdır.

| id | kusur | nasıl yakalandı | durum |
|---|---|---|---|
| **K18** | `timeout_denetimi` **string literal içindeki** örneği kod saydı ⇒ 2 sahte pozitif | H3 kendini denetim: "4 çağrının 2'si timeout'suz" — oysa fikstür metniydi | **DÜZELTİLDİ** (`_kod_metni` maskeleme) — K6'nın tam kardeşi |
| **K19** | `_kontrol` FAIL satırı basıyor ama `hatalar`a yazmıyordu ⇒ kapı **FAIL basıp rc=0** dönebilirdi | ilk koşumda "FAIL G5" satırı + `rc=0` **birlikte** göründü | **DÜZELTİLDİ** (fail-closed) |
| **K20** | pozitif kontrolün **beklentisi** yanlış: `A` ekseninde `yuksek`/`dusmez` pozitif farkta **ayırt edilemez** | test `AssertionError: A ekseni kanaryasi` | **DÜZELTİLDİ** (aynı kelimenin iki eksende zıt hüküm vermesi kullanıldı) |
| **K21** | `Popen`'dan `timeout=` istemek **yanlış pozitif**: `Popen.__init__`'te `timeout` parametresi YOK | `inspect.signature` ölçümü | **DÜZELTİLDİ** + kapsam sınırı ilan edildi |
| **K22** | **`.pytest_cache/`** `writes[]`'te yok; pytest onu günceller | `find .pytest_cache -newermt` | **AÇIK** — sonraki turda `denetim_beyan`a girmeli |
| **K23** | test, gitignored `scratch/t0097_rapor.py`'yi import ediyor ⇒ önbellek bayat olsaydı **`scratch/`'a `.pyc` yazardı** (K9'un tekrarı yeteneği) | ölçüldü: `.pyc` damgası `07:23:19Z` (T-0097'den), bu koşumda **yazılmadı** — ama yetenek duruyordu | **ÖNLENDİ** (`sys.dont_write_bytecode`) |
| **K24** | H2'nin canlı-dal kontrolünü tutucu **kapandıktan sonra** çalıştırdım ⇒ kontrol **VAKUM**'du (yeşil görünürdü) | koşum: "gecilir: tutucu yok" — canlı dal hiç sınanmadı | **DÜZELTİLDİ** (kontrol tutucu canlıyken) |
| **K30** | geçici harness'ım `scripts/__pycache__/` altına **beyan edilmemiş 2 `.pyc`** bıraktı (`olcum_kabi`, `kosum_kapisi`) | kapanışın **iki yönlü** beyan denetimi: `writes[]`te `tests/__pycache__/` var, `scripts/__pycache__/` **yok** | **DÜZELTİLDİ** (kaldırıldı, yüzey geri alındı) — §4'te üç dalla ölçüldü |

> **K20'nin ağırlığı:** kendi pozitif kontrolümün beklentisini **üçüncü kez** yanlış yazdım
> (T-0097/K12, sonra H2, sonra G3-A). Her seferinde hatayı **ölçüm** yakaladı — bu, "beklentiyi
> sezgiden yazma" kuralının **yazılı olmasının yetmediğinin** ikinci kanıtıdır; bu yüzden G3'ün
> beklentileri artık **ilan §5'ten ayrıştırılıyor**.

## §6 — T-0097 araçlarında ölçülen kusurlar (BEYAN — düzeltilmedi)

Bu beş kalem `scratch/t0097_*` içindedir; hepsi **T-0097'nin tescilli kaydıdır** (digest'leri
`kapanis.json` gate 4'te ve notlarda kayıtlı) ⇒ **düzenlenmeleri kapalı raporu bayatlatır**.
Bu yüzden **düzeltilmedi, beyan edildi** — ve düzeltmenin kendisi kapıya taşındı.

| # | bulgu | ölçüm |
|---|---|---|
| B1 | `BIRLESIK_ARANAN` (`kapanis.py:42`) **ÖLÜ KOD** | `grep -c` = **1** (yalnız atama; hiç referans yok) |
| B2 | `denetim_donmus` **kısmi anahtar deliği** | `korunan_sonra`'da **15 anahtarın 1'i** varken araç **`durum: TAM (once+sonra+canli)`**, `dokunulmadi: True`, `degisenler: []` ⇒ eksik 14 anahtar için asla "değişti" denemez |
| B3 | Banner paydası **sabit `/7`** (`kapanis.py:339`) | pay türetiliyor (`sum(x[0].isdigit())`), payda literal ⇒ kapı eklenirse banner **yanlış** basar |
| B4 | "İkinci `timeout=` sitesi" (`marangoz.py:228`) | **H3 ile ölçüldü ve kapıya bağlandı** (bkz. §4) |
| B5 | `.pytest_cache/` beyan sınıflandırmasında **YOK** | `grep pytest_cache kapanis.py` = **0**; sınıflandırma yalnız `"__pycache__/"` alt dizgisine bakıyor (`:193-196`) ⇒ K22/B5 aynı sınıf |

> **GERİ ÇEKME (kendi plan iddiam ölçümle çürüdü).** Plan, `kapanis.py:108`'deki
> `cipa.get("birlesik_kural_sha256")` için **"fail-open"** demiştim. **ÖLÇÜM ÇÜRÜTTÜ:** araç o kalemi
> `"ozet"` sözlüğüne **hiç koymuyor**; `KAPI_DISI_BEYANLAR` altında
> `"DOGRULANAMADI (recete kayitli degil) — KAPI DEGIL, beyan"` diye **açıkça** dışlıyor ve
> `gecen = all(ozet.values())` yalnız 7 gerçek kapıyı sayıyor. ⇒ **Fail-open DEĞİL, dürüst beyan.**
> Bu iddia bu raporla **kayıttan düşürülmüştür**; plan metni düzeltilmedi, sapma burada yazıldı.

## §7 — İlan edilen kapsam sınırları ve sapmalar

1. **H2 ölçütü bilerek KATI yapıldı (plandan SAPMA).** Plan *"tutucu **VE** `sonuc.json` taze"* diyordu;
   bu bir **AND**'dir ve `sonuc.json` silindiğinde kapıyı **fail-open** yapardı. T-0075'in dersi:
   *okunamayan sinyalde durmayan koruma kapı değildir.* ⇒ Durma kararı **tutucu ≥ 1**'e bağlandı;
   `sonuc_taze` yalnız **destekleyici kanıt** olarak raporlanıyor. `test_kosum_canli_kapisi`
   `sonuc.json` **yokken de** durduğunu sınıyor.
2. **`Popen` kapsam DIŞI (ilan edilen sınır, gizli delik değil).** `Popen.__init__`'te `timeout`
   parametresi **yoktur** (`inspect.signature` ile ölçüldü) ⇒ ondan `timeout=` istemek **yanlış
   pozitif üretir**. `Popen` için daha zayıf bir ölçüt uygulanır (`.wait(timeout=)` bekçisi); bekçisi
   bulunamayan `Popen` **`sinirlama`** listesinde **adıyla** raporlanır — gizlenmez, ama `rc`'yi
   düşürmez, çünkü orada `timeout`'un varlığı bu kaptan ölçülemez.
3. **Plan öncülü ölçümle yanlışlandı.** Plan, B1–B5 için *"bunlar henüz dondurulmamış araçlarda"*
   deyip **"düzeltilir"** diyordu. **Ölçüm:** beşi de `scratch/t0097_*` içinde, yani **T-0097'nin
   tescilli kaydında**. ⇒ "düzeltilir" dalı **uygulanamaz**; governance'ın `DOKUNULMAZ: scratch/t0097_*`
   maddesi geçerlidir. Kalemler **beyan edildi**, düzeltme **kapıya taşındı** (B4 örneği: H3).

## §8 — Kusur defteri: normalize şema (bu görevin çıkış noktası)

**Ölçülen sorun:** defter **raporlar arası birleştirilemiyor**. `data/eval/*.json` taraması:
yalnız **r18** kararlı `K1…K17` kimlikleri kullanıyor; **r10/r15** `id: None`; **r13** `1,2,3,4`.
⇒ *Tekrarı önlemenin ön koşulu, tekrarı ölçmektir* — ve bugün tekrar **ölçülemiyor**.

Bu raporda defter şu şemayla yazılır (`kusurlar[]`):

```json
{"id": "K15", "yer": "S6 saglik sayaci", "metin": "...", "kanit": "olcum",
 "sinif": "olcum_kabi", "durum": "KAPI", "kapi": "G1", "tekrar": 1}
```

`sinif ∈ {ilan, olcum_kabi, kosum_isletmesi, beyan, surucu}` · `durum ∈ {KAPI, KAPATILDI, ERTELENDI, ACIK}`
⇒ sonraki raporlar **aynı kimlikleri** kullanırsa tekrar sayısı **ölçülebilir** hale gelir.

## §9 — Açıkça YAPILMAYANLAR

* **İlan kusurları (K1/K2/K3/K8/K13) kapatılmadı** — donmuş ön-kayıt belgesinin ölçüm sonrası onarımı
  T-0067'ye aykırıdır. Çözüm **ileriye dönük** olmalıdır: ilan **dondurulmadan önce** iddialarını
  betikle hesaplayan bir **ilan kapısı**; K13 için **asgari etki büyüklüğü ilanı**; K8 için
  **tarifiyle yazılan digest**.
* **Kapılar T-0097'nin 17 kusurunu geriye dönük ONARMAZ** — yalnız **tekrarını** önler.
* **`scratch/` altındaki kapı araçları taşınmadı** (T-0086 aracı yerinde) — taşıma ayrı karardır.
* **Gerçek bir koşum başlatılmadı** ⇒ "canlı koşumda dur" kapısı **sahte fikstürle** sınandı.
* **`CLAUDE.md` DEĞİŞTİRİLMEDİ.** Sonucu beyan edilir: damgalı taban (`213 passed`) **zaten bayattı**
  (T-0097 ölçümü: 237); bu görev sayıyı **248**'e taşıdı ⇒ damga **daha da bayatladı**. Çıpa belgesini
  güncellemek **operatörün** kararıdır.

## §10 — Korunan yüzeyin tam digestleri (önce = sonra)

| artefakt | sha256 (tam) | hüküm |
|---|---|---|
| `data/eval/anka_r18_ceket_ilani_2026-09-21.md` | `1efacbb64f2d4a26279cbfc4b17921b37f02846b284abcdf8a7a68c0a731ad70` | AYNI |
| `data/eval/anka_r18_ceket_2026-09-21.md` | `17fe88adf8c44b208ff80e08a4931cc0a96e4501cbe6bb553a4172f436dc9124` | AYNI |
| `data/eval/anka_r18_ceket_2026-09-21.json` | `17f5d9861bdd53ad1f26fb65da090974d71ed056e31f55391354b5fc5de5d064` | AYNI |
| `data/eval/anka_r18_build_2026-09-21.md` | `84c5f4dbe65894a68ee5e8414cdf0aeccf04840aec2d7e952bb750117ac209ad` | AYNI |
| `data/eval/anka_r18_build_2026-09-21.json` | `780e7b50d94ef5ab0735bc2c7a9c82e6ec274400d0bb2e52ef7ffecd51900641` | AYNI |
| `scratch/t0097_marangoz.py` | `63908889bf77dd1628a35aa4862886ac10902a196d5d6b75ec00786b4496f7c2` | AYNI |
| `scratch/t0097_rapor.py` | `94d0605865bfb696266437b7b44c34bd64e2b531eb72b02aebd878ba7a1d2afb` | AYNI |
| `scratch/t0097_kapanis.py` | `60d4bcc2a1c8b77c6cb9591e16aa746469e7bf64d3ef6ff6a74f125507726ff2` | AYNI |
| `scratch/t0097_kos/sonuc.json` | `6e55c266b8af8776994a362bb3fd0dee4735d3540eef1f0e0666d4d2e38cf1ae` | AYNI |
| `scratch/t0097_kos/kapanis.json` | `35c43e70c0d497a54c387fe1a6388cacb339bb224343a383801276d181173e5c` | AYNI |
| `scripts/evaluate_carpenter_anka.py` | `0f017773bb015ef79f7fac02f51b2a14ed9d27a213fef72574ce16357c37188b` | AYNI |
| `src/llm/frozen_guard.py` | `75b3bf6e4384bc808230fb6b8f06c3187c93d78a45d742a4fb75f85f9ac3097b` | AYNI |
| `data/anka_a1.pt` | `f32d492d9361c668bc86e4e4b0f99f5c473af552486837452475f8b0aa3eb5b4` | AYNI |
| `data/anka_a1r.pt` | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | AYNI |
| `data/train_carpenter_specialization_anka_r18.bin` | `c9d964d98777358d6acb3db2c04d18a4e9b5bf515f1d8125077ce097857dfb78` | AYNI |

**15/15 AYNI · 0 DEĞİŞEN · 0 EKSİK** ⇒ T-0098 yalnız **YENİ** dosya ekledi.

### Üretilen yeni yüzey

| dosya | satır | sha256 |
|---|---|---|
| `scripts/olcum_kabi.py` | 458 | `be5ca2c12f0b88e39e2d5f1d39a213908218c71d625d9f0ba414f3b182aa2624` |
| `scripts/kosum_kapisi.py` | 386 | `91179be67c056175331801392c78c9f35f8b9e4c0be6b5d5ba3940968c3e70be` |
| `tests/test_olcum_kabi_kapilari.py` | 422 | `5a78d1a046192336bc37219b834d7aa53e37b92556de5993efcd096ca2bd0ef3` |

**Yan ürün beyanı (K9 sınıfı):** `tests/__pycache__/test_olcum_kabi_kapilari.cpython-314-pytest-9.0.3.pyc`
(beyan edilmiş) · `.pytest_cache/v/cache/{nodeids,lastfailed}` (**beyan EDİLMEMİŞ** ⇒ K22 açık).
`scratch/` altında **hiçbir şey değişmedi**: `scratch/__pycache__/t0097_rapor.cpython-314.pyc`
damgası `07:23:19Z` (T-0097 oturumu) — bu koşumda yeniden yazılmadı.

**K30 kapanışı — `scripts/__pycache__/`:** kapanışın iki yönlü beyan denetimi, benim geçici
harness'ımın bıraktığı **2 beyan edilmemiş `.pyc`**'i yakaladı; **kaldırıldılar** ve yüzey
**geri alındı** (20 dosya, en yeni `03:42:47Z` < `claimed_at 10:24:39Z`). Kimin yazdığı
**tahmin edilmedi, üç dalla ölçüldü** (§4): `pytest` yolu **yazmıyor** ⇒ K23'ün guard'ı
doğrulandı; pozitif kontrol (guard'sız import) **yazıyor** ⇒ mekanizma gerçek. Geriye kalan
**yapısal** boşluk beyan edilir: `scripts/__pycache__/` **20 dosya** taşıyor ve **hiçbiri**
hiçbir görevin `writes[]`'inde yok — bu, K22/K29 ile **aynı sınıf**tır (beyan sınıflandırması
`.pyc` üreten **tüm** dizinleri kapsamalı, yalnız `"__pycache__/"` alt dizgisini değil).

> **Makine eşi.** Bu raporun ikizi
> `data/eval/anka_r19_olcum_kabi_kapilari_2026-09-21.json`'dur: aynı ölçümün **normalize kusur
> defteri** (§8 şeması, K1…K29) + kapı/test tabloları + sapmalar + **geri çekme kaydı**.
> **Karşılıklı digest çıpası bu iki dosyanın İÇİNE yazılmaz** — yazılsaydı döngü olurdu (her
> yazım diğerinin digest'ini değiştirir). İkisinin de **TAM sha256**'sı ayrı kanaldan,
> `.agent-bus/notes/T-0098.md`'de, **iki dosya da dondurulduktan sonra** hesaplanıp kayda geçer.
