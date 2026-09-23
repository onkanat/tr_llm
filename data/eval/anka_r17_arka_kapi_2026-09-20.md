# T-0095 — Anka + sonda kolları CHAT ARKA KAPISINDAN sınandı (ham metin çıktıları)

**Damga (UTC):** 2026-09-20T15:45:13.593505+00:00 · **Görev:** T-0095 · **Yürütücü:** claude (TEK)

**Operatör sorusu:** *“chat arayüzünün arka kapısı ile temel model Anka ve Modül model Marangozu test et text çıktıları kullan çıktıları görmek istiyorum. gerekirse yeni marangoz eğitelim.”*

## §0 — Ölçüm künyesi (hepsi artefakttan okundu)

| kalem | değer |
|---|---|
| kapı | `AgentGateway.create_http_server + urllib POST /api/query` |
| cihaz (istenen → gerçek) | `mps` → **`mps`** (mps_available=True) |
| sözlük | `data/rebuild/vocab_anka_r1_33114.json` (f9940a8d8e1f7cd9…) |
| model sayısı | 5 |
| soru sayısı | 6 (`M1 vernik`, `M2 kurt dişi`, `M3 şerit testere`, `M4 gönye`, `M5 zeytin ağacı`, `G1 genel`) |
| üretim uzunluğu | **45 jeton** — `src/rag/epistemic_agent.py:311` (`max_new_tokens=45`); arka kapının KENDİ davranışı, değiştirilmedi |
| HTTP | bütün sorgular **200** (beş model × 6 soru = 30) |

**Ölçüt çıpası (kör değil):** referansın kendine sözcük-F1'i **1.0000**, boş yanıt **0.0000** ⇒ tavan ve taban ayrışıyor. Aşağıdaki F1 bu ölçeğin içinde okunur.

## §1 — Arka kapı gerçekten soketten mi geçti?

Evet. Betik `ask()`'i doğrudan çağırmaz: `AgentGateway.create_http_server()` ile bir `HTTPServer` ayağa kaldırır, `urllib.request` ile `POST /api/query` atar; ölçülen değer **HTTP yanıt gövdesindeki** `response_text` alanıdır. Yol:

```
urllib POST /api/query  →  GatewayHTTPHandler.do_POST  →  AgentGateway.ask()
                        →  EpistemicCuriosityAgent.process_query()
                        →  generate_tokens(max_new_tokens=45)  →  decompiler.decompile_sentence
                        →  {"response_text": …}
```

## §2 — HAM METİN ÇIKTILARI (operatörün görmek istediği)

Aşağıdaki her hücre **beş modelin aynı soruya verdiği gerçek çıktıdır**, kırpılmadı. En altta külliyatın insan yazımı referansı var — kıyas için konuldu, model çıktısı değildir.

### M1 vernik

> **Soru:** Ahşap atölyesi güvenlik ve imalat rehberi: Ahşabın son kat yüzeyinde dolgu verniği nasıl tatbik edilmelidir?

> **Külliyat referansı (insan yazımı):** Dolgu verniği uygulamasında ahşabın açık gözeneklerini doldurarak son kat cila için cam gibi pürüzsüz ve homojen bir alt zemin hazırlayan astar verniktir.

| model | arka kapının çıktısı (`response_text`) | RAG | F1 |
|---|---|---|---|
| **TABAN** | , [?] ve [Özel İsim] gibi). bu nedenle, bir [?] olarak kullanılır. [Özel İsim]- i [Özel İsim]), [?] ( 1980-25). [Özel İsim] [?] ( 1980-20 | 0.275 | 0.0851 |
| **R0** | teknik çözüm: Tanımlanan özellikler doğrudan kestane ağacına aittir. | 0.275 | 0.0000 |
| **R4** | teknik çözüm: bu gereksinim için marangozlukta ve parçaya uygun açıyla yanaşılmalıdır. | 0.275 | 0.1212 |
| **R10** | teknik çözüm: bu gereksinim için marangozlukta kırlangıç tercih edilir. | 0.275 | 0.0645 |
| **R20** | teknik çözüm: Tanımlanan özellikler doğrudan kestane ağacına aittir. | 0.275 | 0.0000 |

### M2 kurt dişi

> **Soru:** Geleneksel ahşap zanaatı sorusu: Usta marangoz olarak kurt dişi birleştirme geçme tekniğinin püf noktalarını açıkla.

> **Külliyat referansı (insan yazımı):** Kurt dişi birleştirme uygulamasında açılı zikzak dişlerle ahşap parçaların boylamasına eklenerek sonsuz kereste elde edilmesini sağlayan endüstriyel eklemedir. Birleşme yanaklarının birbirine boşluksuz ve kasmadan oturması mukavemet için şarttır.

| model | arka kapının çıktısı (`response_text`) | RAG | F1 |
|---|---|---|---|
| **TABAN** | , [Özel İsim] [?].) ve [Özel İsim] ( 1980-25), [Özel İsim] ( 1980-25), [Özel İsim] ( 1980-2005), [Özel İsim] ( 19 | 0.237 | 0.0408 |
| **R0** | teknik çözüm: bu gereksinim için marangozlukta kırlangıcın yüzeyine yerleştirilmesidir. | 0.237 | 0.0541 |
| **R4** | teknik çözüm: Tanımlanan özellikler doğrudan ahşap parçaya uygun açıyla yanaşılmalıdır. | 0.237 | 0.0526 |
| **R10** | teknik çözüm: Tanımlanan özellikler doğrudan çam ağacına aittir. | 0.237 | 0.0000 |
| **R20** | teknik çözüm: usta Tavsiyesi: genel [?] bahçe mobilyasında ve ahşabına çok dayanıklıdır; dış [?] bahçe mobilyası ve tekne güvertelerinde kullanılır. | 0.237 | 0.0417 |

### M3 şerit testere

> **Soru:** İki kasnak arasında dönen sürekli esnek şerit bıçağıyla kavisli kesimler, biçme ve kalın keresteyi dilimleme makinesidir.

> **Külliyat referansı (insan yazımı):** Bu işlev için marangozlukta 'şerit testere' tercih edilir.

| model | arka kapının çıktısı (`response_text`) | RAG | F1 |
|---|---|---|---|
| **TABAN** | DERIV_lI [?], [Özel İsim] ve diğerlerinde bulunan bir bitkidir. [Özel İsim] [?], [?] ve [?] olarak da bilinir. [Özel İsim], 180 [?]; Cm boyundadır. [Özel İsim], 180 [?]; | 0.382 | 0.0000 |
| **R0** | bu gereksinim için marangozlukta kırlangıç tercih edilir. | 0.382 | 0.6667 |
| **R4** | bu gereksinim için marangozlukta kullanılır. | 0.382 | 0.4615 |
| **R10** | teknik çözüm: bu gereksinim için marangozlukta ve iç [?] bahçe mobilyasıdır. | 0.382 | 0.3158 |
| **R20** | bu gereksinim için marangozlukta kullanılır. | 0.382 | 0.4615 |

### M4 gönye

> **Soru:** Atölye şefinin değerlendirmesi: Ahşap atölyesinde gönye aletinin doğru kullanım tekniği nedir?

> **Külliyat referansı (insan yazımı):** Teknik Çözüm: Gönye ahşap parçaların 90 derece dik açılarını, düzlemselliklerini ve paralelliklerini kontrol etmeye yarayan l biçimli ölçüm aletidir. Aletin kesici ağzı daima keskin tutulmalı ve parçaya uygun açıyla yanaşılmalıdır.

| model | arka kapının çıktısı (`response_text`) | RAG | F1 |
|---|---|---|---|
| **TABAN** | , [?] ve [Özel İsim] gibi). bu nedenle ilgilidir ve [?], [Özel İsim] ( 1980). bu nedenle, [?] ( 1980). [Özel İsim] [Özel İsim], [Özel İsim] ( 1980 | 0.285 | 0.0690 |
| **R0** | teknik çözüm: Tanımlanan özellikler doğrudan kestane ağacına aittir. | 0.285 | 0.1053 |
| **R4** | teknik çözüm: Tanımlanan özellikler doğrudan ahşap parça birleşimdir. | 0.285 | 0.1579 |
| **R10** | teknik çözüm: Tanımlanan özellikler doğrudan çam ağacına aittir. | 0.285 | 0.1053 |
| **R20** | teknik çözüm: Tanımlanan özellikler doğrudan kırlangıç ağacına aittir. | 0.285 | 0.1053 |

### M5 zeytin ağacı

> **Soru:** Kıvrımlı, kontrastlı damarlarıyla eşsiz bir desene ve sertliğe sahiptir

> **Külliyat referansı (insan yazımı):** Bu gereksinim için zeytin ağacı son derece uygundur. Kıvrımlı, kontrastlı damarlarıyla eşsiz bir desene ve sertliğe sahiptir; dekoratif sunum tahtaları ve mutfak eşyalarında popülerdir.

| model | arka kapının çıktısı (`response_text`) | RAG | F1 |
|---|---|---|---|
| **TABAN** | DERIV_lI COPULA_AORIST. [Özel İsim] [?], [?] ve [?] gibi) ( 1980, [Özel İsim]- [?]). bu türlerdeki bir [Özel İsim] cinsidir. bu türlerde, [?] ( [Özel İsim]) cinsidir | 1.000 | 0.1176 |
| **R0** | teknik çözüm: Tanımlanan özellikler doğrudan kayın ağacına aittir. | 1.000 | 0.0000 |
| **R4** | usta marangoz: bu gereksinim için marangozlukta ve parçaya uygun açıyla yanaşılmalıdır. | 1.000 | 0.2286 |
| **R10** | teknik çözüm: bu işlem için marangozlukta ve parça aletlerin birbirine boşluksuz ve kasmadan oturması mukavemet için şarttır. | 1.000 | 0.1951 |
| **R20** | teknik çözüm: Tanımlanan özellikler doğrudan kestane ağacına aittir. | 1.000 | 0.0000 |

### G1 genel

> **Soru:** Türkiye'nin başkenti neresidir?

> **Külliyat referansı (insan yazımı):** KULLIYATTA YOK

| model | arka kapının çıktısı (`response_text`) | RAG | F1 |
|---|---|---|---|
| **TABAN** | [Özel İsim], 1986 yılında inşa edilmiştir. [Özel İsim], 1968 yılında inşa edilmiştir. [Özel İsim], 1968 yılında inşa edilmiştir. [Özel İsim], 1 | 0.000 | — |
| **R0** | teknik çözüm: bu gereksinim için marangozlukta kırlangıç tercih edilir. | 0.000 | — |
| **R4** | teknik çözüm: bu gereksinim için marangozlukta ve [?]- ı [Özel İsim], Danimarka'lı bir devlet adamıdır. | 0.000 | — |
| **R10** | teknik çözüm: bu gereksinim için marangozlukta ve iç [?] bahçe aletlerinde kullanılan hibrit bir ağaçtır. | 0.000 | — |
| **R20** | teknik çözüm: bu gereksinim için marangozluktaki en rahat işlenen, yoğun işkence ve nihai yapıya sahiptir; dış [?] ise yüzde 125 derece olmalıdır. | 0.000 | — |

## §3 — Telemetri (arka kapının kendi alanları)

| model | sha256 (kısa) | benzersiz yanıt / soru | en sık yanıt (kaç kez) | F1 ort. |
|---|---|---|---|---|
| **TABAN** | `b93cc1cd54093fc6…` | 6/6 | , [?] ve [Özel İsim] gibi). bu nedenle, bir [?] olarak kulla (1×) | 0.0625 |
| **R0** | `ba313677466108ba…` | 5/6 | teknik çözüm: Tanımlanan özellikler doğrudan kestane ağacına (2×) | 0.1652 |
| **R4** | `f3bd5be6aa71702b…` | 6/6 | teknik çözüm: bu gereksinim için marangozlukta ve parçaya uy (1×) | 0.2044 |
| **R10** | `1163b918abd6f52c…` | 5/6 | teknik çözüm: Tanımlanan özellikler doğrudan çam ağacına ait (2×) | 0.1361 |
| **R20** | `1d04365518cb067a…` | 5/6 | teknik çözüm: Tanımlanan özellikler doğrudan kestane ağacına (2×) | 0.1217 |

*“Benzersiz yanıt” sütunu soruya KOŞULLANMA ölçüsüdür*: 6 soruya 6 ayrı yanıt veren model soruya duyarlıdır; aynı cümleyi tekrarlayan model cümle kalıbını öğrenmiştir, içeriği değil.

## §4 — Denetim: kapının dokunduğu yüzeyler

**Değişen yüzey sayısı:** 1

### BULGU (K15) — arka kapı `data/future_train_vector.jsonl`'e YAZDI

Arka kapı **salt-okunur değildir**. `process_query` içinde tasarlanmış bir yol var (`src/rag/epistemic_agent.py:339-366`):

```
match_score >= 0.85  VE  (entropy_post > tau  VEYA  çıktı 'bilgi yok' der)
   ⇒  kayıt `data/future_train_vector.jsonl`'e EKLENİR   ("Epistemik Kayıt")
```

Bu koşumda yazan tek sorgu: **TABAN / M5 zeytin ağacı**

Yazan/yazmayan ayrımı **ölçüldü** — aynı M5 sorgusu (rag=1,000) beş modelde de aynı belgeyi buldu, ama yalnız TABAN'ın son-entropisi tau'yu aştı:

| model | rag | entropy_post | epistemic_failure | kayıt yazıldı |
|---|---|---|---|---|
| TABAN | 1.0000 | 4.3457 | True | True |
| R0 | 1.0000 | 0.8292 | False | False |
| R4 | 1.0000 | 1.1132 | False | False |
| R10 | 1.0000 | 0.4544 | False | False |
| R20 | 1.0000 | 0.8744 | False | False |

**Yönetişim boşluğu (kapatılmadı, beyan edildi):** `data/future_train_vector.jsonl` `data/**` altındadır ⇒ CLAUDE.md kural 3'e göre **salt-okunur**; ama `.agent-bus/frozen.json`'un 10 deseninden **hiçbirine uymuyor** (`data/*.pt`, `data/*.bin`, … hiçbiri). Yani ne kural 3 ne donmuş desen bu yolu makine düzeyinde koruyor. T-0094'ün 20 adet `.bin.meta.json` bulgusuyla **aynı sınıf**: donmuş liste bir **dosya**yı saymıyor.

**Yapılan:** dosya **0 bayttı** (`e3b0c442…`, boş dosya digest'i); koşum sonunda **1 kayıt** (2,2 KiB, `964a168d…`). Kayıt `scratch/t0095_arka_kapi/KANIT_future_train_yazilan_kayit.jsonl` altına **kopyalandı** (kanıt korunur). **Geri alma yapılmadı:** yazma `data/**` altına elle müdahale gerektirir ve bu oturumda `data/` yazımı **engellendi** (kural 3 doğru çalışıyor). Karar operatöründedir: boş hâline döndürülsün mü, yoksa kayıt kalsın mı?

### Değişmeyenler (ölçüldü)

| yüzey | sonuç |
|---|---|
| `data/qdrant_db/**` (6 dosya) | **DEĞİŞMEDİ** — hepsi baş/son aynı digest |
| `data/anka_a1r.pt` | **DEĞİŞMEDİ** `b93cc1cd54093fc6…` |
| `scratch/t0094_sonda/arm_R*.pt` (4) | **DEĞİŞMEDİ** |
| `src/**`, `scripts/**` | **İZLENEN DOSYA DEĞİŞMEDİ** — `git status --porcelain` ⇒ TEMIZ — izlenen hiçbir dosya değişmedi |

**K16 — kendi kusurum, düzeltildi:** git çıktısının ilk sürümü tek bir `?? scripts/evaluate_carpenter_anka.py` satırıydı ve rapor bunu *“değişiklik YOK”* diye yazıyordu ⇒ **iddia kendi kanıtıyla çelişiyordu**. `??` (izlenmeyen) ile `M/A/D` (izlenen değişik) artık **ayrı** ölçülür ve hüküm yalnız ikincisine bağlanır. `??` satırı bu görevde üretilmedi: T-0094'ün dosyasıdır ve oturum başındaki git durumunda zaten `??` idi.


## §5 — Sınırlar (yapılmayanlar, beyan)

- **Hiçbir `src/**` dosyası değiştirilmedi.** Kapının `max_new_tokens=45`'i, `tau`'su, `similarity_threshold=0.85`'i **olduğu gibi** bırakıldı.

- **Lexicon uyuşmazlığı (ölçüldü, düzeltilmedi):** `AgentGateway.create_default` varsayılanı `data/lexicon/roots.tsv` (**52.374** satır); Anka külliyatının lexicon'u `data/lexicon/roots_anka_r1.tsv`. `run_agent_arena.py` lexicon **geçmiyor** ⇒ arka kapı Anka'ya **taban lexicon**'u ile hizmet etti. Yani bu ölçüm arka kapının **BUGÜNKÜ** hâlidir; `literal/PN` ayrışmasının (T-0093/U6) bir kısmı buradan da gelebilir. **Bu koşumda izole edilmedi.**

- **M5 (`rag=1,000`) tek bir belgeye tam eşleşmedir** — o sorgu külliyat cümlesinin kendisidir; RAG skoru bu yüzden 1,0. Diğer beş sorgu 0,24–0,38 bandında ve **hiçbiri** 0,85 eşiğini geçmiyor ⇒ **RAG koşullaması pratikte devreye girmedi**.

- **F1 sütunu yeni bir ölçüttür, T-0094'ün ROUGE'u DEĞİLDİR.** Farklı n, farklı üretim uzunluğu (45 ↔ 128), farklı zarf. İki sayı **kıyaslanmaz**.

- Beş kol arasında **sıralama iddiası yok**: F1 farkları bu örneklemle ayrıştırılmadı; T-0094 zaten sonda kollarının sıralanamadığını ölçmüştü (R²=0,0162).

## §6 — Hüküm ve “yeni marangoz eğitilelim mi?” sorusu

### Ölçülen tablo

| | TABAN `anka_a1r.pt` | sonda kolları R0–R20 |
|---|---|---|
| ceket eğitimi | **yok** (0 epoch) | 0,12–0,16 epoch (T-0094) |
| çıktı biçimi | yer tutucu/ansiklopedik dağınıklık: `, [?] ve [Özel İsim] gibi` | **marangoz kalıbı**: `teknik çözüm: bu gereksinim için marangozlukta …` |
| soruya koşullanma | yok | **yok** — aynı cümle farklı sorulara tekrar ediyor |
| T-0094 A ekseni (genel CE) | — (kimlik) | **+%24,5…+%35,8** ⇒ eşiği (+%10) AŞIYOR |
| T-0094 B ekseni (noktalama) | — (kimlik) | +0,28…+1,82 ⇒ eşiği geçiyor |

### Ölçülmüş cevap

1. **“Modül model Marangoz” diye bir checkpoint YOK.** Ölçüldü: `data/anka_*.pt` altında yalnız `anka_a1.pt` ve `anka_a1r.pt` var (ikisi de TABAN); kristal zinciri 18 Eyl'de silindi; `data/anka_a1r_ceket.pt` **yazılmadı** (T-0094'ün denetimi yokluğunu doğruluyor). Marangoz kelimesini hak eden tek ağırlık **sonda kolları**dır ve onlar **0,16 epoch**'luk birer sondaydı.

2. **Arka kapıdan bakınca kollar hiç de boş değil:** TABAN'ın dağınık çıktısına karşı kollar **dilbilgisel olarak doğru, alan içi Türkçe** üretiyor. Yani ceket **biçimi** öğrenilmiş.

3. **Ama içerik öğrenilmemiş:** kollar altı sorunun çoğuna **aynı cümleyi** veriyor (ör. R0: M1 ve M4 ⇒ *“Tanımlanan özellikler doğrudan kestane ağacına aittir.”*). 0,16 epoch'ta bu beklenen sonuçtur; **“marangoz öğrenilemez” demek değildir** — **“0,16 epoch yetmez”** demektir.

4. **Yeni marangoz eğitimi T-0094'ün kapısından geçmedi:** dört kolun dördü de A ekseninde eşiği aştı. Yani **uzun bir koşum, replay oranı seçilmeden başlatılamaz**; seçim için gereken sonda da kolları sıralayamadı (R²=0,0162). Bu bir **engel değil, sıralama sorunu**: doğru sonda tasarımı *blok sırasını sabitleyip yalnız oranı oynatmak* ya da tersiydi (T-0094/K10).

5. **Karar operatöründedir.** Ölçüm şu üç seçeneği ayırıyor:

   - **(a)** Yeni sonda: sıra sabit + oran değişken (ya da tersi), uzunluk artırılmış ⇒ hangi replay oranının A eksenini +%10 içinde tuttuğunu **ölçer**.

   - **(b)** Doğrudan tam giydirme: replay oranı **seçilmeden** — T-0094'ün uyardığı adım; A ekseninin ne olacağı önceden bilinmez.

   - **(c)** Ceketi yeniden derlemeden önce **lexicon'u Anka'nınkiyle hizala** (bu koşumda ölçülen §5 açığı) — çünkü temsil ayrışmasının (T-0093/U6) bir kaynağı tam olarak bu olabilir.

   Bu rapor **öneri üretmez**; hangi yolun seçildiğini ve gerekçesini operatör verir.

## §7 — Digest tablosu (tam)

| dosya | sha256 |
|---|---|
| `data/anka_a1r.pt` | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `scratch/t0094_sonda/arm_R0.pt` | `ba313677466108ba0e18dbb851390a0635c592da8a631cf95eb4729532df2120` |
| `scratch/t0094_sonda/arm_R4.pt` | `f3bd5be6aa71702bbaa60a8622e1be2cf00d0b9c15d4737cedcf667bdca9e574` |
| `scratch/t0094_sonda/arm_R10.pt` | `1163b918abd6f52ced42b916d8edd085daa835ef912fe37d459cfa9750f125a4` |
| `scratch/t0094_sonda/arm_R20.pt` | `1d04365518cb067abca89a122913dd1911158196cfb1c9149b7cdba4f675d6fa` |
| `data/rebuild/vocab_anka_r1_33114.json` | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| `data/lexicon/roots.tsv` | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/lexicon/roots_anka_r1.tsv` | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| `data/future_train_vector.jsonl` | `964a168d5075bbf0f31a1f7f5b48b759fafe8f60946d861393af633dbe08410c` |
| `data/pedagogy_canonical/carpenter_canonical.jsonl` | `d0d9703970d8d41eda64f8a0b8f3064d3c45177bdf77250bab7c71e02fc38ef0` |
| `scratch/t0095_arka_kapi.py` | `e248f97b1b4a74b62174f5fb1954acdede4e1eb7afe38e304fef2309449681b1` |
| `scratch/t0095_rapor_uret.py` | `9992bc6cfa90b25ca8a125e20466fdfc1c058ee564951fc71ad07833dc014849` |
| `scratch/t0095_arka_kapi/KANIT_future_train_yazilan_kayit.jsonl` | `964a168d5075bbf0f31a1f7f5b48b759fafe8f60946d861393af633dbe08410c` |

*Bu raporun kendi digest'i tabloda YOKTUR — yazıldığı anda kendi içeriği değişir ve taşıdığı değer bayatlardı (T-0094 §7.0 dersi). Okuyucu kendisi hesaplar.*

**Sayı denetimi:** bu raporun tablolarındaki her sayı `sonuc.json`'dan betikle üretildi (elle yazılmadı); `data/eval/anka_r17_arka_kapi_2026-09-20.json` aynı kaynaktan gelir.

## §8 — Kapanış kanıtları (betikle ölçüldü)

| kanıt | sonuç |
|---|---|
| Donmuş yüzey taraması (T-0095 `claimed_at=2026-09-20T15:44:18Z` sonrası) | taranan **1.478** · donmuş desene eşleşen **69** · sonrası yazım **0** ⇒ **TEMIZ** |
| `scratch/*.py` py_compile kanaryası | 163 betik · bozuk **0** ⇒ **TEMIZ** |
| Rapor ↔ JSON sayı denetimi | 48 kontrol · uyuşmazlık **0** ⇒ **TEMIZ** |
| `venv/bin/pytest -q` | `1 failed, 237 passed in 92.29s (0:01:32)` |

Düşen test(ler): `tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints`

`data/future_train_vector.jsonl` **bilinçli olarak tabloya konmadı**: yukarıda §4'te ayrı ele alınıyor (arka kapının tasarlanmış yazma yolu; donmuş desene uymuyor).
