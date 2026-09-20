# T-0090 · Onaysız küçük açıklar — `chat_prompt.py` rc=2 · stage-1 `--pretrain` · fixture sözlüğü · README arena satırları

**Görev:** T-0090 · `status: done`
**Damga (ölçüldü — betik hesapladı, elle yazılmadı): `2026-09-20T11:13:24Z`**
**Yetki:** operatör onayı — *"Onay bekleyen işler için devam et"* ⇒ dört iş akışından **(3) Onaysız
küçük açıklar** ve **(4)'ün README kısmı**.
**Kaynak (özet değil, tam metin okundu):** T-0088 kapanış raporu `§10` açık maddeleri **1, 3, 4, 5**
(`data/eval/anka_r11_kucuk_borclar_2026-09-20.md:183-203`).

> **Ölü artefakt ADLARI bu raporda ANILMAZ.** Adlar T-0086 aracının kanonik manifestinde durur
> (digest birebir doğrulanır). Gerekçe ölçülmüştür: bir yolu **adıyla anan** her satır sayacı
> şişirir — bu sınıf T-0083, T-0088 ve T-0089'da **üç kez** üredi, bu görevde **dördüncü kez**
> (bkz. §7/1). Raporda adı yazmamak, sayacı **kasten** şişirmemek içindir.

---

## 1. İlan edilen kapılar (ölçümden ÖNCE yazıldı) ve sonuç

| Kapı | İlan edilen ölçüt | Ölçülen | Hüküm |
|---|---|---|---|
| **Y1** | üç sessiz dal → **3/3 `rc=2`**, mesaj **stderr**'de, stdout'ta DEĞİL | 3/3 rc=2; stderr'de 1, stdout'ta **0** | **KALDI** |
| **Y1b** | pozitif dal: kapı **ateşlenmez**, betik kapıların ALTINDAKİ satıra ulaşır | ulaştı (iki ayrı kap) | **KALDI** |
| **Y2** | stage-1 + `--pretrain` → `rc=0` ve ilk kayıp > 5,0 | rc=0 · ilk kayıp **10,5279** | **KALDI** |
| **Y3** | `venv/bin/pytest` → yeni düşüş **yok** | 1 failed, **227 passed** (taban 223) | **KALDI** |
| **Y4** | T-0086 aracı: `README.md olu ≤ 2` (artış YOK) | **2 → 2** | **KALDI** |

**Çürütme maddeleri (önceden yazıldı):** (a) dal rc=2 olup stdout'a basarsa ⇒ kısmi;
(b) Y1b'de kapı ateşlenirse ⇒ kapı **aşırı geniş** ve Y1 kanıt değildir;
(c) Y3'te yeni düşüş olursa ⇒ fixture değişikliği geri alınır ve madde kapsam dışı yazılır.
**Üçü de ateşlenmedi.**

---

## 2. Y1 — `chat_prompt.py`: üç sessiz dal (T-0088 §10/4'ün tam sınıfı)

**Ölçülen taban (değişiklikten ÖNCE, aynı kap):** üç dalın **üçü de `rc=0`**.

| Dal | Komut (kısaltılmış) | ÖNCE | SONRA | ÖNCE stdout/stderr | SONRA stdout/stderr |
|---|---|---|---|---|---|
| K1 | `--vocab YOK_boyle_bir_sozluk.json` | `rc=0` | **`rc=2`** | — | 0 / **1** |
| K2 | `--vocab <33.114> --model YOK_….pt` | `rc=0` | **`rc=2`** | — | 0 / **1** |
| K3 | lexicon çözülmez (`cwd=$TMPDIR`) | `rc=0` | **`rc=2`** | — | 0 / **1** |

Kusur sınıfı: `main()` içindeki `print` + **çıplak `return`** ⇒ süreç `rc=0` ile çıkar ve çağıran,
*"dosya yok"* ile *"başarıyla koştu"*yu **hiçbir sinyalden** ayırt edemez. `print` bir uyarı
değildir: stderr'e gitmez, önem derecesi taşımaz, boru hattında görünmez.

**Düzeltme:** `chat_prompt.py`'ye `_durdur(mesaj) -> NoReturn` eklendi —
`print(..., file=sys.stderr, flush=True)` + `sys.exit(2)`. Emsal: `train_dpo.py` `_durdur` (T-0089),
orada da DPO hiç yapılmadan başarı dönüyordu.
**Beyan edilen genişleme:** dosyadaki **iki mevcut `sys.exit(2)` sitesi** (T-0087'den) de aynı
yardımcıya taşındı ⇒ dosya içi **tek** durma davranışı. Onların `rc`'si zaten 2'ydi; değişen yalnız
**akıştır** (stdout → stderr). Beş site artık birebir aynı sözleşmeyi taşır.

### Y1b — pozitif dal (kapı AYIRT ediyor mu, yoksa her şeyi mi durduruyor?)

| Kap | Komut | Sonuç |
|---|---|---|
| **Gerçek checkpoint** (elle, 356 MB) | `--vocab <33.114> --model data/anka_a1r.pt` | Üç kapı da **geçildi**; istem satırına ulaştı (`Model: anka_a1r.pt`), `DURDURULDU` **yok** |
| **Ucuz kap** (otomatik test) | `--vocab <33.114> --model data/lexicon/roots.tsv` | `rc=1`, `DURDURULDU` **yok**, `Model Ağırlıkları Yükleniyor` satırı stdout'ta **var** |

**İddia sınırı (beyan):** otomatik teste alınan kap **ucuz** olandır — `--model` var olan ama
checkpoint olmayan bir dosyaya işaret eder ⇒ üç kapı geçilir, `torch.load` sesli hata verir ve süreç
**kendi kendine biter**. Gerçek checkpoint'li koşum otomatik teste **alınmadı**: stdin `EOF`'ta REPL
**sonsuz döngüye** giriyor (§6/1 — kapsam dışı, düzeltilmedi, açık madde olarak yazıldı).

---

## 3. Y2 — stage-1 `--pretrain` (T-0088 §10/5: *"ölçülmedi"* → **ölçüldü**)

Komut, stage-1'in **birebir bayrak kümesi** (`--device cpu --data data/train.bin --steps 2
--from-scratch --vocab <33.114> --save-path $TMPDIR/…`):

| Dal | `--pretrain` | Sonuç |
|---|---|---|
| **A** | **yok** | **`rc=1`** — `RuntimeError` @ `train.py:275`: *"SFT maskelemesi bu partide HICBIR hedef birakmadi"* |
| **B** | var | **`rc=0`** · ilk kayıp **10,5279** · bitiş 8,3653 · 356 M `.pt` (`$TMPDIR`) |

**Hüküm:** stage-1 bu bayrak olmadan **HİÇ koşamıyordu** ⇒ boru hattı 1. aşamada ölüydü.
**Ama kusur SESSİZ DEĞİL:** `train.py:275` canlı kapısı ilk partide durur (T-0073 düzeltmesi
çalışıyor). Yani D1 sınıfının *"sessiz sıfır-kayıp"* tuzağı burada **kapalı**; kalan kusur, boru
hattının bu aşamayı hiç geçememesiydi. Düzeltme: `stage1_cmd`'e `--pretrain` eklendi
(`--from-scratch` **korundu**).

**Kap (beyan):** ölçüm sandbox içinde alındı ve sandbox **MPS'i gizler** ⇒ `--device mps` isteyen
koşum `train.py:48` fail-closed kapısında durur (bu, ölçüm kabının kusurudur, görevin değil). Bu
yüzden ölçüm `--device cpu` kabında alındı. **Soruyu değiştirmez:** hedef seçimi
(`mask_prompt_targets`) ve onu denetleyen kapı, tensörler cihaza taşınmadan **ÖNCE, CPU'da** çalışır
(`train.py:266-278`).

**Ek olarak ölçüldü (kusur DEĞİL):** `run_goal_pipeline.py`'nin `run_cmd`'i
`subprocess.run(cmd_list, check=True)` kullanır ⇒ alt süreç `rc≠0` dönerse `CalledProcessError` ile
boru hattı **durur**. Yani stage-1 hatası yutulmaz (T-0089'un "sessiz `return`" sınıfı burada **yok**).

---

## 4. Y3 — `tests/test_prompt_contract.py` fixture'ı (T-0088 §10/3)

`env` fixture'ı 31.357 girişli **BAYAT** sözlüğü yüklüyordu; güncel külliyat **33.114**. Yani test
takımı, ürünün gördüğü sözlükten **farklı** bir tokenizer kuruyordu. Fixture artık dosyada zaten var
olan `GUNCEL_VOCAB` sabitini kullanır.

| Ölçüm | ÖNCE | SONRA |
|---|---|---|
| `venv/bin/pytest` | 1 failed, **223 passed** | 1 failed, **227 passed** |

Artış **tam +4** = yeni `tests/test_chat_prompt_gates.py`'nin 4 testi. Düşen tek test yine
`test_agent_gateway.py::…::test_gateway_http_server_endpoints` — **sandbox soket yasağı**
(`socketserver.py:478 PermissionError`), görevle ilgisiz, ilan edilmiş davranış.
⇒ **Fixture değişikliği hiçbir testi düşürmedi** (çürütme maddesi (c) ateşlenmedi).

### Yeni test dosyası: `tests/test_chat_prompt_gates.py` (4 test)

Üç **duran** dal (`rc=2` + mesaj **stderr**'de + stdout'ta **değil**) ve **bir geçen** dal.
Sonuncusu kritiktir: yalnız *"duruyor mu"* bakan bir test, **her şeyi durduran** bir kapıyı da
geçirirdi ⇒ **vakum kapı**. Ölü ad anılmaz (T-0086/T-0089 sınıfı).

---

## 5. Y4 — README (T-0088 §10/1)

Ölçülen kalem: `run_agent_arena` komutları ve seçenek tablosu **ZORUNLU** `--model`/`--vocab`
içermiyordu ⇒ **bugün çalışmayan** komutlardı; ayrıca sözlük artık gömülü sabit değil **parametre**
olduğu için `vocab_base_32852.json` iddiası eksikti.

| Düzeltilen | Ne yapıldı |
|---|---|
| §4 arena komutları (3 adet) | `--model data/anka_a1r.pt --vocab data/rebuild/vocab_anka_r1_33114.json` eklendi + zorunluluk notu |
| §3 `chat_prompt.py` komutu | Aynı sınıf: `--vocab`/`--model` **ZORUNLU** (T-0087) ⇒ komut tamamlandı + `rc=2` notu |
| §5 (b2) sözlük iddiası | "sözlük olarak 32.852 yükler" → **TABAN 32.852, güncel külliyat 33.114 ve `--vocab` ile AÇIKÇA verilir** |
| Seçenek tablosu — arena satırı | İki **ZORUNLU** bayrak + `--repetitions`/`--device`/`--port`; "koşmadan durur" notu |
| Seçenek tablosu — `chat_prompt.py` satırı | İki **ZORUNLU** bayrak + `rc=2` davranışı (T-0087/T-0090) |

**Ölü atıf sayacı — sınıflandırılmış (çıplak toplam DEĞİL):**

| Ölçüm | ÖNCE | SONRA | Fark |
|---|---:|---:|---:|
| `README.md` (T-0086 aracı) | 2 | **2** | **0** |
| Dokunulan diğer dosyalar (`chat_prompt.py`, `run_goal_pipeline.py`, `run_agent_arena.py`, `test_prompt_contract.py`, yeni test) | — | — | **0** |
| **Tüm ağaç (ham toplam)** | 2447 | 2449 | **+2** |

**İki artışın ikisi de TEK bir satırdan gelir — ama biri doğrudan, biri ÖLÇÜM ARACININ
kendisinden:**

* **+1 doğrudan:** `.agent-bus/state/tasks/T-0090.json` — bu görevin **kendi beyanı** (README
  kalemini tarif ederken ölü adı **anmak zorunda kaldı**). Kod/belge kusuru **değil**; kayıt
  olduğu gibi bırakıldı — beyanı, sayacı düzeltmek için **sonradan** değiştirmek kaydı bozmak
  olurdu.
* **+1 dolaylı (ölçüldü, ilk kez):** T-0086 aracı **kendi çıktısını** (`data/eval/anka_r9_…json`)
  da ağaç sayar. Yeni bir ölü-atıf taşıyan dosya listeye girdiği için aracın kendi kaydı da bir
  ad daha taşır ⇒ bir sonraki koşumda **+1**. Yani araç, kendi ölçtüğü yüzeyin **parçasıdır**.
* **Sınırsız şişme YOK — idempotans ölçüldü:** ağaç **değişmeden** üç ardışık koşum:
  `2449 · 2449 · 2449` ⇒ etki **tek seferliktir**, sonra sabitlenir.
* **Aracın kendi payı ayrıştırıldı:** aracın artefaktı ham toplamın **934** adını taşır
  (2449'un **%38,1**'i) ⇒ KANARYA da çıkarıldığında "gerçek ağaç" **2445 değil**, **2445 − 934
  = 1511**'dir. Aracın ilan ettiği `gerçek_olü_kanarya_haric: 2445` **kendi kaydını içerir**.

README'nin mevcut 2 atfı **mirastır** (T-0089'da ölçüldü: bu görevde hiç eklenmedi) ve
T-0081/T-0083 kararıyla **tarihsel kayıt** olarak korunur; bu görev onları **silmedi**.

---

## 6. Bu görevde ölçülen YENİ bulgular (kapsam dışı — **düzeltilmedi**, beyan edilir)

1. **`chat_prompt.py` REPL, stdin `EOF`'ta sonsuz döngüye giriyor** — **sayıyla ölçüldü**:
   `Hata Oluştu: EOF when reading a line` basılıyor, ardından istem **bütün blok hâlinde tekrar
   basılıyor** ve süreç **hiç bitmiyor**. Gerçek checkpoint'li koşumun standart çıktısı dosyaya
   alındı (`$TMPDIR/t0090/poz.out`, süreç elle durduruldu):

   | Ölçüm | Değer |
   |---|---|
   | pencere (doğum → son yazma) | **322,98 sn** (zaman dilimi farkından bağımsız; mutlak damga kullanılmadı) |
   | üretilen çıktı | **4.724.070.497 bayt** (~4,4 GiB) · **94.355.597 satır** |
   | **benzersiz satır** | **37** |
   | istem döngüsü (aynı blok) | **6.290.372** |
   | `EOF` hata satırı | **6.290.371** |
   | hız | **19.476 döngü/sn** · 14,6 MB/sn |
   | sha256 (kanıt, dosya **silindi**) | `8aa741ec6a69c49ea0a3459c9ba1a5248a3e8dd0fb4f8b4f4530502cb471f3fe` |

   Yani bu bir *yavaşlama* değil: **saniyede ~19,5 bin kez** sonsuz döngü. Boru hattında/otomatik
   koşumda **diski ve CPU'yu dolduran** bir asılmadır. **T-0067 gereği tasarıma sessizce
   yazılmadı** — kapsam dışı bırakıldı ve buraya yazıldı.
   **Beyan:** 4,4 GiB'lik geçici kanıt dosyası, yukarıdaki tam digest + satır/benzersiz/hız
   sayıları alındıktan **sonra silindi** (disk hijyeni); kanıt artık bu tablodadır, dosyada değil.
2. **Yukarıdakinin doğrudan sonucu:** `chat_prompt.py`'nin uçtan uca pozitif dalı **otomatik teste
   alınamaz** (asılıyor) ⇒ testteki pozitif kap ucuz kaptır (§2/Y1b, iddia sınırı orada beyan edildi).
3. **`run_cmd` `check=True` kullanıyor** ⇒ sessiz yutma yok. Bu bir bulgu **değil**, ölçülmüş bir
   *güvence*dir; T-0089 sınıfının burada bulunmadığını kayda geçirir.
4. **T-0086 aracı kendi çıktısını kendi girdisi olarak sayar (öz-referans) — ilk kez ölçüldü.**
   Aracın artefaktı `data/eval/anka_r9_…json`, taradığı ağacın **içindedir** ve kendi per-file
   kaydını da taşır ⇒ (i) ham toplamın **934** adı **aracın kendi kaydıdır** (2449'un %38,1'i);
   (ii) ağaca **yeni** bir ölü-atıf dosyası girdiğinde araç **bir sonraki koşumda** fazladan
   **+1** yazar. Etki **tek seferliktir ve sınırlıdır** — ağaç sabitken üç ardışık koşum
   `2449 · 2449 · 2449` verdi (**idempotent**). Yani araç **kaçak değil**, ama **sayacı kendi
   kaydıyla kirlidir**: `gerçek_olü_kanarya_haric` alanı "gerçek ağacı" değil, **ağaç + aracın
   kendi kaydını** verir. **Düzeltilmedi** (araç donmuş değil ama bu görevin kapsamı dışı) —
   ölçüldü ve buraya yazıldı.

---

## 7. Kendi kusurlarım (ölçüm öncesi/sonrası, aynen)

1. **"Emekli/ölü adı anma sayacı şişirir" sınıfı DÖRDÜNCÜ kez üredi** — bu kez **görev beyanımda**:
   `.agent-bus/state/tasks/T-0090.json` ölü adı anıyor ⇒ doğrudan **+1**. Ölçüm bunu **iki**
   artışa çevirdi (biri aracın öz-referansı, bkz. §5 ve §6/4): ham toplam **2447 → 2449**, ama
   **gerçek ağaç** (aracın kendi kaydı çıkarılınca) **1511**. Sınıf T-0083
   (emekli notu), T-0088 (araç sözlüğü), T-0089 (yeni test dosyası docstring'i) ve şimdi
   **(görev spec'i)**. **Kuralı bilmek üremeyi engellemiyor; biçim her seferinde yeni.**
   Bu raporda ad **kasten anılmadı**; ama beyan dosyasında anıldı ve **her iki sayı da** yazıldı.
2. **Önce ölçmeden bir kalemi yanlış tarif ettim.** Bu göreve başlarken "README'de 2 canlı ölü-atıf
   **mirası**" maddesini bir *düzeltilecek kalem* sanıyordum; kaynağı (§10/1) okuyunca bunun bir
   kalem **olmadığı**, T-0089'un ölçtüğü **miras** olduğu ve gerçek kalemin **arena komutları +
   `--vocab` iddiası** olduğu çıktı. Ders: özetten değil **kaynaktan** oku ([[ozet-bayat-olabilir-kaynagi-oku]]).
3. **Sandbox MPS'i gizledi** ve ilk stage-1 ölçümüm `train.py:48` kapısında `rc=1` verdi. Neredeyse
   *"stage-1 zaten rc=1"* diye yazacaktım — oysa o **kabın** kusuruydu (`--device mps` sandbox'ta
   yok). Kap düzeltilince gerçek taban `rc=1` **çıktı ama başka sebepten**; iki `rc=1`'i ayırt
   etmeseydim rapor yanlış olurdu ([[kopyalanan-betik-kabini-degistirir]] ailesi).
4. **Denetim aracının çıktı JSON'unu üzerine yazdım** (`data/eval/anka_r9_olu_atif_denetimi_2026-09-20.json`)
   — T-0088'in ürettiği artefaktın **önceki hâlinin digest'ini almadan** koştum. Araç deterministik
   olduğu ve ağaç değişmediği için kayıp **yok**; ama ders [[kabul-kosusu-olctugu-artefakti-degistirir]]:
   **ölçüm aracını koşturmadan önce hedef artefaktın digest'ini al.** "Önce" hâli `$TMPDIR/t0090/denetim_once.json`.

---

## 8. Açık kalan maddeler (kapanmadı — beyan edilir)

1. **REPL `EOF` sonsuz döngüsü** (§6/1) — ölçüldü, **düzeltilmedi** (kapsam dışı).
2. **`run_goal_pipeline.py` uçtan uca koşulmadı** (T-0088 §10/6 aynen duruyor): `data/qdrant_db` +
   Gemini/Ollama öğretmen + ağır checkpoint. Bu görevde ölçülen **stage-1 kapısıdır**, uçtan uca
   başarı **değildir**.
3. **`scripts/retrain_clean_models.py` bütünüyle ölü** (T-0089 §1) — dokunulmadı.
4. **AdamW momentleri kaydedilmiyor** (T-0090 kapsamı dışı; onaylı iş akışı (4)'ün ikinci yarısı).
5. **D1 (en-uzun-kök) + D4 (kesme/rakam)** — onaylı iş akışı; D1 donmuş `src/compiler/**` yazımı
   gerektirir ve **ilan edilen kuralı zaten ölçümle ÇÜRÜTÜLMÜŞTÜR** (`anka_r2_on_olcum:30`:
   uzunluk ekseni **net −1** ⇒ ayırt edici değil) ⇒ kör uygulanmayacak.
6. **`scratch/**` sürümlenmiyor** (T-0088 §10/7) ⇒ T-0086 aracı ve kanarya fikstürleri taze klonda yok.
7. **T-0086 aracının öz-referansı** (§6/4): sayacı kendi çıktısını içerir (934 ad) ve yeni bir
   ölü-atıf dosyası eklendiğinde bir sonraki koşumda +1 yazar. **Ölçüldü, düzeltilmedi**;
   düzeltmenin doğru biçimi (aracın kendi çıktısını `KAPSAM`tan çıkarması) bir sonraki tura aittir.

---

## 9. Tam digest tablosu (önek DEĞİL)

| Yol | sha256 |
|---|---|
| `chat_prompt.py` | `d6614977fc9e366dd72adcd835ac3fdca3569411626f88e27ebf0d071d18a44a` |
| `scripts/run_goal_pipeline.py` | `58b76dc5489eb37092732422a34c11fcd92205ce19bcd3f328e282ae880af1b4` |
| `tests/test_prompt_contract.py` | `ab2eb6864e73ea4c7b3c47ab0bd72675e8094d7b25d30142810fb01682d6a5d6` |
| `tests/test_chat_prompt_gates.py` (YENİ) | `554d7fa2b623a27b8dbfec8b25d4af7f59dc51cb052b59da02674292f37101ab` |
| `README.md` | `bb2080bb2c41308417675299005ae26bee72c7505199c667402a3b8e0aed2000` |
| `data/eval/anka_r9_olu_atif_denetimi_2026-09-20.json` (araç çıktısı) | `5ee47b746d6773c297939025c19b628d56533afb01df38fada9dbcb802203c5f` |

**Değişmeyen ama doğrulanan:** `data/**` altındaki donmuş yüzeyde bu görevde **yazma yok**;
`.pt`/`.bin` üretilmedi (stage-1 sondayı `$TMPDIR`'e yazdı, donmuş yola değil).

---

## 10. Hüküm

**Beş ilan edilen kapının beşi de KALDI; çürütme maddelerinin hiçbiri ateşlenmedi.**
Kapatılan kusurlar: `chat_prompt.py`'nin **üç sessiz `rc=0` dalı** (→ `rc=2`, stderr),
**stage-1'in koşamaması** (`--pretrain`), **bayat test fixture'ı**, **README'nin çalışmayan arena
komutları + eksik sözlük iddiası**.

**İddia sınırı (beyan):** bu görev **eğitim koşumu yapmaz**, uçtan uca boru hattını **koşmaz** ve
`data/**` donmuş yüzeyine **yazmaz**. Ölçülen şey **yapılandırma kapılarıdır** — kapıların
`rc`'si ve akışıdır, üretilen modelin kalitesi değildir. `chat_prompt.py` için kapatılan sınıf
"**sessizlik**"tir; REPL'in EOF asılması (§6/1) **hâlâ açıktır** ve bu hüküm onu kapsamaz.
