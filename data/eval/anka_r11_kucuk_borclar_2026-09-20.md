# T-0088 · KÜÇÜK BORÇLAR (4 KALEM) — KAPANIŞ RAPORU

**Tarih:** 20 Eyl 2026 · **Yürütücü:** claude · **Durum:** `done` (COMMIT EDİLMEDİ — operatör yetkisinde)

**Kapsamın kaynağı (kendi cümlem DEĞİL):** `data/eval/anka_r10_okuma_varsayilanlari_2026-09-20.md` §7 (devredilen
satırlar) + §10 (açık maddeler). Operatör bir önceki turda dört kalemlik "Küçük borçlar" işini seçti; bu rapor o
dört kalemi **ölçerek** kapatır.

| # | Kalem | Kaynak satır | Sonuç |
|---|---|---|---|
| a | `src/llm/prompt_contract.py:35` bayat sözlük varsayılanı | T-0087 §7 | **KAPANDI** |
| b | `scripts/run_agent_arena.py:49` ölü checkpoint varsayılanı | T-0087 §7 | **KAPANDI** (kapsam GENİŞLEDİ — §3) |
| b2 | `scripts/run_goal_pipeline.py` ölü soyağacı adları | T-0087 §7 | **KAPANDI** (kapsam GENİŞLEDİ — §4) |
| c | `USER_GUIDE.md:106` çürütülmüş satır | T-0087 §10/1 | **KAPANDI** |
| d | Denetim aracının etiket sözlüğü + girintili blok kuralı | T-0087 §10/2 | **KAPANDI** — *girintili blok iddiası ÇÜRÜDÜ*, etiket sözlüğü ÖLÇÜLDÜ |

---

## 1. İlan edilen kapılar ve ölçüm

| Kapı | İlan | Ölçüm | Sonuç |
|---|---|---|---|
| **K1** | `TokenizerConfig()` ve `describe()` argümansız **durur**; pozitif kontrol çalışır | `ValueError: TokenizerConfig.vocab_path VERILMEDI…` · `ValueError: describe(): config VERILMEDI…` · `[SOZLESME] zarf=etiket surum=2 vocab=data/rebuild/vocab_anka_r1_33114.…` | ✓ |
| **K2** | `prompt_contract.py` içinde ölü soyağacı adı **0** | denetim aracı: `ESLESME YOK` (`grep -c kristal` = 0) | ✓ |
| **K3** | `run_agent_arena.py` içinde **canlı** ölü atıf 0; `--model` verilmezse SESLİ durur | `ham=1 emekli=1 **olu=0**` · argümansız `rc=2` · yalnız `--model` → `rc=2` · olmayan yol → `rc=2` · **pozitif kontrol** banner'ı bastı | ✓ |
| **K4** | `run_goal_pipeline.py` içinde **canlı** ölü atıf 0; dört yol parametresi varsayılansız | `ham=1 emekli=1 **olu=0**` (öncesi: `ham=19 emekli=0 olu=19`) · argümansız `rc=2` · olmayan sözlük → `rc=2` | ✓ |
| **K5** | `USER_GUIDE.md:105` **değişmez** (ölçümle doğru), :106 **değişir** | 105 iddiası yeniden ölçüldü ve **DOĞRU** çıktı (`test_model.py:25,38,40-42`); 106 satırı ölçümle **yanlış** çıktı ⇒ değiştirildi | ✓ |
| **K6** | Araç girdileri: KANARYA birebir · negatif kontrol canlı | kanarya `.py` 5/2/3 ✓ · `.md` 3/2/1 ✓ · NEGATİF `scripts/t0027_measure_models.py` `ham=26 olu=26` ✓ | ✓ |
| **K7** | `pytest` sonucu taban çizgiden **sapması birebir beyan edilir** | taban **1 failed, 215 passed** → şimdi **1 failed, 216 passed**; sapma = **+1 test** (yeni T-0088 kanaryası) | ✓ |
| **K8** | 10 donmuş desene **0 yazma** | `find … -newermt "2026-09-20 00:00"` = **boş**; `data/vocab.json` = `4b40eebee9c996c8dd3150f6b3c8896de85764ee1cf0b08e6c636e5c002a0e69` (beyan edilenle **birebir**); `data/*.pt|*.bin` = **25** dosya | ✓ |

**Tek düşen test** (tabanda da düşüyordu): `tests/test_agent_gateway.py::…test_gateway_http_server_endpoints` —
`socketserver.py:478 PermissionError`. Bu **sandbox soket yasağı**dır, kod kusuru değildir (T-0082'de de aynı).

---

## 2. (a) `src/llm/prompt_contract.py` — bayat sözlük varsayılanı

**Önce:** `vocab_path: str = "data/vocab.json"` (31.357, BAYAT) · `describe(config=None)` argümansız çağrıda o
varsayılana düşüyordu. **Sonra:** varsayılan **YOK**; kapı **çağrı anında** ve iki ayrı sitede durur.

**Varsayılan BAŞKA BİR ADA taşınmadı** (T-0085 emsali): uydurma bir ad, hangi külliyata ait olduğu belirsiz bir okuma
hedefi yaratırdı. Ayrıca `literal_entity_mode` uyarı metnindeki **silinmiş checkpoint adı** kaldırıldı, uyarının
kendisi korundu. `tests/test_prompt_contract.py` bu yolu **açıkça** geçirir (`GUNCEL_VOCAB`) ve yeni kanarya
`test_canary_e_tokenizer_config_fail_closed` üç dalı sınar (iki durma + **pozitif kontrol**; pozitif kontrol
olmasa "her zaman duruyor" da kapıyı geçerdi).

**Ç3 sınandı:** varsayılana yaslanan **canlı çağıran** var mı? → Yok (testler açık yol veriyor).

---

## 3. (b) `scripts/run_agent_arena.py` — ölü checkpoint varsayılanı **+ ölçülmüş ek kusur**

**Kalem (b) buydu:** `--model default="data/kristal_model.pt"` kaldırıldı; varsayılan başka ad taşınmadı.

**Ölçüm sırasında çıkan ve kalemi BÜYÜTEN olgu (beyan edilir):** çağrı yeri
`AgentGateway.create_default(model_path=…, device=…)` — **`vocab_path` GEÇMİYOR**. T-0087 o parametreyi zorunlu
kıldığı için betik **zaten ölüydü**:

```
RuntimeError: DURDURULDU: vocab_path verilmedi. Varsayilan KALDIRILDI (T-0087)…
```

Yani "(b)'yi düzeltmek" tek satırlık varsayılan silme değil, **çağrı yerini yeniden yapılandırmaktı**: `--vocab`
bayrağı eklendi, ikisi birlikte ve **çağrı anında** zorunlu; ayrıca *verilen yol yoksa* da durulur (sözlük
okunamazsa `Vocabulary.load` boş sözlük üretebilir — ayrı bir sessizlik sınıfı).

**Ölçülen dallar:** argümansız `rc=2` (**iki** bayrağı adıyla sayar) · yalnız `--model` → `rc=2` · olmayan yol →
`rc=2` · **pozitif kontrol** (gerçek iki yolla) banner'ı bastı ⇒ kapı *yalnız durmuyor*, geçiriyor da.

**Dokunulmayan:** `kristal_bellek` (94. satır) yaşayan bir **koleksiyon adıdır**, soyağacı atfı değil.

---

## 4. (b2) `scripts/run_goal_pipeline.py` — 19 ölü atıf, gömülü varsayılanlar kaldırıldı

**Ölçüm (öncesi):** `ham=19 emekli=0 olu=19` — üç ad: `kristal_model.pt` (13), `kristal_carpenter_model.pt` (5),
`kristal_model_sft.pt` (1). Hepsi **bu betiğin kendi ürettiği** sonra da tükettiği yollardı.

**Yapılan:** dört yol parametresi **varsayılansız** zorunlu kılındı (`--base-model`, `--carpenter-model`,
`--sft-model`, `--vocab`); bütün üretim/tüketim siteleri bunlara bağlandı; `check_frozen_save_path` **6 çağrısı** ve
`--allow-frozen-write` semantiği **korundu**.

**Ölçüm sırasında çıkan ve beyan edilen üç ek olgu:**

1. **Üç derleme çağrısı donmuş çıktıya izinsiz yazıyordu.** `compile_jsonl_to_bin(...)` çağrılarında
   `allow_frozen_write` **geçilmiyordu**; hedefleri `data/*.bin` (donmuş) olduğu için betik **koşulsuz**
   `RuntimeError` verirdi. Bayrak (zaten var olan) artık geçiriliyor ⇒ davranış *"her hâlükârda dur"*tan
   *"operatör onayıyla geç"*e döndü; **yeni bir yetki icat edilmedi**.
2. **4. aşama sessizce kayboluyordu.** `train_dpo.py`'nin **kendi** varsayılanları silinmiş yollardır
   (`88-89`); verilmezse betik `Hata: SFT referans model dosyası … bulunamadı!` basıp **`return` eder (rc=0)** ⇒
   boru hattı 5. aşamaya DPO yapılmamış gibi devam ederdi. Yollar açıkça geçildi; **semantik korundu**
   (referans = base'in kopyası, aktif/kayıt = base).
3. **Derleme sözlüğü gömülüydü** (`data/rebuild/vocab_base_32852.json`, **TABAN 32.852**) ve checkpoint'lerle
   hizası **kapıya bağlı değildi** ⇒ sözlük parametreye çevrildi ve **tüm eğitim aşamalarına** (`--vocab`) geçilir.

**Dokunulmayan ve açık madde olarak bırakılan:** 1. aşamadaki `train.py` çağrısında `--pretrain` **yok**
(D1 sessiz sıfır-kayıp tuzağı); kusur mu, `data/train.bin` mi gerektirmiyor **ölçülmedi** ⇒ T-0067 gereği
tasarıma sessizce yazılmadı, buraya yazıldı.

---

## 5. (c) `USER_GUIDE.md` — çürütülmüş satır

* **105. satır (`test_model.py`) DEĞİŞTİRİLMEDİ** — iddiası yeniden ölçüldü ve **doğru** çıktı: `test_model.py:25`
  `data/vocab.json` sabit; `:38` `data/kristal_model.pt`; `:40-42` dosya yoksa **sessiz `return`**.
* **106. satır (`chat_prompt.py`)** ölçümle **çürüdü**: "`--vocab` bayrağı **yok** ⇒ sözlük değiştirilemez"
  iddiası artık yanlış — T-0087 ile **iki bayrak da var** ve ikisi de `sys.exit(2)` ile durur. Satır kaldırıldı;
  yerine **tarihli, ölçülmüş** durum yazıldı. Ayrıca tablonun başlığı ("sessizce yanlış sonuç vermezler, sesli
  dururlar") kendi satırıyla çelişiyordu (`test_model.py` **sessiz** dönüyor) ⇒ başlık damgalı hâle getirildi.
* **Kapsam genişlemesi (beyan):** §7'deki iki `run_arena` komutu da `--model`/`--vocab` içermiyordu ⇒ fail-closed
  kapıdan sonra **çalışmaz** komutlardı; düzeltildi ve seçenek listesine iki **ZORUNLU** bayrak eklendi.

---

## 6. (d) Denetim aracı (T-0086) — iddia ÇÜRÜDÜ, sözlük ÖLÇÜLDÜ

1. **Girintili blok kuralı — ÇÜRÜTÜLDÜ (düzeltme DEĞİL).** "Kural girintili emekli bloğunu görmüyor" iddiası
   ölçüldü: aracın **kendi fonksiyonu** (kopyalanmadan, import edilerek) girintili etiketli blokta koşuldu ⇒
   satırlar **ayıklandı**. Neden: `s = ham.strip()` zaten var. Fikstür bu ekseni artık **sınar** (öncesinde hiç
   sınanmamıştı) — `scratch/anka_r9_kanarya.py` sonuna girintili etiketli blok eklendi.
2. **Etiket sözlüğü — ÖLÇÜLDÜ, GENİŞLETİLMEDİ.** Kalıba uyan blok-başı = **8**; etiketli olup **kaçırılan** blok =
   **0** ⇒ ölçülmüş boşluk **yok**. Kalıbı genişletmek sayıyı yalnız `LIVE → ayıklandı` yönünde **düşürür**
   (canlıyı gizleme riski) ⇒ ilan edilmiş muhafazakâr yöne aykırı; **ayrı ve kapılı bir karar** olarak bırakıldı.
3. **Bayat negatif kontrol — teşhis yanlıştı, araç düzeltildi.** `NEG_KONTROL = "chat_prompt.py"` T-0087'den sonra
   bayatladı; araç bunu **"AŞIRI AYIKLAMA"** diye **yanlış teşhis** ediyordu. Artık `olu == 0` iki okumaya ayrılır
   (**kontrol bayatladı** / **aşırı ayıklama**) ve hüküm `AYIRT EDİLEMEDİ` olur, ayıklanan satırlar **elle
   incelenmek üzere basılır**. Kontrol, proje kararıyla değiştirilmeyen bir ölçüm betiğine (`t0027_measure_models.py`,
   26 canlı atıf) taşındı.

---

## 7. Kalem dışı ama ZORUNLU kalan düzeltme: `tests/test_compiler_entrypoints.py`

**Bu, kalem listesinde yoktu; (b2) onu KIRDI ve kırık bırakmak seçenek değildi.**

`test_ast_run_goal_pipeline_has_frozen_guards` (ii) maddesi şunu assert ediyordu:

```python
prev_args[0] == "data/kristal_model_sft.pt"     # ← SİLİNMİŞ artefaktın adı, sabit
```

Yani test, **ölü literali canlı tutan son kilit taşıydı**: (b2)'yi yapmak testi kırıyordu. Ölçüt **değiştirilmedi**,
yalnız **ad sabitlenmesi bırakıldı**: *guard'ın hedefi, kopyalamanın HEDEF argümanıyla aynı olmalı*.
`check_frozen_save_path` sayısı (`== 6`) ve `run_cmd` öncesi sıra denetimi **aynen** korundu.

* **Kanarya (ilan edilmiş, ayrı kapsam):** guard hedefi kopya hedefinden farklı yapıldı ⇒ test **DÜŞTÜ** (`rc=1`);
  gerçek ağaçta **GEÇTİ** (`rc=0`). Yani yeni ölçüt **ayırt edici**, vakum geçiş değil.
* **Dosya geri yükleme birebir doğrulandı:** mutasyon öncesi/sonrası `sha256` **aynı**
  (`4c31cc32651df3778b68a34c0e2c1c7f19b9efe52ade3985aa10eb82c1641ee2`).
* **Yönetişim:** bu dosya `writes[]`'te **yoktu** ⇒ düzenlemeden **ÖNCE** açık kiralama alındı
  (`bus_acquire_lease(["tests/test_compiler_entrypoints.py"], task_id="T-0088")` → `ok: true`) ve bu raporda
  **kapsam genişlemesi** olarak beyan edilir.

---

## 8. Çürütme maddeleri (tasarımda ilan edilmişti)

* **Ç1 (d) — ATEŞLEDİ.** "Girintili blok kuralı eksik" iddiası **çürüdü**; kayıt *çürütme* olarak yazıldı, düzeltme
  olarak değil. Fikstür yine genişletildi (eksen artık sınanıyor).
* **Ç2 (K3/K4) — KISMEN ATEŞLEDİ.** Ölü varsayılanı kaldırmak *tek başına* davranışı değiştirmiyordu: arena
  **zaten** T-0087'den beri duruyordu (ölçüldü). Doğru çerçeve "kusur düzeltildi" **değil**, *"ölü varsayılan
  kaldırıldı; durma yapılandırma anına taşındı ve çağrı yeri yeniden bağlandı"*.
* **Ç3 (K1) — ATEŞLEMEDİ.** Varsayılana yaslanan canlı çağıran bulunamadı.
* **Ç4 (K5) — ATEŞLEMEDİ.** `USER_GUIDE.md:105` iddiası ölçümle **doğrulandı**, değiştirilmedi.

---

## 9. Kendi kusurlarım (ölçüm öncesi/sonrası, aynen)

| # | Kusur | Etki / düzeltme |
|---|---|---|
| 1 | **Kaydedilmiş kanıtın üzerine yazdım.** Aracı `CIKTI_JSON`'u yamalamadan koştum ⇒ `data/eval/anka_r9_olu_atif_denetimi_2026-09-20.json` (07:45, 76 K) yeniden yazıldı (10:19, 78.808 B). Dosya **izlenmiyor** ⇒ eski baytlar **kurtarılamaz** | **Beyan edilir.** Ölçülmüş azaltım: yanındaki `.md` (07:45) 07:45 ölçümlerinin **tamamını** saklıyor (kanarya 4/1/3 · 3/2/1, NEGATİF `chat_prompt.py ham=11 olu=11`, kapsam tablosu) ⇒ **bulgu kaybı yok**, yalnız makine-okur kopya değişti. Sonraki tüm koşumlar yamalandı (`$TMPDIR`) |
| 2 | **Tasarım sayısını elle yazdım** ([[tasarim-sayisi-betikle-hesaplanmali]] ailesinin **3.** tekrarı): kanarya deltasını "+2 ham/+2 emekli" ilan ettim; ölçüm **+1/+1** verdi (sayaç yalnız **ölü yol taban adı taşıyan** satırı sayar) | Fikstür başlığı ve `KANARYA` sözlüğü ölçülen 5/2/3'e çekildi; hata **fikstür metnine** yazıldı (T-0087 K2(b) emsali: kusur **İLANDA**) |
| 3 | **Bayat kontrolü kod kusuru sandım:** araç "AŞIRI AYIKLAMA" bastı; gerçek olgu **kontrolün bayatlaması**ydı | Ayrı `AYIRT EDİLEMEDİ` dalı + ayıklanan satırları basma eklendi (§6/3) |
| 4 | **Seçenek metnim kusurluydu:** "`run_goal_pipeline.py` ölü **varsayılanları**" yazmıştım; ölçüm **0 ölü varsayılan, 18 ölü literal** verdi | Kapsam düzeltmesi olarak beyan edilir; sayı ölçümden gelir |
| 5 | **Sondam şişirdi:** etiket taramamda `.md` içinde de `#`'i yorum saydım ⇒ "eşleşmeyen aday" 46 çıktı; araç `.md` için `>` bloğu kullanıyor | Sonda artefaktı olarak beyan edilir; belirleyici sayı **"kaçırılan etiketli blok = 0"** |
| 6 | **4. aşama ilk düzeltmem semantiği kaydırıyordu:** `--active-model sft_model` yazdım; bu, DPO kaydını **kopyaya** yöneltirdi | `train_dpo.py:88-93` okundu; `--ref-model sft_model --active-model base_model` ile **özgün semantik** kuruldu |
| 7 | **Beyan ile ölçüm çelişti:** `scratch/anka_r9_kanarya.md` `writes[]`'te beyan edildiği hâlde **referans digest'ine göre net değişmedi** (birebir: `989977987be9b8e907876163b2af598c82f71abe747afb1ca08f6cb53ad57b46`). *Referans digest alınmadan ÖNCE* değişip değişmediği buradan **ölçülemez** — `scratch/**` izlenmiyor, eski hâlin digest'i yok | Doğru olan **ölçümdür**: ölçülebilir pencerede "değişti" demiyorum; `writes[]` bir **izin**tir, **sonuç** değil |

---

## 10. Açık maddeler (kapanmadı — beyan edilir)

1. **`README.md` bayat, DOKUNULMADI** (`writes[]` dışı): `194`, `197`, `200` satırlarındaki `run_agent_arena` komutları
   `--model`/`--vocab` içermiyor ⇒ **bugün çalışmaz**; `237` seçenek tablosu da aynı; `221` "sözlük olarak
   `vocab_base_32852.json` yükler" iddiası (b2)'den sonra **eksik** (yol artık parametre). Düzeltme metni hazır,
   uygulanmadı.
2. **`train_dpo.py` T-0087 sınıfının canlı 6. ve 7. sitesini taşır:** `:51` `data/vocab.json` (**BAYAT, 31.357**)
   sabit-kodlu ve `--vocab` kabul **etmez**; `:88-89` iki **ölü checkpoint varsayılanı** ve bunlar `return` ile
   **sessiz** (rc=0). Dosya `writes[]` dışı.
3. **`tests/test_prompt_contract.py` `env` fixture'ı hâlâ `data/vocab.json` yüklüyor** (31.357) — modül düzeyinde
   ve **açık** bir madde olarak yorumla işaretli; testin kendisi artık güncel sözlüğü açıkça geçiriyor.
4. **`chat_prompt.py:314-316` asimetrisi:** sözlük yolu **verilmezse** `sys.exit(2)` (sesli), ama **verilip de
   bulunamazsa** `return` (rc=0) ⇒ ikinci dal **sesli değil**. Dosya `writes[]` dışı.
5. **`run_goal_pipeline.py` 1. aşamasında `--pretrain` yok** (D1 tuzağı adayı) — kusur mu, veri kümesi mi
   gerektirmiyor **ölçülmedi**.
6. **`run_goal_pipeline.py` uçtan uca koşulmadı** (`data/qdrant_db` + Gemini/Ollama öğretmen + ağır checkpoint);
   ölçülen şey **yapılandırma kapısıdır**, uçtan uca başarı **değildir**.
7. **`scratch/**` `.gitignore`'da** ⇒ T-0086 aracı ve **kanarya fikstürleri sürümlenmiyor**: taze bir klonda araç
   KANARYA dosyalarını bulamaz ve doğru şekilde **VAKUM** diye durur. Aracın kalıcılığı **depo dışı** bir gerçeğe
   bağlıdır.
8. **Etiket sözlüğünü genişletme kararı** bilinçli olarak **alınmadı** (kapılı, §6/2).

---

## 11. Tam digest tablosu (önek DEĞİL)

| Dosya | Önce (T-0088 öncesi) | Sonra |
|---|---|---|
| `src/llm/prompt_contract.py` | `280c129d89f507383bab3c9ce2c907b55806b9ff4790630439874a13952a8f1a` | `56d0139d30484285e01471b56d6b8952e80842c52acb589d6080c4f991281c9a` |
| `tests/test_prompt_contract.py` | `e5eb105a056ea568962a7eafa647f503f2b6dde016152e3ed0125fffec3183a7` | `077a8c4f3ff0fdd863edf8503aac442bb511214e8955cee286b03b59066431d2` |
| `scripts/run_agent_arena.py` | `58087abbdf5b3905bb88f59edc28416b28d3233eec071cceb6e0bd0ce3d060bd` | `f1d8000e07ab0f97dfb910a595fda92fd405b0673089512dc2ca686d717cd962` |
| `scripts/run_goal_pipeline.py` | `2a88708edc0b9a8d156d070371fb43ba9e4f2b26014a72ae2d979f4c49c4d874` | `4c31cc32651df3778b68a34c0e2c1c7f19b9efe52ade3985aa10eb82c1641ee2` |
| `USER_GUIDE.md` | `4d5e8a992a4b8440478b55a434551fcc69a9cbfb00557c2b9d803a689ebbfe8e` | `d954e620bf4974806677abe4ff263ae11d2083149f5a95d42b9bbb2e4a5dd88a` |
| `tests/test_compiler_entrypoints.py` | `481138b318647be73484fa1c7ea7f839413b766157ae4478c0a5c401a7a70012` (`git show HEAD:`) | `423c99b70e4f77a692ff610dada36c1f47488a7e36d812bb2e849c631e3eee20` |
| `scratch/anka_r9_olu_atif_denetimi.py` | `0223938a7e6e669a9461d71031b514599f4c0b63e37e740038f4db388223ae27` | `090b0b8c7220c1c4139653d1081500c45e9a30eb4431a8dc9c6cc363534f0ea0` |
| `scratch/anka_r9_kanarya.py` | `935b83fccb72f3f7160d4433c45192d23239286bdcdcedec0150d3d1e64a93b2` | `86ce050ee07d9a1af4c62c1995ef6bf5a341b18e249bc4c3f0096c11aa13a5cd` |
| `scratch/anka_r9_kanarya.md` | `989977987be9b8e907876163b2af598c82f71abe747afb1ca08f6cb53ad57b46` | **aynı** (net değişim yok — §9/7) |
| `data/vocab.json` (DONMUŞ) | `4b40eebee9c996c8dd3150f6b3c8896de85764ee1cf0b08e6c636e5c002a0e69` | **aynı** (K8) |

**Denetim aracının ağaç ölçümü (KANARYA hariç):** ölü atıf **2438 → 2419** · KOD `156/2/154` → `138/3/135` ·
SCRATCH `406/0/406`, BELGE `199/0/199`, KAYIT `1679/0/1679` (**değişmedi**) · manifest digest
`4e1c4d51a896936909b1750b52088e8939e03e966b42fd8ea84934e0993e6e61` (birebir).

---

## 12. Hüküm

**Dört kalem de kapandı.** İkisi ilan edildiğinden **büyük** çıktı (arena'nın çağrı yeri T-0087'den beri ölüydü;
`run_goal_pipeline.py`'de 3 donmuş-yazma çağrısı ve sessiz dönen 4. aşama bulundu) — bu **kapsam genişlemesi**
ölçümle kanıtlanmış ve beyan edilmiştir. Kalem (d)'nin **girintili blok** iddiası **çürüdü**; doğru hâli araca
fikstür olarak eklendi.

**Bu görev COMMIT ETMEZ** (operatör yetkisinde) ve **eğitim koşusu yoktur**; hiçbir `.pt`/`.bin` üretilmedi.
`scratch/**` sürümlenmez (§10/7).
