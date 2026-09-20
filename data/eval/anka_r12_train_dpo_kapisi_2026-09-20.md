# T-0089 · `train_dpo.py` Yapılandırma Kapıları (fail-closed)

**Görev:** T-0089 · `from: claude` · `to: claude` · oluşturma `2026-09-20T07:46:22Z` ·
sahiplik `2026-09-20T07:46:30Z` · TTL 180 dk
**KAYNAK:** T-0088 kapanış raporu §10/2 (kanonik açık liste) ·
`data/eval/anka_r11_kucuk_borclar_2026-09-20.md`
**Operatör talimatı:** *"Sıradaki açık maddeyi kapat: train_dpo.py"*
**Damga:** bütün ölçümler 20 Eyl 2026, bu oturumda, bu makinede.

---

## 0. Hüküm (özet)

**K1…K8 kapılarının TAMAMI KALDI.** Sekiz kapı için ölçüm komutu ve çıktısı §3'te.
Dört çürütme maddesi (Ç1…Ç4) yanıtlandı; **hiçbiri çürütülmedi** (§4).
Kendi kusurlarım — **beş tanesi ölçüm kabında, ikisi yeni dosyalarımda** — §5'te, düzeltmeleriyle.

**Tek cümlelik sonuç:** `train_dpo.py` argümansız koşumda *bayat sözlüğü yükleyip 5.117 tercih
çiftini okuduktan sonra* "bulunamadı" deyip **`rc=0`** dönüyordu, yani DPO hiç yapılmadan
**başarı** raporluyordu; artık üç eksik bayrağı adıyla anıp **`rc=2`** ile, model yüklemeden
önce duruyor.

**İddia sınırı (K4).** Sözlük↔checkpoint kapısı **sessizliği değil OPAKLIĞI** giderir.
Ölçüldü (Kontrol A): `strict=False` şekil uyuşmazlığını **YUTMAZ**, `RuntimeError` fırlatır —
yani o sitede veri kaybı yoktu, yalnız eyleme dönük olmayan bir torch duvar metni vardı.
Kapıya "sessiz kırpmayı engelledi" payesi **verilmez**.

---

## 1. Ölçülen kusur (değişiklikten ÖNCE, `git HEAD`)

`git show HEAD:train_dpo.py > $TMPDIR/t0089/head_train_dpo.py`

| # | Kusur | Yer (HEAD) |
|---|---|---|
| 6. site | `vocab_path = 'data/vocab.json'` sabit-kodlu; betik `--vocab` **kabul etmez** ⇒ yol DEĞİŞTİRİLEMEZ | `:51` |
| 7. site | `sft_model_path = 'data/kristal_model_sft.pt'` · `active_model_path = 'data/kristal_model.pt'` — ikisi de **SİLİNMİŞ** | `:88`, `:89` |
| sessizlik | ref model yoksa `print` + `return` ⇒ **rc=0** | `:96-98` |
| sessizlik | DPO veri seti yoksa aynı sınıf | `:61-63` |
| soyağacı | `--active-model` verilip yol yoksa aktif model **ref'ten** başlatılır (adı verilenden FARKLI model) | `:108-116` |

### 1.1 Taban ölçümü — çıktının TAMAMI

```
$ PYTHONPATH=/Users/hakankilicaslan/Git/tr_llm venv/bin/python $TMPDIR/t0089/head_train_dpo.py
============================================================
 KRİSTAL-VEKTÖREL MİMARİSİ: DPO HİZALAMA EĞİTİMİ (STAGE-3)
============================================================
Cihaz: cpu
Sözlük Yüklendi. Kelime dağarcığı boyutu: 31357          <-- BAYAT sözlük, sessizce
DPO veri seti yükleniyor: data/pedagogy/dpo_all_tokenized.jsonl...
  -> Yüklenen Tercih Çifti Sayısı: 5117 (Uzunluk sınırı nedeniyle 1081 adet çift atlandı)

[1] Aktif ve Referans modeller yükleniyor...
Hata: SFT referans model dosyası 'data/kristal_model_sft.pt' bulunamadı!
rc=0                                                     <-- BAŞARI döndü
```

**Bu satır görevin çekirdeğidir:** çağıran boru hattı `rc=0` görür, DPO'nun hiç yapılmadığını
**hiçbir sinyalden** öğrenemez.

### 1.2 Aynı sınıfın ikinci ve üçüncü yüzü (ölçüldü)

```
H2  $ ... head_train_dpo.py --vocab data/rebuild/vocab_anka_r1_33114.json
    Sözlük Yüklendi. Kelime dağarcığı boyutu: 31357     <-- VERİLEN sözlük YOK SAYILDI
    Hata: SFT referans model dosyası 'data/kristal_model_sft.pt' bulunamadı!
    rc=0

H3  $ ... head_train_dpo.py --vocab yok.json --ref-model yok.pt --active-model yok.pt
    Hata: SFT referans model dosyası 'yok.pt' bulunamadı!      <-- üç yol da AÇIKÇA verildi
    rc=0                                                      <-- yine BAŞARI
```

H3 önemlidir: kusur "varsayılan kötü" değil, **yolun varlığının hiç denetlenmemesi**dir.

---

## 2. ÖNCE → SONRA (`rc` tablosu, hepsi ölçüldü)

| Dal | Komut | ÖNCE (HEAD) | SONRA |
|---|---|---|---|
| A argümansız | `train_dpo.py` | **0** | **2** — üç bayrağı adıyla anar |
| B kısmi | `--vocab <gerçek>` | **0** (yok sayıldı, 31357 yüklendi) | **2** — yalnız eksik İKİSİNİ anar |
| C yolu olmayan | `--vocab yok.json --ref-model yok.pt --active-model yok.pt` | **0** | **2** — `--vocab VERILEN YOL YOK: yok.json` |
| D pozitif | gerçek üçlü + uydurma `--data` | **koşulmadı** (¹) | **2** — `--data` adıyla ⇒ ilk üç kapı GEÇTİ |
| E K4 negatif | `--vocab data/vocab.json --ref-model data/anka_a1r.pt --active-model data/anka_a1r.pt` | **0** (sözlüğe kör) | **2** — `fark 1757` |
| F Ç1 pozitif | gerçek çift (33114↔33114) + **boş** `--data` | **koşulmadı** (¹) | **1** — kapı YOK, K4 GEÇTİ, veri döngüsünde `IndexError` |

**(¹) Neden koşulmadı:** HEAD'in kayıt yolu `dpo_model_save_path = active_model_path`'tir
(`:232`) — D'nin gerçek üçlüsünde bu **donmuş** `data/anka_a1r.pt`'dir. HEAD'de D veya F
koşmak **30 DPO adımı eğitip** donmuş yola yazmayı denemek olurdu. Bu görevde eğitim koşusu
**yasaktır** (§Sınırlar) ⇒ o iki hücre **bilerek ölçülmedi** ve buraya sayı **yazılmadı**.

### 2.1 D ve F'nin ayırt ediciliği

* **D**, ilk üç kapının *gerçek* değerlerle **geçtiğini** kanıtlar: durma mesajı `--data`'yı
  anar, yani `--vocab/--ref-model/--active-model` kapıları ateşlemedi. Aksi hâlde mesaj
  onlardan birini anardı.
* **F**, K4'ün *(§4 Ç1)* pozitif dalıdır: `Sözlük Yüklendi. Kelime dağarcığı boyutu: 33114`
  → iki model yüklendi → *"Modeller hazırlandı. Referans model donduruldu."* → eğitim
  döngüsüne girdi. **Hiç `DURDURULDU` basmadı.** Kesilme nedeni kapı değil, **veri yoludur**
  (boş dosya ⇒ `IndexError: index 0 is out of bounds for axis 0 with size 0`).
  Yani K4 gerçek çifti **kabul ediyor**.

---

## 3. İlan edilen kapıların ölçümü

Kapılar ölçümden **ÖNCE** ilan edildi (T-0089 `spec`), sonra **değiştirilmedi**.

| Kapı | Ölçüm | Sonuç |
|---|---|---|
| **K1** sabit sözlük yok, `--vocab` zorunlu, yol yoksa rc=2, başka ada taşınmaz | `grep -n "data/vocab\.json" train_dpo.py` → **2**: `:126` EMEKLİ yorumu (ayıklandı) · `:150` DURDURULDU mesaj metni. Canlı **atama = 0**; tek atama `:137 vocab_path = argv_deger(("--vocab",))`. B/C dalları rc=2 | **KALDI** |
| **K2** iki model yolu zorunlu, ölü varsayılan yok, canlı `kristal_model` literali = 0 | Araç (T-0086): `train_dpo.py` **ham=3 emekli=3 ölü=0** — üç anma da EMEKLİ bloğunda (soy ağacı kanıtı), canlı kodda 0 | **KALDI** |
| **K3** üç sessiz durma da sesli | A (üçü eksik) · B (ikisi eksik) · C/E (yol yok) hepsi rc=2. Sessiz soyağacı dalı **kaldırıldı** | **KALDI** |
| **K4** sözlük↔checkpoint satır sayısı kapısı | Negatif: E → rc=2 `fark 1757`. Pozitif: F → GEÇTİ. Saf fonksiyon testi iki dalı da ölçer | **KALDI** (iddia sınırı §0) |
| **K5** donmuş-yazım semantiği korunur, AST testi DEĞİŞMEDEN geçer | `--save-path/--output-model` varsayılanı `active_model_path` (**özgün semantik korundu**); `check_frozen_save_path` + `--allow-frozen-write` yerinde. `git diff HEAD -- tests/test_compiler_entrypoints.py` → AST testi fonksiyonuna **0 +/- satır**; `pytest tests/test_compiler_entrypoints.py` → **12 passed** | **KALDI** |
| **K6** boru hattı `--vocab` geçirir, kapı çalışır, yanlış cümle değişir | `run_goal_pipeline.py:352` `--vocab vocab_path` geçirir; argümansız koşum **rc=2** (`--base-model ve --carpenter-model ve --sft-model ve --vocab ZORUNLUDUR`). USER_GUIDE yanlış cümle damgalı ölçümle değiştirildi; README satırı güncellendi; TARİHSEL bloklar **silinmedi** | **KALDI** |
| **K7** pytest sapması | **1 failed, 216 passed** → **1 failed, 223 passed** · sapma **TAM +7** = yeni dosyadaki 7 test. Düşen tek test tabanda da düşen sandbox soket yasağı (`socketserver.py:478 PermissionError`) | **KALDI** |
| **K8** donmuş 10 desen, eğitim yok | §3.1 | **KALDI** |

### 3.1 K8 — donmuş yüzey (yöntem: baseline'daki YOLLAR yeniden hash'lendi)

Komut sırasını yeniden üretmek kırılgan olurdu (zsh glob sırası yeniden kurulamaz);
onun yerine **baseline'ın kendi yol kümesi** kanonik alındı.

```
baseline: $TMPDIR/t0089/once_donmus.txt  (28 satır)
          sha256 79f05cea4f4a2844344d6ac7d8f209ed3d9fc19e19808e38e7b43e6548b2579e
  değişen digest : 0
  silinen dosya  : 0
  bağımsız numaralandırma: data/*.pt = 2, data/*.bin = 23 (toplam 25 = baseline 25)
  yeni eklenen   : []        kaybolan : []
```

Eşiğe bağlı yazma denetimi — **eşik = görevin kendi `claimed_at`'i (UTC)**:

```
data/realistic_rag/**      7 dosya   0 yazılan
data/b1_5_splits/**       16         0
data/pedagogy_canonical/** 6         0
data/lexicon/**            3         0
src/compiler/**           12         0
data/*.pt                  2         0
data/*.bin                23         0
data/vocab.json            1         0
data/vocab_entity.json     1         0
src/llm/tokenizer.py       1         0
                    TOPLAM 72 dosya   0 yazılan
```

`data/vocab.json` `4b40eebee9c996c8dd3150f6b3c8896de85764ee1cf0b08e6c636e5c002a0e69`
ve `data/anka_a1r.pt` `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293`
**önce = sonra**. **Eğitim koşusu YOK; hiçbir `.pt`/`.bin` üretilmedi.**

---

## 4. Çürütme maddeleri (ölçümden ÖNCE ilan edilmişti)

### Ç1 (K4) — *"Gerçek çiftte GEÇİYORSA ve uyuşmayanda DURUYORSA ayırt edicidir; ikisinde de duruyorsa VAKUMDUR."*
**ÇÜRÜTÜLMEDİ — kapı ayırt edici.**
* **Pozitif dal (CLI, F):** gerçek 33114↔33114 → K4 **geçti**, eğitim döngüsüne ulaşıldı.
* **Negatif dal (CLI, E):** 31357↔33114 → **rc=2**, `fark 1757`.
* **Saf fonksiyon:** `test_sozluk_checkpoint_uyumu_uyusani_gecirir_uyusmayani_durdurur` iki dalı da ölçer.
* **SINIR:** uçtan uca **eğitim** koşumu yapılmadı; F, veri döngüsünün ilk satırında kesildi.

### Ç2 (K3) — *"`--active-model` verilip yol yokken ref'ten başlatmayı kullanan CANLI çağıran varsa fail-closed YANLIŞTIR."*
**ÇÜRÜTÜLMEDİ.** Çağrı yüzeyi ağaç genelinde tarandı; **iki** site var:
* `scripts/run_goal_pipeline.py` — `shutil.copyfile(base_model, sft_model)` (`:346`) 4. aşamadan
  (`:356`) **ÖNCE** koşar ve `base_model` yoksa orada `FileNotFoundError` ile düşer ⇒ 4. aşama
  hiç başlamaz ⇒ **sessiz soyağacı bu boru hattında ATEŞLENEMEZDİ**. Ayrıca `:355`
  `check_frozen_save_path(base_model, ...)`.
* `scripts/retrain_clean_models.py:127` — `train_dpo.py`'yi **hiç yol bayrağı olmadan** çağırır;
  betik bütünüyle **ölü** (tek anan yer bir AST testi, `main()` çağrılmıyor) ve kendi
  gövdesinde zaten ölü yollar taşır. **Değiştirilmedi** (açık madde).
* **"ref'ten başlat" niyeti kaybolmadı:** aynı yol iki bayrağa da verilir (açık niyet).

### Ç3 (K6) — *"4. aşamaya `--vocab` eklendikten sonra yapılandırma kapısı DÜŞERSE boru hattı kırılmıştır."*
**ATEŞLENMEDİ.** `venv/bin/python scripts/run_goal_pipeline.py` → **rc=2**, mesaj dört bayrağı sayar.

### Ç4 (K7) — *"pytest sapması beyan edilen yeni test sayısından farklıysa düzeltme bir şeyi kırmıştır."*
**ATEŞLENMEDİ.** Sapma **TAM +7**; yeni dosya **7 test** içerir. Düşen tek test tabanla aynıdır.

---

## 5. Kendi kusurlarım (ölçüm çürüttü — kayda geçer)

### 5.1 Ölçüm kabında (dördü)

| # | Kusur | Etki / düzeltme |
|---|---|---|
| **C1** | HEAD kopyasını `$TMPDIR`'de **`PYTHONPATH` olmadan** koştum. `sys.path.append(dirname(__file__))` depoyu değil `$TMPDIR`'i ekler ⇒ `ModuleNotFoundError: No module named 'src'` ⇒ **rc=1** | Bu, **"taban rc=0"** iddiamı yanlışlıkla **çürütüyordu**. Dikkatsiz bir koşum tabanı yanlış yazardı. Kap `PYTHONPATH=<depo>` ile kuruldu; gerçek taban **rc=0** |
| **C2** | `find … -newermt "2026-09-20 00:00"` donmuş denetimi **1 YANLIŞ POZİTİF** verdi: `data/anka_a1r.pt` yerel mtime `02:55` (UTC+3) ama **UTC damgası `2026-09-19T23:55:08Z`** = T-0080 Faz 5 kaydı; T-0089 `07:46:30Z`'de sahiplenildi ⇒ dosya **7 sa 51 dk ÖNCE** yazılmış | Eşik görevin `claimed_at`'ine (UTC) bağlandı ⇒ **0**. Yerel damgayla UTC damgayı kıyaslamak yanlış "bayat/yeni" verir |
| **C3** | Yazımdan sonra araç `train_dpo.py` için **`olu=2`** ölçtü | EMEKLİ bloğumun **BAŞLIK** satırı etiketsizdi (etiket **3.** satıra düşmüştü). Aracın ilan edilmiş kuralı yalnız başlığı etiketli bloğu ayıklar. Etiket başlığa taşındı ⇒ **`olu=0`**. **Her iki sayı da burada raporlanır** (etiketleme gizlenmez) |
| **C4** | Etiket kuralını **varsaydım** | Aracın kendi kodu okundu (`ETIKET_BELGE = r"TAR[İI]HSEL"`, blok = ardışık `>` satırları, etiket **başlıkta**) ve **dört dalda** sınandı: etiketli→ayıklandı · etiketsiz→yakalandı · etiket blok DIŞINDA→yakalandı · etiket 2. satırda→yakalandı. Araç **kör değil** |

### 5.2 Yeni dosyalarımda (ikisi de kendi ürettiğim canlı ölü-atıf)

| # | Kusur | Düzeltme |
|---|---|---|
| **D1** | **YENİ** test dosyamın docstring'i ölü artefakt adını **etiketsiz** andı ⇒ araç KOD kapsamında `tests/test_train_dpo_entrypoint.py` için **`olu=1`**. T-0086 aracının yakalamak için var olduğu sınıfı **ben ürettim** | Ad KOD yüzeyinden **çıkarıldı**; kanıt (tam komut + stderr + rc) bu rapora, **KAYIT** kapsamına taşındı. Araç ölçümü: dosya artık tabloda **yok** (`olu=0`) |
| **D2** | `USER_GUIDE.md`'ye eklediğim damgalı not ölü yolu **etiketsiz** andı ⇒ BELGE'de **+1** | Blok **gerçekten** tarihsel kayıt olduğu için başlığına `TARİHSEL` etiketi kondu (adı silmek yerine — ad okuyucu için taşıyıcı). BELGE 202 → **201** |

**Ağaç etkisi (T-0086 aracı, damgalı):** KOD **133 → 130** · BELGE **202 → 201** ·
hedef dosyalarımın dördünde **0 canlı ölü-atıf** (`README.md`'deki 2 **mirastır**; oraya
hiç eklemedim). Yani görev ağacı **temizledi**, kirletmedi.

**Not ([[emekli-notu-sayaci-sisirir]]'in kaçınılmaz yönü):** *bu rapor* kaldırdığım artefaktı
adıyla anmak zorundadır ⇒ KAYIT kapsamının sayacını **kendi satırları kadar** şişirir. Bu bir
regresyon değil, kanıt yüzeyinin tanımıdır; araç KAYIT'ı ayrı kapsamda basar.

---

## 6. Şartname beyanımın ölçümle düzeltilmesi

Şartname §"sessizlik" için *"`--active-model` YOKSA aktif model **sessizce** ref'ten başlatılır
(… uyarı yok)"* yazıyordu. **Ölçüm:** eski `else` dalı **bir bilgi satırı basıyordu** —
`"  * Aktif model referans modelden başlatılıyor: <yol>"` (HEAD `:115`).

⇒ Durum **"sessiz"** değil, **"uyarısız/etiketsiz"**dir. İddia ölçüme göre daraltıldı.
(Çürütülmeyen kısım: yüklenen model, **adı verilenden FARKLI**dır ve bunu söyleyen bir **uyarı
yoktu**.) [[denetim-kapsami-iddiadan-dar]] ailesi: sayı doğruyken *iddia* genişti.

---

## 7. Değişen dosyalar ve TAM digest tablosu

Önek **kullanılmadı** ([[hash-iddialari-tam-digest-ile-denetlenir]]).

| Dosya | ÖNCE (HEAD) | SONRA |
|---|---|---|
| `train_dpo.py` | `70d9ea96256a17d39fa9da7588d668f2dfed8bf1f5a01b2250e991ae843caa0c` | `e94c4ff1099d916d7f2b042ff67d705ae139c4d2902324b21813c44262531a50` |
| `tests/test_train_dpo_entrypoint.py` | — (yeni dosya) | `9f59eb91a4c67699e5f55797a412ef705104321374d2e28efa32c1ccf7a7a843` |
| `scripts/run_goal_pipeline.py` | `2a88708edc0b9a8d156d070371fb43ba9e4f2b26014a72ae2d979f4c49c4d874` | `8582f6042730948faa8c2200bdfc54fbce6c99954ab320793b5558b077b3b30c` |
| `USER_GUIDE.md` | `4d5e8a992a4b8440478b55a434551fcc69a9cbfb00557c2b9d803a689ebbfe8e` | `1f3d6f0105467b6b65f51e08e8befe96c7f57b8097bf851c9df06032722f6975` |
| `README.md` | `818f59053665b75c500b7a5c4e7a9eb2105d63958fc1aeefaa2094a9b274d068` | `abf618385376d51c8bb93701ec4bf8682b1b6dd16f3c3d37637dcfb7b0b409d1` |

Digest'ler **bütün yazmalar bittikten SONRA** alındı; bu raporun kendi digest'i kapanış
mesajında ve not dosyasında verilir.

**Dokunulmayanlar (beyan):** `scripts/retrain_clean_models.py` (bütünüyle ölü — açık madde),
`CHANGELOG.md` ve `wiki/log.md` (TARİHSEL kayıt), `tests/test_compiler_entrypoints.py`
(T-0089 `writes[]`'inde **yok**; `M` durumu T-0088'den devralındı — AST testi fonksiyonuna
**0 satır** dokunuldu).

---

## 8. Açık kalan işler (kapanmadı — beyan edilir)

1. **`scripts/retrain_clean_models.py` bütünüyle ölü ve `train_dpo.py`'yi yol bayrağısız
   çağırıyor.** Bugün zaten kırılır (kendi gövdesinde ölü `data/kristal_model*.pt` yolları var).
   T-0089 kapsamı dışı bırakıldı; **değiştirilmedi**.
2. **AdamW momentleri kaydedilmiyor** (T-0077'den devralınan açık madde) — DPO koşumundan
   devam optimizer durumunu kaybeder.
3. **`README.md`'de 2 canlı ölü-atıf mirası** (T-0089'da sayısı **artırılmadı**, ama
   temizlenmedi).
4. **Uçtan uca DPO eğitim koşumu ölçülmedi** — kapılar ölçüldü, **koşum ölçülmedi**. Bu görev
   eğitim koşusu yasağı taşıdığı için **bilerek** yapılmadı; kapıların pozitif dalı bunu
   telafi etmez.

---

## 9. Yönetişim

`writes[]` (8, KISA ve iki yönlü): `train_dpo.py` · `tests/test_train_dpo_entrypoint.py` ·
`scripts/run_goal_pipeline.py` · `USER_GUIDE.md` · `README.md` ·
`data/eval/anka_r12_train_dpo_kapisi_2026-09-20.md` · `…json` · `.agent-bus/notes/T-0089.md`.
Kiralar: `bus_acquire_lease`, 8 yol, hepsi `T-0089`/`claude`. Donmuş **dosya** kiralanmadı
(kiralamaya konu donmuş dosya **yok**; donmuş desenlere **hiç yazılmadı**).
`data/**` salt-okunur; tek yazma `data/eval/`dir. **Commit YOK, push YOK, `git add -A` YOK**
(operatör yetkisi). Eğitim koşusu **yok**.
