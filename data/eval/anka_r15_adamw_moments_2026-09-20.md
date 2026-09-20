# T-0092 · AdamW momentleri kaydediliyor — kusur ÖLÇÜLDÜ, kapatıldı, maliyeti BEYAN edildi

**Görev:** T-0092 · `status: done`
**Damga (ölçüldü — betik hesapladı, elle yazılmadı): `2026-09-20T11:52:54Z`**
**Yetki:** operatör onayı — *"Onay bekleyen işler için devam et"* ⇒ dört iş akışından **(4) AdamW
momentleri**. Kaynak: T-0080 kapanış raporu **§8/3** (*"AdamW momentleri kaydedilmiyor ⇒
`anka_a1r.pt`'den devam eden koşum optimizer durumunu kaybeder"*), özetten değil **tam metinden**
okundu.

> **Donmuş yüzeye YAZMA YOK.** `data/**` donmuş desenleri, `src/compiler/**`,
> `src/llm/tokenizer.py` — hiçbirine yazılmadı. `data/anka_a1.pt` ve `data/anka_a1r.pt` yalnız
> **okundu** (§9). Bu görev **model eğitmedi**; koşumlar 2 adımlık CPU sözleşme sınamalarıdır.

---

## 1. Ölçümden ÖNCE ilan edilen karar ve çürütme maddeleri

**İlan:** `train.py`'ye **yan dosya (sidecar)** mekanizması eklenecek — `--save-optimizer` ile
`<save-path>.opt.pt`, `--load-optimizer` ile geri yükleme, ve yan dosyanın **hangi modele ait
olduğunu** doğrulayan bir **eşleşme kapısı** (`model_sha256`). **Varsayılan KAPALI**: bayrak
verilmezse davranış bit-bit aynı kalır.

**Çürütme maddeleri (ölçümden ÖNCE yazıldı):**

1. *"Varsayılan koşumda yan dosya oluşur **veya** model digest'i değişirse ⇒ değişiklik GERİ
   ALINIR."* — **ateşlenmedi** (§4/G1).
2. *"G4/G5'in DURAN dalı ölçülemezse ⇒ o kapı İDDİA EDİLMEZ, rapora 'ölçülemedi' yazılır."* —
   **ateşlenmedi**; dört duran dal da ölçüldü (§4).
3. *"G3'te kayıt sayısı 101 çıkmazsa ⇒ mekanizma çalışmıyor demektir; 'kısmen çalıştı' diye
   yazılmaz."* — **ateşlenmedi**; gerçek programda **101** çıktı (§4/G3).

---

## 2. Ölçüm 0 — Kusur: varsayım değil, ÖLÇÜM

Sonda `$TMPDIR/t0092_probe2.py` (v2; v1'deki kendi araç kusurum §6/1'de beyan edildi).
Referans: 2 AdamW adımı sonrası durum.

| Ölçüm | Sonuç |
|---|---|
| Referans `optimizer.state` kaydı | **101** |
| **(A) MEVCUT devam** (yalnız `model.state_dict()` yazılmış) | **0 kayıt ⇒ momentler KAYIP** |
| **(B) Yan dosyalı devam** | **101 kayıt** |
| Gidiş-dönüş bit-özdeş mi? | **True** (farklı alan: **[]** — hiçbiri) |
| `param_group` | lr `0.001==0.001` · wd `0.01==0.01` |

**Kaybın ÖLÇÜSÜ** (aynı partiden tek adım; kayıp güncellemeden **ÖNCE** ölçülür ⇒ fark yalnız
güncellemededir):

| Hal | Kayıp | `||delta||` | oran |
|---|---|---:|---:|
| (A) MOMENTLİ (referans) | 0,337373 | **4,735192** | 1,0000× |
| (B) MOMENTSİZ (mevcut davranış) | 0,337373 | **8,194721** | **1,7306×** |
| (C) Yan dosyadan moment | 0,337373 | **4,735192** | 1,0000× |

**(C) ile (A) farkı TAM `0,000e+00`** ⇒ gerçek geri yükleme, yaklaşıklık değil.
Yani mevcut davranış, devam koşumunun **ilk güncellemesini %73 daha büyük** yapıyordu.

**M5 — BAYAT İDDİA ÇÜRÜDÜ:** *"checkpoint'lerin %21'i mask tamponu"* iddiası **bu artefakt için
geçersiz**: `data/anka_a1r.pt` (101 anahtar) içinde `mask`/`cached` anahtarı **YOK** (ölçüldü).
Checkpoint bugün **saf parametredir**.

---

## 3. Yapılan değişiklik (tasarım gerekçeli)

**Yan dosya seçildi, tek-dosya biçimi DEĞİL.** Gerekçe: `torch.save(model.state_dict())`
biçimini değiştirmek mevcut **tüm** yükleyicileri (`chat_prompt.py`, `run_agent_arena.py`,
`test_model.py`, `src/llm/prompt_contract`) kırardı. Yan dosya model biçimine **dokunmaz** ⇒
**sıfır yükleme kırılması**.

| Öğe | Ne yapar |
|---|---|
| `--save-optimizer` | `<save-path>.opt.pt` yazar: `optimizer.state_dict()` + `adim` + `model_sha256` |
| `--load-optimizer` | Yan dosyayı yükler; **digest uyuşmazsa rc=2 ile DURUR** |
| Eşleşme kapısı | Yüklemede model dosyasının digest'i **yeniden hesaplanır** ve karşılaştırılır |
| Yan dosya YOK + bayrak YOK | Koşum **devam eder** ama stderr'e **açık uyarı** basar (sessiz düşme yok) |
| Yan dosya YOK + `--load-optimizer` | **rc=2 ile DURUR** |
| `check_frozen_save_path` | **Yan dosya için de** ve `torch.save`'dan **ÖNCE** çağrılır |
| Yazma sırası | **MODEL ÖNCE, YAN DOSYA SONRA** (bilinçli — §7/2) |

**Varsayılan KAPALI.** `save_optimizer`/`load_optimizer` `sys.argv`'de **açık bayrak aranarak**
belirlenir; varsayılan `False`.

**Ek olarak `--seed N` eklendi (ölçüm aracı).** Gerekçesi ölçüldü: `train.py`'de **hiç tohum
kurulmuyordu** ve `KristalDataset.get_batch` global `torch.randint` kullanıyor ⇒ **aynı komut
farklı ağırlık üretiyordu** ve *"değişiklik davranışı değiştirdi mi?"* sorusu **ölçülemezdi**.
G1'in (b) ayağı bu bayrağa dayanır. Bayrak eklenmeseydi G1 **iddia edilemezdi**.

---

## 4. Kapılar — DURAN ve GEÇEN dal BİRLİKTE (vakum kapı yasak)

Tümü `tests/test_train_optimizer_state.py` içinde, **gerçek `train.py` süreci** üzerinden
(`subprocess`), 2 adımlık CPU koşumlarıyla. **10/10 GEÇTİ** (53,28 sn).

| Kapı | Ne ölçer | Dal | Sonuç |
|---|---|---|---|
| **G0** | `--seed` aynı tohumda AYNI, **farklı tohumda FARKLI** ağırlık | geçen **+** ayırt eden | ✓ |
| **G1** | `--save-optimizer` YOK ⇒ yan dosya OLUŞMAZ; sıfırdan koşumda uyarı basılmaz | geçen | ✓ |
| **G2** | Yan dosya OLUŞUR; `model_sha256` = modelin **GERÇEK** digest'i; `adim`=2; durum **boş DEĞİL** | geçen | ✓ |
| **G3** | Devam koşumu momentleri **yükler** | geçen | ✓ |
| **G4a** | Yan dosya yok + bayrak yok ⇒ **DURMAZ**, stderr'e uyarır, stdout'a **basmaz** | duran-değil | ✓ |
| **G4b** | Yan dosya yok + `--load-optimizer` ⇒ **rc=2**, model yazılmaz | **duran** | ✓ |
| **G5** | Digest uyuşmuyor ⇒ **rc=2** | **duran** | ✓ |
| **G5b** | `model_sha256` alanı **hiç yok** ⇒ **rc=2** ("doğrulanamadı" ≠ "doğru") | **duran** | ✓ |
| **G6a** | Göreli donmuş model yolu ⇒ **DURUR**, artefakt **oluşmaz** | **duran** | ✓ |
| **G6b** | Model yolu serbest ama `.opt.pt` donmuş ⇒ **DURUR** (mesaj **yan dosya** yolunu anar) | **duran** | ✓ |

**G0'ın İKİ dalı birlikte gerekli:** yalnız *"aynı tohum → aynı"* ölçmek **vakum kapıdır** —
hiçbir şey yapmayan bir bayrak da o dalı geçerdi. Ayırt eden dal ölçüldü:

```
d1 (seed 1234) = 48b6da7231af67480727904c565e0ef3338256a4a5f0428095a7d6e4dd94c423
d2 (seed 1234) = 48b6da7231af67480727904c565e0ef3338256a4a5f0428095a7d6e4dd94c423  ⇒ AYNI
d3 (seed  999) = 81082fd6e7792e3003dc011d8931237a75d166215e811b25a8533356a2e7d920  ⇒ FARKLI
```

**G6b NEDEN AYRI BİR KAPIDIR (ve gereksiz tekrar DEĞİLDİR):** `data/x_v2` (uzantısız) donmuş
desenlerin **hiçbirine** girmez, ama `.opt.pt` eki onu `data/*.pt` desenine sokar. Ölçüldü —
kapı mesajı **yan dosya** yolunu anar:

```
RuntimeError: Donmuş yola yazma engellendi: data/anka_t0092_sonda_v2.opt.pt (allow_frozen_write=False)
```

⇒ yan dosya kontrolü **erişilebilir ve ayırt edici**dir.

### G1 — "varsayılan bit-bit aynı" iddiasının ÖLÇÜLEN SINIRI

Model digest'i bayrak AÇIK ve KAPALI koşumlarda **birebir aynı**:

```
ON  (--save-optimizer) m.pt = 48b6da7231af67480727904c565e0ef3338256a4a5f0428095a7d6e4dd94c423
OFF                    m.pt = 48b6da7231af67480727904c565e0ef3338256a4a5f0428095a7d6e4dd94c423
```

**Ama "log bit-bit aynı" DEĞİLDİR ve öyle yazılmaz.** `diff` üç satırın değiştiğini gösterdi;
üçü de **anlamsal değil**: (i) adım süresi `0.29s` vs `0.27s`, (ii) toplam süre `1.16` vs `2.37`
sn (yan dosya yazımının gerçek maliyeti), (iii) **kayıt yolunu taşıyan** son satır. Kayıp
satırları (`10.5462` · `9.6450`) ve tüm diğer satırlar **aynı**. Yeni `[ckpt]` satırı
**basılmadı**, çünkü `--save-every` verilmemişti ⇒ varsayılan log **şekil olarak** değişmedi.

---

## 5. Ölçülen maliyet (istenen kalem)

| Kalem | Ölçüm |
|---|---|
| Model dosyası | 373.727.988 bayt = **356,4 MiB** |
| **Yan dosya** | 747.482.741 bayt = **712,9 MiB** = modelin **%200,01**'i |
| Neden tam %200 | AdamW parametre başına **2 moment** (`exp_avg`, `exp_avg_sq`), ikisi de float32; 93.424.986 parametre |
| **`sha256_file` süresi** | **0,160 sn** (2232 MiB/sn) — model başına, **her yüklemede** bir kez |
| Koşum süresi etkisi | 1,16 sn → 2,37 sn (2 adımlık sözleşme koşumu; eğitimde ihmal edilebilir) |

**Yorum (iddia sınırı):** sha256 maliyeti **ihmal edilebilir**; asıl maliyet **disk ayak izi**dir —
her checkpoint için **iki kat** yer. Bu bir **tercih**tir, ölçülmüş bir bedeldir.

---

## 6. Kendi kusurlarım (ölçüm kabı ve test tasarımı)

1. **v1 sondam SAHTE bir bulgu üretti (önceki oturum).** Optimizer durumunu **parametre NESNESİ
   kimliğiyle** karşılaştırdım (`o3.state.get(p1)`); `optimizer.state` nesne anahtarlıdır ⇒ her
   arama `None` döndü ve *"gidiş-dönüş bit-özdeş DEĞİL"* **yanlış** sonucu çıktı. Doğru yol
   `state_dict()` (indeks anahtarlı). v2 düzeltti, sonuç **True**. v1'in "False"u **bulgu değil,
   aracımın kusuruydu**.
2. **Kabuk ölçümümü kirletti: zsh unquoted değişkeni kelimelere AYIRMAZ.** Donmuş kapı ölçümünü
   `venv/bin/python train.py $C ...` ile koştum; `$C` **tek bir argüman** olarak geçti ⇒ koşum
   **varsayılan** bayraklarla (vocab 32.852, 100 adım, SFT) çalıştı. Kapı sonucu yine geçerliydi
   (kapı her şeyden önce çalışır) **ama diğer bayrakların uygulandığını sanmıştım**. Dizi
   (`C=(...)` + `"${C[@]}"`) ile yeniden ölçtüm. Ders: *"komut bir şey yazdırdı" ≠ "ölçtü"*.
3. **Kanarya sayacım hiçbir şey doğrulamıyordu.** Güçlendirdiğim AST testini çapraz denetlemek
   için yazdığım bağımsız sayaç, `torch.save` bir `ast.Attribute` olduğu için adı yalnız `torch`
   okuyordu ve ekrana **`save=0`** bastı ⇒ "bağımsız doğrulama" **boştu**. `.attr` eklenince
   `guard=2 save=3` / `3,3` / `2,2` çıktı. Kusur **aracımdaydı**, görevde değil.
4. **İlk G6 testim donmuş yüzeye 356 MB yazdı.** Testi **mutlak** yolla kurdum; donmuş kapı
   mutlak yolda ateşlenmediği için koşum **tamamlandı** ve `data/anka_t0092_sonda.pt` +
   `.opt.pt` **frozen `data/` altına yazıldı**. Artefaktlar `finally` ile **silindi**
   (ölçüm: `data/` altında kalıntı **YOK**; `git status`'ta iz yok) — ama **test tasarımım**
   kapının çalışacağını **varsaymıştı**. Göreli yola çevrildi; orada kapı `torch.save`'dan
   **önce** durduğu için koşum maliyeti de **sıfırdır**. Bu kusur, §7/1'deki gerçek bulguyu
   **ortaya çıkardı** — yani zarar vermeden işe yaradı, ama **şansla**.
5. **Dosya digest'i karşılaştırmam SAHTE determinizm hatası üretti.** Aynı tohumlu iki koşumu
   **farklı dosya adlarıyla** (`a.pt` / `a2.pt`) kaydedip `cmp`/sha256 ile karşılaştırdım ve
   "FARKLI" gördüm. Sebep: `torch.save` bir **ZIP** kabıdır ve **arşiv üye adları dosya adını
   taşır** (`a.pt/data/0` vs `a2.pt/data/0`). **Tensör tensör** karşılaştırma: **101/101
   özdeş**, ZIP CRC'leri **aynı**. **AYNI taban adla** yeniden koşunca digest'ler **birebir
   eşit** çıktı. Kural [[hash-iddialari-tam-digest-ile-denetlenir]] ailesinin bir üyesi:
   **kabuk farkı, içerik farkı değildir**.
6. **`writes[]` beyanım GEÇ kaldı.** `tests/test_compiler_entrypoints.py`'yi, görev beyanımda
   **yokken** değiştirdim (yazma kira kapsamındaydı — `tests/` dizini kiralı — ama **beyan**
   eksikti). Gerekçe ölçülerek doğdu: testteki `len(torch.save) == 1` **sayı sabitlemesi**
   T-0092'nin ikinci kayıt yeriyle düştü. Spec **sonradan düzeltildi** (oku→yaz→**tekrar oku**),
   `spec_duzeltmeleri[0]` altında **önceki `writes[]` ve önceki spec sha256'sı saklandı**, ve
   durum `bus_send` ile bildirildi. Sıra yanlıştı; kayıt düzeltilmedi, **büyütüldü**.

---

## 7. Bu görevde ölçülen YENİ bulgular

### 7/1 · Donmuş kapı MUTLAK yollarda FAIL-OPEN (KAPSAM DIŞI — DÜZELTİLMEDİ)

**Ölçüm (üç dal, temiz):**

| Dal | `--save-path` | rc | Donmuş kapı | Artefakt |
|---|---|---|---|---|
| A | `data/anka_t0092_sonda.pt` (göreli) | 1 | **ATEŞLENDİ** | yok ✓ |
| B | `data/anka_t0092_sonda_v2` (göreli, yalnız **yan dosya** donmuş) | 1 | **ATEŞLENDİ** (yan dosyayı anar) | yok ✓ |
| C | `<MUTLAK>/data/anka_t0092_sonda.pt` | **0** | **ATEŞLENMEDİ** | **model + yan dosya YAZILDI** ✗ |

**Kök neden:** `src/llm/frozen_guard.py:59` — `fnmatch.fnmatch(os.path.normpath(path), p) or
fnmatch.fnmatch(path, p)`. `os.path.normpath` **mutlak yolu göreli YAPMAZ**; `data/*.pt` deseni
`/Users/.../data/x.pt` ile **eşleşmez** ⇒ desen hiç sınanmamış olur.

**Bu kusur ÖNCEDENDİR:** `frozen_guard.py` son commit **`4251c53` (2026-09-16)**; bu görevde
**DEĞİŞMEDİ** (`git status` boş). Ve yalnız yan dosyayı değil **model kaydını da** etkiler ⇒
`--save-path` ile mutlak yol yazan **her** koşum donmuş korumadan muaftır.

**Neden düzeltilmedi:** (i) dosya T-0092 `writes[]`'inde **değil**; (ii) `frozen_guard.py`
**paylaşılan bir güvenlik modülüdür** — kapsam dışı değişiklik başka görevlerin tabanını
geçersiz kılar; (iii) projenin yerleşik uygulaması ölçülen yeni kusuru **beyan edip**
düzeltmemektir. **Operatör kararı bekler**; düzeltmesi ayrı bir görev ister.

### 7/2 · Yazma sırası REZİDÜEL RİSKİ (BEYAN — ÖLÇÜLMEDİ)

Model **önce**, yan dosya **sonra** yazılır. Sıra bilinçlidir: iki `os.replace` arasında kesinti
olursa yan dosya **bir önceki aralıkta** kalır ⇒ `model_sha256` diskteki **yeni** modelle
uyuşmaz ⇒ bir sonraki devam koşumu **eşleşme kapısında DURUR**. Ters sırada, kesinti "uyuşan
görünen ama yanlış" bir çift üretebilirdi.

**Bu bir BEYANDIR, ölçüm DEĞİLDİR:** kesintiyi üretip kapının durduğunu **ölçmedim**. Ölçülen
şey, kapının **yanlış digest'te durduğudur** (G5) — kesinti senaryosunun **aynı kapıya**
düştüğü ise **çıkarımdır**, ölçüm değil.

### 7/3 · `tests/test_compiler_entrypoints.py` sayı sabitlemesi (kapandı, §6/6)

`len(save_indices) == 1` → **guard sayısı ≥ save sayısı** ilişkisine çevrildi. Neden yalnız konum
yetmez: erken guard **bütün** save'lerden önce geldiği için *"save'den önce bir guard var mı"*
kontrolü **hiçbir zaman düşemez** ⇒ **VAKUM KAPI**. Kanarya fikstürüyle üç dal ölçüldü:

```
KANARYA-1 (guard'sız 3. save)  guard=2 save=3 → DÜŞTÜ  ✓ (kusuru yakalıyor)
KANARYA-2 (guard'lı 3. save)   guard=3 save=3 → GEÇTİ  ✓ (yanlış pozitif yok)
KANARYA-3 (gerçek train.py)    guard=2 save=2 → GEÇTİ  ✓
```

---

## 8. Açık kalan maddeler

1. **Donmuş kapının mutlak yol açığı** (§7/1) — **düzeltilmedi**; operatör kararı bekler.
2. **İki `os.replace` arası kesinti** (§7/2) — **beyan edildi, ölçülmedi**.
3. **Gerçek eğitim kalitesine etkisi ÖLÇÜLMEDİ.** Momenti geri yüklemenin **kayıp eğrisine** ve
   **nihai modele** etkisi ölçülmemiştir; ölçülen şey **ilk güncellemenin büyüklüğüdür**
   (1,7306×) ve **gidiş-dönüşün bit-özdeşliğidir**. *"Momentler kaydediliyor" ≠ "model daha iyi"*.
4. **`data/anka_a1.pt` / `anka_a1r.pt` moment TAŞIMAZ** — bu görev onları **yeniden üretmedi**;
   mevcut checkpoint'lerden devam eden koşum yine momentlerini sıfırdan başlatır ve **stderr
   uyarısı** alır (G4a). Gerçek kazanç, **bu görevden SONRA** kaydedilen checkpoint'lerde başlar.
5. T-0091'den devreden açıklar **değişmedi**: `plausibility oracle` yok · büyük-harf homografı
   sınıfı · plan↔kod uyuşmazlığı · D4 (kesme/rakam) · REPL EOF asırması · T-0086 öz-referansı.

---

## 9. Tam digest tablosu (önek DEĞİL)

**DEĞİŞEN / ÜRETİLEN:**

| Yol | Bayt | sha256 | Not |
|---|---:|---|---|
| `train.py` | 25,710 | `ecf3a79e9e38600abb8068cbe11066d9e453dc4d83b44c46c604b01d6fabaa7f` | DEGISTI |
| `tests/test_train_optimizer_state.py` | 14,124 | `00de0f8871ba79ee02c9850ae3d2e3c21f7a26a196afedaee38f20e7f79d2e11` | YENI (10 kapi) |
| `tests/test_compiler_entrypoints.py` | 25,006 | `235cc81cf7ac62ed0c9490b743d273f2155bb44c1c0a92c55f5ea54886405bec` | DEGISTI (§7/3) |

**OKUNAN ve DEĞİŞMEYEN (donmuş/ürün yüzeyi) — bu görev bunlara YAZMADI:**

| Yol | Bayt | sha256 | Not |
|---|---:|---|---|
| `src/llm/frozen_guard.py` | 2,965 | `75b3bf6e4384bc808230fb6b8f06c3187c93d78a45d742a4fb75f85f9ac3097b` | DOKUNULMADI — mutlak yol aciginin kaynagi |
| `.agent-bus/frozen.json` | 1,086 | `c2eb9a69492d7a4e6eba78d3a7b3ded1544182fe1d88950e7e7f627dacdf77fa` | DOKUNULMADI (10 desen) |
| `data/anka_a1r.pt` | 373,735,137 | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | DOKUNULMADI — YALNIZ OKUNDU |
| `data/train.bin` | 21,523,330 | `00519ef695265a179336101efb0975aa8c522f288b7d55c9b53aa50bdc74cc5c` | DOKUNULMADI — YALNIZ OKUNDU |

`data/anka_a1r.pt` digest'i `b93cc1cd…` ile **başlar** ve planın kapandı diye yazdığı değerle
**birebir aynıdır** ⇒ bu görev onu **değiştirmedi**.

**Sonda kaynakları ve ÇIKTILARI (kanıt — dosya olarak, özet değil):**

| Yol | Bayt | sha256 | Not |
|---|---:|---|---|
| `$TMPDIR/t0092_probe2.py` | 3,707 | `7deffc82bd40cb7f96b342ed797dc99c2fa567691b2532e233abf4ca09b9eb4f` | kusur + maliyet + gidis-donus (§2) |
| `$TMPDIR/t0092_det.py` | 1,644 | `ea8610fa860159152ad0026e0560de11228b6438d0a28f34676c079583d14669` | tensor tensor karsilastirma + ZIP kabi (§6/5) |
| `$TMPDIR/t0092_kanarya.py` | 4,005 | `98d5e1481320ef7877ebe0e7446f67d4ef4c3c2f3d25618735d16b64facce44f` | AST testinin ayirt ediciligi (§7/3) |
| `$TMPDIR/t0092/probe2.txt` | 766 | `087af0c2b309438cae1764581fe53704331d8c8325e9558273b5c4dfc51a443c` | sonda CIKTISI |
| `$TMPDIR/t0092/kanarya.txt` | 576 | `029a465502e8c14904a16450daaed63fa4e284984669fa238451347542015a4c` | sonda CIKTISI |

Sonda **kaynakları** bu raporun `.json`'unda **tam metin** saklanır ⇒ `$TMPDIR` silinse de ölçüm
**yeniden üretilebilir**.

**Takım sağlığı:** `venv/bin/pytest -q` → **1 failed, 237 passed** (77,54 sn).
Taban **227**; sapma **TAM +10** = yeni test dosyasındaki **10 kapı**. Düşen tek test yine
`test_agent_gateway.py` **sandbox soket yasağı** (`socketserver.py:478 PermissionError`) ve
**ilan edilmiş taban davranışıdır**. **Bu görev o tabanı DEĞİŞTİRMEDİ** — sayı ölçüldü.

**Donmuş yüzey denetimi (Python ile, `find` DEĞİL — T-0091 §7/4 dersi):**

| Ölçüm | Sonuç |
|---|---|
| Son **60 dk** içinde değişen `data/**` `.pt`/`.bin` | **0** |
| `data/anka_t0092_sonda*` kalıntısı | **YOK** (test `finally`'si + elle `rm` ile doğrulandı) |
| `src/compiler/**` · `src/llm/tokenizer.py` · `src/llm/frozen_guard.py` | **değişmedi** |

---

## 10. Ek — yeniden koşulabilir sonda dizini

Üç sonda `$TMPDIR`'de koşuldu (depo ve `scratch/**` kirlenmesin diye) ve kaynakları **bu raporun
`.json`'unda tam metin** saklanır:

| Sonda | Ne ölçer | Koşum |
|---|---|---|
| `$TMPDIR/t0092_probe2.py` | kusur + maliyet + gidiş-dönüş bit-özdeşliği (§2) | `venv/bin/python $TMPDIR/t0092_probe2.py` |
| `$TMPDIR/t0092_det.py` | iki checkpoint'i **tensör tensör** karşılaştırır + ZIP kabı (§6/5) | `... t0092_det.py A.pt B.pt` |
| `$TMPDIR/t0092_kanarya.py` | AST testinin **ayırt ediciliği** (§7/3) | `venv/bin/python $TMPDIR/t0092_kanarya.py` |

Üçü de **depo kökünden** ve **`venv/bin/python` ile** koşulmalıdır
([[kopyalanan-betik-kabini-degistirir]]: `sys.path` `$TMPDIR`'i gösterirse `ModuleNotFoundError`).

**İddia sınırı:** bu görev bir **kayıt/yükleme sözleşmesi** kusurunu ölçtü ve kapattı.
Ölçülen şey **optimizer durumunun korunmasıdır** — modelin **dil kalitesi değil**. Eğitim
kalitesine etkisi **ölçülmemiştir** ve ölçülmüş sayılmamalıdır. Donmuş kapının mutlak yol açığı
**bulundu, beyan edildi ve DÜZELTİLMEDİ**; bu görev onu **kapsamıyordu**.
