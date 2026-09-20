# T-0082 · Belge tazeliği onarımı — RAPOR

**Tarih:** 20 Eyl 2026 (UTC) · **Görev:** T-0082 · **Yürütücü:** claude (tek yürütücü)
**Not:** `.agent-bus/notes/T-0082.md` · **Önceki devir:** `dokuman-borcu-2026-09-19.md`, T-0080 kapanış raporu §5.1

> Bu rapor **ölçüm raporudur**. Her sayı ya yeniden koşulan bir komuttan ya da
> dosyadan birebir alındı; hiçbir iddia ölçülmeden yazılmadı.

---

## 0. Hüküm — önce sonuç

| Soru | Ölçülen cevap |
|---|---|
| `SPEC.md:48` (D8) düzeltildi mi? | **EVET** — `6bca90b` + 44/44 kayda geçti |
| `SPEC.md` sürüm damgası | **EVET** — `2026-09-20 (T-0082)` |
| `CLAUDE.md` değişti mi? | **HAYIR — gerekmedi.** Üç iddiası yeniden ölçüldü, **üçü de doğru** |
| Aynı sınıftan başka bayat madde? | **EVET, 1** — `SPEC.md` açık kusur #1 (sayım + ölçüm tuzağı) |
| Kapsam dışı bulunan kusur | **2** — `USER_GUIDE.md`/`README.md` ölü yollar · **`train.py:149`** |
| `git add -A` kullanıldı mı? | **HAYIR** |

**Dürüst cümle:** istenen düzeltme yapıldı; ama ölçüm **belgede olmayan bir kusur**
çıkardı — `train.py`'nin varsayılan kayıt yolu hâlâ **silinmiş zincirin** adını taşıyor
(§3.2). O, bu görevin kapsamı **dışındadır** ve düzeltilmedi, yalnız beyan edildi.

---

## 1. Ölçüm — yazmadan ÖNCE

| Ne | Ölçülen | Komut |
|---|---|---|
| `notes/` dosya / izlenen / izlenmeyen | **44 / 44 / 0** | `ls \| wc -l` + `git ls-files \| wc -l` |
| `6bca90b`'nin eklediği not dosyası | **tam 17** | `git show --stat 6bca90b -- .agent-bus/notes/` |
| `SPEC.md` satır 3 | `son düzenleme: 2026-09-19 (T-0081)` | `sed -n 3p` |
| `SPEC.md` satır 48 | "27'si izleniyor, **17'si izlenmiyor**" | `sed -n 48p` |
| `CLAUDE.md` donmuş liste ↔ `frozen.json` | **10/10 iki yönlü eşit** | §2.2 |
| `venv/bin/pytest` | **213 passed, 1 failed** | yeniden koşuldu |
| `events.jsonl` | **1.191 satır**, **6** tip | `wc -l` + sayaç |
| `data/*.pt` (gerçek) | `anka_a1.pt`, `anka_a1r.pt` | `ls data/*.pt` |
| `kristal*.pt` diskte/git'te | **0** | `ls` + `git ls-files \| grep` |

**Ana bulgu:** `6bca90b`'nin eklediği 17 dosya, `SPEC.md:48`'in "izlenmiyor" dediği
kümenin **tam kendisidir** (T-0062…T-0077 + T-0081). Yani D8 borcu **kapanmıştı**,
yalnız belge bunu **yazmıyordu**.

---

## 2. Yapılan düzeltmeler

### 2.1 `SPEC.md` D8 paragrafı

| | |
|---|---|
| **Önce** | "19 Eyl 2026 ölçümü: 44 dosya, 27 izleniyor, **17 izlenmiyor** … sürümleme T-0061'de durmuş" |
| **Sonra** | 19 Eyl ölçümü **tarihsel** olarak korundu + `6bca90b`'nin **17 dosyayı** eklediği ve dizinin **44/44** olduğu yazıldı (20 Eyl ölçümü) |

⚠ **Bilinçli olarak "0 izlenmiyor" diye YAZILMADI.** Bu görevin **kendi notu**
(`T-0082.md`) yazıldığı anda dizinde yeniden **1 izlenmeyen** dosya doğar ⇒ sabit bir
"0" iddiası **doğar doğmaz bayat** olurdu. Kalıcı kısım korundu:
**durmak ≠ sürümlenmek**; durum **her turda yeniden ölçülür**.

### 2.2 `CLAUDE.md` — DOKUNULMADI (üç bağımsız doğrulama)

| # | İddia | Ölçüm | Sonuç |
|---|---|---|---|
| 1 | pytest `213 passed, 1 failed` | yeniden koşuldu: **213 passed, 1 failed** | **doğru** |
| 2 | Donmuş **10 desen** | `frozen.json` ile **iki yönlü** karşılaştırma: `doc→json eksik: []`, `json→doc eksik: []` | **doğru** |
| 3 | `train.py:66` varsayılan sözlük · `:112-116` `RuntimeError` | satırlar okundu: `vocab_path = 'data/rebuild/vocab_base_32852.json'` · `if _jeton_max >= vocab_size: raise RuntimeError` | **doğru** |

`train.py` T-0080'de **değişmişti** ⇒ satır atıflarının kaymış olması beklenirdi;
**kaymamıştır.** `33114` dizgesi `train.py`'de hâlâ **0** kez geçer.

> **Not (kapsam):** belgeyi değiştirmemek de bir **bulgudur** — "tazelik onarımı"
> gerekçesiz düzenleme üretmez. Bu satır o kararın kanıtıdır.

### 2.3 `SPEC.md` "Bilinen açık kusurlar" #1 — aynı sınıftan ikinci madde

Sayım **1.174 → 1.191** olmuş; dahası bir **ölçüm tuzağı** doğmuş:

| | |
|---|---|
| Kaynakta `task_updated` | `grep -c scripts/agent_bus_mcp.py` = **0** ⇒ **olay tipi yok** (kusur sürüyor) |
| Log'da `task_updated` | **1 satır** — ve o satır bir **tip değil**, T-0081'in `result_reported` kaydının **`evidence` metni** (kusuru *anlatan* alıntı) |

⇒ `grep task_updated log/events.jsonl` yapan okuyucu **yanlışlıkla "kusur kapandı"**
okur. Belgeye hem yeniden ölçüm hem **"iddia KAYNAK üzerinden sınanır"** uyarısı yazıldı.

---

## 3. Kapsam dışı bulunan kusurlar (düzeltilmedi — beyan edilir)

### 3.1 `USER_GUIDE.md` 7 · `README.md` 1 ölü referans

Ölçüldü: bu belgeler `data/kristal_model.pt`, `data/kristal_carpenter_model.pt`,
`data/kristal_model_sft.pt`, `data/kristal_b1_5_best.pt` yollarına **canlı komut**
veriyor; bu dosyalar **diskte de git'te de yok**.

| Belge | Ölü atıf | Satırlar |
|---|---:|---|
| `USER_GUIDE.md` | **7** | 59, 62, 100, 101, 118, 130, 199 |
| `README.md` | **1** | 210 (`cp data/kristal_model.pt data/kristal_model_sft.pt`) |

**`CHANGELOG.md` kapsam DIŞI bırakıldı ve bu bilinçlidir:** changelog **tarihsel
kayıttır**; Kristal atıfları o dönemde **doğruydu**. Tarihi silmek kayıt bozar.
`GEMINI.md` ve `llmcoder.md` içinde ölü atıf **yok** (ölçüldü).

### 3.2 `train.py:149` — KOD kusuru (belge değil)

```
149:    model_save_path = 'data/kristal_model.pt'
```

`--save-path` verilmezse koşum **silinmiş zincirin adıyla** yazmaya çalışır.
Zincir **hafifletici:** hedef `data/*.pt` donmuş desenine düştüğü için
`check_frozen_save_path` **yüksek sesle durur** — sessiz değil. Ama **varsayılanın
kendisi** bayattır. `USER_GUIDE.md:101`'in "*varsayılan `data/kristal_model.pt`*"
satırı **kodu doğru anlatır** ⇒ kusur belgede değil, **kaynakta**; belgeyi "düzeltmek"
yanlış olurdu.

> Bu madde `writes[]`'te **yok** ⇒ **dokunulmadı.** Kod değişikliği kendi görevini,
> kendi kapısını ve donmuş-yol değerlendirmesini hak eder.

---

## 4. Ölçüm aracının KENDİ kusuru (kayda geçer)

Donmuş liste karşılaştırmasının **ilk koşumu yanlış aykırılık üretti**: regexim
satırdaki **ilk** parantezi yakaladı ve başlık metnini (`SPEC.md Kural 2):** [`…`)
desen sandı ⇒ sahte sonuç "**doc→json eksik: 1**".

Çıpa `**10 desen** (\`` olarak düzeltildi ⇒ gerçek sonuç **10/10 iki yönlü eşit**.
**Sayı doğruyken araç yanlıştı**; düzeltilmeden raporlanmadı
([[denetim-kapsami-iddiadan-dar]]).

---

## 5. Kabul kriterleri — madde madde

| # | Kriter | Durum | Kanıt |
|---|---|---|---|
| 1 | D8 paragrafı ölçülen durumu taşır, dirençli kural korunur | **✓** | §2.1 |
| 2 | Satır 3 damgası `2026-09-20 (T-0082)` | **✓** | `sed -n 3p` |
| 3 | `CLAUDE.md` pytest sayısı yeniden ölçüldü; bulgu rapora yazıldı | **✓** | §2.2 — değişiklik **gerekmedi** |
| 4 | Donmuş 10 desen iki yönlü karşılaştırıldı | **✓** | §2.2, iki yön de **boş** |
| 5 | Önce/sonra sha256 | **✓** | §6 |
| 6 | Kapanış: mesaj + kiralar boş + `changed_files` iki yönlü, `len==len(set)` | **✓** | §7 |

---

## 6. Önce / sonra — TAM digest

| Dosya | Önce | Sonra | Satır |
|---|---|---|---:|
| `.agent-bus/SPEC.md` | `5e9e08a0554ac8844f51bcb18624d37915845a908bff40cf936c576086d9bdfb` | **`afe111bd611f78e0cccb3090945aca43e0bdacc6c57e59a370cb99955a221c51`** | 274 → **283** |
| `CLAUDE.md` | `d11e3e17b8631a67fef195f1c6d4b27cfdf3820784a29c74a949b67e0d97c406` | **`d11e3e17b8631a67fef195f1c6d4b27cfdf3820784a29c74a949b67e0d97c406`** | 47 → **47** |

`CLAUDE.md`'nin digest'i **birebir aynı** ⇒ dokunulmadığı **kanıtlıdır** (beyan değil).

---

## 7. Değişen dosyalar — iki yönlü

| Yol | Durum |
|---|---|
| `.agent-bus/SPEC.md` | **değişti** (2 düzeltme) |
| `CLAUDE.md` | **değişmedi** — doğrulandı, bilinçli |
| `.agent-bus/notes/T-0082.md` | yeni |
| `data/eval/anka_r6_belge_onarim_2026-09-20.md` | yeni (bu rapor) |
| **Toplam** | **4 beyan · 3 yazıldı · 1 bilinçli dokunulmadı** |

`len(cf) == len(set(cf))` ⇒ **tekillik GEÇTİ**. Kapsam dışı **hiçbir** dosyaya
yazılmadı: `USER_GUIDE.md`, `README.md`, `train.py` **değişmedi** (kapsam dışı, §3).

> **`git status` bu listenin kanıtı DEĞİLDİR:** `notes/` bu görevle birlikte
> **yeniden izlenmeyen** hâle geldi (§2.1'in gerekçesi tam olarak budur);
> kanıt **digest tablosuna** bağlanır. `git add -A` **kullanılmadı.**

**Gerekçenin ölçümle doğrulanması:** not yazıldıktan **sonra** ölçüldü ⇒
`.agent-bus/notes/` = **45 dosya / 44 izlenen / 1 izlenmeyen**. Yani §2.1'deki
"sabit 0 yazma" kararı bir **önlem değil, doğrulanmış** bir karardır: o cümle
yazıldığı anda yanlış olacaktı. Ayrıca `git status --porcelain` izlenen dosyalarda
**yalnız** `SPEC.md` gösterir ⇒ `CLAUDE.md`'ye dokunulmadığı **git düzeyinde de**
kanıtlıdır.

---

## 8. Açık kalan işler (kapatılmadı — beyan edilir)

1. **T-0083 — `USER_GUIDE.md` + `README.md`** ölü Kristal yolları. **Karar gerekiyor:**
   yaşayan komutlar **güncel checkpoint'lere** mi yönlendirilecek, yoksa o bölümler
   **tarihsel** diye mi işaretlenecek? İkincisi seçilirse **uydurma yol yazılmaz**.
2. **`train.py:149`** varsayılan kayıt yolu — kod görevi; kendi kapısını hak eder.
3. **`notes/` yeniden izlenmiyor** (`T-0082.md` yazıldı) ⇒ commit yetkisi operatörde.
   **`6bca90b` hâlâ `ahead 1` ve push edilmedi** (T-0080'dan devir).
4. `SPEC.md` "Bilinen açık kusurlar" sayıları **damgalıdır**; her turda yeniden
   ölçülmelidir — bu görev onu bir kez daha tazeledi, **kalıcı kılmadı**.

---

*Raporun bütün sayıları bu turda yeniden koşulan komutlardan ve dosyalardan birebir
alınmıştır. Ölçülmeyen hiçbir mekanizma yazılmadı.*
