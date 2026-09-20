# T-0085 · `train.py` ölü varsayılanı kaldırıldı — `--save-path` ZORUNLU · RAPOR

**Tarih:** 20 Eyl 2026 (UTC) · **Görev:** T-0085 · **Yürütücü:** claude (tek yürütücü)
**Not:** `.agent-bus/notes/T-0085.md` · **Kaynak:** T-0083 raporu §8, açık madde **1**

> Ölçüm raporu. Kapılar **ölçümden ÖNCE** ilan edildi (görev şartnamesi) ve
> **değiştirilmedi**. Ölçüm sonrası ortaya çıkan kusur (aşağıda §2.3) **tasarım
> düzeltilerek değil, rapora yazılarak** kapatıldı.

---

## 0. Hüküm

| Soru | Ölçülen cevap |
|---|---|
| Ölü varsayılan `train.py`'den kaldırıldı mı? | **EVET** — atama artık `None` |
| `--save-path` verilmezse ne olur? | **Sesli `RuntimeError`, `rc=1`, sıfır yazım** (§5/A) |
| Verilirse davranış korundu mu? | **EVET** — `rc=0`, koşum tamam, atomik kayıt (§5/B) |
| Donmuş yol koruması (`check_frozen_save_path`) duruyor mu? | **EVET** — `torch.save`'den önce, AST testi geçti |
| Repo-içi çağıran kırıldı mı? | **HAYIR** — varsayılana yaslanan çağıran **yok** |
| Sınıfın kalanı da sessizce düzeltildi mi? | **HAYIR** — **yalnız `train.py`**; kalan sınıf beyan edildi (§3) |
| `git add -A` | **kullanılmadı** |

---

## 1. İlan edilen kapılar ve sonuçları

| Kapı | Eşik (ilan edilen) | Sonuç | Kanıt |
|---|---|---|---|
| **K1** | sınıf ölçüldü ve LIVE/TARİHSEL sınıflandırıldı | **✓ GEÇTİ** (sayı **yeniden üretilemedi**, §2.3) | §2, §3 |
| **K2** | `train.py`'de ölü varsayılan atama **0** | **✓ GEÇTİ** — küçük-harf `"kristal"` **0** satır | §4 |
| **K3** | `--save-path` yoksa sesli erken durma, `rc≠0`, yazım yok | **✓ GEÇTİ** — `rc=1`, `data/*.pt` değişmedi | §5/A |
| **K3-poz** | verilince **aynı kapıya takılmaz** | **✓ GEÇTİ** — argüman kapısı **0 kez** ateşledi, sonraki kapıya ilerledi | §5/C |
| **K4** | `check_frozen_save_path` korunur ve `torch.save`'den önce | **✓ GEÇTİ** — AST testi `9 passed, 3 deselected` | §5/C, §6 |
| **K5** | `--save-path` verildiğinde davranış değişmez | **✓ GEÇTİ** — `rc=0`, 2 adım, atomik kayıt, artık `.tmp` yok | §5/B |
| **K6** | `venv/bin/pytest` ⇒ 213 passed, 1 failed (sandbox) | **✓ GEÇTİ** | §7 |
| **K7** | `git add -A` kullanılmadı | **✓ GEÇTİ** | §8 |

**Çürütme 1 (önceden ilan edildi):** *"Zorunlu kılmak repo-içi bir çağıranı kırarsa
düzeltme YANLIŞTIR."* — **Gerçekleşmedi.** `scripts/run_goal_pipeline.py` (269/277/285)
ve `scripts/retrain_clean_models.py` (78/95/113) `--save-path`'i **açıkça** veriyor;
hiçbir test varsayılanı aramıyor; `README.md:233` varsayılan **iddia etmiyor**.

**Çürütme 2 (önceden ilan edildi):** *"Kapı veri yüklemesinden SONRA ateşleniyorsa
fail-closed değildir."* — **KISMEN gerçekleşti ve beyan edilir.** Kapı, veri yüklemesi
ve sınır/meta kontrollerinden **sonra**; ama **her yazımdan ve eğitim döngüsünden
ÖNCE** ateşliyor (§5/A'da ölçüldü: durma anında `data/*.pt` değişmemiş). Veri
yüklemesi **saniyeler**, eğitim **saatler**. Kapının argüman ayrıştırmasının hemen
ardına taşınması **AÇIK İŞ**'tir (§9/1); kapıyı ölçümden sonra taşımak ilan edilmiş
kapsamı sessizce değiştirmek olurdu.

---

## 2. Sınıf ölçümü (K1)

### 2.1 Üç kapsam, üç sayı — hangisi söyleniyorsa ölçütü de söylenmeli

| # | Kapsam (araç) | Desen | Adet | Dosya |
|---|---|---|---|---|
| 1 | izlenen ağaç (`grep -rn ... .`) | `kristal[a-z_]*_model\.pt` | **108** | **33** |
| 2 | izlenen ağaç | `kristal[a-z_]*\.pt` (**geniş**) | **153** | **40** |
| 3 | `scratch/` (gitignore'lu) | `kristal[a-z_]*_model\.pt` | **241** | **43** |
| 4 | izlenen ağaç, `.md`+`.json` dahil | `kristal[a-z_]*_model\.pt` | **402** | **112** |

> **İzlenen ağaç ile `scratch/` ayrık kümelerdir** (bkz. §2.2): 1 + 3 = **349 atıf /
> 76 dosya** izlenen+scratch birleşik `.py` yüzeyidir.

### 2.2 ARACIN KAPSAMI — `grep` burada **ugrep** ve `.gitignore`'a uyuyor

Bu, ölçümün kendisinde bulunan bir kapsam kusurudur ve **kayda geçer**:

```
$ grep --version | head -1
ugrep 7.8.4 aarch64-apple-macosx +neon/AArch64 …
$ grep -rn --include="*.py" "kristal" .  | grep -c "scratch/"     →  0
$ grep -rn --include="*.py" "kristal" scratch/ | wc -l            →  241
```

`scratch/` bir sembolik bağ **değildir** (`drwxr-xr-x … 361 … scratch/`), ama
**`.gitignore`'da olduğu için** kökten yapılan özyinelemeli tarama onu **sessizce
atlar**. Yani "`kristal` atıf sayısı" diye tek bir sayı yazmak, ölçütü yazmadan
**yanlış olur**.

- **Sınıf:** [[denetim-kapsami-iddiadan-dar]] ve onun ters yönü
  [[gitignore-kapanis-kanitini-saklar]] — aynı tuzak `git status`'ta ölçülmüştü;
  şimdi **sayacın kendisinde** ikinci kez üredi.
- **Kural (bu rapordan sonra):** sayı, **kapsamı ve deseniyle birlikte** yazılır.

### 2.3 İLAN EDİLEN SAYI YENİDEN ÜRETİLEMEDİ — beyan edilir, düzeltilmez

Şartnamede **K1 = "111 atıf / ~35 dosya"** diye ilan edilmişti. Bugünkü hiçbir
ölçüm bu sayıyı **vermiyor**: dar desen **108/33**, geniş desen **153/40**,
`scratch` dahil **349/76**. İlan edilen sayı, **hiçbir yazılı ölçütün çıktısı
değildir**.

- **Yapılan:** kapı **düzeltilmedi** (ölçümden sonra tasarım düzeltilmez —
  [[tasarim-sayisi-betikle-hesaplanmali]]); kusur **buraya yazıldı**.
- **Ders:** ilan edilen sayı, onu üreten **komutla birlikte** yazılmalıdır. Aksi
  halde sayı doğru olsa bile **hangi ölçütün** olduğu bilinmez.

---

## 3. LIVE / TARİHSEL sınıflandırması — ve neden hepsi düzeltilmedi

Ayrım ölçütü: **LIVE** = yol, çalışma zamanında bir **okuma/yazma hedefi** olarak
kullanılıyor (varsayılan parametre veya `torch.load` argümanı). **TARİHSEL** = yol
yalnız bir **kayıt/ölçüm** metninin içinde anılıyor.

| Sınıf | Temsilci | Ölçülen davranış | Neden T-0085 kapsamı dışı |
|---|---|---|---|
| **LIVE — yazma varsayılanı** | `train.py:149` | `--save-path` verilmezse **o ada yazmaya çalışıyordu**; `data/*.pt` **donmuş** ⇒ çift kusur | **DÜZELTİLDİ** (bu görev) |
| **LIVE — okuma varsayılanı** | `chat_prompt.py:309`, `src/gateway/agent_gateway.py:80`, `src/rag/rag_pipeline.py:226`, `src/gateway/retrain_pipeline.py:97` | `model_path` varsayılanı; `torch.load` **dosyayı bulamayınca sesli patlar** ⇒ sessiz değil | **KOD** değişikliği; ayrı görev + ayrı kapı ister |
| **LIVE — açık argüman** | `scripts/run_goal_pipeline.py` (20 atıf), `scripts/retrain_clean_models.py` (18) | `--save-path` **açıkça** veriliyor; donmuş kapı zaten çağrılıyor | Çağıran; varsayılan kaldırılınca **kırılmıyor** (ölçüldü) |
| **ÖLÇÜM/TARİHSEL betik** | `scripts/t0027_measure_models.py`, `evaluate_*`, `run_experiment_*`, `diagnostics_*` | Bir kez koşulmuş ölçüm betikleri | Geçmişi yeniden yazmak **kanıtı bozar** |
| **TARİHSEL kayıt** | `data/eval/**`, `.agent-bus/notes/**`, `.agent-bus/tasks/**` | Donmuş ölçüm kayıtları | **Değiştirilmesi yasak**: tarihsel kaydı düzeltmek = kanıtı tahrif etmek |
| **TEST FİKSTÜRÜ** | `tests/test_compiler_entrypoints.py:36` | `FROZEN_MODEL = "data/kristal_model.pt"` yalnız `check_frozen_save_path` **fikstürü** | Varsayılanı sınamıyor (ölçüldü) |

**Tek satırı düzeltip sınıfı kapatmış sayılmak** [[denetim-kapsami-iddiadan-dar]]
kusuru olurdu; bu yüzden **yalnız `train.py`** düzeltildi ve kalanı **beyan edildi**.

---

## 4. K2 — sayaç ölçümü, harf duyarlılığı belirtilerek

```
$ grep -c  "kristal" train.py     →  0      (küçük harf, harf DUYARLI)
$ grep -ic "kristal" train.py     →  4      (harf DUYARSIZ)
```

Harf-duyarsız **4** satır, silinen zincir **değildir**: `KristalLM`,
`KristalDataset`, `KristalEmbedding` — **korunması gereken mimari sınıf adlarıdır**
([[mimari-korunur-kristallm]]: *"sınıf adları `Kristal*` KALIR"*). İki sayı
çelişmiyor; **ölçütleri farklı**.

**Emekli-notu tuzağı ölçüldü ve kaçınıldı** ([[emekli-notu-sayaci-sisirir]]): emekli
notu silinen yolu **adıyla anmıyor** (dosya adı kod içinde `<silinmiş zincir>.pt`
diye yazıldı), bu yüzden küçük-harf sayacı **şişmedi**. Not, `# EMEKLI (T-0085):`
etiketiyle **ayrı blok**tur ⇒ denetimde ayıklanabilir.

---

## 5. Ölçülen koşumlar (üçü de küçük val bin ile; gerçek eğitim BAŞLATILMADI)

### A · NEGATİF — `--save-path` VERİLMEDEN

```
Veri kümesi yüklendi: data/anka_a1r_pretrain_val.bin (Blok boyutu: 128). Toplam morfem token sayısı: 2,000,000
[sinir] en buyuk jeton id=33,113 < vocab_size=33,114 (guvenli)
[meta] sozluk/kulliyat provenance: BIREBIR (sozluk_sha256=f9940a8d8e1f7cd9… · giris=33114)
RuntimeError: DURDURULDU: --save-path verilmedi. Varsayilan kayit yolu KALDIRILDI (T-0085):
eski varsayilan, silinmis bir checkpoint zincirinin adini tasiyordu.
Kayit yolunu ACIKCA verin, or. --save-path data/anka_a2.pt
```

**`rc=1`** · `data/*.pt` **değişmedi** (yalnız `anka_a1.pt`, `anka_a1r.pt`; mtime'lar
korundu) ⇒ **sıfır yazım**.

### B · K5 — `--save-path` VERİLEREK (davranış korundu mu?)

Hedef **repo dışı** seçildi (`$TMPDIR/t85_k5.pt`) ⇒ donmuş desene **ve** repoya
yazmaz; kontrol artefaktı digest alındıktan sonra **silindi**.

```
Model mimarisi kuruldu ve cihaza taşındı. (Öğrenme Oranı: 0.001)
Eğitim Başlatılıyor -> Adım Sayısı: 2, Batch Boyutu: 2, Block Boyutu: 128, Hedef: ON-EGITIM
Adım    1/2 | Kayıp (Loss): 10.6716 | Adım Süresi: 0.83s
 EĞİTİM BAŞARIYLA TAMAMLANDI!
Başlangıç Kaybı:   10.6716
Bitiş Kaybı:       8.8890
Eğitilmiş model ağırlıkları '/tmp/claude-501/t85_k5.pt' dosyasına kaydedildi.
```

**`rc=0`** · hedef yazıldı (`356M`, `aafb33f7c6afd6d12272707ed17d9b402472ff75b69d1d91cefa9ac49467b625`)
· artık `.tmp` **yok** (atomik kayıt çalıştı) · repo'da `.pt.tmp`/`zzz*` sızıntısı **yok**.

*Yan bulgu (kayda geçer):* sıfırdan koşumun **ilk kaybı 10,6716**; ilan edilen
canlılık bandı `ln V = ln 33.114 = 10,4076` ([[canlilik-imzasi-moda-bagli]]).
`batch_size=2` için bandın içindedir — bu bir *kapı* değil, **yan gözlemdir**.

### C · K3-POZİTİF + K4 — `--save-path` donmuş desene verilerek

```
File "train.py", line 188, in main
    check_frozen_save_path(model_save_path, allow_frozen_write=allow_frozen_write)
RuntimeError: Donmuş yola yazma engellendi: data/zzz_kapi_testi.pt (allow_frozen_write=False)
```

**Ateşleyen kapı DONMUŞ kapıdır; argüman kapısı 0 kez ateşledi** (`grep -c
"save-path verilmedi"` → **0**). Yani `--save-path` verildiğinde yeni kapı
**takılmıyor**, sonraki kapıya **ilerliyor**. `rc=1` · test dosyası **yazılmadı**.

---

## 6. K4 — AST değişmezi

`tests/test_compiler_entrypoints.py::test_ast_train_py_has_frozen_guards`:
`train.py`'de **tek** `torch.save` çağrısı olmalı ve ondan **hemen önceki** çağrı
`check_frozen_save_path` olmalı.

```
9 passed, 3 deselected
```

Yapı (ölçüldü): `check_frozen_save_path` **tanımı** 16 · **çağrısı** 188 ·
`torch.save` 306 ⇒ **değişmez korundu**.

---

## 7. K6 — takım

```
venv/bin/pytest  →  213 passed, 1 failed
```

Düşen tek test `tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints`
— nedeni `socketserver` **`PermissionError`**, yani **sandbox soket yasağı**;
sandbox dışında geçer. **Sayı değişmedi** (T-0081'de ölçülen damgayla aynı).

---

## 8. Değişen dosyalar — iki yönlü, ve SİLİNEN satırlar

| Yol | Durum | Silinen satır |
|---|---|---|
| `train.py` | **değişti** (+19 / −1) · 333 → **351** satır | `model_save_path = 'data/kristal_model.pt'` |
| `USER_GUIDE.md` | **değişti** (+136 / −2 vs HEAD) · 564 → **586** satır | yalnız **2 bayat tablo hücresi** (§8.1) |
| `data/eval/anka_r8_train_varsayilan_2026-09-20.md` | yeni (bu rapor) | — |
| `.agent-bus/notes/T-0085.md` | yeni | — |
| **Toplam** | **4** — `len(cf) == len(set(cf))` ⇒ **tekillik GEÇTİ** | |

Kapsam dışı **hiçbir** dosyaya yazılmadı: `chat_prompt.py`, `test_model.py`,
`src/**`, `scripts/**`, `CLAUDE.md`, `README.md` **değişmedi**.

### 8.1 "Silinen 2 satır" kanıtı — bilgi kaybı yok

```
-| `--load-path <yol>`  | `--save-path`           | … (örn: `data/kristal_model.pt`). |
-| `--save-path <yol>`  | `data/kristal_model.pt` | Eğitilen yeni ağırlıkların … |
```

Silinen **tek şey** iki **bayat tablo hücresidir**; ikisi de yeni değerlerle
**değiştirildi**. **Tarihsel T-0083 notu SİLİNMEDİ** — kendi damgasıyla duruyor ve
ardına **damgalı bir GÜNCELLEME** satırı eklendi:

> *"GÜNCELLEME (T-0085 · damgalı ölçüm: 20 Eyl 2026) — yukarıdaki not artık
> TARİHSELDİR; silinmedi, kendi damgasıyla korundu."*

Aynı disiplin T-0084'te TARİHSEL banner için uygulanmıştı; burada da ilan edilmiş
kapsam **sessizce gevşetilmedi**.

### 8.2 `numstat` uyarısı — iki sayı çelişmiyor, kümeler farklı

`git diff --numstat` **`HEAD`'e** göredir ve commit edilmemiş **T-0083 (+20)** ile
**T-0084 (+92)** değişikliklerini **de kapsar**. `USER_GUIDE.md` için okunan
**+136 / −2**, T-0085'in **tek başına** katkısı değildir: net satır değişimi
**564 → 586 = +22**'dir ve **−2**'nin tamamı §8.1'deki iki tablo hücresidir
([[iki-sayi-celisiyor-sanma-once-kume]]).

---

## 9. Açık kalan işler (kapatılmadı — beyan edilir)

1. **Argüman kapısının YERİ.** Şu an veri yüklemesinden **sonra**, eğitim
   döngüsünden **önce**. Fail-closed'dur (yazım yok) ama **erken** değildir;
   argüman ayrıştırmasının hemen ardına taşınmalı. **Ayrı görev + ayrı kapı ister**
   (Çürütme 2'de beyan edildi).
2. **LIVE okuma varsayılanları** (§3): `chat_prompt.py:309` ve üç `src/**` modülü.
   `torch.load` sesli patlar (sessiz değil) ama **ölü ad taşır**; **KOD** değişikliği
   ⇒ ayrı görev.
3. **Sınıfın kalanı** (izlenen ağaçta **108 atıf / 33 dosya**; `scratch` ile
   **349 / 76**). Sessizce düzeltilmedi; LIVE/TARİHSEL ayrımı §3'te yazıldı.
4. **`USER_GUIDE.md:6-11` TARİHSEL banner cümlesi** hâlâ *"doğrulanmadığı için iddia
   edilmiyor"* diyor; T-0084'ün açık maddesiyle **aynı** gerilim. Banner'a bu turda
   **dokunulmadı**.
5. **`chat_prompt.py`'a `--vocab` bayrağı yok** ⇒ güncel checkpoint sohbet
   arayüzüne bağlanamıyor (T-0084 §8/2 ile **aynı** madde).
6. **Sayı ilanı kuralı:** K1'in **111/35**'i yeniden üretilemedi (§2.3); bundan
   sonra ilan edilen her sayı **komutuyla** yazılmalıdır.

---

## 10. Bu turda ölçülen KENDİ kusurlarım (kayda geçer)

| # | Kusur | Etki |
|---|---|---|
| 1 | **K1'de ilan ettiğim sayı (111/35) hiçbir ölçütün çıktısı değil** (§2.3) | Kapı "geçti" diye okunabilirdi; sayı **yeniden üretilemedi** diye beyan edildi |
| 2 | **`grep`'in `scratch/`'ı sessizce atladığını fark etmedim** (§2.2) — burada `grep` **ugrep** ve `.gitignore`'a uyuyor | İlk sayım **241 atıf** eksik verdi; kapsam yazılmadan sayı yanlış olur |
| 3 | Bir satırlık düzeltme sanıp **111 atıflık bir sınıfın başına** dokundum | Tek satırı düzeltip "sınıf kapandı" demek [[denetim-kapsami-iddiadan-dar]] olurdu; sınıf **beyan edildi** |
| 4 | `rc` yakalama: `cmd | grep` zincirinde `$PIPESTATUS` **zsh'te yok** (`$pipestatus`) | İlk koşumda `rc` **boş** geldi; çıktı dosyaya alınıp `rc=$?` doğrudan okunarak düzeltildi |

---

*Raporun bütün sayıları bu turda koşulan komutlardan alınmıştır. Ölçülmeyen hiçbir
mekanizma yazılmadı; ölçülen kendi kusurlarım (§10) dahil.*
