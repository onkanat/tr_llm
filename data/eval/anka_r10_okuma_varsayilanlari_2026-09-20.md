# T-0087 · Kapanış Raporu — LIVE okuma varsayılanları: ölü model yolu + bayat sözlük (fail-closed)

**Yürütücü:** claude · **Tarih:** 20 Eyl 2026 · **Görev:** `.agent-bus/state/tasks/T-0087.json`
**KAYNAK:** T-0085 raporu §3 (LIVE/TARİHSEL) + T-0083 raporu §8 · **Yürütücü notu:** `.agent-bus/notes/T-0087.md`

> **Disiplin:** kapılar (K1…K7 + iki çürütme maddesi) ölçümden **ÖNCE** görev kaydına yazıldı ve
> **değiştirilmedi**. Ölçüm sonrası bulunan kusurlar **tasarıma yansıtılmadı**, bu raporun §5'ine yazıldı.

---

## 1 · Özet hüküm

**K1 ✓ · K2(a) ✓ · K2(b) ✗ (İLAN HATALI) · K3 ✓ · K3-poz ✓ · K4 ✓ · K5 ✓ · K6 ✓ · K7 ✓**

Görevin en önemli ölçümü **K4**'tür ve sonucu nettir: bayat sözlük (31.357) ile 33.114 satırlı
checkpoint **DURMUYOR** — üç `[SOZLESME_UYARI]` basıyor, **1757 satırı kırpıyor** ve hemen ardından
**"tam eşleşme (0 eksik, 0 fazla)"** diyerek **güven veriyor**. Bu, "sessiz" değil **"yutucu"** bir
kusurdur: mesaj var, ama son satır operatöre yanlış bir "her şey yolunda" imzası veriyor. Kapı bu yüzden
**kırpmanın değil, yanlış güvencenin** önüne konuldu.

Model yolu için kazanç **erçeklik + mesaj**tır (ölü varsayılan zaten `FileNotFoundError` veriyordu);
sözlük yolu için kazanç **gerçek davranış**tır (K4). Bu ayrım, ilan edilen **Çürütme 2**'nin gereğidir ve §6'da açıkça yazılmıştır.

---

## 2 · Ölçülen zemin (şartname yazılmadan önce)

| Yer | Eski değer | Durum |
|---|---|---|
| `chat_prompt.py:309` | `model_path = 'data/kristal_model.pt'` | dosya **YOK** |
| `chat_prompt.py:285` | `vocab_path = 'data/vocab.json'` | **VAR ama BAYAT (31.357)**; `--vocab` bayrağı **hiç yoktu** |
| `src/gateway/agent_gateway.py:80-81` | aynı iki varsayılan | aynı |
| `src/rag/rag_pipeline.py:225-226` | aynı iki varsayılan | aynı |
| `src/gateway/retrain_pipeline.py:95,97` | aynı iki varsayılan + `save_path` **hiç yoktu** | aynı |

Ölçülen dosya gerçekleri: `data/kristal_model.pt` **YOK** · `data/kristal_carpenter_model.pt` **YOK** ·
`data/vocab.json` **VAR** (`4b40eebe…`, 31.357) · `data/train_future_finetune.bin` **VAR** (`80340a98…`) ·
`data/anka_a1r.pt` **VAR** (`b93cc1cd…`, 33.114 satır).

---

## 3 · Yapılan değişiklik (ilan edilen tasarım)

**Varsayılan silindi → parametre `None` → kullanım anında fail-closed durma.** Böylece *zaten ölü*
çağıranlara dokunmak gerekmedi; yalnız **eskiden sessizce yanlış olan** yol sesli durur.

1. Dört yükleme sitesine **argüman kapısı** (model/vocab) + **sözlük↔checkpoint tutarlılık kapısı**
   (`resize_state_dict`'ten **ÖNCE**; `embedding.embedding.weight` satır sayısı sözlükle karşılaştırılır).
2. `RetrainPipeline`: `_zorunlu()` saf yardımcısı; `compile_backlog_to_bin` sözlüğü **en başta** doğrular,
   `run_training` model/save kapılarını **derlemeden ÖNCE** çalıştırır ⇒ eksik argümanda **hiç yazım yok**.
3. `chat_prompt.py`: **`--vocab` bayrağı eklendi**; `--model`/`--vocab` zorunlu; eksikse `sys.exit(2)`
   (kapı mesajı `data/vocab.json`'un neden bayat olduğunu ve doğru örneği **açıkça** yazar).
4. **Beyan edilen ek kapsam (şartname "dört site" diyordu; ölçüm ikisini daha buldu):**
   - `/model` menüsü **silinmiş** checkpoint'leri seçenek olarak sunuyordu (CANLI ölü atıf sitesi) ⇒ emekliye ayrıldı.
   - `"kristal_carpenter" in model_path` koşulu, Kristal zinciri silinince **hiçbir yolda eşleşemez** hâle
     gelmişti ⇒ **marangoz talimatı sessizce hiç uygulanmıyordu**; ölü önek (`kristal_`) kaldırıldı.
5. `tests/test_agent_gateway.py`: mevcut test sözlüğü açıkça alıyor, canary'ye `vocab_path` verildi ve
   **iki yeni fail-closed canary** eklendi (`create_default` argümansız/yarı-argümanlı · `RetrainPipeline` argümansız).

---

## 4 · Kapılar ve kanıtlar

### K1 — kapsam ölçüldü ve beyan edildi ✓
Kapsam dışı bırakılan site **sessizce atlanmadı**: `src/llm/prompt_contract.py:35`
`vocab_path: str = "data/vocab.json"` (bayat) — tüketici yüzeyi geniş (testler + problar), uyuşmazlık
davranışı bu görevde ölçülmedi ⇒ **ayrı görev** olarak beyan edildi (§7).

### K2(a) — `data/kristal_model.pt` ölü atfı: **ÖLÜ = 0** ✓
| Dosya | ham | emekli (etiketli not) | **ÖLÜ** |
|---|---|---|---|
| `chat_prompt.py` | 1 | 1 | **0** |
| `src/gateway/agent_gateway.py` | 0 | 0 | **0** |
| `src/gateway/retrain_pipeline.py` | 0 | 0 | **0** |
| `src/rag/rag_pipeline.py` | 0 | 0 | **0** |
| **TOPLAM** | 1 | 1 | **0** |

Kalan tek ham satır **kendi emekli notumdur** (`EMEKLİ (T-0087)`), yani atıf değil **kayıttır**
([[emekli-notu-sayaci-sisirir]]).

### K2(b) — "küçük-harf `kristal` 4 dosyada **0 satır**": **İLAN HATALI** ✗
Ölçüm: **ham 26** satır; **23'ü meşru tanımlayıcıdır** ve **silinemez**:
`generate_kristal_vector` (gerçek fonksiyon adı, 9 satır) · `kristal_bellek` (gerçek Qdrant koleksiyon adı, 12 satır)
· `kristal_count` (`get_status()` değişkeni, 2 satır). Kalan 3 satır benim **etiketli** emekli notlarım.
**Kapı harfiyen imkânsızdı**; kusur **kodda değil ilanda**dır. Kapı, ölçümden sonra **değiştirilmedi**
(T-0067 kuralı: bulunan kusur rapora yazılır, tasarıma yansıtılmaz). K2'nin *niyeti* — "ölü **yol**
atfı sıfırlanmalı, `Kristal*` sınıf adları ölü atıf değildir" — K2(a) ile **kanıtlandı**.

### K3 — argümansız ⇒ sesli erken durma (`rc≠0`) ✓
| Site | rc | Kapı mesajı |
|---|---|---|
| `chat_prompt.py` (argümansız) | **2** | `Hata: --vocab verilmedi. Varsayilan KALDIRILDI (T-0087)…` |
| `AgentGateway.create_default()` | **1** | `RuntimeError: DURDURULDU: model_path verilmedi…` |
| `rag_pipeline.build_from_disk()` | **1** | `RuntimeError: DURDURULDU: model_path verilmedi…` |
| `src/rag/rag_pipeline.py` (`__main__`) | **1** | `RuntimeError: DURDURULDU: model_path verilmedi…` |
| `RetrainPipeline().run_training(steps=1)` | **1** | `RuntimeError: DURDURULDU: model_path verilmedi…` |
| `RetrainPipeline().compile_backlog_to_bin()` | **1** | `RuntimeError: DURDURULDU: vocab_path verilmedi…` |

**Muhafız:** `data/*.pt` + `data/*.bin` = **25 dosya**, koşu öncesi/sonrası sha256 ⇒ **DEĞİŞEN/YENİ = 0**.

### K3-poz — argüman verilince kapı **0 kez** ateşler ✓
| Site (argümanlı) | rc | Kapı ateşledi | Ne oldu |
|---|---|---|---|
| `chat_prompt.py --model <yok> --vocab <gerçek>` | 0 | **False** | sözlük yüklendi; durma **model dosyası** yokluğundan (eski kapı) |
| `create_default(model=<yok>, vocab=<gerçek>)` | 1 | **False** | `FileNotFoundError` (model yok) |
| `build_from_disk(model=<yok>, vocab=<gerçek>)` | 1 | **False** | `FileNotFoundError` (model yok) |
| `compile_backlog_to_bin(vocab=<gerçek>)` — **POZİTİF KONTROL** | 0 | **False** | **derleme BAŞARILI**, `.bin` gerçekten yazıldı |

Yani kapı **yalnız yoklukta** durur; argüman verilince yolu tıkamaz (pozitif kontrol bunu kanıtlar).

### K4 — bayat sözlük davranışı: **SESSİZ DEĞİL, YUTUCU** ✓ (görevin en önemli ölçümü)
Ölçüm **düzeltme ÖNCESİ üretim kodunun kendisiyle** yapıldı (`git show HEAD:src/gateway/agent_gateway.py`
birebir alınıp `create_default(model_path="data/anka_a1r.pt", vocab_path="data/vocab.json")` çağrıldı). Ham çıktı:

```
[SOYAGACI] yuklenen=…/data/anka_a1r.pt sha256=b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293 anahtar=101
[SOZLESME_UYARI] Parametre 'embedding.embedding.weight' boyut uyumsuzluğu: eski (33114, 768) -> yeni (31357, 768) (1757 satır kırpıldı (clipping))
[SOZLESME_UYARI] Parametre 'lm_head.weight' boyut uyumsuzluğu: eski (33114, 768) -> yeni (31357, 768) (1757 satır kırpıldı (clipping))
[SOZLESME_UYARI] Parametre 'lm_head.bias' boyut uyumsuzluğu: eski (33114,) -> yeni (31357,) (1757 satır kırpıldı (clipping))
[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).
### SONUC: KOSUM DURMADI. model.embedding satir=31357 (checkpoint 33114, sozluk 31357)
### YANI: 1757 satir SESSIZCE atildi ve cagiran bunu GORMEDI.
```

**Okunuşu:** kırpma *görünür* (3 uyarı) ama **durdurucu değil**, ve son satır **"tam eşleşme"** diyerek
operatöre kırpmanın olmadığını söylüyor. Çağıran `model.embedding` satır sayısını kontrol etmezse kayıp
**fark edilmez**. Mekanizma doğrulandı ve **iki kontrolle** sınırlandı:

| Çift | Fark | `[SOZLESME_UYARI]` | Koşum durdu mu? |
|---|---|---|---|
| `data/vocab.json` (31.357) ↔ `anka_a1r.pt` (33.114) | **+1757** | **3** | **HAYIR** |
| `vocab_base_32852.json` (32.852) ↔ `anka_a1r.pt` | +262 | 3 | **HAYIR** |
| `vocab_anka_r1_33114.json` (33.114) ↔ `anka_a1r.pt` | 0 | **0** | HAYIR (uyarı yok) |

Kontrol satırı, kapının **yanlış pozitif üretmediğini** gösterir: eşleşen çiftte hiç uyarı yoktur.
**Kapı bu yüzden `resize_state_dict`'in içine DEĞİL, yükleme sitelerine** konuldu — orada satır **ekleme**
(ısı-sıcak başlangıç) meşru yoldur; kırpma ise her zaman sözleşme ihlalidir.

### K5 — varsayılana yaslanan çağıranların ölçülen sonucu ✓
| Çağıran | Yaslandığı varsayılan | Ölçülen sonuç | Kırılma |
|---|---|---|---|
| `scripts/prepare_turk_tarihi_pipeline.py:184` `create_default(device="cpu")` | model+vocab | **zaten ölüydü**: varsayılan `data/kristal_model.pt` (YOK) ⇒ eskiden `FileNotFoundError`; şimdi `RuntimeError` | **YOK** |
| `scripts/train_literature_specialization.py:54` (aynı) | model+vocab | aynı | **YOK** |
| `scripts/run_agent_arena.py:49` | `--model` varsayılanı **`data/kristal_model.pt`** (ölçüldü) | **açık argüman** olarak geçiyor ⇒ kapı ateşlemez, davranış **değişmedi** | **YOK** |
| `scripts/run_agent_arena.py:81` `RetrainPipeline(device=…)` | vocab+model+save | `run_training` çağrılırsa artık **derlemeden önce** durur (ölçüldü: rc=1) | **YOK** (çağrı yolu zaten T-0085'ten beri ölü) |
| `scripts/run_goal_pipeline.py:177,309,387,402` | **açık** ölü yollar | eskiden `FileNotFoundError`, şimdi aynı | **YOK** |
| `src/rag/rag_pipeline.py __main__` | model+vocab | **kod değişti**: `--model`/`--vocab` geçiyor ⇒ kapı ateşlemiyor | **YOK** |
| `src/gateway/pedagogical_supervisor.py:631,878` `RetrainPipeline()` + `run_training(steps=50,batch_size=16)` | vocab+model+save | T-0085'ten beri zaten ölü (train.py `--save-path` zorunlu); **şimdi durma noktası öne alındı** ⇒ `data/train_future_finetune.bin` (`80340a98…`) artık **yazılmıyor** | **YOK** |
| `tests/test_agent_gateway.py:149` | vocab | **güncellendi** (görev kapsamı) | — |

**Ölçülen hüküm:** ilan edilen **Çürütme 1 gerçekleşmedi** — *çalışan* hiçbir repo-içi çağıran kırılmadı.

### K6 — `venv/bin/pytest` ✓ (ilan edilenden **+2 test** sapma, açıklamalı)
```
1 failed, 215 passed in 11.34s
FAILED tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints
```
- **Sapma:** ilanda "213 passed" yazıyordu; ölçüm **215 passed**. Fark **bu görevde eklediğim 2 canary**
  testidir (213 + 2 = 215). Sapma **beyan edilir**, sessizce yutulmaz.
- **Düşen tek test** ilan edildiği gibi **sandbox soket yasağı**dır (`socketserver` → `PermissionError:
  [Errno 1] Operation not permitted`), görevle ilgisizdir.
- **Ara koşum kanıtı:** kapı sırası kusuru düzeltilmeden önce **`214 passed, 2 failed`** — düşen ikinci test
  **benim yeni canary'imdi** (bkz. §5.2). Bu ara çıktı **silinmedi**, kanıt olarak burada duruyor.

### K7 — git disiplini ✓
`git add -A` / `git add .` **kullanılmadı**; commit/push **yapılmadı** (`6bca90b` push'u operatör yetkisindedir).
`changed_files` **iki yönlü** doğrulandı, `len(cf) == len(set(cf))` — bkz. §8 kapanış.

---

## 5 · Kendi kusurlarım (ölçümle bulundu)

1. **Sonda tespit dizesinde harf hatası.** K3 boolean tablosunu `'varsayilan KADIRILDI'` (küçük v) ile
   aradım; üretim mesajı **"Varsayilan KALDIRILDI"** (büyük V) basıyor ⇒ **ilk tablo her satırda `False`**
   çıktı, mesajlar ekranda dururken. Düzeltilmiş tespitle yeniden ölçüldü (§4 K3/K3-poz tabloları).
   Bu, [[etiket-adlari-buyuk-harf-tuzagi]] ailesinin **bu oturumdaki tekrarıdır**: sayı doğruydu, **iddia** yanlıştı.
2. **Kapı sırası kusuru — kendi testim düşürdü.** `compile_backlog_to_bin` içinde `vocab_path` kapısını
   `future_train_path` varlık kontrolünün **altına** koymuştum; fikstürde sıra `FileNotFoundError`a takıldı
   ve test `RuntimeError` beklerken düştü. Doğrusu: **yapılandırma hatası veri hatasından ÖNCE** raporlanır.
   Kapı **başa alındı** (ilan edilen tasarım değişmedi; sıra düzeltildi) ve düzeltme **öncesi** ara koşum
   kanıt olarak §4/K6'da duruyor.
3. **Ölçülemez bir kapı ilan ettim** (K2(b)) — kusur kodda değil **ilanda**; §4'te açıkça yazıldı.
4. **Araç kuralı kusuru (ölçümle bulundu).** T-0086 aracının (`scratch/anka_r9_olu_atif_denetimi.py`)
   emekli ayıklaması `s.startswith("#")` kullanıyor ⇒ **girintili yorum bloğunu görmüyor**, yani fonksiyon
   içindeki **etiketli not sayaçtan çıkarılmıyor**. Ölçüldü: `emekli=0` (araç kuralı) ↔ `emekli=1`
   (doğru kural: `s.strip().startswith("#")`). Bu, [[emekli-notu-sayaci-sisirir]]'in **üçüncü üremesidir**
   ve açık madde *"aracın etiket sözlüğünün genişletilmesi"*ne **ölçülmüş gerekçe** sağlar. Araç bu
   görevde **değiştirilmedi** (kendi görevi gerekir) ve **yeniden koşulmadı**: çıktı yolu tarihsel bir
   artefakttır (`data/eval/anka_r9_…`) — onu ezmek T-0086 kanıtını tahrif ederdi.
5. **Şartname kapsamı eksik kalmıştı:** "dört site" denmişti; ölçüm `chat_prompt.py` içinde **iki CANLI
   ölü-path sitesi daha** buldu (ölü checkpoint'i **seçenek olarak sunan** `/model` menüsü ve
   **hiç eşleşemez** hâle gelmiş `"kristal_carpenter"` dalı). İkisi de kapsama **beyan edilerek** alındı (§3.4).

---

## 6 · Çürütme maddeleri (önceden ilan edilmişti)

- **Çürütme 1** — *"Varsayılana yaslanan ÇALIŞAN bir repo-içi çağıranı kırarsa düzeltme YANLIŞTIR."*
  → **Gerçekleşmedi.** Sekiz çağrı yeri ölçüldü (K5); yaslananların hepsi ya **zaten ölüydü** ya da
  argümanı **açıkça** geçiyor. Tek davranış değişikliği supervisor yolunda: durma **öne alındı** ve
  `data/train_future_finetune.bin` artık **yazılmıyor**.
- **Çürütme 2** — *"Ölü varsayılanı kaldırmak yalnız zaten-ölü bir yolu daha erken ve daha açık
  durduruyorsa, kazanç YALNIZ mesaj kalitesidir; bunu 'kusur düzeltildi' diye çerçevelemek YANLIŞ olur."*
  → **İki yolda iki farklı hüküm** (ayrım açıkça yazılır):
  - **Model yolu (ölü checkpoint adı):** kazanç **erçeklik + mesaj**tır. Eski varsayılan da
    `FileNotFoundError` veriyordu ⇒ burada "kusur düzeltildi" demek **abartı** olurdu; doğru çerçeve
    **"ölü varsayılan kaldırıldı, durma yapılandırma hatasına dönüştü"**.
  - **Sözlük yolu (bayat sözlük):** kazanç **gerçek davranış**tır. K4 ölçtü: koşum **durmuyor**, 1757 satır
    atılıyor ve son satır **"tam eşleşme"** diyor. Bu bir **sessiz-yutucu** kusurdur ve kapı onu kapatır.

---

## 7 · Kapsam dışı bırakılanlar (gerekçeli beyan)

| Site | Neden dışarıda |
|---|---|
| `src/llm/prompt_contract.py:35` `vocab_path="data/vocab.json"` | tüketici yüzeyi geniş (testler + problar); uyuşmazlık davranışı bu görevde ölçülmedi ⇒ **ayrı görev** |
| `scripts/run_agent_arena.py:49` `--model` varsayılanı `data/kristal_model.pt` | `scripts/**` ÖLÇÜM/TARİHSEL betikleri **DEĞİŞTİRİLMEZ** (ölçüldü, rapor edildi) |
| `scripts/run_goal_pipeline.py:177,309,387,402` açık ölü yollar | aynı gerekçe (ölçüldü, davranış değişmedi) |
| `data/eval/**` tarihsel kayıtlar, T-0086 aracı ve çıktısı | geçmişi yeniden yazmak **kanıtı tahrif eder** |
| `USER_GUIDE.md:6` TARİHSEL banner · araç etiket sözlüğü · `6bca90b` push | T-0083 §8'in ayrı maddeleri; push **operatör yetkisinde** |

---

## 8 · Tam digest tablosu (önek değil, **tam** sha256)

### Değişen dosyalar (**8**, iki yönlü doğrulandı)
| Dosya | sha256 |
|---|---|
| `chat_prompt.py` | `21c10b5faa7194527fac22b24eb4ef8045333a22d1fe6125797639589048f1dc` |
| `src/gateway/agent_gateway.py` | `fd1226f776629c14e82bc6943da108f3eac790ecc332d58481a749a5e0d1c920` |
| `src/gateway/retrain_pipeline.py` | `7cbe4075c66ebd44aa6db0d1673fda7f6ca0a0a37f6e4f2875581d3b3f1634bd` |
| `src/rag/rag_pipeline.py` | `5f9b238063d5c080680056730076d1e5c2409ed808a5ad89ac7d32bf2782c854` |
| `tests/test_agent_gateway.py` | `c55c7b44b7a0ddb0d5be31f03b9d5e64df2cdf3ce89012a1491958871dbb85e5` |
| `.agent-bus/notes/T-0087.md` | `8432b58ad8695b5ee9b49a325a295da605ddcb890ccc326e91f8f8932de2f3ca` |
| bu rapor (`.md`) | *kendi digest'ini taşımaz* — kapanış bildiriminde kayıtlı |
| `…2026-09-20.json` | *kendi digest'ini taşımaz* — kapanış bildiriminde kayıtlı |

> **Önceden kirli olanlar (beyan):** `agent_gateway.py`, `retrain_pipeline.py`, `rag_pipeline.py` bu görev
> **başlamadan önce de** değiştirilmişti (T-0081/T-0085 işleri) ⇒ üzerine yazıldı. Oturum başında `M` olan
> `.agent-bus/SPEC.md`, `README.md`, `USER_GUIDE.md`, `train.py` bu görevde **DOKUNULMADI**.

### Bu raporun ölçtüğü, **değişmeyen** referans artefaktlar
| Artefakt | sha256 | Rol |
|---|---|---|
| `data/anka_a1r.pt` | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | K4'ün checkpoint'i (33.114) |
| `data/anka_a1.pt` | `f32d492d9361c668bc86e4e4b0f99f5c473af552486837452475f8b0aa3eb5b4` | A1 tabanı |
| `data/vocab.json` | `4b40eebee9c996c8dd3150f6b3c8896de85764ee1cf0b08e6c636e5c002a0e69` | **BAYAT sözlük** (31.357) |
| `data/rebuild/vocab_base_32852.json` | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` | K4 kontrolü |
| `data/rebuild/vocab_anka_r1_33114.json` | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` | **doğru** sözlük |
| `data/train_future_finetune.bin` | `80340a98b23381098bea27ec831e0c231e2efc8de4eca31396209fdfe15595ee` | supervisor yolu artık **yazmıyor** |
| `data/eval/t0069_pre_delete_manifest.json` | `4e1c4d51a896936909b1750b52088e8939e03e966b42fd8ea84934e0993e6e61` | ölü yol kanonik kaynağı |

### Dokunulmayan güvence
- **Donmuş 10 desen:** `data/*.pt` · `data/*.bin` · `data/vocab.json` · `data/vocab_entity.json` ·
  `src/llm/tokenizer.py` · `src/compiler/**` · `data/realistic_rag/**` · `data/b1_5_splits/**` ·
  `data/pedagogy_canonical/**` · `data/lexicon/**` → **hiçbirine YAZILMADI**.
- **Muhafız ölçümü:** `data/*.pt` + `data/*.bin` = **25 dosya**, koşu öncesi/sonrası sha256 ⇒ **0 değişen**.
- `data/rebuild/**` donmuş listede değil ama CLAUDE.md kural 3 gereği **salt-okunur** ⇒ bu görevde **yazılmadı**.

---

## 9 · Kapanış (üç sinyal)

İlan edilen kabul: *"K1…K7 kanıtlı; `changed_files` iki yönlü ve `len==len(set)`; kiralar boşaltılır;
`bus_send` bildirimi."*

1. **`bus_send` bildirimi** — gönderildi (mesaj dosyası adı kapanış kaydında).
2. **Kiralar boş** — `bus_lease_status` ⇒ `[]`.
3. **`changed_files` iki yönlü** — beyan edilen `writes[]` ile ölçülen değişiklik kümesi karşılaştırıldı;
   `len(cf) == len(set(cf))`; eksik ve beyan-dışı **yok**.

## 10 · Açık kalanlar (bu görev kapatmadı — beyan)

1. `src/llm/prompt_contract.py:35` bayat sözlük varsayılanı (ölçüldü, kapsam dışı) — **ayrı görev**.
2. **Araç etiket sözlüğü + girintili blok kuralı** (`s.strip().startswith("#")`) — §5.4 gerekçesiyle.
3. `USER_GUIDE.md:6` TARİHSEL banner cümlesi.
4. `scripts/run_agent_arena.py:49` ve `scripts/run_goal_pipeline.py` ölü varsayılanları (`scripts/**` kapsamı dışı).
5. `6bca90b` hâlâ `ahead 1`; **push operatör yetkisindedir** — `git add -A` yasak olmaya devam ediyor.
