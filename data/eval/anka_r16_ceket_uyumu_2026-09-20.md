# T-0093 — Ceket Anka'ya uyuyor mu? (ölçüm raporu)

**Görev:** T-0093 · **Yürütücü:** claude · **Tarih:** 2026-09-20 · **`claimed_at`:** `2026-09-20T13:10:15Z`

**Operatör sorusu (aynen):** *"Test et Anka ya ceket uyuyormu ?"*

## 0. Sorunun doğru okunması

Kristal zinciri **SİLİNDİ** (18 Eyl operatör kararı) ⇒ **giyilecek ceket checkpoint'i YOK**; `data/*.pt` altında yalnız `anka_a1.pt` ve `anka_a1r.pt` var. Elde kalan "ceket" **veridir**: `data/train_carpenter_specialization.bin`. Test edilebilir soru şudur: **ceket verisi Anka'nın sözlüğü, hedefi ve temsiliyle uyumlu mu** — yani ceket Anka'ya **giydirilebilir** mi?

## 1. Sonuç tablosu (hepsi TAM külliyat üzerinde ölçüldü)

| # | ölçüm | sonuç | hüküm |
|---|---|---|---|
| **U1** | sözlük kimliği | ceket sözlüğü sha256 `7611b6a523a2bb52…` ↔ A1 meta beyanı `7611b6a523a2bb52…` | **AYNI DOSYA** |
| **U2** | önek değişmezliği | id 0…32851: **farklı id 0**; A1-r +262 id ekliyor | **KORUNDU** |
| **U3** | id geçerliliği | 867,650 jeton · max id 32850 · geçersiz id **cekette 0**, A1-r'de 0 | **TEMİZ** |
| **U4** | çözme anlamlılığı | `<BOS>`=12,763 = `<EOS>`=12,763 (**fark 0**), `record_count`=12.763 ile aynı; A1'de fark 1 | **İD KAYMASI YOK** |
| **U5** | canlılık (sıfırdan) | rc=**0** · ilk kayıp **10.5812** (ln V = 10.3998) · → 8.1759 | **CANLI** |
| **U6** | temsil | literal/PN: ceket **8,428.5** ↔ A1 **0.2902** | **AYRIŞIYOR** |

### U6 ayrıntı — temsil ekseni (TAM külliyat, örneklem değil)

| ölçüm | CEKET | ANKA A1 |
|---|---|---|
| jeton | 867,650 | 100,000,000 |
| `<PROPER_NOUN>` | **2** (%0.0002) | **7,936,521** (%7.9365) |
| literal ad | 16,857 (%1.9428) | 2,302,966 (%2.3030) |
| **literal / PN** | **8,428.5** | **0.2902** |
| `<UNK>` | 11,711 (%1.3497) | 2,629,227 (%2.6292) |
| `<ENT>` · `<CAP>` · `<ALL_CAPS>` | **1,088 · 924 · 164** | **0 · 0 · 0** |

A1'in `literal/PN` değeri kanonik T-0079 referansı **0,2902** ile **birebir** (ölçüldü: 0.2902) ⇒ ölçüm kabı doğrulandı. Ceketin literal ad **tipi** 1,354 = T-0078'in kanonik sayısı **1.354** ile birebir.

**Yön okuması:** literal ad **oranı** benzerdir (ceket, A1'in **0.8436 katı**) — asıl ayrışma **yer tutucudadır**: ceket adı **işaretler** (`<ENT><CAP>…</CAP></ENT>`) ya da literal bırakır, A1 ise `<PROPER_NOUN>`'a **çökertir** — A1'in yer tutucu **ORANI** ceketin **34,431 katı**. (Ham *sayı* oranı 3,968,260×'tir ama bu, külliyat boyu farkını (115.3×) temsil farkıyla **karıştırır** ⇒ kıyas **oran** üzerinden yapılır.) `literal_entity_mode` beyanı (ceket `true` / Anka `false`) böylece **meta'dan okunmadan, `.bin` jetonlarından ÖLÇÜLDÜ**.

### U5b — GERÇEK giydirme: Anka A1-r ağırlıklarının üzerine

U5 sıfırdan bir canlılık koşumudur; **giydirmek ise Anka'nın üzerine giymektir**. U2'nin önek invaryantı sayesinde ceket `.bin`'i A1-r sözlüğüyle **yeniden eşleme olmadan** kullanılabilir. Eşleştirilmiş kontrol: **aynı model** (`anka_a1r.pt`), **aynı protokol**, **aynı tohum**; yalnız **veri** değişir.

| veri (10 adım, `anka_a1r.pt` üzerine) | ilk kayıp | son10 ort ± std |
|---|---|---|
| **ceket** `train_carpenter_specialization.bin` | 7.6883 | **6.9998 ± 0.3251** |
| kontrol: Anka'nın **kendi** külliyatı `anka_a1r_pretrain.bin` | 4.035 | **3.9833 ± 0.2867** |

**Sonuç:** rc=0, şekil uyuşmazlığı uyarısı **çıkmadı** ⇒ ceket Anka'ya **sorunsuz giyiliyor**; gradyan canlı. Ama ceket Anka için **1.7573× daha zor** (+3.0165 nat). Bu, "uyuyor"un **bedelini** gösterir: ceket giyilebilir ama Anka'nın dağılımına **yakın değil**.

### U7 — ablasyon: 1,76× zorluk NEREDEN? *(ölçüm sırasında EKLENDİ, beyan edilir)*

Doğal hipotez: "Anka `<ENT>`/`<CAP>` kanalını hiç görmedi, zorluk oradan." Ceket makale sınırlarından ikiye bölünüp aynı protokolle ölçüldü.

| makale grubu | makale | jeton | jeton payı | son10 ort ± std |
|---|---|---|---|---|
| `<ENT>`/`<CAP>` **içeren** | 789 | 59,462 | %6.85 | **7.6779 ± 0.3148** |
| içermeyen | 11,974 | 808,188 | %93.15 | **6.9850 ± 0.2789** |

**Bulgu — hipotezin TERSİ:** markup makaleleri 0.6929 nat (≈%9.9) daha zor, ama jetonun yalnız %6.85'si oldukları için toplam 3.0165 nat'lık açığa ağırlıklı katkıları **0.0475 nat = açığın %1.57'si**. ⇒ **A1'in hiç görmediği temsil kanalı, maliyetin ana kaynağı DEĞİL.** Kalan %98.4 bu ablasyonla **ayrıştırılmadı** (aday: alan kayması · literal-ad temsili · `<INSTRUCTION>` biçimi); **ölçülmedi, iddia edilmiyor**.

## 2. İlan edilen çürütme maddeleri — ateşleme durumu

Maddeler **ölçümden önce** yazıldı ve **değiştirilmedi**.

| # | madde | durum | ölçüm |
|---|---|---|---|
| 1 | U1 U2 fark ise uymuyor | **ATEŞLENMEDİ** | U1 ayni_dosya=True · U2 farkli_id=0 |
| 2 | U4 cop ama pozitif temiz ise id kaymasi | **ATEŞLENMEDİ** | ÖNCÜL OLUŞMADI: pozitif kontrol de doğal düz metne çözülmüyor; iki külliyat da derlenmiş akışa çözülüyor ⇒ karşılaştırma tabanı yok |
| 3 | U5 kayip tam sifir ise olu | **ATEŞLENMEDİ** | ilk kayip=10.5812 |
| 4 | U6 iki oran ayni ise temsil farki yok | **ATEŞLENMEDİ** | ceket literal/PN=8428.5000 · A1=0.2902 ⇒ 29,046× |

## 3. Hüküm

**Mekanik uyum (U1–U5b): TAM UYUYOR.** TAM UYUYOR (5/5): ayni sozluk dosyasi · onek invaryanti fark 0 · gecersiz id 0 · cerceve dengeli · gradyan canli; **giydirme koşumu rc=0** ve şekil uyuşmazlığı yok (U5b).

**Temsil ekseni (U6): AYRIŞIYOR.** AYRISIYOR ve ayrisma BUYUK: ceket adi ISARETLER (<ENT><CAP>) veya literal birakir (PN=2 jeton), A1 <PROPER_NOUN>'a cokertir (PN=%7,94).

**Yön:** Ayrisma A1-r'nin GITTIGI YONLE AYNI: A1-r sozluge 254 literal ad ekledi.

> **İddia sınırı:** Bu olcum GIYDIRILEBILIRLIGI olcer, GIYILMIS KALITEYI degil. 'Uyuyor' != 'iyi calisiyor'.

**Giydirmenin bedeli (ölçülmedi, beyan edilir):** ceket 867.650 jetonluk **küçük** bir alandır; giydirmek replay karışımı gerektirir (`scripts/build_replay_mix.py`, ör. `replay_every=4` ⇒ %25 ceket) ve modele A1'in **hiç görmediği** bir temsil kanalını (`<ENT>`/`<CAP>`; A1'de 100 M jeton boyunca **0**) öğretir.

## 4. Kendi kusurlarım (bu ölçümde)

**K1 — v1 literal-ad filtresi etiket adlarını özel ad saydı.** `t[:1].isupper()` filtresi `POSS_3SG`/`CASE_LOC` gibi **morfem etiket adlarını** da geçirdi (külliyatın en sık jetonları) ⇒ v1 **%29,81** (ceket) / **%36,41** (A1) bastı. Kanonik filtre (`AFFIKS_ONEK` + `KONTROL`) ile düzeltildi ve **fail-closed** pozitif/negatif kontrolle sınandı (negatif küme temiz, pozitif küme 4/4; aksi hâlde `rc=2`). v1 çıktısı **silinmedi**, kanıt olarak saklandı (`79cddd82072a5b4e…`). Aynı aileden: T-0078.

**K2 — "`train.py` `--save-path`'ten SESSİZCE devralıyor" iddiam YANLIŞTI.** Aynı `--save-path` ile ikinci koşumda ilk kayıp 10,5812 yerine 4,8193 çıktı; bunu sessiz soyağacı devralma sandım. Kodu okuyunca mekanizma doğru ama **ilan edilmiş**: `train.py:226` `--load-path` verilmezse `model_load_path = model_save_path` yapar ve dosya varsa koşum *"Mevcut model ağırlıkları '…' tespit edildi, eğitim devam ettiriliyor (Resume)..."* + *"UYARI: AdamW momenti bulunamadi … (T-0092)"* satırlarını **basıyor**. Kusur kodda değil **benim grep penceremdeydi**. Aynı aile: T-0087.

**K3 — donmuş yüzey denetimim bir kez VAKUM geçti.** Kapanışta denetimi ikinci kez yazdığımda `frozen.json`'u **sözlük olarak** okumam gerektiğini kaçırdım: `patterns` bir **liste**, dosya ise **sözlük**. Betik sözlüğün **anahtarlarını** (`_comment`, `version`, `patterns`, `_rationale`) desen sandı ⇒ `os.walk` hiçbir şey bulamadı ve denetim **"taranan 0, değişen 0"** diyerek **geçti**. Sayı "temiz" görünüyordu ama **hiçbir dosyaya bakılmamıştı**. Fail-closed yeniden yazıldı: desen listesi okunamazsa veya `tarama == 0` ise betik **`rc=2` ile DURUR**; doğru sonuç **78 taranan / 0 değişen**. Aynı aile: T-0075 *"okunamayan sinyalde durmayan koruma kapı değildir"*.

## 5. Girdi digest tablosu (TAM digest — önek değil)

| dosya | bayt | sha256 |
|---|---|---|
| `data/train_carpenter_specialization.bin` | 1,735,300 | `13dd81bf700690b78ca15246a2fc1582ee6f8fea587c6d4e78f8ab1e1017b109` |
| `data/anka_a1_pretrain.bin` | 200,000,000 | `383a9c890b75ec96195122b9981d7a73371d3bcaa3b9f6e7c2a188510d917cb8` |
| `data/rebuild/vocab_base_32852.json` | 741,650 | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` |
| `data/rebuild/vocab_anka_r1_33114.json` | 747,458 | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| `data/anka_a1.pt` | 372,124,278 | `f32d492d9361c668bc86e4e4b0f99f5c473af552486837452475f8b0aa3eb5b4` |
| `data/anka_a1r.pt` | 373,735,137 | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `data/train_carpenter_specialization.bin.meta.json` | 714,695 | `b0088fd6ae8969f950006aa31e7797b405f27618bba80d825041d898f8013560` |
| `data/anka_a1_pretrain.bin.meta.json` | 1,028 | `6afd761dc49f57a908acc04eccbc1d50a02902817cc0702ed02a12afe4468320` |
| `data/anka_a1r_pretrain.bin.meta.json` | 1,383 | `6c7aab3290d2b49078e314478594cc57c5b1c44f7f76a576dbb5ceb7600c3ee7` |
| `scratch/t0093_probe.py` | 5,393 | `d6e598a82c22892197c84758dcb1b6126da1d05c21a81a0c60f91bad793e3cf6` |
| `scratch/t0093_probe_v2.py` | 6,013 | `aac3f5e312ff074a0a9919690e884258a58c2ed5418afa3266f9854fbb53ba25` |
| `train.py` | 25,710 | `ecf3a79e9e38600abb8068cbe11066d9e453dc4d83b44c46c604b01d6fabaa7f` |

## 6. Sınırlar ve açık işler

* **Donmuş yüzeye yazma 0** — ölçüldü (fail-closed, `tarama==0` ise betik DURUR): `frozen.json`'un **10** desenine uyan **78** dosya tarandı, ölçüt **mtime >= claimed_at (ZAMANSAL)** ⇒ değişen donmuş dosya **0**; `data/` altında `data/eval/` dışında değişen dosya **0**.
* U5 çıktısı yalnız `$TMPDIR/t0093/`'a yazıldı. `data/*.pt` **değişmedi**: `anka_a1.pt`, `anka_a1r.pt`.
* **Model eğitilmedi**; U5 tek bir 2 adımlık CPU sözleşme koşumudur. Dil kalitesine etki **ölçülmedi**.
* `venv/bin/pytest`: **1 failed, 237 passed in 77.25s (0:01:17)** (rc=1) — düşen test `test_gateway_http_server_endpoints`, nedeni sandbox soket yasağı (`PermissionError` `socket.bind`). CLAUDE.md'deki **damgalı** taban 213 passed, 1 failed (19 Eyl 2026) idi; artış bu arada kapanan T-0088/T-0090/T-0092'nin **eklediği** testlerdendir — bu görev **hiçbir test dosyasına dokunmadı** (`git status` ile doğrulanabilir).
* Bilinen ve **düzeltilmemiş** açık: `src/llm/frozen_guard.py:59` donmuş kapı **mutlak yolda FAIL-OPEN** (T-0092 ölçtü, operatör kararı bekliyor).
