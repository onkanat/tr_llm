# P5 · ÖLÇÜT TAZELEME İLANI (ön-kayıtlı) — kanonik kap eşiklerinin taze insan tavanıyla denetimi

**Damga:** 23 Eyl 2026 · **Yazım:** ölçümden ÖNCE · **Görev:** T-0099 (bus,
kiralamalar: `data/eval/` dir + `scratch/anka_p5_*` + koşullu `scripts/*`,
`tests/*`) · **Koşum YOK** — bu iş yalnız ölçüm + eşik kalibrasyonudur (CPU,
dakikalar). Plan: `~/.claude/plans/enchanted-wiggling-moon.md`.

## 1. Neden (ölçülmüş zemin)

Anka P4 (rc=0, `b8662629…`): ROUGE **0,1346** · tutarsızlık **%14** · kesişim
%2-6 · LM bedeli +%0,29 — kalibre eşiklerin (ROUGE 0,35 · kesişim 10,92 ·
tutarsızlık 5,0) tamamının altında. Üç maruziyet ekseni (adım T-0097 · pay
P3 · epoch P4) kapandı; kuşku ölçüte kaydı: **mevcut eşikler P2/0c'de
ölçülen tek tavan ölçümünden** kalibre (eşik = insan_tavani × k; ROUGE
k=0,84 · kesişim k=0,84 · tutarsızlık k=1,67). Kural gereği eşik ancak yeni
tavan ölçümüyle değişir — bu ilan o ölçümü tanımlar.

## 2. Kaynak denetimi (betikle, 23 Eyl 2026)

Carve kuralı **ÇIKTI-düzeyi birebir eşleşmedir** (arena kaydının `output`'u
aşağıdaki havuzun herhangi bir `output`'unda geçiyorsa KİRLİ). Havuz beyanı
(5 dosya, 12.715 kayıt, 2.778 özgün çıktı):

| kaynak | sha256 | carve hükmü |
|---|---|---|
| `data/pedagogy/carpenter_specialization_dataset.jsonl` | `146a73dc…` | havuz üyesi |
| `data/pedagogy/high_school_foundation_dataset.jsonl` | `59e2f278…` | havuz üyesi |
| `data/pedagogy_canonical/carpenter_canonical.jsonl` | `d0d97039…` | havuz üyesi |
| `data/pedagogy_canonical/high_school_canonical.jsonl` | `8bbdc5c2…` | havuz üyesi |
| `data/eval/anka_r17_heldout_2026-09-20.jsonl` | `c397eb08…` (P4 çıpası birebir) | havuz üyesi |
| `data/pedagogy/arena_base_accumulated.jsonl` | `37652795…` | **899 kayıt · KIRLI 781 · temiz 118** (tamamı `highschool_genz`) |
| `data/pedagogy/arena_carpenter_accumulated.jsonl` | `317586ca…` | **302 kayıt · KIRLI 297 · temiz 5** |

**Kararlar:**
* **B1:** arena_carpenter tavan kaynağı **OLAMAZ** (297/302 eğitimde görülmüş;
  r18 ilanı aynı yargıyı beyan etmişti). Elleme.
* **B2:** ROUGE ve tutarsızlık borusu-tavanları `arena_base` **temiz 118**
  kayıttan yenilenir (n=100, seed 42). Kesişim tavanı **BU P5'TE
  ALINMAZ/TANISALDIR**: kayıtların tümü `highschool_genz` domain'i, kesişim
  girdi-çıktı kelime paylaşımı alan-derin bir ölçüttür — domain uyuşmaz
  kaynaktan kalibre EDİLEMEZ (mevcut 10,92 kapanmış kayıttan kalır).
* **Ezber eşiği** (%10) sabit — dokunulmaz.

## 3. Yöntem (SABİT — P2 deseni kopya, katsayılar değişmez)

* n=100 · seed 42 · `rng.sample` · `kelimeler` ve `lcs_f1` ECA'dan **İMPORT**
  (kopya YASAK); `rouge_l_score` `evaluate_b1_5_rigorous`'tan İMPORT.
* **Çift temsil:** YUZEY (`gw = kelimeler(output)`) + JETON (`gw =
  kelimeler(" ".join(tokenizer.encode(output))`); JETON yolun ölçütteki
  tanımlı yoludur (ECA:478-481 deseni). **Kapı JETON'dan**; YUZEY
  gösterim-uyuşmazlığı denetimidir (P2/0b: LCS-F1 23/100 ayrışma ⇒ yalnız
  TANISAL).
* Tutarsızlık deseni: `scratch/t0096_kos/kesisim_sondasi.py:111-117`
  (yüklemin son konumda yokluğu VEYA 2×2 token döngüsü) — kapanmış dosyaya
  DOKUNULMAZ, desen kopyalanır.
* **İstatistik beyanı:** oran ölçütleri (tutarsızlık · kesişim · ezber) için
  **Wilson %95** aralığı; ROUGE ortalaması için **±1,96·std/√n**.

## 4. İki kol

### A-kolu — reproduce + FAIL-CLOSED
`data/eval/anka_r17_heldout_2026-09-20.jsonl` (539), n=100 seed 42 — kapanmış
kaydın (`scratch/t0096_kos/kesisim_sondasi.py`, sha256 `ae84b81e…`
betiğiyle üretilmiş JSON) **JETON tavanlarının reproduce** edilmesi:

| ölçüt | beklenen (kapanmış kayıt) | kabul bandı |
|---|---|---|
| ROUGE-L (JETON) | **0,4164** | reproduce ± 1,96·std/√100 |
| tutarsızlık (JETON) | **%3,0** | reproduce Wilson %95 |
| kesişim (JETON) | **%13,0** | reproduce Wilson %95 |
| ezber (JETON) | %0,0 | birebir |

**A-kolu bandı dışına çıkarsa sonda rc≠0 (FAIL-CLOSED)** — P5 durur, nedeni
rapora yazılır. (Not: `kelimeler`/tokenizer zinciri değişmediyse birebir
çıkması beklenir; band yalnız beklenmedik ortam farkına karşı sigortadır.)

### B-kolu — yeni tavan (Dal-K kaynağı)
`arena_base` **temiz 118** kayıttan `rng.sample` n=100 (seed 42, dosya
sirası deterministik). Her kaydın **kendi çıktısının** JETON/YUZEY
temsilinin kendi çıktısına ROUGE'u (tavan mekanizması aynen: gidiş-dönüş
kaybı tavanıdır — BORU tavanı kaynaktan bağımsız), tutarsızlık ve
kesişim/ezber (tanısal) aynı desenle ölçülür.

## 5. Dallar (ölçüm ÖNCESİnde ilanlı; en fazla bir dal uygulanır)

* **Dal-T (taze kanıt):** B-kolu ROUGE ve tutarsızlık tavanları, A-kolu
  reproduce değerlerinin yukarıdaki bantları İÇİNDE ⇒ **eşik DEĞİŞMEZ**;
  zincir güncellenmez; sonuç = "mevcut eşikler taze insan-tavanı kanıtıyla
  doğrulandı" (birinci-derece olası sonuç).
* **Dal-K (tavan bant dışı):** ROUGE veya tutarsızlık tavanı A-kolu bandının
  DIŞINDA ⇒ eşik = yeni_tavan × aynı k (ROUGE 0,84 · tutarsızlık 1,67); 0d
  zincir güncellemesi (ECA:68-71 satır sayısı KORUNARAK → `olcum_kabi.py`
  ESIK_REFERANSLARI/K11 kanaryaları → test → tablo şablonu; her halkada
  mutasyon kanıtı) + P4/P3 sonda JSON'larının çift hükmü
  (`data/eval/anka_p5_p4_cift_hukum_2026-09-23.md` — eski hükümler
  DÜZENLENMEZ).
* Kesişim ve ezber eşikleri her iki dalda SABİT (§2 kararlar).

## 6. Beklenti (ilandan okunur; ölçüm sonrası elle yazılmaz)

* A-kolu: §4 tablosundaki değerler birebir (bu ilanın kabul bandı).
* B-kolu: sayısal beklenti YAZILMAZ (tek ölçekli yeni kaynak); yalnız
  **yön beklentisi** beyan edilir: ROUGE/tutarsızlık boru tavanları
  kaynak-bağımsız olduğundan A-kolu değerlerinin yakınında (bant içinde)
  çıkması beklenir — Dal-T birinci-derece olası. B tavanı beklenmedik DÜŞÜK
  çıkarsa Dal-K eşiği düşürür — bu DOĞRU davranıştır (ulaşılamaz eşiğin
  önlenmesi; Ç4 dersi).

## 7. Çıktılar

* `scratch/anka_p5_tavan_sondasi.py` → `scratch/anka_p5_tavan_sondasi.json`
  (carve kanıtı + A/B tavan tabloları + dal kararı girdileri).
* `data/eval/anka_p5_olcut_tazeleme_sonuc_2026-09-23.md` (A/B tablo, dal
  kararı + gerekçe, digest tablosu).
* Dal-K ise §5'teki zincir güncellemesi + çift hüküm raporu.
* Doğrulama: `venv/bin/pytest` yeşil (bilinen gateway istisnası hariç) ·
  `scripts/olcum_kabi.py` rc=0 · `scripts/anka_karsilastirma_tablosu.py`
  eşik satırı kaynak eşiklerle birebir.

## 8. Beyanlar

* `scratch/t0096_kos/**` ve tüm kapanmış kayıtlar DOKUNULMAZ.
* Eğitimde görülmüş kaynaktan (arena_carpenter kirli 297 ·
  carpenter_canonical) tavan ÖLÇÜLMEZ.
* Donmuş yollara yazım YOK · koşum YOK · `git add -A` YASAK.
* Bu ilan ölçümden ÖNCE yazıldı; sonraki hiçbir belge eşikleri bu ilan
  dışında değiştiremez (önce bu ilan güncellenir, sonra zincir).
---

## 9. EKLEME (ölçüm SONRASI, 23 Eyl 2026) — operatör kararı: dal uygulaması

Ölçüm `scratch/anka_p5_tavan_sondasi.py` (rc=0) tamamlandı: A-kolu dört
ölçütte **birebir reproduce** (ROUGE 0,4164 · tutarsızlık %3,0 · kesişim
%13,0 · ezber %0,0 — fail-closed bandı geçti). B-kolu: ROUGE tavan
**0,3835** (A bandı dışı, fark 0,0329 > ±0,0153), tutarsızlık tavan **%39**
(banka %3,0). Bileşen ayrışması: %39'un **tamamı yüklemler-eksik**
(JETON gidiş-dönüşü `highschool_genz` metninde yüklemleri düz kelime
olarak kaybediyor), döngü **0/100**.

**Operatör kararı (AskUserQuestion, 23 Eyl):**
* **ROUGE:** Dal-K UYGULANIR — eşik 0,35 → **0,3221** (= 0,3835 × k=0,84).
* **Tutarsızlık:** mekanik Dal-K (eşik 65,13) **UYGULANMAZ** — ölçülen
  kanıt, tutarsızlık ölçütünün de alan-derin olduğunu gösterdi (kesişimle
  aynı sınıf, ilan §2'nin kesişim gerekçesiyle aynı mantık); tutarsızlık
  tavanı **TANISAL** beyan edilir, eşik **5,0 SABİT** kalır.
* Kesişim ve ezber: ilan §2'deki sabitlik aynen geçerli.

Bu ekleme ön-kayıt metnini SİLMER; mekanik dal uygulamasından sapmanın
kaydıdır (kanıt: `scratch/anka_p5_tavan_sondasi.json` + sonuç raporu).
