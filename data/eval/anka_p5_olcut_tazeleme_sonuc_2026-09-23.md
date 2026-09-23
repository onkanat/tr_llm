# P5 · ÖLÇÜT TAZELEME SONUCU — kanonik kap eşiklerinin tazeleme denetimi (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Görev:** T-0099 · **Koşum YOK** (CPU ölçüm, ~1 dk)
· İlân: `anka_p5_olcut_tazeleme_ilani_2026-09-23.md` (ölçümden ÖNCE yazıldı;
sha256 `f12f84b2…`, Ekleme §9 dahil) · Sonda rc=**0**.

## 1. Kaynak denetimi — reproduce (betik: sonda içinde, kanıt JSON'da)

| kaynak | carve hükmü |
|---|---|
| `arena_base_accumulated.jsonl` | **899 kayıt · KIRLI 781 · temiz 118** (tamamı `highschool_genz`) |
| `arena_carpenter_accumulated.jsonl` | **302 · KIRLI 297 · temiz 5** ⇒ tavan kaynağı OLAMAZ |
| havuz | 5 dosya · 12.715 kayıt · 2.778 özgün çıktı (ilan §2 beyanıyla birebir) |

## 2. A-kolu — reproduce (FAIL-CLOSED): dört ölçütte BİREBİR

`heldout 539` · n=100 · seed 42 — kapanmış kayıt (`t0096_kos/kesisim_sondasi`
JETON tavanları) karşısında:

| ölçüt (JETON) | banka | ölçülen | fark | bant | hüküm |
|---|---|---|---|---|---|
| ROUGE-L | 0,4164 | **0,4164** | 0,0000 | ±0,0153 | BİREBİR |
| tutarsızlık | %3,0 | **%3,0** | 0,0000 | ±3,71 | BİREBİR |
| kesişim | %13,0 | **%13,0** | 0,0000 | ±6,61 | BİREBİR |
| ezber | %0,0 | **%0,0** | 0,0000 | birebir | BİREBİR |

Bant sinyali gereksizdi: ölçüm zinciri (kelimeler/tokenizer/vocab/lexicon)
değişmediyse birebir reproduce — kanıt, mekanizmanın deterministik olduğunu
kapanmış kayıttan bağımsız bir kez daha doğruladı.

## 3. B-kolu — yeni tavan (arena_base temiz 118, n=100 seed 42)

| temsil | ROUGE-L | tutarsızlık | kesişim (tanısal) | ezber |
|---|---|---|---|---|
| YUZEY | 1,0000 | %100 (yapısal — ham kelimede yüklemler etiketli olamaz) | %28 | %0 |
| **JETON** | **0,3835** | **%39** | %29 | %0 |

**İki sinyal:**
* **ROUGE tavanı A bandı DIŞINDA** (0,3835 vs 0,4164; fark 0,0329 > ±0,0153)
  ⇒ Dal-K ateşlendi.
* **Tutarsızlık tavanı %39** (banka %3,0). Bileşen ayrışması
  (`anka_p5_tutarsiz_ayrisma.py`): **%39'un tamamı yüklemler-eksik** (39/100),
  döngü **0/100** — JETON gidiş-dönüşü `highschool_genz` metninde yüklemleri
  `TENSE_/COPULA_` etiketi yerine düz kelime olarak kaybediyor. Bu,
  ilan §2'nin kesişim için ilan ettiği **alan-derinlik sınıfının aynısıdır**:
  tutarsızlık tavanı da alan-uyuşmaz kaynaktan kalibre EDİLEMEZ.

## 4. Hüküm — Dal-K (operatör kararıyla: ilan Ekleme §9)

Ön-kayıtlı Dal-K mekanik uygulaması ile ölçülen kanıt ROUGE için uyumlu,
tutarsızlık için ÇATIŞIYORDU (mekanik uygulama eşik 5,0 → **65,13** üretir ve
kapıyı fiilen öldürür). Operatör kararı (AskUserQuestion, 23 Eyl):

| eksen | tavan | karar | yeni eşik |
|---|---|---|---|
| ROUGE-L | 0,3835 (Dal-K) | **Dal-K UYGULANDI** | 0,35 → **0,3221** (0,3835 × 0,84) |
| tutarsızlık | %39 (tanısal) | mekanik Dal-K UYGULANMADI | **5,0 SABİT** |
| kesişim | %29 (tanısal) | ilan §2 aynen | **10,92 SABİT** |
| ezber | %0 | ilan §2 aynen | **10,0 SABİT** |

**Zincir güncellemesi (0d, tek geçiş — her halka koşarak kanıtlandı):**
1. `ECA:70` `ESIK_ROUGE = 0.3221` (satır sayısı korundu — G6 çıpası birebir).
2. `olcum_kabi.py` ESIK_REFERANSLARI `(70, ESIK_ROUGE, 0.3221)` + K11'e
   **eski eşik 0,35 DÜŞMELİ kanaryası** eklendi.
3. `tests/test_olcum_kabi_kapilari.py` eşik beyanı 0,3221'e güncellendi.
4. Tablo şablonu otomatik okudu: `ROUGE≥0.32` (kaynak eşik).

## 5. Doğrulama (üç kanal)

* `scripts/olcum_kabi.py` **rc=0** · 27 PASS · yeni eşik G6'da birebir
  (`:70 ESIK_ROUGE = 0.3221`) · **4/4 K11 kanaryası REDDEDİLME kanıtı**
  (dâhil: eski 0,35 eşiği).
* `venv/bin/pytest` **290 passed, 1 failed** — tek istisna bilinen
  `test_gateway_http_server_endpoints` (sandbox `socketserver`
  `PermissionError`; sandbox dışında geçer).
* `scripts/anka_karsilastirma_tablosu.py` rc=0 — eşik satırı kaynak
  eşiklerle birebir.

## 6. Çift hüküm (ayrı rapor: `anka_p5_p4_cift_hukum_2026-09-23.md`)

Temel 2.0 (0,0056) · T-0096 (0,3035) · P3 (0,1014) · P4 (0,1346) — **dört
hüküm değişmez** (hepsi iki eşikte ALTINDA). Marj notu: T-0096'nın 0,3035'i
yeni eşiğe 0,0186 kaldı — 0,32 bandına dokunan bir koşum artık kapıda GECER
hükmü verebilir (beyanlı davranış; eşik ölçülen tavana bağlı).

## 7. Digest tablosu (betikle hesaplandı)

| artefakt | sha256 |
|---|---|
| `scratch/anka_p5_tavan_sondasi.py` | `7326de5f…` |
| `scratch/anka_p5_tavan_sondasi.json` | `1daa0b32…` |
| `scratch/anka_p5_tutarsiz_ayrisma.py` | `27157e2a…` |
| `data/eval/anka_p5_olcut_tazeleme_ilani_2026-09-23.md` (Ekleme §9 dâhil) | `f12f84b2…` |
| `scripts/evaluate_carpenter_anka.py` (ECA, güncel) | `1f1e8375…` |
| `scripts/olcum_kabi.py` (güncel) | `8da6a9d6…` |
| `tests/test_olcum_kabi_kapilari.py` (güncel) | `62990cff…` |
| `scripts/anka_karsilastirma_tablosu.py` (güncel) | `fa638828…` |

*Kaynak digest'leri (ölçüm girdileri): ilan §2 tablosu; A-kolu çıpası
`heldout c397eb08…` P4 raporuyla birebir.*

## 8. Kapanış

**P5 KAPANDI:** eşik tazeleme ölçümü tamamlandı; A-kolu birebir (mevcut
zincirin mekanizması taze kanıtlandı), ROUGE Dal-K uygulandı (0,35 →
0,3221), tutarsızlık/kesişim/ezber eşikleri ölçülen kanıtla SABİT kaldı
(alan-derinlik sınıfı). Sıradaki koşum ancak yeni külliyat arzıyla açılır
(P4 teşhisi) — eşik tazelemesi o koşumun ölçüt zeminini güncelledi.

## 9. Kalan (operatör kararı bekliyor)

* P5 kayıtlarının commit/push onayı (ECA + olcum_kabi + test + tablo + ilan/
  sonuç/çift-hüküm raporları; scratch kayıtları gitignore'da — kanıt kanalı
  bu digest tablosu).
* Hafıza güncellemesi önerisi: "eşik tazelemede alan-derinlik sınıfı"
  (tutarsızlık tavanının %3→%39 ayrışması) kalıcı ders.