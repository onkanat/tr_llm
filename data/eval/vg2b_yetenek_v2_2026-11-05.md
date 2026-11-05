# VG-2b — Vektörel Gezgin v2: Yetenek Ağı v2 — 2026-11-05

## İLAN

İş emri 2 (arka-log kalemi 2) uygulanır: yetenek ağı **v2** — (1) TELEMETRİ:
aktivasyon-olay kaydı JSONL + SONUÇ SİNYALİ (aktivasyon zirvesindeki yeteneğin
alanı ↔ router'ın seçtiği uzman uyumu: uyum → +1, uyumsuz → −1, zemin yoksa
kayıt-yalnız); (2) AĞIRLIK EVRİMİ: deterministik bantlı kural ×(1±eps),
eps=0,05, bant [0,5; 2,0], idempotent payload geri-yazımı + geçmiş telemetriye;
(3) GATEWAY GÖZLEM→DANIŞMA yükseltmesi: **OP-İN, default KAPALI** — açıkken
zirve+uyum+sinyal META'ya yazılır ve telemetriye kaydedilir, uyumsuzluk
yalnız LOGLANIR, **router KARARI DEĞİŞMEZ** (P2.8 router kanıtlı); (4) KISIT
KARTLARI: 'kisit' tipi BASKILAYICI kartlar (çarpan ≤ 1 — aktivasyon skorunu
DÜŞÜRÜR, asla yükseltmez), 5 GERÇEK kısıt (repo kuralları, kanıt dosyaları,
uydurma yok), etki = ağırlık × benzerlik × kısıt-çarpanı; (5) 5-sorgu demo
tekrarı (v1 sorguları) + kısıt etkisi + rapor.

SERT KURALLAR (başa bağlı): **commit YOK** (controller'a devredilir); Qdrant
yazımı **YALNIZ `yetenek_agi`** koleksiyonu (kısıt ekimi + ağırlık
güncellemeleri — kurulumdaki fail-closed ad-kapısı zorlar); **mevcut
davranış-default'ları birebir** (kısıtsız koleksiyonda aktivasyon sonucu v1
ile anahtar-setine kadar aynı; gateway danışması default kapalı); TDD (RED
önce — ilk koşum `ImportError: AGIRLIK_MAX` ile RED); minimal-diff.

## HÜKÜM

**GEÇTİ.** Kanıtlar (hepsi koşum-çıktısı, uydurma yok):

| Kapı | Ölçüm | Sonuç |
|---|---|---|
| TDD (v2) | `tests/test_yetenek_agi_v2.py` | **52/52 GEÇTİ** (RED: ilk koşum `ImportError` → GREEN) |
| v1 birebir | `tests/test_yetenek_agi.py` (dokunulmadı) | **33 passed, 1 skipped** — v1 sözleşmeler değişmedi |
| Tam süit | `venv/bin/python -m pytest tests/ -q` | **641 passed, 1 skipped, 1 xfailed, 0 failed** (5:16) |
| Gerçek koleksiyon dumanı | `QDRANT_URL=…:6333 -k Dumani` | **GEÇTİ** (41 nokta: 36 yetenek + 5 kısıt; evrilmiş ağırlıklarla) |
| Kısıt ekimi (idempotent) | demo koşum 1: 36→41; koşum 2: 41→41 | upsert ÇOĞULTMAZ (nokta kimliği sha256-türeli) |
| Demo birebir-öncesi | 5 v1 sorgusu, eşik=0,02, top_k=3 | zirveler+skorlar+etkiler **v1 probuyla birebir** (kısıtlar ekiliyken bile top-3 değişmedi) |
| Ağırlık evrimi | 8 olay (4 uyum + 3 uyumsuz + 1 kayıt-yalnız) | 7 ağırlık değişimi; read-back koleksiyonda doğrulandı |
| Evrim geçmişi telemetri | `data/eval/yetenek_telemetri.jsonl` | 26 satır: 19 aktivasyon + 7 `agirlik_evrimi` kaydı (kimlik, eski→yeni, sinyal, eps) |
| Yeniden-oynatma determinizmi | birim testi: aynı olaylar 2× (zamansız) | **birebir aynı geçmiş** (TestEvrimIsle) |
| Gateway default KAPALI | birim testleri | anahtar YOK + telemetri YAZILMAZ; kararı-değiştirmez testi birebir eşit |
| Determinizm (canlı) | demo: aynı sorgu 2× | **SAĞLANDI** |

## RAPOR

### 1) Telemetri formatı — `data/eval/yetenek_telemetri.jsonl` (GÖREV-1)

JSONL; satır = tek JSON olay; `sort_keys=True` → aynı olay birebir aynı satır.
İki olay tipi:

**`aktivasyon`** (her aktivasyon gözlemi; gateway danışması açıkken veya
`evrim_isle`/demo açıkça yazdığında):
```json
{"olay": "aktivasyon", "zaman": "2026-11-05T03:22:26+00:00",
 "kaynak": "gateway|demo_once|demo_simulasyon|evrim|…",
 "sorgu_ozeti": "zımpara nedir",                       // ≤120 karakter
 "aktif_yetenekler": [{"ad": "Marangoz Alan Ceketi (carpenter)",
   "kimlik": "marangoz_ceketi", "skor": 0.032606, "etki": 0.040758}, …],
 "zirve_kimligi": "marangoz_ceketi", "zirve_uzmani": "carpenter",
 "router_uzmani": "carpenter",                          // yoksa null
 "uyum": true, "sinyal": 1,                             // +1 / −1 / null
 "karar_notu": "VG-2: 3/36 kart etkin (eşik=0.02, top_k=3); …",
 "kisit_etkileri": {"uygulanan": {"…": 0.5}, "kisitlar": ["alan_safligi", …]}}
```

**`agirlik_evrimi`** (her ağırlık değişimi — GÖREV-2 geçmişi):
```json
{"olay": "agirlik_evrimi", "zaman": "…", "kaynak": "demo_simulasyon",
 "kimlik": "marangoz_ceketi", "eski_agirlik": 1.25, "yeni_agirlik": 1.3125,
 "sinyal": 1, "eps": 0.05}
```

**SONUÇ SİNYALİ** (`sonuc_sinyali`, src/rag/yetenek_agi.py:812): zirve kartın
router-uzman eşleşmesi (KIMLIK_UZMAN_HARITASI: alan ceketleri → kendi uzman
sınıfları, derleyici/tokenizer çekirdeği → grammar_core). Uyum → `sinyal=+1`;
uyumsuz → `sinyal=−1`; router yok VEYA zirve kart uzman kümelerine
eşlenmiyorsa (ör. vektör bellek yönetimi) → `uyum/sinyal=null` — **kayıt-yalnız,
ağırlık DEĞİŞMEZ** (karşılaştırma zemininin olmadığını cezaya çevirmek dürüst
değil).

### 2) Ağırlık evrimi — kural + simülasyon sonuçları (GÖREV-2)

**Kural** (`agirlik_evrimi`, satır 929): saf, deterministik, bantlı —

```
uyum (+1)    → agirlik × (1 + eps)        eps = 0,05 (EVRIM_EPSILON)
uyumsuz (−1) → agirlik × (1 − eps)
sinyal yok   → DEĞİŞMEZ (kayıt-yalnız)
bant         → [AGIRLIK_MIN=0,5; AGIRLIK_MAX=2,0] kırpılır, 6 basamak yuvarlanır
```

Geri yazım: `YetenekAgi.agirlik_yaz` (satır 582) — nokta kimliği sabit
(sha256-türeli) → **idempotent upsert, çoğaltma yok**; doc/vektör BİREBİR
korunur (embedding metni ağırlık içermez; vektörler payload'daki
`token_ids`/`crystal_tags`'den yeniden üretilir — tokenizer gerekmez); bant
dışı ağırlık ve bilinmeyen kimlik **fail-closed ValueError**. YALNIZ
`yetenek_agi` koleksiyonu (kurulum ad-kapısı).

`evrim_isle` (satır 645): olayları canlı işler — aktivasyon → zirve→uzman
sinyali → telemetri kaydı → sinyal ±1 ise zirve kartın ağırlığı bantla
güncellenir → geçmiş telemetriye. Saf çekirdek `evrim_uygula` (satır 949)
sözlük→sözlük çalışır → **yeniden-oynatma determinizmi birim-testli**
(aynı olay dizisi 2× → birebir aynı geçmiş; bilinmeyen kimlik fail-closed).

**Demo simülasyonu** (8 olay; sinyaller SİMÜLE — 4 gerçek-beklenti uyum,
3 kasıtlı yanlış-yönlendirme, 1 kayıt-yalnız; zirveler CANLI koleksiyondan):

| Olay | Sorgu | Router (simüle) | Zirve (canlı) | Sinyal |
|---|---|---|---|---|
| 1 | 'zımpara nedir' | carpenter | marangoz_ceketi (carpenter) | **+1** |
| 2 | 'iş kanunu maddesi' | legal | legal_ceketi (legal) | **+1** |
| 3 | 'gideceğim kelimesini çözümle' | grammar_core | kristal_derleyici (grammar_core) | **+1** |
| 4 | 'ceviz ağacı özellikleri' | gardener | bahcivan_ceketi (gardener) | **+1** |
| 5 | "Qdrant'ta ne var" | grammar_core | vektor_bellek_yonetimi (uzman YOK) | **null** — kayıt-yalnız |
| 6 | 'zımpara nedir' | pedagogy ✗ | marangoz_ceketi | **−1** |
| 7 | 'iş kanunu maddesi' | carpenter ✗ | legal_ceketi | **−1** |
| 8 | 'ceviz ağacı özellikleri' | legal ✗ | bahcivan_ceketi | **−1** |

**Ağırlık öncesi → sonrası** (read-back, gerçek koleksiyon):

| Kart | Öncesi | Sonrası | Sinyal dizisi |
|---|---|---|---|
| marangoz_ceketi | 1,25 | **1,246875** | +1 sonra −1 (net ×0,9975) |
| legal_ceketi | 1,25 | **1,246875** | +1 sonra −1 |
| bahcivan_ceketi | 1,25 | **1,246875** | +1 sonra −1 |
| kristal_derleyici | 1,10 | **1,155** | yalnız +1 (×1,05) |
| vektor_bellek_yonetimi | 1,10 | **1,10 (değişmedi)** | kayıt-yalnız |

**Evrim öncesi/sonrası ZİRVE etki farkı** (aynı 5 sorgu):

| Sorgu | Zirve | etki öncesi → sonrası |
|---|---|---|
| 'zımpara nedir' | Marangoz Ceketi | 0,040758 → 0,040656 |
| 'iş kanunu maddesi' | Legal Ceketi | 0,193828 → 0,193343 |
| 'gideceğim kelimesini çözümle' | Kristal Derleyici | 0,256260 → **0,269073** (+%5 — tek saf-uyum kartı güçlendi) |
| 'ceviz ağacı özellikleri' | Bahçıvan Ceketi | 0,134761 → 0,134424 |
| "Qdrant'ta ne var" | Vektör Bellek Y. | 0,287752 → 0,287752 (değişmedi — kayıt-yalnız kanıtı) |

Ham tablolar: `scratch/vg2b_demo.json`. Not: evrilmiş ağırlıklar GERÇEK
koleksiyonda kalır (v2'nin amacı bu); varsayılanlara dönmek için
`scripts/seed_yetenek_agi.py --url http://192.168.1.9:6333` yeniden koşulur
(idempotent upsert ağırlıkları envanter varsayılanına geri yazar).

### 3) Gateway DANIŞMA kancası — OP-İN, default KAPALI (GÖREV-3)

`src/gateway/agent_gateway.py` (minimal diff, +44 satır):

- **:34** import: `KANCA_TOP_K, TELEMETRI_YOLU, danisma_raporu`
- **:68-69** `__init__`'e opsiyonel parametreler: `yetenek_danisma: bool =
  False`, `yetenek_telemetri_yolu: str = TELEMETRI_YOLU` (en sona ekli —
  konumsal çağrı kırılmaz)
- **:89-91** saklama + sözleşme yorumu (default KAPALI — mevcut davranış birebir)
- **:385-418** `ask()` içinde: v1 gözlem bloğu **birebir** (`yetenek_aktivasyonu`
  anahtarı ve hata→None sözleşmesi korunur) + içte OP-İN danışma bloğu

**Default KAPALI kanıtı** (birim testi, `TestGatewayDanisma`):
`yetenek_danisma` verilmeden → yanıtta `yetenek_danisma` anahtarı **HİÇ YOK**
(test_danisma_default_kapali_anahtar_yok) ve **telemetri dosyası YAZILMAZ**
(test_danisma_default_kapali_telemetri_yazmaz). `scripts/baslat_canli_gateway.py`
DOKUNULMADI — canlı açılış v1 gözlem kancasıyla birebir sürer.

**Açıkken**: aktivasyon zirvesi + router uzmanı + uyum/sinyal
`sonuc["yetenek_danisma"]` META'sına yazılır (`danisma_raporu`,
yetenek_agi.py:898), olay telemetriye kaydedilir (`kaynak="gateway"`).
**Uyumsuzlukta yalnız LOG**:
```
[VG2B_DANISMA] uyumsuzluk: aktivasyon zirvesi 'Legal Alan Ceketi (hukuk)' → 'legal' ama router 'carpenter' seçti — KARAR DEĞİŞMEDİ (yalnız kayıt; P2.8 router kanıtlı)
```
**Router KARARI DEĞİŞMEZ kanıtı** (test_danisma_karari_degistirmez_ve_uyumsuzluk_loglanir):
aynı sorgu kapalı vs açık koşumda `active_expert`, `router_experts` VE
`response_text` birebir eşit; uyumsuz senaryoda bile eşit. Danışma/aktivasyon
hatası ask'i bozmaz (dürüst None). Kararı gerçekten değiştirmek AYRI bir
ölçüm işidir — v2 bunu bilinçli olarak YAPMAZ.

### 4) Kısıt kartları — 5 GERÇEK kart, baskılayıcı (GÖREV-4)

`kisit_envanteri()` (yetenek_agi.py:994) — her kart repo kuralı + kanıt
dosyası; `tip='kisit'`, etki **BASKILAYICI** (çarpan ≤ 1, asla yükseltmez —
çarpan > 1 fail-closed ValueError):

| Kimlik | GERÇEK kural | Kanıt (repo'da VAR) | Koşul | Hedef | Çarpan |
|---|---|---|---|---|---|
| donmus_sozluk_33_114 | Donmuş sözlük 33.114 — yeni jeton yok (roots.tsv/compiler/tokenizer donmuş yol desenleri; vocab↔checkpoint sözleşmesi kilitli) | `.agent-bus/frozen.json` | hedef_etkin | sozluk_genisletme | **0,5** |
| alan_safligi | Ceket dışı sorguda alan kartı baskılanır (ölçülmüş gürültü bandı 0,005–0,024 < bant 0,03 < ilgili isabet dibi 0,033) | `scripts/yetenek_probe.py` | gurultu_bandi (bant=0,03) | alan ceketleri (hedef_alan=['alan']) | **0,5** |
| hukum_unk_yasagi | 'hükmü' UNK — hukuk şablonlarında kullanılmaz (ölçülmüş) | `scripts/prepare_legal_dataset_v3.py` (satır 90) | hedef_etkin | sft_veri_uretimi | 0,9 |
| yazma_yetkisi_fail_closed | Bilinmeyen koleksiyona yazım yok (T-0148 5B) | `src/gateway/agent_gateway.py` | hedef_etkin | bilgi_enjeksiyonu | 0,9 |
| uretim_tavani_200 | Üretim tavanı 200 jeton (`safe_max_tokens=min(200,…)`, DoS/VRAM kilidi) | `src/gateway/agent_gateway.py` | hedef_etkin | ajan_kapisi | 0,9 |

**Motor uygulaması** (`aktivasyon_vektor_ile`, satır 528): havuzdaki
`tip='kisit'` noktaları yetenek adayı OLMAKTAN çıkıp KURAL olarak uygulanır:
`etki = ağırlık × benzerlik × kısıt-çarpanı` (`kisit_carpanlari`, satır 773 —
çarpım ≤ 1, kimlik-sıralı deterministik). Kısıt kartının KENDİSİ
`aktif_yetenekler`'e girmez. **Kısıtsız koleksiyonda sonuç anahtar-setine
kadar v1 ile birebir** (test_kisitsiz_bellekte_sonuc_v1_birebir) — v1
dağıtımları etkilenmez.

**Demo etkisi** (tek sorgu: 'sözlüğe yeni kelime ekle bilinmeyen kelime azalt'):
- **Sözlük Genişletme**: ham etki 0,239520 → **0,119760** (×0,5 donmuş sözlük)
  → sıra **1'den 3'e DÜŞTÜ** (baskılamasaydı zirve olacaktı; zirve Kristal
  Derleyici 0,190533). Kısıt kartı sırayı kanıtlı değiştirdi.
- Aynı sorguda **alan_safligi** gürültü bandındaki alan kartlarını da baskıladı
  (Bahçıvan/Pedagoji/Legal ×0,5 — ham skorları 0,03 altında), yazma yetkisi
  Bilgi Enjeksiyonu ×0,9, üretim tavanı Ajan Kapısı ×0,9 — `kisit_etkileri`
  alanında raporlanır.
- Buna karşın 5 v1 sorgusunun top-3'ü kısıtlar ekiliyken **birebir korundu**
  ('zımpara' 0,0326 > bant 0,03 → marangoz dokunulmaz; diğer hedefler ya eşik
  altında ya top-3 dışı) — birebir korunum dumanla kanıtlı.

### 5) Testler — `tests/test_yetenek_agi_v2.py` (52 test, TDD: RED önce)

- **TestTelemetri** (11): JSONL yaz + read-back sayı; deterministik serileştirme;
  sinyal uyum +1 / uyumsuz −1 / router-yok ve zirve-uzmansız kayıt-yalnız; olay
  formatı; sorgu-özeti kesimi; danışma raporu telemetri kaydı.
- **TestAgirlikEvrimiKurali** (8): ×(1±eps) aritmetiği; üst/alt bant kırpma;
  sinyalsiz değişmez; bant sabitleri; saf `evrim_uygula` + yeniden-oynatma
  determinizmi; geçmiş kaydı; bilinmeyen kimlik fail-closed.
- **TestAgirlikYaz** (6, yerel Qdrant storage_path): yaz + read-back;
  idempotent ikinci yaz (sayı sabit); bant dışı fail-closed ×2; bilinmeyen
  kimlik; etki∝ağırlık uçuşu (skor sabit).
- **TestEvrimIsle** (3): uyum/uyumsuz/kayıt-yalnız döngüsü (telemetri 3
  aktivasyon + 2 evrim kaydı); koleksiyona yazım + çoğaltmama;
  **yeniden-oynatma determinizmi** (2 bağımsız koşum, zamansız geçmiş birebir).
- **TestKisitEnvanteri** (7): 3–5 kart; tip='kisit'; çarpan ∈ (0,1]; koşul/hedef
  sözleşmesi; **kanıt yolları repo'da VAR**; kimlik benzersizliği + yetenek
  envanterinden ayrıklık; iş emri örnek kurallarının hedefleri.
- **TestKisitCekirdek** (6): çarpanlı etki aritmetiği (çarpansız = v1);
  baskılamanın sırayı değiştirmesi; hedef_etkin eşik-altı uygulanmaz;
  gurultu_bandi bant-içi/bant-dışı; çarpım ≤ 1 + hedef dışı dokunmaz.
- **TestMotorKisitli** (3): motor kısıt uygular + `kisit_etkileri` raporlar;
  **kısıtsız bellekte anahtar seti v1 birebir**; bozuk kısıt (çarpan>1) fail-closed.
- **TestKisitliTohumlama** (3, yerel Qdrant): birleşik tohum idempotent (41);
  kısıtlar aktivasyonda yetenek DEĞİL; bozuk kart tohumlamada durur.
- **TestGatewayDanisma** (5): default KAPALI anahtar YOK; default KAPALI
  telemetri YAZMAZ; açıkken META + telemetri (`kaynak="gateway"`);
  **karar değişmez + uyumsuzluk LOG'u**; hata ask'i bozmaz.

Koşum: `venv/bin/python -m pytest tests/test_yetenek_agi_v2.py -q` →
**52 passed**. Tam süit (VG-2b sonrası): **641 passed, 1 skipped, 1 xfailed,
23 subtest, 0 failed**.

### 6) Değişen dosyalar (commit YOK — controller'a devredilir)

- **YENİ**: `src/rag/yetenek_agi.py` v2 bölümü aslında mevcut dosyaya eklendi
  (aşağıda); `tests/test_yetenek_agi_v2.py` (52 test);
  `scripts/yetenek_v2_demo.py` (demo); `data/eval/yetenek_telemetri.jsonl`
  (telemetri defteri, 26 satır); bu rapor.
- **DEĞİŞTİ (minimal)**: `src/rag/yetenek_agi.py` (+~600 satır v2: sabitler
  :69-91, ad→uzman eşlemesi :94-121, çarpanlı `etki_hesapla`/`aktivasyon_seti`
  :334-365, kısıt geçişli `tohumla` :474-481, kısıtlı
  `aktivasyon_vektor_ile` :528-579, `agirlik_yaz` :582, `evrim_isle` :645,
  `_agirlik_oku` :717, kısıt çekirdeği :732-810, telemetri+sinyal :812-926,
  evrim :929-990, `kisit_envanteri` :994; v1 davranış-default'ları birebir);
  `src/gateway/agent_gateway.py` (+44: import :34, 2 opsiyonel param :68-69,
  saklama :89-91, OP-İN danışma bloğu :396-418 — default kapalıyken anahtar
  seti birebir); `scripts/seed_yetenek_agi.py` (+8: kısıt envanteri de ekilir
  — 36 yetenek + 5 kısıt; yeniden koşum ağırlıkları varsayılana geri yazar).
- **DOKUNULMADI**: `tests/test_yetenek_agi.py` (v1 testleri, 33 passed);
  `scripts/baslat_canli_gateway.py`; `scripts/yetenek_probe.py`; Qdrant'ta
  `yetenek_agi` DIŞINDA hiçbir koleksiyona yazılmadı.
- **NOT (dürüst)**: paylaşılan çalışma ağacında kardeş subagent'ın
  `src/rag/epistemic_agent.py` (+39) ve `src/rag/merak.py` (+64) değişiklikleri
  görünür — bunlar BU işin DEĞİL; tam süit bu değişikliklerle birlikte 641
  passed verdiğinden çakışma yok, controller birleştirmede ikisini birlikte
  görmelidir.

### 7) Vizyon → v2 eşleşmesi (dürüst)

Uyan: telemetri-tabanlı ağırlık evrimi (deterministik + bantlı + geçmişli);
gözlem→danışma (OP-İN, kararı değiştirmeyen zenginleştirme); kısıtlar vektör
uzayında kural kartları olarak yaşar ve aktivasyonu BASKILAR; hepsi ölçümlü
(eps/bant/çarpan/kısıt bantları sayısal gerekçeli).

v2'de YOK olan (v3+ için açık borç): (1) evrim yalnız zirve kartı günceller —
ikinci/üçüncü sıradaki kartlara sinyal yayılımı yok; (2) sinyal kaynağı
gateway'de OP-İN'dir — canlı açılışta danışma kancası bilinçli olarak
kablosuz bırakıldı (default kapalı kanıtı testli); (3) kısıtlar
`hedef_etkin`/`gurultu_bandi` iki koşuluyla çalışır — yeni koşul türleri
(KISIT_KOSULLARI sabitine eklenerek) ayrı iş; (4) ağırlık evrimi karar
DEĞİŞTİRMEZ — aktivasyonun router'ı etkilemesi ayrı ölçüm ister (iş emrinin
kendisinin kuralı).
