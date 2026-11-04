# P2.5 FAZ-2C — LEGAL CEKETİ v3: BÜYÜK VERİ (463→4.813) + REPLAY — İLAN→HÜKÜM→RAPOR

**Tarih:** 2026-11-04 · **Repo:** ~/Git/tr_llm · **Cihaz:** MacBook Air M2 (MPS, 8 GB)
**Deliverable:** `data/pedagogy/legal_canonical_sft_v2.jsonl` (4.813 kayıt) + `data/anka_legal_v3.pt`
(YENİ — `anka_legal.pt`/`anka_legal_replay.pt` DOKUNULMADI) + bu hüküm dosyası.
Commit YOK. Qdrant YOK. Mevcut dosyalara yazma YOK (tüm çıktılar yeni dosya).

---

## 1. İLAN (kapılar koşumdan önce donduruldu — `data/eval/p25c_legal_v3_ilan_2026-11-04.md`,
sha256 `b5cab73b…d3b5cd6f`; koşum 18:24 +03'te başladı, İLAN saat ~18:20'de yazıldı)

Ölçülmüş zemin (2A/2C hükümlerinden birebir):

| Koşum | Tarif | SFT train | ΔCE (val-50) | kart4 | etiket16 | OOD uyum32 ort |
|---|---|---|---|---|---|---|
| TABAN pure | — | — | — | 0/20 | 0/20 | (öz 1,00) |
| tarif | 1e-5, 3e, saf SFT | 413 | −0,048 | 0/20 | 0/20 | 0,7083 |
| p26lr | 5e-5, 3e, saf SFT | 413 | −0,194 | 0/20 | 0/20 | 0,4375 |
| tani12e | 5e-5, 12e, saf SFT | 413 | −0,463 | 1/20 | 0/20 | 0,1250 |
| REPLAY (2C) | 5e-5, 10e, %19,95 pencere + KL β=0,2 | 313 | −0,306 | 0/20 | 0/20 | 0,3646 |

**Donmuş kapılar (İLAN §3):** K1 val-50 CE ≤ 4,782289 − 0,50 = **4,282289** ·
K2 kanonik-20 **etiket16 ≥ 5/20** (tohum-42; nüfus beyanı: örneklem v2 listesinden çekilir —
nüfus 463→4.813 olduğundan idx listesi 2A/2C'den farklı, kural aynı) ·
K3 OOD ilk-32 argmax uyumu **≥ 0,40** (3 sabit sorgu, taban-pure'e karşı).

**Donmuş tarif (İLAN §4):** `train_p26_sft_replay.py` CLI, DEĞİŞTİRİLMEDEN:
init=öğretmen=`anka_taban_pure.pt`, pencere-bin `kulliyat_v2.bin`, `--pencere-n 1166`
(train 4.663/4 → %20,003 karışım), `--kl-beta 0.2`, `--peak-lr 5e-5 --min-lr 1e-5
--warmup 20`, `--epochs 3`, seed 42. Artefakt-seçim kuralı (nöbet semantiği, İLAN §4.4):
`_e1/_e2/final` 2A-rig val-50 CE ile taranır, en düşüğü `anka_legal_v3.pt` artefaktıdır.

## 2. HÜKÜM — **KALDI (1/3 kapı: K1 GEÇTİ — seride İLK KEZ)**

| Kapı | Donmuş kural (İLAN §3) | Ölçüm (2A rig: val-50, mode=True) | Hüküm | Açık |
|---|---|---|---|---|
| **K1** val CE | final ≤ **4,282289** | **4,259029** (Δ = **−0,523260**) | **GEÇTİ** | — (barı 0,023 nat aştı) |
| **K2** kanonik-20 | etiket16 ≥ **5/20** | **0/20** (kart4 0/20; ortak-önek 0,9) | **KALDI** | **5 kayıt** |
| **K3** OOD argmax | uyum32 ort ≥ **0,40** | **0,0833** (0,00 / 0,22 / 0,03) | **KALDI** | **0,317** |

Ölçüm zemini kanıtı (programatik): taban-pure val-50 CE yeniden ölçüldü — çıpa
`p25_legal_taban_cipasi_2026-11-04.json` ile fark **0,000000**; OOD gen_ids **3/3 birebir**;
kanonik taban özeti özdeş (ortak-önek 0,0, 0/20). Nöbet taraması: `_e1` 4,352029 →
`_e2` 4,277363 → final 4,259029 — monoton düşüş, **aşırı-eğitim nöbeti tetiklenmedi,
artefakt = final** (İLAN §4.4; kopya gerekmedi).

## 3. RAPOR

### 3.1 Veri: `legal_canonical_sft_v2.jsonl` (FAZ-1 kapıları aynen, ölçekli)

- **4.813 kayıt** = 4.350 YENİ + 463 eski (eski satırlar BAYT-BİREBİR kuyruk). Üretici:
  `scripts/prepare_legal_dataset_v2.py` (FAZ-1 modülü import — kapılar birebir).
- Yeni tip dağılımı: a_kavram_tanimi **154** (28/kanun dengesi kaldırıldı) ·
  b_madde_sorusu **3.244** (PER_LAW_CAP kaldırıldı) · b3_tum_madde **808** (YENİ: b1 dışı tüm
  madde satırları, 490'ı Geçici/Ek) · c_kavram_ayrimi **145** (otomatik aynı-kanun bent
  çiftleri) · d_madde_dizisi **157** (ardışık-madde alıntısı). Dedupe: 158 (yeni-eski463
  örtüşmesi). Uydurma YOK: her alıntı korpus madde satırından; mülga/mezar/bent-biçim
  filtreleri FAZ-1 birebir; **b1∪b3 = 4.052 madde** (17 kanunun non-mülga ≥60 kr tam
  kapsamı: 4.053 adayın 4.052'si — 1'i etiket-yinelenen madde).
- Token/UNK (mode=True): **877.721 token**, UNK **16.049 (%1,83)**, ort. 182/kayıt
  (FAZ-1: 80.455 token, %1,62 — **10,9×** büyüme; UNK oranı istikrarlı).
- **VAL sözleşmesi kanıtı:** v2 son-50 == FAZ-1 son-50 **BİREBİR** (`True`); eski 463 blok ==
  eski dosya **BİREBİR** (`True`); eski jsonl sha `69670bc6…34734e9c` değişmedi (mtime 17:26).
  CLI val-150 de REPLAY koşumuyla özdeş: init val CE **4,641116** ve pencere-CE **3,9633**
  birebir yeniden üretildi (canlı kanıt).

### 3.2 Koşum ölçümleri (betik DEĞİŞTİRİLMEDİ; log: `p25c_legal_v3_egitim_log_…txt`)

- **547 opt-adım** (planlanan 549; 1.095 mikro-parti / accum-2 — son tek mikro-partinin
  gradyanı adım olmadı; 2C'deki 125/130 deseninin aynısı), 3 epok, warmup 20,
  5e-5 → 1e-5 cosine (son adım LR 1,00e-05), seed 42.
- **Pencere karışımı %20,003:** pencere-n **1.166**, SFT train **4.663** (CLI son-150 val
  dilimi) → 1.166/(4.663+1.166). SFT maruziyeti 4.663×3 = **13.989 satır-geçişi**
  (tani12e: 4.956; REPLAY: 3.130).
- **KL kanıt:** dal **1.063/1.095** mikro-partide ateşlendi (β=0,2, öğretmen=pure, öz-çapa);
  adım-ort KL 0,103 → ~0,20 (maks 0,225) — pencere satırlarında öğretmenden ölçülebilir saptı.
- Süre **3.175,3 sn** (2C: 248,2 sn — kayıt ölçeği 10,4×, koşum 12,8×; makine geneli load 8,8
  çekişmesiyle adım-süresi 30-37 sn'lik geçici spike'lar gördü, koşum ort 5,8 s/adım;
  son-20 adım 2,895±0,64 s). MPS tepe **1,72 GB** (önerilen 5,33 GB — çökme yok).
- Betiğin kendi rig'i (val-150, mode=False): masked SFT val CE **4,641116 → 3,966010**
  (Δ −0,675106); pencere-CE (kulliyat_v2): **3,9633 → 3,8146** (Δ −0,1487) —
  **genel-korpus pencere alanı bozulmadı, iyileşti**. Loss 3,8970 → 3,3573.
- Tokenizer-modu farkı (İLAN beyanı): eğitim mode=False, ölçüm mode=True — 2C'de ölçülen
  etkisi −0,002 nat; K1 açısından ihmal edilebilir.

### 3.3 Kapı hükümleri ve desen karakterizasyonu (argmax üretimler çözümlendi)

- **K1 GEÇTİ — ilk kez:** 4,259029 ≤ 4,282289. ΔCE −0,523 nat — 2C'nin 0,194 nat açığı
  veri ölçeğiyle kapandı. **CE ekseninde "epokla değil veriyle" teşhisi doğrulandı.**
- **K2 KALDI — 5 kayıt açık, ama desen niteliksel değişti:**
  - ortak-önek **0,0 (taban) → 0,9** (2C: 0,45): 18/20 kayıtta TAM 1 jeton örtüşme —
    **[Mevzuat] ilk jetonu edinildi** (b-kartlarının ilk jetonu; gold `m e v z u a t` TEK
    jeton). Kalan 2/20 sıfır (K-tip kartlar).
  - **erken-EOS 0/20** (2C: 9/20) — üretimler 48-jeton tavana kadar akıyor.
  - **kart-tip ayrımı YOK:** 84% çoğunluk formatına (b1∪b3 "Mevzuat Hükmü/Yanıtı" çerçevesi)
    çöküş — "Kanun Tanımı:"/"Kavram Ayrımı:" kartlarına da aynı "Mevzuat…" çerçevesi üretiliyor.
  - **gövde halüsinasyonu:** "Mevzuat <PROPER_NOUN> Türk medeni Kanunu (Kanun No. 477),
    madde 3 (Kanun No. 477), madde 3…" — 477 gibi kanun numaraları eğitim verisinde yok
    (gerçek numaralar 657-6698); tekrar döngüleri var. CE 4,26'ya düşerken üretimin olgusal
    çerçevesi oturmadı.
- **K3 KALDI — 0,317 açık:** uyum32 **0,0833** (marangoz#0 0,00 / marangoz#1 0,22 /
  bahçıvan#0 0,03; REPLAY: 0,3646). OOD CE Δort **+0,2138** (+0,3303/+0,3665/−0,0555 —
  2A bağlam tavanı +0,15 not edilir). Desen: v3 **alan-dışı sorgularda da legal-kart
  üslubuna** geçiyor (marangoz#0'a "mevzuat <PROPER_NOUN> Türk patent kanunu (Kanun No.
  4000)…"); marangoz#1'de tabanın "teknik çözüm" döngüsüne kısmen dönüş (0,22). KL β=0,2
  %20 pencere, 4,5× maruziyet altında öğretmen argmax davranışını **tutamadı**.

### 3.4 Karşılaştırma tablosu (hepsi aynı taban, aynı 2A rig, aynı val-50)

| Koşum | Tarif | SFT train | ΔCE (val-50) | kart4 | etiket16 | OOD uyum32 ort | OOD CE Δort |
|---|---|---|---|---|---|---|---|
| TABAN pure | — | — | — | 0/20 | 0/20 | (öz 1,00) | — |
| tarif | 1e-5, 3e, saf SFT | 413 | −0,048 | 0/20 | 0/20 | 0,7083 | −0,012 |
| p26lr | 5e-5, 3e, saf SFT | 413 | −0,194 | 0/20 | 0/20 | 0,4375 | −0,014 |
| tani12e | 5e-5, 12e, saf SFT | 413 | −0,463 | 1/20 | 0/20 | 0,1250 | +0,077 |
| REPLAY (2C) | 5e-5, 10e, %19,95 pencere + KL β=0,2 | 313 | −0,306 | 0/20 | 0/20 | 0,3646 | +0,052 |
| **v3-BÜYÜK VERİ** | **5e-5, 3e, %20,003 pencere + KL β=0,2, SFT v2 4.663** | **4.663** | **−0,523** | **0/20** | **0/20** | **0,0833** | **+0,214** |

**Okuma (ölçümlü):** (1) **K1 veriyle kapanır** — 10,4× veri + 3 epok, Δ'yi −0,31'den
−0,52'ye taşıdı; bar ilk kez geçildi. (2) **K2 veriyle çözülmez** — 2C'nin "kart şablonu
veriyle edinilir" hipotezi bu koşumda NEGATİF ölçüldü: 13.989 satır-geçişi kart kimliğini
(kart-tip + kanun/madde eşlemesi) kurmadı; edinilen tek şey çoğunluk çerçevesinin ilk jetonu.
(3) **K3 alan-öğrenimiyle ters-korelasyon** 2A'da yumuşatılabilirdi (KL+pencere); 4,5×
maruziyette β=0,2/%20 ankrajı yetersiz kaldı (0,365→0,083). Üç kapı mevcut tarif uzayında
bir arada kapanmıyor; ölçülen cephe: alan-öğrenimi ↑↑ · alan-dışı uyum ↓↓ · kart kimliği
~ (1-jeton önek + halüsinasyonlu gövde).

### 3.5 Çıktılar ve sha256 (programatik — `hashlib`; elle dökülmedi)

| Dosya | sha256 |
|---|---|
| `data/pedagogy/legal_canonical_sft_v2.jsonl` (4.813 kayıt; son-50 FAZ-1 ile özdeş) | `1c822c9f41bd0896f65ed85ff24f59d883346b46f26f8ba6c58922ad2bf6524f` |
| `data/pedagogy/legal_canonical_sft_v2.jsonl.meta.json` | `400d1c9627783f1e8df023115be37b2cd0485dfc1a37327a7d62b2617af4e63b` |
| `scripts/prepare_legal_dataset_v2.py` (üretici, FAZ-1 kapıları import) | `cebf96b22da3badd5a7333d4d059305e24c04c5bf896a0015ae9d6f8a3110029` |
| `data/anka_legal_v3.pt` (YENİ artefakt; en iyi epok = final e3; kopya YOK) | `4de1228219f99d79051ce94efe9c4a1115425c4bf25e1b6938053cb370ec6f7e` |
| `data/anka_legal_v3.pt_e1.pt` / `_e2.pt` (ara kanıt; val-50 4,352029 / 4,277363) | `e4a15f26…58e4276` / `855a43b3…4e92b8d` |
| `data/eval/p25c_legal_v3_ilan_2026-11-04.md` (İLAN — koşumdan önce donduruldu) | `b5cab73b73a740c8e73104d206b606776d19200b7d00f12308f7321d3b5cd6f3` |
| `data/eval/p25c_legal_v3_history.json` (547 adım loss/LR/KL) | `199577af40ff5d17aebdc14878c7f202aaeb639e7aba3869d75d9e1ae413baac` |
| `data/eval/p25c_legal_v3_hukum_2026-11-04.json` (betik hükmü) | `50a84ad9518bb0a612ca0821c6a1a9bfaa3e11aa1bf83303b61d2dc8730cb4ed` |
| `data/eval/p25c_legal_v3_olcum_rig_2026-11-04.json` (2A-rig ölçüm kanıtı) | `817ffd7728bc3ae24b7c61400a39ede923ecca9e9a124990fba882bb36cdb496` |
| `data/eval/p25c_legal_v3_egitim_log_2026-11-04.txt` (koşum logu) | `51ab4651ed2ffb2b1879ec17bdf00fc73a2fa2d3327dec64d8731389226331cb` |
| `data/eval/p25c_legal_v3_hukum_2026-11-04.md` (BU dosya) | — (yazım sonrası üst ajana bildirilir) |

**Dokunulmamışlık kanıtı (SERT KURAL):** `anka_legal.pt` mtime 17:37, `anka_legal_replay.pt`
17:55, `legal_canonical_sft.jsonl` 17:26 — üçü de bu koşum öncesi hallerinde; eski jsonl
sha `69670bc6…34734e9c` 2C hükmündekiyle birebir.

Ölçüm rigi: `~/.hermes/cache/scratch/p25c_v3_olcum_rig.py` — `train_legal_ceket.py`
DEĞİŞTİRİLMEDEN import edildi (değerlendirme kodu birebir; model/SFT yolu parametreli).
Provenans (betik çıkışı): init=öğretmen `5479a2f3…d263360` (2A/2C ile aynı — taban
değişmedi), pencere bin `7f91317a…b57745c`, SFT v2 `1c822c9f…f6524f`.

### 3.6 Sonraki adımlar (öneri — her biri yeni donmuş İLAN gerektirir)

1. **K2 için:** (a) kart-tip dengesizliğini kırmak — b-çerçevesi %84; tip-dengeli örnekleme
   veya kart-başı ağırlıklama; (b) kanun/madde numarası → alıntı eşlemesine ayrı hedefli
   kart kalemi (halüsinasyonun kaynağı koşulsuz alan-istatistiği; eşlemeyi açıkça öğreten
   desen yok). Epok artışının K2'ye katkısı 2C'de tüketildi; beklenti düşük.
2. **K3 için:** β↑ (0,3-0,5) veya pencere-n↑ — ama K1 ile ters-korelasyon bu koşumda daha
   sert ölçüldü (−0,523 ↔ 0,083); K1'i koruyarak K3'ü 0,40'a çekmek mevcut karışım
   ekseniyle görünmüyor. Alternatif eksen: pencere bin'in hukuk-ağırlıklı pencerelere
   çevrilmesi (2C önerisi — alan-içi ankraj, alan-dışı serbestlik).
3. **Ölçüm serisinin ana bulgusu:** üç kapı (alan-CE ↓, kart kimliği ↑, alan-dışı uyum ~)
   tek koşumda bir arada kapanamıyor — kapılar arasında ölçülen takas cephesi 6 koşumla
   haritası çıkarıldı; bir sonraki İLAN bu cepheyi AÇIKÇA kabul edip hangi iki kapının öncelikle
   hedefleneceğini seçmelidir.