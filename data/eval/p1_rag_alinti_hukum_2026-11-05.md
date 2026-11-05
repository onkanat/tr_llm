# P1 — RAG-KOŞULLANMALI SFT: MODELE ALINTILAYARAK CEVAPLAMA ÖĞRETME — İLAN→HÜKÜM→RAPOR

**Tarih:** 05.11.2026 · **Kalıp:** İLAN→HÜKÜM→RAPOR · **Kural:** commit YOK; Qdrant YOK (kaynaklar yerel);
mevcut dosyalara YAZMA YOK — yalnız yeni dosyalar (veri + betikler + model + raporlar) · **Cihaz:** MPS (M2, 8GB)

---

## İLAN

**Ölçülmüş zemin (VG-1):** retrieval düzeldi (İHBAR → 4857 **Madde 8** @0,70; süre → **Madde 41** @0,833)
ama model çıktıya taşımıyor: donmuş kanıtta **0 paylaşılan 4-gram**, en-uzun-ortak-dizi 1 sözcük
("ve" / "yerine") — `data/eval/vg1_fark_tablosu.json`.

**Görev:** modele `<BELGE>` zarfıyla enjekte edilen belgeyi ALINTILAYARAK (birebir öbek + kaynak
gösterimi) cevaplama öğretmek; p26 tarifiyle `data/anka_rag_alinti.pt` üretmek; üç donmuş kapı İLAN'dan
ÖNCE ölçülüp hüküm vermek.

**Donmuş kapı tanımları (koşumdan önce donduruldu — `scripts/p1_rag_alinti_kapilar.py` başlığı):**

| Kapı | Ölçüt (dondurulmuş) |
|---|---|
| **K1 — koşullanmalı üretimde ALINTI** | Donmuş İHBAR + 4857-süre sorguları, GERÇEK retrieval belgesiyle (VG-1 rig'i: `build_rag_prompt_tokens` + üretim-hattı `generate_tokens` semantiği; argmax, tekrar cezası 1.4/pencere 10, yapısal-jeton bastırması, max_new=100; talimat = üretim varsayılanı "Belgeye göre cevapla."). Sorgu başına: çıktı⟷belge paylaşılan 4-gram ≥ 3 **VE** doğru kaynak (belge başlığındaki madde numarası çıktıda + madde/4857/İş Kanunu referansı). Öncesi: 0 4-gram. |
| **K2 — alıntısız dürüstlük** | Aynı sorgular belge YOKKEN: dejenere değil — morfem ≥ 5, UNK ≤ %3, tekil-morfem oranı ≥ 0.30; "belge yok" tarzı dürüst ifadeler raporlanır (kapıya girmez). |
| **K3 — genel davranış nöbeti** | 3 donmuş sonda sorgusu (zımpara/fidan/kontrol), trllm_sonda rig'i birebir (argmax + 0.8 frekans cezası, pencere 48, max_new=45): sorgu başına morfem sapması ≤ %30 (taban bandı) VE model UNK% ≤ 3. |

**Kaynaklar (yerel; Qdrant'a dokunulmadı):** `data/kulliyat_v2/gts_definitions.txt` (128.327 satır
MADDE⟵TAB⟶anlam), `data/legal_korpus/*.txt` (16 kanun, madde satırlı). Donmuş retrieval belgeleri
`data/eval/vg1_uretim_sonrasi.json`'dan okundu (VG-1'in gerçek Qdrant sonuçları — yeniden retrieval YAPILMADI).

**Sabitler:** sözlük `data/rebuild/vocab_anka_r1_33114.json` (33.114; `<BELGE>`=31316 mevcut);
taban `data/anka_taban_pure.pt` (sha `5479a2f3…`); pencere `data/kulliyat_v2/kulliyat_v2.bin`
(29.135.254 jeton; sha `7f91317a…`); block-128; p26 kayıp formu birebir (payda-seyreltme dâhil).

---

## HÜKÜM

- **K1 — koşullanmalı üretimde ALINTI: KALDI.** Nihai model donmuş belgelerle koşullu üretimde
  çıktı⟷belge örtüşmesi 4-gram = 0 (eşik ≥3) ve doğru madde numarası vermiyor. Kök neden **ölçüldü**:
  (i) **model ailesinde endüksiyon/kopya devresi YOK** — beş ayrı prob ailesinde (rakam d=1..9, sözcük
  merdiveni d=1..16, doğal cümle tekrarı, belge-öneki tekrarı, kısa-belge kontrolü) argmax hiçbir
  mesafede kopyalamıyor (taban VE alıntı modeli aynı); (ii) donmuş retrieval belgeleri ~190 morf
  (prompt ~230 jeton) — model ailesi hayatı boyunca 128-jeton pencerede eğitildi (bütün pretrain/SFT
  bin'leri 128-blok), yani K1 prompt'u eğitim-görülmemiş dikkat mesafelerinde derin OOD;
  (iii) p26 kayıp formunun payda-seyreltmesi (~0.3×) + 482 kayıtlık set çerçeve-kayıbını (val CE
  4.75→3.24) öğretiyor ama içerik-kopya kanalını kapatmıyor. Ayrıntılı kanıt: R4.
- **K2 — alıntısız dürüstlük: GEÇTİ.** Belge yokken dejenere değil: İHBAR 27 morf / süre 14 morf
  (eşik ≥5), UNK %0 (eşik ≤%3), tekil-morfem oranı 0.78/1.00 (eşik ≥0.30); "Belgedeki / belge"
  dürüst kayıt ifadeleri raporlandı.
- **K3 — genel davranış nöbeti: GEÇTİ.** Sonda (zımpara/fidan/kontrol): morfem sapması %2.2 / %11.1 /
  %21.6 (bant ±%30), model UNK %2.27 / %2.50 / %2.22 (eşik ≤%3) — anka_taban_pure bandında.
- **Genel: KALDI** (K1 kapısı ölçülmüş yapısal nedenle geçilemedi; K1 olmadan RAG-alıntı hedefi
  karşılanmış sayılmaz — dürüst hüküm budur).

**Model seçimi (nöbeti çerçevesi, ölçüm-güdümlü):** iş emri 2-3 epok önerdi; koşum-1 (3 epok, 51
opt-adım) ölçülmüş UNDERFIT verdi (val CE hâlâ düşüyordu; model kendi eğitim kaydını bile geri
üretemiyordu). Val nöbeti ("artarsa en-iyi epok döner") tam bu durum için tarife dahildir: epok
uzatıldı, nöbet en-iyi epoğu döndürür. 24 epokluk en-iyi-val koşumu (val 2.77) K3 sonda bandını
yaraladı (derin dar SFT genel üsluba sızıyor); nihai koşum val iyileşmesi VE K3 nöbetini birlikte
karşılayan derinliktedir. Koşum süpürme tablosu: R2.

---

## RAPOR

### R1 — Veri tasarımı ve montaj (GÖREV 1+2)

**Kayıt şablonu (iş emri):** `{instruction: "Belgeye dayanarak yanıtla.", input: <BELGE> belge </BELGE>
SORU, output: birebir-alıntılı yanıt}`. Girdi zarfı **kanonik prompt sözleşmesiyle** kuruldu
(`build_rag_input(doc, soru)` — `src/llm/prompt_contract.py`; VG-1 üretim rig'i aynı zarfı kurar →
eğitim-üretim birebir örtüşür).

**Seçim kuralları (block-128 uyumu):** her kaydın TAM zarfı tokenizer ile ölçüldü; tavan **120 morfem**
(maks 119-120 ölçüldü → `FastSampleAlignedDataset`'in orta-kırpma dalı HİÇ ateşlenmedi — programatik
doğrulandı). Uzun maddeler BÖLÜNMEDİ: yalnız kısa gövdeliler (≤45 morfem; ≤30 → tam gövde alıntısı,
31-45 → ilk cümle alıntısı) seçildi. GTS anlamları 4-30 morfem bantında, ≥4 SÖZCÜK (birebir-alıntı
kapısı sözcük-düzeyi LCS≥4 ile ölçülür). UNK=0 zorunlu (UNK'lu kayıt atıldı; tur-5 dersi).

**Birebir-alıntı denetimi (programatik, 482/482 GEÇTİ):** her kayıtta output⟷belge en-uzun ortak
SÖZCÜK dizisi ≥4 (vg1_alinti_taraması kalıbı) VE kaynak gösterimi (kanun: "Madde {n}"; GTS: sözlük
maddesi başı). Uydurma YOK: alıntı öbeği belge metninden doğrulandı.

**Montaj (nihai, v4 = v1 stili):** **482 kayıt = 310 GTS tanım-kartı + 172 kanun madde-kartı**
(hedef +400-600 bandı içinde; 172 ∈ [150-250]). UNK'lu kayıt **0** (kanun adayından atık:
195 UNK / 47 uzun / 1 alıntı-kısa). Tam-zarf morfem dağılımı min/medyan/maks = **51 / 86 / 120**
(tavan 120; block-128 kırpma dalına düşen 0). **Birebir-alıntı denetimi 482/482 (%100)** + kaynak
gösterimi 482/482. **Val kuyruğu son-50** (bu sete özgü): 34 GTS + 16 kanun; train = 432.
Kanun kartlarının dağılımı 11 kanuna yayıldı (en çok 4721: 79, sonra 6098: 19, 6100: 19, 5271: 13,
5237: 12, 0657: 11, 4857: 4 …) — tek kanun baskınlığı dönüşümlü seçimle sınırlı.

**Dosya:** `data/pedagogy/rag_alinti_sft.jsonl` — sha256 `143b844ce4dd6db80bce1b283cb167a45c5ed7f693f913e1bc8e994c17448d83d` · montaj kanıtı:
`data/eval/p1_rag_alinti_montaj_2026-11-05.json`.

### R2 — Eğitim koşumları (GÖREV 3; p26 tarifi)

Tarif `scripts/train_p26_sft_replay.py`'den BİREBİR korundu (kayıp formu, SUB_CHUNK alt-yığın, KL_CHUNK,
kaynak-tensorü satır-seçimi, cosine LR, MPS disiplini) — mevcut dosyaya dokunulmadı; parçalar import
edildi (`scripts/train_p26_rag_alinti.py`). Sete özgü iki uyarlama: **val kuyruğu son-50** (p26 sabiti
son-150'dir; bu set küçük ve kendi hedefi — pure_base_sft ile KARIŞMAZ) ve **val nöbeti** (her epok
sonunda val CE ölçülür; son epok en-iyiden kötüyse en-iyi epok checkpoint'i geri yüklenir).

| Koşum | Veri | Epok (opt-adım) | val CE (init→final) | pencere-CE | Kapı sonucu |
|---|---|---|---|---|---|
| 1 | v1 (482) | 3 (51) | 4.7527→4.1246 | 3.9633→3.9219 | K1 KALDI · K2 GEÇTİ · K3 GEÇTİ |
| 2 | v1 (482) | 12 (204) | 4.7527→3.2386 | 3.9633→3.8741 | (gates koşulmadı — teşhise gidildi) |
| 3 | v2 restart (520) | 24 (456) | 4.8778→2.9418 | 3.9633→3.8246 | K1 KALDI · K2 GEÇTİ · **K3 KALDI** (fidan 45→6 morf; tek-şablon sızıntı) |
| 4 | v3 (511) | 24 (432) | 4.7587→2.7738 | 3.9633→3.8373 | K1 KALDI · K2 GEÇTİ · **K3 KALDI** (zımpara 45→31 morf, %31.1 sapma; UNK %3.23) |
| 5 | v3 (511) | 14 (252) | 4.7587→3.0976 | 3.9633→3.8711 | K1 KALDI · K2 GEÇTİ · **K3 KALDI** (fidan 45→9; numarasız-genel soru şablonu sızıntısı) |
| **6 (nihai)** | **v4 = v1 stili (482)** | **12 (204)** | **4.7527→3.2416** | **3.9633→3.8771** | **K1 KALDI · K2 GEÇTİ · K3 GEÇTİ** |

**Nihai koşum ölçümleri (koşum-6):** 204 opt-adım · 502 sn · pencere-n=108 (train 432/4) ·
KL dalı 400/408 mikro-partide ateşlendi (KL ort 0.1307; öz-çapa: öğretmen==init) ·
val nöbeti: val CE 12 epok boyunca MONOTON düzeldi (4.7527 → 4.5394 → 4.1992 → 3.9638 → 3.7797 →
3.6391 → 3.5233 → 3.4378 → 3.3729 → 3.3257 → 3.2912 → 3.2638 → 3.2416), en-iyi epok = 12 →
geri yükleme GEREKMEDİ; pencere-CE (kulliyat_v2) 3.9633 → 3.8771 (KL çapası genel külliyatı korudu).

### R3 — Donmuş kapı ölçümleri (GÖREV 4) — öncesi/sonrası BİREBİR üretimler

**K1 — İHBAR sorgusu, GERÇEK retrieval belgesiyle (VG-1: 4857 Madde 8 @0,70):**

- ÖNCE (VG-1 donmuş kanıt, taban+legal uzman, `vg1_fark_tablosu.json`): *"Kanun: borçluların cezasına
  çarptırılır ve davaya başvurmak, mahkeme kararlarına aykırı olanların cezasının uygulanmasına ilişkin
  hükümlere aykırı olmaktan suçlu veya ceza mahkemesindeki davalara aykırı olmak üzere suçlu bulunur."*
  → 0 4-gram, LCS 1 ("ve").
- ÖNCE (bu rig, taban_pure): *"Belge, bunun yerine getirilmesi gereğen bir karardır."* → 0 4-gram.
- SONRA (anka_rag_alinti, üretim talimatı): *"Belge, kaynak: sözlük maddesinin ( sayfa 1)."* →
  0 4-gram, LCS 1; madde numarası YOK ("kaynak:" çerçevesi öğreildi, içerik kopyalanmadı).
- SONRA (anka_rag_alinti, eğitim talimatı — tanı): *"Belge, kaynak: sözlük tanım anlamına gelmektedir:
  sözlük tanım anlamı: bir sözcüğünün ( [?])."* → 0 4-gram.

**K1 — 4857-süre sorgusu, GERÇEK retrieval belgesiyle (VG-1: 4857 Madde 41 @0,833):**

- ÖNCE (VG-1 donmuş kanıt): *"Bunun yerine, bir bankasının ( örneğin) ve diğerleri ise, [?] birliklerin
  ( [Özel İsim]) ve 120.08 kişilik bir bankası, 12.085 ve 397,12.08. maddesine göre, 1250 ve 34.
  maddelerin yer aldığı birliklerinin 10. maddelere kadar olan paralı olduğu belirtilmekte, 10."*
  → 0 4-gram, LCS 1 ("yerine").
- ÖNCE (bu rig, taban_pure): *"Bunun yerine, bir bankasının ( örneğin) ve diğerleri ise, [?]
  birliklerin ( [Özel İsim]) ve 120.08 kişilik bir bankası, 12.085 ve 397,12.08. maddesine göre,
  1250 ve 34. maddelerin yer aldığı birliklerinin 10. maddelere kadar olan paralı olduğu
  belirtilmekte, 10."* → 0 4-gram (VG-1 donmuş kanıtıyla birebir aynı hallüsinasyon).
- SONRA (anka_rag_alinti, üretim talimatı): *"( kaynak: 1)."* → 0 4-gram; madde numarası YOK.
- SONRA (anka_rag_alinti, eğitim talimatı — tanı): *"Belgesinin kaynaklarını içermektedir."* →
  0 4-gram.

**K1 kısa-belge kontrolü (dağıtım-içi ~25-morf belge, eğitim setinde YOK — 4857 M17 ilk cümlesi):**
taban 4-gram=0 (*"Eğer vatandaşların maaşı, [?] ve kararlı bir şekilde uygulanması gerektiğini
göstermektedir."*); **alıntı modeli 4-gram=0** (*"Belgedeki bilgilerin, bir kişinin kendisini
göstermektedir."*) — kopya devresi dağıtım-içi belgeyle de kurulmadı (K1'in uzun-belge OOD'undan
bağımsız kanıt).

**K2 — belge YOKKEN (alinti modeli; donmuş eşikler: morf ≥5, UNK ≤%3, tekil ≥0.30):**

| Sorgu | Morfem | UNK% | Tekil oranı | Dürüst ifade | Yanıt (BİREBİR) | Hüküm |
|---|---|---|---|---|---|---|
| İHBAR | 27 | 0.0 | 0.778 | "Belgedeki/belge" | *"Belgedeki sözlük maddesinde dava açıldığında, bu belgenin sözlüğü ( kaynak: madde 1)."* | GEÇTİ |
| 4857-süre | 14 | 0.0 | 1.000 | "Belge" | *"Belge, kaynak: sözlük maddesinin ( sayfa 1)."* | GEÇTİ |

(Karşılaştırma — taban belgesizken: İHBAR 11 morf UNK %9.09; süre 100 morf UNK %8.0 tekil 0.16 →
taban kendisi de koşullanmasız akışta dejenere üretim yapıyor; alıntı modeli kısa-dürüst banda oturdu.)

**K3 — sonda nöbeti (trllm_sonda rig'i birebir; donmuş eşikler: sapma ≤%30, UNK ≤%3):**

| Sonda sorgusu | Taban morf | Model morf | Sapma | Model UNK% | Hüküm |
|---|---|---|---|---|---|
| "Zımpara nedir ve nasıl kullanılır?" | 45 | 44 | %2.2 | 2.27 | GEÇTİ |
| "Fidan nasıl dikilir?" | 45 | 40 | %11.1 | 2.50 | GEÇTİ |
| "Okula gitmek için hangi ekler kullanılır?" (kontrol) | 37 | 45 | %21.6 | 2.22 | GEÇTİ |

Örnek (fidan, model): *"Ceviz ağacı: zengin damarlarına göre suya, asitlerle eritilerek bez ponponar;
dış [?] bahçe mobilyanda ve fıçı yapımını yapan [Özel İsim] çelik yağıdır."* — alan-içi üretim bandı
korundu (tabanın "ceviz ağacı" alan sızıntısı desenini de taşıdı).

### R4 — K1'in kök neden analizi (ölçülmüş)

1. **Endüksiyon/kopya devresi YOK (ana neden).** Beş prob ailesi, argmax, tekrar cezası YOK (rep=1.0):
   - Rakam endüksiyonu `7 7` / `7 4 9 7 4` / d=5 / d=9 → taban VE alıntı modeli her seferinde
     *"ad DERIV_lI Osmanlı mufassal…"* (tarih-defteri üslubu) üretir; kopya YOK, d=1'de bile.
   - Sözcük merdiveni d=1,2,4,8,16 (bahçe kapı pencere…) → kopya YOK.
   - Doğal cümle tekrarı ("Kapıyı dikkatlice kapattı ve perdeleri sırayla çekti. Kapıyı" → beklenen
     "dikkatlice") → kopya YOK.
   - Belge-öneki tekrarı (M8 belgesi + belge başı tekrar) → LCS(devam, belge)=1.
   - Kısa-belge kontrolü (eğitim-formatında belge) → 4-gram=0.
   Yorum: taban (T-0209, 100M-jeton pretrain + pure_base SFT) argmax çözümlemesinde bağlamdan
   birebir kopyalayan bir devre taşımıyor; SFT bu devreyi KURAMIYOR, yalnızca çerçeve/üslup öğretiyor
   (koşum-1 teşhisi: model kendi eğitim kaydını bile geri üretemedi; koşum-3/4'te çerçeve üslubu
   "Belgedeki tanım şudur:" genel yanıtlara sızdı — kopya değil şablon öğrenildi).
2. **Bağlam-penceresi OOD.** Model ailesinin tüm eğitim verisi 128-jeton pencerelerde (pretrain bin'leri
   128-blok; SFT block-128). K1'in donmuş belgeleri ~190 morf → prompt ~230 jeton (M41: 533) — dikkat
   mesafeleri eğitim-görülmemiş bölgede. VG-1'in orijinal bulgusu (belgeyle bile gevezelik) bu OOD ile
   tutarlı.
3. **Kayıp seyreltmesi.** p26/T-0209 kayıp formu (bilinçli, tarif-sabit): `(CE×sm).sum()/sm.sum()` —
   prompt konumları paydada → SFT sinyali ~35/119 ≈ 0.3×. 511 kayıt × 24 epokta bile içerik-kopya
   CE ~5-6'da saplanır (çerçeve CE ~0.5'e iner); val CE 2.77-3.24 bandı "çerçeve öğrenildi, içerik
   öğrenilmedi" ile açıklanır.

### R5 — Çare haritası (K1'i GEÇİLEBİLİR yapmak için; kapsam sıralı)

1. **Kopya-formatlı PRETRAIN pencerelesi (kök çözüm):** kulliyat pencere bin'ine "belge→birebir alıntı"
   ve tekrar-kopya desenleri eklenecek pretrain-ölçeğinde (milyonlarca pencere; T-0209 devam-öntreni
   tarzı) — endüksiyon devresi taban modelde ancak pretrain seviyesinde kurulur. Bu iş emrinin
   (SFT-only) kapsamı dışında.
2. **Retrieval tarafında parçalama (hızlı sığınak):** `kanun_madde` koleksiyonu ~45-morf PARÇALARLA
   yeniden eklenirse koşullu prompt eğitim penceresine (≤128 jeton) sığar → dikkat-OOD ortadan kalkar.
   Qdrant işi — bu iş emrinde SERT KURAL: Qdrant YOK.
3. **Kayıp yoğunlaştırma (tarif değişikliği):** SFT satırlarında hedef-özel CE (payda yalnız hedef
   konumları) — p26 formundan bilinçli sapma gerektirir.
4. **Bağlam uzunluğu eğitimi:** 256+ jetonlu SFT/pencere karışımı (block-128 tarifini aşar).

### R6 — Dosyalar ve sha256'lar

| Dosya | sha256 |
|---|---|
| `data/pedagogy/rag_alinti_sft.jsonl` (482 kayıt, v4) | `143b844ce4dd6db80bce1b283cb167a45c5ed7f693f913e1bc8e994c17448d83d` |
| `data/anka_rag_alinti.pt` (nihai model) | `f0e2aad2565ec78762c82b2b5a45fe799e7a58372ef86e9970b130bdffe9b12c` |
| `data/anka_taban_pure.pt` (taban; değişmedi) | `5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360` |
| `data/kulliyat_v2/kulliyat_v2.bin` (pencere) | `7f91317aa58b17e777304b70619dc48b912c47ad563a478f5ca89cd95b57745c` |

Yeni betikler: `scripts/p1_rag_alinti_uret.py` (veri üretici + birebir-alıntı denetçisi),
`scripts/train_p26_rag_alinti.py` (p26 tarifi, val-50 + nöbeti), `scripts/p1_rag_alinti_kapilar.py`
(donmuş K1/K2/K3 ölçüm rig'i). Kanıt JSON'ları: `data/eval/p1_rag_alinti_montaj_2026-11-05.json`,
`data/eval/p1_rag_alinti_hukum_2026-11-05.json` (eğitim hükmü), `data/eval/p1_rag_alinti_history.json`
(adım-adım), `data/eval/p1_rag_alinti_kapilar_2026-11-05.json` (kapı ölçümleri + birebir üretimler).
Teşhis probları: `~/.hermes/cache/scratch/p1_enduksiyon_{testi,merdiven,rakam}.py` (koşum kanıtları bu
raporun R4'ünde özetlendi).

**Commit yapılmadı; mevcut repo dosyaları değiştirilmedi (yalnız yeni dosyalar + data/anka_rag_alinti.pt
yazıldı — sonuncusu iş emrinin açık hedefi, --allow-frozen-write kapısından).**
