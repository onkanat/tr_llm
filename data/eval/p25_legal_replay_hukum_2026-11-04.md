# P2.5 FAZ-2C — LEGAL CEKETİ REPLAY KARIŞIMIYLA (p26 tarifi) — İLAN→HÜKÜM→RAPOR

**Tarih:** 2026-11-04 · **Repo:** ~/Git/tr_llm · **Cihaz:** MacBook Air M2 (MPS, 8 GB)
**Deliverable:** `data/anka_legal_replay.pt` (YENİ kayıt — `anka_legal.pt`'ye DOKUNULMADI) + bu hüküm dosyası.
Commit YOK. Qdrant YOK. İzlenen dosyalara dokunma YOK (yalnız yeni rapor/hüküm + yeni model).

---

## 1. İLAN (kapılar koşumdan önce donduruldu — 2026-11-04 17:53 +03)

### 1.1 Ölçülmüş zemin (P2.5 FAZ-2A hüküm dosyalarından, birebir alıntı)

| Koşum (2A) | Tarif | val-50 CE (final) | ΔCE (final−init) | kart4 | etiket16 | OOD uyum32 (ort) |
|---|---|---|---|---|---|---|
| TABAN pure | — | 4,782289 | — | 0/20 | 0/20 | 1,00 (taban kendine) |
| tarif (1e-5, 3e, saf SFT) | pencere/KL YOK | 4,734058 | −0,048 | 0/20 | 0/20 | 0,7083 |
| p26lr (5e-5, 3e, saf SFT) | pencere/KL YOK | 4,588212 | −0,194 | 0/20 | 0/20 | 0,4375 |
| tani12e (5e-5, 12e, saf SFT) | pencere/KL YOK | 4,318863 | −0,463 | 1/20 | 0/20 | 0,1250 |

Zemin ölçüm rigi (birebir korunur): `scripts/train_legal_ceket.py` — masked SFT val CE
(`FastSampleAlignedDataset`, BLOCK=128) **val = son-50 kayıt**, tokenizer
`literal_entity_mode=True` (FAZ-1 sözleşmesi); kanonik-20 örneklemi tohum 42 (sabit idx listesi);
OOD-3 = `marangoz_canonical_sft.jsonl#0`, `#1`, `bahcivan_canonical_sft.jsonl#0`, üretim maks 48,
uyum = ilk-32 pozisyon jeton örtüşmesi (kısa üretim kalan konumları uyumsuz sayar).

### 1.2 DONMUŞ KAPILAR (koşumdan önce donduruldu — ölçüm 1.1 rigiyle yapılır)

- **K1 (val CE):** REPLAY artefaktının 2A-rig val-50 masked SFT CE'si **≤ 4,782289 − 0,50 = 4,282289**
  (yani Δ = final − init ≤ −0,50 nat).
- **K2 (kanonik-20):** tohum-42 örnekleminden **≥ 5/20** kayıtta **ilk-16 jeton birebir doğru**
  (kart türü + kânun/kavram etiketi — 2A hüküm metriği **etiket16**; 2A'da üç koşumda da 0/20).
  kart4 sayısı ikincil olarak ayrıca raporlanır.
- **K3 (OOD argmax):** 3 sabit alan-dışı sorguda taban-pure ilk-48 argmax üretimiyle **ilk-32 uyum
  ortalaması ≥ 0,40** (2A tarif 0,71 / tani12e 0,12 — KL çapasıyla orta bantta tutma hedefi).
  OOD CE deltası bağlam için ayrıca raporlanır (2A kapısındaki +0,15 CE tavanı yalnız not edilir).

HÜKÜM: üç kapı da geçerse GEÇTİ; herhangi biri kalarsa KALDI + hangi kapı kaç nat/puan açık.

### 1.3 Donmuş koşum tarifi (betik DEĞİŞTİRİLMEDEN, CLI yeter — --help doğrulandı)

```
venv/bin/python scripts/train_p26_sft_replay.py \
  --sft-path data/pedagogy/legal_canonical_sft.jsonl \
  --init-model data/anka_taban_pure.pt \
  --teacher-model data/anka_taban_pure.pt \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --pencere-n 78 \
  --kl-beta 0.2 \
  --peak-lr 5e-5 --min-lr 1e-5 --warmup 20 \
  --epochs 10 \
  --seed 42 \
  --save-path data/anka_legal_replay.pt --allow-frozen-write \
  --history-path data/eval/p25_replay_legal_history.json \
  --hukum-path data/eval/p25_replay_legal_hukum_2026-11-04.json
```

Tarif sapmaları ve gerekçeleri (koşumdan önce beyan):

1. **--pencere-n 1087 → 78.** Gerekçe: betiğin val dilimi `recs[-150:]` → legal 463 kayıtta
   train=313, val=150. 1087 pencere karışımı %77,6 pencere yapardı; %20 pencere hedefi
   313/4≈78 → **78/(313+78) = %19,95**. Görev talimatının örtük aritmetiği (313/4≈78) uygulandı.
2. **--epochs 1 → 10.** Gerekçe: (313+78)/32 ≈ 13 opt-adım/epok → 10 epok ≈ 125-130 adım
   (görev: "8-10 epok ≈ 100-130 adım"). SFT küçük; aşırı-eğitim nöbeti post-hoc izlenir (1.4).
3. **Betik epok-sonu değerlendirme + en-iyi geri yükleme DESTEKLEMİYOR** (kaynak satır 335-338 yalnız
   `_e{N}.pt` ara kaydı yazar; val ölçümü yalnız koşum başı/sonu). Nöbet + en-iyi epok POST-HOC
   karşılanır (bend 1.4).
4. **Tokenizer modu farkı (bilinçli beyan):** betik `KristalTokenizer(comp, vocab)` kurar →
   `literal_entity_mode=False` (tokenizer.py:114 varsayılan; apostrof sessizce silinir —
   hukuk metni apostrof-yoğun). 2A ceket EĞİTİMİ mode=True idi; ölçüm rigi 2A ile AYNI kalır
   (mode=True). Eğitim/ölçüm tokenizasyonu bu eksende aynı DEĞİL — betik değiştirme yasağının
   tarifsel sonucu; K1 yorumunda hatırlatılacak.
5. **Val dilimi:** eğitim betiği son-150'yi val tutar (train=313; 2A ceket son-50 val ile 413
   eğitmişti). K1 karşılaştırması 2A'nın ölçüm setiyle (son-50, mode=True) yapılır — zemin
   4,782289 birebir aynı rigden.

### 1.4 Artefakt-seçim kuralı (nöbet semantiği; koşumdan önce donduruldu)

Koşum 10 epoğu tamamlar; betik `_e1.._e9` ara kayıtlarını + finali (`anka_legal_replay.pt`) yazar.
Her epok-sonu kaydı 2A-rig val-50 CE ile ölçülür; **artefakt `data/anka_legal_replay.pt` = en düşük
val-50 CE'li epok kaydıdır** (2A'nın "en-iyi epok geri yükleme" semantiğinin betik-dışı karşılığı;
2A hükümleriyle aynı seçim kuralı — tani12e satırı da en-iyi-epok seçimiyle ölçülmüştü).
K1/K2/K3 bu artefakt üzerinde ölçülür. Final-epok kaydı sha256'si betiğin hüküm JSON'unda kalır.

### 1.5 Ölçüm rigi kimliği

- `train_legal_ceket.py` DEĞİŞTİRİLMEDİ — değerlendirme kodu scratch kopyada (`~/.hermes/cache/scratch/`)
  import yoluyla kullanılır, model kaydı parametreleştirilir (`anka_legal_replay.pt`, `_e{N}.pt`).
- Taban-pure OOD gen_ids mevcut koşumda yeniden ölçülür ve 2A çıpa JSON'undakiyle
  (`p25_legal_taban_cipasi_2026-11-04.json`) birebir çapraz-doğrulanır.
- Tüm sha256 değerleri programatik üretilir (`compute_sha256`), elle dökülmez.

---

## 2. HÜKÜM — **KALDI** (0/3 kapı)

| Kapı | Donmuş kural (İLAN 1.2) | Ölçüm (2A rig: val-50, mode=True) | Hüküm | Açık |
|---|---|---|---|---|
| **K1** val CE | final ≤ 4,782289 − 0,50 = **4,282289** | **4,476628** (Δ = −0,305660) | **KALDI** | **0,194 nat** |
| **K2** kanonik-20 | etiket16 ≥ **5/20** | **0/20** (kart4 da 0/20) | **KALDI** | **5 kayıt** |
| **K3** OOD argmax | uyum32 ort ≥ **0,40** | **0,3646** (0,2812 / 0,5625 / 0,2500) | **KALDI** | **0,035** |

- K3'ün CE tarafı: OOD CE Δort **+0,0524** — 2A kapısındaki +0,15 tavanının İÇİNDE; kaldıran taraf
  argmax-uyum (0,3646 < 0,40).
- Ölçüm zemini kanıtı: taban-pure çıpası bu koşumda yeniden ölçüldü — val CE farkı **0,000000**,
  OOD gen_ids **3/3 birebir**, kanonik-20 özeti birebir (aynı idx/kart4/etiket16/gen-uzunluk).
  2A zeminine karşı ölçüm özdeşliği programatik doğrulandı.
- Kanonik-20 başarısızlık deseni: ortak-önek ort **0,45** (taban 0,0; tani12e 0,65); **9/20** kayıtta
  3-jeton erken-EOS üretimi. Kart şablonu edinilmedi — 2A'nın "epokla değil VERİYLE çözülür"
  bulgusu replay'de de doğrulandı.

---

## 3. RAPOR

### 3.1 Koşum ölçümleri

- Betik `scripts/train_p26_sft_replay.py` DEĞİŞTİRİLMEDİ (CLI yeter — İLAN 1.3 komutu birebir).
  Provenans (betik çıkışı, programatik): init=öğretmen=`anka_taban_pure.pt`
  sha `5479a2f3…d263360` (2A çıpasıyla aynı — taban değişmedi); pencere bin `kulliyat_v2.bin`
  sha `7f91317a…b57745c`; SFT `legal_canonical_sft.jsonl` sha `69670bc6…34734e9c`
  (2A hükümdekiyle birebir — veri değişmedi).
- **Pencere karışımı %19,95**: pencere-n **78**, SFT train **313** (betiğin `recs[-150:]` val
  dilimi) → 78/(313+78)=19,95 %. 1087 bırakılsaydı %77,6 pencere olurdu — İLAN bend 1.3-1
  gerekçesiyle 78'e ayarlandı, rapora yazıldı.
- **125 opt-adım** (250 mikro-parti / accum 2; betiğin tamsayı tahmini 130 — accum artığı epoklar
  arası taşar), **10 epok**, warmup 20, peak 5e-5 → min 1e-5 cosine (son adım LR 1,03e-05).
- Süre **248,2 sn** (ön-hesap 14,1 sn ayrı); s/adım son-20 **2,128±0,839**; MPS tepe **1,83 GB**
  (önerilen 5,33 GB — p26 mikro-parti/KL-yığın onarımları işledi, çökme yok).
- **KL kanıt**: KL dalı **245/250** mikro-partide ateşlendi (β=0,2, öğretmen=pure, öz-çapa);
  adım-ort KL 0,103 → 0,152 (maks 0,171) — model pencere satırlarında öğretmenden ölçülebilir
  saptı (adım-0'da ≈0 beklentisiyle uyumlu).
- Betiğin kendi rig'i (val-150, mode=False): masked SFT val CE **4,641116 → 4,312275**
  (Δ −0,3288); pencere-CE (kulliyat_v2): **3,9633 → 3,8780** (Δ −0,0853) — **genel-korpus
  pencere alanı bozulmadı, iyileşti** (pencere satırlarında CE de çalışıyor, yalnız KL değil).
- Loss: 3,8371 (adım 1) → 3,5211 (adım 125).
- **Epok eğrisi (2A-rig val-50 CE, artefakt-seçim ölçütü)**:

| epok | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 (final) |
|---|---|---|---|---|---|---|---|---|---|---|
| val-50 CE | 4,7509 | 4,6737 | 4,6180 | 4,5720 | 4,5424 | 4,5204 | 4,5050 | 4,4914 | 4,4834 | **4,4766** |

  Monoton düşüş — **aşırı-eğitim nöbeti tetiklenmedi**; en iyi epok = 10 = final. İLAN 1.4
  artefakt-seçim kuralı betiğin yazdığı `data/anka_legal_replay.pt` ile otomatik sağlandı
  (kopyalama gerekmedi; `_e1.._e9` ara kayıtları eğri kanıtı olarak duruyor).
- Tokenizer-modu kontrolü (İLAN 1.3-4 riski ölçüldü): aynı final modelde val-150 2A-rig
  (mode=True) **4,3102** vs betik (mode=False) **4,3123** → fark −0,002 nat — mod farkı val CE'yi
  **açıklamıyor**; K1 açığının nedeni değil.

### 3.2 Kapı hükümleri

- **K1 KALDI — 0,194 nat açık**: final 4,476628; bar 4,282289; iyileşme −0,3057 nat (hedef −0,50).
- **K2 KALDI — 5 kayıt açık**: etiket16 0/20 (bar ≥5); kart4 0/20 (tani12e'de 1/20 idi).
- **K3 KALDI — 0,035 açık**: uyum32 ort 0,3646 (bar 0,40). Per-sorgu: marangoz#0 0,2812 /
  marangoz#1 0,5625 / bahçıvan#0 0,2500; OOD CE Δort +0,0524.

### 3.3 Karşılaştırma tablosu (2A üç koşumu + REPLAY; hepsi aynı taban, aynı 2A rig, aynı val-50)

| Koşum | Tarif | SFT train | ΔCE (val-50) | kart4 | etiket16 | OOD uyum32 ort | OOD CE Δort |
|---|---|---|---|---|---|---|---|
| TABAN pure | — | — | — | 0/20 | 0/20 | (öz 1,00) | — |
| tarif | 1e-5, 3e, saf SFT, pencere/KL yok | 413 | −0,048 | 0/20 | 0/20 | 0,7083 | −0,012 |
| p26lr | 5e-5, 3e, saf SFT | 413 | −0,194 | 0/20 | 0/20 | 0,4375 | −0,014 |
| tani12e | 5e-5, 12e, saf SFT | 413 | −0,463 | 1/20 | 0/20 | 0,1250 | +0,077 |
| **REPLAY** | **5e-5, 10e, %19,95 pencere + KL β=0,2 (öğretmen pure)** | **313** | **−0,306** | **0/20** | **0/20** | **0,3646** | **+0,052** |

**Okuma (ölçümlü):** REPLAY alan-öğreniminde p26lr ile tani12e arasında (−0,306), OOD'da orta
bantta (0,365) durdu. KL+pencere OOD'u tani12e'ye göre **+0,24 puan yukarıda tuttu**
(0,125→0,365) ama alan-öğrenimini **−0,157 nat kıs**tı — 2A'nın ters-korelasyonu %20 pencere +
KL 0,2 ile **yumuşadı, kalkmadı**; üç kapı bu karışımla da bir arada kapanamadı. Nedenler:
(a) SFT maruziyeti 10×313=3.130 satır-geçişi — tani12e'nin 12×413=4.956'ının %63'ü (betiğin
son-150 val dilimi kaçınılmaz kılıyor); (b) her epokta gradyanın ~%20'i pencere satırlarına
akan CE + β=0,2 öğretmen-çekimi (KL>0 ölçüldü). Epok eğrisi hâlâ monoton düşüyor ama son-3-epok
kazancı ~0,007-0,008 nat/epok — **projeksiyon** (ölçüm değil): +6 epok ≈ 4,44 bandı; K1'in
0,194 nat açığı epok artışıyla kapanmaz.

### 3.4 Çıktılar ve sha256 (programatik üretildi — `compute_sha256`/`hashlib`; elle dökülmedi)

| Dosya | sha256 |
|---|---|
| `data/anka_legal_replay.pt` (YENİ artefakt; en iyi epok = final e10; `anka_legal.pt` DOKUNULMADI) | `457e3bb1c7a2d7876a3bd34285a124aa3c41e6c613be09f696b3b38c2c747270` |
| `data/anka_legal_replay.pt_e1..e9.pt` (9 ara kayıt — eğri kanıtı; shalar ölçüm JSON'unda) | — |
| `data/eval/p25_replay_legal_hukum_2026-11-04.json` (betik hükmü) | `ff059a3565cb903a1ad1e9deb733919c8cf72856180de2e40c20f3d02f4d608d` |
| `data/eval/p25_replay_legal_history.json` (125 adım loss/LR/KL) | `4d30072794a2557d5b1c3340900d22faf2e0b13627bfbd70d63e2676516b05e9` |
| `data/eval/p25_legal_egitim_replay_log_2026-11-04.txt` (koşum logu) | `7dc66e510625b2e7e08fde5e57429fb7b3a3633e79b7e84e0b7c12ee6534a4ae` |
| `data/eval/p25_replay_legal_olcum_rig_2026-11-04.json` (2A-rig ölçüm kanıtı; scratch kopyası) | `95a88430836a81baee4a12befc32798d589f0aee6e2a0b3f81c19bc121f96090` |
| `data/eval/p25_legal_replay_hukum_2026-11-04.md` (BU dosya) | — (yazım sonrası üst ajana ayrıca bildirilir) |

Ölçüm rigi: `~/.hermes/cache/scratch/p25_replay_olcum_rig.py` — `train_legal_ceket.py`
DEĞİŞTİRİLMEDEN import edildi (değerlendirme kodu birebir; yalnız model kaydı parametreleştirildi).
Not: rig v1'in çapraz-doğrulama alanında `ood_gen_ids_birebir=False` yazan bir karşılaştırma hatası
vardı (dict==list, hep False); programatik yeniden-karşılaştırma **3/3 birebir** — scratch JSON
ve repo kopyası düzeltildi, bu notla belgelendi. p26 varsayılan hüküm dosyalarının
(`p26_replay_history.json`, `p26_replay_hukum_2026-11-04.json`) ÜZERİNE YAZILMADI — tarih damgalı
`p25_replay_legal_*` adlarına yazıldı.

### 3.5 Sonraki adımlar (öneri — her biri yeni donmuş İLAN gerektirir)

1. **K1**: epok artışı yavaş kapanır (3.3 projeksiyonu); asıl kaldırac pencere-n azaltımı
   (78→~40) VEYA pencere bin'in hukuk-ağırlıklı pencerelere çevrilmesi — ikisi de tarif sapması.
   Betiğin son-150 val dilimi SFT'yi 313'te sabitliyor; maruziyet tavanını yalnız epokla büyütüyor.
2. **K2**: kart-şablonu veriyle çözülür (2A bulgusu + bu koşumun 0/20'si) — replay/epok/KL
   ekseninin dışında; hedefli kart-şablonu SFT kalemi gerekir.
3. **K3** 0,035 açıkta: pencere-n azaltımı K1'i besler ama OOD'u düşürebilir (2A ters-korelasyonu);
   KL β artışı (0,2→0,3) alternatifi ayrıca ilan-edilmeli.