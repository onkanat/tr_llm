# T-0096 · FAZ 0 İLANI — Anka marangoz (ceket) tam giydirme koşumu

> **Bu dosya koşumdan ÖNCE yazılmıştır ve sonradan DEĞİŞTİRİLMEZ.**
> Ölçüm kuralları, seçim kuralı ve çürütme maddeleri burada ilan edilir; sonuçlar görüldükten
> sonra eşik/kural ayarlaması yapılmaz (T-0067: *tasarım sayısı ölçüm sonrası kusura göre
> düzeltilmez, rapora yazılır*).

| kalem | değer |
|---|---|
| görev | **T-0096** (yürütücü: claude) |
| operatör talimatı | *"Marangoz eğitilmeli olası en yüksek hassasiyete 3-5 saat eğitim kabul edilebilir."* |
| operatör seçimi | replay oranı **%25** (`--replay-every 4`), AskUserQuestion ile |
| tarih | 20 Eyl 2026 |
| taban | `data/anka_a1r.pt` (T-0077'de kapandı) |
| durum | **Faz 0 — koşum başlamadı** |

---

## §1 — ÖLÇÜLEN DÜZELTME: T-0094/K8 **ÇÜRÜDÜ** (hata ölçüm kabındaydı)

T-0094, ilanın (§6) epoch sayılarını *"~8× yanlış"* ilan etmişti ve bu bulgu T-0095 raporuna
da taşınmıştı (`anka_r17_arka_kapi_2026-09-20.md:191,199,203` — *"0,16 epoch"*).

**Bu görevde doğrudan ölçüldü** — `scripts.train_step_demo.KristalDataset.get_batch`
gerçekten çağrılarak:

```
$ venv/bin/python -c "... KristalDataset(p, block_size=128); x,y = ds.get_batch(batch_size=8) ..."
data/train_carpenter_specialization_anka.bin
   len(dataset.data) = 784,797 jeton
   x.shape=(8, 128)  y.shape=(8, 128)  => adim basina 1,024 jeton
   1 tam gecis = 766.40 adim
scratch/t0094_sonda/mix_r4.bin
   len(dataset.data) = 1,046,429 jeton
   x.shape=(8, 128)  y.shape=(8, 128)  => adim basina 1,024 jeton
   1 tam gecis = 1021.90 adim
```

`train.py:362` `dataset.get_batch(batch_size=batch_size)` çağırır ve kayıp **8 pencerenin
tamamı** üzerinden hesaplanır ⇒ **adım başına 1.024 jeton**, 1 blok (128) değil.

| | iddia (T-0094/K8) | **ölçülen** |
|---|---|---|
| adım başına jeton | 128 | **1.024** |
| sonda kolu (1.000 adım) | 0,16 ceket epoch | **R0 1,30 · R4 0,98 · R10 1,17 · R20 1,24** |
| ceket metası `steps_formulu` | "8× yanlış" | **DOĞRU** — `ceil(784797/(128*8))*3 = 2301` (3 epoch) |

**Sonuç:** T-0094'ün **ilanı (§6: "R0: 1,30 · R4: 0,98 epoch") DOĞRUYDU**; yanlış olan
*"ölçülen 0,16"* sayısıydı. İki sayı çeliştiğinde kusur **ilanda değil, ölçüm kabının kendi
varsayımındaydı** — [[iki-sayi-celisiyor-sanma-once-kume]] sınıfı.

**Bunun koşum tasarımına etkisi (ölçüm sonrası değil, ÖNCESİNDE yazılıyor):** sonda kolları
*"açlıktan"* değil, **~1 tam ceket epoch görmüş olmasına rağmen** şablon üretiyordu (T-0095:
altı sorunun çoğuna aynı cümle; `kesisim %0`). Yani **"daha çok epoch" tek başına çözüm
değildir**; bütçe körlemesine uzun koşuya değil, **süre üzerinde ölçümlü aramaya** harcanır.

---

## §2 — SABİT yapılandırma (sonda kollarıyla BİREBİR aynı)

A/B eşiklerinin kıyaslanabilir kalması için sonda ile aynı tutulur
(`scratch/t0094_sonda_kos.sh` ile satır satır karşılaştırıldı):

```
env PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
venv/bin/python train.py \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --data  scratch/t0096_kos/mix_r4.bin \
  --device mps --block-size 128 --batch-size 8 --lr 2e-4 --steps <SEGMENT> \
  --load-path <onceki> --save-path scratch/t0096_kos/seg_k.pt \
  --save-optimizer --load-optimizer \
  --loss-report --seed <42+k> \
  --allow-frozen-write        # YALNIZ son teslim adımında, data/anka_a1r_ceket.pt için
```

* **SFT** — `--pretrain` **YOK**. Ceket külliyatında `<OUTPUT>` = 12.763 ⇒ maskeleme gerçek
  hedefleri bırakır (T-0073'ün *"her pencere maskelenir"* tuzağı bu külliyatta **yok**).
* **Replay:** `data/train_chat_balanced.bin` (`443f93d3…`, 3.120.000 jeton), `--replay-every 4`.
* **LR programı YOK** (sabit 2e-4) — *beyan edilen sınır*: karşılaştırılabilirlik için sonda ile
  aynı. Gerekçe ve bedeli raporda; bu ilan onu **kapatmaz**, **açık bırakır**.
* **`--save-every` KULLANILMAZ:** ölçüldü (`train.py:406`), aynı `--save-path` üzerine
  **atomik olarak ÜZERİNE** yazar ⇒ ara checkpoint serisi üretmez. Onun yerine **segment**
  mimarisi: her segment ayrı süreç, ayrı checkpoint.

---

## §3 — Segment planı (bütçenin satın aldığı: SÜRE üzerinde arama)

Ölçülen adım süresi (T-0094 kolları, `arms.jsonl`): R0 0,439 · R4 0,502 · R10 0,519 ·
R20 0,530 sn/adım ⇒ karışım için **~0,50 sn/adım** alınır (tahmin değil, ölçüm).

| segment | adım | kümülatif | ceket epoch | tahmini süre |
|---|---|---|---|---|
| S1 | 1.000 | 1.000 | 0,98 | 8,3 dk |
| S2 | 2.000 | 3.000 | 2,94 | 16,7 dk |
| S3 | 3.000 | 6.000 | 5,87 | 25,0 dk |
| S4 | 4.000 | 10.000 | 9,79 | 33,3 dk |
| S5 | 6.000 | 16.000 | 15,66 | 50,0 dk |
| S6 | 8.000 | 24.000 | 23,49 | 66,7 dk |

**Toplam 24.000 adım** = 3,33 saat (0,50 sn/adım) … 3,67 saat (0,55 sn/adım) saf eğitim
+ 6 eşli değerlendirme × ~2 dk ≈ **3,5–3,9 saat** ⇒ operatörün 3–5 saat bandı içinde.
(Adım süresi 0,70 sn'ye çıkarsa 4,7 saat + değerlendirme = bant **üstüne** çıkar; bu durumda
koşum **S5'te kesilir** ve kesinti rapora yazılır — sessizce uzatılmaz, T-0052/[[mps-yuku-adim-suresini-bozar]].)

Erken segmentler kısa tutuldu çünkü ilgi bölgesi **düşük epoch** tarafındadır (§1).

---

## §4 — İLAN EDİLEN SEÇİM KURALI (ölçümden ÖNCE, sonra değiştirilmez)

Her segmentten sonra `scripts/evaluate_carpenter_anka.py --ceket-ekseni --n 100 --seed 42
--baseline data/anka_a1r.pt` koşulur (eşli: unutma ekseni tabana karşı).

1. **Sert kapı:** aday `unutma_gec == true` olmalı (T-0059: A ≤ **+%10** VE B ≥ **−5,0 puan**).
2. **Dışlama:** `ezber_orani >= %10` **VEYA** `tutarsizlik_orani >= %5` olan segment seçimden
   **çıkarılır** ve **gerekçesiyle ilan edilir** — sessizce düşürülmez (plan K2 eşikleri).
3. **Seçim:** sert kapıyı geçenler arasında **en yüksek `rouge_l_ort`**; eşitlikte sırasıyla
   daha düşük `ezber_orani` → daha düşük `tutarsizlik_orani` → **daha az kümülatif adım**.
4. **Erken durma (TEK kural):** bir segmentte `ezber_orani >= %10` **VE** `rouge_l_ort` önceki
   segmente göre **artmıyorsa** koşum DURUR (külliyat ezberlendi, içerik öğrenilmedi).
5. Hiçbir segment sert kapıyı geçmezse hüküm **"kapıyı geçen yok"**tur; en iyi ceket ekseni
   beyan edilir ve **"PASS" YAZILMAZ** ([[kapi-orneklem-zayifsa-gecmek-kanit-degil]]).

**Teslim:** seçilen segment checkpoint'i `data/anka_a1r_ceket.pt` olarak yazılır.

---

## §5 — İLAN EDİLEN ÇÜRÜTME MADDELERİ (kendi iddiamı yanışlayabilmeliyim)

| # | öncül | çürütücü gözlem | hüküm |
|---|---|---|---|
| **Ç1** | "%25 replay unutmayı taşınabilir tutar" | **her** segmentte `unutma_gec == false` | %25 yetersiz; marangoz iddia edilmez |
| **Ç2** | "kısıt takvimdir, daha çok epoch çözer" | `ezber >= %10` **S3'ten önce** | kısıt **külliyattır** (784.797 jeton); takvim değil |
| **Ç3** | "ceket bu boyutta yakınsar" | `rouge_l_ort` hiçbir segmentte **0,15**'i geçmez | yakınsamıyor; neden ayrı araştırılır |
| **Ç4** | "ceket soru-içeriğine bağlanır" | `kesisim_orani` %80'e çıkmaz | üretim soruya bağlanmıyor (T-0095 bulgusu sürüyor) |
| **Ç5** | "erken durma kuralı ateşler" | kural hiç ateşlemezse | kural **ayırt edici değil**; rapora yazılır, "sorun yok" sayılmaz |

---

## §6 — Yönetişim ve sınır (açıkça YAPILMAYANLAR)

* **Donmuş hedefler YALNIZ:** `data/anka_a1r_ceket.pt` ve `data/anka_a1r_ceket.pt.opt.pt`
  (`data/*.pt` deseni) — `--allow-frozen-write` açıkça verilir, **üst dizin** kiralanır
  (donmuş dosya kiralanamaz, [[donmus-dosyaya-kiralama-verilemez]]).
* Replay karışımı **`scratch/t0096_kos/`** içine kurulur ⇒ donmuş yüzeye **yazılmaz**.
  (Karışım deterministiktir: `build_replay_mix.py --seed 42`; meta'sı provenance taşır.)
* **DOKUNULMAZ:** `data/train_carpenter_specialization_anka.bin`, `data/anka_a1r.pt`,
  `data/train_chat_balanced.bin`, `data/pedagogy/**`, `scratch/t0094_sonda/**`,
  `src/**`, `scripts/**`. Hepsi için **baş/son sha256** raporda.
* **Koşum sandbox DIŞINDA** (MPS sandbox'ta görünmez: `built True / avail False`).
  Koşum sırasında makinede başka GPU tüketen iş çalıştırılmaz (T-0052: 0,43 → 3,45 sn/adım).
* Arka kapı (`POST /api/query`) **çağrılmaz** — o yol `data/future_train_vector.jsonl`'e yazar
  (T-0095/K15) ve bu görevin kapsamı değildir.
* `git add -A` / `git add .` **YASAK**; commit/push yok.
* `src/llm/frozen_guard.py:59` mutlak-yol FAIL-OPEN açığı **kapatılmaz** — yalnız beyan edilir,
  bu koşum ona **güvenmez** (hedefler göreli yolla ve açık bayrakla verilir).
* D1 (en-uzun-kök) ve D4 (kesme/rakam) **kapsam dışı**.

---

## §7 — Bilinen, kabul edilen kusurlar (peşinen beyan)

1. **LR programı yok** (§2) — sabit 2e-4. Uzun koşuda programlı düşüş daha iyi son checkpoint
   verebilir; bu koşum onu **ölçmez**. Karşılaştırılabilirlik bilinçli olarak öne alındı.
2. **`--load-optimizer` zinciri `model_sha256` kapısına bağlı** (T-0092); zincir kırılırsa
   koşum **rc=2 ile durur**, sessizce momentsız devam etmez.
3. **AdamW momentleri `anka_a1r.pt` için YOK** ⇒ **S1 momentsız** başlar; ilk güncelleme
   ölçülmüş olarak **1,7306×** büyüktür (T-0092). S2'den itibaren zincir momentsı taşır.
4. **Ceket külliyatı küçük** (784.797 jeton / 11.708 kayıt ≈ 67 jeton/kayıt) ⇒ ezber riski
   yapısal. §4/2 ve §4/4 bunu **ölçüp dışlamak** için var; yok saymak için değil.
