# P2.6 Tur-4 — anka_taban_v3r4 Donmuş 4 Kapı Ölçümü

**Tarih:** 04.11.2026 · **Kalıp:** İLAN→HÜKÜM→RAPOR · **Kural:** commit YOK, eğitim KOŞMA (yalnız değerlendirme), izlenen dosyalara dokunma YOK (yalnız bu rapor + scratch probe'lar) · **Cihaz:** MPS (macOS)
**Önceki tur:** [P2.6 Tur-3 raporu — anka_taban_v3r3 4 kapı ölçümü](p26_tur3_kapi_olcum_2026-11-04.md) (hüküm: 1/4) · [Tur-2](p26_tur2_kapi_olcum_2026-11-04.md) (1/4) · [Tur-1](p26_kapı_olcum_2026-11-04.md) (1/4)

---

## İLAN

P2.6 tur-4 çıktısı `data/anka_taban_v3r4.pt` modelinin **dört donmuş kapısı** resmi komutlarıyla ölçülür. Eşikler tur-1/2/3 ile BİREBİR aynı, önceden donmuş; hüküm koymadan önce dördü de ölçülür.

| Kapı | Metrik (adıyla) | Eşik |
|---|---|---|
| KAPI-1 | masked SFT val CE (`pure_base_sft.jsonl` son-150, T-0209 deseni) | ≤ 2,32 |
| KAPI-2 | külliyat pencere-CE (`eval_ce_val.py`, VAL bin, 500 pencere, tohum 1337) | ≤ 3,35 |
| KAPI-3 | dil bilgisi paketi güveni (test_model.py, 4 SFT görevi) | ≥ 0,99 VE 4 görevde top-1 doğru |
| KAPI-4 | üretim çöküşü yok (sonda: 3 donmuş sorgu, argmax + 0,8 frekans cezası, max_new=45) | toplam morfem ≥ 100 VE UNK ≤ %2,5 |

**Tur-4 farklılığı (veri-bileşimi kalemi):** tur-4, tur-3'ten devralınan replay'e ek olarak **zenginleştirilmiş `pure_base_sft_v3.jsonl`** ile eğitildi: **+446 zaman/kip kaydı** (733 geçen adaydan kürasyonla; 40 fiil × 30 biçim ızgarası, olumsuz + 2. şahıs + 15 yeni gövde; bkz. [p26_zaman_doz raporu](p26_zaman_doz_2026-11-04.md)). Zaman/kip dozu **204 → 650** (3,2×). Dosya 4.763 → 5.209 satır (ölçüldü, bu oturum). Val karşılaştırılabilirliği: **v3'ün son-150 val kuyruğu v2'ninkiyle ve v2'ninki v1'inkiyle BAYT-ÖZDEŞ** (bu oturumda yeniden ölçüldü: üç dosyanın son-150 SHA'ları aynı, `89c9b3e714786fd4` öneki) — dolayısıyla KAPI-1'in donmuş val kümesi tur-4 koşumunun val kümesiyle aynıdır.

**Resmi komutlar:**

```
KAPI-1: venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur4_gate_sftval.py
          (trllm_p26_tur3_gate_sftval.py'nin v3r4 uyarlaması; TABAN/V3/V3R/V3R2/V3R3/V3R4 altılısı, aynı rig)
KAPI-2: venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r4.pt \
          --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin \
          --samples 500 --seed 1337
KAPI-3: venv/bin/python test_model.py --checkpoint data/anka_taban_v3r4.pt \
          --vocab data/rebuild/vocab_anka_r1_33114.json
KAPI-4: venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur4_sonda.py
          (trllm_p26_tur3_sonda.py kalıbının v3r4 uyarlaması; 3 donmuş sorgu aynen)
```

**Ölçüm kimlikleri (sha256, bu oturum):**

```
2cb4735530ae54037cf44db3b0a12365ed13acf5d21532655a443a4d549d778d  data/anka_taban_v3r4.pt  (tur-4 koşum kaydıyla birebir: p26_tur4_hukum cikti_sha256)
40c9459ac3f4693d2bbb915f80b888f0763dd9df19cd38b15a3ea1fb6dfb4a49  data/anka_taban_v3r3.pt  (tur-3 hüküm kaydıyla birebir)
2e0e77ccc4bdd0cc432819d841aee12a81ce10d1cdf4decd1f598b0118b0dcd7  data/anka_taban_v3r2.pt  (tur-2 hüküm kaydıyla birebir)
2ad5b24013f79969eedb9131cffc1917b103d6b37e1b600037e0d1eab7b3f791  data/anka_taban_v3r.pt   (tur-1 hüküm kaydıyla birebir)
eb2335b8ca1ee5969f42a96cd687b2ae48d5c53d7fb6682a76d5f326f33b020c  data/anka_taban_v3.pt    (T3.5 kaydıyla birebir)
5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt  (T-0209 hüküm kaydıyla birebir)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json (donmuş, 33.114)
f0834dee8175e3535d1c47d65f071d40cae1e1d029fa39948d05599d72dfb0df  data/pedagogy/pure_base_sft_v3.jsonl (tur-4 eğitim verisi, kalem-2 kaydıyla birebir)
```

**Referans bantlar:** TABAN masked SFT val CE 2,3043 / külliyat pencere-CE 3,8205; V3 3,4789 / 3,1936; V3R 2,7380 / 3,2118; V3R2 2,5848 / 3,2181; V3R3 2,4827 / 3,2230. Tur-4 koşum içi (`p26_tur4_hukum_2026-11-04.json`): init_val_ce 2,482663 → final_val_ce **2,439043**; pencere-CE (TRAIN bin) 3,2944 → **3,2978**; öğretmen V3, KL β=0,2, 1 epok, 198 opt-adım, SFT train 5.059, pencere 1.265. Tur-3 paket güvenleri: 0,9981/0,4103/0,9196/0,8252 (top-1 3/4; görev-2'de `TENSE` top-2, p=0,0564); tur-3 sonda: 86 morfem (30/11/45) / 2 UNK (%2,33); TABAN sonda bandı: 127/1/%0,79.

---

## HÜKÜM

**KAPI-1 — masked SFT val CE ≤ 2,32: KALDI.** V3R4 = **2,439044** (PPL 11,4621). Eşiğe açık **+0,119044 nat**; TABAN bandına (2,3043) açık +0,135 nat. Tur-3→tur-4 kazancı **−0,0436 nat** — kapanış hızı monoton yavaşlıyor (−0,741 → −0,153 → −0,102 → −0,044). Koşum içi bitiş değeri (2,439043) bağımsız kapı rig'i ile örtüşüyor (2,439044; 6 basamakta aynı).

**KAPI-2 — külliyat pencere-CE ≤ 3,35: GEÇTİ.** V3R4 = **3,2264** (PPL 25,1893; VAL bin, 500 pencere, tohum 1337). Eşiğe pay **0,124 nat**. Tur-3'ye ödün +0,0034 nat, V3'e toplam ödün +0,0328 nat — KL β=0,2 çapası pencere-CE'yi v3 bandında tutmaya devam ediyor; TABAN'a iyileşme −0,5941 nat.

**KAPI-3 — paket ≥ 0,99 VE 4/4 top-1: KALDI.** V3R4 dört görev güveni (test_model.py sırasıyla kök/zaman-kip/durum/çoğul): **0,9988 / 0,9325 / 0,9807 / 0,9218**; top-1 **4/4 doğru — MERDİVENDE İLK KEZ** (TABAN dışında). **Görev-2 'gideceğim' top-1 ÇEVRİLDİ: `TENSE` (ID 32156) 0,9325** — 3× zaman/kip dozu (204→650) root prior'ını devirdi ('root' 0,4103 → 0,0289, 2. sıra). AMA güven eşiği dönmedi: görev-2 0,9325, görev-3 0,9807, görev-4 0,9218 0,99 altında.

**KAPI-4 — toplam morfem ≥ 100 VE UNK ≤ %2,5: KALDI.** V3R4 3 sorguda toplam **72 morfem** (37/11/24) ve **2 UNK (%2,78)** — iki koşul da kaçtı. Tur-3'ün kırılganlık öngörüsü (R5.5: "payda düşerse oran tekrar üstüne fırlar") gerçekleşti: UNK sayısı aynı (2) ama payda 86→72 düşünce %2,33 → %2,78. Zımpara 30→37 (hâlâ tavan değil), fidan 11'de sabit (erken EOS), **kontrol GERİLEDİ** (45 tavan → 24 — kontrol sorgusu ilk kez erken EOS verdi).

**Genel hüküm: 1/4 kapı GEÇTİ (yalnız KAPI-2 — dört turda da geçen tek kapı).** Zaman/kip doz kalemi görev-2'yi top-1'e taşıdı (tur-3 reçetesinin doğrulaması) ve dört görevin top-1'ini ilk kez TAMAMLADI; ama güven eşiği (0,99) ve üretim uzunluğu eksenleri donmuş eşiklerin altında kaldı. Ceket hattı 4 kapı geçilmeden AÇILMAZ; v3r4 donmuş kapılara göre ceket tabanı DEĞİL.

---

## RAPOR

### R1 — KAPI-1: masked SFT val CE (altılı aynı rig)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur4_gate_sftval.py` (val = `pure_base_sft.jsonl` son-150, `FastSampleAlignedDataset(block=128)`, batch 16, shuffle=False; çıktı aynen):

```
cihaz: mps
val kayıt: 150 (pure_base_sft.jsonl son-150, T-0209 deseni)
[SFT-VAL] TABAN (pure) (data/anka_taban_pure.pt): CE=2.304304 | PPL=10.0172 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3 (data/anka_taban_v3.pt): CE=3.478886 | PPL=32.4236 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R (tur-1) (data/anka_taban_v3r.pt): CE=2.737953 | PPL=15.4553 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R2 (tur-2) (data/anka_taban_v3r2.pt): CE=2.584789 | PPL=13.2605 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R3 (tur-3) (data/anka_taban_v3r3.pt): CE=2.482663 | PPL=11.9731 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R4 (tur-4) (data/anka_taban_v3r4.pt): CE=2.439044 | PPL=11.4621 | eksik_anahtar=0 beklenmedik=0 | örnek=150
```

- Çıpalar birebir yeniden üretildi: TABAN 2,304304 (T-0209 kaydı 2,3049), V3 3,478886 (T3.5 ile birebir), V3R 2,737953 (tur-1 hüküm), V3R2 2,584789 (tur-2 hüküm), V3R3 2,482663 (tur-3 hüküm) — merdiven süreklilik kanıtı; TABAN+V3 çıpaları görev gereği bu oturumda aynı rigde yeniden ölçüldü.
- V3R4 koşum içi final_val_ce (2,4390435099601744) kapı rig'iyle (2,439044) örtüşüyor — v3 val-150 kuyruğunun v1'le bayt-özdeşliği (bu oturumda yeniden ölçüldü) sayesinde koşum raporu ile bağımsız ölçüm aynı val kümesinde.
- Kapanış hızı: V3→V3R −0,741 → V3R→V3R2 −0,153 → V3R2→V3R3 −0,102 → V3R3→V3R4 **−0,044** nat. V3 eşiğine kapatılan toplam 1,040 nat (=açığın %90'ı); kalan **0,119044 nat**. Tur-3'ün "mevcut trendle ≈2,38" projeksiyonu gerçekleşmedi (gelen −0,044) — doz kalemi SFT val CE'ye sınır-kazanç veriyor. Mevcut hızla kalan 0,119 nat ≈ 3 tur demek; tek veri kalemiyle donmuş 2,32 kapanmıyor.
- Tüm yükleme raporları `eksik=0 beklenmedik=0` — mimari sapması yok. Aşırı-eğitim sinyali YOK (val CE indi: 2,483→2,439).
- **Hüküm: KALDI** (2,439044 > 2,32).

### R2 — KAPI-2: külliyat pencere-CE (resmi VAL bin)

Komut: `venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r4.pt --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337` (çıktı aynen):

```
[CE-VAL] model: data/anka_taban_v3r4.pt
[CE-VAL] vocab: data/rebuild/vocab_anka_r1_33114.json (33114 giriş) | val: data/kulliyat_v2/kulliyat_v2_val.bin (582,705 token)
[CE-VAL] block_size=128 | samples=500 | batch=16 | seed=1337 | device=mps
[CE-VAL] yükleme: atılan cos/sin/mask=0 | eksik=0 | beklenmedik=0
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.2264  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 25.1893
[CE-VAL] süre: 6.7 sn
```

- Çıpa teyidi (aynı rig, bu oturum): TABAN **3,8205** (PPL 45,6285), V3 **3,1936** (PPL 24,3756), V3R **3,2118** (PPL 24,8233), V3R2 **3,2181** (PPL 24,9800), V3R3 **3,2230** (PPL 25,1030) — önceki tüm kayıtlarla birebir; pencere-CE sütunu bu oturumda altılı tam.
- Kıyas: TABAN 3,8205 / V3 3,1936 / V3R 3,2118 / V3R2 3,2181 / V3R3 3,2230 / **V3R4 3,2264**. Replay'in düz-metin ödünü V3'ye göre toplam **+0,0328 nat** (tur başına: +0,018/+0,006/+0,0049/**+0,0034**) — ödün büyüyor ama ivmesi yavaşlıyor; KL β=0,2 çapası ödünü ~0,03 nat bandında tutmayı sürdürüyor.
- Koşum içi TRAIN pencere-CE (3,2978) ile resmi VAL ölçümü (3,2264) farkı 0,071 nat — önceki turların avlu farklarıyla (0,066/0,068/0,071) aynı büyüklük; resmi kapı ölçümü VAL bin'dir.
- **Hüküm: GEÇTİ** (3,2264 ≤ 3,35; eşiğe pay 0,124 nat).

### R3 — KAPI-3: dil bilgisi paketi (test_model.py)

Komut: `venv/bin/python test_model.py --checkpoint data/anka_taban_v3r4.pt --vocab data/rebuild/vocab_anka_r1_33114.json`. `<OUTPUT>` sonrası top-1 öngörüleri (görev sırasıyla kök/zaman-kip/durum/çoğul):

| Görev | Girdi | V3R4 top-1 | Güven | Doğru mu? | V3R3 tur-3 | TABAN (bu oturumda aynı komutla) |
|---|---|---|---|---|---|---|
| kök morfemi bul | kitaplarda | `root` | **0,9988** | doğru | `root` 0,9981 | `root` 0,9997 |
| zaman/kip tespit | gideceğim | **`TENSE`** | **0,9325** | **DOĞRU — top-1 ÇEVRİLDİ** | `root` 0,4103 (yanlış) | `TENSE` 0,9981 |
| durum ekleri | okula | `case` | **0,9807** | doğru | `case` 0,9196 | `case` 0,9998 |
| çoğul eki | çocuklar | `PLURAL` | **0,9218** | doğru | `PLURAL` 0,8252 | `PLURAL` 0,9986 |

- Alet doğrulaması: TABAN bu oturumda aynı komutla 0,9997/0,9981/0,9998/0,9986 ve 4/4 doğru — tur-1/2/3 ölçümleriyle birebir; V3/V3R/V3R2/V3R3 güvenleri de bu oturumda yeniden ölçüldü ve TÜM önceki kayıtlarla birebir çıktı (loglar: `p26_tur4_gate3_{taban_pure,v3,v3r,v3r2,v3r3}.log`).
- **Görev-2 'gideceğim' top-5 kanıtı (V3R4, bu oturum):**

```
1. TENSE   | 0,9325 (ID 32156)   ← top-1 ÇEVRİLDİ — DOĞRU
2. root    | 0,0289 (ID 16656)   ← eski sahte-kral 2. sıraya düştü
3. <EOS>   | 0,0030 (ID 3)
4. <UNK>   | 0,0015 (ID 0)
5. PLURAL  | 0,0011 (ID 70)
```

  Tur-1: `TENSE` top-5 dışı (p<0,0162) → tur-2: dışı → tur-3: top-2 (0,0564) → **tur-4: top-1 (0,9325)**. 3× doz (204→650 zaman/kip kaydı; olumsuz + 2sg + 15 yeni gövde) tur-3'ün reçetesini doğruladı: 'root' prior'ı 7,3×'tan 31×'e tersine döndü (0,4103 vs 0,0564 → 0,0289 vs 0,9325). TENSE sonrası 2. tahmin ':' 0,9972 — yanıt şablonu da öğrenilmiş.
- Diğer görevlerin top-5'leri: görev-1 `root` 0,9988 (sonra `<PROPER_NOUN>` 0,0001); görev-3 `case` 0,9807 (sonra `PLURAL` 0,0099); görev-4 `PLURAL` 0,9218 (sonra `case` 0,0072, `POSS_3PL` 0,0070, `TENSE` 0,0068).
- Güven değişimleri tur-3→tur-4: görev-1 +0,0007 (0,9988), görev-2 **+0,5222** (0,4103→0,9325), görev-3 +0,0611 (0,9807), görev-4 +0,0966 (0,9218). Güven ortalaması 0,79→0,96; top-1 sayısı 3/4→**4/4**.
- **Hüküm: KALDI** — top-1 4/4 İLK KEZ tamam ama eşik ÇİFT koşullu: görev-2 0,9325, görev-3 0,9807 ve görev-4 0,9218 0,99 eşliğinin altında (TABAN'ın 0,9997/0,9981/0,9998/0,9986 bandına açık kalan mesafe).

### R4 — KAPI-4: üretim sağduyu sondası (donmuş 3 sorgu)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur4_sonda.py` (trllm_p26_tur3_sonda.py kalıbı; MPS, T-0157 maske, argmax + 0,8 frekans cezası, max_new=45, decompiler yüzey; sorgular tur-1/2/3 ile BİREBİR).

| Sorgu | V3R4 morfem | V3R4 UNK | V3R3 tur-3 | TABAN (bu oturum) |
|---|---|---|---|---|
| Zımpara nedir ve nasıl kullanılır? | **37 (erken EOS, tavan değil)** | 1 | 30 (erken EOS) | 45 |
| Fidan nasıl dikilir? | **11 (erken EOS)** | 0 | 11 (erken EOS) | 45 |
| Okula gitmek için hangi ekler kullanılır? (kontrol) | **24 (erken EOS — GERİLEDİ)** | 1 | 45 (tavan) | 37 |
| **TOPLAM** | **72** | **2 (%2,78)** | 86 / 2 (%2,33) | 127 / 1 (%0,79) |

- **Zımpara 30→37:** uzun 'nedir/nasıl' + zaman/kip kayıtlarının etkisi sürüyor ama tavan (45) yine tutmadı: "Verniklerin yüzeye yapışarak, camdan yapılmış bir aletle bağlanmayı sağlayan veya ahşap parçalarının uç kısımlı olduğumuzda [?] ( demir)" — 1 UNK.
- **Fidan 11'de sabit:** "Bu işini yapmak için kullanılan bir söz" — tur-3 çıktısıyla aynı yüzey; erken-EOS davranışı veri dozundan etkilenmiyor.
- **Kontrol GERİLEDİ (45→24):** tur-1'den beri tavana yakın akan kontrol sorgusu İLK KEZ erken EOS verdi ("Eğer bir şeyin varlığı, [?] ve her birinin kendilerini gösterdikten sonra [Özel İsim] ( s)"). Zaman/kip dozu görev-eksenini iyileştirirken üretim-uzunluğu ekseninde ödün aldı.
- **UNK %2,78 > %2,5:** UNK sayısı değişmedi (2) ama payda 86→72 düşünce oran tur-3 raporu R5.5'in öngördüğü gibi tekrar eşiğin üstüne fırladı. UNK koşulu tur-3'ten tur-4'e KÖTÜLEŞEREK döndü.
- TABAN, donmuş sorgularla bu oturumda 127/1/%0,79 — tur-1/2/3 ölçümleriyle birebir, eşik altında rahat.
- **Hüküm: KALDI** (morfem 72 < 100 VE UNK %2,78 > %2,5 — kapının iki koşulu da kaçtı).

### R5 — Tuhaflıklar

1. **Görev-eksen ↔ üretim-eksen takası:** tur-4, görev-2'yi top-1'e taşırken (0,4103→0,9325) kontrol sorgusunun üretim uzunluğunu feda etti (45→24). İki eksen aynı dozdan zıt yönde etkilendi — tek kalemle dört kapının aynı anda dönmeyeceğinin en net ölçümü.
2. **KAPI-1 kapanış hızı doygun:** −0,741 → −0,153 → −0,102 → −0,044 nat/tur. Doz artışına (194→446 kayıt) rağmen kazanç küçüldü; kalan 0,119 nat bu hızla ~3 tur. SFT val CE ekseninde veri kalemi sınır-kazanç veriyor; eşik çevrimi epok/öğrenme-oranı kalemini ya da val dağılımına daha yakın bir kayıt bileşimi gerektiriyor.
3. **Güven eşiği son bariyer:** top-1 4/4 tamam; kalan mesafe kalibre güven (0,99). TABAN bantları 0,998+; V3R4 üç görevde 0,92-0,98 bandında. Doz kalemi güveni de monoton kaldırıyor (+0,52/+0,06/+0,10) — bir tur daha aynı strateji işleyebilir ama morfem/UNK eksenini birlikte çevirmek için uzun-yanıt kartı kalemi şart.
4. **Çıpa çakışması yok:** altı modelin SFT val CE'si, pencere-CE'si (altılı), güvenleri (altılı) ve TABAN sondası bu oturumda yeniden ölçüldü ve TÜM önceki kayıtlarla birebir — rig tutarlı, merdiven dört turla tamamen karşılaştırılabilir.
5. **UNK oranı payda-kırılgan:** %2,78'in payı (2 UNK) üç turda değişmedi; oran oynayan şey payda (86→72). UNK koşulunun sağlam çözümü üretim-hattı kelime dağarcığı kalemi — doz kalemiyle iyileşmez.

### R6 — Merdiven toplam tablosu ve duruş hükmü

| Merdiven | masked SFT val CE | külliyat pencere-CE (VAL) | Görev-1 güven (kök) | Görev-2 güven (zaman/kip) | Görev-3 güven (durum) | Görev-4 güven (çoğul) | top-1 | Sonda morfem / UNK% |
|---|---|---|---|---|---|---|---|---|
| TABAN (pure) | 2,3043 | 3,8205 | 0,9997 | 0,9981 | 0,9998 | 0,9986 | **4/4** | 127 / %0,79 |
| V3 (saf korpus devam-ön-eğitimi) | 3,4789 | 3,1936 | 0,0512 | 0,0491 | 0,0726 | 0,1595 | 3/4 | 55 / %3,64 |
| V3R (tur-1 replay) | 2,7380 | 3,2118 | 0,9797 | 0,0884 | 0,6377 | 0,3844 | 3/4 | 74 / %4,05 |
| V3R2 (tur-2 replay) | 2,5848 | 3,2181 | 0,9964 | 0,3146 | 0,7675 | 0,4415 | 3/4 | 100 / %3,00 |
| V3R3 (tur-3 replay + veri-zenginleştirme) | 2,4827 | 3,2230 | 0,9981 | 0,4103 | 0,9196 | 0,8252 | 3/4 | 86 / %2,33 |
| **V3R4 (tur-4 replay + 3× zaman/kip dozu)** | **2,4390** | **3,2264** | **0,9988** | **0,9325** | **0,9807** | **0,9218** | **4/4** | **72 / %2,78** |

*Provenans: SFT val CE altılısı, pencere-CE altılısı, güven altılısı ve TABAN sondası bu oturumda ölçüldü (V3/V3R/V3R2/V3R3 çıpaları önceki kayıtlarla birebir; V3/V3R/V3R2 sonda değerleri kendi tur raporlarından). Donmuş 4 kapı ekseninden hiçbiri TABAN bandına dönmedi; V3R4 görev top-1 ekseninde TABAN'a İLK ULAŞAN basamak.*

**Duruş analizi:**

- **Zaman/kip doz kalemi görev-2'yi ÇEVİRDİ:** TENSE 0→top-2→**top-1 0,9325**; top-1 sayısı ilk kez 4/4; dört görevin güveni de yükseldi (ortalama 0,79→0,96). Tur-3 R5.2 reçetesi ("doz artırılıp çeşitlendirilmeden top-1 çevrilmez") deneysel olarak doğrulandı.
- **Ama donmuş hedefler aynı:** KAPI-1'e kalan 0,119 nat (kapanış hızı −0,044/tur ile ~3 tur), KAPI-3'te üç güven 0,99 altı (0,9325/0,9807/0,9218), KAPI-4'te morfem 72<100 VE UNK %2,78>%2,5 (iki koşul birden kaçtı; tur-3'ün geçtiği UNK koşulu da geri döndü). 1/4 — dört turda da geçen tek kapı KAPI-2.
- **Ceket hattı bu noktada AÇILMAZ; v3r4 donmuş kapılara göre ceket tabanı DEĞİL.** Merdiven artık iki kampa ayrıldı: görev/CE eksenleri (KAPI-1/3) replay+doz kalemiyle monoton ilerliyor; üretim-uzunluğu eksenleri (KAPI-4) aynı kalemden zıt yönde etkileniyor (kontrol 45→24). Sıradaki kalemler: (1) güven eşiği için doz'un devamı ya da kalibrasyon kalemi, (2) **uzun-yanıt alan kartı kalemi** (morfem/UNK eksenini KAPI-1'i yavaşlatmadan çevirmek için — tur-4'ün takas bulgusu bunu zorunlu kılıyor), (3) KAPI-1'in 0,12 nat'ı — veri kalemi doygun; epok/ön-eğitim bileşimi gözden geçirilmeden kapanmaz.

---

**Ölçüm oturumu:** 04.11.2026, MPS; tüm koşumlar salt-okuma, repo izlenen dosyalarına dokunulmadı; loglar `~/.hermes/cache/scratch/p26_tur4_gate{1,2,3,4}_*.log` (çıpa teyitleri: `p26_tur4_gate2_cipa_{taban,v3,v3r,v3r2,v3r3}.log`, `p26_tur4_gate3_{taban_pure,v3,v3r,v3r2,v3r3}.log`).