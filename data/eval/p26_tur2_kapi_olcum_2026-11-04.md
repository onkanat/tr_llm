# P2.6 Tur-2 — anka_taban_v3r2 Donmuş 4 Kapı Ölçümü

**Tarih:** 04.11.2026 · **Kalıp:** İLAN→HÜKÜM→RAPOR · **Kural:** commit YOK, eğitim KOŞMA (yalnız değerlendirme), izlenen dosyalara dokunma YOK (yalnız bu rapor + scratch probe'lar) · **Cihaz:** MPS (macOS)
**Önceki tur:** [P2.6 Tur-1 raporu — anka_taban_v3r 4 kapı ölçümü](p26_kapı_olcum_2026-11-04.md) (hüküm: 1/4 — yalnız KAPI-2 geçmişti)

---

## İLAN

P2.6 SFT-replay tur-2 çıktısı `data/anka_taban_v3r2.pt` modelinin **dört donmuş kapısı** resmi komutlarıyla ölçülür. Eşikler tur-1 ile BİREBİR aynı, önceden donmuş; hüküm koymadan önce dördü de ölçülür.

| Kapı | Metrik (adıyla) | Eşik |
|---|---|---|
| KAPI-1 | masked SFT val CE (`pure_base_sft.jsonl` son-150, T-0209 deseni) | ≤ 2,32 |
| KAPI-2 | külliyat pencere-CE (`eval_ce_val.py`, VAL bin, 500 pencere, tohum 1337) | ≤ 3,35 |
| KAPI-3 | dil bilgisi paketi güveni (test_model.py, 4 SFT görevi) | ≥ 0,99 VE 4 görevde top-1 doğru |
| KAPI-4 | üretim çöküşü yok (sonda: 3 donmuş sorgu, argmax + 0,8 frekans cezası, max_new=45) | toplam morfem ≥ 100 VE UNK ≤ %2,5 |

**Resmi komutlar:**

```
KAPI-1: venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur2_gate_sftval.py
          (trllm_p26_gate_sftval.py'nin v3r2 uyarlaması; TABAN+V3+V3R+V3R2 dördülüsü, aynı rig)
KAPI-2: venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r2.pt \
          --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin \
          --samples 500 --seed 1337
KAPI-3: venv/bin/python test_model.py --checkpoint data/anka_taban_v3r2.pt \
          --vocab data/rebuild/vocab_anka_r1_33114.json
KAPI-4: venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur2_sonda.py
          (trllm_p26_sonda_v3r.py kalıbının v3r2 uyarlaması; 3 donmuş sorgu aynen)
```

**Ölçüm kimlikleri (komut çıktısı, bu oturum):**

```
2e0e77ccc4bdd0cc432819d841aee12a81ce10d1cdf4decd1f598b0118b0dcd7  data/anka_taban_v3r2.pt  (tur-2 koşum kaydıyla birebir)
2ad5b24013f79969eedb9131cffc1917b103d6b37e1b600037e0d1eab7b3f791  data/anka_taban_v3r.pt   (tur-1 hüküm kaydıyla birebir)
eb2335b8ca1ee5969f42a96cd687b2ae48d5c53d7fb6682a76d5f326f33b020c  data/anka_taban_v3.pt    (T3.5 kaydıyla birebir)
5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt  (T-0209 hüküm kaydıyla birebir)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json (donmuş, 33.114)
```

**Referans bantlar:** TABAN masked SFT val CE 2,3043 / külliyat pencere-CE 3,8205; V3 3,4789 / 3,1936; V3R tur-1 2,737953 / 3,2118. Tur-2 koşum içi: SFT val CE 2,738→**2,585**; pencere-CE (TRAIN bin) 3,278→**3,286**. Tur-1 paket güvenleri: 0,9797/0,0884/0,6377/0,3844 (top-1 3/4); tur-1 sonda: 74 morfem / 3 UNK; TABAN sonda bandı (bu oturum): 127/1.

---

## HÜKÜM

**KAPI-1 — masked SFT val CE ≤ 2,32: KALDI.** V3R2 = **2,584789** (PPL 13,2605). Eşiğe açık **+0,265 nat**; TABAN bandına (2,3043) açık +0,280 nat. Tur-1→tur-2 kazancı **−0,153 nat** (tur-1'in −0,741 nat'ine karşı ~5× yavaşlama — tur-1 raporunun "kolay kazanç alındı" öngörüsü doğrulandı). Koşum içi bitiş değeri (2,585) bağımsız rig ile birebir örtüşüyor.

**KAPI-2 — külliyat pencere-CE ≤ 3,35: GEÇTİ.** V3R2 = **3,2181** (PPL 24,9800; VAL bin, 500 pencere, tohum 1337). Eşiğe pay **0,132 nat**. Tur-1'e ödün +0,006 nat (öngörü ~+0,02 bandında), V3'e toplam ödün +0,025 nat — KL β=0,2 çapası pencere-CE'yi v3 bandında tutmaya devam ediyor; TABAN'a iyileşme −0,602 nat.

**KAPI-3 — paket ≥ 0,99 VE 4/4 top-1: KALDI.** V3R2 dört görev güveni (test_model.py sırasıyla kök/zaman-kip/durum/çoğul): **0,9964 / 0,3146 / 0,7675 / 0,4415**; top-1 **3/4 doğru**. Görev-1 ilk kez 0,99 bandına girdi (0,9964, doğru jeton `root`). Görev-2 ('gideceğim' zaman/kip) top-1 yine **'root'** — DÜZELMEDİ: doğru etiket `TENSE` top-5'te bile görünmüyor (olasılık < 0,0162); güven 0,0884→0,3146 (3,6×) arttı ama kütleyi yanlış etikete yığdı. Görev-3/4 orta bant, 0,99'a uzak.

**KAPI-4 — toplam morfem ≥ 100 VE UNK ≤ %2,5: KALDI.** V3R2 3 sorguda toplam **100 morfem** (14/41/45) ve **3 UNK (%3,00)**. Morfem koşulu ilk kez karşılandı — ama SINIR DEĞERİNDE (100 ≥ 100, pay sıfır). UNK koşulu karşılanmadı (%3,00 > %2,5): UNK üretimi sorgu-başı tam 1'er, iki turda da aynı (3) — orandaki iyileşme (%4,05→%3,00) yalnız payda büyümesinden. Fidan sorgusundaki erken-EOS KIRILDI (15→41 morfem); Zımpara hâlâ 14'de erken kesiliyor; kontrol 45 tavana akıcı.

**Genel hüküm: 1/4 kapı GEÇTİ (yalnız KAPI-2 — tur-1 ile aynı kapı).** KAPI-1 kapanış hızı 0,741→0,153 nat/tur'a düştü; kalan 0,265 nat bu trendle bir turda (≈2,43) kapanmaz. MERDİVEN PROTOCOL GEREĞİ BURADA DURUR: üçüncü tur tanımlı değil. Ceket hattı bu noktada AÇILMAZ.

---

## RAPOR

### R1 — KAPI-1: masked SFT val CE (dördülü aynı rig)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur2_gate_sftval.py` (val = `pure_base_sft.jsonl` son-150, `FastSampleAlignedDataset(block=128)`, batch 16, shuffle=False; çıktı aynen):

```
cihaz: mps
val kayıt: 150 (pure_base_sft.jsonl son-150, T-0209 deseni)
[SFT-VAL] TABAN (pure) (data/anka_taban_pure.pt): CE=2.304304 | PPL=10.0172 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3 (data/anka_taban_v3.pt): CE=3.478886 | PPL=32.4236 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R (tur-1) (data/anka_taban_v3r.pt): CE=2.737953 | PPL=15.4553 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R2 (tur-2) (data/anka_taban_v3r2.pt): CE=2.584789 | PPL=13.2605 | eksik_anahtar=0 beklenmedik=0 | örnek=150
```

- Çıpalar birebir yeniden üretildi: TABAN 2,304304 (T-0209 kaydı 2,3049, Δ −0,0006), V3 3,478886 (T3.5 ile birebir), V3R 2,737953 (tur-1 hüküm ölçümüyle birebir — merdiven süreklilik kanıtı).
- V3R2 koşum içi bitiş değeri (2,585) kapı rig'iyle örtüşüyor → koşum raporu ile bağımsız ölçüm tutarlı.
- Tüm yükleme raporları `eksik=0 beklenmedik=0` — mimari sapması yok.
- **Hüküm: KALDI** (2,584789 > 2,32; açık +0,264789). Tur-1→tur-2 −0,153164 nat; aşırı-eğitim sinyali YOK (val CE indi: 2,738→2,585 — tur-1 raporunun "ikinci geçişte SFT val CE İNMELİDİR" koşulu sağlandı).

### R2 — KAPI-2: külliyat pencere-CE (resmi VAL bin)

Komut: `venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r2.pt --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337` (çıktı aynen):

```
[CE-VAL] model: data/anka_taban_v3r2.pt
[CE-VAL] vocab: data/rebuild/vocab_anka_r1_33114.json (33114 giriş) | val: data/kulliyat_v2/kulliyat_v2_val.bin (582,705 token)
[CE-VAL] block_size=128 | samples=500 | batch=16 | seed=1337 | device=mps
[CE-VAL] yükleme: atılan cos/sin/mask=0 | eksik=0 | beklenmedik=0
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.2181  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 24.9800
[CE-VAL] süre: 7.6 sn
```

- Çıpa teyidi (aynı rig, bu oturum): TABAN **3,8205** (PPL 45,6285), V3 **3,1936** (PPL 24,3756) — önceki kayıtlarla birebir.
- Kıyas: TABAN 3,8205 / V3 3,1936 / V3R 3,2118 / **V3R2 3,2181**. Replay'in düz-metin ödünü V3'ye göre toplam **+0,025 nat** (tur-1 sonrası +0,018; tur-2 ilavesi +0,006) — KL β=0,2 çapası ödünü ~0,02-0,03 nat bandında tuttu; tur-1 raporu tahmini (~3,23) ile uyumlu.
- Koşum içi TRAIN pencere-CE (3,286) ile resmi VAL ölçümü (3,2181) farkı 0,068 nat — tur-1'deki avlu farkıyla (0,066) aynı büyüklük, yön aynı; resmi kapı ölçümü VAL bin'dir.
- **Hüküm: GEÇTİ** (3,2181 ≤ 3,35; eşiğe pay 0,132 nat).

### R3 — KAPI-3: dil bilgisi paketi (test_model.py)

Komut: `venv/bin/python test_model.py --checkpoint data/anka_taban_v3r2.pt --vocab data/rebuild/vocab_anka_r1_33114.json`. `<OUTPUT>` sonrası top-1 öngörüleri (görev sırasıyla kök/zaman-kip/durum/çoğul):

| Görev | Girdi | V3R2 top-1 | Güven | Doğru mu? | V3R tur-1 | TABAN (bu oturumda aynı komutla) |
|---|---|---|---|---|---|---|
| kök morfemi bul | kitaplarda | `root` | **0,9964** | doğru | `root` 0,9797 | `root` 0,9997 |
| zaman/kip tespit | gideceğim | `root` | **0,3146** | **YANLIŞ** (beklenen `TENSE`) | `root` 0,0884 (yanlış) | `TENSE` 0,9981 |
| durum ekleri | okula | `case` | **0,7675** | doğru | `case` 0,6377 | `case` 0,9998 |
| çoğul eki | çocuklar | `PLURAL` | **0,4415** | doğru | `PLURAL` 0,3844 | `PLURAL` 0,9986 |

- Alet doğrulaması: TABAN bu oturumda aynı komutla 0,9997/0,9981/0,9998/0,9986 ve 4/4 doğru — tur-1 oturum ölçümüyle birebir teyit; ölçüm V3R2'de aynı rig'le yapıldı.
- Güven değişimleri tur-1→tur-2: görev-1 +0,0167 (0,99 bandına GİRDİ), görev-2 +0,2262 (0,09→0,31), görev-3 +0,1298, görev-4 +0,0571 — dört görevin HEPSİ ilerledi; ikinci replay geçişi güven ekseninde tekdüze kazanç verdi.
- Görev-2 kritik ayrıntı: doğru etiket `TENSE` (ID 32156) top-5'te YOK (5. sıradaki olasılık 0,0162'nin altında); top-5 = `root`/`bir`/`<EOS>`/`<UNK>`/`bu`. Model görev-2'de cevap kütlesini en sık görülen cevap etiketi olan 'root'a yığıyor — doğru etiket hiç temsil edilmemiş. Bu, epok sorunu DEĞİL veri sorunu kanıtıdır (bkz. R7).
- **Hüküm: KALDI** (görev-1 tek başına 0,99 üstünde olsa da kapı 4 görevde ister: görev-2 top-1 yanlış, görev-3/4 güvenleri 0,77/0,44 — 0,99 yok; top-1 3/4 ≠ 4/4).

### R4 — KAPI-4: üretim sağduyu sondası (donmuş 3 sorgu)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur2_sonda.py` (trllm_p26_sonda_v3r.py kalıbı; MPS, T-0157 maske, argmax + 0,8 frekans cezası, max_new=45, decompiler yüzey; sorgular tur-1 ile BİREBİR).

| Sorgu | V3R2 morfem | V3R2 UNK | V3R tur-1 | TABAN (bu oturum) |
|---|---|---|---|---|
| Zımpara nedir ve nasıl kullanılır? | **14 (erken EOS)** | 1 | 14 (erken EOS) | 45 |
| Fidan nasıl dikilir? | **41** | 1 | 15 (erken EOS) | 45 |
| Okula gitmek için hangi ekler kullanılır? (kontrol) | 45 (tavan) | 1 | 45 (tavan) | 37 |
| **TOPLAM** | **100** | **3 (%3,00)** | 74 / 3 (%4,05) | 127 / 1 (%0,79) |

- **Fidan erken-EOS'u KIRILDI** (15→41): "Bu durum, [?] bir şeyin kendilerine özgü olarak canlı kalmaktadır; ancak her iki cinsiyetten oluşan ve farksız olmaya devam ettiğimizi göstermiş olabilir." — gramerik, akıcı; konu fidandan uzak ama tur-1'deki 15-morfem kesilme davranışı gitti.
- **Zımpara erken-EOS'u DÜZELMEDİ** (14=14): "Usta Cevabı: [Özel İsim] [?], bir şeyin üzerine yerleştirilmiş olan" — iki turda da aynı noktada kesiliyor.
- Örnek kontrol yüzeyi (45 tavan): "Eğer bir şeyin varlığı, [?] ve her birinin kendilerini gösterdikten sonra [Özel İsim] ( veya bu durumda) olarak tanımlanabilecektir…" — yetenek tavanda sürüyor.
- UNK deseni iki turda da sorgu-başı TAM 1'er (toplam 3) — replay'den etkilenmeyen sabit bir üretim-hattı bileşeni; oran iyileşmesi (−1,05 puan) yalnız payda büyümesinden (74→100).
- TABAN, donmuş sorgularla bu oturumda 127/1/%0,79 — tur-1 ölçümüyle birebir, eşik altında rahat.
- **Hüküm: KALDI** (morfem 100 ≥ 100 koşulu sınır değerinde KARŞILANDI; UNK %3,00 > %2,5 koşulu KARŞILANMADI — kapı iki koşulun birlikte gerçekleşmesini ister).

### R5 — Tuhaflıklar

1. **KAPI-2 VAL < koşum içi TRAIN** (3,2181 vs 3,286; fark 0,068): tur-1'deki avlu farkıyla (0,066) aynı — sistemli, yön beklendiği gibi; resmi kapı ölçümü VAL bin'dir.
2. **UNK üretimi replay-bağışır**: iki turda da sorgu-başı tam 1'er UNK (3 toplam). UNK oranının kapıya takılması artık sayı sorunu değil: payda 100'e çıktı, oran %3,00 — üretim-hattı UNK'leri (alan kelime dağarcığı/sözlük kapsamı) ayrı veri bileşeni kalemi olarak takip edilmeli (tur-1 raporu R6 öngörüsü doğrulandı).
3. **Görev-2 iki geçişe de dirençli**: V3'te 'bir', V3R ve V3R2'de 'root'; TENSE hiçbir turda top-5'e giremedi. İkinci epok aynı SFT setini gördü ve düzeltemedi → sorun `pure_base_sft.jsonl` içeriğinde (bkz. R7).
4. **Morfem toplamı tam eşik değeri (100)**: '≥' koşulu karşılandı ama pay sıfır; Zımpara sorgusu iki turda da 14'te kesildi. Fidan'ın kırılması tur-2'nin davranış kazancı; kalan erken-EOS tek sorguya sıkıştı.
5. **Çıpa çakışması yok**: dört kapıda da tüm çıpalar önceki kayıtlarla birebir (KAPI-1: TABAN/V3/V3R; KAPI-2: TABAN 3,8205/V3 3,1936; KAPI-3: TABAN 4/4; KAPI-4: TABAN 127/1/%0,79) — rig tutarlı, ölçümler karşılaştırılabilir.

### R6 — Merdiven toplam tablosu ve duruş hükmü

| Merdiven | masked SFT val CE | külliyat pencere-CE (VAL) | Görev-1 güven (kök) | Görev-2 güven (zaman/kip) | Görev-3 güven (durum) | Görev-4 güven (çoğul) | top-1 | Sonda morfem / UNK% |
|---|---|---|---|---|---|---|---|---|
| TABAN (pure) | 2,3043 | 3,8205 | 0,9997 | 0,9981 | 0,9998 | 0,9986 | **4/4** | 127 / %0,79 |
| V3 (saf korpus devam-ön-eğitimi) | 3,4789 | 3,1936 | 0,0512 | 0,0491 | 0,0726 | 0,1595 | 3/4 | 55 / %3,64 |
| V3R (tur-1 replay) | 2,737953 | 3,2118 | 0,9797 | 0,0884 | 0,6377 | 0,3844 | 3/4 | 74 / %4,05 |
| **V3R2 (tur-2 replay)** | **2,584789** | **3,2181** | **0,9964** | **0,3146** | **0,7675** | **0,4415** | **3/4** | **100 / %3,00** |

*Provenans: TABAN ve V3R2 tüm sütunlar bu oturumda ölçüldü; V3 sütunları T3.5 kayıtlarından, V3R sütunları tur-1 oturum ölçümlerinden (tur-1 raporu); V3R'nin KAPI-1 değeri bu oturumda yeniden ölçülüp birebir teyit edildi. Donmuş 4 kapı ekseninden hiçbiri TABAN bandına dönmedi.*

**Duruş analizi:**

- **KAPI-1 kapanış hızı çöktü:** V3→V3R −0,741 nat, V3R→V3R2 −0,153 nat. Tur-1'in kapadığı 1,159 nat'lık açığın toplamı = 0,894 (=%77); kalan 0,265 nat bu trendle bir turda (≈2,43) kapanmaz. Tur-1 raporunun "kapı-1'in kapanış hızı yavaşlayacaktır" öngörüsü ölçüldü.
- **Aşırı-eğitim sinyali yok:** ikinci geçiş val CE'yi düşürdü (2,738→2,585), pencere-CE ödünü +0,006 nat ile sınırlı kaldı — duruş sebebi aşırı-eğitim değil, kapanış hızının eşiğe yetmemesi.
- **PROTOCOL gereği merdiven burada DURUR:** üçüncü tur tanımlı değil; trend −0,15 nat/tur ile 2,32'ye ulaşılamaz. Ceket hattı (v3 tabanlı) 4 kapı geçilmeden AÇILMAZ.
- **Tur-1 raporunun ikinci hüküm öngörüsü doğrulandı:** "görev-2 ikinci tur sonunda hâlâ yanlış top-1 veriyorsa sorun epok sayısında değil `pure_base_sft.jsonl` içindeki zaman/kip görevi örneklerindedir" — tam ölçüldüğü gibi gerçekleşti.

### R7 — SFT veri çeşitliliği kalemi: hangi görev türleri eksik

KAPI-3/KAPI-4 bulguları, replay ilerlemesinin tıkandığı yerleri veri bileşenine işaret ediyor:

1. **Zaman/kip tespiti (TENSE/MOOD cevap etiketli örnekler) — doğrudan eksik.** İki bağımsız replay geçişi de görev-2'yi düzeltemedi; model cevap kütlesini 'root'a yığarken `TENSE` jetonu top-5'e hiç giremiyor (p<0,016). SFT setinde bu görev tipinde ya örnek yok ya da cevap kalıbı hep aynı biçimde (ör. cevap hep başka bir etiketle) temsil edilmiş. Eklenmesi gereken: farklı fiil gövdeleri × zaman/kip kombinasyonları (TENSE_FUT/PAST/EVIDENTIAL, kipler) ile cevabı doğrudan `TENSE` olan çok sayıda örnek.
2. **Etiket-çeşitliliği genel olarak dar.** Görev-3 'case' 0,77 ve görev-4 'PLURAL' 0,44 hâlâ 0,99 bantının altında; iki turda güvenler artsalar da doygunluğa ulaşamıyor. SFT setinde cevap etiketi uzayı (ör. olumsuz örnekler: "çoğul eki YOKdur" cevaplı görevler, birden çok ek bir arada) genişletilmeli — model üç baskın etikete (root/case/PLURAL) sıkışmış durumda.
3. **Alan-soru uzun yanıt formatı eksik.** 'Zımpara' sorgusu iki turda da aynı 14. morfemde erken-EOS basarken kontrol sorgusu 45 tavana akıcı — yetenek değil, alan-soru (marangoz/bahçıvan kartı) formatına kaçış davranışı. SFT karışımına uzun yanıtlı alan kartları (özellikle "nedir ve nasıl kullanılır" kalıbı) eklenmeli.
4. **Üretim-hattı UNK'leri ayrı kalem.** Sorgu-başı 1'er UNK iki turda sabit: replay bu bileşene erişemiyor. Alan kelime dağarcığı/sözlük kapsamı kalemi olarak ayrı takip edilmeli (KAPI-4'ün UNK koşulu ancak bu kalemden düzelir).

---

**Ölçüm oturumu:** 04.11.2026, MPS; tüm koşumlar salt-okuma, repo izlenen dosyalarına dokunulmadı; loglar `~/.hermes/cache/scratch/p26_tur2_gate{1,2,3,4}_*.log` (çıpa teyitleri: `p26_tur2_gate2_cipa_{taban,v3}.log`, `p26_tur2_gate3_taban_ref.log`).