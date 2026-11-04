# P2.6 Tur-3 — anka_taban_v3r3 Donmuş 4 Kapı Ölçümü

**Tarih:** 04.11.2026 · **Kalıp:** İLAN→HÜKÜM→RAPOR · **Kural:** commit YOK, eğitim KOŞMA (yalnız değerlendirme), izlenen dosyalara dokunma YOK (yalnız bu rapor + scratch probe'lar) · **Cihaz:** MPS (macOS)
**Önceki tur:** [P2.6 Tur-2 raporu — anka_taban_v3r2 4 kapı ölçümü](p26_tur2_kapi_olcum_2026-11-04.md) (hüküm: 1/4) · [Tur-1 raporu](p26_kapı_olcum_2026-11-04.md) (1/4)

---

## İLAN

P2.6 tur-3 çıktısı `data/anka_taban_v3r3.pt` modelinin **dört donmuş kapısı** resmi komutlarıyla ölçülür. Eşikler tur-1/tur-2 ile BİREBİR aynı, önceden donmuş; hüküm koymadan önce dördü de ölçülür.

| Kapı | Metrik (adıyla) | Eşik |
|---|---|---|
| KAPI-1 | masked SFT val CE (`pure_base_sft.jsonl` son-150, T-0209 deseni) | ≤ 2,32 |
| KAPI-2 | külliyat pencere-CE (`eval_ce_val.py`, VAL bin, 500 pencere, tohum 1337) | ≤ 3,35 |
| KAPI-3 | dil bilgisi paketi güveni (test_model.py, 4 SFT görevi) | ≥ 0,99 VE 4 görevde top-1 doğru |
| KAPI-4 | üretim çöküşü yok (sonda: 3 donmuş sorgu, argmax + 0,8 frekans cezası, max_new=45) | toplam morfem ≥ 100 VE UNK ≤ %2,5 |

**Tur-3 farklılığı (veri-bileşimi kalemi):** tur-3, replay'e ek olarak **zenginleştirilmiş `pure_base_sft_v2.jsonl`** ile eğitildi: +194 zaman/kip +40 uzun 'nedir/nasıl' +31 görev-3/4 = **+265 kayıt** (v1: 4.498 → v2: 4.763; ölçüldü, bu oturum). Val karşılaştırılabilirliği: **v2'nin son-150 val kuyruğu v1'inkiyle BAYT-ÖZDEŞ** (ölçüldü, bu oturum) — dolayısıyla KAPI-1'in donmuş val kümesi tur-3 koşumunun val kümesiyle aynıdır.

**Resmi komutlar:**

```
KAPI-1: venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur3_gate_sftval.py
          (trllm_p26_tur2_gate_sftval.py'nin v3r3 uyarlaması; TABAN/V3/V3R/V3R2/V3R3 beşlisi, aynı rig)
KAPI-2: venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r3.pt \
          --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin \
          --samples 500 --seed 1337
KAPI-3: venv/bin/python test_model.py --checkpoint data/anka_taban_v3r3.pt \
          --vocab data/rebuild/vocab_anka_r1_33114.json
KAPI-4: venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur3_sonda.py
          (trllm_p26_tur2_sonda.py kalıbının v3r3 uyarlaması; 3 donmuş sorgu aynen)
```

**Ölçüm kimlikleri (sha256, bu oturum):**

```
40c9459ac3f4693d2bbb915f80b888f0763dd9df19cd38b15a3ea1fb6dfb4a49  data/anka_taban_v3r3.pt  (tur-3 koşum kaydıyla birebir: p26_replay_hukum cikti_sha256)
2e0e77ccc4bdd0cc432819d841aee12a81ce10d1cdf4decd1f598b0118b0dcd7  data/anka_taban_v3r2.pt  (tur-2 hüküm kaydıyla birebir)
2ad5b24013f79969eedb9131cffc1917b103d6b37e1b600037e0d1eab7b3f791  data/anka_taban_v3r.pt   (tur-1 hüküm kaydıyla birebir)
eb2335b8ca1ee5969f42a96cd687b2ae48d5c53d7fb6682a76d5f326f33b020c  data/anka_taban_v3.pt    (T3.5 kaydıyla birebir)
5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt  (T-0209 hüküm kaydıyla birebir)
f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984  data/rebuild/vocab_anka_r1_33114.json (donmuş, 33.114)
```

**Referans bantlar:** TABAN masked SFT val CE 2,3043 / külliyat pencere-CE 3,8205; V3 3,4789 / 3,1936; V3R 2,737953 / 3,2118; V3R2 2,584789 / 3,2181. Tur-3 koşum içi (`p26_replay_hukum_2026-11-04.json`): init_val_ce 2,584789 → final_val_ce **2,482663**; pencere-CE (TRAIN bin) 3,2862 → **3,2944**; öğretmen V3, KL β=0,2, 1 epok, 178 opt-adım, SFT train 4.613. Tur-2 paket güvenleri: 0,9964/0,3146/0,7675/0,4415 (top-1 3/4; görev-2'de `TENSE` top-5 dışı, p<0,0162); tur-2 sonda: 100 morfem (14/41/45) / 3 UNK (%3,00); TABAN sonda bandı: 127/1/%0,79.

---

## HÜKÜM

**KAPI-1 — masked SFT val CE ≤ 2,32: KALDI.** V3R3 = **2,482663** (PPL 11,9731). Eşiğe açık **+0,162663 nat**; TABAN bandına (2,3043) açık +0,178 nat. Tur-2→tur-3 kazancı **−0,102 nat** (tur-1→tur-2 −0,153'e karşı hafif iyileşme — veri-zenginleştirme, yalnız-epok tur-2 trendinin (−0,10 civarı) üzerine çıkamadı). Koşum içi bitiş değeri (2,482663) bağımsız kapı rig'i ile BİREBİR örtüşüyor (val kuyruğu bayt-özdeş — İLAN'daki kanıt).

**KAPI-2 — külliyat pencere-CE ≤ 3,35: GEÇTİ.** V3R3 = **3,2230** (PPL 25,1030; VAL bin, 500 pencere, tohum 1337). Eşiğe pay **0,127 nat**. Tur-2'ye ödün +0,0049 nat, V3'e toplam ödün +0,0294 nat — KL β=0,2 çapası pencere-CE'yi v3 bandında tutmaya devam ediyor; TABAN'a iyileşme −0,5975 nat.

**KAPI-3 — paket ≥ 0,99 VE 4/4 top-1: KALDI.** V3R3 dört görev güveni (test_model.py sırasıyla kök/zaman-kip/durum/çoğul): **0,9981 / 0,4103 / 0,9196 / 0,8252**; top-1 **3/4 doğru**. Görev-2 ('gideceğim' zaman/kip) top-1 yine **'root'** (0,4103) — DÜZELMEDİ; **AMA 194 zaman/kip kaydı ölçülebilir etki yaptı: doğru etiket `TENSE` (ID 32156) İLK KEZ top-5'e girdi — 2. sıra, olasılık 0,0564** (tur-2'de top-5 dışı, p<0,0162). Görev-3/4 güvenleri 0,77→0,92 ve 0,44→0,83 sıçradı (31 görev-3/4 kaydının etkisi) ama 0,99'a ulaşmadı.

**KAPI-4 — toplam morfem ≥ 100 VE UNK ≤ %2,5: KALDI.** V3R3 3 sorguda toplam **86 morfem** (30/11/45) ve **2 UNK (%2,33)**. **UNK koşulu İLK KEZ karşılandı** (%2,33 ≤ %2,5); sorgu-başı sabit 1'er-UNK deseni de ilk kez bozuldu (zımpara 1, fidan 0, kontrol 1). Morfem koşulu KAÇTI (86 < 100): zımpara erken-EOS'u KISMEN KIRILDI (14→30 — 40 uzun 'nedir/nasıl' kaydının aranan etkisi, tavana ulaşmadı) ama fidan GERİLEDİ (41→11); toplam 100→86.

**Genel hüküm: 1/4 kapı GEÇTİ (yalnız KAPI-2 — tur-1 ve tur-2 ile aynı kapı).** Veri-zenginleştirme dört eksik aksa dördüne de ölçülebilir iyileşme verdi (güvenler, TENSE görünürlüğü, UNK oranı, zımpara) ama tek başına hiçbir donmuş eşiği çevirmedi. Ceket hattı 4 kapı geçilmeden AÇILMAZ; v3r3 donmuş kapılara göre ceket tabanı DEĞİL.

---

## RAPOR

### R1 — KAPI-1: masked SFT val CE (beşli aynı rig)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur3_gate_sftval.py` (val = `pure_base_sft.jsonl` son-150, `FastSampleAlignedDataset(block=128)`, batch 16, shuffle=False; çıktı aynen):

```
cihaz: mps
val kayıt: 150 (pure_base_sft.jsonl son-150, T-0209 deseni)
[SFT-VAL] TABAN (pure) (data/anka_taban_pure.pt): CE=2.304304 | PPL=10.0172 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3 (data/anka_taban_v3.pt): CE=3.478886 | PPL=32.4236 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R (tur-1) (data/anka_taban_v3r.pt): CE=2.737953 | PPL=15.4553 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R2 (tur-2) (data/anka_taban_v3r2.pt): CE=2.584789 | PPL=13.2605 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R3 (tur-3) (data/anka_taban_v3r3.pt): CE=2.482663 | PPL=11.9731 | eksik_anahtar=0 beklenmedik=0 | örnek=150
```

- Çıpalar birebir yeniden üretildi: TABAN 2,304304 (T-0209 kaydı 2,3049), V3 3,478886 (T3.5 ile birebir), V3R 2,737953 (tur-1 hüküm), V3R2 2,584789 (tur-2 hüküm) — merdiven süreklilik kanıtı; TABAN+V3 çıpaları görev gereği bu oturumda aynı rigde yeniden ölçüldü.
- V3R3 koşum içi final_val_ce (2,482663154602051) kapı rig'iyle BİREBİR — v2 val-150 kuyruğunun v1'le bayt-özdeşliği (bu oturumda ölçüldü) sayesinde koşum raporu ile bağımsız ölçüm aynı val kümesinde örtüşüyor.
- Kapanış hızı: V3→V3R −0,741 → V3R→V3R2 −0,153 → V3R2→V3R3 **−0,102** nat. V3 eşiğine kapatılan toplam 0,996 nat (=açığın %86'sı); kalan **0,162663 nat**. Mevcut trendle bir tur projeksiyonu ≈2,38 — donmuş 2,32'ye YETMEZ.
- Tüm yükleme raporları `eksik=0 beklenmedik=0` — mimari sapması yok. Aşırı-eğitim sinyali YOK (val CE indi: 2,585→2,483).
- **Hüküm: KALDI** (2,482663 > 2,32).

### R2 — KAPI-2: külliyat pencere-CE (resmi VAL bin)

Komut: `venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r3.pt --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337` (çıktı aynen):

```
[CE-VAL] model: data/anka_taban_v3r3.pt
[CE-VAL] vocab: data/rebuild/vocab_anka_r1_33114.json (33114 giriş) | val: data/kulliyat_v2/kulliyat_v2_val.bin (582,705 token)
[CE-VAL] block_size=128 | samples=500 | batch=16 | seed=1337 | device=mps
[CE-VAL] yükleme: atılan cos/sin/mask=0 | eksik=0 | beklenmedik=0
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.2230  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 25.1030
[CE-VAL] süre: 6.9 sn
```

- Çıpa teyidi (aynı rig, bu oturum): TABAN **3,8205** (PPL 45,6285), V3 **3,1936** (PPL 24,3756), V3R **3,2118**, V3R2 **3,2181** — önceki tüm kayıtlarla birebir; pencere-CE sütunu bu oturumda beşli tam.
- Kıyas: TABAN 3,8205 / V3 3,1936 / V3R 3,2118 / V3R2 3,2181 / **V3R3 3,2230**. Replay'in düz-metin ödünü V3'ye göre toplam **+0,0294 nat** (tur-1 sonrası +0,018; tur-2 +0,006; tur-3 **+0,0049**) — KL β=0,2 çapası ödünü ~0,03 nat bandında tutmayı sürdürüyor.
- Koşum içi TRAIN pencere-CE (3,2944) ile resmi VAL ölçümü (3,2230) farkı 0,071 nat — önceki turların avlu farklarıyla (0,066/0,068) aynı büyüklük, yön aynı; resmi kapı ölçümü VAL bin'dir.
- **Hüküm: GEÇTİ** (3,2230 ≤ 3,35; eşiğe pay 0,127 nat).

### R3 — KAPI-3: dil bilgisi paketi (test_model.py)

Komut: `venv/bin/python test_model.py --checkpoint data/anka_taban_v3r3.pt --vocab data/rebuild/vocab_anka_r1_33114.json`. `<OUTPUT>` sonrası top-1 öngörüleri (görev sırasıyla kök/zaman-kip/durum/çoğul):

| Görev | Girdi | V3R3 top-1 | Güven | Doğru mu? | V3R2 tur-2 | TABAN (bu oturumda aynı komutla) |
|---|---|---|---|---|---|---|
| kök morfemi bul | kitaplarda | `root` | **0,9981** | doğru | `root` 0,9964 | `root` 0,9997 |
| zaman/kip tespit | gideceğim | `root` | **0,4103** | **YANLIŞ** (beklenen `TENSE`) | `root` 0,3146 (yanlış) | `TENSE` 0,9981 |
| durum ekleri | okula | `case` | **0,9196** | doğru | `case` 0,7675 | `case` 0,9998 |
| çoğul eki | çocuklar | `PLURAL` | **0,8252** | doğru | `PLURAL` 0,4415 | `PLURAL` 0,9986 |

- Alet doğrulaması: TABAN bu oturumda aynı komutla 0,9997/0,9981/0,9998/0,9986 ve 4/4 doğru — tur-1/tur-2 ölçümleriyle birebir; V3/V3R/V3R2 güvenleri de bu oturumda yeniden ölçülüp kayıtlarla birebir çıktı (loglar: `p26_tur3_gate3_{v3,v3r,v3r2}_ref.log`).
- **Görev-2 'gideceğim' top-5 kanıtı (V3R3, bu oturum):**

```
1. root    | 0,4103 (ID 16656)   ← top-1 hâlâ 'root' — YANLIŞ
2. TENSE   | 0,0564 (ID 32156)   ← doğru etiket İLK KEZ top-5'te
3. <EOS>   | 0,0324 (ID 3)
4. <UNK>   | 0,0167 (ID 0)
5. bir     | 0,0147 (ID 40)
```

Tur-2'de `TENSE` top-5'te bile yoktu (5. sıra olasılığı 0,0162'nin altındaydı; top-5 = root/bir/<EOS>/<UNK>/bu). 194 zaman/kip kaydı TENSE ailesini görünmezlikten 2. sıraya taşıdı (0 → 0,0564) ve görev-2 güvenini 0,3146→0,4103 çıkardı — **ama top-1'i 'root'dan devirmedi**: 'root' prior'ı 7,3× daha ağır (0,4103 vs 0,0564). Veri kalemi işe yaradı, dozu top-1 çevirmeye yetmedi.
- Güven değişimleri tur-2→tur-3: görev-1 +0,0017 (0,9981, 0,99 bandında kalıcı), görev-2 +0,0957, görev-3 **+0,1521** (0,7675→0,9196), görev-4 **+0,3837** (0,4415→0,8252) — 31 görev-3/4 kaydının etkisi en net bu iki görevde; dört görevin dördü de ilerledi, güven ortalaması 0,63→0,79.
- **Hüküm: KALDI** (görev-2 top-1 yanlış; görev-3 0,9196 ve görev-4 0,8252 0,99 altında; top-1 3/4 ≠ 4/4).

### R4 — KAPI-4: üretim sağduyu sondası (donmuş 3 sorgu)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur3_sonda.py` (trllm_p26_tur2_sonda.py kalıbı; MPS, T-0157 maske, argmax + 0,8 frekans cezası, max_new=45, decompiler yüzey; sorgular tur-1/tur-2 ile BİREBİR).

| Sorgu | V3R3 morfem | V3R3 UNK | V3R2 tur-2 | TABAN (bu oturum) |
|---|---|---|---|---|
| Zımpara nedir ve nasıl kullanılır? | **30** (erken EOS, tavan değil) | 1 | 14 (erken EOS) | 45 |
| Fidan nasıl dikilir? | **11 (erken EOS)** | 0 | 41 | 45 |
| Okula gitmek için hangi ekler kullanılır? (kontrol) | 45 (tavan) | 1 | 45 (tavan) | 37 |
| **TOPLAM** | **86** | **2 (%2,33)** | 100 / 3 (%3,00) | 127 / 1 (%0,79) |

- **Zımpara erken-EOS'u KISMEN KIRILDI** (14→30): "Usta Cevabı: bu işlem, bir [?] veya daha fazla yağlı boyalarla yapılmalıydı; tutkalın üzerine yerleştirilen camdan yapılan ince kumaş" — 40 uzun 'nedir/nasıl' kaydının aranan etkisi göründü; iki turda sabit 14'lük kesilme gitti ama tavan (45) tutmadı, 30'da EOS.
- **Fidan GERİLEDİ** (41→11): "Bu işini yapmak için kullanılan bir söz" — tur-2'nin davranış kazancı tur-3'te tersine döndü; sorgu-bazlı üretim uzunluğu veri-bileşimiyle oynuyor (kararlı değil). Bu yüzden toplam 100→86 düştü ve morfem koşulu tur-2'de sınır-değer geçmişken tur-3'te KAÇTI.
- Kontrol sorgusu 45 tavanda akıcı (yetenek sürüyor): "Bu durum, ilgili uzmanların ve araştırmacılarının ( örneğin [Özel İsim]) olarak adlandırılan bir terimdir; ancak [?]-i hukuku ile ilgilenilmiş olan kişinin kendi içinde bulunduğumuzu".
- **UNK koşulu İLK KEZ karşılandı**: 2 UNK / %2,33 ≤ %2,5. Sorgu-başı sabit 1'er-UNK deseni ilk kez bozuldu (fidan 0 UNK) — tur-2'nin "UNK üretimi replay-bağışır" bulgusu tur-3 veri karışımıyla kısmen yumuşadı.
- TABAN, donmuş sorgularla bu oturumda 127/1/%0,79 — tur-1/tur-2 ölçümleriyle birebir, eşik altında rahat.
- **Hüküm: KALDI** (morfem 86 < 100; UNK %2,33 koşulu karşılandı — kapı iki koşulun birlikte gerçekleşmesini ister).

### R5 — Tuhaflıklar

1. **Sorgu-bazlı uzunluk oynaması:** fidan tur-2'de kırılıp tur-3'te geri dönerken zımpara tersini yaptı (14→30); iki turun toplamı da 100-86 bandında seyrediyor. Veri-bileşimi uzun yanıtlı alan kartları ekledi ama etki sorgular arası dağıldı — davranışsal ölçüm sorgu setine bağımlı, kararlı bir sıçrama değil.
2. **Görev-2 'root' prior'ı hâlâ baskın:** TENSE ilk kez top-5'te (0,0564) ama root 0,4103 ile 7,3× önde. Eksik şey artık temsiliyet DEĞİL, ağırlık: zaman/kip kayıt dozu artırılıp çeşitlendirilmeden (farklı fiil gövdeleri, kipler, olumsuz örnekler) top-1 çevrilmez.
3. **KAPI-1 kapanış hızı monoton yavaşlıyor:** −0,741 → −0,153 → −0,102 nat/tur; veri-zenginleştirme trendi ancak yataylaştırdı. Kalan 0,163 nat bu hızla ~2 tur demek; donmuş eşik tek turda kapanmıyor.
4. **Çıpa çakışması yok:** beş modelin SFT val CE'si, pencere-CE'si (beşli), güvenleri (beşli) bu oturumda yeniden ölçüldü ve TÜM önceki kayıtlarla birebir — rig tutarlı, merdiven karşılaştırılabilir.
5. **UNK oranı ilk kez eşik altında ama kırılgan:** %2,33 payı 0,17 puan; payda 86'ya düşerse (fidan 11) oran tekrar %2,9'a fırlar. UNK koşulunun kalıcı çözümü hâlâ üretim-hattı kelime dağarcığı kalemi.

### R6 — Merdiven toplam tablosu ve duruş hükmü

| Merdiven | masked SFT val CE | külliyat pencere-CE (VAL) | Görev-1 güven (kök) | Görev-2 güven (zaman/kip) | Görev-3 güven (durum) | Görev-4 güven (çoğul) | top-1 | Sonda morfem / UNK% |
|---|---|---|---|---|---|---|---|---|
| TABAN (pure) | 2,3043 | 3,8205 | 0,9997 | 0,9981 | 0,9998 | 0,9986 | **4/4** | 127 / %0,79 |
| V3 (saf korpus devam-ön-eğitimi) | 3,4789 | 3,1936 | 0,0512 | 0,0491 | 0,0726 | 0,1595 | 3/4 | 55 / %3,64 |
| V3R (tur-1 replay) | 2,7380 | 3,2118 | 0,9797 | 0,0884 | 0,6377 | 0,3844 | 3/4 | 74 / %4,05 |
| V3R2 (tur-2 replay) | 2,5848 | 3,2181 | 0,9964 | 0,3146 | 0,7675 | 0,4415 | 3/4 | 100 / %3,00 |
| **V3R3 (tur-3 replay + veri-zenginleştirme)** | **2,4827** | **3,2230** | **0,9981** | **0,4103** | **0,9196** | **0,8252** | **3/4** | **86 / %2,33** |

*Provenans: SFT val CE beşlisi, pencere-CE beşlisi ve güven beşlisi bu oturumda ölçüldü (V3R3 ve TABAN tüm sütunlarla; V3/V3R/V3R2 çıpa yeniden ölçümü önceki kayıtlarla birebir). Sonda: TABAN ve V3R3 bu oturumda; V3/V3R/V3R2 sonda değerleri kendi tur raporlarından. Donmuş 4 kapı ekseninden hiçbiri TABAN bandına dönmedi.*

**Duruş analizi:**

- **Veri-zenginleştirme kalemi ölçülemez kazanç verdi, eşik çevirmedi:** +265 kayıt; SFT val CE −0,102, görev-3 +0,152, görev-4 +0,384 güven, TENSE top-5'e girişi, zımpara 14→30, UNK %3,00→%2,33. Tur-2 raporu R7'de öngörülen "yeni veri kalemi" stratejisi doğruydu — etki yönü tamam.
- **Ama donmuş hedefler aynı:** KAPI-1'e kalan 0,163 nat (trend −0,10/tur ile ≥1 tur daha), KAPI-3'te görev-2 top-1 + iki güven 0,99 altı, KAPI-4'te morfem 86<100. 1/4 — üç turda da geçen tek kapı KAPI-2.
- **Ceket hattı bu noktada AÇILMAZ; v3r3 donmuş kapılara göre ceket tabanı DEĞİL.** Merdiven, protokolün "veri-bileşimi eksikliğini epokla çözme" kuralına uygun biçimde veri kalemiyle uzatıldı; kalan açıklar için sıradaki kalemler: (1) görev-2 doz/çeşitlilik artışı (top-1 çevirmek için), (2) uzun-yanıt alan kartı çeşitliliği (fidan tipi erken-EOS'u kararlı kılmak için), (3) KAPI-1'in 0,16 nat'ı — mevcut trendle ek veri/tur planı olmadan kapanmaz.

---

**Ölçüm oturumu:** 04.11.2026, MPS; tüm koşumlar salt-okuma, repo izlenen dosyalarına dokunulmadı; loglar `~/.hermes/cache/scratch/p26_tur3_gate{1,2,3,4}_*.log` (çıpa teyitleri: `p26_tur3_gate2_cipa_{taban,v3,v3r,v3r2}.log`, `p26_tur3_gate3_{taban,v3,v3r,v3r2}_ref.log`).