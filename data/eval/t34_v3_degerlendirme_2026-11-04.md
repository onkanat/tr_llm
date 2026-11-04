# T3.5 — anka_taban_v3 Değerlendirmesi: Eski Avlu Kimliği + Çifte CE Kapısı + Dil Bilgisi Paketi + Üretim Örneklemesi

**Tarih:** 04.11.2026 · **Kalıp:** İLAN→HÜKÜM→RAPOR · **Kural:** commit YOK, repo dosyalarına dokunma YOK (yalnız bu rapor + scratch probe'lar) · **Cihaz:** MPS (macOS, makine boş)

---

## İLAN

T3.4 devam-ön-eğitimi tamamlanınca üretilen `data/anka_taban_v3.pt` modelinin dört eksende ölçümlü değerlendirmesi:

1. **Eski donmuş avlunun kimliği** — planın eşik ölçütü 2,3049 hangi val `.bin`'ine aittir? Adaylar: `data/anka_a2_pretrain_val.bin` ve `data/anka_a1r_pretrain_val.bin` (herbiri 4.000.000 bayt = 2M uint16 jeton).
2. **Çifte kapı** — (a) v3 eski avluda CE < 2,3049 mü? (b) v3 yeni avluda (`data/kulliyat_v2/kulliyat_v2_val.bin`, 582.705 jeton) tabanın 3,8205'ine göre iyileşme mi?
3. **Dil bilgisi paketi** — `test_model.py` çıktısında tabanın 03.11 ölçümü 0,9998 idi; v3 kaç?
4. **Üretim kalitesi örneklemesi** — trllm_sonda deseni (MPS, T-0157 maske, argmax + 0,8 frekans cezası, max_new=45), 3 sorgu × taban/v3; hız (tok/s), morfem sayısı, UNK oranı; v3 SFT ceketi DEĞİLDİR — akışkan Türkçe beklenmez, morfolojik tutarlılık ve alan karışması tabanla karşılaştırılır.

**Ölçüm kimlikleri (komut çıktısı):**

```
eb2335b8ca1ee5969f42a96cd687b2ae48d5c53d7fb6682a76d5f326f33b020c  data/anka_taban_v3.pt
5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360  data/anka_taban_pure.pt
```
(pure SHA'sı `data/eval/t0209_taban_pure_hukum_2026-10-02.json` kaydıyla birebir.)

**Sabit ölçüm çerçevesi:** sözlük `data/rebuild/vocab_anka_r1_33114.json` (33.114 giriş, donmuş), `scripts/eval_ce_val.py --samples 500 --seed 1337` (pencere 128, batch 16, cihaz mps), `test_model.py --checkpoint/--vocab`. Tüm v3 yükleme raporları `eksik=0 beklenmedik=0` — mimari tabanla birebir.

---

## HÜKÜM

**H1 — Eski avlu: "birebir bulunamadı" (2,3049 pencere CE'si veren aday YOK); kimlik yine de kesinleşti: `anka_a1r_pretrain_val.bin`.**
- a1r adayı tabanla CE **3,4895** ölçtü — 2,3049'dan +1,1846 uzakta (±0,15 penceresinin çok dışında).
- a2 adayı **hiç ölçülemedi**: guard `RuntimeError: val maks jeton id 33910 >= vocab_size 33114` (a2 külliyatı r2 sözlüğüyle, 33.911 girişle üretilmiş; donmuş 33.114'lük sözlükle yüklenebilir bile değil).
- **Kök neden (kanıtlı):** 2,3049 sayısı hiçbir `.bin` pencere CE'si DEĞİL. T-0209 hüküm kaydında iki ayrı ölçüm var: `final_val_ce: 2.304913` (= masked SFT val CE, `pure_base_sft.jsonl` son-150 kayıt, `scripts/train_t0209_taban_pure.py:140-146` deseni) ve `final_pencere_ce: 3.4899` (pencere CE, a1r TRAIN bin'inden 25 pencereyle). Benim tekrar-üretimim: taban masked SFT val CE = **2,304304** (T-0209 kaydından Δ −0,0006) — eşik ölçütünün hangi metriğe ait olduğu böylece kanıtlandı; a1r val bin'indeki 500-pencere ölçümüm (3,4895) ise T-0209'un pencere CE'siyle (3,4899) Δ −0,0004 örtüşüyor → a1r, tabanın külliyat ailesinin donmuş val dilimidir.
- **Plan kusuru bayrağı:** T3.4'te dondırılan "val CE < 2,3049" ölçütü `eval_ce_val.py` ile HİÇBİR val bin'inde tekrar üretilemez; eşik masked SFT val metriğine aittir.

**H2 — Çifte kapı:**
- **(a) Eski avlu, plan kapısı: GEÇİLMEDİ.** v3 pencere CE (a1r) = **3,2734** → 2,3049'un altında değil (+0,9685). Kapının kendi metriğinde (masked SFT val) durum daha net: taban 2,304304 → **v3 3,478886** (+1,1746, %51 gerileme; PPL 10,02→32,42). v3 aynı-metrik kıyasda eski avluda tabana göre iyileşti (3,4895→3,2734, −0,2161) ama planın eşik ölçütünü iki metre de tutmuyor.
- **(b) Yeni avlu: İYİLEŞME (kapı geçer).** v3 CE = **3,1936** vs taban 3,8205 → **−0,6269** (%16,4); PPL 45,6285→24,3756 (**−%46,6**). (Tabanın yeni avlu değerini bu oturumda yeniden ölçtüm: CE 3,8205 / PPL 45,6285 — T3.4-pilot beyanının birebir kendisi.)

**H3 — Dil bilgisi paketi: ÇÖKTÜ.** Taban bugün 4 görevde de yüksek güven veriyor (top-1 olasılıkları 0,9997 / 0,9981 / **0,9998** / 0,9986 — 03.11'in 0,9998 ölçümüyle birebir örtüşme). v3'te aynı dört top-1: **0,0512 / 0,0491 / 0,0726 / 0,1595**; üstelik 'gideceğim' görevinde top-1 tamamen yanlış (`bir`), diğer üçünde doğru jeton kaldı ama güven %90+ eridi. H2(a)'daki SFT-val gerilemesiyle tutarlı: saf-korpus devam eğitimi SFT biçim yetkinliğini yıkmış.

**H4 — Üretim: morfoloji sağlam, uzunluk çöktü, GTS-tarzı register, alan sızıntısı azaldı.** v3 üretimleri decompiler'dan gramerik Türkçe yüzey veriyor (ör. "işini yaptıran bir söz", "kendilerine özgü olarak canlı kalmak", "şeylerini ortaya koymak için kullanılırlar") — beklenen akışkan-cevap DEĞİL ama morfolojik tutarlılık korundu. Toplam üretilen morfem 135→55 (−%59,3; erken `<EOS>`); UNK oranı %1,48→%3,64 (küçük örneklem, benzer büyüklük). Tabanın "fidan→ceviz ağacının ahşap/mobilya özellikleri" alan sızıntısı v3'te kayboldu; yerine külliyatın GTS tanım katmanının register'ı geldi ("…için kullanılırlar; çünkü"). Hız: taban 19,5–57,6 tok/s (ilk üretim ısınması dahil), **v3 57,6–62,2 tok/s** (03.11 taban+ceket aralığı 30-44 tok/s ile uyumlu düzeyde).

**Genel hüküm:** v3, düz metin modellenmesinde NET İYİLEŞME (yeni avlu −0,627 CE; eski avlu bile −0,216 CE) ama SFT/talimat biçiminde NET GERİLEME (masked SFT val +1,175 CE; dil bilgisi paketi 0,9998→0,0726). Planın 2,3049 kapısı v3'te GEÇİLMEDİ. Devam eğitiminde SFT karışımı/KL çapası eklenmedikçe (Plan-2 P2.6 yönü) bu kapı bu yoldan geçilemez; v3'ün ceket tabanı olarak kullanılması SFT-regresyonu nedeniyle önerilmez — taban (pure) o rolde korunmalı ya da v3 üstüne kısa bir SFT-replay turu atılmalı.

---

## RAPOR

### R1 — Eski avlu kimliği (Görev 1)

**Ön denetim (maksimum jeton kimliği):**

```
$ venv/bin/python -  (numpy memmap taraması)
anka_a1r_pretrain_val: len=2,000,000 token, max_id=33113, min_id=0, PAD(0) sayısı=62,493
anka_a2_pretrain_val:  len=2,000,000 token, max_id=33910, min_id=0, PAD(0) sayısı=39,108
```

**Aday 1 — a1r (taban model):**

```
$ venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_pure.pt \
    --vocab data/rebuild/vocab_anka_r1_33114.json --val data/anka_a1r_pretrain_val.bin --samples 500 --seed 1337
[CE-VAL] vocab: data/rebuild/vocab_anka_r1_33114.json (33114 giriş) | val: data/anka_a1r_pretrain_val.bin (2,000,000 token)
[CE-VAL] block_size=128 | samples=500 | batch=16 | seed=1337 | device=mps
[CE-VAL] yükleme: atılan cos/sin/mask=0 | eksik=0 | beklenmedik=0
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.4895  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 32.7692
[CE-VAL] süre: 9.2 sn
```

**Aday 2 — a2 (taban model):**

```
$ venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_pure.pt \
    --vocab data/rebuild/vocab_anka_r1_33114.json --val data/anka_a2_pretrain_val.bin --samples 500 --seed 1337
RuntimeError: DURDURULDU: val maks jeton id 33910 >= vocab_size 33114. --vocab ile bu külliyata uygun sözlüğü verin.
```
(a2, meta beyanına göre `data/rebuild/vocab_anka_r2.json` (33.911 giriş) ile üretilmiş, üretici görev T-0104; donmuş 33.114 sözlük hattına hiç uymaz.)

**2,3049'un kaynağı — tekrar-üretim probu** (`~/.hermes/cache/scratch/trllm_t35_gate_sftval.py`; T-0209 deseni: `evaluate_val_loss` + `FastSampleAlignedDataset(block=128)`, val = `pure_base_sft.jsonl` son-150, batch 16, shuffle=False):

```
$ venv/bin/python ~/.hermes/cache/scratch/trllm_t35_gate_sftval.py
cihaz: mps
val kayıt: 150 (pure_base_sft.jsonl son-150, T-0209 deseni)
[SFT-VAL] TABAN (pure) (data/anka_taban_pure.pt): CE=2.304304 | PPL=10.0172 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3 (data/anka_taban_v3.pt): CE=3.478886 | PPL=32.4236 | eksik_anahtar=0 beklenmedik=0 | örnek=150
```

T-0209 hüküm kaydı (kanıt):

```
$ cat data/eval/t0209_taban_pure_hukum_2026-10-02.json
{ "durum": "BASARILI", "sha256": "5479a2f3…", "init_val_ce": 2.3408584713935854,
  "final_val_ce": 2.304913580417633, "init_pencere_ce": 3.5, "final_pencere_ce": 3.4899,
  "opt_steps": 171, "sure_sn": 1725.5 }
```

**Sonuç tablosu:**

| Ölçüm | Değer | 2,3049'dan uzaklık | Yorum |
|---|---|---|---|
| Taban @ a1r val (pencere CE, 500 pencere) | 3,4895 | +1,1846 | pencere CE ölçeği |
| Taban @ a2 val | ölçülemez | — | guard: r2 külliyat, donmuş sözlüğe uymaz |
| Taban @ masked SFT val (tekrar-üretim) | **2,304304** | −0,0006 | **eşik ölçütü budur** (T-0209: 2,304913) |
| Taban pencere CE (T-0209 kaydı, a1r train bin) | 3,4899 | — | benim a1r val ölçümümle Δ −0,0004 → aynı külliyat ailesi |

**Hüküm R1:** ≈2,3049 pencere CE'si veren aday yok → *"eski avlu birebir bulunamadı"*. Eşik sayısının masked SFT val CE olduğu, tabanla 2,304304 (Δ −0,0006) tekrar-üretimiyle kanıtlandı. Eski donmuş avlunun kimliği buna rağmen kesindir: **`anka_a1r_pretrain_val.bin`** — donmuş sözlükle yüklenen tek aday, T-0209 eğitim hattının kullandığı `anka_a1r_pretrain.bin` ile aynı aile, pencere CE'si T-0209 kaydıyla Δ 0,0004 örtüşme. a2, r2-sozlük hattının (T-0104) kalıntısıdır ve bu model hattına ait değildir.

### R2 — Çifte kapı (Görev 2)

**Kapı (a) — v3 eski avluda (a1r):**

```
$ venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3.pt \
    --vocab data/rebuild/vocab_anka_r1_33114.json --val data/anka_a1r_pretrain_val.bin --samples 500 --seed 1337
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.2734  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 26.3997
[CE-VAL] süre: 7.0 sn
```

**Kapı (b) — v3 yeni avluda (kulliyat_v2 val, 582.705 jeton):**

```
$ venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3.pt \
    --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.1936  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 24.3756
[CE-VAL] süre: 6.3 sn
```

**Tabanın yeni avluda yeniden ölçümü (3,8205 beyanının teyidi):**

```
$ venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_pure.pt \
    --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.8205  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 45.6285
[CE-VAL] süre: 8.5 sn
```

**Kapı tablosu (pencere CE, --samples 500 --seed 1337):**

| Avlu | Taban (pure) CE | v3 CE | Δ CE | Δ PPL | Hüküm |
|---|---|---|---|---|---|
| Eski (a1r) | 3,4895 (PPL 32,77) | 3,2734 (PPL 26,40) | **−0,2161** (−%6,2) | −%19,4 | eşit-metrikte iyileşme; ama plan kapısı 2,3049'a göre **GEÇİLMEDİ** (+0,9685) |
| Yeni (kulliyat_v2) | 3,8205 (PPL 45,63) | 3.1936 (PPL 24,38) | **−0,6269** (−%16,4) | −%46,6 | **İYİLEŞME — kapı geçer** |

**Kapının kendi metriğiyle (masked SFT val, T-0209 deseni):**

| Model | CE | PPL | 2,3049 eşiğine göre |
|---|---|---|---|
| Taban (pure) | 2,304304 (tekrar-üretim; T-0209: 2,304913) | 10,02 | eşik = tabanın kendisi |
| **v3** | **3,478886** | 32,42 | **+1,1746 GERİLEME — kapı geçilmedi** |

Not: pencere CE ölçümlerinde PAD-hedef uyarısı hiçbir koşumda çıkmadı (seçilen 500 pencerede PAD hedefi yok). Yeni avlada %2 train sızıntısı bilinçli ve beyanlıdır (`kulliyat_v2.bin.meta.json`: "katmanli %2, train'de %2 sizinti bilinçli") — mutlak değer olarak sızıntı içerir, taban-v3 kıyas aynı avlu üzerinde yapıldığı için hüküm geçerlidir.

### R3 — Dil bilgisi paketi (Görev 3)

```
$ venv/bin/python test_model.py --checkpoint data/anka_taban_v3.pt --vocab data/rebuild/vocab_anka_r1_33114.json
$ venv/bin/python test_model.py --checkpoint data/anka_taban_pure.pt --vocab data/rebuild/vocab_anka_r1_33114.json   (karşılaştırma koşumu)
```

`<OUTPUT>` sonrası 1. tahmin olasılıkları (SFT dil bilgisi görevleri; tam loglar scratch'te: `t35_test_model_v3.log`, `t35_test_model_pure.log`):

| SFT görevi | Taban top-1 (olasılık) | v3 top-1 (olasılık) |
|---|---|---|
| "kök morfemini bul" (kitaplarda) | `root` 0,9997 | `root` **0,0512** |
| "zamanını/kipini tespit et" (gideceğim) | `TENSE` 0,9981 | **`bir` 0,0491 — yanlış jeton** |
| "hâl eklerini tespit et" (okula) | `case` **0,9998** | `case` **0,0726** |
| "çoğul eki tespit et" (çocuklar) | `PLURAL` 0,9986 | `PLURAL` 0,1595 |

Tabanın bugünkü koşumu 03.11 ölçümüyle (0,9998; sentez raporu "✓ yüksek güven") birebir örtüşüyor. v3'te doğru jeton 4 görevden 3'ünde hâlâ top-1 ama güven %90+ erimiş, 1'inde top-1 tamamen yanlış. İkinci-adım tahminlerinde de benzer erime: taban `:` 0,9999-1,0000 verirken v3 `:` 0,1676-0,6913 düzeyinde. Orta-cümle tahminleri: `<EOS>` sonrası iki model de `<BOS>` 1,0000 (belge-sınır davranışı korunmuş); cümle içi top-1 olasılıkları tabanda 0,58-0,72, v3'te 0,19-0,23 (olay dağılımı düz metne yayılmış — pencere CE iyileşmesiyle tutarlı).

### R4 — Üretim kalitesi örneklemesi (Görev 4)

Probe: `~/.hermes/cache/scratch/trllm_sonda_v3.py` (orijinal `trllm_sonda.py` kopyası; değişiklikler: modeller TABAN+V3, 3. sorgu nötr kontrol). MPS, T-0157 maske, argmax + 0,8 frekans cezası, max_new=45, decompiler ile yüzey üretimi. Tam log: `~/.hermes/cache/scratch/t35_sonda_v3.log`.

**TABAN (pure):**

| Sorgu | tok/s | Morfem | UNK | Yüzey (decompiler) |
|---|---|---|---|---|
| Zımpara nedir, ne işe yarar? | 19,5 | 45 | 1 | "Teknik rapor: bu işlem için [?]' kullanılmalıydı veya ahşabın doğal yağlarda eritilerek bez ponponla dairesel harekete geçtikçe ardar; tutkal temizlenir ve sertleşen kuruyunca koparmak zorunludur" |
| Fidan nasıl dikilir? | 57,6 | 45 | 0 | "Ceviz ağacı: zengin damar desenine ve asil kahverengi tonlarda, sert kokulu bir ağaçtır; sandık içlerinde ise yüzde 1-5 oranını milimetrenin altından görünen lüks mobilya tür" |
| Bir gün önce yaptığın işi kısaca anlat. | 43,2 | 45 | 1 | "Bu belge, [?]'in kendilerini göstermektedir; ancak her şey yolculuktan sonra gelenlere kadar uzanan ve daha sonraki dönem insan olarak tanımlanırken, aynı zamanda bir arayı" |

**TABAN V3:**

| Sorgu | tok/s | Morfem | UNK | Yüzey (decompiler) |
|---|---|---|---|---|
| Zımpara nedir, ne işe yarar? | 57,6 | 12 | 0 | "Usta Cevabı: [Özel İsim]'mek işini yaptıran bir söz" |
| Fidan nasıl dikilir? | 62,2 | 15 | 1 | "[?], bir şeyin kendilerine özgü olarak canlı kalmak" |
| Bir gün önce yaptığın işi kısaca anlat. | 60,0 | 28 | 1 | "Bunun üzerine [Özel İsim], [?] ve o zamanlardaki gibi görünen şeylerini ortaya koymak için kullanılırlar; çünkü" |

**Sayısal özet:** taban 135 morfem / 2 UNK (%1,48), üçünde de 45-jeton tavana takıldı (EOS yok); v3 55 morfem / 2 UNK (%3,64), erken `<EOS>` ile bitirdi (−%59,3 uzunluk). Hız v3'te 57,6-62,2 tok/s (taban 19,5 ilk-ısınma dahil 19,5-57,6; 03.11 aralığı 30-44 tok/s).

**Yorumlar:**
- **Morfolojik tutarlılık: KORUNDU.** v3 çıktıları derleyici-gramerin geçerli morfem zincirleri ("yaptır+PART_An", "özgü ol+GERUND_ArAk", "kullanıl+TENSE_AORIST+PERSON_3PL") ve decompiler yüzeyleri eksiksiz Türkçe ("işini yaptıran bir söz", "şeylerini ortaya koymak için kullanılırlar").
- **Alan sızıntısı değişimi:** tabanda 03.11'de görülen "fidan→ceviz ağacının ahşap/mobilya bilgisi" sızıntısı v3'te kayboldu — v3 alan-sözlüğü enjeksiyonu yerine genel/nötr üretim veriyor. Buna karşılık yeni bir **register kayması** var: külliyatın GTS tanım katmanının (1,44M jeton) üslubu ("…bir söz", "…için kullanılırlar") v3 üretimlerine belirgin biçimde sızmış.
- **Uzunluk çöküşü + SFT güven erimesi (R3) birlikte okunduğunda:** v3 talimat-biçiminde koşullanmayı yitirmiş durumda; bu, T3.4 koşumunun saf-korpus (SFT karışımı ve KL çapası olmadan) yapılmasının doğrudan sonucudur.

### R5 — Tuhaflıklar / bayraklar

1. **Eşik-metrik uyumsuzluğu (plan kusuru):** T3.4'te "val CE < 2,3049" olarak dondurulan ölçüt, `eval_ce_val.py` pencere CE'siyle ölçülebilir bir sayı değil; masked SFT val CE'sidir. T3.5'ten sonra bu eşik ya (i) SFT-val probuyla (`trllm_t35_gate_sftval.py` scratch'te duruyor) denetlenmeli, ya da (ii) pencere CE ölçeğinde tabanın kendi ölçümleriyle (eski avlu 3,4895 / yeni avlu 3,8205) yeniden dondurulmalı.
2. **a2 adayının hat dışı kalıntı olduğu:** `anka_a2_pretrain_val.bin` r2-sözlük (33.911) hattından; donmuş 33.114 sözlüğüyle yüklenemez. Bu dosya taban hattının avlusu olarak elenir (guard çıktısı kanıttır).
3. **T3.4 son koşumunun eğitim günlüğü repoda bulunamadı:** yalnız 50-adımlık pilotun günlüğü var (`data/kulliyat_v2/pilot_t34_run.log`: başlangıç 3,5928; son-60 ortalama 3,7409 ± 0,2142, pilot çıktısı `pilot_t34.pt`). Ana ajanın beyanı "kayıp 3,59→3,31" bu raporda ölçüm olarak DEĞİL, beyan olarak geçer; bu değerlendirmedeki her sayı kendi komut çıktısından gelir.
4. **v3'ün SFT regresyonu kapıyı iki koldan da kapatıyor:** pencere CE iyileşmesine rağmen (eski −0,216, yeni −0,627) talimat-biçimi metriği +1,175 geriledi. Devam-ön-eğitim turuna %10-20 SFT-replay karışımı (Plan-2 P2.6) eklenmeden 2,3049 kapısı bu yoldan geçilmez görünüyor.
5. Küçük not: tabanın ilk sonda üretimi 19,5 tok/s ile ölçüldü (MPS ısınması); sonraki üretimler 43-58 tok/s. Karşılaştırmalarda sonraki üretimler esas alınmalı.

### Ölçüm envanteri (kanıt dosyaları, scratch)

| Kanıt | Dosya |
|---|---|
| Eval çıktıları (5 koşum) | `~/.hermes/cache/scratch/t35_eval_pure_a1r.log`, `t35_eval_pure_a2.log`, `t35_eval_v3_a1r.log`, `t35_eval_v3_kuldiy.log`, `t35_eval_pure_kulliyat.log` |
| Kapı tekrar-üretim probu | `~/.hermes/cache/scratch/trllm_t35_gate_sftval.py` + `t35_gate_sftval.log` |
| test_model çıktıları | `~/.hermes/cache/scratch/t35_test_model_pure.log`, `t35_test_model_v3.log` |
| Sonda probe + çıktı | `~/.hermes/cache/scratch/trllm_sonda_v3.py` + `t35_sonda_v3.log` |

**Kabul çizgisi hüküm özeti:** Eski avlu kimliği a1r (kanıtlı; 2,3049 = masked SFT val CE, tekrar-üretim Δ −0,0006) · Kapı (a) GEÇİLMEDİ (v3 pencere CE 3,2734 > 2,3049; kapının kendi metriğinde 3,4789 vs 2,3043 = +1,1746 gerileme) · Kapı (b) GEÇTİ (v3 3,1936 vs 3,8205, −0,6269; PPL −%46,6) · Dil bilgisi paketi 0,9998 → 0,0726 (hâl-eki testi; güven çöküşü) · Üretim: morfoloji sağlam, uzunluk −%59, GTS register kayması, alan sızıntısı azaldı, 57,6-62,2 tok/s.