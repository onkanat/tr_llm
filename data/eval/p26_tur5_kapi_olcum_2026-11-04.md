# P2.6 Tur-5 — anka_taban_v3r5 Donmuş 4 Kapı Ölçümü (Uzun-Yanıt + Format-Çeşitliliği Kartı)

**Tarih:** 04.11.2026 · **Kalıp:** İLAN→HÜKÜM→RAPOR · **Kural:** commit YOK; izlenen eski dosyalara YAZMA YOK (yeni: `pure_base_sft_v4.jsonl` + `anka_taban_v3r5.pt`); kapılar koşumdan önce donduruldu ([İLAN](p26_tur5_ilan_2026-11-04.md)) · **Cihaz:** MPS (macOS)
**Önceki turlar:** [Tur-4](p26_tur4_kapi_olcum_2026-11-04.md) (1/4) · [Tur-3](p26_tur3_kapi_olcum_2026-11-04.md) (1/4) · [Tur-2](p26_tur2_kapi_olcum_2026-11-04.md) (1/4) · [Tur-1](p26_kapı_olcum_2026-11-04.md) (1/4)

---

## İLAN (özet — tam metin: `p26_tur5_ilan_2026-11-04.md`, koşumdan önce donduruldu)

Tur-5'in veri kalemi **uzun-yanıt alan kartı + format çeşitliliği**: `pure_base_sft_v4.jsonl` = v3 (5.209) + **526 yeni kayıt** (montaj: v3 kafası + yeniler ORTADA tohum-42 + v3 kuyruğu; **val-150 BAYT-ÖZDEŞ**).

| Kalem | Kayıt | Ölçüm |
|---|---|---|
| (a) uzun-yanıt: nedir+nasıl 148 · prosedür 71 · karşılaştırma 63 · neden-önemli 62 | **344** | çıktı morfem **81-140, ort 105,8** (her kayıt kristal tokenizer ile ölçüldü) |
| (b) format: olumsuz 46 · çok-özlü tanım 45 · parafraz 46 · kısa-nedir 45 | **182** | — |
| **toplam yeni** | **526** | **UNK 0/52.722 token (%0,00)**; saflık 0 ihlal; çakışma 0 |

- Konu çeşitliliği: yeni uzun blok ~340 nötr konu = mevcut 40'lık bloğun **≈8,5×'ı** (5-8× hedefi karşılandı); mevcut bloğun 3 çakışan konusu (Ayna/Su/Sözlük) aday havuzundan çıkarıldı.
- Val zinciri: v1=v2=v3=v4 son-150 SHA aynı (`89c9b3e714786fd4`), bağımsız `cmp` çapraz-teyidi EXIT=0 — KAPI-1'in donmuş val kümesi değişmedi.
- Koşum: `train_p26_sft_replay.py` değiştirilmeden; `--pencere-n 1396` (SFT train 5.585/4; karışım %20,00), KL β=0,2, LR 5e-5→1e-5, warmup 20, batch 16 × accum 2, 1 epok, tohum 42, save-every 50 → `data/anka_taban_v3r5.pt`.
- Kapılar tur-1/2/3/4 ile eşikleri BİREBİR aynı dörtlü; hükümden önce dördü de ölçüldü.

## HÜKÜM

**KAPI-1 — masked SFT val CE ≤ 2,32: KALDI.** V3R5 = **2,427070** (PPL 11,3257). Eşiğe açık **+0,107070 nat** (tur-4 açığı 0,119'dan daraldı: −0,0120 nat/tur). Koşum içi final (2,427070) bağımsız kapı rig'iyle 6 basamakta aynı. Kapanış hızı: −0,741 → −0,153 → −0,102 → −0,044 → **−0,012** nat/tur — doygunluk DERİNLEŞTİ; kalan 0,107 nat bu hızla ~9 tur eder (tek veri kalemiyle pratikte kapanmaz).

**KAPI-2 — külliyat pencere-CE ≤ 3,35: GEÇTİ.** V3R5 = **3,2304** (PPL 25,2905; VAL bin, 500 pencere, tohum 1337). Eşiğe pay **0,120 nat**. Tur-4'ye ödün +0,0040 nat; V3'e toplam ödün +0,0368 nat — KL β=0,2 çapası pencere-CE'yi V3 bandında tutmayı sürdürdü (ödün/tur: +0,018/+0,006/+0,0049/+0,0034/**+0,0040**); TABAN'a iyileşme −0,5901 nat. **Beş turda da geçen tek kapı.**

**KAPI-3 — paket ≥ 0,99 VE 4/4 top-1: KALDI.** V3R5 dört görev güveni (test_model.py sırasıyla kök/zaman-kip/durum/çoğul): **0,9981 / 0,9868 / 0,9848 / 0,8227**; top-1 **4/4 doğru — KORUNDU**. **'gideceğim'→TENSE top-1 KORUNDU ve GÜÇLENDİ: 0,9325 → 0,9868** (tur-4'te çevrilen eksen kaybolmadı; uzun-yanıt kartı görev eksenini bozmadı). AMA güven eşiği yine dönmedi: görev-2 0,9868, görev-3 0,9848 ve görev-4 0,8227 0,99 altında; görev-4 tur-4'ten GERİLEDİ (0,9218 → 0,8227).

**KAPI-4 — toplam morfem ≥ 100 VE UNK ≤ %2,5: KALDI.** V3R5 3 sorguda toplam **58 morfem** (31/11/16) ve **2 UNK (%3,45)** — iki koşul da kaçtı. Tur-4'ten GERİLEDİ (72/2/%2,78 → 58/2/%3,45; payda düşünce oran fırladı — tur-3 R5.5 kırılganlığı sürüyor). **Kontrol sorgusu 45 bandına DÖNMEDİ: 24 → 16** — İLAN'daki uzun-yanıt hipotezinin ana göstergesi TERS yönde ölçüldü. Zımpara 37→31, fidan 11'de sabit. TABAN çıpası bu oturumda birebir: 127 (45/45/37) / 1 / %0,79.

**Genel hüküm: 1/4 kapı GEÇTİ (yalnız KAPI-2 — beş turda da geçen tek kapı).** Uzun-yanıt kartı görev eksenini korudu ve TENSE güvenini yükseltti ama üretim-uzunluğu eksenini ÇEVİREMEDİ; iki eksen yine aynı anda dönmedi. Ceket hattı AÇILMAZ; v3r5 donmuş kapılara göre ceket tabanı DEĞİL.

---

## RAPOR

### R1 — KAPI-1: masked SFT val CE (yedili aynı rig)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur5_gate_sftval.py` (val = `pure_base_sft.jsonl` son-150 = v4 son-150, `FastSampleAlignedDataset(block=128)`, batch 16, shuffle=False; çıktı aynen):

```
cihaz: mps
val kayıt: 150 (pure_base_sft.jsonl son-150, T-0209 deseni)
[SFT-VAL] TABAN (pure) (data/anka_taban_pure.pt): CE=2.304304 | PPL=10.0172 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3 (data/anka_taban_v3.pt): CE=3.478886 | PPL=32.4236 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R (tur-1) (data/anka_taban_v3r.pt): CE=2.737953 | PPL=15.4553 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R2 (tur-2) (data/anka_taban_v3r2.pt): CE=2.584789 | PPL=13.2605 | eksik_anahtar=0 | örnek=150
[SFT-VAL] V3R3 (tur-3) (data/anka_taban_v3r3.pt): CE=2.482663 | PPL=11.9731 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R4 (tur-4) (data/anka_taban_v3r4.pt): CE=2.439044 | PPL=11.4621 | eksik_anahtar=0 beklenmedik=0 | örnek=150
[SFT-VAL] V3R5 (tur-5) (data/anka_taban_v3r5.pt): CE=2.427070 | PPL=11.3257 | eksik_anahtar=0 beklenmedik=0 | örnek=150
```

- Çıpalar birebir yeniden üretildi: TABAN 2,304304 (T-0209 kaydı 2,3049), V3 3,478886, V3R 2,737953, V3R2 2,584789, V3R3 2,482663, V3R4 2,439044 — TÜM önceki kayıtlarla birebir; merdiven süreklilik kanıtı tamam.
- Koşum içi init_val_ce (2,439044) = tur-4 final = V3R4 çıpası; v4'ün val-150 özdeşliği sayesinde tur-5 koşumu tam olarak tur-4'ün kaldığı val kümesinden başladı ve bitirdi (final 2,427070 — rig ile örtüşüyor).
- **Hüküm: KALDI** (2,4271 > 2,32; kalan 0,107 nat; hız −0,012/tur ile doygun).

### R2 — KAPI-2: külliyat pencere-CE (resmi VAL bin)

Komut: `venv/bin/python scripts/eval_ce_val.py --model data/anka_taban_v3r5.pt --vocab data/rebuild/vocab_anka_r1_33114.json --val data/kulliyat_v2/kulliyat_v2_val.bin --samples 500 --seed 1337` (çıktı aynen):

```
[CE-VAL] model: data/anka_taban_v3r5.pt
[CE-VAL] vocab: data/rebuild/vocab_anka_r1_33114.json (33114 giriş) | val: data/kulliyat_v2/kulliyat_v2_val.bin (582,705 token)
[CE-VAL] block_size=128 | samples=500 | batch=16 | seed=1337 | device=mps
[CE-VAL] yükleme: atılan cos/sin/mask=0 | eksik=0 | beklenmedik=0
[CE-VAL] ORTALAMA CROSS-ENTROPY: 3.2304  (500 pencere × 128 hedef = 64,000 konum)
[CE-VAL] PERPLEXITY (PPL): 25.2905
[CE-VAL] süre: 5.7 sn
```

- Kıyas: TABAN 3,8205 / V3 3,1936 / V3R 3,2118 / V3R2 3,2181 / V3R3 3,2230 / V3R4 3,2264 (tur-4 oturumunda aynı rigle ölçülmüş çıpalar) / **V3R5 3,2304**. Ödün/tur ivmesi ~0,004 nat bandında yatay — çapa çalışıyor.
- Koşum içi TRAIN pencere-CE (3,3032) ile resmi VAL ölçümü (3,2304) farkı 0,073 nat — önceki turların avlo farklarıyla (0,066-0,071) aynı büyüklük.
- **Hüküm: GEÇTİ** (3,2304 ≤ 3,35; pay 0,120 nat).

### R3 — KAPI-3: dil bilgisi paketi (test_model.py)

Komut: `venv/bin/python test_model.py --checkpoint data/anka_taban_v3r5.pt --vocab data/rebuild/vocab_anka_r1_33114.json`. `<OUTPUT>` sonrası top-1 öngörüleri:

| Görev | Girdi | V3R5 top-1 | Güven | Doğru mu? | V3R4 tur-4 |
|---|---|---|---|---|---|
| kök morfemi bul | kitaplarda | `root` | **0,9981** | doğru | `root` 0,9988 |
| zaman/kip tespit | gideceğim | **`TENSE`** | **0,9868** | **DOĞRU — top-1 KORUNDU** | `TENSE` 0,9325 |
| durum ekleri | okula | `case` | **0,9848** | doğru | `case` 0,9807 |
| çoğul eki | çocuklar | `PLURAL` | **0,8227** | doğru | `PLURAL` 0,9218 |

- 'gideceğim' top-5 (V3R5): `TENSE` 0,9868 → `root` 0,0038 → `<EOS>` 0,0004 → `<UNK>` 0,0003 → `case` 0,0003. Tur-4'ün çevirdiği eksen bir kart daha sonra da DEVŞİRİLMEDİ: TENSE marjı 0,9325 → 0,9868 (+0,0543); sahte-kral 'root' 0,0289 → 0,0038. Tur-4→tur-5 güven değişimleri: görev-1 −0,0007, görev-2 **+0,0543**, görev-3 +0,0041, görev-4 **−0,0991**.
- **Hüküm: KALDI** — top-1 4/4 ve TENSE korunumu (İLAN koşulu) tamam; ama üç güven 0,99 altında (0,9868/0,9848/0,8227), görev-4 tur-4'ten geriledi.

### R4 — KAPI-4: üretim sağduyu sondası (donmuş 3 sorgu)

Komut: `venv/bin/python ~/.hermes/cache/scratch/trllm_p26_tur5_sonda.py` (tur-4 kalıbı; MPS, T-0157 maske, argmax + 0,8 frekans cezası, max_new=45, decompiler yüzey; sorgular tur-1/2/3/4 ile BİREBİR).

| Sorgu | V3R5 morfem | V3R5 UNK | V3R4 tur-4 | V3R3 tur-3 | TABAN (bu oturum) |
|---|---|---|---|---|---|
| Zımpara nedir ve nasıl kullanılır? | **31 (erken EOS)** | 1 | 37 (erken EOS) | 30 (erken EOS) | 45 |
| Fidan nasıl dikilir? | **11 (erken EOS)** | 0 | 11 (erken EOS) | 11 (erken EOS) | 45 |
| Okula gitmek için hangi ekler kullanılır? (kontrol) | **16 (erken EOS)** | 1 | 24 (erken EOS) | 45 (tavan) | 37 |
| **TOPLAM** | **58** | **2 (%3,45)** | 72 / 2 (%2,78) | 86 / 2 (%2,33) | 127 / 1 (%0,79) |

- **Kontrol 24 → 16: uzun-yanıt hipotezinin ana göstergesi TERS ölçüldü** — 8,6× uzun-yanıt dozuna (40→344 kayıt) rağmen kontrol sorgusu 45 bandına dönmek bir yana daha da kısaldı.
- **Fidan 11'de üç turdur sabit** ("Bu işini yapmak için kullanılan bir söz") — dozdan etkilenmeyen kalıcı erken-EOS adası.
- Zımpara 31: "Verniklerin yüzeye yapışarak temizlenmeyi sağlayan bir yöntem oldukları için, bu işlemcinin [?] ile yapılmış olur" — 1 UNK; yüzey tur-4'ünkine yakın ama daha kısa.
- UNK sayısı üç turdur sabit (2); oran oynatan payda: 86→72→**58** (tur-3 R5.5 öngörüsü üçüncü kez doğrulandı).
- TABAN, donmuş sorgularla bu oturumda 127/1/%0,79 — tur-1/2/3/4 ölçümleriyle birebir (alet doğrulaması).
- **Hüküm: KALDI** (morfem 58 < 100 VE UNK %3,45 > %2,5 — iki koşul birden kaçtı).

### R5 — Tuhaflıklar

1. **Uzun-yanıt hipotezinin güçlü biçimi ÇÜRÜTÜLDÜ:** "uzun-yanıt kayıtları AZ kaldığı için erken-EOS geri döner" öngörüsü 40→344 kayıt dozuna rağmen donmuş sorgularda ters döndü (kontrol 24→16; toplam 72→58). Erken-EOS, bu kartın ulaşabildiği değişkenden (yanıt-uzunluğu dozu) bağımsız görünüyor. En olası bağ: sonda sorgularının TALİMAT önekleri (`Marangozluk sorusunu yanıtla.` / `Bahçıvanlık sorusunu yanıtla.` / `Soruyu kısaca yanıtla.`) verinin talimat dağılımında YOK (bilinçli saflık kararı — probe bulaşması önlenir) ve uzun-yanıt davranışı bu OOD talimat bağlamına TRANSFER OLMAmaktadır. Erken-EOS'un talimat-bağlamlı prior'ı, veri dozuyla değil ancak (a) talimat ailelerini kapsayan uzun-yanıt kayıtlarıyla (probe dizgisine bulaşmadan) ya da (b) EOS prior'ına doğrudan müdahaleyle (reçete dışı) çevrilir.
2. **İki-eksen takası yön DEĞİŞTİRDİ:** tur-4'te görev ekseninin kazancı üretim ekseninden ödünle geldi (kontrol 45→24); tur-5'te görev eksenini KORUDU (TENSE +0,054, 4/4, KAPI-1 −0,012) ama üretim yine geriledi (24→16). Yani takas simetrik değil: veri kartları görev ekseninde birikimli iyileşme üretiyor; üretim-uzunluğu ekseninde hiçbir kart (zaman/kip dozu, uzun-yanıt kartı) kalıcı kazanç üretmedi (tur-1'den tur-5'e sonda: 74→100→86→72→58 — MONOTON DÜŞÜŞ).
3. **KAPI-1 kapanış hızı çöktü:** −0,741 → −0,153 → −0,102 → −0,044 → −0,012 nat/tur. Uzun kayıtlar daha çok maskeli konum ürettiği hâlde kazanç tur-4'ten de küçüldü; replay'in SFT val CE'si veri kalemlerine karşı doygun — kalan 0,107 nat eşik-kapanması için epok/ön-eğitim bileşimi kalemi gerekir (İLAN öngörüsü 3 doğrulandı).
4. **Güven ekseninde karışık sinyal:** 'gideceğim' 0,9868 ile şimdiye kadarki en iyi TENSE güveni (TABAN 0,9981'e açık 0,0113) ve görev-3 0,9848; buna karşılık görev-4 0,9218→0,8227 geriledi. Uzun-yanıt kartı kısa-yanıt görevlerinin güvenini seyreltebilir (dikkat payı uzun üretimlere kaydı) — bir sonraki kart için kısa-format dozunun (b kaleminin) artırılması doğal aday.
5. **Çıpa çakışması yok:** 7'li SFT val CE, V3R5 pencere-CE, TABAN sondası bu oturumda yeniden ölçüldü; tüm çıpalar önceki kayıtlarla birebir — rig beş turda da tutarlı, merdiven tamamen karşılaştırılabilir.

### R6 — Merdiven toplam tablosu ve duruş hükmü

| Merdiven | masked SFT val CE | külliyat pencere-CE (VAL) | Görev-1 güven (kök) | Görev-2 güven (zaman/kip) | Görev-3 güven (durum) | Görev-4 güven (çoğul) | top-1 | Sonda morfem / UNK% (zımpara/fidan/kontrol) |
|---|---|---|---|---|---|---|---|---|
| TABAN (pure) | 2,3043 | 3,8205 | 0,9997 | 0,9981 | 0,9998 | 0,9986 | **4/4** | 127 / %0,79 (45/45/37) |
| V3 (saf korpus devam-ön-eğitimi) | 3,4789 | 3,1936 | 0,0512 | 0,0491 | 0,0726 | 0,1595 | 3/4 | 55 / %3,64 |
| V3R (tur-1 replay) | 2,7380 | 3,2118 | 0,9797 | 0,0884 | 0,6377 | 0,3844 | 3/4 | 74 / %4,05 |
| V3R2 (tur-2 replay) | 2,5848 | 3,2181 | 0,9964 | 0,3146 | 0,7675 | 0,4415 | 3/4 | 100 / %3,00 |
| V3R3 (tur-3 replay + veri-zenginleştirme) | 2,4827 | 3,2230 | 0,9981 | 0,4103 | 0,9196 | 0,8252 | 3/4 | 86 / %2,33 (30/11/45) |
| V3R4 (tur-4 + 3× zaman/kip dozu) | 2,4390 | 3,2264 | 0,9988 | 0,9325 | 0,9807 | 0,9218 | 4/4 | 72 / %2,78 (37/11/24) |
| **V3R5 (tur-5 + uzun-yanıt + format kartı)** | **2,4271** | **3,2304** | **0,9981** | **0,9868** | **0,9848** | **0,8227** | **4/4** | **58 / %3,45 (31/11/16)** |

*Provenans: 7'li SFT val CE, V3R5 pencere-CE, V3R5 güvenleri ve TABAN sondası bu oturumda ölçüldü; TABAN/V3/V3R/V3R2/V3R3/V3R4 çıpaları önceki oturum kayıtlarıyla birebir çıktı (R1 kanıtı). V3/V3R/V3R2/V3R3/V3R4 pencere-CE ve güven değerleri kendi tur oturumlarının kayıtlarından; sonda üçlüleri aynı şekilde tur raporlarından.*

**Duruş analizi — iki eksen aynı anda döndü mü? HAYIR.**

- **Görev/CE ekseni (KAPI-1/3):** tur-5 kartı eksenin tamamını KORUDU ve iki yerde ilerletti: 'gideceğim'→TENSE 0,9325→**0,9868** (merdivenin en yüksek TENSE güveni; TABAN'a 0,0113 açık) ve KAPI-1 açığı 0,119→0,107 nat. Top-1 4/4 ikinci kez tamam.
- **Üretim-uzunluğu ekseni (KAPI-4):** İLAN'ın ana öngörüsü — kontrol sorgusunun 45 bandına dönüşü — gerçekleşMEDİ (24→16; toplam 72→58; UNK %3,45). Sonda merdiveni beş turda monoton düştü: 74→100→86→72→58.
- **Bilimsel sonuç:** uzun-yanıt dozu (8,6×) erken-EOS'un gerekçesi DEĞİLDİ; tur-4'ün "takas" bulgusu tur-5'te "koruma + ilerleme, ama üretim yine düşüş" biçimine dönüştü. Donmuş sorguların talimat önekleri veri dağılımının dışında (bilinçli saflık) ve uzun-yanıt davranışı oraya taşınmıyor. Erken-EOS prior'ı artık TALİMAT-BAĞLAMLI bir hedef olarak anlaşılmalı: sıradaki kalemler (1) talimat-aile kapsamı genişletilmiş uzun-yanıt kartı (probe dizgisi bulaşmadan, örn. farklı uğraş talimatlarıyla), (2) KAPI-1'in 0,107 nat'ı için epok/ön-eğitim bileşimi kalemi, (3) güven eşiği için kısa-format dozunun artırılması (görev-4'ün 0,8227'ye gerilediği yer — b kaleminin ikinci görevi). Ceket hattı bu noktada AÇILMAZ; **v3r5 donmuş kapılara göre ceket tabanı DEĞİL.**

---

**Ölçüm kimlikleri (sha256, bu oturum):**

```
62c082fd4c843324cddf67748f656b357379c0df4bb29d1165a2597c0cbecb54  data/anka_taban_v3r5.pt  (tur-5 koşum kaydıyla birebir: p26_tur5_hukum cikti_sha256)
f90cf88b9f79037571c055dadc4f6322ee5bd2ead6a64796b32b4e3172368369  data/pedagogy/pure_base_sft_v4.jsonl (tur-5 eğitim verisi, 5.735 satır; İLAN kaydıyla birebir)
2cb4735530ae54037cf44db3b0a12365ed13acf5d21532655a443a4d549d778d  data/anka_taban_v3r4.pt  (init; tur-4 hüküm kaydıyla birebir — koşum PROVENANS satırıyla teyit)
eb2335b8ca1ee5969f42a96cd687b2ae48d5c53d7fb6682a76d5f326f33b020c  data/anka_taban_v3.pt    (öğretmen; koşum PROVENANS satırıyla teyit)
f0834dee8175e3535d1c47d65f071d40cae1e1d029fa39948d05599d72dfb0df  data/pedagogy/pure_base_sft_v3.jsonl (dokunulmadı)
```

**Koşum künyesi** (`p26_tur5_hukum_2026-11-04.json`): init_val_ce 2,439044 → final_val_ce **2,427070**; pencere-CE (TRAIN bin) 3,2978 → 3,3032; öğretmen V3, KL β=0,2, 218 opt-adım, 445,4 sn; KL dalı 427/437 mikro-partide ateşlendi, KL ort 0,1044; son-20 adım 2,149±0,274 sn; SFT train 5.585 · pencere 1.396 (%20,00) · val 150.

**Ölçüm oturumu:** 04.11.2026, MPS; koşum dışı tüm adımlar salt-okuma; commit YOK; loglar `~/.hermes/cache/scratch/p26_tur5_{egitim,gate1_sftval,gate2_pencere,gate3_testmodel,gate4_sonda}.log`; üretici/montaj artefaktları scratch'te (`trllm_p26_veri_v4_*`, `trllm_p26_veri_gen_v4.py`, `trllm_p26_veri_montaj_v4.py`, `trllm_p26_tur5_{gate_sftval,sonda}.py`).