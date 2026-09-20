# T-0068 TASARIM — BELGE zarf/içerik ayrıştırması (eksen C'nin −0,6255'i nereden geliyor?)

**Durum:** ÖLÇÜMDEN ÖNCE ilan edilmiştir · damga 2026-09-18T10:22:03Z · görev T-0068
**Kapı DEĞİL:** betimleyici tanı ölçümü. Hiçbir checkpoint seçmez, T-0064 F4 kararını açmaz.
**Bu dosyadaki her sayı `scratch/t0068_write_design.py` tarafından hesaplanmıştır** (elle transkript yok).

## 1. Soru ve iki çürütülmüş hipotez

T-0067 eksen C, genel alanda (`data/realistic_rag/test_natural_150.jsonl`) belge-koşullu
cevap-bölgesi CE'sinde ceketin ebeveyne göre **bozduğunu** ölçtü:
fark(f3−f4) = **-0.6254858080546061**, GA [-0.667942; -0.582812],
f3 5.209674 → f4 5.835160, **150/150 öğede bozulma, 0 iyileşme**.
Mekanizma **açıklanmadı**. İki aday hipotez ölçülüp **çürütüldü**:

| hipotez | iddia | ölçüm (betik: `scratch/t0068_bin_scan.py`) | hüküm |
|---|---|---|---|
| danışman (Q1) | ceket eğitiminde **%95** boş `<INPUT>` gördü, `</INPUT>` kapatmaya şartlandı | ceketin gerçek eğitim akışında (`data/train_carpenter_specialization.bin`, sha256 kayıtla birebir) boş `<INPUT> </INPUT>` **3300/12763 = %25.86** | **ÇÜRÜDÜ** |
| bizim (T-0067) | C1 istemleri `<BELGE>` taşıyor, ceket bu zarfı hiç görmedi | taranan **17** eğitim `.bin`'inin **14**'inde `<BELGE>` x0 (`f3_clean`'in kendi verisi `train_balanced_sft_v2.bin` dâhil; belge içeren tek üç dosya: `train_deep_sft.bin` **1545** · `train_chat_balanced.bin` **614** · `train_f4_replay_mix.bin` **63**) ⇒ **iki kol eşit yoksul**, farkı açıklayamaz | **ÇÜRÜDÜ** |

Her iki hipotez de tek bir gözlemden geliyordu. Bu tasarım soruyu **üç bağımsız bileşene ayırır.**

## 2. Girdiler (canlı sha256; bayatlık denetimi geçti)

| ne | yol | sha256 |
|---|---|---|
| korpus (DONMUŞ, yalnız okunur) | `data/realistic_rag/test_natural_150.jsonl` | `a233323011b9be23c839a6c0e4b8f9a2b769dfd8e7702130a3cb2d75041b08c2` |
| ebeveyn | `data/kristal_model_f3_clean.pt` | `fc964af53534b818329c0da6e86aabfb729144615b85bfb6adc6e3c40e767aee` |
| ceket | `data/kristal_model_f4_r05.pt` | `bc35352af126179e21b803159de13250fe54c5a4d940a110afe94ad1f76ba2fe` |
| referans ölçüm | `data/eval/t0067_axis_c_2026-09-18.json` | `1d38acd566ee96409e51631166398fee4c345061f327334778e5f2cf310f0bb5` |
| T-0067 görev kaydı | `.agent-bus/state/tasks/T-0067.json` | `6311b1bb6677d5082a97d4632e6b5ab7d19750e75ba35b3216165877b206a801` |

Betik, canlı sha256'ları eksen C kaydındakilerle karşılaştırdı; **uyuşmazsa tasarım üretilmezdi**
(bayat referansla pozitif kontrol kurulamaz).

## 3. Koşul inşası (`scratch/t0068_conditions.py` — tasarım ile koşan kod AYNI modül)

Aynı 150 satırdan türetilir. `instruction` ve `output` **değişmez**; yalnız `input` içindeki
`<BELGE>` bloğu değişir:

| koşul | dönüşüm | örnek (`Piroksenit`) |
|---|---|---|
| **K1 belge_var** | `input` olduğu gibi | `<BELGE> Doğal ve tarihsel kayıtlarda Piroksenit öne çıkar. Çünkü Piroksenit, n…` |
| **K2 belge_yok** | blok **çıkarılır** | `Metne göre Piroksenit hangi yapısal özelliğiyle tanımlanmaktadır?…` |
| **K3 bos_belge** | belge **metni** silinir, **etiketler kalır** | `<BELGE> </BELGE> Metne göre Piroksenit hangi yapısal özelliğiyle tanımlanmakta…` |

**Sözleşme denetimi (150/150 satır, betik):** K1'de `<BELGE>` x1 · K2'de x0 · K3'te x1 ve iç metin boş ·
**sorgu metni üç koşulda özdeş** · K1'de belge metni boş değil (ayrıştırma anlamsız olmasın).
**İhlal çıktığında tasarım üretilmezdi**; 0 ihlal bulundu.

*Yapısal sonuç (üç koşulda da geçerli):* `output` değişmediği için **cevap bölgesi jetonları özdeştir**
⇒ kollar kayıt bazında eşleştirilmiş kıyaslanabilir. Betik bunu `n_answer` eşitliğiyle ayrıca doğrular.

Boyut (karakter): K1 ortalama **324.9** · K2 **58.9** ·
K3 **75.9** · çıkarılan belge metni ortalama **248.0** karakter.
K1−K2 = **266.0** karakter = belgenin tamamı.

## 4. Ölçüm

* **Büyüklük:** cevap-bölgesi ortalama CE — `ce_answer` (`scratch/t0066_measure.py::ce_breakdown`,
  `logsumexp(logits) − logits[hedef]`); T-0067 eksen C ile **aynı kod yolu** ⇒ kıyaslanabilir.
* **Kollar:** `f3_clean` (ebeveyn) ve `f4_r05` (ceket); her koşulda **150 kayıt × 2 kol** = 300 ileri geçiş.
* **Yön kuralı** (T-0065 dersi — alan adı + işaret birlikte okunur):
  `fark = CE(f3_clean) − CE(f4_r05)` · **POZİTİF = ceket daha iyi** · **NEGATİF = ceket bozmuş**.
  K1 için beklenen fark **-0.6254858080546061** (negatif = ceket bozuk).
* **Belirsizlik:** kayıt bazında eşleştirilmiş bootstrap yüzdelik GA (10.000 yeniden örneklem, tohum 42,
  `scratch/t0067_axis_c.py::boot` ile aynı yöntem). Ayrık çift sayıları (iyi/kötü/sıfır) her koşulda yazılır.
* **Cihaz:** mps **zorunlu**, sandbox dışında koşulur; `device` ve `mps_available` JSON'a yazılır.

## 5. İLAN EDİLEN KARAR KURALLARI (ölçümden SONRA değiştirilmez)

**A — ARAÇ POZİTİF KONTROLÜ (önce bu çalışır, yoksa diğerleri geçersiz):**
`|K1_yeni − (-0.6254858080546061)| ≤ 0,005` **VE** kayıt bazında
`max |CE_f4(K1)_yeni − CE_f4(K1)_kayıtlı| ≤ 0,01`.
Tutmazsa: hüküm **KAYITSIZ** — araç bilinen etkiyi yeniden üretemiyor, üç koşulun hiçbiri okunmaz.

**B1 — ZARF-ŞOKU:** `|K3| ≥ 0,75·|K1|` **VE** `K3 < −0,30`
⇒ kayıp **etiketin VARLIĞINA** bağlı; belge içeriği gerekmiyor (gösterim zarfı şoku).

**B2 — İÇERİK-BAĞIMLI:** `|K2| ≤ 0,15` **VE** `K3 ≤ K2 + 0,15` **VE** `K1 < −0,30`
⇒ kayıp **belgeye** bağlı; salt etiket zararsız (T-0067'nin "belge-koşullu" daraltması **doğrulanır**).

**B3 — BELGEDEN BAĞIMSIZ:** `|K2| ≥ 0,75·|K1|` **VE** `K2 < −0,30`
⇒ kayıp belge kaldırıldığında **da** sürüyor ⇒ **genel dil modellemesi bozulmuş** ve
T-0067'nin *"C1 belge-koşulludur ⇒ 'genel dil bozuldu' diye genişletilemez"* **daraltma hükmü ÇÜRÜR**
(bu, bu ölçümün kendi önceki iddiamızı yanlışlayabileceği yoldur).

**B4 — BİLEŞİK:** hiçbiri temiz ateşlemezse üç pay GA'larıyla **ayrıştırma olarak** raporlanır —
`K_belge = K1 − K2` (belgenin katkısı) · `K_etiket = K3 − K2` (salt etiketin katkısı) ·
`K_icerik = K1 − K3` (belge metninin etiket üstü katkısı) — ve **tek-mekanizma hükmü KURULMAZ.**

*Çokluk uyarısı:* üç koşul için ayrı GA raporlanır; **çokluk düzeltmesi ilan edilmemiştir**, bağlayıcı
olan yukarıdaki karar kurallarıdır. GA'lar betimleyicidir.

## 6. Tavan/taban ve ayırt edicilik kontrolleri

* **Tavan kontrolü:** her koşulun ortalama CE'si düzgün dağılım sınırı `ln(32852) = 10.400` ile
  karşılaştırılır. `> 0,9·ln(V) = 9.360` ise **sıkışma uyarısı** yazılır
  (belge çıkınca cevap bilinemez hâle gelir ve farklar sıkışabilir) — bu durumda fark yine okunur ama
  **mutlak düzeyler yorumlanmaz**.
* **Ayırt edicilik:** ayrık çift sayısı `b+c` her koşulda yazılır; `0` ise o koşul için hüküm
  **AYIRT EDEMEDİ** olur, olumsuz cümle kurulamaz ([[tavan-artefakti-kapi-gecmez-kanitsizlik]]).
* **Pozitif kontrol** A kuralıdır: araç, bilinen −0,6255 etkisini üretmezse ölüdür ([[gosterim-uyusmazligi-olcutu-oldurur]]).

## 7. Bu tasarımın İDDİA ETMEDİĞİ şeyler

1. **Mekanizma kanıtı değil, ayrıştırmadır:** B1/B2/B3 bir *bileşen atfı* verir; "ceketin hangi katmanı
   bozuldu" sorusunu açmaz.
2. **K2/K3 doğal bir görev değildir:** belgesiz istem, korpusun kendi görev tanımının dışındadır.
   Yalnız **kollar arası fark** okunur; K2/K3'ün mutlak CE'si "genel dil kalitesi" ölçüsü değildir.
3. **Kapı değildir:** eşik önceden ilan edilmiş olsa da bu ölçüm hiçbir checkpoint seçmez.
4. **Tek korpus:** yalnız `test_natural_150` (genel alan). Sonuç ceketin kendi alanına (marangozluk)
   taşınamaz.

## 8. Bütünlük

* **Ölçümden sonra bulunan kusur bu dosyada DÜZELTİLMEZ** (sha256 bütünlüğü: "eşikler ölçümden önce
  ilan edildi" iddiası bozulur); düzeltme **rapora** gerekçesiyle yazılır.
* Bu dosyanın sha256'sı ölçüm JSON'una gömülür ve ölçüm betiği tarafından **yeniden doğrulanır**.
* `data/realistic_rag/**` ve `data/*.pt` **donmuştur**: yalnız okunur, kiralamayla bile yazılmaz.
* Çıktılar `data/eval/` (izinli istisna) ve `scratch/` altındadır; kiralama T-0068'e bağlıdır.

## EK — `.bin` zarf taraması (bu tablo `scratch/t0068_bin_scan.py` tarafından üretildi; hiçbir satır atlanmadı)

`data/*.bin` dosyalarının tamamı uint16 jeton dizisi olarak **parça parça** okundu (parça sınırında
komşuluk taşınarak); "boş" = `<INPUT>` jetonunu **hemen** `</INPUT>` izliyor.

| dosya | `<INPUT>` | boş `<INPUT>` | boş oranı | `<BELGE>` |
|---|---|---|---|---|
| `train.bin` | 32 | 0 | 0.00% | 0 |
| `train_all_chosen.bin` | 6 | 0 | 0.00% | 0 |
| `train_balanced_sft.bin` | 199933 | 406 | 0.20% | 0 |
| `train_balanced_sft_v2.bin` | 31463 | 406 | 1.29% | 0 |
| `train_carpenter_specialization.bin` | 12763 | 3300 | 25.86% | 0 |
| `train_chat_balanced.bin` | 24375 | 1706 | 7.00% | 614 |
| `train_chat_balanced_clean.bin` | 18875 | 1706 | 9.04% | 0 |
| `train_chat_sft.bin` | 8125 | 0 | 0.00% | 0 |
| `train_corpus.bin` | 0 | 0 | 0.00% | 0 |
| `train_deep_sft.bin` | 87629 | 1440 | 1.64% | 1545 |
| `train_dpo_chosen.bin` | 6 | 0 | 0.00% | 0 |
| `train_f4_replay_mix.bin` | 15023 | 3480 | 23.16% | 63 |
| `train_future_finetune.bin` | 6900 | 0 | 0.00% | 0 |
| `train_infancy.bin` | 7264 | 0 | 0.00% | 0 |
| `train_parenting.bin` | 19342 | 0 | 0.00% | 0 |
| `train_pedagogy_highschool.bin` | 52739 | 406 | 0.77% | 0 |
| `train_wiki.bin` | 0 | 0 | 0.00% | 0 |

*Okuma:* ceketin kendi eğitim akışında (`train_carpenter_specialization.bin`) boş input oranı
**%25.86** — danışmanın **%95** iddiası hiçbir dosyada görünmüyor (en yüksek: %25,86 ceketin
kendi dosyası, sonra %23,16 `train_f4_replay_mix.bin`).
`<BELGE>` yalnız **3** dosyada geçiyor (`train_deep_sft.bin` **1545** · `train_chat_balanced.bin` **614** · `train_f4_replay_mix.bin` **63**); **`f3_clean`'in kendi eğitim
verisi `train_balanced_sft_v2.bin` dâhil 14 dosyada sıfır** ⇒ `<BELGE>` yoksunluğu iki kolu
*eşit* vurur, K1'deki farkı açıklayamaz.
