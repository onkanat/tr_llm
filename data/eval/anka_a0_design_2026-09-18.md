# Anka A0 — VERİ ELDEN GEÇİRME TASARIMI VE İLAN EDİLEN KAPILAR

**Görev:** T-0070 (Anka A0-a) · **damga:** 2026-09-18T12:21:32Z
**Talimat (Onkanat, 18 Eyl 2026):** *"…elinizdeki data-set ve Proje dokümanları ile modeli yeniden
aşama aşama eğitin. **Data setleri model ihtiyacına göre elden geçirin** mimari temel yapısını koruyun."*
+ *"Yeni modelin adı Anka çünkü Kristalin küllerinden doğuyor."*

**Onaylar:** silme = kalıcı `rm` · kapsam = hepsi · **donmuş veri yolu güncelleme izni VAR** ·
ön-eğitim ölçeği = **~100 M jeton / ~1 gün** · yürütücü = **yalnız claude-code**.

> **Bu belge A0'ın yarısıdır.** Burada ölçüm ve **kapı ilanı** var; derleme (corpus build) T-0071'de
> yürütülür ve kapılar **orada** ölçülür. Kapılar ölçümden **ÖNCE** ilan edilir ve **sonradan
> değiştirilmez**; ölçüm sonrası bulunan kusur kapıya değil **rapora** yazılır.

---

## 1. Ölçülen envanter (betikle — `data/eval/anka_a0_bin_audit_2026-09-18.json`)

### 1.1 Sözlük

`data/rebuild/vocab_base_32852.json` · sha256 `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` · **32852 giriş** · `next_id=32852`

Noktalama bloğu **32137–32145**: `['.', ',', '?', '!', '-', ':', ';', '(', ')']` · kesme `'` = **32850**

### 1.2 `.bin` dosyaları — 17 adet

| dosya | jeton | etkin (PAD'siz) | max_id | PAD | nokt. token | noktalama |
|---|---|---|---|---|---|---|
| `data/train.bin` | 10,761,665 | 10,761,665 | 25664 | 0.00% | 0 | — |
| `data/train_all_chosen.bin` | 18,097,297 | 18,097,297 | 28260 | 0.00% | 0 | — |
| `data/train_balanced_sft.bin` | 23,356,537 | 23,356,537 | 32136 | 0.00% | 0 | — |
| `data/train_balanced_sft_v2.bin` | 1,793,433 | 1,793,433 | 32850 | 0.00% | 180,336 | VAR |
| `data/train_carpenter_specialization.bin` | 867,650 | 867,650 | 32850 | 0.00% | 74,379 | VAR |
| `data/train_chat_balanced.bin` | 3,120,000 | 1,550,854 | 32850 | 50.29% | 139,731 | VAR |
| `data/train_chat_balanced_clean.bin` | 2,416,000 | 1,117,492 | 32850 | 53.75% | 104,828 | VAR |
| `data/train_chat_sft.bin` | 1,232,000 | 567,445 | 28259 | 53.94% | 0 | — |
| `data/train_corpus.bin` | 595,555 | 595,555 | 16652 | 0.00% | 0 | — |
| `data/train_deep_sft.bin` | 11,216,512 | 2,769,078 | 31321 | 75.31% | 0 | — |
| `data/train_dpo_chosen.bin` | 1,171,921 | 1,171,921 | 17285 | 0.00% | 0 | — |
| `data/train_f4_replay_mix.bin` | 1,156,930 | 1,014,356 | 32850 | 12.32% | 87,691 | VAR |
| `data/train_future_finetune.bin` | 6,204,420 | 6,204,420 | 31338 | 0.00% | 0 | — |
| `data/train_infancy.bin` | 339,742 | 339,742 | 16654 | 0.00% | 0 | — |
| `data/train_parenting.bin` | 451,358 | 451,358 | 16656 | 0.00% | 0 | — |
| `data/train_pedagogy_highschool.bin` | 2,344,149 | 2,344,149 | 31356 | 0.00% | 0 | — |
| `data/train_wiki.bin` | 8,228,842 | 8,228,842 | 27243 | 0.00% | 0 | — |

**Özet:** ham **93,354,011** jeton → **etkin 81,231,794** (81.2 M) ·
PAD **12,122,217** (13.0%) ·
noktalamasız **12/17** · PAD>%10 olan **5** dosya.

> **D1 bulgusunun DÜZELTİLMİŞ hâli.** Önceki kayıt *"14/14 `.bin`'de `max_id ≤ 32136` ⇒ hiçbir model
> noktalama görmedi"* diyordu. Bugünkü ölçüm bunu **kısmen çürütüyor**: `max_id` kümesi
> `[16652, 16654, 16656, 17285, 25664, 27243, 28259, 28260, 31321, 31338, 31356, 32136, 32850]` — 13 ayrık değer, **32137–32849 arası HİÇ değer yok**. Yani
> dosyalar ya noktalamasız ya (T-0044 sonrası derlenen 5 dosya) noktalama+varlık bloğunu birden
> görmüş. **Taban ön-eğitim** (`train.bin`, `train_wiki.bin`, `train_balanced_sft.bin`) noktalama
> **görmedi** — "hiçbir model görmedi" cümlesi fazla geniş, "taban görmedi" doğrudur.

### 1.3 Diğer kaynaklar — 25 jsonl · 144,357 kayıt · bozuk 0

| dosya | kayıt | ilk alanlar |
|---|---|---|
| `data/benchmark_evaluation_raw.jsonl` | 31 | checkpoint, test_type, persona, soru |
| `data/future_train_archive.jsonl` | 358 | instruction, input, output, decompiled_output |
| `data/simulasyon_bellek_export.jsonl` | 4,073 | id, payload, vectors |
| `data/train_sft.jsonl` | 2 | instruction, input, output |
| `data/pedagogy/arena_base_accumulated.jsonl` | 899 | instruction, input, output, domain |
| `data/pedagogy/arena_carpenter_accumulated.jsonl` | 302 | instruction, input, output, domain |
| `data/pedagogy/carpenter_specialization_dataset.jsonl` | 5,400 | instruction, input, output |
| `data/pedagogy/chat_conversations.jsonl` | 6,000 | instruction, input, output |
| `data/pedagogy/cot_vault.jsonl` | 91 | timestamp, query, instruction, thought_trace |
| `data/pedagogy/dpo_all_tokenized.jsonl` | 6,198 | prompt_ids, chosen_ids, rejected_ids |
| `data/pedagogy/dpo_tokenized.jsonl` | 321 | prompt_ids, chosen_ids, rejected_ids |
| `data/pedagogy/high_school_foundation_dataset.jsonl` | 688 | instruction, input, output, expected_keywords |
| `data/pedagogy/infancy_dataset.jsonl` | 3,682 | instruction, input, output |
| `data/pedagogy/lexical_semantics_dataset.jsonl` | 25,000 | instruction, input, output |
| `data/pedagogy/literature_poetry_dataset.jsonl` | 220 | instruction, input, output, expected_keywords |
| `data/pedagogy/middle_school_chat.jsonl` | 125 | instruction, input, output |
| `data/pedagogy/parenting_dataset.jsonl` | 19,342 | instruction, input, output |
| `data/pedagogy/parenting_deep_dataset.jsonl` | 40,001 | instruction, input, output |
| `data/pedagogy/rag_dataset.jsonl` | 11,963 | instruction, input, output |
| `data/pedagogy/rag_interactive_dataset.jsonl` | 6,599 | instruction, input, output |
| `data/pedagogy/raw_teacher_cards_archive.jsonl` | 48 | client, collection, id, payload |
| `data/pedagogy/turk_tarihi_chat.jsonl` | 6,101 | instruction, input, output |
| `data/pedagogy/turk_tarihi_dpo_tokenized.jsonl` | 406 | prompt_ids, chosen_ids, rejected_ids |
| `data/pedagogy/turk_tarihi_sft.jsonl` | 6,507 | instruction, input, output |

Boş dosya: ['data/future_train_vector.jsonl'] *(bilinen kusur, 0 bayt)*

---

## 2. Arınma hattı ve etkisi (`data/eval/anka_a0_cleaning_2026-09-18.json`)

**Kurtarılan külliyat (repo dışı, HF önbelleği):** 631,148 makale ·
1,443,545,748 karakter · 2 parquet.

Ölçüm örneği: **1500 makale**, sabit tohum `42` (deterministik).

| | jeton | UNK | noktalama | karakter/jeton |
|---|---|---|---|---|
| **ham** | 1,512,048 | **6.59%** | **10.35%** | 2.2281 |
| **arınmış** | 833,293 | **1.61%** | **6.08%** | 2.441 |

UNK **6.59% → 1.61%** (×4.10 azalma) ·
noktalama **10.35% → 6.08%** (blok korunuyor).

**2.1 POZİTİF KONTROL — hat canlı mı?** Naif bir markdown temizleyicisi öğretmek *istediğimiz*
noktalamayı da siler. Bu yüzden noktalama içerdiği bilinen bir referans metni aynı hattan geçirildi:

> `Ankara, Turkiye'nin baskentidir. Nufusu 2026'da 5.700.000 idi; hava -3 dereceydi! Neden? (bkz. 2. madde)`

ham noktalama token'ı **12** → arınmış **12** ⇒
**HAT CANLI**

**2.2 Satır filtresi hassasiyeti** (`satir_min`, bir *tasarım seçimi* — kapı değil):

| satir_min | elenen belge | jeton | UNK | noktalama |
|---|---|---|---|---|
| 0 | 576 | 876,016 | 1.59% | 6.11% |
| 10 | 615 | 862,569 | 1.60% | 6.07% |
| 20 | 642 | 833,293 | 1.61% | 6.08% |
| 40 | 672 | 779,161 | 1.57% | 6.01% |

**Bulgu:** eşik kaliteyi **etkilemiyor** (UNK 1.57–1.61%,
noktalama 6.01–6.11%), yalnız **verimi** değiştiriyor.
⇒ Naif endişe (*"filtre veriyi sessizce yok ediyor"*) **çürüdü**: elenen belgeler kısa **taslak
maddelerdir**, sürücü satır filtresi değil **belge-uzunluk filtresidir** (satır filtresi tamamen
kapalıyken bile 576/1500 eleniyor).
**Karar: `satir_min = 0`** (en yüksek verim 876,016, en düşük UNK) — gerekçe
ölçüm, tercih değil.

**2.3 Verim ve projeksiyon (hesaplanmış, elle yazılmamış):**

- Arınma **karakter verimi %60.4** (jeton verimi %55.1)
- Hedef **100 M jeton** ⇒ ~**244 M arınmış karakter** ⇒
  ~**404 M ham karakter** = külliyatın **%28.0**'i
- Tokenizer hızı **35,520 jeton/sn** ⇒ derleme ~**47 dakika**

⇒ Külliyat ölçek için **fazlasıyla yeterli** (yalnız %28.0'i gerekiyor) — kıtlık yok, seçim var.

---

## 3. ÖN-EĞİTİM BİLEŞİMİ (ilan)

**A1 ön-eğitim verisi = arınmış Wikipedia külliyatı**, hedef **100 M jeton**, tek kaynak.

**Gerekçe:** eski zincirin ölçülmüş kusuru *ölçek* ve *noktalama yokluğu* idi (etkin
81.2 M jeton; taban noktalama görmemiş). Tek kaynak seçmek **atfı ölçülebilir**
kılar: karışım eklenirse bir kayıptaki payı ayıramayız. Diğer jsonl kaynakları **SFT/sohbet
aşamalarına** aittir (pedagoji, Türk tarihi, sohbet) — ön-eğitime karıştırılmaz.

---

## 4. ⚠️ T-0071 İÇİN İLAN EDİLEN KAPILAR (ölçümden ÖNCE, değiştirilmez)

| # | kapı | eşik | neden ayırt edici |
|---|---|---|---|
| **K1** | **NOKTALAMA** | derlenen `.bin`'de `max_id ≥ 32145` **VE** noktalama bloğu (32137–32145) oranı **≥ %5,0** | eski taban dosyaları **tam %0,00**; eşik onları kesin dışlar |
| **K2** | **UNK TAVANI** | arınmış metinde UNK oranı **≤ %4,0** | ham **6.59%** eşiği geçemez; arınmanın *işe yaradığını* arar |
| **K3** | **PAD DİSİPLİNİ** | ön-eğitim `.bin`'inde PAD oranı **≤ %1,0** | eski SFT dosyaları %50–75; ön-eğitimde dolgu **olmamalı** |
| **K4** | **ÖLÇEK** | derlenen jeton **≥ 90 M** | kullanıcı hedefi ~100 M; −%10 tolerans |
| **K5** | **TEKRARSIZLIK** | aynı belge iki kez derlenmez (belge kimliği benzersiz) | külliyatın %28.0'i kullanılıyor ⇒ çakışma riski gerçek |
| **K6** | **İZLENEBİLİRLİK** | her kaynak dosya + jeton + **tam sha256** olarak raporlanır | kaynaksız zincir kopuk bırakır |
| **K7** | **POZİTİF KONTROL** | §2.1 sonucu; **hat ölürse K1–K6'nın TAMAMI kanıtsız sayılır** | ölü hat "geçti" diyemez |

**Kapı hükmü dili (zorunlu):** "GEÇMEDİ" yazılırken **YOKLUK** mu **KANITSIZLIK** mı olduğu ayırt
edilir; **tavan payı + aykırı/ayrık çift sayısı + kısıtlılık bayrağı birlikte** raporlanır. Tavanda
**GEÇEN** bir kapı da boştur.

**GA/ölçek uyarısı:** bir fark raporlanırken **büyüklük ve ölçek oranı birlikte** verilir; katkı
payı büyüklüğe kör okunmaz.

---

## 5. KENDİNİ YANLIŞLAYABİLME (önceden ilan)

**Bu belgenin temel iddiası:** *kurtarılan külliyat + arınma, noktalama yetenekli ön-eğitimin ana
kaldıracıdır.*

**Hangi sonuç bu iddiayı ÇÜRÜTÜR?** — Önceden yazıldı, ölçüldü:

> `arinmis UNK >= ham UNK VE arinmis noktalama < %5`

**Ölçülen:** çürüdü = **False** (arınmış UNK 1.61% < ham 6.59%;
noktalama 6.08% ≥ %5,0) ⇒ **iddia ayakta**, ama bu *A0'ın* hükmüdür —
**A1 ön-eğitiminin kalitesi hakkında hiçbir şey söylemez**.

---

## 6. BU GÖREVİN İDDİA ETMEDİĞİ ŞEYLER

1. **"Anka daha iyi olacak" DEĞİL.** Silme + yeniden derleme bir *sıfırlamadır*; devralınmış
   soyağacı yeniden üretilmez, **yeni** bir model üretilir. Kıyas cümlesi ancak Anka'nın kendi
   aşama kapılarıyla kurulabilir.
2. **Ceketin ölçülmüş kusuru giderilmiş sayılmaz.** T-0067/T-0068 genel alanda belgeden bağımsız
   −0,52…−0,63 CE ölçmüştü; bu veriyle *ilişkili* olabilir ama **giderildiği ölçülmedi**.
3. **Noktalama *yeteneği* ölçülmedi.** K1 yalnız noktalamanın **veride bulunduğunu** ölçer;
   modelin onu **kullanabildiğini** değil. O ölçüm A1 sonrası, ayrı bir kapıdır.
4. **UNK %1,61 bir "iyi" değil**, yalnız *ölçülmüş* bir orandır. Mutlak eşiği yok.

---

## 7. Artefaktlar (tam 64 karakter sha256; bu belgenin kendisi hariç)

| yol | bayt | sha256 |
|---|---|---|
| `scratch/anka_a0_audit.py` | 5949 | `99a205692eb0ad8108dafaf258c1be9e44a499816369cb94f4c3de7303ec1653` |
| `scratch/anka_a0_clean.py` | 10474 | `2f2d4e96155cf10f1671f7d3de4aef8ea8cadd207a9f015d7b0858efbcb13c1d` |
| `scratch/anka_a0_design.py` | 12243 | `4fe533b2e12a834b6bbae643a31d45729cf371bf1f897f66e50b72cabb61bd9a` |
| `data/eval/anka_a0_bin_audit_2026-09-18.json` | 18555 | `e61b0b96e536398ed19ffc1869e0ad9ce72bf61c1c3f1ce824e8d8fcf75dd896` |
| `data/eval/anka_a0_cleaning_2026-09-18.json` | 2585 | `8275d80066ec12234d29da52622701ec6239e8fa9d590862bbb9a0af05e41e24` |

Bu belgenin kendi sha256'sı tabloya **konmadı** (kendine referans değeri yazıldığı anda değiştirir);
nihai değer üreticinin `[sha256]` satırında basılır ve kapanış kanıtına yazılır.

---

## 8. Sonraki adım

**T-0071 (Anka A0-b):** arınmış külliyatı kanonik 32852 sözlükle derle → `data/anka_pretrain.bin`
→ **K1–K7'yi ölç ve raporla**. Sonra **T-0072 (A1): ön-eğitim ~100 M jeton** (MPS, sandbox
DIŞI, makinede başka GPU işi YOK — T-0052: eşzamanlı yük adım süresini 0,43→3,45 sn'ye çıkardı).

İlgili: [[anka-yeni-model-adi]], [[hf-onbelleginde-wikipedia-kurtarildi]], [[mimari-korunur-kristallm]],
[[bayat-bin-tokenizer-kaymasi]], [[pad-dolgusu-hedef-olarak-ogreniliyor]].
