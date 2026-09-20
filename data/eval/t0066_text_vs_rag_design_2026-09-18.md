# T-0066 TASARIM — Temel Model ve Ceketin TEXT / RAG Koşullarında Ölçümü

**Damga:** 2026-09-18T07:45Z · **Yazan:** claude · **Görev:** T-0066
**Statü:** TASARIM — **eşikler ve tahminler bu dokümanla, ölçüm GÖRÜLMEDEN ilan edilir**
**Dayanak:** Kullanıcı talimatı (18 Eyl 2026): *"temel model ve ceketi text ile rag ile test edin."*
**Runtime pilotu:** `scratch/t0066_pilot.py` — yalnız cihaz/hız/zarf ölçtü, **sonuç ölçmedi** (damga: aynı gün, ölçümden önce).

---

## 1. Bu bir kapı DEĞİL, davranış ölçümüdür

T-0062/T-0063 bir **unutma kapısı** kurdu ve F4 kararı kapandı. T-0066 farklı bir soru sorar:
**iki model, belge verildiğinde ve verilmediğinde ne yapıyor?** Hiçbir eşik burada bir checkpoint'i *seçmez*; yalnız davranışı kaydeder.

## 2. Çözülen belirsizlikler (kayda geçirilir)

| terim | karar | gerekçe |
|---|---|---|
| **"ceket"** | `data/kristal_model_f4_r05.pt` | T-0064'te kabul edilen F4 noktası (18 Eyl) |
| **"temel model"** | `data/kristal_model_f3_clean.pt` | T-0062/T-0063'te "taban" olarak kullanılan model; kullanıcının çalıştığı çift |
| **ek kol (betimleyici)** | `data/kristal_model.pt` | `chat_prompt.py` arayüzü **bunu** "Temel Model" diye adlandırıyor → isim belirsizliğini kapatmak için ölçülür, **hiçbir kapıya girmez** |
| **RAG zemini** | `data/realistic_rag/test_natural_150.jsonl` | donmuş; 150 kaydın `input` alanı **zaten** kanonik `<BELGE> {doc} </BELGE> {query}` zarfını taşıyor (150/150 parse edildi) |

**Neden bu RAG zemini:** TEXT ile RAG arasındaki fark **tek değişkenli** olur — aynı kayıt, aynı soru, aynı referans cevap; yalnız **belgenin varlığı** değişir.

## 3. Kollar

### 3.1 Koşullar (girdi zarfı)

| koşul | instruction | input | ne izole eder |
|---|---|---|---|
| **RAG** | dosyadaki | `<BELGE> {doc} </BELGE> {query}` | belge **var** |
| **TEXT** | **aynı** (dosyadaki) | `{query}` | yalnız **belge çıkarıldı** (tek değişken) |
| **TEXT_NOINSTR** *(betimleyici)* | `""` | `{query}` | belge **ve** "belgeye dayanarak" talimatı çıkarıldı |

> **Neden TEXT_NOINSTR var:** TEXT kolu, belgesi olmayan bir isteme *"Belgeye dayanarak soruyu yanıtla"* diyor — bu bilinçli bir asimetridir (tek değişken uğruna). TEXT_NOINSTR, *"belgeyi çıkarmak"* ile *"talimatı çıkarmak"* etkilerini ayırır. **Kapı değildir.**

### 3.2 Modeller

`f3_clean` (temel) · `f4_r05` (ceket) · `kristal_model.pt` (betimleyici, kapı dışı)

### 3.3 Sözleşme (üç kol için de aynı)

- Sözlük: `data/rebuild/vocab_base_32852.json` (**32.852** satır) — T-0046 kanonik tabanı.
- Tokenizer: `literal_entity_mode=True` (kanonik **veri üreticilerinin** kullandığı değer).
- Zarf: `render_example` / `render_prompt` (`src/llm/prompt_contract.py` → `_render_structured_prompt`). **Kalıcı `<BOS>` yazılmaz** (encode ekler).
- Boyut uyuşmazlığı `resize_state_dict` ile kapatılır; `[SOZLESME_UYARI]` satırı **görünür bırakılır** (sessiz kırpma/doldurma yok).
  - `f3_clean`/`f4_r05`: 32.852 → uyuşmazlık **beklenmiyor**.
  - `kristal_model.pt`: **32.816** → 36 satır **doldurulacak** (uyarı basılacak).

## 4. Ölçütler

| # | ölçüt | nasıl |
|---|---|---|
| **1 (birincil)** | Referans cevabın **teacher-forced CE**'si, n=150'nin **tamamında** | tek forward; `model(x, targets=y)`, `sign_mask` model içinde hesaplanır (deterministik) |
| **2 (pozitif kontrol)** | **Belge-kullanımı**: aynı kayıt, **başka** bir kaydın belgesiyle (deterministik permütasyon) CE farkı | swap |
| **3** | **Üretim**: greedy decode, tabakalı n=100 (5 domain × 20), **yüzey ROUGE-L** (`rouge_l_score`, kelime-düzeyi LCS F1) | ham üretimler saklanır |
| **4** | Belge içeriği kullanımı: üretim ile **sorunun/belgenin** içerik-kök örtüşmesi (≥2 kök oranı) | betimleyici |

**İki eksen ayrı raporlanır:** CE (zorlanmış) ve ROUGE (serbest üretim) **farklı sorular** sorar; biri diğerinin yerine geçmez.

## 5. **ÖNCEDEN İLAN EDİLEN** tahminler

| # | tahmin | yön |
|---|---|---|
| **P1** | RAG CE **<** TEXT CE — belge cevabı kolaylaştırır | her iki modelde |
| **P2** | swap-belge CE **>** doğru-belge CE — model belgeyi gerçekten kullanıyor | her modelde |
| **P3** | ceket RAG CE **≤** temel RAG CE × **1,10** — ceket RAG yeteneğini bozmadı | |
| **G1** | ceket RAG yüzey-ROUGE **≥** 0,90 × temel RAG ROUGE | |
| **G2** | ROUGE(RAG) **>** ROUGE(TEXT) | her iki modelde |

## 6. **ÖNCEDEN İLAN EDİLEN** eşikler (ölçümden sonra DEĞİŞTİRİLMEZ)

| karar | eşik |
|---|---|
| **Belge kullanımı KANITLANDI** | P2 farkının **eşleştirilmiş %95 GA alt sınırı > 0** |
| **"Ceket RAG'ı bozmadı"** | **P3 VE G1 birlikte** |
| **"RAG koşulu işe yarıyor"** | **P1 VE G2 birlikte** (iki farklı eksende aynı yön) |
| **Sınır bandı** | P3 oranı 1,05–1,10 arasında kalırsa **açıkça raporlanır** ve karar kullanıcıya bırakılır |

**Mutlak CE karşılaştırması modeller arası YAPILMAZ** — TEXT kolu "belgesiz ama aynı soru"dur; modeller arası kıyas yalnız **aynı koşul içinde** (P3/G1) yapılır.

## 7. Örneklem ve güç

- CE: **n=150** (tümü) — eşleştirilmiş (aynı kayıtlar, aynı pencereler).
- Üretim: **n=100**, 5 domainden **20'şer** (deterministik: `RANDOM_SEED = 42`).
- Swap: **n=150**, permütasyon `π(i) = (i·97 + 13) mod 150` (sabit noktasız olması kontrol edilir).

**İlan edilen sınır (dürüstçe):** n=150'de CE farkı için GA genişliği ~±0,02–0,04 nat mertebesinde olacak; **nokta tahmini hüküm vermez**, GA raporlanır.

## 8. Koşum koşulları

- `device=mps` **zorunlu ve kabul kriteri** (sandbox MPS'i gizler → sandbox **DIŞINDA** koşulur).
- Koşum sırasında makinede GPU tüketen başka iş **çalıştırılmaz** (T-0052). Hız başta/sonda ölçülür; **2–3× sapma = yük sinyali**.
- **Artımlı yazma** (her model×koşul biriminden sonra `save()`); sentinel `rc=0`'a bağlı (T-0063 dersi).
- Ham üretimler `data/eval/t0066_generations_2026-09-18.jsonl`'e yazılır.

## 9. Bu ölçümün CEVAPLAYAMADIĞI şeyler (önceden)

- **Üretim RAG boru hattı DEĞİLDİR.** Kanonik zarfla çalışır; `qdrant`/`rag_pipeline.py`'nin kosinüs eşiği (`match_score >= 0.40`) ve `clean_doc_tags` kusuru bu ölçümün **dışındadır**.
- **Tek RAG kaynağı** (`realistic_rag`); başka korpusla genellenemez.
- **`kristal_model.pt` farklı sözleşmeyle eğitilmiş olabilir** (32.816 satır, `literal_entity_mode` servis yolunda `False`) → o kolun sayıları **diğer ikisiyle aynı ölçekte okunamaz**; bu yüzden kapı dışıdır.
- `TEXT_NOINSTR` betimleyicidir, kapıya girmez.

## 10. Ölçüm-ÖNCESİ geçerlilik kontrolü (referans tarafı — hiçbir model kullanılmadı)

**Damga:** 2026-09-18T07:50Z · eşik **değişmedi**, yalnız kapının dayanağı ölçüldü.

Ham çıktı ROUGE'u hangi tabanda ölçmeli? Adaylar: **yüzey** (decompile edilmiş Türkçe) ve **morfemik** (projenin C ekseninin kullandığı taban). Karar ölçümden önce verilmeliydi, verildi; ama önce **tavan** ölçüldü — referans cevaplar encode→decode→decompile turundan geçirildi (model yok):

| taban | tur-gidiş-dönüş ROUGE tavanı |
|---|---|
| **yüzey** (decompile) | **0,9579** |
| **morfemik** | **0,3074** |

Decompile hatası: **0/150 (%100 başarı)** → yüzey tabanı **karışık değil**, tek temiz taban.

**Sonuç:** eşik **yüzey ROUGE** üzerinde ilan edilmiş kalır (tavanı 0,9579 → ayırt edici ölçek). Morfemik taban da **raporlanır** ama kapıya girmez: tavanı 0,3074 olduğu için ölçek sıkışıktır ve yüksek değerler ayırt edilemez.

> **Sınır kaydı (kapalı karara dokunmaz):** T-0063 C ekseninde bildirilen taban **0,3425**, morfemik tabanın bu turdaki tur-gidiş-dönüş tavanının (**0,3074**) *üstündedir*. İki sayı aynı ölçekte olmayabilir (T-0063 kırpık kafalı model ve farklı sözlükle ölçtü — `scripts/evaluate_carpenter_generation_100.py` 31.357 satırla kurar). Bu bir **ölçek gözlemidir**; **hiçbir kapalı kararı, eşiği veya hükmü değiştirmez** ve F4 kararını yeniden açmaz.

## 11. Çıktılar

| dosya | ne |
|---|---|
| `data/eval/t0066_text_vs_rag_2026-09-18.json` | ölçüm (betik hesaplar, elle sayı yok) |
| `data/eval/t0066_generations_2026-09-18.jsonl` | ham üretimler (denetlenebilir) |
| `data/eval/t0066_text_vs_rag_2026-09-18.md` | rapor |
| `.agent-bus/notes/T-0066.md` | anlatı |
