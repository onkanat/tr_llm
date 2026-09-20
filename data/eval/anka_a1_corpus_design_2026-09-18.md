# T-0072 — Anka A1-a: külliyatı **düz metin** temsiliyle yeniden derleme (TASARIM + KAPI İLANI)

**Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** İlan damgası: `2026-09-18T13:16:56Z`
Derleme koşumunun `baslangic_utc` damgası bundan **sonra** olmalıdır — "önce ilan, sonra ölçüm"
sırasının kanıtı budur.

## 0. Operatör kararı ve dayanağı

**Onkanat (18 Eyl 2026):** *"`literal=False` ile yeniden derle."*

T-0071 kapanışında ölçülen temsil farkı (`<ENT><CAP>i n g i l t e r e</ENT>` = "İngiltere" 12 jeton,
külliyatın **%35.58'i** varlık bloğu) operatöre **A1 öncesi karar noktası**
olarak sunuldu; seçim bu belgeyle uygulanıyor.

### İki temsil, aynı örnek üzerinde YENİDEN ölçüldü (400 makale)

| | `literal=True` (T-0071) | `literal=False` (yeni) |
|---|---|---|
| jeton | 2,125,673 | 1,359,106 |
| jeton/makale | **5,314.2** | **3,397.8** |
| **karakter/jeton** | **2.5951** | **4.0587** |
| UNK oranı | %1.2056 | %1.9955 |
| noktalama oranı | %7.2764 | %11.3805 |
| `<ENT>` | **85,030** | **0** |
| `<CAP>` | 79,778 | 0 |
| varlık bloğu oranı | %35.5802 | %0.5708 |
| `<PROPER_NOUN>` | 0 | **80,990 (%5.96)** |

⇒ **766,567 jeton kazanç (%36.06)**; aynı jeton bütçesiyle
**1.56× daha fazla gerçek metin** okunur.

Bu sayılar **elle yazılmadı** — `scratch/anka_a1_design.py` örneği bu koşumda yeniden tokenize etti.

### ✅ Kabul edilen bedel (kayda geçirilir, gizlenmez)

`literal=False` yolunda sözlükte/morfotaktikte bulunamayan büyük-harf kelime **tek
`<PROPER_NOUN>` jetonuna** çöker (`src/llm/tokenizer.py:318-333`, varsayılan dal).
⇒ **Model özel ad ÜRETEMEZ**; "İngiltere" yerine yer tutucu basar. Ölçülen oran
**%5.96**. Bu bir *yetenek kaybıdır* ve A1 sonrası ölçülecek bir kapının
konusudur; bu belge onu **kabul edilmiş** olarak kaydeder, "sorun yok" demez.

## 1. Değişen tek şey

| | T-0071 | T-0072 |
|---|---|---|
| tokenizer | `literal_entity_mode=True` | **`literal_entity_mode=False`** |
| çıktı | `data/anka_pretrain.bin` | **`data/anka_a1_pretrain.bin`** |
| derleyici | `scratch/anka_a0b_build.py` | **`scratch/anka_a1_build.py`** |

**Değişmeyen:** kurtarılan külliyat (repo dışı HF önbelleği), arınma kuralları, kanonik sözlük
`data/rebuild/vocab_base_32852.json` (32852 giriş), belge düzeyi `<BOS>…<EOS>`, min 300 karakter eleme,
sıra (parquet glob sorted + satır sırası), hedef 100 M + 2 M jeton, seri referans yolu,
artımlı yazma.

**Neden yeni betik:** T-0071'in `scratch/anka_a0b_build.py`'sinin digest'i **kapalı raporda
kayıtlıdır**; onu düzenlemek o raporu bayatlatırdı.

**Neden yeni çıktı yolu:** T-0071'in `data/anka_pretrain.bin`'i **ölçülmüş bir artefakt**;
üzerine yazmak kaydı bayatlatır. İkisi yan yana durur.

## 2. 🚩 İLAN EDİLEN KAPILAR (ölçümden ÖNCE — sonradan DEĞİŞTİRİLMEZ)

T-0070'de ilan edilen **K1–K7 aynen korunur**; üstüne **K8** eklenir.

| kapı | koşul | ayırt edicilik (**zorunlu**: hangi girdi düşürür?) |
|---|---|---|
| **K1 NOKTLAMA** | `max_id`≥32145 VE oran ≥%5,0 | eski 12 .bin'de oran tam %0,00 |
| **K2 UNK TAVANI** | ≤%4,0 | ornekte %2.00 — esigin ALTINDA olmali |
| **K3 PAD** | ≤%1,0 | hiç PAD yazılmamalı |
| **K4 ÖLÇEK** | ≥90 M jeton | hedef dolmadan durursa düşer |
| **K5 TEKRARSIZLIK** | train∩val = 0 | ⚠️ **val, train kesildikten SONRAKI makalelerden alinir ⇒ kesisim inşa gereği imkansiz; K5 bir kulliyat ozelligi degil KENDINI KONTROLTUR** |
| **K6 İZLENEBİLİRLİK** | her kaynak + tam sha256 | sha256 alınmazsa düşer |
| **K7 POZİTİF KONTROL** | noktalama > 0 | hat olurse K1..K6 KANITSIZ |
| **K8 TEMSİL** (YENİ) | `ENT`==0 VE `CAP`==0 VE `<PROPER_NOUN>` ≥%1,0 | literal=True ile derlenirse ENT milyonlarca olur ⇒ DUSER; T-0071'in 'kapilar temsili sormuyor' kusurunu kapatir |

**K8'in gerekçesi:** T-0071'de kapıların hiçbiri **akışın neyden oluştuğunu** sormuyordu; bu yüzden
külliyatın %39,1'i varlık işaretlemesi olduğu hâlde 7/7 "GEÇTİ" çıktı. K8 tam o boşluğu kapatır ve
**iki hattı birbirinden ayırt eder**.

## 3. 🚩 ÇÜRÜTME ŞARTI (ölçümden ÖNCE ilan edildi)

> **literal=False kulliyatinin karakter/jeton orani literal=True'ninkinden KUCUK cikarsa, 'ayni butcede daha fazla metin okur' gerekcesi CURUR ve karar yeniden acilir.**

Örnekte ölçülen: `karakter/jeton` true **2.5951** · false **4.0587**
⇒ `çürüdü = False`.

**Neden şimdi yazıldı:** T-0071 kendi çürütme şartını **ölçümden sonra** yazmıştı ve bunu tasarım
eksiği olarak kaydetmişti. Bu görev o eksiği tekrarlamıyor.

## 4. Bu belgenin İDDİA ETMEDİĞİ şeyler

1. **"Düz metin daha iyi bir model verir" DEĞİL** — yalnız *jeton başına daha fazla gerçek metin*
   ölçüldü. Kalite A1 sonrası ayrı kapıdır.
2. **`<PROPER_NOUN>` kaybı ölçülüp giderilmedi**; yalnız kabul edildi ve kayda geçirildi.
3. **K5 boş bir kapıdır**; "GEÇTİ" hükmü külliyat hakkında bir şey söylemez.
4. Bu belge **hiçbir `.bin` yazmadı**; derleme ayrı adımdır.

İlgili: [[anka-yeni-model-adi]], [[anka-kulliyati-duz-metin-degil]], [[tavan-artefakti-kapi-gecmez-kanitsizlik]],
[[ilan-edilen-kural-ayirt-edici-olmali]], [[tasarlayan-kendi-iddiasini-yanlislayabilmeli]].
