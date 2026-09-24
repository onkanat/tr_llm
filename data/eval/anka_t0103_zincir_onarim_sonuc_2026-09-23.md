# T-0103 · ZİNCİR ONARIM SONUCU — decompiler + graph + self-match (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Görev:** T-0103 (üst: T-0102 KAPANDI) ·
**Koşum YOK** (CPU) · İlân: `anka_t0103_zincir_onarim_ilani_2026-09-23.md`
(ölçümden ÖNCE, sha256 `a418eb3c…`) · Sonda rc=**1** (bant-dışı ayrışması
§2 — beklenti yazımı, kod hatalı DEĞİL) · pytest **290 passed / 1 failed**
(bilinen gateway sandbox istisnası) · `olcum_kabi` rc=**0** (27 PASS).

## 1. Ilanlı beklenti kıyası

| ölçüm | ilanlı beklenti | ölçülen | hüküm |
|---|---|---|---|
| (A) S2 recover (13.064 tam) | %35-45 | **%42,19** (5.512) | BAND İÇİ ✓ |
| (B) 539 DECOMP | 0,950-0,970 + mono-tonal | **0,9494** (+0,0021) | **BANT DIŞI** → §2 ayrıştırma |
| (B) mono-tonal | SONRA ≥ ÖNCE (0,9473) | 0,9494 > 0,9473 | ✓ |
| (C) determinizm | 539/539 | **539/539** | ✓ |
| (D) pytest + olcum_kabi | yeşil | 290/1 (bilinen) + rc=0 | ✓ |

## 2. Bant-dışı ayrıştırma — (B) neden 0,950'yi 0,0006 ile kaçırdı

Bandı geriye-dönük genişletilMEDİ; ilanlı değer kayıtta kalır.

* Kayıt-düzeyi ayrışma (T-0101 JSON ile çift-koşum): **25 kayıt iyileşen**
  (`<UNK>` → `çıta DERIV_lAn GERUND_ArAk` gibi gerçek zincirler), **9
  kayıt bozulan** — bypass eş-değer alternatif zincirleri seçiyor
  (`oluşturulma`+… vs `oluştur`+`Il`+`mA`+…): ikisi de graph'ta geçerli,
  scoring az-morfem tercih ediyor; bazılarında decompile yüzeyi ref'ten
  sapıyor. Bu **morfolojik belirsizlik** eski kodda da vardı (self-parse
  UNK zaten kayıpsız değildi); net etki pozitif (+16 kayıt).
* `ara`-META düzeltmesi: 15 kaydın çözülmemiş `POSS_3SG CASE_LOC_N`
  etiketleri `arasında` olarak çözüldü ✓.
* **Bandın kendisi +0,0006 iyimser yazılmış** — kumanda n=300'den
  birebir-ölçekleme yerine geniş tahmin; ölçülen iyileşme gerçek ama
  küçük (539'da UNK ~142 token / 20.424 = %0,70 havuzdan dönüşen pay).
  **Hüküm:** iyileşme ölçüldü ve mono-tonal sağlandı; bant yazımı
  ders-dosyasına işlendi (beklenti kumanda n=300'den bantlanırken
  +0,002-0,005 civarı dar bant beyan edilmeliydi).

## 3. Uygulanan değişiklikler (donmuş `src/compiler/**`, kiralamalı, ilanlı)

1. **`core.py` (`9c157a04…`):** optional `vocab` (None ⇒ 11+ arama
   yerinde davranış birebir aynı — `vocab-sız bıçaklanış` probe'u
   birebir beyanlı) + self-match bypass YALNIZ OOV'de.
2. **`morphotactics.py` (`21f85454…`):** `mark_terminal(VOICE_CAUSATIVE,
   VOICE_PASSIVE)` + `DERIV_lAş/lAn` NOUN kaynaklı geçişler. (Süreç
   dersi: ilk edit `mark_terminal(VERB_ROOT)`'ı EZDİ — `horozlaş`
   sessiz boş döndü; terminal kümesi yazdırılarak yakalandı.)
3. **`decompiler.py` (`241e54fb…`):** `ara` META çarpışması İKİ katman:
   `decompile_sentence` `is_meta` dalına lookahead (ardından suffix
   gelen meta-kök gerçek kök) + `decompile_tags` guard. `sorgu: ara` /
   `belge: tanım: cevap: var` davranışı korundu.

Örnek onarımlar: `bıçaklanış`→`bıçaklan INF_Iş` · `hiddetlendir`→
`hiddetlen VOICE_CAUS_DIr` · `aratıl`→`arat VOICE_PASS_Il` ·
`horozlaş`→`horoz DERIV_lAş` · T-0101 akıl-sağlığı
(`kullanılmalıdır`, `marangozlukta`) DEĞİŞMEDİ.

## 4. Sınırlar (beyanlı) — sonraki işler

| sınıf | kitle | nereye |
|---|---|---|
| S2 kök-OOV (donmuş vocab'da yok) | %27,0 | vocab yenileme / Katman 2 |
| S2 yeni-id + kompozisyon (`ıver`, isim bileşikleri) | 91/107 yol-yok | vocab yenileme sonrası |
| encode-tarafı kesme/tırnak kaybı | T-0102'nin 279'lık sınıfın ana bileşeni | `tokenizer.py:306` DONMUŞ — operatör kararı |
| S1 (`tech<id>` hedefi) | %13,25 | Katman 4 — operatör öngörüsü: etki küçülecek, bu ölçümle uyumlu |

## 5. Digest tablosu

| artefakt | sha256 |
|---|---|
| `src/compiler/core.py` | `9c157a04…` |
| `src/compiler/morphotactics.py` | `21f85454…` |
| `src/compiler/decompiler.py` | `241e54fb…` |
| `scratch/anka_t0103_zincir_onarim_sondasi.py` | `4b6ca454…` |
| `scratch/anka_t0103_zincir_onarim_sondasi.json` (per-kayıt üretim DAHİL) | `00242614…` |
| `data/eval/anka_t0103_zincir_onarim_ilani_2026-09-23.md` | `a418eb3c…` |

*Kaynak çıpası: T-0101 ÖNCE DECOMP 0,9473 (`46bf1537…`) · vocab 33.114 ·
ECA kapısı DEĞİŞMEDİ (satır 368-369 vocab'sız kurulum duruyor; kapının
bu onarımdan yararlanması ayrı operatör kararı).*

## 6. Kapanış

**T-0103 KAPANDI (kısmi, band-dersi beyanlı):** DECOMP tavan 0,9473 →
**0,9494** (+0,0021; mono-tonal ✓); S2 compile-recover %0,03 → **%42,19**
(5.512/13.064 — bant içi); determinizm 539/539; yüzey doğrulama yeşil.
Operatör hipotezinin düzeltmesi: gerçek engel graph zincirleri DEĞİL
önemsiz self-match parse'tı; en sık ekler graph'ta zaten vardı. Kalan
boşluk donmuş vocab/tokenizer işi (§4 tablo). **Katman 2 (İngilizce
kalıntı temizliği) sıradaki en büyük tek kalem — operatör onaylı.**