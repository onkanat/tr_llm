# G6a (T-0137) RAPOR — TAM ARZ ÜRETİMİ (İLAN-5) — 2026-09-26 (claude)

ILAN ≠ RAPOR: ilan `bahcivan_g6a_ilan_2026-09-26.md` İLAN-5'te koşum ÖNCESİ damgalı;
hükümler BETİKTEN (`scratch/t0137/arz_uretim_olc.py`, elle sayı YOK).

## Hüküm

**ARTZ_URETIM_YETERSIZ** — `scratch/t0137/arz_uretim_hukum.json`
(`9e6fa9a7904fe673594ebec6150a448141a5fa8aedd302a081a31b588de137b5`), rc=1.

| Ayraç (İLAN-5) | Ölçüm (betikten) | Eşik | Dal |
|---|---|---|---|
| n (çift) | **2.115** | ≥ 2.115 | GEÇTİ |
| kalip_koruma | **2.115/2.115** | ≥ %90 | GEÇTİ |
| kalip_çekimlilik | **5** aile | = 5 | GEÇTİ |
| dil karışımı | **0** örnek | 0 (mutlak) | GEÇTİ |
| birebir soru tekrarı | **274 (%12,96)** | ≤ %5 (≤ 106) | **AŞTI — hüküm YETERSİZ** |
| JSON parse hatası | 0 | — | — |

Aile dağılımı (ölçülen): nedir 425 · nasil 425 · ne_zaman 425 · hastalik 425 ·
islev 415 — hedef dağılım birebir tuttu (konu-tekrarı hükümsüz aday: benzersiz
konu 1.055, konu-aile çifti 1.841).

## Aşım bölümü (İLAN-5 dalı: KAYIT YAPILMAZ — data/pedagogy'a birleşim YOK)

Betik: `scratch/t0137/arz_asim_analiz.py` (`487e70fa…`) → `arz_asim_bolumu.json`
(`3eb51b53…`):

- **242 tekrarlı soru → 274 fazla kopya** (aşım bölümü).
- Aile dağılımı: ne_zaman **85** · hastalik **63** · nasil **59** · islev **38** ·
  nedir **29**.
- Kaynak: bağımsız subagent'ların temalarının çakışması (konu-tekrarı
  hükümsüzdü; birebir soru tekrarı ayracı yakaladı — ayraç işliyor).

## Koşum ve araç

- 45 subagent dalı (pilotla aynı araç ve kalıp şablonları — İLAN-2/İLAN-5);
  dosyalar `scratch/t0137/arz_uretim_<aile>_<NN>.jsonl` (30 dosya; 45 dal =
  8×50 hastalik + 1×25 hastalik kalıntı + 8×50 islev + 1×15 islev kalıntı +
  9×50 nedir/nasil/ne_zaman … toplam 44 koşum + 1 kalıntı dalı).
- Sınıflandırıcı "timed out" arızalarında spawn'lar bekletilip yenilendi
  (operatör gözlemi: tüm istekler glm-5.3-flash:cloud'a yazılıyor — İLAN-2
  düzeltmesiyle tutarlı; eşikler değişmedi).
- Her dosya ara doğrulamadan geçti (JSON şema; "TOPLAM= hatalar= YOK"
  deseni) — ara doğrulama betik sayımıdır, subagent beyanı DEĞİLDİR.

## Kanıt digest tablosu (tam sha256)

| Dosya | sha256 |
|---|---|
| `arz_uretim_hukum.json` (hüküm) | `9e6fa9a7904fe673594ebec6150a448141a5fa8aedd302a081a31b588de137b5` |
| `arz_uretim_olc.py` (ölçüm betiği) | `9ea544f2974c4e8e71cd0d9ef60f783da8655f0639442ff75881582c082aa67d` |
| `arz_asim_bolumu.json` (aşım analizi) | `3eb51b53512cdc0a53d5de5c01e3994ea86411015ce2ace14d0dfa696c980489` |
| `arz_asim_analiz.py` (analiz betiği) | `487e70fab1a1a78ee5f28bbc4dad4035f9a8c0cde8011a45c58e2262e1491647` |

Üretim dosyalarının (30 × `arz_uretim_*.jsonl`) tam digest tablosu notta
birleşim fazında verilecektir (külliyat nihai biçimine oturmadan digest
yazılmaz — "donmadan digest yazma" kusur sınıfı).

## Dürüst kayıtlar

1. **İlk ölçüm hükmü YETERSİZ:** ilk koşumda külliyat 2.115'e ulaştı ama
   birebir soru tekrarı %12,96 — aşım bölümü 274. Hüküm betikten; külliyat
   `data/pedagogy/bahcivan_arena.jsonl`'a BİRLEŞTİRİLMEDİ (İLAN-5 dalı).
2. **Konu-tekrarı hükümsüzdü ama birebir soru tekrarı ayracı yakaladı** —
   ayraç işliyor (İLAN-5 ayrac 2 amaçlandığı gibi çalıştı).
3. Subagent raporları model çıktısıdır; tablo betik sayımına dayanır.
4. Ara doğrulamalar dosya başına koşuldu ("TOPLAM= hatalar= YOK" deseni);
   nihai sayım ölçüm betiğindendir.

## İLAN-6 — Yeniden üretim 1. tur (2026-09-26)

330 çift üretildi (9 dal; tamamı betikle ara doğrulandı). Birleşim hüküm
betikten (`arz_birlestir.py`, `73d910c5…`): **BIRLESIM_YETERSIZ** (rc=1) —
girdi 2.445 → 350 çıkarıldı (274 eski aşım + yeni çakışma), birleşim
**2.095 < 2.115 (20 eksi)**; kalıp 2095/2095, dil 0, tekrar 0 — sorun
YALNIZ ARZ. `data/pedagogy` YAZILMADI (dal kuralı işledi). Kök neden ölçümü
(betik): 330 yeni çiftin **74'ü eski külliyatla birebir soru çakıştı**
(ne_zaman 44 · nasil 11 · nedir 10 · islev 9; hastalik 0) + 5 yeni-içi tekrar.
İlan: İLAN-6 damgası (koşum ÖNCESİ).

## İLAN-7 — İkinci yeniden üretim + birleşim (2026-09-26) — HÜKÜM: BIRLESIM_GECTİ

İLAN-7 damgası (koşum ÖNCESİ) ile 120 çift üretildi (5 dal: ne_zaman 40 ·
nasil 30 · nedir 20 · islev 15 · hastalik 15 — tamamı betikle doğrulandı).
Birleşim hüküm betikten:

| Ayraç (İLAN-6/7) | Ölçüm (betikten) | Eşik | Dal |
|---|---|---|---|
| birleşim n | **2.199** | ≥ 2.115 | GEÇTİ |
| birebir soru tekrarı | **0 (%0,00)** | ≤ %5 | GEÇTİ |
| kalip_koruma | **2199/2199** | ≥ %90 | GEÇTİ |
| aile sayısı | **5** | = 5 | GEÇTİ |
| dil karışımı | **0** | 0 (mutlak) | GEÇTİ |
| parse hatası | 0 | — | — |

- **Hüküm: BIRLESIM_GECTİ** (rc=0) — hüküm `bahcivan_birlesim_hukum.json`
  (`27b4b804…`); çıktı `data/pedagogy/bahcivan_arena.jsonl` (2.199 çift,
  `da7bdddd670f157cd126ba0eb499db5316dfdd27b653ec7dd28dcafc20fb6794`) yazıldı
  (kiralanmış writes[]).
- Girdi 2.565 → 366 çıkarıldı (274 eski aşım + İLAN-6/7 çakışmaları); çıktı
  alanları: `soru`, `cevap`, `kaynak` (dosya adı), `aile`.
- Aile dağılımı (birleşimde ölçülen): hastalik 453 · islev 429 · nasil 450 ·
  ne_zaman 428 · nedir 439.

## Kanıt digest tablosu (nihai külliyat — tam sha256)

| Dosya | sha256 |
|---|---|
| `data/pedagogy/bahcivan_arena.jsonl` (nihai külliyat, 2.199 çift) | `da7bdddd670f157cd126ba0eb499db5316dfdd27b653ec7dd28dcafc20fb6794` |
| `scratch/t0137/bahcivan_birlesim_hukum.json` (hüküm) | `27b4b804ae49b09e206eea5a41020c7e1dceabbbf27f98b6827d96308b38d7c2` |
| `scratch/t0137/arz_birlestir.py` (birleşim betiği) | `73d910c54e4a764769ce61612515e37663c3ae0d5699857389acbe355155ba76` |
| 59 × `scratch/t0137/arz_uretim_*.jsonl` (üretim dosyaları) | tam tablo: `scratch/t0137/arz_uretim_digests.txt` |

## Sonuç dalları (güncel)

İLAN-6 **BIRLESIM_YETERSIZ** → İLAN-7 yeniden üretim (koşum ÖNCESİ damga) →
**BIRLESIM_GECTİ** ⇒ arz üretim fazı KAPANDI; **İLAN-3 (derleme: T-0134
D3-istisna + `bin_dekod_dogrula` 0 sapma) açılır** — fine-tune (LM-bedeli
kapısı 0,0230, çıpa 0,1392) sonraki faz.

## Dürüst kayıtlar (ek)

5. **İLAN-6 tamponu çakışmaya gitti:** 330 üretimin 74'ü eski külliyatla
   birebir soru çakıştı (tema çakışması); birleşim hüküm dalı işledi; İLAN-7
   istemlerinde ÇAKIŞMA UYARISI sertleştirildi.
6. **İki dosyada son bayt `\n` eksikti** (nedir_13, islev_13 — Write aracı
   artefaktı); külliyat nihai biçime oturduktan SONRA tek byte (0x0A) eklendi
   (içerik/çiftler değişmedi; `bahcivan_arena.jsonl` digest'i ETKİLENMEZ);
   digest tablosu (`arz_uretim_digests.txt`) düzeltme SONRASI yazıldı.
7. Üretim dosyalarının digest'i külliyat nihai biçime oturmadan YAZILMADI
   ("donmadan digest yazma" kusur sınıfına uymak için).

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · tokenizer +
`src/compiler/**` DOKUNULMAZ · kanonik eval betiği DOKUNULMAZ · bu rapor
ROUGE ölçmez · elle sayı YOK · `git add -A` YASAK · commit operatör kapısıdır.