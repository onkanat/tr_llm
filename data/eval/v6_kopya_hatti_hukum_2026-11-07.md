# HÜKÜM — KOPYA HATTI (RAG belge-kopyalama), 07.11.2026

**Soru:** ANKA'ya RAG belgesini gerçekten kopyalama davranışı ağırlıklarda kazandırılabilir mi?
**Yanıt (ölçüm):** Bu ailede, **oturmuş bir tabanın üstüne sonradan KAZANDIRILAMAZ.** Dört pilot,
hepsi negatif; son ikisi K1 sigortasını tetikleyerek durdu.

## Pilotlar

| # | Tarif | Veri | Süre | K4 `kopya_orani` | K1 (SFT val CE) | Hüküm |
|---|---|---|---|---|---|---|
| — | v5 taban (referans) | — | — | **0,00** | 2,2219 | başlangıç |
| 1 | v5'ten 600 adım, lr 1,5e-4, %20 kopya | morfoz metin | 22 dk | **0,00** | 2,3236 | KALDI |
| 2 | v5'ten 2.500 adım, aynı | morfoz metin | 100 dk | **0,00** | 2,2858 | KALDI |
| 3 | v5'ten %100 kopya, 3.000 pencere × ~1 tur | morfoz metin | 6 dk | ölçülmedi | **2,4593 > 2,45** | K1 erken duruş (149. adım) |
| 4 | aynı, **doğal metin** düzeltmesiyle | doğal metin | 8 dk | ölçülmedi | **2,4666 > 2,45** | K1 erken duruş (149. adım) |

## Kanıtlanan mekanizma

1. **Tek geçiş yetmiyor.** Bin'de 56.904 benzersiz kopya penceresi var; 2.500 adımda her biri
   ~1 kez görüldü. Öğrenme sinyali var ama hizalama yok: eğitim penceresinde belge bölgesi CE
   **6,82 → 1,78**, cevap bölgesi CE **3,91 → 3,33**; greedy üretim hâlâ SFT'nin kısa-etiket
   önselini (`TENSE_PROG`) ve soru metnini basıyor.
2. **Tekrar K1'i yıkıyor.** Pencere akışının %100'ü `<BELGE>` zarfı olunca 149 adımda 0,245 CE
   kaybı; K1 bütçesi (2,45 − 2,2219) **0,228**. Kopyanın ihtiyacı ~8 tur → bütçe öğrenmeden
   önce tükeniyor.
3. **Suçlu metin yüzeyi değildi.** Pilot-3'te kendi hatam vardı (belge/cevap metni `vocab.decode`
   ile morfem-morfoz kurulmuştu: "domates fide si dikim i"); doğal metne çevirdim (pilot-4) —
   sonuç **aynı**. Yani engel biçim egemenliğinin kendisi.
4. Mimari ders (P1) ile tutarlı: davranış devreleri **ön-eğitimde** kurulur; bitmiş okuryazarlığa
   sonradan eklenen biçim onunla doğrudan yarışır.

## Durum / karar bekleyen

- Canlı yığın **v5 tabanında çalışıyor** (`8080 HTTP 200`, `taban_sha256 = 1eded7fa… EŞLEŞME: True`).
  Kopya deneylerinin hiçbiri canlıya dokunmadı; `pure` diskte değişmedi; Qdrant'a yazım yok.
- Deney dosyaları **commit edilmedi**: `scripts/build_copy_korpus.py`, `tests/test_copy_korpus.py`,
  `scripts/k4_kopya_olcum.py`, `scripts/train_v4_ogrenci.py` (PAD maskesi + `--pencere-pad-maske`).
  Testler: `test_copy_korpus` **11 passed**, `test_v4_ogrenci` yeni PAD testleri geçiyor.
- **Açık karar (kullanıcıya soruldu, yanıt bekliyor):**
  - **B (önerilen):** pure'dan 3.000 adım, %20 kopya (3.000 benzersiz pencere × 4 tur) — ~2 sa.
    K1 kapısı bu aşamada anlamsız; mekanizma kanıtı (eğitim-penceresi kopyalama + K4).
  - **A:** pure'dan kopya-karışımlı tam ön-eğitim, 1 epok ≈ 11-13 sa, gateway kapalı.
  - **C:** kopya hattını kapat, ekstraktif RAG (belgeyi alıntı olarak döndür + `koşullandı & 4-gram 0`
    koruması) — ağırlığa dokunmadan.
  - **D:** dur, kararı sonra ver.
- **İşletme notu:** 8 GB MacAir'de gateway açıkken eğitim çekirdek tarafından SIGKILL edildi
  (`memorystatus: … 4607 MiB of 7272 MiB wired, killing`). Uzun koşumlarda gateway kapatılır,
  sonra açılıp sha çıpası doğrulanır.
