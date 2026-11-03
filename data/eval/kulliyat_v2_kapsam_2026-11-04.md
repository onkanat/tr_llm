# Külliyat v2 Birleşmesi ve Kapsam Ölçümü — T3.3 (2026-11-04)

## İLAN (ölçümden önce; eşikler burada donar, sonradan DEĞİŞTİRİLMEZ)

**Görev:** Faz-3 T3.3 — külliyat birleşmesi + kapsam ölçümü.
**Yöntem:** İLAN (bu bölüm) → koşum → HÜKÜM (eşiklerle karşılaştırma) → RAPOR.

### Girdi varlıkları (ilan anındaki durum)

| Varlık | Boyut | Not |
|---|---|---|
| `data/raw_corpus.txt` | 3.093.507 bayt · 27.162 satır | her satır kaynak olarak alınır |
| `data/kulliyat_v2/gts_definitions.txt` | 7.446.794 bayt · 128.327 satır | `MADDE<TAB>anlam`; yalnız ANLAM sütunu kullanılır (olduğu gibi) |
| `data/train_wiki.bin` | 36.011.478 bayt = 18.005.739 uint16 token | T3.2 koşumu; ADIM 0'da resume ile 22M bütçeye yükseltilir |
| `data/rebuild/vocab_anka_r1_33114.json` | 33.114 ID | frozen anka vocab; encode yeni token eklemez |

### Komutlar

```bash
# ADIM 0 — wiki resume (bütçe 18.005.739 -> 22.000.000 token)
venv/bin/python scripts/prepare_wiki_dataset.py --budget 22000000

# ADIM 1 — birleşme: metin kaynakları -> text bin; text ++ wiki -> kulliyat_v2.bin
venv/bin/python scripts/assemble_kulliyat_v2.py

# ADIM 2 — kapsam ölçümü (tek geçiş, np.bincount)
venv/bin/python scripts/assemble_kulliyat_v2.py --coverage-only
```

### Birleşme tarifi (dondurulmuş)

- Metin akışı: `raw_corpus.txt` her satır + `gts_definitions.txt` ANLAM sütunu (toplam 155.489 giriş; sekmesiz GTS satırı: 0).
- Satır-bazında tam yinelenenler tek `set` ile atılır (kaynaklar arası dahil); boş satırlar atılır.
- Tokenizasyon: `KristalTokenizer.encode` (frozen vocab) — satır başına `<BOS> … <EOS>` sınırları encode içinde hazır.
- Çıktı: `data/kulliyat_v2/kulliyat_v2.bin` (uint16, `text ++ wiki` sırası; dosya boyutu = toplam token × 2).
- Kapsam ölçümü: `kulliyat_v2.bin` üzerinde tek geçiş `np.bincount`; 33.114 ID'nin en az 1 kez görülenleri; 0 kez görülenler `data/kulliyat_v2/zero_gradient_ids.json`'a.

### DONMUŞ EŞİKLER (sonradan değiştirilmez)

| Kapı | Eşik | Taban (bilgi) |
|---|---|---|
| G1 — toplam külliyat | `kulliyat_v2.bin` ≥ **25.000.000 token** | plan: wiki 22M + raw ~0,7M + GTS ~1,5M ≈ 26M |
| G2 — vocab kapsamı | 33.114 ID'nin ≥ **%50**'si en az 1 kez görünür | eski 2,5M token'lık corpus'ta %14,7 görünüyordu |

**Tutarsızlık protokolü:** G2 tutmazsa wiki resume bir kez daha **+5M** bütçeyle (27M) çalıştırılır, birleşme+kapsam yeniden ölçülür; yine tutmazsa durulur ve "kaynak eklenmeli" diye raporlanır. (Wiki token ekleme G1'i de aynı yönde besler; retry hakkı kapılardan hangisi tutmazsa kullanılır.)

## HÜKÜM (ölçüm → donmuş eşiklerle karşılaştırma)

**Koşum 1 (wiki bütçe 22M — ADIM 0 planı):**

| Kapı | Ölçüm | Eşik | Sonuç |
|---|---|---|---|
| G1 toplam token | 24.124.810 | ≥25.000.000 | **GEÇİLEMEDİ** (−875.190) |
| G2 vocab kapsamı | %95,34 (31.572/33.114) | ≥%50 | GEÇTİ |

G1 tutmadı: görev girdisindeki "raw ≈2,5M token" beyanı ölçümle tutmadı (gerçek ~679K; bkz. RAPOR), İLAN'daki plan toplamı (22 + 0,7 + 1,5) de 25M'nin altında kalıyordu — açık öngörülebilirdi. Protokolün izin verdiği **tek +5M retry** G1 için kullanıldı: wiki bütçesi 27M'ye yükseltildi, birleşme ve kapsam yeniden ölçüldü.

**Koşum 2 (retry — wiki bütçe 27M):**

| Kapı | Ölçüm | Eşik | Sonuç |
|---|---|---|---|
| G1 toplam token | **29.135.254** | ≥25.000.000 | **GEÇTİ** |
| G2 vocab kapsamı | **%95,58** (31.649/33.114) | ≥%50 | **GEÇTİ** |

**SON HÜKÜM: G1 GEÇTİ · G2 GEÇTİ — külliyat v2 kapısı tek retry sonrası AÇILDI.** Eşikler değiştirilmedi; retry hakkı tükendi.

## RAPOR (ölçümler ve kanıtlar)

### ADIM 0 — wiki resume (`scripts/prepare_wiki_dataset.py`)

| Koşum | Komut | Süre | Hız | Sonuç |
|---|---|---|---|---|
| resume 1 | `venv/bin/python scripts/prepare_wiki_dataset.py --budget 22000000` | 5,1 dk | 13,2K tok/s | 18.005.739 → **22.003.183 token** (16.015 makale) |
| retry | `venv/bin/python scripts/prepare_wiki_dataset.py --budget 27000000` | 6,2 dk | 13,4K tok/s | 22.003.183 → **27.013.627 token** (20.990/631.148 makale) |

`data/train_wiki.bin`: 54.027.254 bayt = 27.013.627 uint16 token. Akışlı koşum, RSS düşük; parça-0'ın sıralı uzun-makale bölümünde kalındı (~370M token'lık tam külliyatın bütçeyle kesilmiş öneki).

### ADIM 1 — birleşme (`scripts/assemble_kulliyat_v2.py`)

Metin fazı (RESUME'lu; 2,9 dk; ~12,4K tok/s):

| Kaynak | Satır | kept | dup | blank | Token |
|---|---|---|---|---|---|
| raw_corpus.txt | 27.162 | 15.862 | 1.816 | 9.484 | **679.246** |
| gts_definitions.txt (yalnız ANLAM sütunu) | 128.327 | 116.415 | 11.912 | 0 | **1.442.381** |
| metin toplam | 155.489 | 132.277 | 13.728 | 9.484 | **2.121.627** |

Notlar: raw_corpus.txt'nin "~2,5M token" beyanı ölçümle tutmadı — satırların %35'i boş, %6,7'si birebir yinelenik; kalan 15.862 satır 679K token. GTS 128.327 satırın tamamında sekme vardı (anlam sütunu olduğu gibi alındı, madde başı eklenmedi). Satır bazında tam yinelenenler tek set ile atıldı (kaynaklar arası dahil). Her satır `KristalTokenizer.encode` ile `<BOS>…<EOS>` sınırlarıyla kodlandı (frozen anka vocab; uint16 güvenli).

Birleşme: `text (2.121.627) ++ wiki (27.013.627)` → **`data/kulliyat_v2/kulliyat_v2.bin` = 58.270.508 bayt = 29.135.254 token** (uint16; boyut = token×2 betikte doğrulandı).

### ADIM 2 — kapsam (tek geçiş `np.bincount`)

| Koşum | Toplam token | Görülen ID | Sıfır ID | Kapsam | G2 eşik ≥%50 |
|---|---|---|---|---|---|
| koşum 1 | 24.124.810 | 31.572 | 1.542 | %95,34 | GEÇTİ |
| koşum 2 (final) | 29.135.254 | 31.649 | **1.465** | **%95,58** | **GEÇTİ** |

Tabanla karşılaştırma: eski 2,5M token'lık corpus'ta %14,7 görünürken 29,1M token'da %95,6 (~6,5 kat). Sıfır-ID listesi: `data/kulliyat_v2/zero_gradient_ids.json` (1.465 ID; `id=1 <PAD>` doğal üye — veri akışında hiçbir zaman görünmez).

### Çıktı varlıkları

| Dosya | Boyut | İçerik |
|---|---|---|
| `scripts/assemble_kulliyat_v2.py` | 12.713 bayt | yeni betik: birleşme + `--coverage-only` kapsam |
| `data/kulliyat_v2/kulliyat_v2.bin` | 58.270.508 bayt | final külliyat (29.135.254 token) |
| `data/kulliyat_v2/kulliyat_v2_text.bin` | 4.243.254 bayt | metin katmanı (2.121.627 token) |
| `data/kulliyat_v2/zero_gradient_ids.json` | 16.172 bayt | 1.465 sıfır-ID listesi |
| `data/kulliyat_v2/kulliyat_v2_stats.json` | 928 bayt | birleşme istatistikleri |
| `data/kulliyat_v2/kulliyat_v2_kapsam.json` | 334 bayt | kapsam ölçümü özeti |
| `data/eval/kulliyat_v2_kapsam_2026-11-04.md` | bu rapor | İLAN→HÜKÜM→RAPOR |

Aritmetik doğrulama: 679.246 + 1.442.381 = 2.121.627 ✓ · 2.121.627 + 27.013.627 = 29.135.254 ✓ · 58.270.508 = 29.135.254 × 2 ✓ · 31.649 + 1.465 = 33.114 ✓

Commit YAPILMADI; tüm çıktılar çalışma ağacında untracked.