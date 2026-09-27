# G6b (T-0138) RAPOR — D3-İSTİSNA DERLEME (İLAN-3) — 2026-09-27 (antigravity)

İlan: `data/eval/berber_g6b_ilan_2026-09-27.md` (ayrı dosya — ILAN ≠ RAPOR).

## Hüküm (betikten, elle sayı yok)

**DERLEME_GECTI** — `scratch/t0138/berber_derleme_hukum.json` (`f656cbd7ba0295d0b23927382f7ef87adea175807ef746ee3952d4004ca39640`), rc=0.

| Ayraç (İLAN-3) | Ölçüm (betikten) | Beklenen / Eşik | Dal |
|---|---|---|---|
| derleme_records | **4.230** | 4.230 (2.115 ham + 2.115 SFT) | GEÇTİ (TAM TAVAN) |
| JSON parse hatası | **0** | 0 | GEÇTİ |
| encode hatası | **0** | 0 | GEÇTİ |
| normalize encode hatası | **0** | 0 | GEÇTİ |
| short_skipped | **0** | 0 | GEÇTİ |
| `bin_dekod_dogrula` | **OK** (4/4 tam uyum) | 0 sapma | GEÇTİ |
| BOS sayımı | **4.230** | == records | GEÇTİ |
| EOS sayımı | **4.230** | == records | GEÇTİ |
| OUTPUT sayımı | **4.230** | == records | GEÇTİ |
| /OUTPUT sayımı | **4.230** | == records | GEÇTİ |
| SFT zarf sayısı | **2.115/2.115** | 2.115 | GEÇTİ |
| ham eşleşme | **2.115/2.115** | 2.115 | GEÇTİ |
| sozluk_giris | **33.114** | 33.114 | GEÇTİ |
| toplam_jeton | **249.838** | — | ort 59,06 jeton/kayıt |

## Çıktı Dosyaları

- İkili veri: `scratch/t0138/berber_arena.bin` (uint16 akış, 249.838 jeton)
- Meta verisi: `scratch/t0138/berber_arena.bin.meta.json` (boundaries + digest çıpaları)
- Adapter: `scratch/t0138/berber_d3_adapter.jsonl` (2.115 satır)

## Kanıt Digest Tablosu

| Dosya | sha256 |
|---|---|
| `scratch/t0138/berber_arena.bin` | `104f96940e0fd07eda33db97de21fddfb4252f87e40311b1372c6d5b9d3f7728` |
| `scratch/t0138/berber_arena.bin.meta.json` | `768ae8fdf621acec42119a9c430734fd2c4ce860d1ba68c82021467d51dca6d2` |
| `scratch/t0138/berber_derleme_hukum.json` | `f656cbd7ba0295d0b23927382f7ef87adea175807ef746ee3952d4004ca39640` |
| `scratch/t0138/berber_d3_adapter.jsonl` | `8c614529d29007f3dbfb9d81d2df0be1333bc68d4a980ca51c33f2c525f6fa0a` |
| `data/pedagogy/berber_arena.jsonl` | `2786e9149b50ada3606ca399b44ee9ad35c2c7c282fb4a00d49f4d6bf46d40a1` |
