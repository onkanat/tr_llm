# G6a (T-0137) RAPOR — DERLEME (İLAN-3) — 2026-09-26 (claude)

ILAN ≠ RAPOR: ilan `bahcivan_g6a_ilan_2026-09-26.md` İLAN-3'te koşum ÖNCESİ
damgalı; hüküm BETİKTEN (`scratch/t0137/bahcivan_derle.py`, elle sayı YOK).

## Hüküm

**DERLEME_GECTİ** — `scratch/t0137/bahcivan_derleme_hukum.json` (rc=0).

| Ayraç (İLAN-3) | Ölçüm (betikten) | Eşik | Dal |
|---|---|---|---|
| JSON parse hatası | **0** | 0 | GEÇTİ |
| encode hatası | **0** | 0 | GEÇTİ |
| normalize encode hatası | **0** | 0 | GEÇTİ |
| kayıt sayısı | **4.398** | ≤ tavan 4.398 (2×2.199) | GEÇTİ (tavan TAM) |
| SFT zarf eşleşme | **2.199/2.199** | = 2.199 | GEÇTİ |
| ham kayıt eşleşme | **2.199/2.199** | = 2.199 | GEÇTİ |
| `bin_dekod_dogrula` | **4/4 OK** (BOS/EOS/OUTPUT//OUTPUT) | 0 sapma | GEÇTİ |
| sozluk_giris | **33.114** | = 33114 | GEÇTİ |

- Toplam jeton: **399.212** (kayıt başına ort. 90,77).
- Kütüphane: `src.llm.dataset_compiler` IMPORT (D3-istisna
  `tokenize_specialization_jsonl` + `bin_dekod_dogrula`) — kanonik kod
  DOKUNULMAZ; tokenizer kurulumu `vocab_anka_r1_33114` + `roots.tsv` +
  `literal_entity_mode=True` (İLAN-3 beyanı birebir işlendi).
- Kayıt yapısı: 2.199 ham (adapter JSON'un encode'u) + 2.199 normalize SFT
  (`{"instruction": "Bahçıvan uzmanı olarak cevapla.", "input": <soru>,
  "output": <cevap>}` — q derleyicinin kendi `:`-çıkarımıyla, betik yeniden
  kurulumuyla doğrulandı).

## Kanıt digest tablosu (tam sha256; bin nihai biçime oturduktan sonra)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/bahcivan_arena.bin` (derleme çıktısı, 4.398 kayıt) | `bd5f19606aaae573bda8d3a87cfa79b11ee06b2f9a6305da07842d63d39e76b9` |
| `scratch/t0137/bahcivan_arena.bin.meta.json` (boundaries + digest çıpaları) | `03525d888fe6c9be35f51317f3c3c2bab99848912776fae0116de4f5313e3f93` |
| `scratch/t0137/bahcivan_derleme_hukum.json` (hüküm) | `1c2592223458826cc8ee88a377cfdeda4f04ccbfdd813fdf4a2b62d86428ba26` |
| `scratch/t0137/bahcivan_derle.py` (derleme betiği) | `348b939df1da206f71ea9449bf72aea69b1a084bdc31dbc2da94f4bb1faa2cc6` |
| `scratch/t0137/bahcivan_d3_adapter.jsonl` (ara dosya, 2.199 satır) | `3287e20c7b7cdcf2f446d822e4cc2f40458f37893edc559851fb616eb6a410e8` |
| records (betikten, kayıt-başına JSON + `\n` akışı) | `7482db8c77d2903171a22fcf910aba332299c630b691b866d3094a32cc816e83` |

## Dürüst kayıtlar

1. **İlk koşum DERLEME_SAPMA (rc=2) — betik içi ölçüm yöntemi kusuru:** İLAN-3
   ayracı 4'ün ilk betik uygulaması zarf sayımını `vocab.decode` üzerinden
   yaptı; morfem-tabanlı decode zarf metnini birebir geri vermediği için sayım
   `0` yazdı (derleme kayıtları kendisi kusursuzdu: 4.398 kayıt, bin_dekod 4/4
   OK). Betik, ayracı DEĞİŞTİRMEDEN, zarf sayımını derleyicinin kendi
   `:`-çıkarımıyla yapısal yeniden kurulum olarak yeniden yazıldı (SFT kaydı
   `tokenize_specialization_jsonl`'ın normalize şemasıyla birebir beklenen
   biçimden üretilip kayıt akışında aranır). Hüküm İKİNCİ koşumda betikten
   yenilendi; iki hüküm dosyası da aynı yoldadır (sonuncusu geçerli).
2. **Ara hüküm dosyası hükümlü koşumda yazıldı** (DERLEME_SAPMA kaydı kanıta
   düştü) — rapor iki koşumu da beyan eder; nihai hüküm rc=0 koşumundan.
3. Derleme çıktıları data/ donmuş yoluna YAZILMADI (`scratch/t0137/` — kiralanmış
   writes[]); `check_output_path` gerekmedi (hedef donmuş değil).
4. SFT kayıtları derleyicinin varsayılan zarfını ("Ahşap uzmanı olarak
   cevapla.") İÇERMEZ — İLAN-3 zarfı "Bahçıvan uzmanı olarak cevapla."
   parametresiyle geçildi; ham kayıtlar adapter JSON'un encode'udur.

## Sonuç dalları

İLAN-5/6/7 arz zinciri BIRLESIM_GECTİ → İLAN-3 derleme **DERLEME_GECTİ** ⇒
**İLAN-3b (fine-tune) açılır** — ayrı İLAN damgası + operatör kapıları
(data/*.pt yazım, taban DONUK, LM-bedeli kapısı 0,0230, çıpa 0,1392; MPS koşum
sandbox dışı; `--vocab` ZORUNLU 33114; `--pretrain` ZORUNLU).

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · `src/llm/tokenizer.py`
+ `src/compiler/**` + `src/llm/dataset_compiler.py` DOKUNULMAZ (import yalnız) ·
`ROL_ZARFI` DOKUNULMAZ · bu rapor ROUGE ölçmez · elle sayı YOK · `git add -A`
YASAK · commit operatör kapısıdır.