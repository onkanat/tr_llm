# KANARYA SİLME KANITI — `data/train_future_finetune.bin` (+meta) — T-0146

**Damga:** 2026-09-27 (UTC) · **Yürütücü:** claude · **Onay:** operatör kararı
27 Eyl 2026 ("kanaryanın silinmesi" — P5 kapanış sonrası; commit `ec0d01e` onay-turu)

## Gerekçe

`data/train_future_finetune.bin` (12M) + `.meta.json` — **13 Eyl 2026 09:57**
damgalı, **P5'ten 2 hafta önceki** kanal-testi artefaktı. T-0145 P5
koşumları (3 koşum, 27 Eyl 2026) bu dosyaya **BİR BAYT yazmadı** —
mtime+digest **ÖNCE==SONRA çıpasıyla** kanıtlı (koşum-3: 1789282659.99 ==
1789282659.99; P5 raporu §7.2 madde-3; notes; commit `ec0d01e`).
`data/**` donmuş-salt-okunur desende; silme bu kanıt-dosyasıyla
BELGELENEREK operatör kararıyla yapılır (kanıt önce taşınır kalıbı —
T-0137 kapanış-dersi).

## Silme-ÖNCESİ ölçüm (27 Eyl 2026; bu dosyanın yazılmasından hemen önce)

| Dosya | Boyut | mtime | sha256 |
|---|---|---|---|
| `data/train_future_finetune.bin` | 12M | 1789282659 (13 Eyl 09:57:39) | `80340a98b23381098bea27ec831e0c231e2efc8de4eca31396209fdfe15595ee` |
| `data/train_future_finetune.bin.meta.json` | 178B | 1789282660 (13 Eyl 09:57:40) | `b9eaf68267b990818d4b95cd5e31340c7697fa95acb6d4a3b157cf3c124504d7` |

## Silme kapsamı

- **SİLİNİR:** yalnız yukarıdaki İKİ dosya.
- **DOKUNULMAZ:** `data/**` altındaki her başka yol (donmuş desenler
  `data/*.pt`, `data/*.bin`, `data/vocab.json` vb. dahil); `data/eval/`
  P1–P5 artefaktları; Qdrant sunucu (192.168.1.5:6333; 10 koleksiyon);
  `data/future_train_vector.jsonl` 0-bayt çıpa `e3b0c442…`.
- Bu dosyalar git-izlenmeyen (`data/*.bin` gitignore-sınıfı); silme
  commit-içeriğini değiştirmez — kanıt-dosyası ayrı commit-onayıyla.

## Silme-SONRASI teyit (27 Eyl 2026)

- `ls data/train_future_finetune*` → **no matches found** (iki dosya da YOK)
- `data/eval/` → **410 dosya, sağlam** (P1–P5 artefaktları dahil)
- Qdrant sunucuya dokunulmadı; GERÇEK kanal
  `data/future_train_vector.jsonl` dokunulmadı (0-bayt çıpa).
- Silme komutu: `rm data/train_future_finetune.bin data/train_future_finetune.bin.meta.json`