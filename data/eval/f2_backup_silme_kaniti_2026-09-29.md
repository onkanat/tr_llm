# T-0188 — f2_backup_2026-09-16 silme kanıtı

- **Damga:** 2026-09-29T05:00:43Z (BETİKTEN, `date -u`) · **Yürütücü:** claude
- **Operatör onayı:** AskUserQuestion 2026-09-29 — "Sil (Önerilen)" seçildi (F2 VAZGEÇ,
  T-0180 K5, ile gerekçesi düşen hedefsiz backup).
- **Konum:** `scratch/f2_backup_2026-09-16/` (donmuş-desen DIŞI; `data/**` değildir)
- **Boyut (BETİKTEN `du -sh`):** 53M

## Öncesi tam-digest tablosu (BETİKTEN `shasum -a 256`, koşum `bdmyj9qte`)

| Dosya | SHA-256 | Boyut |
|---|---|---|
| train_balanced_sft.bin | `55863e36b62414c6ada30956b30dc64a5254fe0717582384d43abd5d9cf46a24` | 45M |
| train_balanced_sft.bin.meta.json | `8d34e25eb8b88b779c91dc1f3c646075baf77859e88b847fad8adec45869de50` | 150B |
| train_carpenter_specialization.bin | `13dd81bf700690b78ca15246a2fc1582ee6f8fea587c6d4e78f8ab1e1017b109` | 1.7M |
| train_carpenter_specialization.bin.meta.json | `b0088fd6ae8969f950006aa31e7797b405f27618bba80d825041d898f8013560` | 698K |
| train_chat_balanced.bin | `3b2b9cd3cf3f47147c6c30b1705c812dfa7a8a287788269a6fa73dc94fccbf10` | 6.0M |
| train_chat_balanced.bin.meta.json | `f5ca63fdc62422ee7bd8103ae21be353410b5732cc3cbe12e219b9d19b21b418` | 379B |

## Silme + sonrası teyit

- Silme: `rm -r scratch/f2_backup_2026-09-16` (T-0188 kiralaması altında; üst-dizin
  `scratch/` kiralama-yöntemiyle açık-erişimli).
- Sonrası: BETİKTEN `find scratch -ipath "*f2_backup*"` → 0 satır (post-silme kanıtı
  aşağıda hüküm-JSON'la birlikte ayrıca notlanır).
- Geri-dönüş: YOK (digest-tablosu yalnız içerik-birimlik kanıttır; dosyalar
  git-izlenmeyen backup'tı).