# T-0152 — `data/qdrant_db/` disk-artefakt temizliği: SİLME-ÖNCESİ KAYIT

**Damga (koşum-öncesi, betikten değil `date -u`):** 2026-09-28T06:37:27Z
**Görev:** T-0152 (bus; kiralamalar: `data/qdrant_db/`, `data/eval/`,
`.agent-bus/notes/` — ÜST-DİZİN).

## Meşruiyet

T-0148 Tur-B 4B onarımının kapanış-maddesi: VectorMemory kurucu default
`storage_path='data/qdrant_db'` **KALDIRILDI** (`storage_path: Optional[str]=None`
— T-0148 commit `80bd26b`). Diskte kalan artefaktlar P4 koşumlarından
(dolaylı-yazım kanıtı, P4 kusur-B) ve kod artık buraya YAZMAZ. Dizin
**git-IGNORED** (`git check-ignore` = IGNORED ölçüldü) — silme-kanıtı yalnız
bu digest-kaydıyla yaşar (T-0146 kanarya-silme kalıbı: silme-ÖNCESİ tam
digest+mtime kaydı; rm sonrası 'no matches' teyidi).

## Silme-öncesi tam envanter (ölçüldü, `shasum -a 256` + `stat -f '%m %z'`)

| Dosya | SHA-256 | mtime (epoch) | Boyut (B) |
|---|---|---|---|
| `data/qdrant_db/.lock` | `67a987c432005e3afcc13a871b3ddabaeee0f4504058da65b0769016d3083b44` | 1788979059 | 13 |
| `data/qdrant_db/meta.json` | `4c35797760c04484b6381f237d8bd7ec13b155857fafb9052dc4e289ade51e4b` | 1789410794 | 2176 |
| `data/qdrant_db/collection/simulasyon_bellek/storage.sqlite` | `79c76eb766d3e582bd1128dd855b1d34a7a02eebc1356f1598331968941373e4` | 1789410794 | 16384 |
| `data/qdrant_db/collection/test_temp_coll/storage.sqlite` | `6862b8a9abdf270b38adafbd32d1a022fc08bf1ec63cfe3162c187af1471b492` | 1789112645 | 12288 |
| `data/qdrant_db/collection/kristal_bellek/storage.sqlite` | `f38f7086a1c7396af9b2359393352ed4a31429cf6ff428b7b1257c25711fcb6d` | 1789112561 | 237568 |
| `data/qdrant_db/collection/muhakeme_bellek/storage.sqlite` | `6862b8a9abdf270b38adafbd32d1a022fc08bf1ec63cfe3162c187af1471b492` | 1789235371 | 12288 |

Toplam: 6 dosya, **304.437 bayt (~304K)**. Not:
`test_temp_coll` ↔ `muhakeme_bellek` sqlite'ları bit-özdeş (boş-şablon sqlite).

## DOKUNULMAZLAR (silme-etkisi DIŞI)

- **Canlı sunucu 192.168.1.9:6333** — silme YALNIZ yerel-disk; canlı
  `kristal_bellek` 36-point SABİT (K6 çıpası; salt-okuma kanıtı koşum-sonrası
  yeniden ölçülür).
- `data/**` diğer tüm yollar; `data/eval/` 446 dosya (yanlış-silme teyidi
  koşum-sonrası yeniden sayılır); `data/pedagogy_canonical/**` frozen.

## Silme adımı (operatör onayı sonrası)

`rm -rf data/qdrant_db/` → `find data/qdrant_db` 'no matches' + `data/eval`
dosya-sayısı teyidi + canlı 36-point teyidi → kanıt-dosyası SONUÇ bölümüyle
tamamlanır (aynı dosyaya, silme-sonrası ek).

## SONUÇ (silme-sonrası; operatör onayı alındı)

Operatör onayı: AskUserQuestion "Sil — kanıtla kapat" (28 Eyl 2026).
`rm -rf data/qdrant_db/` çalıştı. Kanıt-teyitleri (silme-sonrası ölçüldü):

- `find data/qdrant_db` → **"No such file or directory"** (rc=1 — no matches;
  T-0146 kalıbı teyidi).
- `data/eval` dosya-sayısı → **447** (silme-öncesi 446 + bu kanıt-dosyası;
  yanlış-silme YOK).
- Canlı sunucu 192.168.1.9:6333 `kristal_bellek` → **green, points 36,
  indexed 36 SABİT** (yerel-disk silmesi canlıyı etkilemedi — DOKUNULMAZLAR
  bölümü teyidi).

**Hüküm: T0152_TEMIZLIK_TAMAM** — 6 artefakt silindi, kanıt bu dosyada
(silme-öncesi tam digest-tablo + silme-sonrası teyitler).