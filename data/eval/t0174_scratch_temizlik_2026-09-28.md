# T-0174 scratch temizlik digest tablosu (silme-ÖNCESİ, BETİKTEN)

- **Tarih:** 2026-09-28T20:00:49Z (BETİKTEN, `date -u`) · **Yürütücü:** claude · **Görev:** T-0174
- **Kural:** silme-öncesi tam-digest kaydı (kapanış-kanıtı sınıfı); referans-taraması `grep -rn` tests/src/scripts/modules → `_gecici_mps`/`anka_r5_kos.pid` eşleşme 0; tests'te scratch-dir glob 0.

| Dosya | Boyut | mtime | SHA256 (silme-öncesi) | Gerekçe |
|---|---|---|---|---|
| scratch/anka_r5_kos.pid | 5 B | 2026-09-19 14:17:46 | `add38bad5475be5d7c06ecf3409ea72cfcb192b1c95c3c3a58323533e790a9c9` | Bayat PID 21781 — `ps -p 21781` ölçümü: süreç YOK |
| scratch/_gecici_mps_dogrulama.py | 3.757 B | 2026-09-21 15:06:33 | `9049222855d87bac292ba1f03969bdf93036c32fa0b45ee95207d9831e3eac0e` | "geçici" koşum-yardımcısı, kapalı koşum kalıntısı |
| scratch/_gecici_mps_sonda.py | 3.920 B | 2026-09-21 15:05:59 | `31cee74fb4b6274108d77b2d02ec14f348f021b0d2d72990cb04993552f13450` | "geçici" sonda, kapalı koşum kalıntısı |

- scratch/ toplam 509 dosya, 2,9 GB — büyük gövde (t012x…t014x koşum logları) bu turda DOKUNULMAZ; disk-dolgu temizliği ayrı altyapı turunda operatör-onaylı liste ile yapılır.
- `data/_archive/` (6,1M rag_pilot_invalidated) ve `.agent-bus/checkpoints/` (330M tar, 26 Eyl oturum yedeği) DOKUNULMAZ.