# T-0175 disk-dolgu envanteri — SİLME YOK, yalnız ölçüm (BETİKTEN)

- **Tarih:** 2026-09-28T20:12:54Z (BETİKTEN, `date -u`) · **Yürütücü:** claude
- **Yöntem:** `du -sh` + `find -size` + `shasum -a 256` + `stat`. Silme bu turda YOK; aday-tablo operatör-onayına sunulur.

## Genel ölçümler

- `data/` büyük-üçlü (DONMUŞ, aday DEĞİL): `anka_a2.pt` 361M · `anka_base_v2.pt` 356M · `anka_a1r.pt` 356M; `anka_{a2,a1r}_pretrain.bin` 191M×2
- `data/` diğer: `b1_5_splits/` 139M (donmuş) · `rebuild/` 66M · `simulasyon_bellek_export.jsonl` 43M
- `scratch/` toplam **2,9 GB** (506 dosya) · `.agent-bus/` 335M · `data/eval/` 8,3M
- 100M-üstü donmuş-dışı tek dosya: `.agent-bus/checkpoints/notes_and_eval_backup.tar.gz` (330M)

## Silme-aday tablo (kapalı-koşum kalıntıları; her satır BETİKTEN SHA256)

| # | Yol | Boyut | mtime | SHA256 | Sınıf / gerekçe |
|---|---|---|---|---|---|
| 1 | `scratch/t0113/seg_3.pt.opt.pt` | 757.284.041 B | 25 Eyl | `0f1fe683f77e59c0a3f658f96ecfd772d702ae9f99337905afea26eb126d78db` | T-0113 done — optimize checkpoint kopyası; soy-bilgi data/anka_base_v2.pt'te (d0f415f3) + digest kaydında |
| 2 | `scratch/t0113/seg_3.pt` | 378.634.784 B | 25 Eyl | `b1c748f244c3d2c088a9104f05a8d555e139444cb255f7043e69a3c20655801b` | T-0113 done — ham seg checkpoint kopyası |
| 3 | `scratch/t0137/bos_model.pt` | 373.734.923 B | 26 Eyl | `765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9` | G6a done (KABUL_YOK) — koşum-taban kopyası |
| 4 | `scratch/t0138/bos_model.pt` | 373.734.923 B | 27 Eyl | `765e4584e3e5199abbf03c02bc7af2951e4fefa80397507ca855cc1761eaaca9` | **#3 ile BIT-ÖZDEŞ** (digest aynı) — ikiz kopya; en az biri risksiz |
| 5 | `scratch/t0138/wiki_replay.bin` | 199.934.464 B | 27 Eyl | `49f88cb8d13f42cdc7f34e4d1fbd20600294292e516bf58d8b651660b4c4a2e7` | T-0138 done — wiki-replay külliyat kopyası |
| 6 | `scratch/t0129/g5_kosum/m_test.pt` | 373.734.923 B | 26 Eyl | `12c6188ec036658734338f4e8cbfaeee38b22f4ab75219706c9498905c7b1f0a` | G5 koşum ölçüm-kabı kopyası (T-0129 done) |
| 7 | `scratch/t0062_corpus/_train_hashes.npy` | 341.568.984 B | 18 Eyl | `541a3771db4ea2996b2474e3874405855e8cbd86be153656acd119befa3a369e` | T-0062 dönemi hash-önbelleği |
| 8 | `scratch/f2_backup_2026-09-16/train_balanced_sft.bin` | 46.713.074 B | 13 Eyl | `55863e36b62414c6ada30956b30dc64a5254fe0717582384d43abd5d9cf46a24` | **KOŞULLU:** F2 hedef-kümesi kararı AÇIK — karar kapanmadan DOKUNULMAZ önerilir |
| 9 | `.agent-bus/checkpoints/notes_and_eval_backup.tar.gz` | 345.729.613 B | 26 Eyl | `196ab4a63589a305898357d6878091ae92df5df9143ea01e7cd6c90b1d7ae063` | 26 Eyl oturum yedeği (RESUME_SESSION.md halefi taze; içerik notes/+eval zaten git'te ve data/eval'de) |

- **Toplam aday: ~2,79 GB** (koşulsuz #1–7+9 = 2,74 GB; #8 hariç).
- `data/_archive/` (6,1M) ve tüm `data/*.pt`/`*.bin` (donmuş) aday DEĞİL.
- `scratch/t012x…t014x` küçük koşum-log klasörleri (~25M toplam) kanıt-değeri taşır — aday DEĞİL.
- Silme-uygulaması yalnız operatör-onayı ile, satır-satır (her satır silinmeden ÖNCE digest yeniden ölçülür — dosya değişmişse o satır atlanır), T-0174 digest-kaydı sınıfında raporlanır.