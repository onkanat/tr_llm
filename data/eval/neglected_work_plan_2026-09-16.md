# İhmal Edilen İşler Planı — test + dokümantasyon + commit (16 Eyl 2026)

**Damga:** 2026-09-16T19:40Z · **Yazan:** claude (danışman) · **Yetki:** Onkanat "dinlenme modu" talimatı
(*"test ve dokümantasyonlar ihmal edildi bu görevleri planla; otomatik onay istemeyen işleri yapmakta serbestsin;
git commit ve push gereklerini unutma; KristalLLM dokümantasyonu ve Antigravity'ye güven sınırını biraz esnet"*)

---

## 0. Ölçülen durum
| Ölçüm | Değer |
|---|---|
| Test dosyası / modül-betik | 30 / 112 |
| `train.py`, `train_dpo.py`, `evaluate_sft_benchmarks.py` | **doğrudan testi YOK** (`grep trainer` → 0 dosya) |
| Dokümanların son güncellemesi | hepsi **2026-09-15** (T-0028/T-0029 hizalama turu) — tarih taze, **içerik bugünkü işi yansıtmıyor** |
| Commit bekleyen | **115 kayıt** (28 değişmiş + 87 untracked; **57'si `data/` altında → sahnelenmeyecek**) |
| Son commit | `9e1c1df` (T-0018…T-0031) → o günden beri T-0032…T-0054 birikti |

## 1. TEST BORCU (öncelik sırası)
1. **[YÜKSEK] `train.py`'nin maskeleme döngüsü testi yok** — `for k in range(seq_len): … if token_id == eos_id: is_output=False` bloğu (satır 157-175). Burası **ölçülmüş bir kusurun yeri**: T-0053'te `<EOS>`→ ilk `<PAD>` hedefinin maskesiz kaldığı ölçüldü (185 EOS'un 16'sı). Test: sentetik bir x/y dizisiyle (OUT içeren, EOS'lu, dolgulu) `targets_np`'nin **hangi konumlarda -100** olduğunu birebir doğrula + sızıntı konumunu **pinle** (kusur düzeltilirse test kırmızı yansın ya da bilinçli olarak güncellensin).
2. **[ORTA] `KristalDataset.get_batch`** — 128 hizalama/padding davranışı ve `y = data[i+1:…]` kaydırması test edilmiyor; PAD yoğunluğu (%47,8) buradan geliyor.
3. **[ORTA] `evaluate_sft_benchmarks.py`** — CE/val-loss yolu test edilmiyor; **maskeli/maskesiz ölçek farkı** (T-0053 F4: 1,58 vs 6,85) bu betikte de yanlış okunabilir.
4. **[DÜŞÜK] `train_dpo.py`** — kapi testi var (T-0049) ✓ ama DPO adımı/kayıt yolu test edilmiyor.

## 2. DOKÜMANTASYON BORCU (bugünkü işin yansıması)
Bugün değişen ve dokümanlara **girmemiş** olanlar:
- **Veri hattı:** `train_scientific_sft.py` / `retrain_clean_models.py` / `run_goal_pipeline.py` / `evaluate_sft_benchmarks.py` artık **`data/train_balanced_sft_v2.bin`** okuyor; sözlük **`data/rebuild/vocab_base_32852.json`** (32.852) ve `train.py` modeli **yüklenen sözlükten** kurup `resize_state_dict` ile +36 satır padding yapıyor (`[SOZLESME_UYARI]` ile görünür).
- **Donmuş-yazım kapıları:** üç eğitim girişi + `retrain_clean_models.py` artık **`--allow-frozen-write`** istiyor; kanonik yardımcı `src/llm/frozen_guard.check_frozen_save_path`.
- **PAD kayıp maskesi:** `train.py`'de **varsayılan aktif** (`--no-pad-mask` ile kapanır) → kayıp ölçeği değişti, eski değerlerle kıyaslanamaz.
- **F3 checkpoint'leri:** `f3`, `f3_sft`, `f3_clean`, `f3_clean_sft` (+ kısa koşum `f3a`/`f3b`) ve `kristal_model_f3.pt`'nin **F4 tabanı olmaması** gerektiği.
- **CLAUDE.md'deki bilinen sınırlama** (vocab 31.357/32.816 anlatısı) → bugünkü gerçek: 32.852 + resize.
- **`wiki/log.md`** bugünkü oturumu içermiyor.

## 3. GIT (yol-kapsamlı; `git add -A` ASLA)
- **Sahnelenecek:** `scripts/*.py`, `src/**`, `tests/*.py`, `train.py`, `train_dpo.py`, `data/eval/*.json` (rapor/plan kayıtları), `.agent-bus/notes/T-00*.md`, `wiki/*.md`, dokümanlar, `.gemini/skills/**`.
- **Sahnelenmeyecek:** `data/` altındaki veri artefaktları (57 untracked: `b1_5_splits/`, `_archive/`, `benchmark_evaluation_raw.jsonl`, `.bin/.pt` — hepsi büyük/donmuş), `scratch/` (danışman çalışma alanı).
- **Kimlik:** commit mesajı sonunda `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- **Zamanlama:** T-0054 (F3-clean) doğrulandıktan SONRA → tek tutarlı paket.

## 4. GÜVEN SINIRI (biraz esnetildi — Onkanat talimatı)
- **Antigravity:** görevler artık **2 artefakta kadar / daha uzun koşumlara kadar** açılabilir (önceki "yalnız küçük iş" politikası esnetildi). **Değişmeyen:** spec'te kiralama bölümü zorunlu, `changed_files = writes[]` beyanı, ve her turun **bağımsız doğrulaması** benden.
- **Dokümantasyon:** KristalLLM dokümanlarında **doğrudan düzenleme yetkisi** (önceden her doc değişikliği ayrı görevdi) — ama değişiklikler `data/eval/` altında kayıt altına alınır ve ölçülen sayılar dokümana yazılırken **repo genelinde grep** ile eski değer avı yapılır (T-0028 V1 dersi).

## 5. YÜRÜTME SIRASI
1. **T-0054 izle + doğrula** (F3-clean) → F4 geçiş kararı *(devam ediyor)*
2. **Test 1** (maskeleme döngüsü) — yeni test dosyası, kendi doğrulamamla
3. **Doküman hizalaması** — README/CHANGELOG/USER_GUIDE/CLAUDE.md/wiki/log.md
4. **Commit + push** (yol-kapsamlı)
5. Test 2-3 (get_batch, evaluate_sft_benchmarks) — sırayla
