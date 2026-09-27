# G6b (T-0138) İLAN-3b — BERBER FİNE-TUNE KOŞUMU — 2026-09-27 (antigravity)

ILAN ≠ RAPOR: rapor `data/eval/berber_g6b_finetune_sonuc_2026-09-27.md` (AYRI dosya).
Hükümler BETİKTEN, elle sayı YOK.

## 1. Taban ve Checkpoint Disiplini

- Taban: `data/anka_base_v2.pt` (`d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`, MÜHÜRLÜ 2.0-sealed).
- `data/*.pt` yazımı YASAKTIR.
- Koşum çıktısı: `scratch/t0138_g6b_kos/` (kiralanmış). Checkpoint silme OPERATÖR KAPISIDIR.
- Taban-çıpa zinciri: marangoz heldout ROUGE çıpası **0,1392** (kapı: 0,0230).

## 2. Carve ve Veri

- **Oran %10 (212/2.115 çift), tohum 42, deterministik.**
- Heldout: `scratch/t0138/berber_heldout.jsonl` (212 satır, D3 şeması: `{"instruction": ZARF, "input": <soru>, "output": <cevap>}`).
  sha256: `d063ec27f66e8cecea348706c36e405a3f9f540a1b9d9505e6eb16cfb555abb7`.
- Eğitim adapter'ı: `scratch/t0138/berber_egitim_adapter.jsonl` (1.903 satır, D3-istisna şeması).
  sha256: `b775d0884ed7747eb00812bfffc804721d7533db2812ab7d9163e50570077963`.
- Eğitim bin: `scratch/t0138/berber_egitim.bin` (3.806 kayıt, 224.994 jeton, uint16 akış).
  sha256: `0306b2e2e0204e2391f0617184acecdd5fca04e0347a3b91ecb5c0625850df85`.
- Maske oranı tanısı: L1_ESIK = **4.034** (boş pencere 0/400).

## 3. Berber Taban-Çıpa

- Kanonik kap `scripts/evaluate_carpenter_anka.py` (`--model data/anka_base_v2.pt --heldout scratch/t0138/berber_heldout.jsonl --train-source scratch/t0138/berber_egitim_adapter.jsonl --ceket-ekseni --output scratch/t0138/berber_taban_cipa.json --device mps`).
- Ölçülen Berber taban ROUGE-L: **0,0302** (medyan 0,017, DECOMP 0,0397; sonda sha `708bb03f…`).
- A9 ARTİS çıpası bu değere sabitlenmiştir: Çıpa = **0,0302**, Kapı = **0,0230**.
  Koşul: `ft_ROUGE - 0,0302 >= +0,0230`.

## 4. Modüler Ceket Eğitimi (train_module.py)

- **Eğitici:** `train_module.py` (Değişmez Taban + Değişken Ceket mimarisi).
- **Taban:** `data/anka_base_v2.pt` (%100 DONUK; 93.424.986 parametre sabit).
- **Veri (Salt Berber):** `scratch/t0138/berber_egitim.bin` (3.806 kayıt, 224.994 jeton, D3-istisna SFT).
  *Operatör ve Başmühendis ilkesi:* Marangoz, bahçıvan veya diğer dikeyler KARIŞTIRILMAZ.
- **Hedef Katmanlar:** `attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2,lm_head` (37 katman).
- **Parametreler:** r=16, alpha=32, lr=2e-4, batch=8, block=128, steps=1000, seed=43, device=mps.
- **Çıktı Modülü:** `scratch/t0138/berber.mod.pt` (LoRA Yetenek Modülü, ~1,87M parametre).

## 5. Ölçüm Ayracı Seti

| # | Ayraç | Ölçüm | Eşik |
|---|---|---|---|
| A9 | Berber Yetenek ARTIS | Berber Heldout | ft ROUGE - çıpa (0,0302) ≥ +0,0230 |
| A11 | A-ekseni CE (Dil Tabanı) | Wikipedia Dilimi | CE artış ≤ +%10 |
| A12 | B top-1 (Kararlılık) | Kanonik Doğrulama | düşüş ≤ 5,0 (Wilson) |
| A13 | Tutarsızlık / Ezber | Berber Heldout | tutarsızlık < %5 · ezber < %10 |

- **ÖLÇÜM KABI:** `scripts/modul_ile_olcum.py`
  (`--module scratch/t0138/berber.mod.pt --model data/anka_base_v2.pt --baseline data/anka_base_v2.pt --heldout scratch/t0138/berber_heldout.jsonl --train-source scratch/t0138/berber_egitim_adapter.jsonl --ceket-ekseni --output scratch/t0138/berber_ft_berber_sonda.json --device mps`).

## 6. Kapanış

Modül `scratch/t0138/berber.mod.pt` doğrulama sonrası korunur/arşivlenir.
Digest tablosu raporda dondurulur.
Commit `git add -A` YASAKTIR.
