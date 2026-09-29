# T-0177 İLAN PROTOKOLÜ: Külliyat-Düzeyi Eğitilebilirlik Kapısı (T-0075 kuralının uygulanması)

- **Tarih:** 2026-09-28T20:20:27Z (BETİKTEN, `date -u`)
- **Görev:** T-0177 · **Yürütücü:** claude · **Durum:** İLAN AŞAMASI (Edit/koşum ÖNCESİ)

---

## 1. Amaç ve ÖLÇÜLMÜŞ gerekçe

T-0075 (18 Eyl) + bedel-kaydı (21 Eyl): `train.py`'daki "SFT maskelemesi 0 hedef ⇒
`RuntimeError`" kapısı **parti başına** ateşler; karışık külliyatta
`P(ateşleme) = (1-p)^B` — `train_future_finetune.bin` (p=%7,23, B=8) → **%57,5 ölüm**;
wiki-replay vakasında ise kapı ölü (P(8/8 pencere `<OUTPUT>`-suz) ≈ 1,5e−5) ve
külliyatın **%24,85'i** sessizce eğitilemez kaldı (hüküm geri çekildi).
İlan edilen kural o tarihte sabitlendi: **karar külliyat düzeyinde verilir** —
koşum başında örnekleyip `p≈0` ise DUR, `p>0` ise parti-başına **say + uyar + atla**.
Bu görev o kuralın uygulanmasıdır. Güncel kod-çıpası (BETİKTEN grep): per-batch
`RuntimeError` `train.py` SFT-dalı (MASKE bloğu) — değişecek; PAD-kapısı DOKUNULMAZ.

## 2. Sabitler (koşum-öncesi ilan; yumuşatma yok)

- **SEED = 1777** · **N = 400** örneklem pencere (deterministik `torch.Generator`).
- **K1 (koşum-başı, yalnız SFT dalı — `pretrain=False`):** N pencere örnekle,
  `mask_prompt_targets` (+`maske_pad_hedefleri`, mevcut pad_mask_active ile) uygula;
  ölç: eğitilebilir hedef oranı `k/(N·block_size)` ve 0-hedef pencere oranı `p`.
  `[kulliyat]` satırı YAZDIRILIR (hedef oranı + p + N + seed).
- **K2 (p == tam 0):** `RuntimeError` — düz-metin külliyatta SFT ölüdür (T-0073 sınıfı);
  mevcut fail-closed karakterinin koşum-başına taşınması. Yüksek sesle durur.
- **K3 (0 < p):** devam; parti-başına 0-hedef partisi **sayılır**, `[uyari]` satırı
  basılır, **parti ATLANIR** (gradyan yok), koşum sürer; koşum sonunda toplam sayım
  raporlanır. Eski parti-başına `RuntimeError` bu daldan KALDIRILIR.
- **K4 (PAD-kapısı dokunulmaz):** "PAD maskelemesi bu partide HİÇBİR hedef bırakmadı"
  kapısı aynen kalır ( hizalama-dolgusu olayı; külliyat-anlamı taşımaz).
- **K5 (vekil):** `has_output` (`<OUTPUT>` id üye-testi) ↔ gerçek maske sonrası
  hedef>0 → örneklemde **400/400** uyum beklenir (T-0075 vekil-kanıtı).
- **K6 (davranış-koruma):** `--pretrain` dalı bit-özdeş (kapı SFT-dalı); loss-report
  formatı değişmez; mevcut pytest koşumları (`test_train_args`, `test_train_optimizer_state`,
  `test_train_scheduler`) tamamı `--pretrain` (BETİKTEN grep: 10/10 + 3) → etkisiz.
- **K7 (pytest):** baseline 309 + 3 yeni test (tests/test_kulliyat_kapisi.py):
  (i) p=0 külliyatta RuntimeError (mutasyon: düz-metin .bin); (ii) 0<p'de say+atla
  kararı; (iii) vekil 400/400 sentetik karışım. Yeni düşen ∅.
- **K8 (canlı-dokunulmazlık):** Qdrant/gateway 0 istek; donmuş data/** okuma-dışı
  (yalnız testlerde tmp); `src/llm/dataset.py`, `scripts/train_step_demo.py`
  mask gövdesi DOKUNULMAZ.

## 3. Yüzey

- `train.py`: koşum-başı kapı fonksiyonu + SFT-dalına bağlama + parti-bası sayım.
- `tests/test_kulliyat_kapisi.py`: YENİ.
- `data/eval/t0177_kulliyat_kapisi_hukum_2026-09-28.json`: hüküm BETİKTEN.