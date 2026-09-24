# T-0106 FAZ A SONUÇ — DECOMP ROUGE ölçüt geçişi (tanısal, hüküm dışı)

**Tarih:** 2026-09-24 · **Görev:** T-0106 · **İlan:** `anka_t0106_fazA_decomp_rouge_ilani_2026-09-24.md`
**Kapsam:** DECOMP ikinci temsilin ölçüm kabına geçişi (aday beyanı) + r2 tavan ölçümü + duman doğrulaması.

---

## 1. Kod değişiklikleri (uygulandı)

| Dosya | Değişiklik |
|---|---|
| `scripts/evaluate_carpenter_anka.py` | `yuzey` fallback **beyanlı** (`except → yuzey=gm; decomp_istisna += 1`; sessiz fallback kaldırıldı); `rouge_l_decomp_ort/_medyan/_std` + `decompile_istisna` + `yuzey_unk_sayi` metrik alanları; **`ham_gm`** alanı (TÜM n örnek gm dizisi — P3/P4'te kaybolan yeniden-ölçülebilirlik); `TAVAN_ROUGE_DECOMP`/`ESIK_ROUGE_DECOMP_ADAY` sabitleri (eşik bloğu sonrası); `esikler` bloğuna `rouge_decomp_aday`/`tavan_rouge_decomp`. **Hüküm DEĞİŞMEZ:** `rouge_gec` RAW `ESIK_ROUGE=0,3221`. |
| `scripts/olcum_kabi.py` | ESIK_REFERANSLARI'na `(83, "TAVAN_ROUGE_DECOMP", 0.9509)`, `(84, "ESIK_ROUGE_DECOMP_ADAY", 0.7988)`; K11'e `(84, …, 0.9509)` düşmeli kanaryası. |
| `tests/test_olcum_kabi_kapilari.py` | (d) bloğu sabit-değer beyanı. 11 passed. |
| `scripts/anka_karsilastirma_tablosu.py` | `rouge_decomp` kolonu (tanısal). |
| `tests/test_b1_5_split_leakage.py` | `test_canary_import_examples_raises_on_missing_vocab` ortam-bağımsız onarım (gts/roots VAR, yalnız vocab YOK mock'u). |

Satır çıpaları KORUNUNDU: K2 kanarya bloğu (satır 68-77) dokunulmadan, yeni sabitler NK çiftini BÖLMEYEN satırlara girdi; `olcum_kabi.py` rc=0, **30 PASS · 0 HATA** (yeni :84 kanarya dahil).

## 2. A3 — r2 DECOMP tavan ölçümü (TAMAMLANDI)

Betik: `scratch/anka_t0106_decomp_tavan_r2.py` (P5-A betik İMPORT deseni değil; kanonik parçalar import, VOCAB/LEXICON sha-pinned — uyuşmazlıkta rc=2).
Çıktı: `anka_t0106_decomp_tavan_r2_2026-09-24.json` sha256 `e1b81b064b32f66c…` rc=0.

| | r1 çıpa (P5-A) | **r2 (bu ölçüm)** |
|---|---|---|
| RAW ROUGE-L | — | 0,4188 |
| **DECOMP ROUGE-L** | 0,9509 | **0,9543** (Δ RAW'a +0,5355) |
| RAW'ı geçmeyen kayıt | — | 0/100 |
| UNK / decompile istisna | — | 23 / 0 |

→ r2 tavan r1 bant-İÇİNDE (0,9509 ile 0,9543, Δ +0,0034 tek orneklem gürültüsü mertebesi).
**Aday eşik beyanı DEĞİŞMEDİ:** `ESIK_ROUGE_DECOMP_ADAY = 0.7988` (0,9509 × 0,84) —
k=0,84 kalibrasyonu RAW dağılımında yapıldı; DECOMP'a taşınması doğrulanmadı; **hüküme BAĞLANMAZ**.

## 3. Duman koşumu — AYNI kabin çıpası (a1r/r1, n=20)

Çıktı: `anka_t0106_duman_a1r_n20.json` sha256 `9f02e1e7b86759b3…` rc=0.

| Eksen | Ölçülen | Çıpa (T-0105/kabin) | Hüküm |
|---|---|---|---|
| A CE | **3,5352 ± 0,8032** | 3,5352 (birebir) | ✓ |
| B top-1 | **%49,56** | 49,56 (birebir) | ✓ |
| ceket ROUGE RAW | 0,0000 | a1r zemin 0,0 (p2_taban_zemin) | ✓ band-içi |
| ceket DECOMP | 0,0000 (aday 0,7988, bağlanmaz) | — | ✓ |
| ezber/tutarsızlık/kesişim | 0,00 / %100 / 0,00 | zemin | ✓ |
| `ham_gm` | **20/20 dolu** | — | ✓ |
| decompile istisna / yuzey UNK | 0 / 641 | — | ✓ |

Yeni alanların TÜMÜ dolu; hüküm sözlüğü değişmedi (RAW eşik); A/B çıpası birebir
⇒ ölçüt geçişi **geri-uyumlu**: mevcut hükümler etkilenmedi.

## 4. Kapanış koşulları (ilan §4 — 4/4)

1. `pytest tests/test_olcum_kabi_kapilari.py` → **11 passed** ✓
2. `scripts/olcum_kabi.py` rc=0, 30 PASS, :84 kanarya PASS ✓
3. Tam pytest → bilinen gateway sandbox istisnası dışında YEŞİL (canary-import onarımıyla) ✓
4. Duman ECA: yeni alanlar dolu, `rouge_l_ort` çıpa bandında, A/B birebir ✓

## 5. Beyanlı sınırlar

* DECOMP eşik **hüküm değil ADAY'dır** (r1+r2 tek orneklem her ikisi; bağlama
  kararı ayrı ilan + r2 gold-gm çoklu-orneklem doğrulaması ister — bu görevde YAPILMADI).
* `ham_gm` gm dizilerini kalıcı kaydeder; P3/P4 kaybı (checkpoint silinme) bu şemayla
  bir daha tekrar etmez.
* a2 koşum (Faz B) sonda çağrıları `--vocab/--lexicon` passthrough ile r2 pinli
  olacaktır (sürücüde B2 genişletmesi).

**Durum: FAZ A KAPANDI.** Faz B (a2-tabanlı ceket koşumu, TEK DEĞİŞKEN r1→r2 derleme)
başlıyor — ilan dosyası ayrı.