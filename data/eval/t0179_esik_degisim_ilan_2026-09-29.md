# T-0179 İLAN PROTOKOLÜ: RAG_MATCH_THRESHOLD 0,40 → 0,33 (operatör-onaylı eşik-kararı)

- **Tarih:** 2026-09-29T03:16:20Z (BETİKTEN, `date -u`)
- **Görev:** T-0179 · **Yürütücü:** claude · **Tetik:** T-0178 önerisi operatör-onayı
  (28 Eyl AskUserQuestion: "0,33 (Önerilen)")

## 1. Ölçülmüş gerekçe

T-0178 BETİKTEN ölçüm: ayrım-aralığı [0,075 ; 0,3556) — OOV-maks 0,0556
(kendi kabında) / 0,075 (onarım beyanı); pozitif-min 0,3556. Eşik 0,33
pencere-ortası; pozitif-yakalama 20/20; OOV ayrımı ~4,4×.

## 2. Kod-değişimi (tek sabit; İLAN'dan sonra)

- `src/rag/rag_pipeline.py:30` → `RAG_MATCH_THRESHOLD: float = 0.40` → **0.33**.
- Dokunulan çıpa: `rag_pipeline.py` `fe1b37f40605a7e0…`, `tests/test_rag_pipeline.py`
  `df4784ff94950176…` (BETİKTEN shasum).
- `src/rag/vector_memory.py` DOKUNULMAZ.

## 3. Test-davranış-güncelleme (İLAN'beyanlı; test-SİLME YOK)

- **Vaka (i)** `test_rag_pipeline_gate_below_threshold`: MockMemory
  `return_score=0.39 → 0.30` — "eşik-altı skor koşullama YOK" niyeti korunur;
  0,39 yeni eşikte (0,33) artık pozitif-bandın İÇİNDEDİR (ölçülmüş karar);
  0,30 negatif-vakası olarak valid (0,33-altı).
- **Vaka (iii)** sınır-durumu: `is_context_usable(0.33) True`,
  `(0.33001) True`, `(0.32999) False`; 0,40-regresyon satırları KORUNUR
  (0,33-üstü True).
- Diğer eşik-etkili testler etkisiz (0,85 benzerlik-çapa — Kapı-D ayrı ölçek;
  `rag_score=0.40` yüklemi True-True davranış değişmez).

## 4. Yeniden-ölçüm kanıtı (değişim-SONRASI ayrı koşum)

- Betik: T-0178 ölçüm-betiği, koşum-3 (`--cikti-ek "3"`, ayrı hüküm-yolu
  `hukum3…`) — beklenti: `mevcut_esik_gecme 20/20` (0,33); tarama tablosu
  aynı dağılımı verir (skor-yolu kodu değişmedi — betik-düzey teyit); karar-
  sınıflandırma çıpa-skorlarla 20/20 (çapada 0,33 hesap).
- İstisna-beyanı: betik `K4` çıpa-uyum kapısı koşum-2'de bilinen yüksek-tail
  sparse-rank sınıfına tabi; koşum-3'te kapı kararı `mevcut_esik_gecme 20/20`
  temel sonuçtur, K4 bant-disi sayımı BEYAN olunur.

## 5. pytest

Ayrı koşum (baseline 317 — T-0177 sonrası); beklenen: vaka (i)/(iii)
güncellenmiş halleriyle tam geçiş; yeni-düşen ∅.

## 6. Hüküm

`data/eval/t0179_esik_degisim_hukum_2026-09-29.json` — BETİKTEN; commit AYRI
operatör-onayı.