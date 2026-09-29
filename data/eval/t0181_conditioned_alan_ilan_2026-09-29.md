# T-0181 İLAN — `conditioned` alanı HTTP yanıtına ekleniyor (operatör emri)

**Damga:** 2026-09-29T03:43:10Z (BETİKTEN, `date -u`) · **Yürütücü:** claude (T-0181)

## Arka-plan (ölçülmüş)

T-0179 canlı-smoke (29 Eyl 03:37Z, `rag_score 0,3911`): T-0179 eşik-değişiminin
(0,40→0,33) canlı-daki KARARI HTTP 17-anahtar yüzeyinde görünmezdi —
`conditioned` yalnız epistemic_agent iç-hesabıydı (satır :379) ve `is_high_similarity`
farklı bir kapıydı (Kapı-D, `similarity_threshold=0,85`; T-0143 beyanı). Operatör
kararı: alan eklensin.

## Beyan (değişim-kapsamı; koşum-öncesi)

1. `src/rag/epistemic_agent.py` — `process_query` return-sözlüğüne
   `"conditioned": conditioned` (satır :379 hesabı; **davranış değişmez** — yalnız
   çıktı-görünürlüğü).
2. `src/gateway/agent_gateway.py` — `ask()` map'ine
   `"conditioned": res.get("conditioned", False)`.
3. `tests/test_agent_gateway.py` — **1 yeni test**:
   alan-mezuniyet + pozitif-kontrol; stub-memory skorları
   `0,3911 → True` / `0,30 → False` (`RAG_MATCH_THRESHOLD == 0.33` çıpa-assert'i);
   0,3911 değer-i birebir hüküm-koşumu-3 çıpasıdır (T-0178).
4. `is_high_similarity`/Kapı-D **DOKUNULMAZ**; `similarity_threshold` dokunulmaz.
5. Mevcut test-YOK (test-SİLME YOK); "17-anahtar" beyanı "18-anahtar" olur —
   `ps`-çıpası bekleme: canlı-smoke tekrarı ayrı koşumda restart gerekir (bu turda
   koşum yok; canlı gateway 0 istek).

## Kapılar (koşum-öncesi beyan)

- K1 statik: py_compile + pyflakes (iki src + tests).
- K2 yeni-test (pozitif-kontrol 0,3911/0,30 ayrımı; alanın bool türü).
- K3 pytest AYRI koşum: baseline 317, yeni-düşen ∅ hedefi; düşme → DUR rc=2.
- K4 canlı-dokunulmazlık: gateway 0 istek; data/** (data/eval/ dışı) 0 yazım.

Hüküm: `data/eval/t0181_conditioned_alan_hukum_2026-09-29.json` (BETİKTEN).