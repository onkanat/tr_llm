# İLAN-3 — T-0157 koşum-1 ölçüm-kabı import-istisnası beyanı (koşum-2 ÖNCESİ)

**Damga (BETİKTEN, koşum-2 öncesi):** `2026-09-28T07:56:21Z`
**Meşruiyet:** Koşum-1 (`venv/bin/python scripts/dogrulama_t0157_gateway_okunabilirlik.py`,
log `t0157_gateway_okunabilirlik_kosum_2026-09-28.log`) **rc=1 ile DÜŞTÜ** —
betik `:61`'de `from src.llm.prompt_contract import build_rag_prompt_tokens`
ile YANLIŞ modülden import yaptı; gerçek tanım `src/rag/rag_pipeline.py`'da
(`epistemic_agent.py:39` bu modülden import eder). Bu bir **ölçüm-kabı
(betik) hatasıdır — üretim-kod hatası DEĞİL**; T-0155 koşum-1 / T-0151
koşum-1 sınıfıdır.

## Koşum-1 durumu (ölçülen)

- İstisna import aşamasında (satır 61) meydana geldi: **hüküm-JSON
  YAZILMADI** ve **K1-K6'dan HİÇBİRİ koşmadı** (K1 kaynak-çıpası dahi
  ölçülemedi). Hiçbir kapı sonucu YOKTUR; bu koşum hiçbir kapı hakkında
  kanıt içermez.
- Koşum-1 artefaktları **DOKUNULMAZ**: `…kosum_2026-09-28.log` olduğu
  gibi korunur; hüküm/rapor eski adlarıyla hiç yazılmadığından üzerine
  yazma riski yoktur.

## Onarım (yalnız ölçüm-kabı; üretim-kod DEĞİŞMEDİ)

1. Betik `:61` importu `from src.rag.rag_pipeline import
   build_rag_prompt_tokens` olarak düzeltildi (üretim-import'uyla birebir).
2. Betiğin çıktı-adları **koşum-2 adlarına** çevrildi:
   `HUKUM_YOL = …hukum2_2026-09-28.json`,
   `RAPOR_YOL = …rapor2_2026-09-28.md` (İLAN-1'deki çıktı-beyanı koşum-1
   için geçerli kaldı; koşum-2 bu İLAN-3 ile beyan edilir).
   Koşum-2 logu: `…kosum2_2026-09-28.log` (yönlendirme-adı).
3. Statik-ön onarım-sonrası yeniden koşuldu: py_compile OK; pyflakes
   **rc=0** (İLAN-2 beyaz-listesi `clean_query_tags` bulgusu betikte YOK —
   betik yüzeyi temiz; beyaz-liste İLAN-1 K5 üretim-dosyası maddesi için
   aynen geçerlidir).

## Koşum-2 disiplini

- **Kapılar, hedef-değerler, sorgular, DOKUNULMAZLAR DEĞİŞMEZ** (İLAN-1
  sabit; İLAN-2 K5 netleştirmesi geçerli). **Yumuşatma YOK.**
- Koşum-2 **sandbox DIŞI** (MPS; yoksa DUR) ve **sarmal-sız**
  (`nohup …; echo rc=$?` rc'yi yutar — T-0155 dersi).
- Koşum-2'yi `hukum2`/`rapor2` yazması dışında betik İÇERİĞİ koşum-1
  betiğiyle aynıdır; kapı mantığına dokunulmadı.