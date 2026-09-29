# T-0193 İLAN — Çıktı-biçim hizası tanısı (ROUGE gösterim-payı ayrışımı)

- **Damga:** BETİKTEN 2026-09-29T11:13:20Z; T-0193 claim `ok:true` ·
  kiralama `data`+`scripts`+`data/eval`+`.agent-bus/notes` `ok:true` (conflict 0)
- **Emir:** operatör (29 Eyl) — "çıktı-biçim hizası tanısı için görev aç ve koşuma başla"

## Soru

T-0192'de ROUGE-L **0,1058** (eşik 0,35) DUS — düşüşün kaç payı
GÖSTERİM-uyumsuzluğu (model çıkışı `<ENT>`-etiketli morfolojik jeton-
dizimi; referans doğal-cümle), kaç payı hakiki üretim-kalite açığı?

## Koşum öncesi çıpalar (BETİKTEN, 11:13:20Z)

- `data/anka_b1_5_best.pt` — `e5eb114e 5bd71d58 00ba423d 844b2e2a eed6b7 8beaf3 a4278690 a43fb51e…`
- `data/b1_5_splits/test.jsonl` — `f106e7d2…` · train.jsonl — `c2d8480b…`
- `src/compiler/decompiler.py` — `a7280cab 25ec0ba4 48bab3d5 b61f8f35 0df6a75f 497de6c2 da9f447c 19238dc4` (yalnız-OKU; import)
- `scripts/evaluate_b1_5_rigorous.py` — `77e265a0 ee1a1379 96a11be5 5ab71fba 2213b12d 706c89e1 f79d31dd 13f3c9c9` (tarihsel dokunulmAZ; `rouge_l_score` import-metrik)

## Tanı-desini (İLAN-önceden-sabit)

1. **BİREBİR-YÜZEY (pozitif-kontrol K1):** T-0192 koşum-kalıbı birebir
   (elle-parça-zarf `" ".join`, encode-son-EOS kırpım, argmax,
   max_new=128, 256-kırpım, eos/`</OUTPUT>` dur; rng.seed=42 → N=100).
   Yeniden-sayım T-0192 hüküm değerleriyle birebir kontrol: n=100 ·
   Ezber 0,0 · Tutarsızlık 22,0 · ROUGE-L 0,105808… · conditioning 6,0.
2. **İki gövde:** raw (T-0192 yüzey) + DECOMP
   (`decompile_sentence(gen_text, capitalize=True)` — gateway insan-yüzeyi,
   epistemic_agent.py:412 kanıtı). ROUGE-L tek-metrik-tanım
   (`rouge_l_score` import).
3. **Eşikler (önceden-sabit):** delta = decomp − raw.
   - **HIZA_TAM:** decomp ≥ 0,35 VE delta > 0,10
   - **HIZA_KISMİ:** 0,05 ≤ delta ≤ 0,10 (kalite-gap de ölçülür)
   - **HIZA_YOK:** delta < 0,05
   - rc: K1+K2 geçer → rc=0 (hüküm-sınıfı bilgilendirici); K1 düşer →
     `T0193_TANI_DUR` rc=2.
4. **Bileşen-kanıt:** 22 tutarsız üretimin ayrışımı (yüklemsiz vs döngü;
   idx-listesi) + format-envanteri (tag-jeton-payı, `<ENT>` blok/üretim,
   ort. jeton-sayısı).
5. Çıktı-yolları ayrı (`data/eval/t0193_*`) — T-0192 artefaktları EZİLMEZ.

## Koşum-disiplin

Sandbox-DIŞI MPS zorunlu; `src/compiler/**` yalnız-okuma; tarihsel betik
dokunulmAZ; canlı 8080'e 0 istek; donmuş yollar yalnız-okuma; hüküm
BETİKTEN (İLAN/sonuç ayrı dosyalar).

## Hüküm-yolu

`data/eval/t0193_cikti_hiza_hukum_2026-09-29.json` — eşik-hukum BETİKTEN.