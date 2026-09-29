# T-0186 HÜKÜM — VOWEL_DROP derin-mekanizması tanısı

- **Damga:** BETİKTEN 2026-09-29T05:14:19Z (probe-RC betik-çıktısı; JSON damgası 05:14:31Z)
- **İLAN:** `t0186_vowel_drop_analiz_ilan_2026-09-29.md` (05:02:11Z, koşum-öncesi)
- **Yürütücü:** claude · keşif-ajanı + BETİKTEN probe'lar

## Üç-soru cevabı (İLAN'ın tanı-kabülleri #1–#3)

1. **Lexicon ARZ:** VAR — `VOWEL_DROP`-tag'li satırlar BETİKTEN sayaç:
   `roots_anka_r1.tsv` 153 · `roots_anka_r2.tsv` 153 · `roots.tsv` 151 (anı-JSON
   `anka_r1_lexicon_2026-09-19.json`: `VOWEL_DROP: 152` + `VOWEL_DROP,GEMINATION: 1`.
   Ekim-kuralı BETİKTEN: `import_gts.py:67-69` (son-harf ∈ {l,r,n}; takı ünsüz;
   len>2). → **ARZ_YETMEZ DEĞİL.**
2. **TOKENIZER_KIRIK şıkkı:** ÇÜRÜYOR — `src/llm/tokenizer.py` içinde VOWEL/vowel
   grep'i BOŞ (tek isabet tokenizer.py:218 comment); bu by-design bir eksik DEĞİL:
   `encode()` kompilatörü iç-süreç olarak çalıştırır (T-0141 zinciri) ve token-
   vektör LEMMA-formalıdır (probe kanıtı aşağı). → **TOKENIZER_KIRIK DEĞİL.**
3. **DECOMPILER_KIRIK şıkkı:** DETERMİNİSTİK ZİNCİRDE KIRIK YOK — roundtrip-probe
   (R1-lexicon, T-0186 yolları, BETİKTEN): FAZ-B `örnek=9188, bit-özdeşlik=0.999782`;
   düşen 2 örnek (BIT_UYUMSUZ `çırağsa` + YOL_0 `ucul`) **VOWEL_DROP-sınıfı DEĞİL**
   (VOIcing/VOICE_PASS ailesi). id/morpheme-dökümü probe (BETİKTEN 05:14:19Z):
   `burun→['burun']` · `burnu→['burun','POSS_3SG']` (stems yüzey `burn`) ·
   `ağzı→['ağız','POSS_3SG']` · `hacrimiz→['hacir','POSS_1PL']` (Sınıf-A'nın eski
   kırık-örneği DOĞRU çözülüyor); decompile_tags tag[0]=lemma-kanıtı
   (decompiler.py:92-110: root_lemma=clean_tags[0]) + attrs→mutate_stem
   (phonology.py:53-59). → **DECOMPILER_KIRIK DEĞİL (deterministik kanada).**

## Yargı (İLAN'ın Yargı-sınıflarımdan)

**T0186_MEKANIZMA_ACIKLANDI** — ünlü-düşme mekanizması deterministik zincirde
BÜTÜN; token-vektör lemma-formalı olduğundan alternasyon yüzey-katmanında
(phonology.mutate_stem) çözülür ve kök-seçim lemma-guard'dan (core.py A2,
T-0150) geçer. **Derin-mekanizma kuyruğunun varsaydığı "altında kırık bir
katman daha var" hipotezi BETİKTEN kanıtla ÇÜRÜTÜLDÜ.**

## BELIRSIZ-BAND beyanı

- LM-tarafı üretimde ünlü-düşme-özel kusur KANITI: YOK (son hüküm-JSON üçünde
  `grep VOWEL_DROP` = 0; canlı-bes-test'te kusurlar `[?]`/PROPER_NOUN sınıfı).
  LM-yeti ayrı ölçüm yüzeyidir (görev-kapsamında koşum yok — İLAN beyanı).

## Kanıt-yolları

`data/eval/t0186_roundtrip_probe_2026-09-29.md` + `_hukum_.json` (rc=2 — ancak bu
DAKİKA'daki rc=2, T-0147 çıpasının roots.tsv-bağlılığından; VOWEL_DROP-tanımda
kanıt-değeri: düşen-2 örneğin sınıfı) · probe-koşum günlüğü BETİKTEN (05:14:19Z).