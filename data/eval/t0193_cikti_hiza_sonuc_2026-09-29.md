# T-0193 SONUÇ — Çıktı-biçim hizası tanısı (ROUGE gösterim-payı ayrışımı)

- **Hüküm:** **T0193_TANI_GECTİ** (rc=0) · **HIZA_SINIFI: `HIZA_YOK`**
  (delta yok-eşiği altında) · BETİKTEN hüküm-JSON:
  `data/eval/t0193_cikti_hiza_hukum_2026-09-29.json`
- **Soru (İLAN):** T-0192 ROUGE 0,1058 düşüşü gösterim-payı mı, kalite-açığı mı?

## Eşik-tablo (İLAN-önceden-sabit birebir)

| Kapı | Ölçüm (BETİKTEN) | Sonuç |
|---|---|---|
| 6-çıpa koşum-başı sha | BETİKTEN (decompiler.py + tarihsel-eval sha dahil) | **PASS** |
| K1 pozitif-kontrol (raw sayım ≡ T-0192 hüküm) | 4 alan fark 0; ROUGE fark <1e-6 | **PASS** |
| K2 decompiler-sağlık | 100/100 istisnasız, **boş-decompile 0** | **PASS** |

## Anahtar-ölçüm (ÖNCEDEN-SABİT eşiklerle BETİKTEN)

| Yüzey | mean ROUGE-L |
|---|---|
| RAW (T-0192 jeton-dizim) | 0,105808… |
| DECOMP (kanonik insan-yüzeyi) | 0,132540… |
| **delta = gösterim-payı** | **+0,0267** → `< 0,05` ⇒ **HIZA_YOK** |

**TANIK-ÖRNEK (idx-0, BETİKTEN):** DECOMP yüzey gerçekten doğal:
`"Metin, Maran 'ına karşı bir komplo yaptı. bu durum, Roma 'ın [?]
'ın ölümüyle sonuçlandı."` — gösterim onarılmış halde ROUGE hâlâ
0,0000 ⇐ içerik referansla örtüşmüyor (`komplo` hariç).

## Bileşen-kanıt

- **Tutarsızlık %22'nin parçalanması: 22/22 yüklemsiz** (son-5-jetonda
  TENSE_/COPULA_ yok), bunlardan yalnız 1'i ayrıca döngü (idx-listesi
  hüküm-JSON'da). Önceden-sabit "yüklemsizlik ağırlıkta" gözlemi
  ölçüldü: %21-puan payı yüklemsizlik.
- **Format-envanter:** ort. tag-jeton-payı %32,3 · `<ENT>` blok 1,43
  üretilmiş-üretim · ort. jeton 34,0. Biçim etiket-yükü yüksek ama
  hizalama-onarımı skoru taşımıyor.

## Hüküm-okuma

Çıktı-biçim hizası **ROUGE açığını açıklamıyor** — decompiler yüzeyine
geçiş yalnız +0,0267. T-0192'nin iki DUS kapısı (ROUGE 0,1058,
Tutarsızlık %22,0) **hakiki üretim-kalite açığı**: model anka_b1_5_best
(2-epoch-regen taban) soru-ilişkili içerik üretmiyor ve yüklemsizlik
yüksek. İleride ROUGE-yükseltme çalışması gösterim-başına
yapmamalıdır (T-0193-eşik-sınıfı yanlış-direksiyonu engeller);
hedef üretim-tarafı: yüklemlilik + soru-ilişkili içerik.

## Koşum-disiplin ve beyanlar

1. K1 birebir-yüzey kanıtı: yeniden-koşum T-0192 raporuyla sayım-düzeyinde
   özdeş (argmax deterministik; seed/sample çıpası) — T-0192 sayıları
   tekrar-bilirlik-kaniti de oldu.
2. `src/compiler/**` yalnız-okuma (decompiler import; dokunuş yok,
   sha çıpası İLAN'da) · tarihsel betik dokunulmAZ · canlı 8080'e 0
   istek · donmuş yalnız-okuma.
3. koşum 11:14→11:15Z, ~4 dk, RC=0; BETİKTEN rc-yutma yok.

## Bus-disiplin

T-0193 claim ok · kiralama `data`+`scripts`+`data/eval`+
`.agent-bus/notes` ok (conflict 0) · hüküm/sonuç BETİKTEN ayrı
dosyalar · commit ayrı operatör-onayı.