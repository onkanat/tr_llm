# T-0186 İLAN — VOWEL_DROP derin-mekanizması tanısı

- **Damga:** BETİKTEN 2026-09-29T05:02:11Z (`date -u`; İLAN koşum-ÖNCE) · claim: T-0186 `ok:true` · kiralama: `data/eval` + `.agent-bus/notes` `ok:true`
- **Tür:** TANI (ölçüm + kök-mekanizma konumlandırma); KOD mutasyonu YOK; onarım ayrı emir.

## Koşum-öncesi sabit tanı-kabülleri (İLAN)

1. **Ölçüm-yüzeyi YOK-savunan değil:** her üç soru (lexicon-arzı, eval-artefakt-kanıtı,
   tokenizer/yüzey-katmanı) BETİKTEN komutla cevaplanır; `grep` boş-çıktı = "YOK (boş)"
   olarak raporlanır, sessiz-atlanır değil.
2. **Yargı-sınıfları KABULDEN ÖNCE sabit:** tanı sonunda mekanizma şu üç yargıdan BİRİNİ
   alır —
   - `ARZ_YETMEZ`: lexicon'da ünlü-düşme alternasyonu eğitilmiş arz yok (satır-sayacı
     BETİKTEN);
   - `TOKENIZER_KIRIK`: tokenizer/encode katmanı alternasyonu ayırt etmiyor (id-dökümü
     BETİKTEN);
   - `DECOMPILER_KIRIK`: morfem-dizi doğru, yüzeyleştirme kırık (src/compiler kalıbı).
   Hiçbiri kesinleyemezse kalan şıklar `BELIRSIZ_BAND` beyanlı.
3. **Ölçüt-ölü yasağı (T-0121 dersi):** sadece "VOWEL_DROP etiketi geçiyor" değil —
   etiket gerçek üretim-zincirinde (morfem→yüzey) kullanım sayacı da ölçülür; ikisi
   ayrı satırdadır.
4. **Kanayak:** T-0149 üç-sınıf tanısındaki (Sınıf-A/B/C) eval-artefakt kanıtı
   yeniden-okunur; elle-sayı YASAK — sayfa-dışı değer BETİKTEN.

## DOKUNULMAZLAR

`data/lexicon/**`, `src/llm/tokenizer.py`, `src/compiler/**` salt-okunur; canlı 8080'e
0 istek; `data/*.pt` yalnız yükleme; canlı üretim-borcu çizelgesine dokunulmaz.

## Hüküm-biçimi

`data/eval/t0186_vowel_drop_tani_2026-09-29.json` — BETİKTEN damga + üç-soru cevabı +
yargı-sınıfı + (onarım-yönü önerisi) . rc=0 yalnız üç-soru-cevabı tamam ise; aksi rc=2.