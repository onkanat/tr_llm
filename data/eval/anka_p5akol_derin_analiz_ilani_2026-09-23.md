# P5-A · A-KOLU DERİN ANALİZ İLANI (ön-kayıtlı) — JETON çıktılarının decompiler'dan geçirilmesi

**Damga:** 23 Eyl 2026 · **Yazım:** ölçümden ÖNCE · **Görev:** T-0100
(kiralamalar: `data/eval/` dir + `scratch/anka_p5akol_*`) · **Koşum YOK** —
CPU ölçüm, dakikalar. Üst görev: T-0099 (P5 ölçüt tazeleme).

## 1. Neden (kumanda sınaması kanıtı — ölçümden önce gözlemlendi)

P5 A-kolu (heldout reproduce) ROUGE tavanını **0,4164** ölçmüştü (JETON =
`vocab.decode` dizisinin ham metni). Kumanda sınamasında aynı orneklem'in
**3 kayıdı** kanonik decompiler'dan
(`src/compiler/decompiler.py:MorphemeDecompiler.decompile_sentence` —
ECA:482-484 deseni) geçirildi:

| örnek | RAW ROUGE | DECOMP ROUGE |
|---|---|---|
| örnek 1 | 0,4545 | **0,7500** |
| örnek 3 | 0,2745 | **0,9714** |
| örnek 5 | 0,3922 | **0,9189** |

**Bulgu:** 0,4164 "tavanı" büyük ölçüde HAM decode kaybıdır; decompiler
ekleri (`TENSE_*`/`CASE_*`/`POSS_*`) yüzey Türkçesine çözümleyerek kaybın
büyük kısmını onarıyor (ör. `kullanıl TENSE_NECESS COPULA_AORIST` →
`kullanılmalıdır`). Kalan kayıp: `<UNK>` (`[?]`) kelimeleri + küçük biçim
farkları. Bu, tavanın **temsil katmanı** yorumunu ölçülebilir hâle getirir.

## 2. Sorular (derin analiz)

1. **Üç temsil, aynı orneklemde:** RAW / DECOMP / YUZEY ROUGE — DECOMP tavanı
   kaç? RAW−DECOMP farkının kaynağı ne?
2. **Sözlük boyutu (33.114) ile bağı:** `<UNK>` oranı — referans
   kelimelerinin kaçı sözlüğe giremedi; UNK'lu kayıtların DECOMP ROUGE'u
   UNK'suzlara karşı ne kadar düşük?
3. **root.tsv (52.582 satır) ile bağı:** JETON dizilerindeki kök adaylarının
   lexicon vuruş oranı (`find_stems`); vuruş olmayan kökler hangi sınıfta?
4. **Jeton listesi kalitesi:** ek etiketlerin `affix_info` (morphotactics
   graph) ile çözümleme oranı vs `COMMON_FALLBACK`'a düşen oran; decompile
   istisnası (fallback) oranı.
5. **Ölçüt çıpası:** ECA kapısı ROUGE'u HAM `gm`'den sayıyor (ECA:481). DECOMP
   yolunun ölçütte kullanılması ayrı bir ölçüt kararıdır — bu analiz yalnız
   **tanısal**, eşik önerisi YAPMAZ.

## 3. Yöntem (SABİT)

* **Birebir aynı orneklem:** `heldout 539` · n=100 · seed 42 · `rng.sample`
  (P5 sonda betiği `scratch/anka_p5_tavan_sondasi.py` ile aynı dizi).
* Kanonik import: `MorphemeDecompiler` (`src.compiler.decompiler`),
  `kelimeler`/`lcs_f1` (ECA), `rouge_l_score` (`evaluate_b1_5_rigorous`) —
  kopya YASAK. Desen kopyası yalnız kapanmış `t0096_kos` betiğinden DEĞİL,
  P5 betiğinden (`7326de5f…`) alınır.
* `decompile_sentence` çağrısı ECA deseniyle try/except; istisna kayıtlar
  RAW'a düşer ve **fallback bayrağıyla** raporlanır.
* Girdi digest'leri: vocab `33.114` (`vocab_anka_r1_33114.json`) ·
  lexicon `roots_anka_r1.tsv` 52.582 satır · heldout `c397eb08…`.

## 4. Beklenti (ilandan okunur)

* Kumanda sınamasındaki üç kayıt birebir reproduce edilir (aynı seed, aynı
  zincir).
* **Yön beklentisi:** DECOMP ort ROUGE'u her kayıtta RAW'a eşit veya daha
  büyük; ortalama DECOMP tavanı 0,4164'ün belirgin ÜSTÜNDE (kumanda sınaması
  0,75-0,97 bandına işaret ediyor). Garanti değil — ölçülür.
* UNK oranı sözlük boyutunun sınırlaması olarak raporlanır; UNK'lu kayıtlar
  düşük DECOMP ROUGE bekler.
* AÇIK KONU beyanı: kapı ROUGE'unun RAW yerine DECOMP'tan sayılması
  (ECA:481 değişikliği) bu analizde YAPILMAZ; ölçüm + kanıt çıktığında ayrı
  operatör kararı ister.

## 5. Çıktılar

* `scratch/anka_p5akol_derin_analiz.py` → `scratch/anka_p5akol_derin_analiz.json`
* `data/eval/anka_p5akol_derin_analiz_sonuc_2026-09-23.md` (tablo + yorum +
  digest tablosu).
* Doğrulama: `venv/bin/pytest` yeşil (bilinen gateway istisnası hariç).

## 6. Beyanlar

* Kapanmış kayıtlar ve `t0096_kos/**` DOKUNULMAZ · donmuş yollara yazım YOK ·
  `git add -A` YASAK · eşik değişikliği YAPILMAZ (tanısal).