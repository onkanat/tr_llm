# T-0104 · UNK ANATOMİSİ + TEMİZLİK PAKETİ + R2 KÖK/VOCAB + YENİDEN DERLEME (İLÂN)

**Damga:** 23 Eyl 2026 · **Görev:** T-0104 · **Yürütücü:** claude ·
Üst: T-0103 (KAPANDI). Operatör istekleri: *"UNK temizle ve derle, eğit"* ·
*"Bunun çözümü roots.tvs ve tokenlist"*. Bu ilan build-ölçümünden ÖNCE yazıldı;
bantlar kumanda sayılarından okundu (3-kez-yanlış-yazıldı dersi).

## 1. Kumanda bulguları (hüküm-öncesi ölçüm, tamamı beyanlı)

Gerçek bin `data/anka_a1r_pretrain.bin`: jeton 100.000.000 ·
**unk 2.376.511 = %2,3765**.

UNK bağlam sınıfları (±4 jeton komşuluk, n=3.000): **rakam %61,1** ·
ASCII(ad/yabancı) %27,3 · **İngilizce %4,0** · Türkçe %7,6.

**T-0102 hipotezi düzeltildi:** "UNK'ların ~%46'sı İngilizce-komşu"
bağlam-istatistiğiydi; doğrudan ölçümde İngilizce fonksiyon-kelime komşuluğu
yalnız **%4**. Asıl havuz:
1. **Apostrofli sayı+ek** — `1953'te`/`2017'de`/`%12'dir` → bütün kelime UNK
   (tek jeton); `%` zaten encode'da DÜŞÜYOR; rakam-komşu UNK'ların asıl içeriği.
2. **NK-aralığı-dışı semboller** — `–` (325), `*`, `=`, `&` bağımsız simgeler.
3. **Markdown dosya/galeri satırları** — `jpg`, `file:`, `Dosya:…jpg|küçükresim`
   (Resim Galerisi/Gallery bölümleri; neredeyse sıfır dil-beceri değeri).
4. **Türkçe kök-OOV** — encode-düzeyi (düzeltme: kelime-compile sayımı
   AŞIRI sayıyordu; apostrofli özel adlar "Türkiye'de" zaten `türkiye CASE_LOC`
   olarak compile ediliyor): tür 7.376 · jeton-payı ~%4,4 · başlıkta
   `idi` 51, `vb` 36, `rapçi`, `finalde`, `kalbin`, `ismi`.

**Ayırt-edicilik dersi (kumanda-1):** saf ASCII-alpha ölçütü ayırt edici
DEĞİL (medyan belge %36 ASCII-kelime — `ve/bir/daha/kadar/sonra/olarak`
Türkçe fonksiyon kelimeleri zaten ASCII); bu kural belgelerin %97'sini ezer.
**İngilizce cümle-filtresi VERİMSİZ ölçüldü:** eşik≥2'de ek UNK %4,3 için
%5,6 jeton bedel ⇒ **UYGULANMAYACAK** (beyanlı ret).

## 2. İlanlı temizlik paketi (G) — deterministik, sıralı, arin() sonrası

1. **Satır-eleyici:** `^(Dosya|File|Image|Resim):…$` satırları, dosya-adı
   satırları (`*.(jpg|png|gif|svg|webp|ogg|mp3|wav)`), galeri başlıkları
   (`Resim Galerisi|Gallery|Galeri|Resimler|Görseller|Dosyalar`) düşer.
   Ölçüm: UNK −%3,3 · bedel −%0,92.
2. **Sembol-norm:** `–/— → -` (NK-aralığında 32141) + bağımsız
   `[*=:&|]{1,3}` satır düşürme. (Kümelenmiş etki ölçümü 1+2: UNK −%10,5.)
3. **Sayı-apostrof ek düşürme:** `r"\b(\d[\d.,:]*)'\w+"` → `\1`
   (`1953'te`→`1953`). Ölçüm: UNK −%28,4 · bedel ~+0,3% (UNK tekli → rakam
   jetonları). **Beyanlı bilgi kaybı:** sayı-eki dilbilgisi bilgisi düşer
   (o jetonlar UNK'tan hiçbir şey öğrenmiyordu).
4. **HTML entity:** `&[a-zA-Z]+;` → boşluk (`&nbsp;`, `&ndash;`).
5. **LaTeX komut:** `\\[a-zA-Z]+` → boşluk (`\cdot`, `\begin`…).
6. **Satır-içi dosya adı parçası:** `\w+\.(jpe?g|png|gif|svg|webp|ogg|mp3|wav)`
   → boşluk (G1'in satır-başı kuralının yetişmediği orta-satır kalıntılar).
7. **Tek yıldız:** `\*` → boşluk (KURALLAR'ın `\*{2,}` kalıbının
   tamamlaması; `*magister` tarzı kalıntı).

**Revizyon 3 (r2-census pozitif kontrolünün yakaladığı durum):** ilk
aday listesinde markup çöpü (`&ensp`, `\cdots`, `Georgia.jpg`…) vardı —
bunlar kök DEĞİL temizlik kalıntısı; G4-G7 yukarıya eklendi ve aday
filreye alfabetik test eklendi (aday = `^[harf][harf'-]*$`, ≥2 harf,
rakam-yok, ENG_MARK-yok).
**Revizyon 3 koşum (rc=0):** belge 3.641 · OOV tür 18.651 (TR-harfli
5.112) · aday 798 · +797 kök · **N=33.911** · pozitif **197/200=%98,5**
(bant ≥%90 GEÇTİ). Kalan 3 (`Hee-ae`, `Ji-hoon'un`, `Reis-ül`) §5'in
beyanlı kusur sınıfında (ek-çözümleme YAPILMAZ — çekimli biçim
derleyicide çözülemiyor; sözlükte VAR ama tek-başına compile edilemez).

Ret beyanı: İngilizce cümle-filtresi (§1) ve İngilizce kelime ekleme
(model diline İngilizce öğretmek — karşı) UYGULANMADI.

## 3. R2 kök/vocab (operatör yönü: "roots.tsv ve tokenlist")

* `data/lexicon/roots_anka_r2.tsv` = roots_anka_r1.tsv (sha256
  `ea874a73…` korunur) + **encode-düzeyi OOV türleri, külliyat-frekans ≥5**
  (census: deterministik örneklem n=6000, seed 42; kök = kelime-kendisi,
  NOUN/"-"). **Revizyon 1 (build-öncesi):** ilk kural "TR-harfli" filtresi
  idi; build-smoke K9 probe'u yakaladı — **ASCII-Türkçe kelimeler**
  (`idi`, `vb`, `ismi`, `finalde`) filtre dışında kalıp UNK kalıyordu.
  Yeni kural: aday = frekans≥5 OOV tür; **HARİÇ: rakamlı türler** (sayı-
  apostrof G-kuralı ile çözülür; rakam jetonları NK-aralığında var)
  **+ İngilizce fonksiyon-kelime kümesi** (kumanda-2 ENG_MARK, 44 kelime —
  `and`, `by`, `the`… sözlüğe eklenmez, §5 ret ile tutarlı).
  Beyanlı kusur: ek-çözümleme YAPILMAZ — çekimli biçim ("finalde")
  ayrı kök olarak eklenir; morfem-genelliği bilinçli feragat.
* **Revizyon 2 (build-smoke K7c'nin yakaladığı kusur):** T-0103 bypass'ın
  self-match eşleştirmesi matched_prefix üzerinden — `istanbul` girdisi,
  `İstanbul` lemma id'si stoi'da VARKEN, kısa-önek adaylarına
  (`is`) düşüp **boş parse** üretiyordu. `core.py` düzeltmesi: filtre
  artık **lemmanın stoi-üyeliğine** bakar (öz-parse yalnız gerçekten
  UNK'taysa düşürülür). T-0103 probları (`bıçaklanış`, `hiddetlendir`,
  `aratıl`, `horozlaş`) ve akıl-sağlığı (`kullanılmalıdır`,
  `marangozlukta`) birebir korundu. **Beyanlı kusur:** ek-çözümleme YAPILMAZ — çekimli biçim
  ("finalde") ayrı kök olarak eklenir; morfem-genelliği bilinçli feragat.
* `data/rebuild/vocab_anka_r2_<N>.json` = vocab_anka_r1_33114.json (sha256
  `f9940a8d…` korunur) + yeni kök token id'leri. **Önek değişmezliği:
  id 0..33113 ve `next_id=33114` BİREBİR korunur** (fail-closed assertion;
  V1-V4 kapı deseni, `scratch/anka_r3_vocab.py`).
* Compiler: `CrystalCompiler(lex, graph, vocab=vocab)` — T-0103 self-match
  bypass üretimde İLK KEZ devrede.
* Model: embedding + output head 33.114 → N satır (train.py:323
  `resize_state_dict`); yeni satırlar rastgele init, önek satırları devralınır.

## 4. Beklenti bantları (kumanda sayılarından okunan)

| ölçüm | ilanlı bant | kaynak |
|---|---|---|
| yeni bin unk_orani | **1,40-1,54** (Revizyon 4) | ONCE 2,3765 × −%28,4..−%35 (eski bant 1,55-1,70) × Rev3 etkisi −%9,4 relatif (kumanda: örneklem unk 1,2758→1,1559; jeton bedel −%0,71) = alt 1,40 · üst 1,54. Üst bant ASLA artmaz — temizlik eklenirken unk üst bandı büyümez (ilk taslakta 1,72 yazılmıştı; kendi yakaladığım mantık-kusuru) |
| K2-unk (YAZILAN bin) | ≤ **2,0** | eşik |
| K4 jeton | ≥ 90M (hedef 100M) | değişmez |
| jeton bedel (temizlik) | +0,5% üst sınırı | ölçüm +0,3% · Rev3 −%0,71 (DÜŞÜRÜR) |
| K1 noktalama | max_id ≥ 32145 · oran ≥ 5,0 | değişmez |
| K3 pad / K5 tekrarsızlık / K6 izlenebilirlik | değişmez | değişmez |
| val/train belge-id kesişimi | 0 | değişmez |
| eğitim devam-modu canlılık | sıfırdan-lnV bandı GEÇERSİZ | moda-bağlı dersi |
| resize sonrası pytest + olcum_kabi | yeşil (bilinen gateway istisnası) | değişmez |

**Smoke yüzeyi beyanı:** smoke (20.000 jeton) üzerinde `K4_olcek` ve
`K10_unk_bant` beklenen-FAIL'dir (ölçek ve tam-bin bantı ölçülmez);
tam koşumda tüm kapılar fail-closed rc'ye bağlıdır. Smoke koşumunda
K1-K8+K9 pozitif GEÇTİ; G2 unk=0,78 (smoke-örneklem) · G3 PN=4,605
(smoke-örneklem; tam-bin hüküm SERI'de).

**Revizyon 5 (ilk tam-koşum fail-closed rc=2'nin hükmü; ONCE-çapa tazeleme):**
ilk koşum `data/anka_a2_pretrain.bin` 100.000.000 jeton (sha256
`da724af570636c55…`) · val 2.000.000 (`96ece8010ca5f611…`). Üç kapı KALDI —
hepsi T-0080'in a1r'ye kalibre edilmiş bayat eşikleri. ONCE çapaları
AYNI kabinle (bin_olc) a1r bin'den ölçüldü:

| kapı | a1r ONCE (train/val) | a2 SONRA | yargı |
|---|---|---|---|
| unk_orani | 2,3765 / 3,1246 | **1,2246 / 1,9554** | hedef aşılı (−%48,5) |
| PN_oranı | 7,1363 / 8,617 | **6,761 / 8,1544** | DÜŞÜŞ (bypass PN-kırılma onarımının gölge etkisi) |
| kapsam (train) | 86,3653 / 47,572 | **86,6828 / 48,6981** | ONCE üzeri |

**Kusur beyanı (bayat-bant ailesi, iki kök):**
1. **Census örneklemi çifte-kullanım** — kumanda bantı (1,40-1,54) census
   örnekleminden çıkan köklerle hesaplanmıştı; kökler tam-binde census-örneklem
   DIŞINDAKI OOV'leri de çözerek örneklem-tahminden büyük düşüş verdi
   (−%48,5 vs tahmin −%40). Kumanda kabının örtük-varsayımı ilanı çürüttü
   (bilinen ders: kabın örtük-varsayımı).
2. **G3_PN=3,0 / G5=95 eşikleri a1r çapasıyla uyumsuz** — a1r bin bile
   geçemezdi (PN %7,14 · kapsam %86,37): T-0080'de farklı birim/pay ile
   kalibre edilmişler.

**Yeni ilanlı bantlar (fail-closed kalır):**
- `K10_unk_bant`: **1,10-1,54** — alt bant muhafazakâr kalır (aşırı-temizlik
  sinyali hâlâ kollanır); ölçüm 1,2246 ∈ bant.
- `G3_PN`: **≤ 7,50** (train) = a1r çapası 7,1363 + %5 pay; val çapası 8,617
  bilgi olarak kaydedilir. Ölçüm 6,761 PASS (ONCE'dan düşük — bypass PN
  kırılmasını onardı; istenen etkinin gölgesi).
- `G5_kapsam`: **train ≥ 85,0** · **val ≥ 45,0** = a1r çapaları
  (86,37/47,57) − muhafazakâr pay. Ölçümler 86,68/48,70 PASS.
- Diğer tüm kapılar değişmez. Hüküm: `scratch/anka_t0104_hukum.py`
  (ckpt sayıları + bu bantlar, fail-closed rc).

## 5. Sınırlar (beyanlı)

* `src/llm/tokenizer.py` DONMUŞ — sayı-apostrof kök-çözümü (1953'te → 1953+de
  graph zinciri) tokenizer işi; G-kuralı bunu temizlikte atlar (ek bilgisi düşer).
* `and/by/the/des/till` gibi yabancı kelimeler UNK KALIR (İngilizce kelime
  sözlüğe eklenmez).
* `data/*.pt` donmuş: eğitim çıktısı **yeni** `data/anka_a2.pt`;
  `data/anka_a1r.pt` salt-okunur (digest kayıt altında).
* A-ekseni (Wikipedia) yeniden derleniyor; SFT/ceket (B-ekseni) DOKUNULMAZ.
* Eğitim: `--pretrain` ZORUNLU; `--lr` AÇIKÇA; GPU'yu tüketen başka iş yok;
  uzun koşum artımlı checkpoint yazar.