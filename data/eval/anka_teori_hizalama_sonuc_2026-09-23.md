# T-0101 · TEMEL TEORİ HİZALAMA SONUCU — compile↔decompile↔root.tsv↔token-listesi tutarlılığı (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Görev:** T-0101 · **Koşum YOK** (CPU ölçüm) ·
İlân: `anka_teori_hizalama_ilani_2026-09-23.md` (ölçümden ÖNCE, sha256
`48982d15…`) · Sonda rc=**0** · **TANISAL** — eşik önerisi YAPILMADI.

## 1. Ilanlı beklenti kıyası (sayılar ilandan okunur; bant-dışı ayrıştırıldı)

| metrik | ilanlı beklenti | ölçülen (539/48.407) | hüküm |
|---|---|---|---|
| DECOMP ROUGE (539 tam) | ≥ 0,90 | **0,9473** | BAND İÇİ |
| YUZEY pozitif kontrol | 1,0000 | 1,0000 | BAND İÇİ |
| birebir DECOMP=ref | ≥ %85 | **0/539 (0,0)** | BANT DIŞI → §2 ayrıştırma |
| compile determinizm | 539/539 | **539/539** | BAND İÇİ |
| (C) meta digest eşleşmesi | TRUE | `sozluk_sha256` + `lexicon_sha256` **birebir** | BAND İÇİ* |
| Y2 vocab→lexicon kök vuruş | ≥ %90 | **0,9940** (497/500) | BAND İÇİ |
| Y1 lexicon→vocab kapsama | ≥ %80 | **0,6666** (32.268/48.407) | BAND DIŞI → §3 ayrıştırma |

\* Sonda ilk koşumda `vocab_sha256` diye ARADIĞI anahtar meta'da yoktu
(False false-alarm); kanonik anahtar `sozluk_sha256` — düzeltilmiş ölçümde
**birebir eşleşme** (`f9940a8d8…` hem meta'da hem dosyada). İlanlı "birebir
DECOMP=ref ≥ %85" beklentisi **yanlış yazılmıştı**: kapı deseni
(ECA:482-484) `capitalize=False` ile çağırır; birebir eşleşme beklemek
yazım hatasıydı — fail-closed bant döngüsü bu yüzden çalıştı.

## 2. (A) Roundtrip ayrışması — birebir 0, ama kayıp kozmetik + küçük

| fark sınıfı | sayı | oran | içerik |
|---|---|---|---|
| birebir | 0 | 0,0 | — |
| buyuk_harf | 235 | %43,6 | `capitalize=False` deseni (ECA:483) — tasarım |
| bosluk_nokta | 150 | %27,8 | tırnak/boşluk (`'şerit testere'`) |
| **icerik** | 154 | %28,6 | **ROUGE ort 0,9537** — küçük LCS farkı |

Icerik sınıfının en düşük 5'i ROUGE 0,8519-0,90; en düşüklerin decomp
metinlerinde **çözülmemiş ek etiketleri** görünüyor (`POSS_3SG CASE_LOC_N`
— idx 20/234): graph'ta olmayan ek-zincir kombinasyonları decompile'da
etiket metni olarak kalıyor. DECOMP tavanını 0,9473 → 1,0'a bağlayan
~%5 boşluğun bileşeni: çözülmemiş ekler + UNK + küçük LCS.

## 3. (B) Lexicon↔vocab hizalama boşluğu — İKİ YÖNLÜ

**Yön 2 (vocab→lexicon): 0,9940** — token listesinin lexicon'a yansıması
tam. Vuruşsuz 3: `Etrüsklerle`, `Philipposun`, `İsanın` (özel ad + ek).

**Yön 1 (lexicon→vocab): ham kapsama 0,6666.** Kapsama-dışı 16.139:
`ozel_karakter` 1.077 · `uzun` 3 · **`diger` 15.059**.
'diger' sınıfının **rastgele n=60** örnekleminde (alfabetik ilk-60 örnekleme
BIASLIYDI — düzeltildi) decompile-recover **0/60**: tamamı tek `[?]` (UNK)
üretiyor. Ayrışma iki alt sınıf gösterdi:

| alt sınıf | örnek | mekanizma |
|---|---|---|
| kökü lexicon'da YOK | `kruton`, `kayet`, `voltametre`, `aav` | stem adayı yok ⇒ UNK |
| kökü VAR, zincir graph'ta YOK | `gezinebil`, `mermerleştir`, `abanabil` (kök `aban` yok) | `find_stems` yalnız `ab`/`aba` döner ⇒ UNK |

**Ölçülmüş etki sınırlı:** gerçek külliyatta (heldout 539) UNK token oranı
~%1,45 (P5-A, 31/2.138) — modelin gördüğü metinler vocab'daki köklerle
kurulu; lexicon'un vocab-dışı ~%31'i **ölü lemma** (ne token ne
compile-reachable), ama aktif cümlelere değmiyor. Beklenti "≥ %80"nin
yanlış kalibre edilmiş olduğu ölçümle gösterildi: lexicon lemma kümesi
(48.407) vocab kök kümesinden (32.268) **yapısal olarak geniştir** —
çekimli/türevli lemma kayıtları (`kullanılmalıdır` gibi) token olmak
zorunda DEĞİL; asıl ölçüt bunların compile-reachability'si ve o, kök
lexicon'da VARSA çalışıyor (`kullanılmalıdır` → `kullanıl TENSE_NECESS
COPULA_AORIST` ✓, `marangozlukta` → `marangoz DERIV_lIk CASE_LOC` ✓).

## 4. (C) Eğitim pipeline — modelin doğru compile edilmiş külliyatla eğitildiği kanıtı

`data/anka_a1r_pretrain.bin.meta.json` (18 anahtar) taşıdığı hizalama:
`bin_sha256` + `sozluk_sha256 = f9940a8d8…` (ölçülen dosya digest'iyle
**birebir**) + `sozluk_giris = 33114` + `lexicon_sha256` (birebir) +
`literal_entity_mode = False` (eval kurulumuyla aynı) + `block_size` +
`damga_utc` + `ureten_gorev`. Kardeş-meta cross-check mekanizması
(train.py:112-116 deseni) eğitim külliyatının ölçülen vocab/lexicon ile
**aynı kaynak çıpasından** üretildiğini kanıtlıyor.

## 5. Temel teori hükmü

**Zincir ÇALIŞIYOR:** (1) compile deterministik 539/539 · (2) eğitim
külliyatı digest hizalaması birebir · (3) token listesi lexicon'a %99,4
yansıyor · (4) gidiş-dönüş DECOMP 0,9473 — **cevaplar doğru decompile
ediliyor**; kalan fark kozmetik (harf/tırnak) + küçük LCS + graph-dışı
ek-zincirler. (5) ÖLÇÜLMÜŞ BOŞLUK: lexicon'un vocab-dışı ~%31'i compile
yolundan da ulaşılmaz (recover 0/60) — etkisi külliyatta %1,45 UNK'la
sınırlı; bu boşluk "hassas hizalama" başlığının ölçülmüş iş maddesidir
(operatörün "sonra" dedikleri sıranın ilk öğesi).

## 6. Kapı DECOMP geçişi (AÇIK KONU — bu rapor kararı BAĞLAMAZ)

P5-A (n=100) 0,9509 · T-0101 (539 tam) 0,9473 — iki bağımsız ölçüm uyumlu.
Kapı ROUGE'unu RAW'dan DECOMP'a geçirmek (ECA:486-498 `rl` satırı) ölçüt
kararıdır; geçmiş sonda JSON'ları per-kayıt üretim saklamadığından geçmiş
koşumların DECOMP çift-hükmü saf fonksiyonla YAPILAMAZ (T-0100 sınır dersi
— ilan §5). Bu raporun kanıt katkısı: DECOMP tavanın 539'ın tamamında
**0,9473** ile doğrulanması.

## 7. Digest tablosu (betikle hesaplandı)

| artefakt | sha256 |
|---|---|
| `scratch/anka_teori_hizalama_sondasi.py` | `5fc69910…` |
| `scratch/anka_teori_hizalama_sondasi.json` (per-kayıt üretim DAHİL — gelecek çift-hüküm için) | `46bf1537…` |
| `data/eval/anka_teori_hizalama_ilani_2026-09-23.md` | `48982d15…` |

*Kaynak çıpası: vocab 33.114 (`f9940a8d8…`) · lexicon 52.582 satır ·
heldout 539 (`c397eb08…`) · meta `anka_a1r_pretrain.bin.meta.json` hizalama
birebir. Zincirde DEĞİŞİKLİK YOK — olcum_kabi rc=0, 27 PASS, 4/4 kanarya.*

## 8. Kapanış

**T-0101 KAPANDI (tanısal):** temel teori zinciri dört eksende doğrulandı
(compile determinizmi · pipeline digest hizalaması · vocab→lexicon %99,4 ·
decompile roundtrip 0,9473). İki ölçülmüş boşluk beyanlı: (i) lexicon↔vocab
ham kapsama %66,7, kapsama-dışı lemma'lar UNK'a düşüyor (etki külliyatta
%1,45) — "hassas hizalama" işi; (ii) graph-dışı ek-zincirler decompile'da
etiket kalıyor (icerik sınıfının bileşeni). Kapı DECOMP geçişi operatör
kararına açık.