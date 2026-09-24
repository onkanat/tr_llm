# T-0101 · TEMEL TEORİ HİZALAMA ANALİZİ İLANI (ön-kayıtlı) — compile↔decompile↔root.tsv↔token-listesi tutarlılığı

**Damga:** 23 Eyl 2026 · **Yazım:** ölçümden ÖNCE · **Görev:** T-0101
(kiralamalar: `data/eval/` dir + `scratch/anka_teori_hizalama_*`) ·
**Koşum YOK** — CPU ölçüm, dakikalar. Operatör yönü: kapı DECOMP geçiş
kararından ÖNCE temel teori zinciri ölçülmeli — model compile'a göre
eğitiliyor (README:73 akışı: prompt → compiler → model → modül →
**decompile**); eğitimin ve decompile'ın doğru çalıştığının kanıtı
önceliklidir ("hassas hizalama · veri-seti büyüklüğü · modül kalitesi
SONRA").

## 1. Üç ölçüm (SABİT, ilandan okunur)

### (A) Compile↔decompile roundtrip — heldout 539 TAMAMI
* Yol: `ref → tokenizer.encode → vocab.decode → MorphemeDecompiler.
  decompile_sentence` (P5-A n=100 örneklemiyle AYNI zincir; bu ilan orneklemi
  **539'ın tamamına** büyütür — P5-A'nın 0,9509'u örneklem değeridir, tam
  külliyat değeri ölçülür).
* Çıktılar: YUZEY ROUGE (=1,0 pozitif kontrol) · RAW ROUGE · DECOMP ROUGE ·
  **birebir (exact-match) DECOMP=YUZEY sayımı**.
* **Kayıp ayrışması (betikle, elle YOK):** DECOMP≠YUZEY her kayıtta
  `kok_miss` kökleri sınıflanır — (i) rakam/`[sayı]`, (ii) çekimli/başharf-
  büyük bütün kelime, (iii) kesme sınıfı (P5-A'da `l ×3`), (iv) UNK,
  (v) diğer. Her sınıftan **≥10 örnek metin** JSON'a yazılır (hükmün
  gözle denetlenebilirliği).

### (B) root.tsv↔token-listesi — İKİ YÖNLÜ ([[cift-yonunu-alan-adindan-oku]])
* **Yön 1 (lexicon→vocab):** `roots_anka_r1.tsv` (52.582 satır) lemma'larının
  `vocab_anka_r1_33114.json` kapsama oranı — lemma birebir tek token mı?
  Kapsama dışı lemma'lar sınıflanır: UNI alt-parça'a ayrışan / çekimli /
  özel-karakter / diğer; sınıf başına sayı + örnek.
* **Yön 2 (vocab→lexicon):** vocab kök token'larının (etiket/meta olmayan
  token'ların rastgele n=500 örneği, seed 42) lexicon `find_stems` vuruş
  oranı — token listesinin lexicon'a yansıması.
* **Beklenti (ilandan okunur; ölçümden sonra birebir kıyaslanır):**
  Yön 1 kapsama ≥ %80 · Yön 2 vuruş ≥ %90. Garanti değil — ölçülür; band
  dışı çıkarsa rapor NEDENİ ayrıştırmak zorundadır (fail-closed: sessiz
  "uyuyor" yazma).

### (C) Eğitim pipeline tutarlılığı
* `data/anka_a1r_pretrain.bin.meta.json` varsa digest/giriş cross-check
  `vocab_anka_r1_33114.json` ile (kardeş-meta mekanizması, train.py:112-116
  deseni); yoksa BEYAN edilir (sessiz geçilmez).
* **Compile determinizmi:** aynı cümlenin iki kez compile'ı bit-özdeş id
  dizisi (kod standardı: determinizmin pozitif kontrolü) — 539 referansın
  hepsinde.
* A1-r eğitim külliyatının tokenizer'ı ile eval tokenizer'ı aynı literal
  mod (`literal_entity_mode=False`) — P5-A kurulumuyla birebir.

## 2. Beklenti (3-kez-yanlış dersi: sayılar ilandan okunur, elle yazılmaz)

| metrik | ilanlı beklenti | kaynak |
|---|---|---|
| DECOMP ROUGE (539 tam) | n=100 örnekleminin 0,9509'u ±0,017 bandında; **≥ 0,90** | P5-A |
| YUZEY pozitif kontrol | **1,0000 birebir** | P5-A (0/100 kayıtta <1,0) |
| birebir DECOMP=YUZEY | ≥ %85 | P5-A delta dağılımı |
| kök vuruş (Yön 2, vocab→lexicon) | ≥ %90 | P5-A n=100'de %93,26 |
| Yön 1 kapsama | ≥ %80 (sözlük 33.114 < lexicon 52.582 ⇒ %100 beklensin DEĞİL) | sözlük boyut bağı (P5-A: sınırlayıcı değil) |
| compile determinizm | 539/539 bit-özdeş | kod standardı |

Band dışı değer **hüküm değil, ayrıştırma konusudur**: hangi sınıf kaydın
bandın dışına çektiği betikle raporlanır.

## 3. Kanonik İMPORT (kopya YASAK)

`MorphemeDecompiler` + `CrystalCompiler` + `LexiconManager` +
`build_default_graph` (src/compiler) · `KristalTokenizer`/`Vocabulary`
(src.llm.tokenizer) · `kelimeler` (ECA) · `rouge_l_score`
(evaluate_b1_5_rigorous) · P5-A betiğindeki kök-aday filtresi deseni
(`scratch/anka_p5akol_derin_analiz.py` — t0096_kos'tan DEĞİL, git'te
kayitlı P5 betiğinden).

## 4. Çıktılar ve doğrulama

* `scratch/anka_teori_hizalama_sondasi.py` →
  `scratch/anka_teori_hizalama_sondasi.json` (özet + sınıf örnek metinleri +
  per-kayıt DECOMP metni — **gelecek çift-hükümler için per-kayıt üretim
  saklanır**, T-0100 sınır dersi).
* `data/eval/anka_teori_hizalama_sonuc_2026-09-23.md` — üç ölçüm tablosu +
  beklenti kıyası + ayrışma + digest tablosu.
* Zincir salt-okunur doğrulama: `venv/bin/pytest` yeşil (bilinen gateway
  istisnası dışında) — zincirde DEĞİŞİKLİK YOK.

## 5. Beyanlar ve sınırlar

* **TANISAL — eşik önerisi YAPILMAZ.** Kapı ROUGE'unun RAW→DECOMP geçişi
  (ECA:481-498) bu analizden sonra AYRI operatör kararıdır; bu ilan onu
  bağlamaz.
* Donmuş yollara yazım YOK (`data/*.bin`, `data/lexicon/**`,
  `src/compiler/**`, `src/llm/tokenizer.py` YALNIZ İMPORT) · kapanmış
  kayıtlara dokunma · `git add -A` YASAK · koşum açılmaz · ölçüm CPU
  (sandbox içinde çalışır; MPS gerekmez).
* Beklenti bandı dışına çıkan her metrik için ayrıştırma ZORUNLU — "temel
  teori çalışıyor" hükmü band içi sayı + ayrıştırma İKİSİYLE birlikte
  verilir.