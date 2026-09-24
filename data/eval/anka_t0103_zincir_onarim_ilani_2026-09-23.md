# T-0103 · ZİNCİR ONARIM İLANI (ön-kayıtlı) — decompiler + graph + self-match (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Yazım:** ölçümden ÖNCE · **Görev:** T-0103
(kiralamalar: `src/compiler` dir + `data/eval` dir +
`scratch/anka_t0103_zincir_onarim_sondasi.*`) · **Koşum YOK** (CPU ölçüm) ·
Üst görev: T-0102 (KAPANDI). Operatör onayı birebir: *"Bunları uygula ve
ölç. 1. Katman 3 (decompiler kesme + graph zincirleri) — en ucuz,
0,9473 → yukarı; graph zincirleri 13.064 lemma'yı UNK'tan kurtarır."*

## 1. Kumanda bulguları (ilansız küçük problar — hüküm üretmez; P5-A deseni)

Operatör hipotezinden farklı ama daha basit bir mekanizma çıktı; betikle
ölçüldü, ilana yazılmadan uygulamada kullanılmaz:

1. **Affix tag'ler vocab'da VAR** (örneklem 16/16: `INF_mA`, `INF_Iş`,
   `PART_An/DIk`, `DERIV_lA/lAn/lAş/lIk/lI/sIz/CI`, `POTENTIAL`,
   `TENSE_PAST`, `CASE_ACC` ✓).
2. **En sık S2 ek yüzeyleri graph'ta ZATEN VAR** (`me/ma`→`INF_mA`,
   `abil/ebil`→`POTENTIAL`, `ış/iş`→`INF_Iş`, `la/le`→`DERIV_lA`,
   `laş`→`DERIV_lAş`).
3. **Gerçek engel — önemsiz self-match parse:** S2 kelimeleri zaten
   lexicon lemma'sı; `find_stems` kelimenin KENDİSİNİ en-uzun kök olarak
   döndürüyor ⇒ `compile` tek-morfem öz-parse ediyor
   (`bıçaklanış` → `tv=['bıçaklanış']`) ⇒ tanım gereği vocab-dışı ⇒ UNK.
   **Bypass kumandası (n=300, seed 42, S2):** recover 112 = **%37,3** ·
   kök-OOV 81 (%27,0 — donmuş vocab'da çözülmez) · yol-yok 107 (%35,7).
4. **yol-yok alt sınıflama (betikle):** id-vocab-içi-geçiş-uyumsuz **16** /
   yeni-id-kompozisyon **91** (yeni id vocab'da YOK ⇒ donmuş vocab altında
   ölü — Katman 2/vocab yenileme işi). Ölçümlü adaylar:
   `DERIV_lAş/lAn` NOUN kaynaklı eksik (`horozlaş`, `uyaklan`) ·
   CAUSATIVE/PASSIVE state'leri **terminal işaretli DEĞİL** (`hiddetlendir`,
   `sövdür`, `aratıl`, `böldürül` — bare lemma yüzeyi yol bitmiyor).
5. **Decompiler'da gerçek bir hata — META kök çarpışması:**
   `decompile_tags:84` `root_lemma.lower() in META_TAGS` — `ara` hem gerçek
   lexicon kökü (`ara-` fiili) hem meta marker; `ara POSS_3SG CASE_LOC_N`
   affix yoluna girmeden ham etiket kalıyor ⇒ **15/539 kayıttan**
   T-0101'in çözülmemiş-etiket sınıfının en büyük dilimi (`arasında`
   üretilmiyor).
6. **Kesme/tırnak kaybı ENCODE tarafında:** `src/llm/tokenizer.py:306`
   (mode=False yolu) apostrophe'u düşürüyor; tipografik tırnak `“` regex
   alternatiflerinin hiçbirine uymuyor. **Tokenizer DONMUŞ** — bu sınıf
   (T-0102'nin 279 kelime sınıfının ana bileşeni) bu görevde ÇÖZÜLMEZ;
   beyan edilir. Decompiler'ın kendisi sağlıklı (`İstanbul'a`, `Ali'nin`
   ✓ probe).

## 2. Uygulama beyanı (3 dosya — donmuş `src/compiler/**`, kiralamalı)

1. **`src/compiler/core.py`:** `CrystalCompiler.__init__`'e OPTIONAL
   `vocab: Optional[Vocabulary] = None` — varsayılan None ⇒ mevcut 11+
   arama yerinde davranış BİREBİR aynı. `compile()` içinde self-match
   bypass YALNIZ OOV'de: `vocab` VARSA ve kelimenin kendisi vocab-dışıysa
   self-match adayı düşülür (diğer adaylar varken) — vocab-içi
   kelimelerin parse'ı hiç değişmez.
2. **`src/compiler/morphotactics.py`:** yalnız semantik-doğru, vocab-içi
   id'li eklemeler: `mark_terminal(VERB_VOICE_CAUSATIVE, VERB_VOICE_PASSIVE)`
   + `DERIV_lAş`, `DERIV_lAn` geçişlerinin `NOUN_ROOT` kaynaklı halleri.
   COPULA_AORIST ile CAUSATIVE `DIr` çarpışması bilinçli EKLENMEZ
   (yanlış-parse kirliliği).
3. **`src/compiler/decompiler.py`:** META kök çarpışması düzeltilir —
   meta-kök takip eden tag'lerin TÜMÜ suffix ise affix yoluna girer
   (`arasında` üretilir); değilse eski davranış (ham birleştirme) korunur.

## 3. Ölçümler (SABİT, ilandan okunur)

* **(A) S2 tam külliyat recover (13.064):** ÖNCE %0,03 (T-0102 beyanlı,
  yeniden koşulmaz) · SONRA edits-ile. **Beklenti bandı (kumanda n=300):
  %35-45.** Band dışı ⇒ ayrıştırma zorunlu.
* **(B) 539 roundtrip DECOMP:** ÖNCE **0,9473** (T-0101 beyanlı — yeniden
  koşulmaz) · SONRA sonda'da (compiler vocab-enjeksiyonlu) ölçülür.
  **Beklenti bandı: 0,950-0,970** (mono-tonal: SONRA < ÖNCE ise
  değişiklik GERİ alınır — fail-closed).
* **(C) Compile determinizm:** 539 kayıdın iki koşumu birebir.
* **(D) Yüzey doğrulama:** `venv/bin/pytest` yeşil (bilinen gateway
  istisnası dışında) + `venv/bin/python scripts/olcum_kabi.py` rc=0.

## 4. Sınırlar ve beyanlar

* **ECA kapısı DEĞİŞMEZ** (satır 368-369 vocab'sız kurulum duruyor) —
  kapının bu onarımdan yararlanması (ECA'ya vocab geçişi) AYRI operatör
  kararıdır; DECOMP kapı geçişi zaten açık konu.
* Donmuş `vocab`/`tokenizer` sınırı: kök-OOV %27 + yeni-id/kompozisyon
  91/107 + encode-tarafı kesme sınıfı bu görevde çözülmez (Katman 2 /
  vocab yenileme işi).
* S1 (`tech<id>` hedef sınıfı, %13,25) BU görevde kurulmaz — operatörün
  öngörüsü doğrultusunda Katman 4 etkisi Katman 3 ölçümünden sonra
  değerlendirilecek.
* `git add -A` YASAK · kapanmış kayıtlara dokunma · hüküm ölçümleri
  yalnız ilanlı sonda betiğinden.

## 5. Çıktılar

* `scratch/anka_t0103_zincir_onarim_sondasi.py` → `.json` (per-kayıt
  üretim DAHİL — gelecek çift-hüküm için saklanır).
* `data/eval/anka_t0103_zincir_onarim_sonuc_2026-09-23.md` — beklenti
  kıyası tablosu + değişiklik listesi + digest tablosu.