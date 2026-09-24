# T-0102 · SAPMA BİRİKİMİ AYRIŞMASI İLANI (ön-kayıtlı) — küçük sapmaların üst üste gelmesi (23 Eyl 2026)

**Damga:** 23 Eyl 2026 · **Yazım:** ölçümden ÖNCE · **Görev:** T-0102
(kiralamalar: `data/eval/` dir + `scratch/anka_t0102_sapma_ayrisma_*`) ·
**Koşum YOK** — CPU ölçüm. Üst görev: T-0101 (temel teori, KAPANDI).
Operatör yönü: "compile/decompile sürecinde küçük küçük sapmalar var ve
testlerimizde üst üste gelerek saptırıyor" — her sapma sınıfının payı
ölçülmeden hükümler saptırmaya devam eder.

## 1. Üç ölçüm (SABİT, ilandan okunur)

### (A) 'diger' 15.059 lemma TAM ayrışması (örneklem DEĞİL)
* Katı-önek kök testi — **self-match HARIÇ** (T-0101 danışmasındaki
  artifact: `find_stems` lemma'nın kendisini de dönerdi; "299/300 kök adayı
  VAR" önermesi bu yüzden şişti; düzeltme ilanlı).
* Sınıflar: **S1** (katı-önek kökü YOK — kendisi kök, ör. `strafor`,
  `nükleofilik`) · **S2** (katı-önek kökü VAR + ek yüzeyi, ör.
  `bıçaklanış` → `bıçaklan` + `ış`).
* S2'de: ek yüzeyi uzunluk dağılımı + en sık 20 ek yüzeyi (tam külliyatta);
  S2'nin **TAMAMINDA** compile-recover oranı (kumanda n=300: 0/254 — tam
  külliyatta doğrulanır); S1'in vocab-dışı olduğunun kanıtı (tanım
  gereği, sayaçla).
* **Beklenti (kumanda n=300'den, seed 42):** S2 %80-90 (%84,7) · S1
  %10-20 (%15,3) · S2 recover ≈ 0. Garanti değil — tam külliyatta ölçülür.

### (B) A1-r bin UNK kitlesinin sınıf payları (2.376.511 UNK, %2,38)
* Bin np.memmap uint16 (donmuş — YALNIZ okunur). Her UNK konumu (önceki,
  sonraki) komşuluğuyla sınıflanır: PROPER_NOUN-bağlam / UNK-koşusu
  (çok-jetonlu bilinmeyen) / noktalama-komşu / ek-etiket-komşu / diğer.
* `[sayı]` jeton sayımı ayrı raporlanır (külliyat placeholder incidence).
* **Beklenti (kumanda):** PROPER_NOUN-bağlam payı ~%20-30 · UNK-koşusu
  ~%17-18 · pencere yoğunluğu %2,06-2,49. Garanti değil — ölçülür.

### (C) DECOMP tavan boşluğunun bileşen dağılımı (sapma birikimi)
* T-0101 sonda JSON (`scratch/anka_teori_hizalama_sondasi.json`,
  `46bf1537…`) kayıtlarında token-düzeyi fark sınıfları: buyuk_harf /
  noktalama-tırnak / **[sayı]** / **UNK** / **çözülmemiş-etiket**
  (graph-dışı ek, `POSS_3SG` sınıfı) / diger-kelime.
* Her sınıfın DECOMP ROUGE kaybına katkısı; **bileşenlerin toplamı =
  1 − 0,9473 boşluğuyla hesap tutarlılığı** (bölüştürme kapanmalı —
  çift sayım beyanlı yöntemle önlenir).
* **Beklenti:** kayıt-düzeyi sınıf oranları T-0101'dekilerle uyumlu
  (buyuk_harf %43,6 · bosluk_nokta %27,8 · icerik %28,6); token-düzeyi
  paylar kayıt-düzeyinden küçük olur (bir kayıt tek harf sapması
  taşıyabilir).

## 2. Beyanlar ve sınırlar

* **TANISAL — eşik önerisi YAPILMAZ.** `tech<id>`/entity mekanizması
  KURULMAZ (yalnız ölçüm); tasarım kararı ilanlı ölçümden sonra
  operatöredir.
* Donmuş yollara yazım YOK (`anka_a1r_pretrain.bin` YALNIZ memmap okuma) ·
  kapanmış kayıtlara dokunma · `git add -A` YASAK · koşum açılmaz.
* Kanonik İMPORT: T-0101 sonda betiğindeki kurulum deseni
  (`scratch/anka_teori_hizalama_sondasi.py`) + `kelimeler` (ECA) +
  `rouge_l_score` (evaluate_b1_5_rigorous).

## 3. Çıktılar

* `scratch/anka_t0102_sapma_ayrisma.py` →
  `scratch/anka_t0102_sapma_ayrisma.json`.
* `data/eval/anka_t0102_sapma_ayrisma_sonuc_2026-09-23.md` — üç ölçüm
  tablosu + beklenti kıyası + **T-0101 ile birlikte değerlendirme** bölümü
  + digest tablosu.
* Zincir salt-okunur doğrulama: `venv/bin/pytest` yeşil (bilinen gateway
  istisnası dışında).