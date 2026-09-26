# G6a (T-0137) RAPOR — KÖK-NEDEN TANISI: ÜRETİM-FAZI EKSENLERİ — 2026-09-26 (claude)

Operatör emri (2026-09-26T19:12Z): "üretim eksenlerini ölç: sıcaklık, decoding,
OUTPUT-zarf, sonlandırma". ILAN ≠ RAPOR: ilan
`bahcivan_g6a_uretim_eksenleri_ilan_2026-09-26.md` (`8bcda8d0…`, koşum ÖNCESİ
damga + ilk-koşum dürüst kaydı). Hüküm BETİKTEN
(`scratch/t0137/bahcivan_uretim_eksenleri.py` `1552c6bb…` →
`bahcivan_uretim_eksenleri.json` `00d497cc…`, TOPLAM_SN=1614; elle sayı YOK).
TANISAL görev — **kabul ayracı YOK; KABUL_YOK değişmez.**

## Hükümler (betikten; kıyas D0 = greedy T=0 tavan 128, aynı koşum)

| Dal | ROUGE | tutarsızlık | üretim ort | kesme | dal |
|---|---|---|---|---|---|
| D0 greedy T=0, tavan 128 | 0,0315 | %46,0 | 70,11 kelime | 30/100 | referans |
| E1 T=0,7 | 0,0421 (**+0,0106**) | **%22,0 (−24 pp)** | 54,83 | **5/100** | SICAKLIK_KILIT_AZALDI + ROUGE_ARTTI |
| E2 beam-3 | 0,0158 (−0,0157) | %24,0 (−22 pp) | 50,42 | 20/100 | BEAM_KILIT_AZALDI + ROUGE_DUSTU |
| E3 zarsız istem | 0,0372 (+0,0057)* | %37,0 (−9 pp) | 61,75 | 22/100 | ZARF_GEREKLI (out_id-ilk-10 payı **0,0**) |
| E4 greedy tavan 256 | 0,0324 (+0,0009) | %45,0 (−1 pp) | 97,84 | 17/100 | TAVAN_NOTR |

*E3 ROUGE `<OUTPUT>`-sonrası cevap-kısmı üzerinden — indirekt; asıl ölçüt
out_id-ilk-10 payı 0,0 (0/100): **zarsız istemde model `<OUTPUT>`'a hiç
geçmiyor** — istem-sonu OUTPUT jetonu üretim sözleşmesinin GEREKLİ parçası.

## Kesme kırılımı (tutarsızlığın taşıyıcısı ölçüldü)

| Dal | kesme | kesilenlerde tutarsız | sonlanmışlarda tutarsız |
|---|---|---|---|
| D0 | 30/100 | 30/30 (**%100**) | 16/70 (%22,9) |
| E1 T=0,7 | 5/100 | 4/5 | 18/95 (%18,9) |
| E2 beam-3 | 20/100 | 20/20 (%100) | 4/80 (%5,0) |
| E3 zarsız | 22/100 | 21/22 | 16/78 (%20,5) |
| E4 tavan 256 | 17/100 | 17/17 (%100) | 28/83 (**%33,7**) |

**Sentez (ölçülenler):**
1. **Kesilen üretim ~%100 tutarsız** (tüm dallar) — cevap bitmeden kesildiği
   için son 5 jetonda yüklemsiz kalıyor: tutarsızlık ölçütünün büyük payı
   **tavan-kesme artefaktı**. D0'ın 46 tutarsız üretiminin 30'u kesme.
2. **Tavanı kaldırmak çözmedı (TAVAN_NOTR):** E4'te kesme 30→17'ye düştü ama
   sonlanmış-tutarsızlık %22,9→**%33,7**'ye çıktı — model cevabı bitirdikten
   sonra durmuyor; 128-256 bandında dolgu/döngü üretip geçte sonlanıyor.
   Kusur kesmede değil **sonlandırma davranışında**: cevap-EOS'u öğrenilmemiş.
3. **T=0,7 en güçlü tek müdahale:** kesme 30→5 (döngü kırılıyor + üretim
   kısalıyor) + ROUGE ARTTI +0,0106 (dal-arası, gürültü bandı üstü). Ama
   tutarsızlık %22 >> %5 eşiği ve ROUGE 0,0421 — A9 bandı +0,0230'ın
   (+0,0185) hâlâ ALTINDA; kabul ayracı değişmez.
4. **OUTPUT-zarf GEREKLI** (E3 0/100) — istem sözleşmesi davranışın parçası;
   İLAN-3b §5 kısıtı ikinci kez veriyle desteklendi.

## Koşum-gürültüsü ölçüsü (PK2 — ÖLÇÜM-2 kıyası uyumsuz)

D0, ÖLÇÜM-2 ile BİREBİR parametrelerle (seed 42, n 100, max_new 128, nötr)
koşuldu; ROUGE **0,0315 vs 0,0365**, tutarsızlık **%46 vs %43**, üretim
**70,11 vs 61,32 kelime** — ayrı koşumda greedy bit-deterministik DEĞİL (MPS,
ayrı process). **Koşumlar-arası gürültü bandı ≈ ROUGE 0,0050 / tutarsızlık
3 pp / üretim 9 kelime.** Bu bandın iki sonucu:
- Dal-arası farklar (aynı koşum) eşiği aştıkça güvenilir; eşik-altı dal
  farkları (E3 +0,0057 dahil) koşum-gürültüsü içinde ayrışmayabilir.
- **FAZ-3b zarf denemesi hükmü düzeltmesi (dürüst kayıt):** zarf ikizi AYRI
  koşumdu — ROUGE_DUSTU (−0,0050, "eşik-tam") ve tutarsızlık −7 pp, koşumlar-
  arası gürültü bandının İÇİNE düşüyor; "marangoz zarfı Bahçıvan'a taşınmaz"
  yön hükmü kalır ama **ölçü büyüklüğü gürültü-altıdır** — eşik-tam karar
  gücü YOK. T-0114 kardeş kıyasındaki −15 pp vs −7 pp farkı da aynı bantta.
- İlan edilen eksen eşikleri (±0,005 ROUGE / 10 pp kilit) koşum-gürültüsü
  ölçeğine yakın ilan edilmişti; bundan sonra koşumlar-arası kıyaslar
  aynı-koşum ikizlerle yapılmalıdır.

## Dürüst kayıtlar

1. **İLK koşum pozitif-kontrol DUSTU** — kıyas TÜR-uyuşmazlığı (kanonik
   `uret` decode string listesi döndürür, sarmal id listesi; içerik birebir,
   `max|Δlogits|` 0,0; `uret1==uret2` True). Ayrac değişmeden kontrol
   id-normalize edildi; hüküm İKİNCİ koşumdan (İLAN dürüst kaydı).
2. **Sarmal üretici kanonik kapla ilk-3 örnekte bit-özdeş** (36/128/58 jeton)
   — pozitif kontrol geçti; MPS determinizmi aynı-process içi doğrulandı.
3. E4 tavan 256 tutarsızlık kırılımı İLAN'da beyanlanmıştı; "sonlanmış"
   sınıfı EOS vs `</OUTPUT>` ayrımını yapmaz (sarmal break-davranışı) —
   beyanlı sınırlama, JSON beyan alanında kayıtlı.
4. E1 T=1,0 dalı bu koşumda ölçülmeden (İLAN beyanı: 0,7 kök bulguya bağlı
   izleyen dal); beam uzunluk-normalizasyonu YOK koşuldu (beyanlı).

## Kanıt digest tablosu (tam sha256)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/bahcivan_uretim_eksenleri.json` (artımlı sonuç + hüküm) | `00d497cc54b737c1d764e52aa48c5558dfe4b2be65254bf7868a0ba72aeadbcf` |
| `scratch/t0137/bahcivan_uretim_eksenleri.py` (betik, 2. sürüm) | `1552c6bb5ed43f5115bb186054d30ad076446d9b8c50e2e53cfa158e39efb739` |
| `scratch/t0137/uretim_eksenleri.log` (2. koşum log) | `66c12f5bfd8d460bcc137e60a6bd9c5a0420f329f684e96699c313358244035d` |
| `data/eval/bahcivan_g6a_uretim_eksenleri_ilan_2026-09-26.md` (İLAN) | `8bcda8d0ada94ecac9c3d25c98b3a62ececa33ddfe9cc5b9733fa1f5ef79faed` |
| girdi (salt-okunur): ft `bd68c450…` · heldout `36416b82…` · ÖLÇÜM-2 nötr `be84f93e…` | (önceki faz kanıtlarıyla birebir) |

## Sonuç dalları

- **SICAKLIK_KILIT_AZALDI + ROUGE_ARTTI** (T=0,7) — tek güçlü müdahale;
  **BEAM_KILIT_AZALDI + ROUGE_DUSTU**; **ZARF_GEREKLI**; **TAVAN_NOTR**.
- Kök-neden zinciri: CE_DUSTU (veri öğrenildi) → kusur ÜRETİMDE → üretim
  kusurunun iki bileşeni: **kesme-artefaktı** (~%100 tutarsız) +
  **sonlandırma davranışı** (cevap-EOS öğrenilmemiş; tavan kaldırılsa çözülmez).
- İzleyen eksenler (beyanlı — ölçülmeden ilan edilmez): cevap-sonlandırma
  eğitimi (EOS davranışı için külliyat/sft ekseninde), T=0,7'nin heldout
  dışında tekrarlanabilirliği, marangoz-zarf yerine Bahçıvan-kalıp zarfı
  (FAZ-3b zayıflamasından sonra gürültü-uyarımıyla).
- Kabul hükmü değişmez: **KABUL_YOK** (A9 BAND_ALTINDA + A13 IHLAL; T=0,7
  ROUGE'su bile A9 bandı altında). Checkpoint silme + commit operatör
  kapıları bekliyor. G6b (T-0138) antigravity'ye açık.

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · `uret` /
`render_prompt` / tokenizer / compiler DOKUNULMAZ · data/** salt-okunur ·
elle sayı YOK · `git add -A` YASAK · commit operatör kapısıdır.

---

## FAZ-5 — MARANGOZ vs BAHÇIVAN: değer ve yöntem farkı (kaynaklı kıyas; yeni ölçüm YOK)

Operatör emri: "E4 bitince hüküm ve raporu yaz, Marangoz eğitiminde elde
edilen değer ve yöntem ile olan farkı açıkla". E4 zaten FAZ-4'te kapandı —
hüküm ve rapor yukarıdadır; bu bölüm kıyası ekler. Değerler
`bahcivan_marangoz_kiyas_tablo.py` betiğinden (JSON kaynaklı, elle sayı
YOK; tablo `bahcivan_marangoz_kiyas_tablo.json` `2da72389…`).

### Değer kıyası (betikten)

| Eksen | marangoz ft | Bahçıvan ft |
|---|---|---|
| taban çıpa (ROUGE) | 0,1392 | 0,0236 |
| ft ROUGE (ÖLÇÜM) | **0,1374** (f −0,0018, KORUNDU) | 0,0365 (+0,0129, BAND_ALTINDA) |
| tutarsızlık | **%2,0** | %43,0 |
| aynı-koşum greedy D0 (FAZ-4) | — | 0,0315 / %46,0 |
| T=0,7 dalı (FAZ-4) | — | 0,0421 / %22,0 |
| cevap-CE (taban→ft) | A-ekseni +%0,446 (unutma YOK) | 5,2103→3,3682 (CE_DUSTU 1,8421, 28σ) |
| külliyat çift | 5.400 | 2.199 |
| soru/cevap ort (kelime) | 12,37 / 21,08 | 4,14 / 33,88 |
| soru tekrarı | %37,1 | %0,0 |

### Yöntem farkının açıklaması (ölçülenler + kaynaklı yorum)

1. **Yöntem ÖZDEŞ kurulum — fark veride:** ikisi de P3 sürücü, aynı mimari
   (6/6/768/RoPE/tying-false), aynı mix deseni (%20 wiki + %40 ceket +
   %40 sft), aynı carve oranı (%10, kesişim 0), aynı lr bandı (1e-4).
   Marangozda işe yarayan yöntem Bahçıvan'da aynı işledi; rampa farkı
   yöntemden değil aşağıdaki dört eksenden.
2. **Arz ölçeği (5×):** marangoz 5.400 çift vs Bahçıvan 2.199 — marangoz
   rampası 1,6 epoch'ta 0,1014'e ulaştı (P3); Bahçıvan 6,8 epoch'ta
   +0,0129 kaldı. Ama kök-neden CE ölçümü verinin ÖĞRENİLDİĞİNİ (28σ)
   gösterdi — ölçek ekseninde kök-neden YOK; kusur üretimde.
3. **İstem-cevap biçimi tersine asimetri:** marangoz 12→21 (uzun soru,
   kısa cevap — üretim 128 tavana az çarpıyor); Bahçıvan 4→34 (kısa soru,
   uzun cevap — model marangoz biçimine kayıp üretimi uzatıyor: 61-70
   jeton-parça, referans 33,9) ⇒ **kesme %30** ve kesilenler ~%100
   tutarsız. Marangozda tutarsızlık %2 — biçim-tavan uyumu farkın taşıyıcısı.
4. **Taban çıpası farkı (0,1392 vs 0,0236):** marangoz yeteneği tabanda
   VAR (seg_3-devralma) — fine-tune "koruma + derinleştirme" rejiminde
   işledi; Bahçıvan içeriği tabanda YOK — "yeni yetenek" rejimi gerektirdi
   ve rampa kısa kaldı. A9 bandı (+0,0230) bu yüzden Bahçıvan rampasında
   daha yüksek arza bağlı.
5. **Sonlandırma davranışı (FAZ-4 yeni):** cevap-EOS öğrenilmemiş — tavan
   kaldırılsa çözülmüyor (TAVAN_NOTR; sonlanmış-tutarsızlık %33,7). Marangoz
   cevap-kalıbında bu davranış ölçülmedi (beyanlı); kesme-kırılım ölçümü
   Bahçıvan koşumuna özeldir.
6. **Zarf uyumu:** ROL_ZARFI marangoz metni marangozda doğal; Bahçıvan
   dikeyinde FAZ-3b unk 6,8× ve gürültü-bantlı etki — taşınmaz.

### Dürüst kayıt (ölçüm-tanım notu)

Üretim tarafındaki "kelime" sayımı (`kelimeler(gm)`, `gm = " ".join(decode)`)
aslında **jeton-parça** sayımıdır — FAZ-4 ölçümünde `ort_kelime` ile
`ort_jeton` birebir aynı çıktı (70,11/54,83). Referans tarafı gerçek
kelime. İki kanat da kapın kanonik ölçümüyle aynı yöntem olduğundan
kıyaslar aynı ölçekte kalır; ama "61 kelime vs ref 34 kelime" ifadelerindeki
üretim sayıları jeton-parça ölçeğidir (kök-neden raporundaki değerler dahil)
— uzunluk oranları (üretim/ref ≈ 2,0) bu ölçekle beyanlıdır.

## Sınırlar (FAZ-5)

Kıyas bölümü YENİ ölçüm koşmaz; tüm değerler önceki faz kanıt
JSON'larından betikle okunur (`bahcivan_marangoz_kiyas_tablo.py`) —
yeni ilan/hüküm beyanı YOK; kabul ayracı değişmez (KABUL_YOK).