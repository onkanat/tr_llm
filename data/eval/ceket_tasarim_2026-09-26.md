# ANKA CEKET TASARIM DOKÜMANI (G3a, T-0135)

**Tarih:** 2026-09-26 · **Yürütücü:** claude (onaylayıcı/tasarım kanadı) · **Görev:** T-0135 (G3a, TUR-3 operatör onayı)
**Operatör emri (birebir):** *"Çeket modelin tasarım şekli dökümante edilecek; Bahçıvan ve Berber bu doküman üzerine geliştirilecek ve Temel model üzerinde test edilecek."*

Bu belge, ceketin (uzmanlık katmanının) tasarımını ölçülmüş bulgulara sabitler. **Her iddia bir
T-kimliğine ve §12'deki digest tablosundaki tam sha256'ya bağlanır; çıpasız cümle yoktur.** Bahçıvan
(G6) ve Berber bu belgeyi giriş şartnamesi olarak kullanır. Kaynak dosyalar değiştirilmedi; tüm
digest'ler bu oturumda `shasum -a 256` ile ölçüldü (elle sayı yok, önek yasak).

---

## §0 Beyanlı sınırlar (tüm bölümleri bağlar)

1. **ESIK_ROUGE 0,3221** (`scripts/evaluate_carpenter_anka.py:70`) ve **TAVAN_ROUGE_DECOMP 0,9509**
   (`:83`) **DOKUNULMAZ**dir — P5 Dal-K kalibrasyonu (tavan 0,3835 × 0,84) [D22] ve P5akol DECOMP
   tavan ölçümü [D23] ile sabitlenmiştir; kanarya satırları (`scripts/olcum_kabi.py:326,330`,
   [D21]) eski değerin geri yazılmasını DÜŞÜRÜR. Ceket tasarımı bu eşikleri değiştiremez.
2. **Mimari tartışma dışı:** 6 katman / 6 kafa / 768 gizli / RoPE / weight-tying YOK —
   `data/anka_base_v2.meta.json` `mimari` alanı birebir [D19]; tokenizer DONMUŞTUR
   (frozen.json Kural 2) — ceket tasarımı tokenizer'a DOKUNMAZ.
3. **Kapalı deneyler yeniden açılmaz:** F4 kararı (T-0064, [D2]) ve lr ekseni kapanışı, epoch-arzı
   kapanışı (T-0097), decoding tanısı kapanışı (T-0110…T-0112) kapanmış hükümlerdir; yalnız yeni
   ölçülmüş eksenle gerekçe güncellenir (bkz. §1-3 notu).
4. **Ölçüt-ölü sınıfı:** her yeni ölçütte pozitif kontrol ZORUNLU ([D18] §5; T-0121/T-0123 dersi).
5. **Kapanış-kanıtı kusur sınıfı yasağı:** ilan ve rapor AYRI dosyada; ILAN_YOL == RAPOR_YOL betikte
   RED'dir (T-0130 kusuru, [D16] §5 + [D18] §2). Elle yazılmış sayı YOK; hüküm betikten çıkar.

---

## §1 Ne zaman CEKET, ne zaman TABAN — karar ağacı

**Mimari çerçeve (operatör kararı, 15 Eyl 2026 — bağlayıcı):** *"Kristal LLM her zaman kaslı bir z
kuşağı lise öğrencisi"* — taban DEĞİŞMEZ genel-Türkçe yetkinliğidir; carpenter ve benzeri uzmanlık
modülleri DEĞİŞKEN cekettir, tabanın ÜSTÜNE giyilir [D1]. Regenerasyon planı F4 aşaması ceket
aşamasıdır [D1].

**Ölçülmüş ayrım kuralı:** iyileştirme isteği konumlandırılırken tek soru — **DİL BECERİSİ mi,
ALAN UZMANLIĞI mı?**
- Dil becerisi (noktalama, rakam, kesme işareti, akıcılık) → **TABAN**. Ölçüm: 14/14 `.bin`'de
  max token id ≤ 32136 (noktalama bloğu 32137'de başlar — hiçbir model noktalama görmedi);
  tek dosyada 3.744 rakam karakterinin 1.678'i (%44,8) `<UNK>` [D1, T-0020/T-0021 turu].
  Karar (operatör, 15 Eyl): rakam + kesme tabana taşınır; tokenizer değişmez, yeni taban sürümüyle gelir [D1].
- Alan uzmanlığı (marangozluk kalıp/cevap davranışı, RAG bağlam kullanımı) → **CEKET** [D1].

**Karar ağacı (uygulama sırası):**
1. Kusur DİL BECERİSİ mi? Evet → taban geçişi (ankabase_v2 gibi; ön-eğitim/devam eğitimi), ceket değil.
2. Kusur tabanda DEĞİL, tam ince ayarda doğdu mu? Kanıt deseni: öğretmen zorlamalı ppl — taban
   **33,38 (en iyi)** · seg_1 76,85 · seg_6 599,72 · seg_6+modül 86,41 ⇒ taban sağlam, bozukluk
   aşağı akışta [D5-chain; ölçüm T-0096 zinciri, P2 3.1]. Ceket/modül ONARIR (bkz. §2).
3. Yetenek (içerik üretimi) cekete YÜKLENMEZ (§2 — beş tekrar); ceket kalıp + cevap-davranışı +
   rol zarfı öğretir; içerik gereksinimi taban külliyatında karşılanır (P4: epoch arzı doydu,
   boşluk külliyatta [D6-bağ; T-0107/0108/0109 arz zinciri]).
4. **Taban-çıpa deseni (kardeş testi):** ceket adayının eğitilmemiş tabanı kanonik kapta
   ROUGE 0,0000 vermelidir — ölçüt canlıysa eğitilmemiş taban SIFIR üretir [D18 taban-çıpa
   beyanı]; ölçüt canlı DEĞİLSE (T-0121 TF-tanısı ÖLÇÜT_ÖLÜ) ölçüt çöpe gider, pozitif kontrolle
   yeniden yazılır [D18].

**Bugünkü taban:** `data/anka_base_v2.pt` (G2/T-0133) — seg_3 zincirinden kanonik
`resize_state_dict` ile 33.114'e sıfır-kırpma hizalı; kafa (33114, 768) ×3 tensör benim torch
okumamla kanıtlı; `[SOZLESME_UYARI]` 0; manifest kendi-kendine tutarlı (model+vocab digest 2/2
birebir, benim yeniden hesabım) [D19]. Mühür **'2.0-pre-seal / mühür bekliyor'** — §9'a bak.

---

## §2 Delta BİLGİ ekleyemez, ONARIR — beş bağımsız tekrar

**Ölçülmüş zincir (T-0096 altı deneme + P2 Aşama 3.2):**

| tekrar | eksen | koşum | sonuç |
|---|---|---|---|
| 1 (r29) | kapasite | r 16→64 (×4 parametre) | ROUGE farkı 1,70 SE — ayırt edilemedi [D20/D20a] |
| 2 (r31) | giriş | +embedding | kesişim %1 → %0 [D20b] |
| 3 (r32) | bütçe | 1000→3000 adım | kesişim %1 → %0; ezber %28→%38 [D20c] |
| 4 (r33) | yetkin temel | temel = seg_6 | kesişim %10 → %10 [D20d] |
| 5 (P2-3.2) | sağlıklı taban | Temel 2.0+ince ayar (CE 3,4438), 4.000 adım r16 | yetenek **+0** (kesişim %0→%0, ROUGE 0,0056→0,0068 — gürültü altı); **LM bedeli +%4,03** [D5] |

(r34 kapanış turu — H-A elendi; seg_1 temelinde modül yine içerik eklemedi: kesişim %0→%0,
ROUGE 0,1097→0,0997 [D20e].) Her koşumda **biçim** kazanıldı (lm_head hedeflenince
ROUGE 0,0191→0,0696), **içerik kazanılmadı** — çıktı zarf yankısı.

**Onarım tekrarlanır:** r33'te seg_6 Wikipedia hasarı %68 geri alındı (A CE 6,2086→4,3902, −%29,29)
**yetenek sabitken**; P3'te carve geri dönüşü ROUGE 0,0056→**0,1014** (18×) yaptı, Wikipedia CE
+%1,43 ile bedel kapısını geçti [D6]. P2-3.2'de sağlıklı tabanda da onarım/davranış kalıcı, içerik
+0 [D5].

**LM bedeli tabanla küçülüyor (ölçülmüş):** pilot anka_a1r tabanında (mix'i görmemiş) 1.000
adımda modül CE **+%13**; Temel 2.0 tabanında 4.000 adımda **+%4,03** [D5] — bozuk-hedef
artefaktı hipoteziyle uyumlu: taban SFT mix'i gördükçe onarılacak hasar ve bedel azalır.

**Mekanizma notu (ölçülmüş):** onarım "tabana dönüş" DEĞİLDİR — Frobenius ‖·−taban‖_F
674,17→804,42 (+%19,3): tabandan UZAKLAŞAN bir düzeltme yönü; yetenek eksenlerine dokunmuyor [D5].

**CEKET TASARIM KURALI (§2 çıkarımı):**
1. Ceketten **bilgi/icerik bekleme** — beş bağımsız tekrar kanıtlanmış sınır. Ceket öğrendiği şey:
   biçim, kalıp-davranışı, rol zarfı uyumu, onarım.
2. Ceketin tabana verdiği **LM bedeli beklenen ve kapılıdır** (§3); bedel taban ne kadar SFT-mix
   sağlıksa o kadar küçük — ceket EĞİTİMİ taban değişikliği gerektirmez.
3. "İyileşti" iddiası için Frobenius sınaması ZORUNLU (tabana-dönüş ≠ iyileşme; iki kardeş ölçüm).
4. İçerik gereksinimi görülürse çözüm ceket değil **taban külliyatıdır** (P4 hükmü: adım/epoch/pay
   üçü kapandı, boşluk külliyatta [D6-bağ]).

---

## §3 LM bedeli kapısı 0,0230 (2 × ROUGE SE)

- Kanonik kapta ROUGE-L standart hatası **0,0115** (n=100) — bedel kapısı **2×SE = 0,0230**.
  Hüküm mantığı: `f = ceket − çıpa` ROUGE farkı; f ≥ +0,0230 ARTIŞ_SİNYAL · |f| ≤ 0,0230
  KORUNDU · f ≤ −0,0230 KALİTE_BEDELİ. T-0115'te ilan edildi ve uygulandı
  (f −0,0012 → KORUNDU) [D8]. T-0126/27/28 hükümlerinde LM_BEDELİ False satırları aynı kapıyla
  yazılmıştır [D13][D14][D15].
- **DECOMP eksenli ölçümlerde TAVAN_ROUGE_DECOMP 0,9509 çıpadır** [D22][D23]; ham decode
  tavanı decompiler artefaktıdır (ham kaybın %87'si) — ham ROUGE sayısı DECOMP tavanıyla
  kıyaslanamaz (T-0106/T-0107 kapanışı) [D18].
- **Kural:** ceket eğitim koşumu tamamlandığında ROUGE çıpayla karşılaştırılır; KALİTE_BEDELİ
  dalında ceket kabul edilmez (G6 Bahçıvan şablonunda aynı kapı). Eşik 0,3221'e dokunma = §0.1.

---

## §4 Rol zarfı mekanizması

- **T-0114 (ölçüm):** zarflı koşullama kilit doğumunu azaltıyor — S3 segmentinde −15 pp [D7].
- **T-0115:** zarflı koşumda kanonik ROUGE **korundu** (f = −0,0012, kapı 0,0230 içinde) [D8].
  Dikkat: zarflı koşum carve-kırıcıdır — `egitim_satirlari` ham-string setiyle held-out'u çıkarır;
  zarflı kopya `--train-source` ile nötrle birebir eğitim kümesine oturtulmalı [D8 ilan/beyan].
- **T-0120/T-0122 (kalıcı entegre):** `ROL_ZARFI` sabiti `src/llm/prompt_contract.py`'de;
  `render_prompt(..., rol_zarf=False)` salt-eklenti; `--rol-zarf` CLI. Tokenizer DONMUŞ,
  dokunulmadı. PC-1: zarflı üretim T-0115 `zarfli.json` ham_gm 100/100 birebir —
  ROUGE 0,1342 (zarflı çıpa) · nötr çıpa 0,1354; **iki çıpa karıştırılmaz** [D9].
- **CEKET TASARIM KURALI:** ceket üretim/ölçüm koşumları çıpa damgası SEÇİCİDİR — `--rol-zarf`
  verildiyse 0,1342 zarflı çıpa, flag'siz 0,1354 nötr çıpa; raporda hangi çıpanın kullanıldığı
  koşum-beyanıyla yazılır [D9]. Zarf şokunun bedelsizliği ölçülmüştür (T-0115) — ceket şablonu
  zarfı kalıcı benimser, tokenizer dokunuşu YASAKTIR.

---

## §5 Kalıp-dengesi zinciri (kilit mekanizması → müdahale ölçümleri)

**Kilit teşhisi (T-0124/T-0125):** tutarsız cevapların altkümeleri arza YAKIN — "arz eksikliği"
ÇÜRÜDÜ; mekanizma **kalıp-kilit**: `ne_ise_yarar` NEG %24,2 vs POZ %3,5 [D11]. Arz KARMIŞIK
(genel pay J≥0,8 %0,23) ama kalıp-içi 8,8× homojen (ne_ise_yarar %3,69 vs diger %0,43, 16,34×SE)
— kilit KÜRESEL değil **KALIP-ÖZEL** [D12].

**Müdahale ölçüm zinciri (tek değişken külliyat bileşimi, taban seg_3, taban-çıpa deseni):**

| kol | külliyat | post NEG ort ΔCE | pozitif işaret | hüküm |
|---|---|---|---|---|
| T-0123 baseline | kanonik | −1,52 | 0/33 | referans [D10] |
| T-0126 AZALT | ne_ise_yarar'ın %50'si düşürüldü | −1,2379 | **0/33** | TAŞIMADI [D13] |
| T-0126 AZALT+COGALT | + 3 grup 2× kopya-oversample | −0,9646 | **5/33** | KISMİ_SİNYAL — ilk pozitif kırılma [D13] |
| T-0127 ÇEŞ | ne_ise_yarar cevapları Gemini paraphrase (J≥0,8 %0,00) | −1,2461 | **1/33** | TAŞIMADI [D14] |
| T-0128 BİRLEŞİM | ÇEŞ külliyat + 2× oversample (14.109 satır) | −0,7382 | **10/33** | KISMİ_SİNYAL — REKOR [D15] |

Sıralama **BİRLEŞİM > denge > çeşit ≈ azalt > baseline**; denge + çeşitlilik **sinerjik**
(kalıp kilitlenmesi ~üçte bir oranında çözülüyor; tek başına yetersizler birlikte taşıyor).
ROUGE: 0,1439 (+0,0085), LM_BEDELİ False; ezber %0 [D15]. Ders: **kopya-oversample dengeyi
taşıyor, çeşitliliği değil**; cevap-çeşitliliği tek başına kilidi kırmıyor — kilidin omurgası
soru-KALIBININ dar temsili [D13][D14].

**CEKET TASARIM KURALI:** ceket külliyatında kalıp-ağırlık dengesi (hedef kalıbı azalt + komşu
kalıpları çoğalt) VE cevap-çeşitlendirme **birlikte** uygulanır; tek kol tasarlanmaz. Eğitim
hükmü eşiği: post NEG ort ΔCE ≥ −1,0 VE pozitif işaret > 0 (ilanlı, koşum-öncesi; T-0126
eşikleriyle aynı bant [D13 ilan]).

---

## §6 Ceket-üstü tutarsızlık bedeli — G4 hükmü (T-0130)

- T-0128 sonrası kanonik kap tutarsızlığı **%11,0 (11/100)** — denge çıpasının (%5) 2 katı;
  eşik ≤%5 İHLAL (post-hoc etiket doğru yazılmıştır) [D15].
- **G4 teşhis hükmü (benim bağımsız tekrar koşumum, betikle):**
  `TUTARSIZLIK_TAMAMEN_ÇOĞALTMA_BÖLGESİNDE` — 11/11 tutarsızlık **çoğaltma bölgesinde**
  (%12,22, Wilson [6,96; 20,57]); paraphrase bölgesinde **0/10** (%0,00); **nedir %38,89** (7/18),
  nasil %6,25, diger %5,36 [D16][D17].
- **ci_ortusuyor: True (beyanlı):** n=10 küçük; Wilson üst sınırlar örtüşüyor — hüküm gözlemsel
  %100 dağılımına dayanır, n=10'dan bağımsızlıksız değildir [D16].
- Hata biçimi: `not_yuklem` (cümle sonu yüklem eksikliği/kesilme) + `teknik çözüm :` yankı
  kalıbının tekrarı — G4 tam envanter 11/11 [D16 §3].
- **CEKET TASARIM KURALI:** tutarsızlığın kaynağı çeşitlendirilmiş cevaplar DEĞİL; çoğaltılmış
  satırlardır. Dolayısıyla ceket külliyatında **aşırı çoğaltma (oversampling) tavanlıdır** ve
  tasarım hedefi çoğaltma oranı değil **cevap-sonlandırma kalıbı dengesidir** (yüklemin tamamlanmış
  cümlelerle temsili). Ceket kabulünde tutarsızlık çıpa **%5** (T-0128 %11 İHLAL deseni) izlenir;
  tutarsızlık ölçümü G4 kardeşiyle (offline morfem analizi) yapılır; **ILAN_YOL == RAPOR_YOL
  yasaktır** — ilan ayrı dosyada, ön-kayıt digest'i notta saklanır [D16][D17][D18].

---

## §7 Derleme sözleşmesi (ceket külliyatı → bin)

1. **D3-istisna çift-kayıt yapısı ZORUNDA:** kanonik marangoz bin'i satır başına İKİ kayıt taşır —
   (1) ham `render_example` + (2) normalize SFT ("Ahşap uzmanı olarak cevapla." + ":" split q).
   T-0127 kusuru: tek-kayıt derleme 364 adım üretti (beklenen ~687), SFT maske dağılımı kaydı,
   kıyas kırıldı; FAZ 2 öncesi bin-yapı denetimi koşumu kurtardı [D14].
2. **Bin dekod zorunlu (her FAZ 1'de):** BOS/EOS segment say + `<OUTPUT>` sayısı / kayıt sayısı /
   jeton-per-satır — meta'ya değil YAPINA bak [D14].
3. **Kanonik fonksiyon İMPORT edilir, kopyalanmaz** (derle_ces.py 2. sürüm doğru yol); T-0134
   (G3b) `src/llm/dataset_compiler.py` + `bin_dekod_dogrula` kütüphanesini bu sözleşmeyle üretir;
   kabul **bit-özdeş bin** (eski betikle tam sha256 birebir) [T-0134 şartnamesi].
4. Külliyat-vocab uyumu fail-closed: sözlük külliyattan küçükse derleyici/eğitici görünür
   `[SOZLESME_UYARI]` basar, sessiz kırpma YOK (T-0132 benim iki-dal koşumumla kanıtlı) [D18].

---

## §8 Bahçıvan / Berber şablonu (G6 girdisi)

G6'da Bahçıvan (ceket eğitici) ve Berber (değerlendirici) bu belgenin §1-§7 kuralları üzerine
geliştirilir; **Temel model (anka_base_v2) üzerinde test edilir.** Zorunlu şablon maddeleri:

1. **Taban-çıpa deseni:** eğitilmemiş taban kanonik kapta ROUGE 0,0000 üretmeli (ölçüt canlılığı
   pozitif kontrolü) [D18].
2. **Eşik önceden ilan:** post NEG ΔCE bandı (−1,0; 0) ve işaret payı eşiği, LM bedeli kapısı
   0,0230, tutarsızlık çıpa %5 — koşum-öncesi, **ayrı ilan dosyasında** (ILAN≠RAPOR) [D16][D18].
3. **Çıpa damgası:** `--rol-zarf` kullanımında 0,1342, nötrde 0,1354 — raporda beyanlı [D9].
4. **Derleme:** §7 D3-istisna + bin dekod FAZ 1'de; T-0134 kütüphanesi kullanılır.
5. **LM bedeli + Frobenius ikilisi:** "iyileşti" iddiası için her iki sınama ZORUNLU [D5].
6. **Artefakt disiplini:** model `.pt`'leri doğrulama sonrası silinir, digest tabloda dondurulur;
   disk temizliği kapanış şartnamede (T-0126/27/28 pratiği) [D13][D14][D15].
7. **Operatör kapıları otomatikleşmez:** commit, .pt silme, eşik/TAVAN, donmuş yol, görev açma —
   SPEC v1.2 protokolü ile claude inbox izleyicisi yalnız FAZ onaylarını taşır [D15].

---

## §9 Versiyon mührü (G2 bağlantısı)

`data/anka_base_v2.meta.json`: `version = "2.0-pre-seal"`, `muhur_durumu = "muhur bekliyor"`
(T-0133 manifest, benim bağımsız okumam) [D19]. **Bu belge (G3a) mührü kapar:** §1-§8 kuralları
taban-2.0 üzerine ceketlerin (Bahçıvan/Berber, G6) bağlayıcı tasarım şartnamesidir. Mührün
kapanışı OPERATÖR KAPISIDIR — bu belgenin teslimi ile talep edilir; otomatikleşmez.

---

## §10 Tam digest tablosu (tüm çıpalar; önek YASAK, tamamı bu oturumda ölçüldü)

| id | artefakt | sha256 |
|---|---|---|
| D1 | `data/eval/regeneration_plan_2026-09-15.json` | `96fd4e3f019611415811a75c46762864495431364b0b6bce27c2b6c05e7371f5` |
| D2 | `data/eval/f4_decision_2026-09-18.md` | `048886ffff1cfe7b331f05d6a8dc0f1240249ff1cb492df5465c6498b320c5f3` |
| D3 | `data/eval/t0067_role_2026-09-18.md` | `765c5494b437fefb1efe8b2ac3c86569133d4cd9d3f6995dd82a9fa1f8e0858f` |
| D4 | `data/eval/t0068_belge_ayristirma_2026-09-18.md` | `647ebf458605f058fa51a9598d3f5bc6f7390b3a2af2e2f86e87b7500deb01e6` |
| D5 | `data/eval/anka_p2_asama32_modul_sonuc_2026-09-23.md` | `05bc07cd9cd0c22ebe0a416bc5b0815a348b0749055584cb004a6fe72dbd8370` |
| D6 | `data/eval/anka_p3_sonuc_2026-09-23.md` | `a603ae2bf71820a051a578702959f296727db8c024d3dd5b546b379170b2b718` |
| D7 | `data/eval/anka_t0114_sonuc_2026-09-25.md` | `907f75567594930b201c14f7780cdafa5f2254998434be23532104bb10710d78` |
| D8 | `data/eval/anka_t0115_sonuc_2026-09-25.md` | `3cf0602abc9d362d123b3e70f09e1124b76409f398936f1337a68a57f679e094` |
| D9 | `.agent-bus/notes/T-0120.md` | `1fe5c5712da4fed45fae839da55e3810bc55b184cac6829088c60c0c91d15f6b` |
| D10 | `.agent-bus/state/results/T-0123.json` | `7c71352612af167507473e3e6860de129225b71d7d1504a86f797bdac1a44851` |
| D11 | `.agent-bus/state/results/T-0124.json` | `e7965d325cdcea4da9b1f0059cc964425df428bce527f499c125454795bdbc6f` |
| D12 | `.agent-bus/state/results/T-0125.json` | `dbb587e7888518f1d69f43a5d64d66a67cc2016b51d83ed6e6259b67514fca3f` |
| D13 | `.agent-bus/state/results/T-0126.json` · ilan `anka_t0126_ilani_2026-09-25.md` · sonuc `anka_t0126_sonuc_2026-09-25.md` | `47e18a2dff7c4f6731a99fd5610c85e8f9517ccedf5446ce073811bf0260e103` · `5fe3c8d2c95c176fd58307c76564b7c877e8418e4316d386a4a6f38004934013` · `7795708501303a3358ca5d40fb15858e7f19cede8a76293b62c3172177abc5ee` |
| D14 | `.agent-bus/state/results/T-0127.json` · ilan · sonuc | `1a1d30b3b656a2e32630341c43fe6dffc1ef704b6ff69c82dc70d5e98a3eecf5` · `7b515708664366fab9643c3e204ef79670613f2d3586224e12acf77c3da13a25` · `76633799944ed9c3606b1e7581bc1ac7ecc5892c10c5a193318d3b4a2bd39cac` |
| D15 | `.agent-bus/state/results/T-0128.json` · ilan · sonuc | `2d628554a167f150bae47c551fa12454aac12af936dbc77d85cf1aa636e2398a` · `e6cd51e929a1a3b9522d2177ba4cea34da4e056d4f1b0e9e90ae55a8ce720eb4` · `59362b9746fe45cf97ac765d71123784ad3fae18d37b49ca6a119e85e95d99bf` |
| D16 | `data/eval/anka_g4_tutarsizlik_teshisi_2026-09-26.md` | `b206f925bc5ead692de88425ee012a078f596aeb45ce91c9703bdfc5611e0e80` |
| D17 | `scratch/t0129/g4_teshis/g4_sonuc.json` (benim koşum) | `29637787e1f83e24b61414c217fc5eadd62a8f5178028c94e5e73f54155552ff` |
| D18 | `data/eval/anka_t0129_132_dogrulama_2026-09-26.md` | `110220a5c9298299a8486df33c5d2bc2689072c38c0b34e1aa02dd1a2caafe77` |
| D19 | `data/anka_base_v2.pt` · `data/anka_base_v2.meta.json` | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` · `7b2d1e5242f8392929a4ae7ef28321481e47bf07424f75e0d74f666bd6ca943c` |
| D19a | `.agent-bus/notes/T-0133.md` | `d02fb1df28676cbc467058118342b51f8fc559613faa551a9a1c02f6ad557ea6` |
| D20 | `data/eval/anka_r29_kapasite_yetenek_2026-09-21.modul.json` | `3b678640d9660a090bb82a8dd571f573419a642e3786968271a7b324a654a671` |
| D20b | `data/eval/anka_r31_gomme_yetenek_2026-09-22.modul.json` | `ca2cdb10d852250522c15c519535eea8916a146d13867cccb9ae5b21fedc1804` |
| D20c | `data/eval/anka_r32_adim_yetenek_2026-09-22.modul.json` | `e0f0b2dd4a79269fd513a86ab8c882770b91e28d735dca6222d09a0d324a6202` |
| D20d | `data/eval/anka_r33_yetkin_temel_yetenek_2026-09-22.modul.json` | `e77c53d56702f1f19a8996c90ed188ad8b21c142e40dd9e610b26f05666f466c` |
| D20e | `data/eval/anka_r34_son_tur_yetenek_2026-09-22.modul.json` | `bae7af6c546217bd9dab7a3ce8b0278a00f2c2d8da27f394defcbcb6aed1d715` |
| D21 | T-0126 kanonik kayıtlar (içinde kanonik_kayit_sha256) | `bf22e742…` (r29), `61420c5c…` (r33), `e854b195…` (r34) — kayıt dosyalarının içinde |
| D22 | `scripts/evaluate_carpenter_anka.py` (ESIK/TAVAN sabitleri :70/:83/:84) | `a8667e21e8b762aa1ec47cbff1fee150df015e1b0474a0675a06af8d3254aad1` |
| D23 | `scripts/olcum_kabi.py` (kanarya satırları :326,330) | `fd911be933007f363228bd66ccf1a3d58ddd0cbbc10291529d54629fb3cbfdd7` |

---

## Beyanlı sınırlar ve açık sorular

1. Bu belge MEVCUT ölçümlerin sentezidir; **yeni ölçüm üretmez** (ilansız ölçüm YASAK). Her sayı
   kaynağından spot-doğrulandı (D5 sayıları D5'te; G4 sayıları D16/D17'de; r-serisi D20…).
2. Açık gerilim (kapanmış, yeniden açılmaz): A′ unutma ekseninde %5 ceket geçerken
   (T-0062 [D2]), kabul edilen ceket rol-ekseninde **−0,6255 CE, 150/150 kötüleşme** üretti [D3]
   ve bozulma **BELGEDEN BAĞIMSIZ** (kayıp %84,6 belge yokken de var) [D4] — F4 kararı yürürlükte
   ama "aleyhine olmadığı tek kol" cümlesi ölçülmüş eksen kümesiyle sınırlandırılmalıdır [D2].
   Bu bedel §2'ye yazılır: **ceketin maliyet tablosu büyüktür** — Bahçıvan tasarımı her yeni
   ceket için maliyet eksenlerini (A′/B′/C + rol-ekseni) ilan etmek ZORUNDADIR.
3. Dönüştürülmeyen hipotezler: bozuk-hedef artefaktı ayrımı (doğru-maskeyle eğitilmiş tam ince
   ayar checkpoint'inde modül koşumu) ölçülmemiştir — açık soru [D5].