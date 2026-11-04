# VG-3 — ΔH Yeniden Ölçüm (Çıpalı Envanter + Aux) + J/CI + Drift + Dondurma Hükmü (2026-11-04)

## İLAN

**Kapsam:** Kalem-4'ün bitişiğindeki şartın (P2.4b §8 uyarısı: *"ΔH, bellek ortamına duyarlıdır — eşik dondurulmadan önce bellek envanterinin çıpalanması gerekir"*) yerine getirilmesi: envanter artık **ÇIPALI** (VG-1/VG-2 sonrası, bu kalem bittikten sonra değişmeyecek); 107 sorguluk set, **aux zinciri dahil** (gts_bellek + kanun_madde) gerçek canlı hat üzerinde yeniden ölçülür; J tabloları + bootstrap CI yeniden kurulur; önceki iki ölçümle (n=36, n=107-eski-envanter) drift tablolaşır; dondurma önerisi VERİLİR ama dondurma DENETLEYİCİ kararıdır.

**Çıpalı envanter (ölçüm-öncesi salt-okunur GET kanıtı, betik stdout'unda):**

| koleksiyon | nokta |
|---|---|
| anka_bellek | 46 |
| muhakeme_bellek | 91 |
| simulasyon_bellek | 4.073 |
| gts_bellek | 128.327 |
| kanun_madde | 4.068 |
| yetenek_agi | 36 |

**Sert kurallar (hepsi tutuldu):** git commit YOK; eğitim YOK; **Qdrant SALT-OKUNUR** (yalnız `query_points`/koleksiyon `GET`); izlenen dosyalara DOKUNULMADI — bu kalem yalnız YENİ `data/eval/` kanıt dosyaları üretir, betikler scratch'tedir (`p24c_olcum.py`, `p24c_topla.py`, `p24c_ek_analiz.py`).

**Güvenlik kanıtı:** `data/future_train_vector.jsonl` sha256 önce/son aynı (`e3b0c44298fc1c14…`, DEĞİŞMEDİ); future-train yazımı koşum-öncesi scratch'e yönlendirildi (`p24c_future_train_yonlendirme.jsonl`, **2 kayıt** düştü — §6). Aynı 107 sorgu, aynı model (entropy_pre p24b ile **107/107 bit-düzeyinde özdeş**) — ortam etkisi izole edildi.

---

## HÜKÜM (denetçi CHANGELOG'da dondurur)

### 1. DONDURMA ÖNERİSİ: YOK — **KALDI**

Çıpalı envanterde (aux dahil) **hiçbir sinyal+eşik çifti "J CI'sı 0'ı dışlar" ölçütünü geçmiyor**:

| aday | nokta J | J CI95 (1000 yeniden-örnekleme) | hüküm |
|---|---|---|---|
| **delta_h @ −0,85** (kalem-4 dondurma-adayı) | +0,113 | **[−0,045; +0,292] → 0'ı İÇERİR** | **ÇÜRÜTÜLDÜ** |
| delta_h @ −0,80 / −0,75 (yeni nokta-plato bölgesi) | +0,113 / +0,090 | [−0,045; +0,292] / [−0,063; +0,258] | 0'ı içerir |
| delta_h J_p5-argmax | +0,071 @ +0,90 | J_p5 = **0,000** | dejenere — robust bölge YOK |
| post_h @ +2,25 / +2,50 / +2,75 / +2,81 / +3,00 | +0,155…+0,195 | hepsi [−0,015…−0,105; …] → 0'ı içerir | dondurulamaz |
| post_h @ +0,65 (tek 0-dışlayan) | +0,154 | [+0,043; +0,253] | tetik **%78,5–91,6** → RED (P2.4b ile aynı hüküm) |

- **delta_h sinyali çıpalı envanterde istatistiksel ayrım gücünü yitirdi.** Kalem-4'te 0'ı dışlayan TEK adaydı (eski envanter: J=+0,231, CI [+0,024; +0,445]); aux ekimi aynı 107 sorguda nokta J'yi +0,231→+0,113'e düşürdü ve CI'yı 0'ın içine çevirdi. Mekanizm ölçüldü (§6): aux, **belirsiz sorguların TAMAMINI** (28/28) koşulluyor — ΔH'nin taşıdığı "koşullanma başarısızlığı" bilgisi belirsiz-rest ayrımını kaybetti.
- **post_h** operasyonel-uslanabilir bölgede (tetik ~%45–65) hiçbir eşikte 0'ı dışlamıyor; 0'ı dışlayan tek eşik (+0,65) tetik %84+ olduğundan kalibrasyon hedefiyle (normalde sessizlik) çelişiyor — öneri DIŞI.
- **Çıpalı envanterde dondurulabilir eşik YOKTUR; denetçi eşik donduramaz.** Kalem-4'ün "dondur-yalnız-çıpalama-koşuluyla" önerisi, çıpalı zemin ölçümüyle **geçersizleşti** (bu, kalem-4 uyarısının doğrulanmasıdır, çürütülmesi değil — aşağıdaki kural).

### 2. OPERASYONEL KURAL ÖNERİSİ — ÇIPALAMA KURALI (HÜKMÜN ASIL ÜRÜNÜ; ölçümsel kanıtla)

> **Çıpalam kuralı:** Eşik, bellek **envanterine çıpalanmış** bir zeminde ölçülür (koleksiyon adları + nokta sayıları belgeye yazılır). **Envanter değişince — yeni koleksiyon ekimi, mevcut koleksiyona ekim/silme, aux zinciri değişikliği — eşik YÜRÜRLÜKTEN DÜŞER** ve yeniden ölçüm + yeniden çıpalam yapılmadan kullanılmaz. Eşik belgesi her zaman: sinyal + değer + tetik oranı + J/CI + **çıpalı envanter tablosu**.

**Kuralın ölçümsel kanıtı (bu kalem):** Yalnız envanter değişti (aux ekimi; model, set, sorular, pre-entropiler bit-düzeyinde aynı):
- delta_h@−0,85: J **+0,231 → +0,113** (ΔJ=−0,118); CI [+0,024;+0,445] → [−0,045;+0,292] (**0-dışlar → 0-içerir**).
- delta_h@−0,90/−0,95: ΔJ=−0,153; post_h@3,00/3,16: ΔJ=−0,094.
- Bu, kalem-4'te ölçülen simulasyon_bellek etkisinden (J 0,27→0,17) **büyük ve aynı yönlü** ikinci bir envanter-etkisidir: iki bağımsız ekim olayı, iki kez aynı sonuca (J düşer, eşik kayar) yol açtı. Kural artık varsayım değil, iki-olgulu ölçümdür.

### 3. Denetçiye ek bulgular (karar bu kalemin dışında)

1. **Sinyal tasarımı aux çağında eskidi:** GTS sözlüğü (128.327) her şeyi "tanımladığı" için koşullanma-başarısızlığı artık gerçek belirsizliği işaretlemiyor. Yeni merak-sinyali (ör. kaynak-koşullu eşik: aux kaynaklı belgelerde ayrı eşik; skor-kapılı ΔH; ya da entropi-dışı sinyal) denetçi tasarımıdır — bu kalem ölçtü, tasarlamadı.
2. **Aux, epistemik-başarısızlık (A-koşul) kapısını çöp girdiye açtı:** #91 (`"?"`, kenar) gts_bellek'te skor 1,0 ≥ 0,85 yüksek-benzerlik + H_post 4,21 ≥ 2,5 → future-train kaydı üretildi. Aux kaynaklı belgelerde `is_high_similarity` kapısının gözden geçirilmesi önerilir.
3. **"Retrieval skoru 0,70–0,83 bandına çıktı" iddiası bu sette doğrulanmadı:** 107 sorguda yalnız **1** sorgunun en-iyi skoru [0,70; 0,83] bandında; aux kaynaklı 32 skorun medyanı ~0,53, dağılım 0,30–1,00 (§6). (VG-1'in üretim testindeki bant gözlemi bu sorgu setine taşınmamış.)

---

## RAPOR

### §1 Ölçüm kanıtı (GÖREV 1)

- **Düzeneği P2.4/P2.4b ile birebir + aux:** `AgentGateway.create_default(anka_taban_pure + anka_router_v2 + bahcıvan/marangoz ceketleri + anka_bellek/simulasyon_bellek @ http://192.168.1.9:6333)` ve **`QDRANT_EXTRA_COLLECTIONS=gts_bellek,kanun_madde`** (VG-1 operasyonel notu). Gateway kurulum stdout'unda `[AUX] canlı retrieval zincirine bağlandı: gts_bellek` / `…kanun_madde` satırları + betik içi `aux_kurulu == ['gts_bellek','kanun_madde']` assert'i.
- **Koşum:** kurulum 3,3 sn; **107 sorgu 108,4 sn; ask_hata=0** (107/107). `gw.ask(mode="RAG", force_rag=False, max_new_tokens=45)` == canlı `/api/ask`.
- **Determinizm:** p24b↔p24c `entropy_pre` **107/107 bit-düzeyinde özdeş**; `ham_entropy_pre` 107/107 özdeş. Post-entropide **yalnız 2/107** sorgu değişti (#2, #22) — her ikisinde belge, skor, kaynak ve koşullanma durumu AYNI; **MPS float determinizmi dışı** (yakın-eşik argmax sapması), retrieval-drift değil; J analizinde gürültü düzeyinde.
- **τ=2,5 (mevcut donmuş tetik) yine 107/107 (%100) tetikledi** — mutlak pre-entropi eşiği yine ayrım üretmiyor.
- Ham kayıt: scratch `p24c_olcum_kayitlar.ndjson` (107 NDJSON) → toplanmış: `data/eval/p24c_esik_dagilim_2026-11-04.json`.

### §2 Kategori × ölçüm — eski-envanter (p24b) → çıpalı-envanter (p24c) eşli değişim

| kategori | n | cond p24b→p24c | H_post medyan p24b→p24c | ΔH medyan p24b→p24c | ΔH [min–maks] p24c |
|---|---|---|---|---|---|
| alan | 28 | 19 → **25** | 1,58 → 1,72 | −3,28 → −3,30 | [−7,44 … +0,88] |
| nötr | 28 | 9 → **19** | 3,70 → 3,56 | −1,58 → **−1,92** | [−5,04 … −0,25] |
| **belirsiz** | 28 | 17 → **28 (TAMAMI)** | **4,21 → 3,22** | **−1,77 → −2,64** | [−6,59 … +1,35] |
| kenar | 23 | 14 → **19** | 4,02 → 3,64 | −4,09 → −4,09 | [−9,62 … −0,21] |

Koşullanma 59→91/107. **Ayrım mekanizması tersine döndü:** p24b'de belirsizlerin H_post'u (4,21) rest'ten belirgin yüksekti; aux ekimiyle belirsiz H_post 3,22'ye indi ve rest ile örtüşür hale geldi — H_post ve ΔH kategorileri **çıpalı envanterde ayırt etmiyor** (nokta J'ler §3).

### §3 Eşik taramaları (p24c, çıpalı envanter) — tetik: belirsiz=pozitif, J = belTPR − restFPR

**delta_h** (ΔH = H_post − H_pre ≥ t) — seçmeli satırlar:

| t | belTPR | restFPR | J | tetik (n=107) |
|---|---|---|---|---|
| −2,50 | 0,464 | 0,392 | +0,072 | %41,1 |
| −1,50 | 0,357 | 0,266 | +0,091 | %29,0 |
| −1,00 | 0,214 | 0,177 | +0,037 | %18,7 |
| **−0,75 (yeni nokta-plato)** | 0,179 | 0,089 | **+0,090** | %11,2 |
| −0,50 | 0,107 | 0,089 | +0,019 | %9,3 |
| 0,00 | 0,107 | 0,025 | +0,082 | %4,7 |

Nokta-J maksimumu yalnız **+0,126** (plato [−0,78; −0,72), öneri −0,75) — p24b'deki +0,254'ten yarıya indi; J_p5-argmax dejenere (+0,90; J_p5=0,000). Bootstrap argmax medyanı −1,42 (IQR [−2,68; −0,80]) — p24b'de −0,88 idi.

**post_h** (H_post ≥ t) — seçmeli satırlar:

| t | belTPR | restFPR | J | tetik |
|---|---|---|---|---|
| +2,00 | 0,750 | 0,620 | +0,130 | %65,4 |
| +2,50 (eski hatırlama-öncelikli) | 0,714 | 0,570 | +0,145 | %60,7 |
| **+2,75 (yeni nokta-J maks bölgesi)** | 0,714 | 0,519 | **+0,195** | %57,0 |
| +3,00 | 0,607 | 0,494 | +0,113 | %52,3 |

Nokta-J maksimumu +0,208 @ ~2,81; J_p5-argmax yine +0,65'ı seçer (tetik %84+) → öneri DIŞI (P2.4b §5b hükmü aynen geçerli).

### §4 Bootstrap CI (sorgu-seviyesi 1000 yeniden-örnekleme, kategori-tabakalı, tohum 20261104 — p24b ile aynı)

| sinyal @ eşik | nokta J | J CI95 | belTPR CI95 | tetik CI95 | 0'ı dışlar? |
|---|---|---|---|---|---|
| delta_h @ −0,85 | +0,113 | [−0,045; +0,292] | [0,071; 0,393] | [%7,5; %19,6] | **HAYIR** |
| delta_h @ −0,75 | +0,090 | [−0,063; +0,258] | [0,036; 0,321] | [%5,6; %17,8] | HAYIR |
| post_h @ +2,50 | +0,145 | [−0,067; +0,336] | [0,536; 0,857] | [%51,4; %70,1] | HAYIR |
| post_h @ +2,75 | +0,195 | [−0,015; +0,391] | [0,536; 0,857] | [%47,7; %65,4] | HAYIR (kıl payı) |
| post_h @ +0,65 | +0,154 | [+0,043; +0,253] | [0,893; 1,000] | [%78,5; %91,6] | EVET → ama tetik %84+ ⇒ RED |

Doğrulama: aynı tohumla yeniden hesaplanan p24b CI'ları P2.4b raporuyla birebir (delta_h@−0,85: [+0,024;+0,445]; post_h@2,50: [+0,002;+0,381]) — drift karşılaştırması yöntem-tutarlıdır.

### §5 Drift — üç ölçüm, sabit eşikler (GÖREV 2)

| eşik | n=36 eski ortam | n=107 eski envanter | **n=107 ÇIPALI envanter** | aux etkisi (ΔJ) | tetik (p24c) |
|---|---|---|---|---|---|
| delta_h @ −1,50 | J=+0,269 | J=+0,173 | **J=+0,091** | −0,082 | %29,0 |
| delta_h @ −0,90 | J=+0,285 | J=+0,241 | **J=+0,088** | −0,153 | %15,0 |
| delta_h @ −0,87 | J=+0,285 | J=+0,218 | **J=+0,100** | −0,118 | %14,0 |
| **delta_h @ −0,85** | J=+0,323 | J=+0,231 (CI 0-DIŞLAR) | **J=+0,113 (CI 0-IÇERİR)** | **−0,118** | %13,1 |
| post_h @ +2,50 | J=+0,238 | J=+0,203 | **J=+0,145** | −0,059 | %60,7 |
| post_h @ +3,16 | J=+0,138 | J=+0,198 | **J=+0,103** | −0,094 | %49,5 |

- Drift'in kendisi eşli-bootstrap'ta sınırda (ΔJ CI @−0,85: [−0,250; +0,015]) ama dondurma ölçütü mutlak CI'dır ve o NET başarısız.
- **Driftin imzası:** envanter değişiklikleri J'yi hep AŞAĞI oynatıyor (36→107 ortam: −0,10; aux ekimi: −0,12) ve argmax'ı sola/erişilemez bölgeye kaydırıyor (−1,50 → −0,88 → dejenere).

### §6 Aux (gts/kanun) ekiminin ΔH dağılımına etkisi (GÖREV 2 — mekanizma)

- **Yükseltme kuralı:** aux, YALNIZ ana zincirin belgesi 0,33 koşullanma kapısını geçemediğinde sorgulanır; kök-eşleşmeli + 0,33+ en yüksek aux adayı yükselir (epistemic_agent.py:260-282).
- **Ölçüm:** 32/107 sorguda yanıt belgesi aux'tan geldi (**gts_bellek 24, kanun_madde 8**); kategori dağılımı: **belirsiz 11, nötr 10, alan 6, kenar 5**. Tüm 32 flip koşullanma False→True (geri-dönüş 0).
- **"Koşullanma post-entropiyi düşürmeli" yarım-true çıktı:** aux-flip 32 sorgunun **20**'sinde post-H düştü, **12**'sinde ARTTI (ör. #39 alan 1,48→4,66; #93 kenar 0,004→4,06; #11 nötr 0,11→4,61). GTS/kanun belgeleri topikal olarak ilişkili ama modelin tanımadığı bağlama itiyor — aux koşullanması entropiyi tek-yönlü düşürmüyor, **yeniden dağıtıyor**.
- **"ΔH işareti değişmeli mi?" → HAYIR, işaret sistematik değil:** 5/107 sorguda işaret değişti, iki yönlü (#2, #55, #77: +→−; #39, #83: −→+). Baskın etki işaret değil **genlik sıkışması**: belirsiz ΔH medyanı −1,77→−2,64 (rest'e yaklaştı), belirsiz H_post 4,21→3,22.
- **Belirsiz tetik listesi @−0,85 (10 → 6):** #26, #27, #76, #77 artık tetiklenmiyor (aux belgesi ΔH'lerini −0,85 altına düşürdü: ör. #26 −0,76→−2,43); kalan 6: **#28, #80, #81, #83, #87, #88** — bunlar aux'lu olsa bile entropisi çökmeyen "sert-çekirdek" belirsizler (#81 simulasyon, #87/#88 anka kaynaklı; #28/#80/#83 gts/kanun kaynaklı ama skorları 0,33–0,59 bandında). Rest tetik 10→8.
- **Aux skor dağılımı (32 sonda):** medyan ~0,53; bantlar [0,30–0,35):3, [0,35–0,40):7, [0,40–0,45):3, [0,45–0,50):1, [0,50–0,55):8, [0,55–0,60):4, [0,60–0,65):1, [0,65–0,70):1, **[0,80–0,85):3, [1,00):1**. Tüm sette [0,70; 0,83] bandındaki en-iyi skor: **1/107**.
- **Yan-etki (future-train A-yolu):** 2 scratch kaydı — #49 (alan, anka, 1,0; P2.4b'den biliniyor) + **YENİ: #91 (`"?"`, kenar, gts_bellek, skor 1,0, H_post 4,21)** — aux, çöp girdiye yüksek-benzerlik üretti (HÜKÜM §3.2).

### §7 Kanıt dosyaları

| dosya | içerik |
|---|---|
| `data/eval/p24c_esik_dagilim_2026-11-04.json` | tüm p24c ölçümü: 107 ham kayıt (entropy_pre/post, ΔH, skor+kaynak, conditioned, süre) + çıpalı envanter + kategori özetleri (p24b eşli) + delta_h/post_h taramaları + platolar + bootstrap CI (p24c VE aynı tohumla p24b yeniden-hesap) + J-kantilleri (0,05 ızgara) + üçlü drift + aux etkisi (32 flip detayı, skor dağılımı, işaret değişimleri) + determinizm + future-train kayıtları |
| `data/eval/p24c_delta_h_dondurma_2026-11-04.md` | bu rapor (İLAN→HÜKÜM→RAPOR) |
| scratch: `p24c_olcum.py`, `p24c_topla.py`, `p24c_ek_analiz.py`, `p24c_olcum_kayitlar.ndjson`, `p24c_olcum.log`, `p24c_future_train_yonlendirme.jsonl` | ölçüm betiği, toplayıcı, ek-analiz, ham NDJSON, koşum log'u, yönlendirilen future-train kayıtları (repo-dışı) |

**Değiştirilen izlenen dosya: YOK.** Commit yok, eğitim yok, Qdrant'a yazma yok. **Değişen izlenen dosya: YOK** (`git status` yalnız yeni `data/eval/p24c_*` dosyaları gösterir). Dondurma kararı (bu ölçümle: **dondurma YOK — KALDI**) ve çıpalam kuralının yürürlüğe alınması denetçinindir.
