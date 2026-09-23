# T-0097 · FAZ 0 İLANI — Anka ceket külliyatını büyütme + tam giydirme koşumu

> **Bu dosya derleme ve koşumdan ÖNCE yazılmıştır ve sonradan DEĞİŞTİRİLMEZ.**
> Ölçüm kuralları, kapılar ve çürütme maddeleri burada ilan edilir; sonuçlar görüldükten sonra
> eşik/kural ayarlaması yapılmaz (T-0067: *tasarım sayısı ölçüm sonrası kusura göre
> düzeltilmez, rapora yazılır*).

| kalem | değer |
|---|---|
| görev | **T-0097** (yürütücü: claude) |
| operatör talebi | *"külliyatı büyütmek planla"* |
| operatör kararları | (1) ceket/marangoz · (2) marangoz derinliği · (3) **doğrudan tam koşum, kısa sonda YOK** · (4) **katalog tablolarını genişlet** · (5) **eski held-out SABİT** — AskUserQuestion ile |
| tarih | 21 Eyl 2026 |
| taban | `data/anka_a1r.pt` (`b93cc1cd5409…`, 373.735.137 B) |
| kontrol kolu | **T-0096'nın kayıtlı altı noktası** (`scratch/t0096_kos/sonuc.json`) |
| durum | **Faz 0 — derleme başlamadı** |

---

## §1 — ÖLÇÜLEN TEŞHİS (derlemeden ÖNCE, betikle; elle yazılmadı)

### 1.1 — Üretim zinciri çözüldü: **125 olgu → 450 perspektif → ×12 = 5.400 satır**

`scripts/generate_specialization_dataset.py` bir **katalog açılımıdır**. Tablolar
`(ad, açıklama)` **2'li tuple** listeleridir ve betik `if __name__ == '__main__':` (`:466`)
korumalıdır ⇒ **import etmek yan etki üretmez** (doğrudan çalıştırılarak doğrulandı).

| bölüm | tablo | **olgu (satır)** | perspektif | aday |
|---|---|---|---|---|
| §1 | `WOOD_SPECIES` | 25 | ×5 (A–E) | 125 |
| §2 | `TOOLS` | 25 | ×4 (A–D) | 100 |
| §3 | `JOINERY` | 20 | ×3 | 60 |
| §4 | `PHYSICS_AND_DRYING` | 15 | ×3 | 45 |
| §5 | `FINISHES` | 15 | ×3 | 45 |
| §6 | `WORKSHOP_SCENARIOS` | 25 | ×3 | 75 |
| | **TOPLAM** | **125** | | **450** |

§8 (`:397-406`) her adayı **11 istem öneki** ile çoğaltır ⇒ `450 + 450×11 = 5.400`.
Sayaçla doğrulandı: `Toplam Üretilen Ham Aday Sayısı: 5,400`, `Toplam Altın SFT Örnek Sayısı: 5,400`.

### 1.2 — Külliyatın benzersiz bilgi içeriği **16.405 karakter**

| ölçüm | değer |
|---|---|
| benzersiz olgu (ad) | **125** |
| benzersiz açıklama | **125** |
| açıklama karakteri (toplam) | **16.405** |
| açıklama uzunluğu: ort / medyan / min / max | **131,2 / 127 / 90 / 205** |

Külliyat **784.797 jeton ≈ 2.071.864 karakter** (0,37814 jeton/karakter kalibrasyonu) ⇒
benzersiz bilgi külliyatın **%0,79'u**; **%99,2'si iskele** (şablon + perspektif + önek çoğaltması).

### 1.3 — Kural filtresi **NO-OP**

§7 `evaluate_5d_rubric` (`:194`) 5.400/5.400 adayı **geçirir** — hiçbirini elemez.
⇒ Külliyatta *"kalite süzgecinden geçmiş altın SFT"* iddiası **VAKUM**'dur.

### 1.4 — Mevcut açıklamalar **zaten** çok cümlelik teknik metindir

Ort. **131,2 karakter**, çok maddeli, zanaat terimi yoğun (örn. *"Doğal silika ve yağ
barındırdığından suya, güneşe ve çürümeye aşırı dayanıklıdır; dış mekan bahçe mobilyası ve tekne
güvertesinde rakipsizdir."*). ⇒ Eksik olan **açıklama zenginliği DEĞİL, olgu SAYISIDIR**.

### 1.5 — Diskteki "yeni marangoz içeriği" taraması

| aday | ölçüm | hüküm |
|---|---|---|
| `carpenter_specialization_dataset.jsonl` | 4.861 satır kullanılıyor (539'u held-out) | yeni içerik YOK |
| `pedagogy_canonical/carpenter_canonical.jsonl` | aynı 5.400 öğenin persona-normalize ikizi | **yeniden-ifade** |
| `arena_carpenter_accumulated.jsonl` | 302 kayıt, **30 cevabı held-out'ta birebir** | **KİRLİ — dışlandı** |
| `future_train_archive.jsonl` | 188 benzersiz `rag_document` = **32.635 karakter** | temiz, **birincil kaynak** |
| `raw_teacher_cards_archive.jsonl` | 48 olgu (`topic`/`base_concept`) | temiz, **birincil kaynak** |
| `simulasyon_bellek_export.jsonl` | 4.000 kayıt `domain=gts_sozluk` | **sözlük**, marangoz değil |
| `pedagogy/rag_dataset.jsonl` | `belge: X sorgu: X` → `X` | kopyalama görevi |
| `realistic_rag/**` | jeoloji/arkeoloji + `is_counterfactual` | alan dışı + **karşı-olgu** |

⇒ Derinlik **yalnız katalog genişletmeyle** elde edilir (operatör kararı 4).

### 1.6 — Neden eşleşen adım kıyası **temiz** bir deneydir

Külliyat `g` katına çıkar ve adım sayısı sabit tutulursa epoch `1/g` olur ⇒ toplam
**olgu-maruziyeti** `(125·g)·(E/g) = 125·E` — **DEĞİŞMEZ**. Jeton bütçesi de sabittir
(24.000 × 1.024 = **24,6 M**). ⇒ Eşleşen adımlarda T-0096'ya karşı kıyas, **jeton bütçesi ve
olgu-maruziyeti sabitken yalnız ÇEŞİTLİLİĞİ** değiştirir.
**Koşulu V4b'dir:** yeni olguların olgu-başına jetonu mevcutlarınkiyle **±%20** içinde olmalı.
Aksi hâlde `g` jeton ekseninde de kayar ve kıyas bulanır.

---

## §1.7 — ERRATUM (bu ilandan SONRA, derlemeden ÖNCE ölçüldü; §1.5 DEĞİŞTİRİLMEDİ)

§1.5'teki iki kaynak **yanlış nitelendirilmişti**. §1.5'in metni **olduğu gibi bırakılmıştır**
(sessizce düzeltmek kaydı bozardı); düzeltme burada, **fark yazılarak** yapılır.

| §1.5'te yazılan | ölçülen gerçek |
|---|---|
| `raw_teacher_cards_archive.jsonl` → *"48 olgu (`topic`/`base_concept`) — temiz, **birincil kaynak**"* | **YANLIŞ.** 48 kartın yalnız **~5'i marangoz** (indeks 0,1,2,3,4). Kalan **43'ü** edebiyat/fizik/biyoloji/matematik (kafiye, Newton yasaları, mitoz-mayoz, periyodik tablo…). 47. kart bir **decompiler artefaktı** taşır (`tezat sanat POSS_3SG ne COPULA_AORIST … <UNK>`). Alan adları `payload.topic` / `payload.base_concept`'tir (üst düzeyde değil). |
| `future_train_archive.jsonl` → *"188 benzersiz `rag_document` … temiz, **birincil kaynak**"* | **FAZLA İYİMSER.** Marangoz içeren belge **69**; ama bunların çoğu **katalogdan TÜRETİLMİŞTİR** (örn. *"Usta Marangoz: Teknik Çözüm: Kumpas ahşap kalınlığını…"* = 125 olgunun önek-açılmış hâli). Mevcut 125 ile **4-gram örtüşmesi SIFIR** olan paragraf sayısı: **6** — ve bunların 3'ü aynı zıvana içeriğinin parçası ⇒ **≈3 olgu**. |
| `data/knowledge/ahsap_ve_marangozluk_rehberi.md` | 7 bölüm; §1–§3 `future_train_archive` ile **aynı metin**, §4–§7 mevcut ağaç türlerinin genişletmesi ⇒ **türev**. |

**Ölçülen sonuç:** diskteki **gerçekten yeni** marangoz olgusu sayısı **≈ 0–8**'dir; arşiv
katalogdan türetildiği için **döngüseldir**.

**Bunun ilana etkisi (kapı DEĞİŞMEZ):**

* V4 (**≥125 yeni olgu**) **aynen durur**. Düşürülmez, gerekçeyle değiştirilmez.
* §7.3'ün dürüstlük notu **genişler**: yeni olguların *"bir kısmının"* değil, **~%95'inin** tek
  kaynağı yürütücünün yazdığı metindir. **Doğruluk yükü tamamen yürütücüdedir.**
* Kapıya ulaşılamazsa hüküm **"hedefe ulaşılamadı"** olur ve **uydurma olguyla doldurulmaz**.
* Bu erratum **derlemeden ÖNCE** yazılmıştır; hiçbir ölçüm sonucu görülmeden yazılmıştır, yani
  ön-kayıt sırası **bozulmamıştır** (T-0096 kapanış kapısı: ilan mtime < ilk derleme mtime).

---

## §1.8 — ERRATUM II (derlemeden ve koşumdan ÖNCE; §1–§1.7 DEĞİŞTİRİLMEDİ)

**Kaynak:** bağımsız bir arka-plan çözümleme ajanı (Antigravity **değil**; operatör kısıtı gereği
Antigravity'ye yalnız danışılır, görev açılmaz). Bu ajan diskte **kendi ölçümlerini** yaptı ve
aşağıdaki beş bulguyu üretti. Hepsi **derlemeden önce** ölçülmüştür ⇒ **hüküm değil, ön-koşuldur**.

| # | ölçüm | sonuç | ilana etkisi |
|---|---|---|---|
| **E1** | **eğitim-içi ROUGE oracle'ı**: her held-out cevabı için 4.861 satırlık eğitim katmanındaki **en iyi** ROUGE-L | ort **0,9447** · med **0,9545** · **min 0,8571** · **100/100 ≥ 0,35** | **AĞIR.** Held-out'un *her* cevabının eğitimde ≥0,857'lik **ikizi** var; fark yalnız **sarmalayıcı önektir** (`Teknik Çözüm:` vs `Teknik Rapor:`). ⇒ **Bu held-out'ta ROUGE'u külliyatla yükseltmek neredeyse imkânsızdır** — eksik olan içerik değil. |
| **E2** | **`ezber_orani` tavanı**: *eğitim metninin kendisi* `encode→decode` ile geri döndürüldü (300 kayıt), 4-gram kapsama | ort **0,020** · med 0,016 · **0/300 ≥ 0,90** | Ölçütün **kendi tavanı ≈%2**, kapı ise **%90** ⇒ eksen **VAKUM**. S1'in *"ezber %20,00"* değeri ezber değil, **dejenere üretim** (tekrar döngüsü) artefaktıdır. |
| **E3** | **`kesisim_orani` yapısal tavanı**: held-out'ta **boş `input`** taşıyan satır | **343/539** (örneklemde **65/100**) | Boş `input`'ta `set(kelimeler(input)) ∩ set(gw)` **2'ye ulaşamaz** ⇒ örneklemde erişilebilir tavan **≈%35**, kapı **%80**. Eksen **yapısal olarak** kapalı. |
| **E4** | `--replay-every` bayrağı `train.py`'de | **YOK** (`grep -c` = **0**) | §2'nin **komut bloğu zaten doğruydu**; ama replay maddesinin *ifadesi* düzeltilir: oran **`scripts/build_replay_mix.py`**'ye ait bir bayraktır ve **karışım `.bin`'ine gömülür**. Yeni kol, eğitim bayrağını değil **karışımı** yeniden üretmeyi gerektirir. |
| **E5** | 4-gram sızıntı kapıları | Sertifikalı-temiz eğitim katmanının **%36**'sı ≥0,85 kapsama; **%62**'si 8-gram paylaşır | **4-gram KAPI OLARAK KULLANILAMAZ.** Ayırt edici tek ölçüt **önek-normalize tam-dizgi (cevap-grubu) eşitliği**'dir. V6'nın 4-gram katmanı **rapor-only**'ye indirilir; sert olan iki katman (cevap ekseni · kayıt düzeyi) **aynen durur**. |

**Bu ölçümlerin ilana etkisi — dürüst hüküm:**

* **E1, ana ekseni vurur.** İlan zaten **Ç1** ve **Ç3** ile bu tuzağı yakalamak üzere yazılmıştı;
  ama E1 onlardan **daha ileri** gider: ΔROUGE ≈ 0 çıkarsa bu *"külliyat kısıt değil"* değil,
  **"ölçüt doygun"** demektir. Ç3'ün hükmü bu yüzden **genişletilir**: ROUGE düşmezse kapsam
  kayması; **artmazsa** ölçüt doygunluğu. İkisi ayrı yazılır.
* **E2/E3, iki kalite filtresini ölçülemez kılar.** §6.2'nin dışlaması `ezber`/`tutarsizlik`
  üzerinden işler; `ezber` ekseni tavanı %2, kapı %90 ⇒ **bu filtre hiçbir zaman ateşlemez ve
  ateşlerse artefakttır**. §6'nın hükmü artık **A / B / ROUGE** üçlüsüne dayanır; bu **§6'ya
  eklenen bir kısıttır**, eşik değişikliği **değildir** (eşikler aynen kalır, **VAKUM** diye
  raporlanır — T-0096/K3 sürüyor).
* **E4, bir ifade düzeltmesidir**; komut bloğu değişmez. Yeniden üretilecek olan **karışımdır**.

**Bağımsız ön-kayıtlı tahmin (bu ilana kaydedilir ve sonuçla karşılaştırılır):**

> Bağımsız ajan, **hiçbir koşum yapılmadan önce**, şunu öngördü: **eşleşen altı noktanın
> HEPSİNDE Δ(s) = ROUGE_yeni(s) − ROUGE_T-0096(s) ≤ 0**; mekanizma: (i) E1 ⇒ içerik açığı yok,
> (ii) T-0096'da ROUGE S6'da hâlâ **monoton artıyordu** ⇒ bağlayıcı değişken **adımdı**,
> (iii) sabit adımda `g=2` külliyat epoch'u yarıya indirir ⇒ T-0096'nın kendi eğrisi yeni kolu
> **S3–S4 düzeyine (≈0,13–0,21)** yerleştirir, (iv) diskte gerçekten yeni olgu **yok**.

Bu tahmin **kaydedilir** çünkü **yanlışlanabilir**: Δ > 0 altı noktanın **≥4'ünde** çıkarsa
tahmin **ÇÜRÜR** ve bu, külliyat lehine **beklenmedik ve bilgilendirici** bir sonuçtur.
Tahminin kaydedilmesi, sonucu **önceden çivilenmiş bir hüküm** yapmaz; **çürütülebilir bir
öngörü** yapar (T-0068/B3 disiplini).

**Ön-kayıt çıpası — mtime DEĞİL, İÇERİK:** iki erratum ilanın mtime'ını **ileri taşıdı**. Bu
yüzden "ilan mtime < ilk derleme mtime" denetimi **tek başına zayıftır**; asıl çıpa, **kural
içeriğinin** derlemeden önce sabitlenmiş olmasıdır. §4 kapıları · §5 çürütme maddeleri · §6
seçim kuralı metinlerinin **sha256'sı** derlemeden önce hesaplanıp `scratch/t0097_kos/once.json`'a
yazılır ve kapanışta **yeniden hesaplanarak** karşılaştırılır. Kural metni değişirse kapı düşer;
mtime kayması **beyan edilmiş** olduğu için tek başına ihlal sayılmaz.

**DEĞİŞMEYENLER (açıkça):** V4 (**≥125 yeni olgu**) · altı segment sınırı (1.000/3.000/6.000/
10.000/16.000/24.000) · held-out (539 satır, **SABİT**) · tüm eşikler · §6 seçim sırası ·
kontrol kolu (T-0096'nın kayıtlı altı noktası) · §1.6'nın olgu-maruziyeti invaryantı ve onu
koruyan **V4b**. Bu erratum **hiçbir ölçüm sonucu görülmeden** yazılmıştır.

---

## §2 — SABİT yapılandırma (T-0096 ile **BİREBİR** aynı ⇒ kontrol kolu geçerli)

```
env PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
venv/bin/python train.py \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --data  scratch/t0097_kos/mix_r4.bin \
  --device mps --block-size 128 --batch-size 8 --lr 2e-4 --steps <SEGMENT> \
  --load-path <onceki> --save-path scratch/t0097_kos/seg_k.pt \
  --save-optimizer --load-optimizer \
  --loss-report --seed <42+k> \
  --allow-frozen-write        # YALNIZ son teslim adımında, data/anka_a1r_ceket_r18.pt için
```

* **SFT** — `--pretrain` **YOK** (T-0073'ün *"her pencere maskelenir"* tuzağı bu külliyatta yok).
* **Replay:** `data/train_chat_balanced.bin` (`443f93d352b0…`), oran **%25**. `--replay-every`
  **`train.py`'de YOKTUR** (`grep -c` = **0**, §1.8/E4) — bayrak **`scripts/build_replay_mix.py`**'ye
  aittir ve oran **karışım `.bin`'ine GÖMÜLÜR**. ⇒ Bu kol, eğitim bayrağı değiştirilerek değil,
  **karışım yeniden üretilerek** kurulur.
* **LR programı YOK** (sabit 2e-4) — *beyan edilen sınır*: karşılaştırılabilirlik için T-0096 ile
  aynı. Bu ilan onu **kapatmaz**, **açık bırakır**.
* **`--save-every` KULLANILMAZ:** aynı `--save-path` üzerine atomik **ÜZERİNE** yazar ⇒ ara
  checkpoint serisi üretmez. Onun yerine **segment** mimarisi (her segment ayrı süreç).
* **Taban AdamW momenti taşımaz** ⇒ S1 momentsız başlar; ölçülen ilk güncelleme **1,7306×**
  (T-0092). Beyan edilen sınır.

---

## §3 — Segment planı (bütçenin satın aldığı: **SÜRE** üzerinde arama)

Ölçülen adım süresi (T-0096, `sonuc.json`): **0,39–0,59 sn/adım** ⇒ **0,50 sn/adım** alınır
(tahmin değil, T-0096 ölçümü). Kesme kuralı: adım süresi **0,70 sn**'yi aşarsa koşum S5'te
kesilir ve kesinti rapora yazılır — sessizce uzatılmaz.

**Segment sınırları T-0096 ile BİREBİR aynıdır** (kontrol kolunun tek geçerlilik koşulu):

| segment | adım | kümülatif | T-0096 ceket epoch | T-0097 tahmini epoch |
|---|---|---|---|---|
| S1 | 1.000 | 1.000 | 0,98 | **≈0,5** (`1/g`) |
| S2 | 2.000 | 3.000 | 2,94 | ≈1,5 |
| S3 | 3.000 | 6.000 | 5,87 | ≈2,9 |
| S4 | 4.000 | 10.000 | 9,79 | ≈4,9 |
| S5 | 6.000 | 16.000 | 15,66 | ≈7,8 |
| S6 | 8.000 | 24.000 | 23,49 | ≈11,7 |

**Toplam 24.000 adım** ≈ 3,33–3,67 saat saf eğitim + 6 eşli değerlendirme ≈ **3,5–3,9 saat**.
Epoch sayıları **farklıdır ve bu ilan edilmiş bir özdür** (§1.6: olgu-maruziyeti korunur, epoch
sayısı korunmaz).

**`--steps` ELLE YAZILMAZ:** yazılan `.bin`'in kendi meta'sındaki `jeton` alanından **betikle**
hesaplanır (`ceil(jeton/(128·8)) × epoch`). Ölçüm sonrası kusura göre düzeltilmez; sapma rapora
yazılır (T-0067).

---

## §4 — İLAN EDİLEN KAPILAR

**Derleme (V-serisi):**

| # | ölçüm | düşüren girdi |
|---|---|---|
| V1 | sözlük kimliği `f9940a8d8e1f…`, giriş **33.114** | yanlış sözlük |
| V2 | **önek invaryantı**: id 0…32.851 fark **0** | sözlük kayması |
| V3 | `literal_entity_mode=False` ⇒ `<ENT>`=`<CAP>`=`<ALL_CAPS>` = **0** | eski temsil |
| V4 | **yeni olgu ≥ 125** (benzersiz bilgi ≥2×) ve her olgu **12 satır** üretir | eksik/boş tablo |
| **V4b** | **olgu başına jeton: yeni / mevcut = 1,00 ± 0,20** | kıyas bulanır ⇒ §1.6 gerekçesi düşer |
| V5 | yeni adlar mevcut 125 ad ile normalize çakışma **0** | kopya olgu |
| V5b | yeni açıklamaların **4-gram** örtüşmesi mevcut 125 ile **≈ 0** | yeniden-ifade ⇒ genişleme sahte |
| V6 | **sızıntı (KAPILI)** — üç katman: held-out 539 satırın **cevap ekseni** (önek-normalize) **0** · **kayıt düzeyi** **0** · **4-gram** **0** | held-out'a sızan olgu |
| V7 | muhasebe: `kept = satır×2 + 1.500 + 688`; `json`/`encode`/`short` hatası **0** | kaynak yolu değişmiş |
| V8 | ölçek: jeton **> 1.150.000** (mevcut 784.797'nin ≥1,47 katı) | genişletme yapılmamış |

**Koşum (K-serisi, T-0096'dan devralınan sert sözleşmeler):** baş/sözlük boyutu tam eşleşme ·
lexicon boş değil · tokenizer sözlüğü büyütmüyor · altı özel jeton var · `<OUTPUT>`/`</OUTPUT>`
ayrı id · zarf sözleşmesi · `--device mps` mevcut · A ekseni %95 yarı-genişlik marj altı ·
oracle sağlaması (kimlik ROUGE = 1,0, çeldirici ROUGE < 0,35) · ≥2.500 noktalama konumu.
Hepsi **`rc=2`** ile **fail-closed**.

---

## §5 — İLAN EDİLEN ÇÜRÜTME MADDELERİ

Her madde **kendi iddiamı yanlışlayabilmelidir** (T-0068/B3 disiplini): çürütücü gözlem ve eşiği
**önce** ilan edilir.

| # | öncül | çürütücü gözlem | hüküm |
|---|---|---|---|
| **Ç1** | "kısıt külliyattır; olgu sayısı oranı düzeltir" | **eşleşen adımlarda** ROUGE artmaz **VE** A düşmez | külliyat **kısıt değilmiş**; kısıt başka yerde (lr / replay / momentler / katman dondurma) — ***"veri yetmedi" DENMEZ*** |
| **Ç2** | "genişletme unutmayı azaltır" | A, eşleşen adımlarda T-0096'nınkinden **yüksek** | genişletme unutmayı **artırdı** |
| **Ç3** | "genişletme ceketi yakınsatır" | ROUGE eşleşen adımlarda **düşer** | **kapsam kayması**: ölçüt hâlâ **eski 450 perspektifi** sorar ⇒ genişlemeyi **görmüyor** |
| **Ç4** | "yeni olgular ezberlenir" | `ezber ≥ %10` **herhangi** bir segmentte | katalog açılımı ezber üretiyor |
| **Ç5** | "kesişim ekseni işler" | `kesisim` %80'e çıkmaz | T-0096/K3 sürüyor: eşik **VAKUM** (insan tavanı **%13,00**) |
| **Ç6** | "iskele %99,2 olduğu için çeşitlilik kaldıraçtır" | ROUGE kazancı **doyar** (g=2 kazancı, g=1'in iki katı etmez) | iskele **değil**, **mutlak olgu sayısı** kısıt ⇒ daha büyük `g` gerekir |

---

## §6 — İLAN EDİLEN SEÇİM KURALI (ölçümden ÖNCE, sonra değiştirilmez)

Her segmentten sonra `scripts/evaluate_carpenter_anka.py --ceket-ekseni --n 100 --seed 42
--baseline data/anka_a1r.pt` koşulur (eşli: unutma ekseni tabana karşı).

1. **Sert kapı:** aday `unutma_gec == true` olmalı (T-0059: A ≤ **+%10** VE B ≥ **−5,0 puan**).
2. **Dışlama:** `ezber_orani >= %10` **VEYA** `tutarsizlik_orani >= %5` olan segment seçimden
   **çıkarılır** ve **gerekçesi ateşleyen koşuldan TÜRETİLİR** — sabit şablonla birleştirilmez.
   Ateşleyen **her** koşul **ayrı ayrı** listelenir.
   > **T-0096/K7 kusuru burada DÜZELTİLİR:** eski sürücü iki karşılaştırmayı `veya` ile
   > birleştirip ikisini de basıyordu ⇒ 6 gerekçenin **3'ü** `ezber %0.00 >= %10.0` gibi
   > **yanlış bir önerme** taşıyordu. Sayı doğru, **yüklem yanlıştı**.
3. **Seçim:** sert kapıyı geçenler arasında **en yüksek `rouge_l_ort`**; eşitlikte sırasıyla
   daha düşük `ezber_orani` → daha düşük `tutarsizlik_orani` → daha **az** kümülatif adım.
   > Uygulama **sırası** ilanla birebir olmalıdır (T-0096'da kod önce kalite filtresini, sonra
   > sert kapıyı uyguluyordu ⇒ kaydedilen gerekçe değişiyordu).
4. **Erken durma:** bir segmentte `ezber_orani >= %10` **VE** `rouge_l_ort` önceki segmente göre
   **ARTMIYORSA** koşum DURUR.
5. Hiçbir segment sert kapıyı geçmezse hüküm **"KAPIYI GECEN YOK"**tur; en iyi ceket ekseni
   beyan edilir, **"PASS" YAZILMAZ**.

**Teslim kaydı `hedef` alanını HER DALDA taşır** ve ateşleyen dal **`stderr`'e basar**
(T-0096/K5: `teslim` kaydı `hedef` taşımıyordu; kardeş dal taşıyordu ⇒ gözden kaçma, tasarım
değil. Ateşleyen dal hiçbir şey basmıyordu ⇒ karar log'da iz bırakmadı, T-0095 sınıfı).

---

## §7 — BEYAN EDİLEN SINIRLAR VE KUSURLAR

1. **Held-out SABİT** (operatör kararı 5) ⇒ **yeni olgular ÖLÇÜLMEZ**; ölçüt hâlâ **eski 450
   perspektifi** sorar. İddia *"model daha çok marangozluk öğrendi"* **DEĞİL**, *"eşleşen
   adımlarda ROUGE/unutma değişti mi"*dir. **Ç3 tam bu tuzağı yakalamak için ilan edildi.**
2. **Kısa sonda YOK** (operatör kararı 3) ⇒ hipotez **koşumun kendisiyle** sınanır; hipotez
   yanlışsa bedeli **~3,5–3,9 saatlik koşudur** ve bu bedel **burada, önceden** beyan edilir.
3. **Yeni olguların teknik doğruluğunu sınayan ORACLE YOK.** Her olgu `kaynak` alanı taşır, ama
   doğruluk yükü yürütücüdedir; yanlış bir olgu modele öğretilir. Bu bir **AÇIK** olarak rapora
   yazılır. Türetilemeyen olgu **yazılmaz**; kapıya ulaşılamazsa hüküm *"hedefe ulaşılamadı"*
   olur, **uydurma olguyla doldurulmaz**.
4. **`ESIK_KESISIM=80` ULAŞILAMAZ** (insan tavanı %13,00 ⇒ 6,15× uzak, T-0096/K3). Eşik
   **değiştirilmez**, **VAKUM** olarak raporlanır.
5. **Replay oranı taranmaz** (%25 sabit) — T-0096'nın ölçmediği kollar ölçülmemiş kalır.
6. `anka_a1r.pt` **AdamW momenti taşımaz** ⇒ S1 momentsız, ilk güncelleme **1,7306×** (T-0092).
7. **D1** (en-uzun-kök) ve **D4** (kesme/rakam) **kapsam dışı**.
8. **Belge–kod uyuşmazlığı (beyan):** `scripts/generate_specialization_dataset.py` §1/§2/§3
   başlıkları *"× 6 Farklı Perspektif"* der (`:264`, `:297`, `:324`) ama kod **5 / 4 / 3** üretir.
   Dosya **dokunulmaz** (ölçülmüş kaynak), uyuşmazlık **bildirilir**.
9. **`src/llm/frozen_guard.py:59` mutlak-yol FAIL-OPEN** açığı **kapatılmaz** — beyan edilir ve
   üretici ona **GÜVENMEZ** (hedefler göreli yolla + açık bayrakla verilir).
10. **Epoch sayısı korunmaz** (§1.6): eşleşen adımda T-0097 daha AZ epoch görür. Bu bir kusur
    değil **tasarımdır**; ama T-0096 ile kıyas okunurken **unutulmamalıdır**.

---

## §8 — YÖNETİŞİM

* **Antigravity'ye GÖREV AÇILMAZ** (operatör kısıtı: *"bu modele yeni görev açma"*) — yalnız danışma.
* **Kiralama:** donmuş **dosya** kiralanamaz ⇒ **üst dizin** kiralanır (`data/` · `data/eval/` ·
  `scratch/` · `.agent-bus/`), dosya `writes[]`'te **adıyla** beyan edilir.
* `--allow-frozen-write` **yalnız** beyan edilen donmuş hedefler için:
  `data/train_carpenter_specialization_anka_r18.bin` (+`.meta.json`) ve
  `data/anka_a1r_ceket_r18.pt` (+`.opt.pt`).
* **DOKUNULMAZ** (baş/son sha256 `scratch/t0097_kos/once.json`'da damgalı):
  `data/train_carpenter_specialization_anka.bin` · `data/anka_a1r.pt` · `data/anka_a1.pt` ·
  `data/train_chat_balanced.bin` · `data/pedagogy/**` · `data/pedagogy_canonical/**` ·
  `data/eval/anka_r17_*` · `scripts/**` · `src/**` · `scratch/t0096_*` ·
  `scratch/anka_r17_ceket_build.py`.
* `data/pedagogy/carpenter_specialization_dataset.jsonl` **ezilmediğinin kanıtı** raporda
  (baş/son sha256 `146a73dce0bf…` AYNI).
* `git add -A` / `git add .` **YASAK**; commit/push YOK. `data/eval/` dışına rapor yazılmaz.
* `/api/query` arka kapısı **ÇAĞRILMAZ** (T-0095/K15: `data/future_train_vector.jsonl`'e yazar).
* Koşum **sandbox DIŞINDA** (MPS sandbox'ta görünmez); koşum sırasında başka GPU işi **YOK**
  (T-0052: 0,43 → 3,45 sn/adım).

---

## §9 — ÇIKTILAR

* `data/train_carpenter_specialization_anka_r18.bin` (+`.meta.json`) — genişletilmiş ceket
* `data/anka_a1r_ceket_r18.pt` (+`.opt.pt`) — **yalnız** kapıyı geçen segment varsa
* `data/eval/anka_r18_build_2026-09-21.{md,json}` — derleme raporu + V1–V8
* `data/eval/anka_r18_ceket_2026-09-21.{md,json}` — koşum raporu + **T-0096 ile yan yana tablo**
* `.agent-bus/notes/T-0097.md` — koşum **sürerken** yazılan yürütücü notu
