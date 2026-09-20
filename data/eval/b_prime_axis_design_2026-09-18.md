# B′ Ekseninin Onarımı — Tasarım ve **ÖNCEDEN İLAN EDİLEN** Eşik

**Damga:** 2026-09-18T04:25Z · **Yazan:** claude · **Statü:** TASARIM — **ölçüm bu dokümandan SONRA yapılır**
**Görev:** T-0063 · **Dayanak:** Onkanat kararı (18 Eyl): *"önce B′ eksenini onar"*
**Önceki durum:** `data/eval/forgetting_measurement_design_2026-09-18.md` §4.4 — B′ ekseni **geçersiz** ilan edildi.

---

## 1. Kusurun teşhisi (T-0062'de ölçüldü)

Top-1 doğru sayımı, **örneklemdeki çoğunluk id'nin payı kadar bedava** kazanır:

| korpus | çoğunluk payı = sabit-tahmin top-1 | model tabanı top-1 |
|---|---|---|
| `gts_dict` | %54,8 (`,`) | %18,3 |
| `raw_full` | %70,1 (`.`) | %26,1 |

Taban **sabit bir tahmincinin altında** kaldığı için eksende "koruma oranı" yorumu geçersizdi. Ayrıca örnekleme dosya sırasından alınıyordu (kapsam %3,0 / %7,6).

## 2. Ama beceri VAR — kırılan şey ölçüt

Aynı ölçümler şunu da gösterdi (32.852 kelimelik sözlükte doğru noktalama id'sinin **sırası**):

| korpus | taban **ort. rank** | şans düzeyi rank | oran |
|---|---|---|---|
| `raw_full` | **38,0** | ~16.426 | ~430× |
| `gts_dict` | **~86** | ~16.426 | ~190× |

Model doğru noktalamayı ortalamada ilk 40–90 içine koyuyor. Yani **beceri var; top-1 ölçütü onu göremiyor.** Teşhis: eksik olan şey "noktalama becerisi" değil, **marginal-bağımsız bir ölçüt**.

## 3. Yeni eksenin tanımı (ilan)

### 3.1 Örneklem: **DENGELİ 3-sınıf**, tutulmuş zemin
Her korpusta, tutulmuş kayıtlarda **en çok geçen 3 noktalama id'si** alınır ve her birinden **eşit sayıda** konum seçilir:

| korpus | seçilen 3 id | her birinden n | toplam |
|---|---|---|---|
| `raw_full` | `.` (32137), `:` (32142), `,` (32138) | **3.955** | **11.865** |
| `gts_dict` | `,` (32138), `.` (32137), `!` (32140) | **2.069** | **6.207** |

> **Düzeltme notu (ölçümden ÖNCE):** ilk taslakta n, `punct_heldout`'un tek-konum-per-kayıt havuzundan
> türetilip 114/104 sanılmıştı. Havuz **tüm konumlar** üzerinden yeniden sayıldığında çok daha büyük çıktı
> (yukarıdaki sayılar); üst sınır kaldırıldı. Bu **gücü artıran** bir değişikliktir; eşik ve ölçüt aynı kaldı.
>
> **Yan bulgu:** tek-konum-per-kayıt örneklemesi id dağılımını da çarpıtıyormuş. `raw_full`'da **gerçek**
> oranlar `.` %47,4 / `:` %28,5 / `,` %24,1 (n=16.414) iken, tek-konum örnekleminde `.` %70,1 / `,` %17,2 /
> `:` %11,4 çıkıyordu — yani bir kayıtta **ilk geçen** işaret sistematik olarak kayırılıyor (`.` cümle sonu
> olduğu için kayıt ortasında da erken düşüyor). Dengeleme bu çarpıklığı da ortadan kaldırır.

Dengeleme **kritik**: denge sayesinde "hep çoğunluk id'sini tahmin et" stratejisi şans düzeyinde kalır ve **marginal bedava kazanç sağlamaz.** Konumlar kayıt içinde döner (`start = (k·13) mod L`), kayıtlar `(k·7919 mod N)` ile dolaşılır.

### 3.2 Birincil ölçüt: **kısıtlı top-1** (3 id arasında argmax)
Her konumda modelin son-token logit'leri **yalnız seçilen 3 noktalama id'sine** kısıtlanır; argmax doğru id ise isabet.

- **Şans düzeyi = %33,3** (dengeli 3-sınıf).
- **Marginal-yalnız strateji = %33,3** (dengeleme sayesinde).
- Dolayısıyla **%33,3'ün üstü, bağlamın kullanıldığının kanıtıdır.**

### 3.3 İkincil (rapor, kapı değil)
Tam sözlükte doğru id'nin **ort. ve medyan rank**'i; GA; **karıştırılmış-bağlam** ve **yanlış-id** kontrolleri (§4).

## 4. Pozitif kontroller — eksen bunları GEÇMEZSE REDDEDİLİR

| # | Kontrol | Ne kanıtlar | Reddetme ölçütü (ilan) |
|---|---|---|---|
| **K1** | **Karıştırılmış bağlam:** aynı konum, bağlam korpustan rastgele **başka** bir 32-token ile değiştirilir | Model bağlamı kullanıyorsa isabet **düşer** | Karıştırılmış bağlam skoru [%25, %42] aralığında **değilse** (yani şans düzeyinden sapıyorsa) kontrol bozuk → eksen reddedilir |
| **K2** | **Gerçek bağlam > karıştırılmış bağlam:** ikisi arasındaki fark | Becerinin kaynağı bağlamdır | Gerçek ≤ karıştırılmış ise → **eksen reddedilir** |
| **K3** | **Karıştırılmış hedef (permüte):** bağlam **aynen kalır**, hedef her konum için korpustan **başka bir konumun** gerçek id'siyle değiştirilir | Model bağlamla hedef arasındaki **gerçek bağlantıyı** kuruyor mu; yoksa "bu bağlamda şu işaret olur" yanlılığı mı var | Karıştırılmış hedef skoru gerçek skora yakınsa (fark < 10 puan) → ayırt etme yok → **eksen reddedilir** |
| **K4** | **Taban şansın üstünde mi** (binom testi, tek yönlü) | Ölçülecek bir beceri var | p ≥ 0,01 ise → **eksen reddedilir** |
| **K5** | İki bağımsız korpusta **aynı yön** | Zemine bağlı olmadığı | İşaret tutarsızsa → uyarı, karar tek korpusta verilmez |

**K3'ün önemi:** bağlam **aynen** kalır, yalnızca bağlam–hedef **eşleşmesi** bozulur; marginal ve bağlam yanlılıkları paylaşılır. Fark yalnızca "bu bağlam gerçekten bu işarete mi gidiyor" bilgisinden gelir → **marginal-bağımsızlığın doğrudan testi**.

> **Düzeltme notu (ölçümden ÖNCE, aynı damga günü):** K3 ilk yazımda "yanlış-id isabeti" diye muğlak bırakılmıştı; ölçülebilir hale getirmek için **karıştırılmış hedef (permüte)** olarak kesinleştirildi. Hiçbir ölçüm bu değişiklikten önce yapılmadı.

## 5. **ÖNCEDEN İLAN EDİLEN** eşik

> **B′-onarılmış PASS ⇔ bir kolun kısıtlı top-1'i, tabanın kısıtlı top-1'inin ≥ %90'ıdır** (göreli koruma).

Gerekçe: A′ ekseni **%10 göreli bant** kullanıyor; B′'de aynı bandı aynı yönde kullanmak iki ekseni **kıyaslanabilir** kılar. Şans düzeyine göre normalize edilmiş bir ölçüt olduğu için taban değeri bilinmeden ilan edilebilir.

**Ayrıca raporlanır (kapı değil):** mutlak puan düşüşü; şans-düzeyine göre normalize kazanç (`(skor − 1/3) / (taban − 1/3)`); K1–K3 kontrol değerleri; GA.

**Eşik bu dokümanın damgasından SONRA değiştirilmez.** Bir kol sınırda kalırsa (retention %88–92) bu **açıkça** raporlanır ve karar kullanıcıya bırakılır.

## 6. Ne ölçülecek

Beş checkpoint, **aynı konumlar** (eşleştirilmiş → McNemar / eşleştirilmiş bootstrap):
`f3_clean` (taban) · `carpenter_v2` (%0) · `f4_r05` (%5) · `f4_r10` (%10) · `f4_replay` (%25).
İki bağımsız tutulmuş korpus. Cihaz damgası her koşumda yazılır; CPU/MPS farkı tek bir kontrol koşumuyla izole edilir (T-0062 dersi).

## 7. Araç

`scratch/t0063_b_probe.py` — saf fonksiyonlar, deterministik, `scratch/t0062_ab_probe.py`'nin pencerelerini ve `load_clean` yolunu yeniden kullanır. Çıktı: `scratch/t0063_b_<korpus>.json`.

## 8. Sınırlar (dürüstçe)

- ~~**n küçük** (342 / 312 konum)~~ → **GEÇERSİZ (ölçümden önce düzeltildi):** §3.1 düzeltme notu üst sınırı kaldırdı; kanonik n **11.865** (`raw_full`) ve **6.207** (`gts_dict`). %10 göreli düşüş için güç artık yeterli (n=11.865'te %10 göreli fark ~±0,6 puan GA ile ayrılır); yine de GA raporlanır.
- **3 id'ye kısıtlı**: `?` `-` `:` gibi seyreklikler dışarıda kalır (korpusta yeterli örnek yok). Eksen "noktalama"nın tamamını değil, **dengelenebilen üç sınıfını** temsil eder.
- **Kısıtlı argmax** yapay bir görevdir (model normalde 32 bin sınıf arasından seçer); bu bilinerek seçildi, çünkü tam-sözlük top-1'i marginal tarafından ele geçiriliyordu.
- Eksen geçerse bile **eski B′ sayılarıyla kıyaslanamaz** (farklı ölçüt, farklı örneklem).

---

## 9. ÖLÇÜM SONUÇLARI — *bu bölüm ölçümden SONRA eklendi*

> **Damga:** 2026-09-18T06:55Z · §1–§8'in damgası 04:25Z. **§5'teki eşik ve §4'teki kontrol ölçütleri DEĞİŞTİRİLMEDİ**; bu bölüm yalnız sonucu kaydeder.

**Eksen iki bağımsız korpusta da GEÇERLİ** (K1–K4 hepsi geçti):

| korpus | şans | taban | K1 bağlam | K3 hedef | K3 farkı | hüküm |
|---|---|---|---|---|---|---|
| `raw_full` | %33,3 | %43,35 | %32,80 | %32,79 | +10,55 (eşik 10) | GEÇERLİ |
| `gts_dict` | %33,3 | %45,66 | %32,59 | %33,35 | +12,31 | GEÇERLİ |

**İlan edilen ≥ %90 koruma kapısı: dört kolun DÖRDÜ de geçti (her iki korpusta).**

| kol | `raw_full` | `gts_dict` | hüküm |
|---|---|---|---|
| %0 `carpenter_v2` | %97,7 | **%91,5** ← §5 sınır bandında | GEÇTİ |
| %5 `f4_r05` | %98,6 | %101,1 | GEÇTİ *(K5 işaret uyarısı)* |
| %10 `f4_r10` | %96,7 | %96,6 | GEÇTİ *(iki korpusta anlamlı ama küçük kayıp)* |
| %25 `f4_replay` | %103,8 | %100,1 | GEÇTİ |

**§5'in sınır-bandı kuralı işletildi:** `carpenter_v2`/`gts_dict` = **%91,53**, ilan edilen %88–92 bandının içinde → bu hücre **açıkça raporlanır** (yukarıda işaretli). Hükmü değiştirmiyor çünkü aynı kol zaten A′'dan kalıyor.

**K5 işaret tutarsızlığı:** `f4_r05` (raw +0,61 puan kayıp · gts −0,50 puan kazanç) → tasarım gereği **karar tek korpusta verilmez**; hüküm iki korpusta da GEÇTİ olduğu için sonuç etkilenmiyor, ama %5 için kurulabilecek cümle "%5 korur" değil **"%5'te saptanabilir değişim yok"**tur.

**Sınır (dürüstçe):** `raw_full` K3 farkı **10,55** — ilan edilen 10 puan eşiğinin 0,55 puan üstünde. Eşik değiştirilmedi; sınır kayda geçti.

> **SINIR GÜNCELLEMESİ (T-0065, bağımsız doğrulama bulgusu B2):** bu marj istatistiksel olarak **kesin değil**. Eşleştirilmiş %95 GA **[9,33, 11,78]** puan (bootstrap [9,31, 11,85]); H0: fark ≤ 10 için tek yönlü z-testi **p = 0,1884** → **alt sınır 10,0'ın ALTINDA**. Yani `raw_full`'un K3'ü nokta tahmini başarısıdır. §5'teki eşik ve §4'teki kontrol ölçütleri **değiştirilmedi**; eksenin geçerliliği bu marja dayanmaz (asıl kanıt K1/K3'ün şans düzeyine oturmasıdır: 32,6–33,4 → bkz §9 tablo).
>
> **B3:** `carpenter_v2`/`gts_dict` koruması %91,53 (bandın içinde) ama eşleştirilmiş bootstrap %95 GA **[%89,81, %93,25]** → alt sınır %90 kapısının altında; o hücre kapıyı nokta tahminiyle geçmiştir. Kol zaten A′'dan elendiği için hüküm etkilenmez.
>
> **B1 (yön tersliği):** §9'daki hiçbir ifade etkilenmedi (bu bölümde %0 için "kazanç" iddiası yoktu); düzeltme T-0063 notu ve raporundadır. Kayıt: `data/eval/t0065_verification_processing_2026-09-18.md`.
