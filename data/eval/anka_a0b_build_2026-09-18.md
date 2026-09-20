# T-0071 — Anka A0-b: külliyat DERLENDİ ve K1–K7 ölçüldü

**Görev:** T-0071 (Anka A0-b) · **başlangıç** 2026-09-18T12:38:46Z · **bitiş** 2026-09-18T13:02:04Z
**Talimat (Onkanat, 18 Eyl 2026):** *"…elinizdeki data-set ve Proje dokümanları ile modeli yeniden
aşama aşama eğitin. **Data setleri model ihtiyacına göre elden geçirin** mimari temel yapısını koruyun."*

**Bu görevin yeri:** A0'ın ikinci yarısı. T-0070 envanteri ölçtü ve **K1–K7'yi ölçümden ÖNCE ilan
etti** (`data/eval/anka_a0_design_2026-09-18.md` §4); burada **derleme** yapıldı ve **aynı kapılar** ölçüldü. Kapılar değiştirilmedi.

---

## 1. Ne yapıldı (ilan edilen tasarım kararları uygulandı)

| karar | uygulama |
|---|---|
| Tokenizer | `literal_entity_mode=True` + kanonik `data/rebuild/vocab_base_32852.json` |
| **id uzayı genişlemedi** | inşa öncesi **32852** → sonrası **32852** · `next_id=32852` — **DOĞRULANDI** (`SystemExit` bekçisi) |
| Belge kodlama | **makale düzeyi** `<BOS>…<EOS>` (paragraf değil) |
| Metin | `başlık\n` + arınmış markdown · T-0070 kuralları · `satir_min=0` |
| Eleme | arınmış metin < 300 karakter |
| Tekrarsızlık | arınmış metnin sha256'sı + belge `id`'si |
| Sıra | parquet dosya sırası (glob sorted) + satır sırası — **deterministik** |
| Çıktı | `data/anka_pretrain.bin` + `.meta.json` · `data/anka_pretrain_val.bin` + `.meta.json` |
| Artımlı yazma | her **2,000** makalede kontrol noktası (35 ara kayıt); `rc=0`'a bağlı |

**Sözlük değişmedi:** önce `7611b6a523a2bb52…` → sonra aynı ⇒
`sozluk_degismedi` = **True**.

---

## 2. Paralelleştirme ve **EŞDEĞERLİK KANITI**

Derleme CPU'dur (GPU'ya dokunulmadı) ama seri hız **24,484** jeton/sn ⇒
seri koşumda **68 dakika** olurdu. Bunun yerine **6** işçi süreçle paralelleştirildi.

> **Naif güven reddedildi.** Paralel bir hat *sıra* veya *eleme kararı* değiştirebilir; o zaman
> "aynı veri" iddiası kanıtsız kalır. Bu yüzden **seri referans yolu korundu** (`--seri`) ve iki
> yolun çıktısı **2,000,000** jetonluk sondayla **bit düzeyinde** kıyaslandı:

| kol | jeton | seri sha256 | paralel sha256 | sonuç |
|---|---|---|---|---|
| train | 2,000,000 | `07725f2886c53e62…` | `07725f2886c53e62…` | **AYNI** |
| val | 40,000 | `5a909f2c2a41e4fa…` | `5a909f2c2a41e4fa…` | **AYNI** |

⇒ **bit-özdeş = True**. Seri ve paralel yollar **aynı `_isci()` fonksiyonundan** geçer;
fark yalnız **süreç sınırıdır**, dolayısıyla bu test kuyruğu ve sırayı sınar.
**Bu kanıt olmadan paralel sayılar geçersiz sayılırdı.**

**Bellek bağı (ölçüm sonrası bulundu):** `Pool.imap` girdi üreticisini **sınırsız kuyruklar** ve
sonuçları sıra beklerken `_cache`'te biriktirir — 631 bin makalede GB'larca. Akış **2.000'lik
bloklara** bölündü (her blok tüketilmeden sonraki gönderilmez); sıra yine korunur, ölçülen hız
düşmedi (73,140 jeton/sn).

**Ölçülen hız:** akış **73,140** jeton/sn · toplam süre **23.3 dakika**
(akış 23.2 dk).

> ⚠️ **Duman ölçeğindeki oran bir hız ölçümü DEĞİLDİR.** 20 bin jetonluk koşumda hedef 3–4 makalede
> doluyor ve 6 işçinin sözlük kurulumu baskın çıkıyor; orada paralel *yavaş* görünür (3.662 vs
> 10.221 jeton/sn). Bu sayı **hiçbir yerde hız olarak raporlanmadı**.

---

## 3. ÖLÇÜLEN KAPILAR (T-0070 §4'te ilan edildi — değiştirilmedi)

| kapı | ölçülen | eşik | hüküm | not |
|---|---|---|---|---|
| **K1 NOKTLAMA** | `max_id`=32850 · blok oranı **%6.4683** | ≥32145 VE ≥%5.0 | **GEÇTİ** | ayırt edici: 12 eski `.bin`'de oran **tam %0,00** · ⚠️ akışın %39.1'i varlık bloğu — bkz. §4.5 |
| **K2 UNK TAVANI** | **%1.5758** (akış) | ≤%4.0 | **GEÇTİ** | ham örnek %6.59 — eşiği geçemezdi |
| **K3 PAD DİSİPLİNİ** | **%0.0000** | ≤%1.0 | **GEÇTİ** | eski SFT dosyaları %50–75 |
| **K4 ÖLÇEK** | **100,000,000** jeton | ≥90,000,000 | **GEÇTİ** | pay **+10,000,000** · ⚠️ bkz. §4 kısıtlılık |
| **K5 TEKRARSIZLIK** | train⊓val = **0** · benzersiz id 75,776 | kesişim = 0 | **GEÇTİ** | ⚠️ **ayırt edici DEĞİL** — bkz. §4.4 |
| **K6 İZLENEBİLİRLİK** | 2 kaynak · sha256 TAM | her kaynak + tam sha256 | **GEÇTİ** | kaynak jeton toplamı raporda |
| **K7 POZİTİF KONTROL** | noktalama token'ı **12** | > 0 | **GEÇTİ** | **HAT CANLI** |

**K7 yaşamsal:** hat canlı olduğu için K1–K6'nın **tamamı kanıtlıdır**; hat ölseydi hepsi
"KANITSIZ" ilan edilecekti.

### 3.1 Derlenen artefakt (yazılan dosyadan yeniden ölçüldü)

| | jeton | sha256 | PAD | UNK | noktalama | benzersiz id | <EOS> |
|---|---|---|---|---|---|---|---|
| `data/anka_pretrain.bin` | 100,000,000 | `0e55f0c0c24617a9323059c71563a9ce943177ccc90cd56824be1a7f26d2b164` | %0.0000 | %1.5781 | %6.4683 | 27,668 | 73,766 |
| `data/anka_pretrain_val.bin` | 2,000,000 | `4be459821814e9221f731d8a20c03ab790b460beb05619d68ba331028978141b` | %0.0000 | %1.4596 | %6.2816 | 13,041 | 2,008 |

`max_id` = **32850** (kanonik sözlüğün üst sınırı 32851) · `min_id` = 0

**Belge akışı:** 101,594 makale okundu · kullanılan **73,767**
(train) + **2,009** (val) · elenen (kısa) **25,818** ·
tekrar id **0** · tekrar metin **0**

### 3.2 Kaynak izlenebilirliği

| kaynak | jeton | okundu | sha256 |
|---|---|---|---|
| `tr-00000.parquet` | 102,001,712 | okundu | `17df2980dd5a28ce8f449ab7b199e875165bb7ad81305c1c544896d3c3f2b732` |
| `tr-00001.parquet` | 0 | okunmadı | `a358fb4fd5179e8ac282ff266c7ad19d53eafe680911986a7b2c04d941127ca9` |

Toplam kaynak **567 MB** · kullanılan **102.0 M** jeton (ilk dosya).

### 3.3 Sözlük kapsamı ve akışın bitişi (ölçüldü — hata değil, sınır)

| | train | val |
|---|---|---|
| benzersiz id | **27,668** | **13,041** |
| sözlük kapsamı | **%84.22** (32,852 girişin) | %39.70 |
| hiç görünmeyen giriş | **5,184** | 19,811 |
| varlık bloğu (32816–32851) | 35/36 giriş · 39,135,708 token | 35/36 |
| noktalama bloğu (32137–32145) | 9/9 giriş | 9/9 |

**Son belge KESİK:** train'de **73,767** belge kullanıldı ama
**73,766** `<EOS>` var ⇒ **1** belgenin `<EOS>`'u yok; val'de
1. Nedeni: hedef jeton sayısına **tam** oturmak için son makale **ortasından kesildi**,
yani akış belge ortasında bitiyor. Ön-eğitim **rastgele pencere** açtığı için bu zararsızdır
(`KristalDataset` düz akıştan `torch.randint` ile pencere seçer, belge sınırı önemsiz) — ama
"her belge `<EOS>` ile biter" cümlesi **yanlış olurdu** ve kayda geçirilir.

---

## 4. ⚠️ ÖLÇÜM SONRASI BULUNAN KUSURLAR (kapıya KATILMADI)

### 4.1 T-0070'in derleme projeksiyonu GEÇERSİZDİ

T-0070 §2.3 derleme süresini **~47 dakika** diye ilan etti:
`100M ÷ C['jeton_sn']` = `100M ÷ 35,520`. Ama o betikte
`jeton_sn = (ham_top.jeton + arn_top.jeton) / sure` — **iki tokenizasyon akışını tek süreye böler**,
yani *hiçbir* geçişin hızını tanımlamaz; ayrıca bellekteki metni tokenize eder, derlemenin ödediği
**parquet G/Ç'sini içermez**.

**Ölçülen gerçek seri hız 24,484 jeton/sn ⇒ 68 dakika** (ilan edilenin
~1.5 katı).

**Kural gereği bu düzeltme kapı tanımına katılmadı** — kapılar ölçümden önce ilan edilmişti ve
değiştirilemez; kusur **rapora** yazılır. Kapıların hiçbiri süre üzerine kurulu değildi, dolayısıyla
bu kusur K1–K7'nin hiçbirini etkilemez.

### 4.2 K4'ün **kısıtlılık bayrağı** — "tavan" külliyatın sınırı değil, TALEBİN sınırı

K4 **100,000,000** ile geçti, pay **+10,000,000**. Ama bu sayı
külliyatın kapasitesi **değil**: kurtarılan külliyatın arınmış kapasitesi ≈ **357 M
jeton** ve biz **%28.6**'ini kullandık. Derleme hedefe ulaşınca **kendini kesti**.
⇒ "102 M jeton elde edildi" doğru; "külliyat 102 M jeton" **yanlış olurdu**. Kıtlık yok, **seçim** var.

### 4.3 Donmuş-yol bekçisi bu betikte **çağrılmadı** (beyan)

`data/*.bin` deseni donmuştur ve `train.py`/`train_dpo.py`/`run_goal_pipeline.py`/
`retrain_clean_models.py` yazmadan önce `check_frozen_save_path` çağırır. **Bu derleyici betiği o
bekçiyi çağırmaz.** Dayanak bekçi değil **kiralamadır**: operatör Anka işi için donmuş veri yolu
güncelleme izni verdi ve T-0071 `data/` üst dizinini kiraladı (donmuş bir **dosya** kiralanamaz →
üst dizin kiralanır). Bu beyan, okuyucunun "bekçi koştu" varsaymaması içindir.

### 4.4 ⚠️ K5 **ayırt edici değil** — kendi ilan ettiğim kuralı kendi kapım çiğniyor

Kendi kuralım der ki: *ilan edilen kural ayırt edici olmalı* — hangi girdi onu düşürür? K5'in ölçümü
`train ∩ val = 0`. Ama val kümesi train **kesildikten sonraki** makalelerden alınır (tasarım kararı 7),
yani kesişim **inşa gereği imkânsızdır**; K5 ancak **kendi kodumdaki bir hatayla** düşebilir.
Dolayısıyla K5 bir *külliyat özelliği* değil, bir **kendini kontrol**tür.

**Dahası: tekrar eleme mekanizması bu külliyatta HİÇ TETİKLENMEDİ** — 100,000,000 jeton boyunca
`tekrar_id = 0`, `tekrar_metin = 0`.
⇒ Mekanizma **kanıtlanmadı** (bozuk olduğu da gösterilmedi; sıfır vaka = kanıtsızlık). K5 hükmü
"GEÇTİ"dir ama **boş bir kapıdır**; tavan/kanıtsızlık kuralı gereği bu birlikte raporlanır.

### 4.5 🔴 EN ÖNEMLİ BULGU — külliyatın **%39,1'i varlık işaretlemesi** (ölçüm sonrası)

K1 "noktalama var mı" diye soruyordu ve **evet** dedi. Ama kapıların hiçbiri **akışın neyden
oluştuğunu** sormuyordu. Yazılan dosyayı çözünce ortaya çıktı:

```
... 32155 9 · 32155 9 · 32153 7 · 32850 ' · 171 CASE_ABL · 349 beri
    32816 <ENT> · 32818 <CAP> · i n g i l t e r e · 32817 </ENT> · ' · 29 CASE_LOC ...
```

> **"İngiltere" tek bir jeton değil — `<ENT><CAP>i n g i l t e r e</ENT>`, 12 jeton.**

| | eski taban `train.bin` | eski wiki `train_wiki.bin` | **Anka `anka_pretrain.bin`** |
|---|---|---|---|
| jeton | 10,761,665 | 8,228,842 | 100,000,000 |
| `max_id` | 25664 | 27243 | 32850 |
| **`<ENT>` adedi** | **0** | **0** | **4,465,667** |
| **varlık bloğu oranı** | **%0.0000** | **%0.0000** | **%39.1357** |
| varlık içeriği oranı | — | — | **%37.40** |
| işaretleyici oranı | — | — | %8.93 |
| varlık uzunluğu (ort/medyan) | — | — | **8.37 / 8** jeton |
| `CASE_ABL` oranı | %0.0063 | %0.6255 | %0.3925 |
| `POSS_3SG` oranı | %0.0090 | %5.3939 | %2.6742 |

**İddiayı DARALTARAK söylüyorum:** *"yeni külliyat eskiden tamamen farklı"* **YANLIŞ olurdu** —
morfoloji etiketleri (`CASE_ABL`, `POSS_3SG`, `PART_An`…) **eski tabanda da vardı**
(%5.39 `POSS_3SG`). Ölçülen **tek yapısal fark** şudur:

> **Varlık işaretlemesi eskiden %0,0000 — şimdi %39.14.**
> Yani Anka'nın ön-eğitim jetonlarının **~%37,4'ü** harf harf yazılmış varlık **içeriği**,
> **%8,9'u** `<ENT>`/`</ENT>` işaretleyicisidir.

**Neden kapılar yakalamadı:** K1 yalnız *noktalama blok oranı* ve `max_id` sorar; `<ENT>` o eşiği
**geçirir** (noktalama gerçekten var). **Tavanda geçen bir kapı boştur** — kendi kuralım.

**Bu bir tasarım SORUSUDUR, sessizce kabul edilmiş sayılmaz.** T-0071 şartnamesi
`literal_entity_mode=True`'yu **kanoniklikle** gerekçelendirdi ("T-0044 sonrası kanonik hat böyle"),
**ön-eğitime uygunlukla değil**; ölçülen **%39,1** sonucu şartnamede **öngörülmemişti**.
Sonuçları:

1. Ön-eğitim bütçesinin **~%37'si** varlık içinde **harf tahminine** gider — jeton-verimsiz.
2. Eski zincirin modeli bu temsili **hiç görmedi** (o külliyatta varlık bloğu %0) ⇒ Anka ile eski
   zincir arasında **temsil farkı** var; davranış kıyası elmayla elma değildir.
3. Model **düz metin değil**, morfoloji+entity ile **etiketlenmiş** bir akış öğrenir.

⇒ **A1 öncesi karar gerektirir:** (a) bu temsille ön-eğitime devam, (b) düz-metin varyantı da
derleyip kıyaslama, (c) varlık bloğunu dışlayan bir temsil. **Karar operatöründür**; bu rapor
sessizce (a)'yı seçmiş sayılmaz.

### 4.6 Kendi projeksiyonum yanlıştı — T-0070'inkiyle **aynı sınıf** hata

Koşum başlarken ilk 2.000 makaleden **3.901 jeton/makale** ölçüp "100 M jeton ≈ 25.600 makale ⇒
~14 dakika" dedim. **Yanlıştı:** makale başına verim akış boyunca düşüyor (ilk aralık 3.901 → son
aralıklar ~350–1.200), çünkü parquet sırasının **başı uzun makalelerle dolu**. Gerçekleşen süre
**23.3 dakika** oldu. ⇒ Bir **önekten** türetilen hızı bütüne taşımak, T-0070'in iki
akışı tek süreye bölmesiyle **aynı sınıf** hatadır; ikisi de bu raporda kayıtlıdır.

---

## 5. KENDİNİ YANLIŞLAYABİLME — *post-hoc*, önceden ilan EDİLMEDİ

T-0070 §5 kendi çürütme şartını **ölçümden önce** ilan etti ve ölçtü (`çürüdü = False`).
**Bu görev ayrı bir çürütme şartı önceden kaydetmedi — bu T-0071'in tasarım eksiğidir** ve
aşağıdaki şart *ölçümden sonra* yazıldığı için **kanıt gücü düşüktür**; kayda geçirilir:

> *Post-hoc şart:* "1500 makalelik örnek külliyatı temsil etmiyor" — yani tam derlemenin K2/K1
> oranları örneğinkinden **sapmış** olsaydı, örnek-bazlı tasarım çürürdü.

**Ölçülen:** örnek UNK %1.61 → tam derleme **%1.58**;
örnek noktalama %6.08 → tam derleme **%6.47**.
**Kalite oranlarında** sapma küçük — **ama bu hüküm önceden ilan edilmediği için zayıftır.**

### 5.1 ⚠️ Ama örnek **VERİM** ekseninde temsili DEĞİL (ölçüldü — şartı kısmen çürütüyor)

| eksen | T-0070 örneği (1500 makale) | T-0071 tam koşum | oran |
|---|---|---|---|
| kullanılan makale başına jeton | **901.8** | **1,346.1** | ×1.49 |
| elenen oranı | %38.4 | %25.4 | — |

Örnek **rastgele** çekildi, koşum ise parquet sırasını **baştan** tüketiyor; sıranın başı uzun
makalelerle dolu olduğu için iki küme **aynı dağılımdan gelmiyor**. ⇒ Örnek, *kalite oranlarında*
temsili görünürken *verim* ekseninde **1,76 kat** sapıyor. Bu, T-0070 §2.3'ün *kapasite/derleme
süresi* projeksiyonlarının örnekten bütüne taşınamayacağını gösterir — ve **§4.6'daki kendi hatamın
kaynağı da tam olarak budur**.

---

## 6. BU GÖREVİN İDDİA ETMEDİĞİ ŞEYLER

1. **"Anka iyi olacak" DEĞİL.** Bu bir *veri derlemesidir*; hiçbir model eğitilmedi, hiçbir `.pt`
   yazılmadı.
2. **Noktalama *yeteneği* ölçülmedi.** K1 noktalamanın **veride bulunduğunu** ölçer; modelin onu
   **kullanabildiğini** değil. O ölçüm A1 sonrası ayrı bir kapıdır.
3. **UNK %1.58 bir "iyi" değil**, ölçülmüş bir orandır; mutlak eşiği yok.
4. **Ceketin T-0067/T-0068 kusuru giderilmiş sayılmaz**; giderildiği ölçülmedi.
5. **Çıktı kalitesi hakkında hiçbir şey.** Yalnız oranlar ve kapılar ölçüldü.
6. **"Bu külliyat düz metindir" DEĞİL.** Ölçüldü: jetonların **%37.4'i** harf
   harf yazılmış varlık **içeriği**, **%8.9'i** işaretleyicidir (bkz. §4.5).
   "Kanonik hattan geçti" ≠ "ön-eğitime uygundur".

---

## 7. Artefaktlar (tam 64 karakter sha256; bu belgenin kendisi ve bus durum dosyaları hariç)

| yol | bayt | sha256 |
|---|---|---|
| `scratch/anka_a0b_build.py` | 18,877 | `e02923cae014bf5ae71b60480748e43a7af7d60d9546659a2f3ef9579c67a29c` |
| `scratch/anka_a0b_report.py` | 33,572 | `8cb8ab2075834a16ccb7fcb72581c87d0d8acd190816a140cf35a9b0cead45f4` |
| `data/anka_pretrain.bin` | 200,000,000 | `0e55f0c0c24617a9323059c71563a9ce943177ccc90cd56824be1a7f26d2b164` |
| `data/anka_pretrain.bin.meta.json` | 866 | `ed78d9a78fc45ed6290623ffe32227fde8fa9a1c69850a7bddf8d999c9e378b2` |
| `data/anka_pretrain_val.bin` | 4,000,000 | `4be459821814e9221f731d8a20c03ab790b460beb05619d68ba331028978141b` |
| `data/anka_pretrain_val.bin.meta.json` | 864 | `8d586e18bd7f21f8f402493f2334f8e18d5aa13bbc7b495bb11f9505a4856c17` |
| `data/eval/anka_a0b_build_2026-09-18.json` | 7,676 | `c5798955388f7e8228c3d5ec9392a9eb88f970c75b5741a785872183dc8aef3b` |
| `data/eval/anka_a0_design_2026-09-18.md` | 13,009 | `fc06517ab01370aceeab693c209fb71b0caa3990621dc92ceafbde57b8db97c7` |

Bu belgenin kendi sha256'sı tabloya **konmadı** (kendine referans); üreticinin `[md]` satırında
basılır. Bus'a ait `state/tasks|results` dosyaları da konmadı — kapanış çağrısı onları kendisi
günceller ve tablo sahte ihlal üretirdi (T-0068 dersi).

---

## 8. Sonraki adım

**⚠️ ÖNCE KARAR (operatör):** §4.5'teki temsil sorusu — **A1 tam olarak bu külliyatla mı koşacak?**
Üç seçenek ölçülmüş bir fark üzerinde duruyor (varlık bloğu **%0 → %39.1**);
karar verilmeden A1 başlatılırsa temsil **sessizce** seçilmiş olur.

**T-0072 (A1 planlama):** GPU verim sondası (sandbox **DIŞI**) — blok 128/256 × yığın 8/16/32
üzerinde **jeton/sn ölçümü**; adım bütçesi ölçümden ilan edilsin. Bilinen sınır: ~1,55 sn/adım
(blok 128) ile 102 M jeton ≈ 97.656 adım ≈ **42 saat** — kullanıcının **~1 gün** bütçesinin
**üzerinde**, dolayısıyla yığın/blok ayarı gerçek bir açık sorudur.
**T-0073 (A1):** ön-eğitim koşumu (MPS, sandbox DIŞI, makinede başka GPU işi **YOK** — T-0052).

İlgili: [[anka-yeni-model-adi]], [[hf-onbelleginde-wikipedia-kurtarildi]], [[mimari-korunur-kristallm]],
[[uzun-olcum-artimli-yazmali]], [[tasarim-sayisi-betikle-hesaplanmali]], [[cift-yonunu-alan-adindan-oku]].
