# Faz 2 Ön-Ölçümü · D1'in Mekanizması ve Aday Onarımların Çürütülmesi

**Tarih:** 19 Eyl 2026 · **Görev:** T-0080 · **Yürütücü:** claude (tek yürütücü)
**Aşama:** Faz 2 **AÇILMADI** — ön koşul ölçümü
**Hüküm:** **D1 hiçbir el kuralıyla onarılamaz; ölçütü (G1) hata ölçüsü değildir.**

Bu belge, donmuş hiçbir dosyaya **dokunmadan** yapılan ön ölçümün kapanışıdır.
Üç aday tasarım ölçüldü; **üçü de kendi ölçütünde çürüdü.** Kabın kendisi
fail-closed ve pozitif/negatif kontrollüdür.

> **İki cümle:** (1) Bugünkü D1 davranışının nedeni **skorlama değil, `roots.tsv`
> alfabetik sırasıdır** — beraberlikte önek (kısa kök) daima önce gelir, 9/9.
> (2) **Uzunluk ekseni bu etiketlerde ayırt edici değildir** (net −1): "en uzun kök"
> 4 yanlışı düzeltir, 5 doğruyu bozar. D1'in çözümü **plausibility oracle'ıdır**,
> yeni bir skor kuralı değil.

---

## 1. İlan edilen kapılar (ölçümden ÖNCE yazıldı, sonra değiştirilmedi)

| Hipotez | İlan edilen eşik | Ölçülen | Hüküm |
|---|---|---|---|
| **H1** ihlaller beraberlikten mi geliyor | ≥ %90 ⇒ `BERABERLIK_KIRMA` | **%99,03** (2.642/2.668) | **BERABERLIK_KIRMA** |
| **H6** ad alanı ayırma (silmeden) G1'i düşürür | düşüş ≥ %50 | **%65,48** (2.668→921) | **DESTEKLENDİ** → §7'de **ÇÜRÜDÜ** |
| **H3** filtreli sözlük (silerek) G1'i düşürür | düşüş ≥ %50 | %74,95 | DESTEKLENDİ → §8'de **elendi** |
| **H4** V1 kırılması (bugün-doğru 5 vaka) korunur | 5/5 | silme **5/5** · ayırma **5/5** | korundu |
| **H5** D1 hedefi (4 vaka) düzelir | 4/4 | silme **2/4** · ayırma **2/4** | **KALDI** |
| **ÖLÇÜT** G1 `DOGRU`yu ihlal saymaz | yanlış pozitif 0 | **5/9** | **ÖLÇÜT_HATA_ÖLÇÜSÜ_DEĞİL** |
| **H6c** G1 düşüşü karardan mı geliyor | silinme payı < %50 | **%85,75** | **TOTOLOJİ_BASKIN** |
| **H8** uzunluk ekseni ayırt edici mi | net > 0 | **net −1** (4 düzeltir / 5 bozar) | **AYIRT_EDİCİ_DEĞİL** |

**H6 ve H3, ilan edilen eşiği geçti — ama kendi alt-ölçümleri (H6c, H3'ün kapsama
kaybı) onları çürüttü.** Eşik geçmek yetmiyor; §12/§13'e bakınız.

---

## 2. MEKANİZMA: D1 bir skorlama değil, `roots.tsv` sıra artefaktıdır

`data/lexicon/roots.tsv` **alfabetik sıralıdır**. Bir kök, uzantısının **önekidir**;
alfabetik sırada önek **daima önce** gelir. Skor berabere kaldığında kazananı
`sorted()`'un kararlı sırası = trie gezinme sırası = **satır sırası** belirler
(`core.py`). Sonuç: **beraberlik daima kısa köke (öneke) kırılır.**

Ölçüm — yer-gerçeği etiketli **9 vakanın 9'u**:

| Yüzey | Etiket | Beraberlik | Kazanan | Kazanan satır | Rakip | Rakip satır | Kazanan önek mi |
|---|---|---|---|---|---|---|---|
| `bunun` | DOĞRU | VAR | `bu` | 8469 | `bun` | 8684 | **EVET** |
| `onun` | DOĞRU | VAR | `o` | 31630 | `on` | 32023 | **EVET** |
| `ölmeden` | DOĞRU | VAR | `öl` | 48341 | `ölme` | 48365 | **EVET** |
| `topraklar` | DOĞRU | VAR | `toprak` | 41099 | `toprakla` | 41102 | **EVET** |
| `günümüzde` | DOĞRU | VAR | `gün` | 17910 | `günü` | 17980 | **EVET** |
| `ağacı` | YANLIŞ | VAR | `ağa` | 5065 | `ağaç` | 5091 | **EVET** |
| `aklı` | YANLIŞ | VAR | `ak` | 2187 | `akıl` | 2491 | **EVET** |
| `kanadı` | YANLIŞ | VAR | `kana` | 22352 | `kanat` | 22371 | **EVET** |
| `gönlü` | YANLIŞ | VAR | `gön` | 17458 | `gönül` | 17491 | **EVET** |
| | | **9/9** | | **9/9 daha erken** | | | **9/9** |

**Satır farkları 4–304 satır.** Yani `ağa`(5065) ile `ağaç`(5091) neredeyse
komşudur — dosya, önek/uzantı çiftlerini yan yana üretmiş. Kazanan **her vakada**
(a) daha erken satır, (b) diğerinin öneki, (c) daha kısa.

Bunun iki sonucu var:

1. **Planın "yazı-tura" ifadesi yanlıştı.** Beraberlik rastgele değil,
   **deterministik olarak önek lehine** kırılıyor. `data/lexicon/roots.tsv` bir
   satır eklendiğinde D1 kararı **sessizce** değişebilir.
2. **Hem 5 DOĞRU hem 4 YANLIŞ vaka yapısal olarak özdeştir** (önek/uzatma
   beraberliği). Bu yüzden **hiçbir uzunluk kuralı ikisini ayıramaz** — §3.

---

## 3. H8: UZUNLUK EKSENİ AYIRT EDİCİ DEĞİLDİR (V1'in sondası)

"En uzun kök kazanır" kuralı (`V1`) **yalnız beraberlikte** uygulanırsa (yani
birincil anahtar "en az morfem" korunur, V1'in hatası buydu):

| | sayı |
|---|---|
| Beraberlik olan vaka | **9/9** |
| Kazananın önek olduğu vaka | **9/9** |
| "En uzun" kuralının **değiştirdiği** vaka | **9/9** |
| **Düzelttiği yanlış** | **4** (`ağacı` `aklı` `kanadı` `gönlü`) |
| **Bozduğu doğru** | **5** (`bunun` `onun` `ölmeden` `topraklar` `günümüzde`) |
| **NET** | **−1** |

> **Uzunluk ekseni bu etiketlerde sıfır (negatif) ayırt edici güç taşıyor.**
> Beklenen kazanç 4, beklenen kayıp 5 ⇒ kural **durumdan kötüdür**.
> V1'in çürümesinin nedeni bu; H6'nın `ince`'i yıkmasının nedeni de bu.

**Kendi kusurum (düzeltildi, yeniden koşuldu):** ilk koşumda hüküm eşiğini
`net != 0` yazmıştım ve `net = −1` yanıltıcı biçimde `KISMEN_AYIRT_EDICI` etiketi
üretti. Eşik `net <= 0` olarak düzeltildi, betik yeniden koşuldu; düzeltme
`esik_duzeltmesi` alanına yazıldı. (Etiket, sayının kendisini gizlemişti.)

---

## 4. H1 / H1'in kaynak dağılımı

İhlallerin **%99,03'ü (2.642/2.668) skor beraberliğidir**; yalnız **26**'sı
beraberlik değil. Yani D1 **beraberlik kırma** problemidir.

**Seçilen kısa kökün kaynağı:**

| Kaynak | Adet | Pay |
|---|---|---|
| `çekirdek` blok (sıralı 50.474 satır) | **1.878** | **%70,4** |
| `türemiş_gorunumlu` lemma | 733 | %27,5 |
| `kuyruk` bloğu (1.900 satır) | **57** | **%2,1** |

> **Planın "kuyruk bloğu tasfiyesi" kaldıracı D1'in yalnız %2,1'idir.**
> Asıl kaynak **çekirdek bloktadır** ⇒ Fay 1'in kuyruk temizliği D1'i
> **çözmez** (ve çözmesi beklenmemelidir).

---

## 5. H6 tasarımı ve ham sayıları (SİLMEDEN ad alanı ayırma)

18.244 türemiş-görünümlü lemma **sözlükte kalır**; yalnızca onlardan **başlayan
yollar kök adayı sayılmaz**. Hiç yol kalmazsa eski seçime dönülür.

| Ölçüm | Değer |
|---|---|
| Kapsama kaybı | **0** |
| G1 | %2,3943 → **%0,8265** (düşüş **%65,48**) |
| Seçimi değişen yüzey | **18.277** (16,4%) |
| Tamamen-türemiş-kökü kalan yüzey | 14.790 |
| İlan edilen G1 kapısı (≤ %1,0) | **GEÇİYOR** |

Ama §6 ve §7 bu tabloyu geçersiz kılıyor.

---

## 6. H6b: DEĞİŞİMİN CERRAHİSİ — 18.277 karar, 9 etiket

| Geçiş | Adet |
|---|---|
| ihlal → temiz | **394** |
| **temiz → ihlal (YENİ ihlal)** | **145** |
| **yanal** (durum değişmedi) | **17.738** (%97,05) |
| ihlal → ihlal | 2 |

Ek olarak: kök kısaldı **18.036** · uzadı 226 · **bütün-kelime okunuşu yok olan 15**.

**Yani: 18.277 kararın %97'si `G1` durumunu hiç değiştirmiyor** — ve G1 zaten
bir hata ölçüsü değil (§9). Bu 17.738 yanal değişim **hiçbir ölçüte bağlı değil.**

**Bütün-kelime okunuşu yok edilenler** (ilk 15):
`ince` · `resim` · **`için`** · `gösterilen` · `eser` · `Han` · `doğum` · `gelen` ·
`Batı` · `Kara` · **`ile`** · **`üzerine`** · `posta` · `ağı` · `değer`

> ⚠️ **Ölçüm benzersiz YÜZEY üzerindedir; bu liste Türkçenin en sık işlev
> sözcüklerini taşır (`için`, `ile`, `üzerine`).** Benzersiz yüzey sayısı (15)
> külliyat **sıklığındaki** hasarı **olduğundan küçük** gösterir. Bu kabın ölçülen
> sınırıdır — sıklık-ağırlıklı hasar bu turda **ölçülmedi**.

---

## 7. H6c: TOTOLOJİ — karar veren sayı (DOĞRUDAN ölçüm)

`G1` *"aday kümesinde daha uzun köklü geçerli yol **var mı**"* diye sorar.
**H6 tam o adayı siler.** O hâlde seçim hiç değişmese bile G1 düşer.

Bunu **çıkarımla değil doğrudan** ölçtüm: *seçilen analiz birebir aynı olduğu hâlde
G1'i `ihlal`den `temiz`e dönen yüzeyler.*

| | Adet | Pay |
|---|---|---|
| Toplam G1 düşüşü | **1.747** | 100% |
| Karardan gelen net (394 − 145) | **249** | **%14,25** |
| **Aynı kararla flip** (aday silinmesi) | **1.498** | **%85,75** |

İki bağımsız yol **birebir aynı** sayıyı verdi (doğrudan sayım 1.498 · çıkarım
1.747 − 249 = 1.498) — kap kendi içinde tutarlı.

> **H6'nın "G1'i %65 düşürdü" başlığı çürüdü: düşüşün %85,75'i ölçütün KENDİ
> SAYDĞI ADAYIN SİLİNMESİDİR, iyileşme değil.**
> Gerçek karar değişimi net **249**'dur (%14,25) — ve o da `G1` üzerinde
> ölçülmüştür; `G1` ise hata ölçüsü **değildir** (§9). Yani H6'nın **kanıtlanmış
> hiçbir kazancı yoktur**; buna karşılık **17.738 yanal + 145 bozucu** kararı
> kanıtsız vermektedir.
> (Aile: `[[gosterim-uyusmazligi-olcutu-oldurur]]` — ölçüt ölür.)

---

## 8. H3 (silerek ayırma) — elendi

| Ölçüm | Değer |
|---|---|
| Çıkan lemma | 19.338 |
| **Kaybolan yüzey** | **14.790 (%13,27)** |
| Kesişimde G1 | %2,4120 → %0,6043 (düşüş %74,95) |

Kapsama kaybı `<UNK>`/`<PROPER_NOUN>`'u **artırır** ⇒ G2/G3'ü **bozar**;
H6'nın aksine kazanç **kapsama feda edilerek** alınır. Ayrıca aynı totoloji payı
burada da vardır. **Elendi.**

> **Payda disiplini (ilan edilmiş):** silme 14.790 yüzeyi çözülemez kıldığı için
> karşılaştırma **iki sözlükte de çözülebilen yüzeylerin KESİŞİMİNDE** yapılır —
> küçülen payda sahte düşüş üretirdi.

---

## 9. ÖLÇÜT: G1 BİR HATA ÖLÇÜSÜ DEĞİLDİR (en ağır bulgu)

`G1`'in öncülü *"kısa kök seçilmesi = hata"*dır. Yer-gerçeği etiketli 9 vakada
sınandı:

| Vaka | Gerçek durum | G1 ne diyor |
|---|---|---|
| `bunun` `onun` `ölmeden` `topraklar` `günümüzde` | **DOĞRU okunuyor** | **ihlal** ✗ |
| `ağacı` `aklı` `kanadı` `gönlü` | **YANLIŞ okunuyor** | ihlal ✓ |

**9/9 vakada G1 "ihlal" diyor. 5'i doğru okuma.** Yani G1 *"hata"*yı değil,
*"kısa kökle çözülmüş"*ü ölçer. **Duyarlılık %100, özgüllük %0.**

> **Sonuç:** "G1 düştü" ⇒ "iyileşti" **çıkarımı geçersizdir.** G1 bir *tutarlılık*
> göstergesi olarak kalır (Faz 4 kapısı olarak), fakat bir tasarımın
> **doğruluğunu kanıtlayamaz.** Faz 0 raporuna ek olarak işlendi (§11 orada).

---

## 10. Kontroller (fail-closed; düşen kontrol `rc=2` ile DURdurur)

| Kontrol | Beklenen | Sonuç |
|---|---|---|
| **K0** derleyici lexicon'la kurulu | 4 poz + 2 neg | **TEMİZ** |
| **DED** türemiş-görünümlü dedektörü | 3 poz + 2 neg | **TEMİZ** (18.244 bulundu, 10 sn) |
| Lexicon digest kapısı | `fe3005e5…` | TEMİZ |
| 3-kolon şema kapısı | tam 3 alan | TEMİZ |
| İki parquet de okundu | eksik dosya yok | TEMİZ (3.212 makale) |
| Filtreli-derleyici kontrolü | filtre kaybolmayı ölçer | TEMİZ |
| **H4 tavan artefaktı** | ilan DÜZELTİLDİ (aşağıda) | TEMİZ |

**İlan düzeltmesi (koşumdan ÖNCE, kayda geçti):** H4'ü ilk olarak *"bugün-doğru 5
vakanın **düzeltilmesi**"* diye ilan etmiştim. Ön-sondanma gösterdi ki derleyici
bu 5 vakayı **zaten 5/5 doğru** okuyor ⇒ ilan **vakum** olurdu (tavan artefaktı,
`[[tavan-artefakti-kapi-gecmez-kanitsizlik]]`). İlan **"korunuyor mu"** olarak
düzeltildi ve düzeltme betiğe + JSON'a (`ilan_duzeltmesi`) yazıldı.

---

## 11. Bu turda ölçülen KENDİ kusurlarım

| # | Kusur | Nasıl çıktı | Etki |
|---|---|---|---|
| 1 | `Vocabulary.load(...)` sınıf metodu sanıldı | `TypeError: missing 1 required positional argument` | r0e'deki `v=Vocabulary(); v.load(yol)` desenine düzeltildi |
| 2 | **Artımlı yazma yok** — H4/H5/ÖLÇÜT hesaplandı, `kaydet()` edilmedi | İlk koşum `KeyError` ile öldü, üç blok **kayboldu** | Her blok `kaydet(V)` ile kapatıldı (`[[uzun-olcum-artimli-yazmali]]`) |
| 3 | **H4 ilanı vakumdu** | derleyici zaten 5/5 doğru | Koşumdan önce düzeltildi, kayda geçti (§10) |
| 4 | Yeniden adlandırılan anahtar (`sonra_duzelen`→`sonra_korunan`) print'te kaldı | İlk koşum yazımdan önce öldü | §2 ile aynı sınıf; düzeltildi |
| 5 | Sondada tuple alanlarını karıştırdım (`r` = rz, string sandım) | `AttributeError: 'int' has no attribute 'startswith'` | Sonda yeniden yazıldı; §2 tablosu bu düzeltilmiş sondadan |
| 6 | **H8 hüküm eşiği yanlıştı** (`net != 0`) | `net = −1` yanıltıcı `KISMEN_AYIRT_EDICI` etiketi üretti | Eşik `net <= 0` yapıldı, **yeniden koşuldu**; düzeltme JSON'da |
| 7 | H6'nın ilk tasarımı (**silerek**) kapsama kaybını ölçmüyordu | 14.790 yüzey sessizce kaybolacaktı | H6 (silmeden) eklendi; H3 karşılaştırma olarak tutuldu |

---

## 12. Hüküm

1. **D1 bir skorlama problemi DEĞİLDİR.** Nedeni `roots.tsv` **alfabetik sırasıdır**:
   beraberlikte önek (kısa kök) daima önce gelir — **9/9**.
2. **Hiçbir el kuralı D1'i onarmıyor.** V1 (uzunluk, birincil) çürüdü
   (Faz 0); V1-varyantı (uzunluk, yalnız beraberlikte) **net −1**; H6
   (türemiş-görünümlüyü dışla) G1'i düşürüyor ama **düşüşün %85,75'i totolojik**
   ve 17.738 kararı kanıtsız değiştiriyor.
3. **Ölçüt ölü:** `G1` doğru okumaları da ihlal sayıyor (**5/9 yanlış pozitif**,
   özgüllük %0) ⇒ "G1 düştü" bir **kanıt değildir**.
4. **Kanıt/karar oranı yapısal olarak yetersiz:** elimizde **9 etiket** var; aday
   tasarımlar **18.000+ karar** değiştiriyor (~1/2000).
5. **Gereken şey yeni bir skor kuralı değil, bir `plausibility oracle`'dır.**
   Ölçüm, ayırt edicinin **uzunluk / türemişlik / dosya sırası olmadığını**
   gösterdi; kalan tek eksen **kullanım/olasılık**tır.

### Faz 2 için sonuç (öneri — operatör kararı bekler)

- **D1 maddesi Faz 2'den AYRILMALIDIR.** D1 çözülmeden de **Faz 1 (D2 sözlük
  eklemeleri)** ve **D3-mekanizma** yürüyebilir: bunlar ağırlıkla **0-yollu
  hücreleri doldurur** (`hakkı`, `adıyla`, `ağzı`, `hissi`) ve **yeni beraberlik
  üretmez**. ⚠️ Ama bu **varsayım değil, her ekleme için sınanacak** bir iddiadır:
  aynı 9-etiket kapısı her partiden sonra koşulur.
- **D1 için iki seçenek ölçülmeli (ikisi de bu turda YAPILMADI):**
  (a) **Oracle: A1 modelinin kendi olabilirliği** — aday analizleri morfem
  dizisi olarak `tokenizer.encode` ile kodlayıp A1'in log-olabilirliğiyle
  sıralamak. Elimizdeki eğitilmiş varlığı kullanır. ⚠️ Kirli: A1 külliyatı
  **mevcut kusurlu derleyiciyle** derlendi ⇒ yanlılık taşır, beyan edilmeli.
  (b) **Sıklık oracle'ı** — kök lemmasının külliyat sıklığı.
- **Faz 4 kapısı olarak `G1`, "belirsiz-pencere sayısı" ile BİRLİKTE raporlanır**
  (planın `needs_disambiguation` maddesi) — çünkü tek başına G1 yanıltıcıdır.
- **Faz 2 açılmadan önce `roots.tsv` satır sırasının sözleşmeye bağlanması**
  gerekir: bugün bir satır eklemek D1 kararını **sessizce** değiştirir.

---

## 13. Çürütme maddesi

| İddia | Onu çürütecek ölçüm |
|---|---|
| "D1 önek/sıra artefaktıdır" | Beraberlikte kazananın önek **olmadığı** ve doğru çıktığı yaygın bir vaka kümesi |
| "Uzunluk ekseni ayırt edici değil" | 9'dan fazla etiketle net > 0 veren bir uzunluk kuralı |
| "G1 hata ölçüsü değil" | G1 ihlali olmayan yüzeylerin hata oranının ihlal olanlardan **yüksek** çıkması |
| "H6'nın kazancı totolojik" | Aynı-kararla-flip sayısının (1.498) toplam düşüşün yarısından **az** çıkması |
| "17.738 yanal değişim kanıtsız" | Bu değişimlerin bir oracle'da **ölçülebilir** olabilirlik artışı sağlaması |
| "D3 eklemeleri D1'e dokunmaz" | Bir D3 eklemesinden sonra 9-etiket kapısının düşmesi |

---

## 14. Tam digest tablosu

| Artefakt | sha256 (tam) |
|---|---|
| `scratch/anka_r2_on_olcum.py` | `3fa9b71b16883a29c777bf4681cc77a15496ad5fce04f364ac460571b8f5936c` |
| `data/eval/anka_r2_on_olcum_2026-09-19.json` | `291ee703f889d98d3f9e88347e0af3a28de7d1c70bb3ab45c88a62352bfc466a` |
| `data/lexicon/roots.tsv` (donmuş) | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/rebuild/vocab_base_32852.json` (donmuş) | `7611b6a523a2bb524cc21aa9cc64235aa6a2ae99f87f46ac51cb2915e61164ad` |
| `data/eval/anka_r0_kapsam_2026-09-19.md` (+§11 ek) | `d3abb9620c477b50fb148ac692ba919781d862abb26720905203ea881e0270b9` |
| `data/eval/anka_r2_on_olcum_2026-09-19.md` | digest **bu belgeye yazılmaz** (kendini içeremez); **kapanış mesajında ve görev kaydında** tam olarak verilir |
| `tests/morphology_regression_100.json` | `cbe00f31c4dcc34ba2baed5317f1bf38b4e69b130c5271205c1dd5709dab18da` |

**Test durumu (19 Eyl 2026, bu turda koşuldu):** `venv/bin/pytest` → **213 passed,
1 failed**. Tek başarısızlık `tests/test_agent_gateway.py::…test_gateway_http_server_endpoints`
ve nedeni `self.socket.bind(...)` → `PermissionError: [Errno 1] Operation not permitted`
= **sandbox yasağı** (`[[sandbox-hides-mps-device]]` ailesi), bu değişikliğin ürünü
**değildir**. Faz 0'da aynı takım 212 passed / 2 failed vermişti; fark, aynı sınıftan
ikinci soket testinin bu koşumda geçmesidir.

**Donmuş hiçbir yola yazılmadı.** Bu fazda üretilen her şey `scratch/` ve
`data/eval/` altındadır (`data/eval/`, `data/**` kuralının tek istisnasıdır).

**Bayat/hatalı ara çıktı silinmedi:** ilk koşumun (H6b'siz, H8'siz) JSON'u aynı
yolun üzerine yazıldı; H8 eşik düzeltmesinden önceki koşumun çıktısı da öyle.
Üç koşumun **girdileri** (lexicon digest + örneklem) birebir aynıdır; yalnız
**ölçüm blokları** eklendi/düzeltildi ve her düzeltme ilgili alanda beyan edildi.
