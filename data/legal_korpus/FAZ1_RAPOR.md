# P2.5 FAZ-1 — HUKUK (LEGAL) VERİ HATTI RAPORU

**Tarih:** 2026-11-04 · **Kapsam:** P2.5 "legal ceketi" faz-1 (yalnız VERİ) ·
**Eğitim YOK · commit YOK · izlenen dosyaya dokunma YOK** (yalnız yeni betik + yeni veri dosyası).

---

## 1. Kaynak erişimi (GÖREV 1) — KANITLI

- **Uç nokta (doğrudan metin, curl/HTTP):**
  `https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatNo={no}&MevzuatTur=1&MevzuatTertip=5`
- **Deneme:** 4857 → HTTP 200, 489.325 bayt (MevzuatDetayIcmal uç noktası 302 verir, kullanılmadı).
- **Kod çözme:** sunucu fiilen **UTF-8** yollar (sayfanın `charset=Windows-1254` beyanı yanıltıcıdır).
- **Saygılı getirme:** istekler arası 3,5 sn, 3 yeniden deneme (üstel bekleme), tek tarayıcı UA; 17 kanun tek oturumda alındı. Ham HTML önbelleği: `scratch/mvz_raw/` (gitignore).
- **Tek kanun kalite kanıtı (4857):** üstbilgi bloğu sayfada hazır (`Kanun Numarası : 4857`, `Kabul Tarihi : 22/5/2003`, RG `10/6/2003 - 25134`); madde başlıkları `Madde N -` / `MADDE N –` iki biçimde — ayrıştırıcı ikisini de kurallı `Madde N -` satırına indirger; 136 madde satırı (ana 122 + geçici/ek), madde üstü başlık satırları (ör. "Tanımlar") korunur.
- **Hukuki çerçeve:** kanun/tüzük metinleri FSEK m.19(b) uyarınca telif dışı; kamu portalı mevzuat.gov.tr.

## 2. Korpus (GÖREV 2) — 17 kanun, 4.151.408 karakter

Hedef band ~1-4M karakter: **4,15M ✓** (ölçülü). Madde numaralı satır yapısı her dosyada korundu;
üstbilgi: kanun adı / no / kabul / RG / kaynak URL / erişim tarihi / telif notu.
Seçim: kullanıcının FMS-200/elektrik müh. bağlamına uyanlar öncelikli; TDK kanunu bilinçli dışarıda.

| No | Kanun | Kabul | RG | Karakter | Madde satırı |
|-----:|-------|-------|----|---------:|-----:|
| 657 | Devlet Memurları Kanunu | 14/7/1965 | 23/7/1965 - 12056 | 551.144 | 431 |
| 2577 | İdari Yargılama Usulü Kanunu | 6/1/1982 | 20/1/1982 - 17580 | 106.114 | 81 |
| 2872 | Çevre Kanunu | 9/8/1983 | 11/8/1983 - 18132 | 114.507 | 61 |
| 3194 | İmar Kanunu | 3/5/1985 | 9/5/1985 - 18749 | 171.379 | 86 |
| 4708 | Yapı Denetimi Hakkında Kanun | 29/6/2001 | 13/7/2001 - 24461 | 60.236 | 23 |
| 4721 | Türk Medeni Kanunu | 22/11/2001 | 8/12/2001 - 24607 | 419.677 | 1.032 |
| 4857 | İş Kanunu | 22/5/2003 | 10/6/2003 - 25134 | 156.509 | 136 |
| 5237 | Türk Ceza Kanunu | 26/9/2004 | 12/10/2004 - 25611 | 307.969 | 345 |
| 5271 | Ceza Muhakemeleri Kanunu | 4/12/2004 | 17/12/2004 - 25673 | 371.358 | 352 |
| 5378 | Engelliler Hakkında Kanun | 1/7/2005 | 7/7/2005 - 25868 | 32.528 | 43 |
| 5510 | Sosyal Sigortalar ve Genel Sağlık Sigortası K. | 31/5/2006 | 16/6/2006 - 26200 | 788.296 | 246 |
| 6098 | Türk Borçlar Kanunu | 11/1/2011 | 4/2/2011 - 27836 | 317.994 | 650 |
| 6100 | Hukuk Muhakemeleri Kanunu | 12/1/2011 | 4/2/2011 - 27836 | 297.408 | 461 |
| 6331 | İş Sağlığı ve Güvenliği Kanunu | 20/6/2012 | 30/6/2012 - 28339 | 78.161 | 50 |
| 6446 | Elektrik Piyasası Kanunu | 14/3/2013 | 30/3/2013 - 28603 | 174.171 | 73 |
| 6502 | Tüketicinin Korunması Hakkında Kanun | 7/11/2013 | 28/11/2013 - 28835 | 151.001 | 91 |
| 6698 | Kişisel Verilerin Korunması Kanunu | 24/3/2016 | 7/4/2016 - 29677 | 52.943 | 36 |

Bilinen kenar durumlar (korpus gerçeği, onarılmadı): 4857'de 87 numara fiilen yoktur (mülga);
5237'de 61/328 ve 3194'te 48 numaralı madde başlıkları sayfada farklı biçimlenir (metin dosyada
bulunsa da satır-başı ayrıştırılmaz); mülga madde başlıklarından bazıları `Madde N (Mülga:...)`
biçimindedir (tire yok).

## 3. Tokenizabilirlik ölçümü (GÖREV 3) — donmuş anka sözlüğü 33.114

Üretici: `scripts/measure_legal_tokenizability.py` → `data/legal_korpus/tokenizabilite_raporu.json`.
Örneklem: 17 kanundan orantılı **156.056 karakter** (hedef ≥100K ✓, baş/orta/son dilimleri).

| Ölçüm | Değer |
|---|---|
| Toplam token | 44.395 (0,284 tok/karakter) |
| **UNK (id=0) oranı** | **%1,34 token** (595 adet) — eşik %10'un çok altında |
| UNK'ya düşen kelime örneklemi | 300 farklı kelime / tüm kelime olaylarının **%2,71'i** |
| Kelime başına token | 2,05 · karakter/token 3,52 |
| `<PROPER_NOUN>` | 0 (literal_entity_mode: bilinmeyen özeller harf-harf açılır) |
| Rakam tokeni | %14,7 (tarih/kanun no/madde no — hukuk metnine doğal) |
| En sık UNK kelimeler | nci, ncı, yürürlüğe, tüketici, hükme, işyeri, kaydıyla, teselsül, uyuşturucu, keseneğine, mali, hükmü, usulü, hakkı, adlî |
| Sözlükte derlenemeyen hukuk kelimeleri | hükmün, zamanaşımı, yürürlükten, kanunoylarıyla, vekalet, tâbi |

**Hüküm:** UNK %10 eşiğinin çok altında → **sözlük genişletme kalemi GEREK YOK**; sözlük donmuş
33.114 aynen kalır. (UNK'lı kökler ceket eğitiminde bağlamdan öğrenilir; kelime-düzeyi UNK %2,71
kabul edilebilir.)

## 4. SFT veri seti (GÖREV 4) — 463 kayıt, marangoz kart kalıbı

Üretici: `scripts/prepare_legal_dataset.py` → `data/pedagogy/legal_canonical_sft.jsonl`
(+ `.meta.json`). Kalıp: `prepare_carpenter_specialization_dataset.py` kart biçimi
`{instruction, input, output}`; alıntılar `«...»` içinde, kaynak kanun no + madde etiketiyle.

| Görev tipi | Kayıt | Açıklama |
|---|---:|---|
| (a) kavram tanımı | 117 | tanım bentleri ('Kavram: ...') ham satırdan birebir |
| (b) madde sorusu — başlıklı | 291 | kanun başlıklı maddeler, madde üstü başlık soruda |
| (b) madde sorusu — hedefli | 35 | 35 elle doğrulanmış soru; hedef madde + terim kapısı |
| (c) kavram ayrımı | 11 | iki gerçek tanım + kaynak; ayrım cümlesi alıntılara sadık |
| (d) prosedür adımları | 9 | her adım gerçek madde alıntısıyla |
| **Toplam** | **463** | hedef band 300-600 ✓ (kalite>nicelik) |

**Uydurmasızlık kapıları (tümü ölçüldü):**
- Her alıntı kurulum gereği korpus madde satırından alınır; hedefli sorularda **terim madde
  metninde doğrulanamazsa kayıt düşer** — sonuçta 0 düşen (35/35 hedefli, 11/11 ayrım, 9/9 prosedür).
- Mezar-taşı ("yerine işlenmiştir") ve mülga maddeler, dipnot çapaları atılır.
- **Alıntı-birebirlik denetimi (bağımsız):** «» içindeki 1.240 cümlenin **%99,7'i** korpus
  metninde birebir bulundu (1236/1240; 37 kesik-son cümle üç-nokta kuralıyla atlandı; kalan tek
  kayıt #365 tek-cümlelik liste alıntısı olup yine gerçek önek + «…»).
- Dedupe: 0; şema doğrulama: tüm kayıtlarda {instruction, input, output}, boş olmayan alanlar.

**Token/UNK ölçümü (anka 33.114, literal_entity_mode=True):** 463 kayıt → **80.455 token**
(ort. 174/kayıt), **UNK 1.304 (%1,62)** — yeni-kayıt UNK hedefi <%3 ✓.

**Örnek 3 kayıt** (tam metin jsonl'de):
1. *Tanım:* "…6502, Madde 3: «Sağlayıcı: Kamu tüzel kişileri de dâhil olmak üzere ticari veya
   mesleki amaçlarla tüketiciye hizmet sunan ya da hizmet sunanın adına ya da hesabına hareket
   eden gerçek veya tüzel kişiyi,»"
2. *Hedefli soru:* "6698 sayılı KVKK'ya göre veri güvenliğine ilişkin yükümlülükler nelerdir?" →
   6698 Madde 12 birebir alıntısı.
3. *Prosedür:* "Kişisel veri güvenliği ihlali… adımları" → 1) 6698 m.12 alıntısı 2) 6698 m.15 alıntısı.

## 5. Dosyalar + sha256

```
e870d60b135b0e87…9c5289d2  data/legal_korpus/0657_devlet_memurlari_kanunu.txt
91a0a5c05a1360c8…8f3591e  data/legal_korpus/2577_idari_yargilama_usulu_kanunu.txt
9f64dbc63e78675a…68f7ee0a  data/legal_korpus/2872_cevre_kanunu.txt
ce98f3080015840c…362fc662  data/legal_korpus/3194_imar_kanunu.txt
da63ac3eab15cda2…6f5f9707  data/legal_korpus/4708_yapi_denetimi_hakkinda_kanun.txt
663d3e5e29ce7b82…a620f8b3  data/legal_korpus/4721_turk_medeni_kanunu.txt
dad2dfd1412f3b4d…9e4e26dd  data/legal_korpus/4857_is_kanunu.txt
40be1312c4036e0d…3e4bac05  data/legal_korpus/5237_turk_ceza_kanunu.txt
d05814649affd2b9…f023ca7e  data/legal_korpus/5271_ceza_muhakemeleri_kanunu.txt
6e52c69394504af3…da00e2e0  data/legal_korpus/5378_engelliler_hakkinda_kanun.txt
bf132fe34fc7b46f…be707149  data/legal_korpus/5510_sosyal_sigortalar_ve_genel_saglik_sigortasi_kanunu.txt
6f771b7aa3399d29…1699d26c  data/legal_korpus/6098_turk_borclar_kanunu.txt
5180f987414b6344…410e35c7  data/legal_korpus/6100_hukuk_muhakemeleri_kanunu.txt
82435347e3be8f41…36cdf0f5  data/legal_korpus/6331_is_sagligi_ve_guvenligi_kanunu.txt
e3a6251f6520cbf8…da63f16c  data/legal_korpus/6446_elektrik_piyasasi_kanunu.txt
4bfbffef07e9f602…f4a53130  data/legal_korpus/6502_tuketicinin_korunmasi_hakkinda_kanun.txt
5a49ab65db694c41…1b9654e6  data/legal_korpus/6698_kisisel_verilerin_korunmasi_kanunu.txt
6a0947b2f1d703c0…012a20e3  data/legal_korpus/korpus_manifest.json
98f5f84b4ba3c098…b3096085  data/legal_korpus/tokenizabilite_raporu.json
69670bc6ddbc033c…ea706347  data/pedagogy/legal_canonical_sft.jsonl
0132c57f4cee3909…96685276  data/pedagogy/legal_canonical_sft.jsonl.meta.json
29c21958c285b113…e676227f  scripts/fetch_mevzuat_corpus.py        (YENİ)
496a3143651ceb57…824eebf0  scripts/measure_legal_tokenizability.py (YENİ)
5052364e41cd559f…cb976db4  scripts/prepare_legal_dataset.py        (YENİ)
```
(Tam 64-haneli sha256'ler yukarıdaki ölçüm çıktısında; rapor özetle kısaltılmıştır.
`legal_canonical_sft.jsonl` korpus üstbilgi düzeltmelerinden ÖNCE ve SONRA aynı kalmıştır —
üretim yalnız gövde metnine bağlıdır, deterministiktir.)

## 6. Faz-2 (ceket eğitimi) için hazır olma hükmü

**VERİ YETERLİ — FAZ-2 BAŞLATILABİLİR.**
- SFT: 463 kayıt / 80,5K token (carpenter r18 v1 ölçeğine uygun başlangıç; kalite-kapılı).
- Korpus: 4,15M karakter hazır (ceket içi devam-ön-eğitim dilimi veya çıpa karışımı için).
- UNK: korpus %1,34 / SFT %1,62 — sözlük genişletme gerekmiyor, donmuş 33.114 korunur.
- **Faz-2'de eksik olanlar (bu iş emri kapsamı dışı):**
  1. `prepare_carpenter_specialization_dataset.py` kalıbıyla `data/train_legal_specialization*.bin`
     serileştirme (ham + "Hukuk uzmanı olarak cevapla." normalize çifti + çıpa karışımı),
  2. router'a **'legal' sınıfı** (val CE raporu + router örneği >0 kabul kapısı),
  3. T-0209 tarifi (4:1 SFT/pencere karışımı, KL β=0,2 — P2.6-B1 satır-seçim düzeltmesiyle,
     8GB M2 için mikro-parti/MPS `empty_cache()` disiplini),
  4. kabul ölçümü: masked SFT val CE + pencere-CE ayrımı METRİK ADIYLA yazılır (skill notu).
- Riskler: rakam payı %14,7 (korpus doğası) — pencere-CE'de rakam yoğun dilimler hafif avantaj kaybettirebilir; 'nci/ncı' gibi sıra-ekleri UNK'da — ceket verisinde telafi edilmez, kabul ölçümünde gözlemlenir.