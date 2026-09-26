# G4: T-0128 BİRLEŞİM KOLU TUTARSIZLIK BEDELİ ALTKÜME TEŞHİS RAPORU

**Tarih / Saat (UTC):** 2026-09-26T07:22:00Z  
**Görev:** T-0130 / G4  
**Hedef:** T-0128 Kol-BİRLEŞİM deneyindeki %11,0 (11/100) tutarsızlık oranının kaynak altkümesini ayrıştırmak.  
**İlan Tarihi / Saat:** 2026-09-26T07:18:00Z (mtime: `1790407153.1329234`)  
**Statü / Amaç:** Tanısal ölçüm — sonucu G3a ceket tasarım dokümanına doğrudan girdi sağlar.

---

## 1. Yönetici Özeti ve Teşhis Hükmü

* **HÜKÜM:** **`TUTARSIZLIK_TAMAMEN_ÇOĞALTMA_BÖLGESİNDE`**
* **Hüküm Açıklaması:** Tutarsızlıkların tamamı (11/11) çoğaltılmış satırlar bölgesindedir (oran: %12.22, CI: [6.96, 20.57]). Paraphrase edilmiş ne_ise_yarar bölgesinde ise tutarsızlık 0'dır (0/10, %0.00, CI: [0.00, 27.75]). Ancak n=10 küçük olduğundan Wilson CI üst sınırı (%27.75) çoğaltma bölgesi aralığıyla matematiksel olarak örtüşmektedir.
* **G3a Ceket Tasarım Dokümanı İçin Çıkarım:**
  Tutarsızlığın (cümle sonu yüklem eksikliği / cümle kesilmesi) kaynağı çeşitlendirilmiş `ne_ise_yarar` cevapları **değildir**. Sentetik paraphrase edilen cevaplar modelde tutarsızlık oluşturmamıştır (tutarsızlık %0,0). Aksine, tutarsızlık modelin 3 kez tekrar eden çoğaltılmış satırlarında (özellikle `"nedir"` soru kalıbında %38,89 oranında) yoğunlaşmıştır. Dolayısıyla G3a ceket tasarımında aşırı çoğaltma (oversampling) yerine cevap sonlandırma kalıplarının dengelenmesi gerekmektedir.

---

## 2. Altküme Dağılımı ve Tutarsızlık Tablosu

| Altküme | Kapsam / Soru Tipi | n (Örnek) | Tutarsız (k) | Tutarsızlık Oranı (%) | %95 Wilson Güven Aralığı (CI) |
|---|---|---|---|---|---|
| **Altküme A (Paraphrase)** | `ne_ise_yarar` | 10 | 0 | **%0.00** | [0.00, 27.75] |
| **Altküme B (Çoğaltma)** | `diger` / `nedir` / `nasil` | 90 | 11 | **%12.22** | [6.96, 20.57] |
| ↳ *Kırılım: nedir* | Soru kalıbı: `"nedir"` | 18 | 7 | %38.89 | [20.30, 61.38] |
| ↳ *Kırılım: nasil* | Soru kalıbı: `"nasıl"` | 16 | 1 | %6.25 | [1.11, 28.33] |
| ↳ *Kırılım: diger* | Diğer marangozluk soruları | 56 | 3 | %5.36 | [1.84, 14.61] |
| **Altküme C (Belirsiz)** | Eşleşmeyen | 0 | 0 | %0.00 | [0.00, 0.00] |
| **TOPLAM** | *Tüm Örneklem* | **100** | **11** | **%11.00** | **[6.25, 18.63]** |

### İstatistiksel Anlamlılık ve Güven Aralığı Analizi:
- Paraphrase bölgesinde 10 örnekten 0'ı tutarsızdır (%0,0). Wilson CI: `[0.00, 27.75]`.
- Çoğaltma bölgesinde 90 örnekten 11'i tutarsızdır (%12,22). Wilson CI: `[6.96, 20.57]`.
- $n=10$ örneklem kısıtından ötürü Wilson CI aralıkları matematiksel olarak örtüşmektedir (`ci_ortusuyor: True`). Ancak gözlemlenen tutarsızlıkların **11'de 11'i (%100'ü)** çoğaltılmış satırlar bölgesinde ortaya çıkmıştır.
- Özellikle `"nedir"` kalıbındaki 18 sorunun 7'si (%38,89, Wilson: `[20.31, 61.38]`) tutarsız sonuçlanarak modelin en zayıf noktasını oluşturmuştur.

---

## 3. Tutarsız Olan 11 Örneğin Tam Listesi (Örnekleme YOK, Tam Envanter)

| Örnek No | Soru Kalıbı | İstem (Instruction) | Hata Sebebi | Üretilen Morfem Özeti |
|---|---|---|---|---|
| 8 | `nedir` | Geleneksel ahşap zanaatı sorusu: Marangozlukta lif doyum noktası nedir ve n… | `not_yuklem` | usta Cevabı : Kerestenin ahşap zamanla çeker , Çat PLURAL veya eğril TENSE_AORIST . çünkü … |
| 14 | `nedir` | Ahşap atölyesi güvenlik ve imalat rehberi: Ahşap atölyesinde gönye aletinin… | `not_yuklem` | teknik çözüm : teknik çözüm : gönye burun birleşim POSS_3SG : gönye burun birleşim POSS_3S… |
| 25 | `diger` | Masif mobilya standartlarına göre açıkla: İskarpelanın arkası (düz yüzeyi) … | `not_yuklem` | usta Tavsiyesi : Ahşabın lif POSS_3PL CASE_ACC_N kopar INF_mAk için sürü DERIV_lAn ince ma… |
| 28 | `nedir` | Atölye pratiği: Ahşap atölyesinde eğe aletinin doğru kullanım tekniği nedir… | `not_yuklem` | teknik çözüm : teknik çözüm : rende tığ POSS_3SG ters lif yön POSS_2SG CASE_LOC yapıl TENS… |
| 29 | `nasil` | Atölye pratiği: Marangozlukta kamalı zıvana birleştirmesi nasıl yapılır ve … | `not_yuklem` | teknik çözüm : usta Tavsiyesi : zıvana dil POSS_2SG CASE_GEN yanak POSS_3PL CASE_DAT_N ayn… |
| 54 | `nedir` | Ahşap işleme teknik danışmanlığı: Ahşap atölyesinde rende aletinin doğru ku… | `not_yuklem` | teknik çözüm : teknik çözüm : teknik çözüm : teknik çözüm : teknik çözüm : Kerestenin boy … |
| 58 | `diger` | Bir marangoz çırağının sorusuna cevap ver: Usta marangoz olarak açıkla: meş… | `not_yuklem` | teknik çözüm : teknik çözüm : ardıç ağa DERIV_CI doğal <UNK> ve yağ barındır PART_DIk POSS… |
| 71 | `nedir` | Marangoz ustasına danış: Marangozlukta kılcal nem testi nedir ve neden haya… | `not_yuklem` | teknik çözüm : eğer ahşap lif POSS_3PL boyuna göre enin CASE_DAT lif yön POSS_2SG CASE_LOC… |
| 72 | `diger` | Marangozluk alan uzmanlığı: Usta marangoz olarak çözüm üret: İskarpelanın a… | `not_yuklem` | usta marangoz : teknik çözüm : lamba kavela DERIV_lI ahşap mobilya PLURAL CASE_LOC ideal k… |
| 73 | `nedir` | Ahşap işleme teknik danışmanlığı: Ahşap atölyesinde tokmak aletinin doğru k… | `not_yuklem` | teknik çözüm : teknik çözüm : teknik çözüm : teknik çözüm : freze devir POSS_3SG uygulama … |
| 94 | `nedir` | Marangoz ustasına danış: Ahşap atölyesinde eğe aletinin doğru kullanım tekn… | `not_yuklem` | teknik çözüm : teknik çözüm : rende tığ POSS_3SG ters lif yön POSS_2SG CASE_LOC yapıl TENS… |

---

## 4. Tam Digest Tablosu (Önek Karşılaştırması YOK)

| Dosya / Artefakt | SHA-256 (tam digest) | Açıklama |
|---|---|---|
| `data/eval/anka_r17_heldout_2026-09-20.jsonl` | `c397eb08218937ea000cc6120f42a8783e1dcab7948c70b33537af74b50d096e` | Held-out soru bankası |
| `scratch/t0128/kanonik_birlesim.json` | `4e7d0e6aa81fe818155a306a771b088216542f44ea8d1a6ab9a81d16654de8ec` | T-0128 değerlendirme ve üretim çıktısı |
| `scratch/t0128/kulliyat_birlesim.jsonl` | `b8d3aa404a1df29c7285119878c8391ffbc7dd959a90b35dce3d5c10299094ae` | 14.109 satırlık birleşim eğitim külliyatı |
| `scratch/t0129/g4_teshis/g4_teshis_olcum.py` | `d046c241d22efc23dd0f7b239bfdfb030153cd0725e9ac6f484f20c77331a35e` | Bu analiz ve teşhis betiği |
| `scratch/t0129/g4_teshis/g4_sonuc.json` | `29637787e1f83e24b61414c217fc5eadd62a8f5178028c94e5e73f54155552ff` | Makine-okunabilir teşhis çıktısı |

---

## 5. Beyanlı Sınırlar ve Yöntem İlkeleri
1. **Model Gereksinimi:** Model çıkarımı yapılmamıştır; `scratch/t0128/kanonik_birlesim.json` içindeki deterministik `ham_gm` morfem dizileri offline olarak ayrıştırılmıştır.
2. **Kanonik Dokunma Yasağı:** `src/`, `scripts/` ve `train.py` dosyalarına kesinlikle dokunulmamıştır.
3. **Kapsam:** Örnekleme yapılmamış, heldout içindeki tüm 11 tutarsız örnek tek tek incelenmiştir.
