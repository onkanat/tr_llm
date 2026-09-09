# Skill 3: Fonoloji Motoru (Phonology Engine) - Faz 3

Bu doküman, Türkçe Kristal-Vektörel Mimarisi'nin morfem birleşmeleri sırasında meydana gelen ses olaylarını (ünlü uyumu, ünsüz yumuşaması/sertleşmesi, ünlü düşmesi) yöneten **Fonoloji Motorunu** ve morfemlerden yüzey form üreten **Morfolojik Geri Çözücüyü (Decompiler)** detaylandırır.

---

## 🏛️ Sınıf Yapısı ve Karakter Grupları

Ses olayları, `PhonologyEngine` sınıfı (`src/compiler/phonology.py`) altında kural tabanlı ve deterministik olarak işletilir:

*   **`VOWELS`** (Ünlüler): `aeıioöuüAEIİOÖUÜ`
*   **`BACK_VOWELS`** (Kalın Ünlüler): `aıouAIOU`
*   **`FRONT_VOWELS`** (İnce Ünlüler): `eiöüEİÖÜ`
*   **`ROUNDED_VOWELS`** (Yuvarlak Ünlüler): `oöuüOÖUÜ`
*   **`UNROUNDED_VOWELS`** (Düz Ünlüler): `aeıiAEIİ`
*   **`UNVOICED_CONSONANTS`** (Sert Ünsüzler - Fıstıkçı Şahap): `fstkçşhpFSTKÇŞHP`

---

## 📐 Ses Değişim Kuralları ve Algoritmalar

### 1. Kök Mutasyonu (`mutate_stem`)
Kökün sonuna ünlü ile başlayan bir ek geldiğinde kökte meydana gelen morfolojik değişimleri kontrol eder:
*   **Hece Düşmesi (`VOWEL_DROP`):** Kökün niteliklerinde `VOWEL_DROP` varsa ve kök iki heceliyse sondan bir önceki sesli harf düşürülür (örn. *ağız* $\rightarrow$ *ağz-*).
*   **Ünsüz Yumuşaması (`VOICING`):** Kökün niteliklerinde `VOICING` varsa ve son harf sert ünsüz ise yumuşatılır:
    *   `p` $\rightarrow$ `b`
    *   `ç` $\rightarrow$ `c`
    *   `t` $\rightarrow$ `d`
    *   `k` $\rightarrow$ `ğ` (Eğer `k` harfinden önce `n` varsa `g` harfine dönüşür; örn. *renk* $\rightarrow$ *reng-*).

### 2. Şablon Çözümleme (`resolve_affix`)
Morfotaktik graftan gelen soyut ek şablonlarını kökün fonetik durumuna göre somut yüzey biçimine dönüştürür.
*   **Parantezli Yardımcı Sesler:**
    *   **Kaynaştırma Harfleri `(y)`, `(s)`, `(n)`, `(ş)`:** Kök ünlüyle bitiyorsa eklenir (örn. *araba-(y)A* $\rightarrow$ *arabaya*).
    *   **Yardımcı Ünlüler `(I)`, `(A)`:** Kök ünsüzle bitiyorsa eklenir (örn. *gel-(I)m* $\rightarrow$ *gelim*).
*   **Büyük Ünlü Uyumu (`A` ve `I` placeholders):**
    *   `A` (2'li uyum): Son ünlü kalınsa `a`, inceyse `e` olur.
    *   `I` (4'lü uyum): Kalın-düzse `ı`, kalın-yuvarlaksa `u`, ince-düzse `i`, ince-yuvarlaksa `ü` olur (`_resolve_I`).
*   **Ünsüz Sertleşmesi (`D` ve `C` placeholders):**
    *   `D`: Kök sert ünsüzle bitiyorsa `t`, aksi halde `d` olur (örn. *git-DI* $\rightarrow$ *gitti*).
    *   `C`: Kök sert ünsüzle bitiyorsa `ç`, aksi halde `c` olur (örn. *kitap-CI* $\rightarrow$ *kitapçı*).

---

## 🔄 Morfolojik Geri Çözücü (`MorphemeDecompiler`)

Dil modelinin ürettiği morfem ID dizilerini (örn. `["git", "TENSE_FUT", "PERSON_1SG"]`) doğru Türkçe yüzey biçimine (`"gideceğim"`) dönüştürmek için ters fonoloji sentezi (`src/compiler/decompiler.py`) işletilir:
1.  **Kök Belirleme:** İlk token kök leksikonundan alınır (`git`).
2.  **Ek Zincirleme:** Ardışık her bir morfem ID'si (`TENSE_FUT`), graf üzerindeki fonetik şablonuyla (`(y)AcAk`) eşleştirilir.
3.  **Fonetik Sentez:** `PhonologyEngine.resolve_affix` çağrılarak kök ve ekler ses uyumu kurallarınca birleştirilir.
4.  **Meta Etiket Koruması:** `<ARA>`, `</ARA>`, `<BELGE>`, `</BELGE>`, `<INSTRUCTION>` gibi özel XML meta etiketleri ses motoruna sokulmadan ham olarak korunur.

---

## 🧪 Doğrulama ve Test Kapsamı
Tüm fonoloji kuralları ve istisnaları `tests/test_phonology_golden.py` ve ilgili test takımları altında **67/67 (%100)** birim test ile doğrulanmıştır.
