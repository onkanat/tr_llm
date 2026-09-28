# İLAN — T-0161 Soru Filtresi Şartnamesi: Router Yeniden-Eğitim Koşulu

**Damga (BETİKTEN, koşum-ÖNCESİ):** `2026-09-28T17:21:42Z`  
**Meşruiyet:** T-0156 raporunda gözlemlenen `%94,33` oranındaki soru işareti (`?`) varlığının şartnameleştirilmesi, havuz envanterinin çözümlenmesi ve router eğitim koşulunun ampirik doğrulanması.  
**Operatör Kararları ve Donmuş-Desen Koruması:**
- `data/anka_router.pt` yazımı için **ayrı operatör onayı şarttır**; bu görevde donmuş `data/anka_router.pt` üzerine **YAZILMAZ** (in-memory / eval doğrulaması yapılır, dosya korunur).
- writes: `["scripts/dogrulama_t0161_soru_filtresi.py", "data/eval/", ".agent-bus/notes/"]` sınırlarına tam uyum.
- `data/pedagogy_canonical/**` (DONMUŞ), `data/pedagogy/**`, `scripts/dogrulama_t0156_pedagogy_daraltma.py` DOKUNULMAZ.
- Canlı Qdrant `192.168.1.9:6333`'a **0 istek** (VectorMemory KULLANILMAZ).

---

## 1. Kök-Neden ve Havuz Analizi (İki Hipotez / İki Kol)

T-0156'da daraltılmış pedagogy havuzundaki metinlerin `%94,33`'ünün soru cümlesi olduğu gözlemlenmiştir. Şartname metnindeki "soru filtresi" ifadesi iki olası matematiksel yoruma sahiptir:

1. **Kol-A (Literal Yorum — '?' İçerenleri Eleme):**
   - Kural: `not "?" in text`
   - Pedagogy 4-dosya toplam kayıt: 7.540, tekil: 7.048.
   - Soru işareti içermeyen tekil havuz boyutu: **395**.
   - Kısıt: Router eğitimi sınıf başına 500 train + 100 val = **600 tekil örnek** gerektirir.
   - Sonuç: `395 < 600` olduğu için Kol-A altında 600 tekil örnek seçimi matematiksel olarak imkansızdır (yetersiz havuz).

2. **Kol-B (İşlevsel Soru Filtresi — '?' İçerenleri Tutma, Gürültüyü Eleme):**
   - Kural: `"?" in text`
   - Soru işareti içeren tekil havuz boyutu: **6.653**.
   - Kısıt: `6.653 >= 600` koşulu fazlasıyla sağlanır.
   - Sonuç: 600 tekil örnek (`500 train, 100 val`) SEED-42 ile seçilir; seçilen örneklerde `pedagogy_secilen_soru_isareti_orani == 1.0000` olur.

Betiğimiz T-0020 emsali gereğince her iki kolu da nesnel olarak ölçer, Kol-A'nın yetersizliğini belgeler, Kol-B üzerinden model eğitim ve doğrulama koşullarını sarsılmaz biçimde kanıtlar.

---

## 2. Kaynak-Çıpa ve Değişmeyen Sabitler

- `data/anka_base_v2.pt`: `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`
- `data/rebuild/vocab_anka_r1_33114.json`: `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`
- `data/lexicon/roots.tsv`: `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`
- `data/anka_router.pt` (T-0156 çıktısı): `44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a`
- SEED-42 · sınıf-başı 500 train + 100 val · `_metinler` (yalnız `input`, boşsa `instruction`) · sorted(set) + rng.shuffle + ilk 600.
- Model: KristalLM 6/6/768 block 4096 (MPS özellik çıkarımı) + CuriosityEngine(768, 768, tau=2.5) fresh-init seed-42.
- Eğitim: CPU AdamW lr=1e-3, batch=32, 3 epoch CE, generator seed=42+epoch.

---

## 3. Kapılar (Hüküm BETİKTEN; 8/8 şart; rc ∈ {0, 2})

- **K1 KAYNAK-DİGEST & DEVİR KORUMASI:** Taban model, sözlük, leksikon ve T-0156 router SHA'ları birebir eşleşmeli; 4 dosya pedagogy kaynağı doğrulanmalı.
- **K2 BASELINE DEĞERİ:** Fresh-init seed-42 router'ın Kol-B 300-val kümesindeki top-1 doğruluğu kayıt altına alınmalı (rastgele taban 0,3333).
- **K3 EĞİTİM BAŞARISI:** Val top-1 >= 0,75 VE >= baseline + 0,15 VE kayıp azalan VE her sınıf train doğru >= 490/500.
- **K4 DETERMİNİZM:** İki bağımsız eğitim koşumunun `state_dict` SHA-256 özeti bit-özdeş olmalı.
- **K5 VERİ-ENVANTERİ & FİLTRE KANITI:**
  - 4 dosya havuz envanteri (Kol-A: 395 tekil; Kol-B: 6.653 tekil).
  - Seçilen 600 pedagogy örneğinin parenting üyeliği == 0.
  - Grammar_core havuz kesişimi == 0.
  - Val ve train kesişimi == 0, legal örnek == 0.
  - Kol-B seçilen pedagogy örneklerinde soru işareti oranı == 1.0000.
- **K6 ŞEMA & DONMUŞ-MODEL KORUMASI:** 9 kanonik anahtar eksiksiz; `data/anka_router.pt` üzerine **yazılmaz** (donmuş dosya korunur); canlı Qdrant'a 0 istek.
- **K7 KARIŞIM MATRİSİ:** Train ve val için 3x3 karışım matrisleri, per-class doğruluklar ve satır toplamları raporlanmalı.
- **K8 ENTEGRASYON & DURUM BEYANI:** Eğitilen modelin `state_dict`'i runtime `TriModalRouter` ile kusursuz yüklenmeli ve doğrulanmalı.
