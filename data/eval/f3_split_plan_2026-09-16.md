# F3 Parçalama Planı — Tabana yetkinlik tamamlama geçişi

**Damga:** 2026-09-16T15:05Z · **Yazar:** claude (danışman) · **Durum:** **ONAYLANDI (16 Eyl 15:03Z)** — dört karar §6'da; F3-0/F3-1 açılıyor, eğitim parçaları (F3-2/F3-3) yedek doğrulanmadan AÇILMAZ
**Kaynak:** `data/eval/regeneration_plan_2026-09-15.json` F3 bölümü + 15 Eyl'den beri ölçülen değişiklikler

---

## 1. F3 nedir (planın kendi tanımı)
> **F3 — TABANA yetkinlik tamamlama geçişi [ÜRETİM (.pt) — eğitim koşusu]**
> *why_base_not_jacket:* "Noktalama dil becerisidir; dil becerisi tabanın işidir. Ceket yalnız kendi dar alanında noktalama öğrenebilir."
> *adımlar:* (i) ÖNCE YEDEK — mevcut taban checkpoint'leri gitignored, ataları kısmen kayıp → geri dönüşsüz; (ii) **devam eğitimi** (sıfırdan değil): mevcut taban + F2'nin yeniden derlenmiş `.bin`'leri; (iii) sözlük: tabanın gerçek sözlüğü, sessiz kırpma kapalı; (iv) `device == mps` kabul kriteri (sandbox MPS'i gizler).
> *gate_G3:* "Noktalama üretim testi (projede İLK KEZ): model `.` `,` `:` üretebiliyor mu? Taban çizgi **0** — çünkü eğitimde hiç görmedi. Ayrıca teacher-forcing kaybı."
> *rollback:* "Yedekten geri dön; yeni checkpoint **AYRI isimle** yazılır, mevcutların üzerine YAZILMAZ."

## 2. 15 Eyl'den beri değişenler (plan güncellenmeli)
| Konu | Plandaki hâli | Bugünkü ölçülmüş hâli |
|---|---|---|
| Sözlük uç noktası | "tabanın gerçek sözlüğü (32.816)" | **32.852** — kullanıcı kararı (16 Eyl); `train.py` modeli artık **yüklenen sözlükten** kuruyor ve checkpoint `resize_state_dict` ile +36 satır padding'leniyor (`[SOZLESME_UYARI]` ile görünür) |
| Eğitim hattı | eski ad okunuyordu | `train_scientific_sft.py`, `retrain_clean_models.py`, `run_goal_pipeline.py`, `evaluate_sft_benchmarks.py` → **`_v2`** (T-0046) |
| F2 ürünleri | yoktu | `train_chat_balanced.bin` (`443f93d3…`, noktalama 180.336) ve `train_balanced_sft_v2.bin` (`53b8406e…`, noktalama 180.336) **doğrulandı** |
| Donmuş yazım | korumasız | Kapı dizisi kuruldu (T-0047/T-0048/T-0049): eğitim artık `--allow-frozen-write` **ve** hedefi beyan/kiralama istiyor |

## 3. KRİTİK: F3'ün veri seçimi G3'ün ulaşılabilirliğini belirler
Taban çizgi 0'dan çıkmak için eğitim verisinin **noktalama taşıması** şart:
- ⛔ `data/train.bin` (max_id **25.664**) ve `data/train_all_chosen.bin` (**28.260**) → noktalama bloğu **YOK**; bunlarla G3 **ulaşılamaz**.
- ✅ `data/train_chat_balanced.bin` ve `data/train_balanced_sft_v2.bin` → yeni sözlükle derlendi, noktalama ve kesme **fiilen tensor'de** (ölçüldü, doğrulandı).
**Öneri:** F3 devam eğitimi bu iki artefakt üzerinden yapılsın (karışım oranı kullanıcı kararı — bkz. §6).

## 4. Parçalama (politika: bu modele tek parça verilmez)
Her parça tek artefakt/sınırlı kapsam; her birine kapı + rollback yazılır.

| Parça | Kapsam | Çıktı | Kapı | Kim |
|---|---|---|---|---|
| **F3-0** ölçüm | Cihaz, disk, yedek kapasitesi, adım süresi tahmini | rapor (`data/eval/`) | `device == mps` **yürütücünün ortamında** doğrulandı (bu makinede MPS `True` — sandbox dışında ölçtüm; yürütücünün smoke'u CPU'da koşmuştu → teyit şart). Disk: her checkpoint 374–473 MB, bütçe 20 GiB | yürütücü (küçük) |
| **F3-1** yedek | `kristal_model.pt` (450 MB) + `kristal_model_pre_clean.pt` (442 MB) + `kristal_model_sft.pt` (357 MB) + carpenter (374 MB) + `step_b1_final` | `scratch/f3_backup_2026-09-16/` (donmuş desen DIŞI) | Her dosya için **sha256 özdeşliği**; geri yükleme provası (bir dosyayı geri kopyala, sha doğrula) | yürütücü (küçük) |
| **F3-2** kısa koşum | 50–100 adım, devam eğitimi, `_v2` verisi, **yeni ad** (`data/kristal_model_f3a.pt`) | yeni `.pt` + meta | Teacher-forcing kaybı ölçülür; noktalama üretimi **0 → >0** mı; `device` raporda | yürütücü |
| **F3-3** tam koşum | Bütçe kararı sonrası uzun koşum (aynı reçete) | `data/kristal_model_f3.pt` | gate_G3 tam: `.` `,` `:` üretimi + kayıp + kanaryalar | yürütücü |
| **F3-4** doğrulama | Bağımsız ölçüm | `data/eval/` raporu | Her sayı yeniden ölçülür; üretim testi benim koşumumla tekrarlanır | **danışman (ben)** |

**F3-2/F3-3 için mekanik:** eğitim donmuş `data/*.pt`'ye yazacağı için görev `writes[]`'te hedefi **ve** üst dizin kiralamasını beyan eder; `--allow-frozen-write` **açıkça** verilir; mevcut checkpoint'lerin üzerine **yazılmaz** (plan kuralı).

## 5. Riskler
1. **Geri dönüşsüzlük** — yedek (F3-1) tamamlanmadan F3-2 başlamamalı.
2. **CPU'da süre** — MPS doğrulanmazsa uzun koşum pratikte imkânsız; F3-2'nin adım süresi ölçümü bunu erkenden gösterir.
3. **Unutma** — devam eğitimi tabanı daraltabilir; ölçüm tasarımı (F4 kararı) bunun için zaten onaylandı.
4. **Disk** — 20 GiB bütçesinde her yeni checkpoint ~0,4 GB; F3-1 yedeği de yer kaplar (≈1,6 GB).
5. **Sessiz kırpma** — kapı dizisi kapattı; F3 spec'i yine de `[SOZLESME_UYARI]` satırını kabul kriteri yapmalı (+36 padding).

## 6. KARARLAR (Onkanat, 16 Eyl 2026 — dört soruyla onay)
1. **Veri düzeni:** **iki aşama** — `train_balanced_sft_v2.bin` sonra `train_chat_balanced.bin` (mevcut Stage-2/3 sırasıyla aynı; yeni artefakt üretilmez).
2. **Reçete:** **önce kısa ölçüm koşumu** (F3-2, 50–100 adım) — adım süresi/cihaz ve kayıp eğrisi ölçülür, TAM koşumun bütçesi o sayılara göre belirlenir.
3. **Yedek yeri:** `scratch/f3_backup_2026-09-16/` (donmuş desen dışı) + geri yükleme provası.
4. **MPS çıkmazsa:** ölçüme bırakılır — kısa koşum device'ı ve süreyi raporlar; TAM koşum kararı o sayıyla verilir.

**ÖLÇÜLEN DİSK GERÇEĞİ (16 Eyl 15:02Z, planın 20 GiB varsayımını DÜZELTİR):** boş alan **18 GiB** (disk %96 dolu). Yedeklenecek beş checkpoint toplam **~2,0 GB**:
`kristal_model.pt` 450 MB (`02e15b81…`) · `kristal_model_step_b1_final.pt` 446 MB (`01d7918e…`) · `kristal_model_pre_clean.pt` 442 MB (`d1f04433…`) · `kristal_model_sft.pt` 356 MB (`b6764c70…`) · `kristal_carpenter_model.pt` 356 MB (`5f0b8c71…`). Öncelik sırası: taban + ata adayı (896 MB) **zorunlu**, ceket (356 MB) önemli, pre_clean/sft (798 MB) türev. Her yeni F3 checkpoint'i ~0,4 GB daha yer ister → F3-0 disk ölçümü ve F3-2 sonrası yeniden ölçüm şart.

**Not:** Bu bir öneridir; onaylanmadan hiçbir parça için görev açılmaz.

---

## 7. DÜZELTME — gate_G3 formülü (T-0051 ölçümü, 16 Eyl 2026T16:14Z)

§1'de alıntılanan **gate_G3** "model `.` `,` `:` üretebiliyor mu? Taban çizgi **0**" biçimindeydi. T-0051'in kısa koşumu bunun **ÜRETİM için yanlış** olduğunu gösterdi — **iki bağımsız yöntemle**:

| Yöntem | TABAN | f3b (100 adım sonra) |
|---|---|---|
| Yürütücü (3 doğal istem) | 32137=3 · 32138=2 · 32142=1 | 32137=0 · 32138=1 · 32142=0 |
| Danışman (5 ham korpus bağlamı, greedy 32) | '.'11 ','11 ')'1 (~23) | '.'4 ','3 ':'1 (8) |

**Doğru ayrım:** "taban çizgi 0" önermesi **eğitim VERİSİ** için doğru (hiçbir `.bin` 32137+ id'lere ulaşmıyor — D1 bulgusu), ama **ÜRETİM** için değil: tabanın embedding'inde o satırlar var (eğitilmemiş) ve greedy onları seçebiliyor. Üstelik 100 adım sonra üretim **artmıyor, azalıyor** (f3b chat_balanced şablonlarına kaymış olabilir).

**G3 bundan sonra şöyle kurulmalı:** (i) **oran/bağlam** ekseni (noktalama token oranı; noktalama içeren bağlam verildiğinde teacher-forcing olasılığı), (ii) **ölçülmüş taban** cümlesi ("taban: 23/160 ham bağlamda, 6/… doğal istemde"), (iii) yeterli bütçe — F3-2 kısa ölçümdü; yürütücü 500–1000 adım öneriyor, adım süresi ölçümü iki aşama için **~33 dk/1000 adım** diyor.

**Ayrıca (T-0051 F1):** spec'te **kiralama bölümü** zorunlu — T-0051 spec'imde yoktu ve iki donmuş-desenli yazım dizin kirası olmadan yapıldı (kural 1 ihlali; veri kaybı yok). F3 parçalarının spec'lerinde bu bölüm **eksiksiz** yazılacak.

---

## 8. F3-3 SONUCU ve F4 ÖNCESİ ZORUNLU DÜZELTME (16 Eyl 2026T17:43Z)

**Koşum yapıldı ve doğrulandı** (`data/eval/t0052_verification.json`): 2×1000 adım MPS'te → `kristal_model_f3_sft.pt` (`5a0a03c2…`) + `kristal_model_f3.pt` (`eefc3ab8…`). CE iki bağımsız ölçümle: taban ~5,5 → f3_sft ~1,95 → f3 ~2,05.

**ÖLÇÜLEN GERÇEK MALİYET (planın bütçe varsayımını düzeltir):**

| Aşama | blok | adım | sn/adım | süre |
|---|---|---|---|---|
| A (sft_v2) | 64 | 1000 | **0,433** | 7,2 dk |
| B (chat_balanced) | 128 | 1000 | **3,447** | **57,5 dk** |
| **Toplam** | | 2000 | | **~65 dk** |

F3-2'nin 100 adımlık ölçümü B için 1,55 sn/adım demişti → **2,2 kat sapma**. Kural: **TAM koşu bütçesi kısa koşumdan ekstrapole edilmez**; ilk birkaç yüz adımdan sonra oran yeniden ölçülür.

**F4'ÜN ÖN KOŞULU — `<PAD>` YOZLAŞMASI (YÜKSEK; mekanizma DÜZELTİLDİ 18:34Z):** f3'ün en sık ürettiği token `<PAD>` (iki bağımsız ölçüm: 42 ve 15); tabanda ilk 5'te **yok**. ~~Mekanizma: hizalama dolgusu hedeflerde maskelenmiyor.~~ **Bu gerekçe ÖLÇÜMLE ÇÜRÜDÜ:** `train.py:157-175` zaten `<OUTPUT>` öncesi ve `<EOS>` sonrası hedefleri `-100` yapıyor; chat_balanced'da PAD hedeflerinin **%99,87'si maskeli** (12.237'nin yalnız 16'sı maskesiz). Yani dolgu zaten kayıptan çıkarılmış. **Gerçek neden ölçülmedi;** en güçlü şüpheli: PAD **girdi** olarak verinin **%47,8'i** ve `compute_sign_mask`'in kontrol token'larını (PAD dahil) **sözcük sınırı** sayması. → **`kristal_model_f3.pt` F4 başlangıcı olarak kullanılmadan önce neden araştırılmalı.**

**G3 (noktalama) SAĞLANMADI:** iki yöntem de artış göstermiyor. Tek olumlu sinyal: **kesme işareti (32850)** tabanda 0 → f3_sft'te 6-9. Kapı, PAD düzeltmesinden **sonra** yeniden ölçülmeli.
