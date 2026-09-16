# CLAUDE Yönergesi (CLAUDE.md)

Bu dosya, projede komut çalıştırma, kodlama standartları ve temel güvenlik kısıtlarını belirler.

---

## 🛠️ Temel Komutlar

### 1. Doğrulama ve Testler
* **Birim Testlerini Çalıştır:** `venv/bin/pytest` (16 Eyl 2026 itibarıyla **195 passed**)
* **Model Çıkarım Testi (Bilinen Boyut Sınırlılığı):** `venv/bin/python test_model.py`
  > [!WARNING]
  > `test_model.py` betiği modeli hâlâ `data/vocab.json` (**31.357**) boyutuyla başlatır; güncel checkpoint'ler (ör. 32.816 satırlı `kristal_model.pt`) ile şekil uyuşmazlığı `RuntimeError` verir. Uyumlu çıkarım için `src.llm.prompt_contract.resize_state_dict` kullanılmalıdır (T-0027 K7/F1).
  > **Güncel kanonik taban sözlüğü 32.852'dir** (`data/rebuild/vocab_base_32852.json`): eğitim hattı (`train.py`, `train_scientific_sft.py`, `evaluate_sft_benchmarks.py`) 16 Eyl 2026'dan beri bu sözlüğü yükler ve checkpoint'i `resize_state_dict` ile hizalar — uyumsuzluk `[SOZLESME_UYARI]` satırıyla **görünür** biçimde basılır, sessiz kırpma yoktur (T-0046).

### 2. Derleme ve Eğitim
* **Wikipedia Veri Kümesini Hazırla:** `venv/bin/python scripts/prepare_wiki_dataset.py`
* **5 Fazlı Birleşik Veri Kümesini Derle:** `venv/bin/python scripts/prepare_all_phases_with_wiki.py`
* **Modeli Eğit (MPS GPU):** `venv/bin/python train.py --device mps --data <bin> --steps N --load-path <ckpt> --save-path <yeni-ad>`
  > [!WARNING]
  > **Donmuş yola yazım operatör onayı ister:** `train.py`, `train_dpo.py`, `run_goal_pipeline.py` ve `retrain_clean_models.py` `data/*.pt` gibi donmuş desenlere yazmadan önce `check_frozen_save_path` çağırır; onay verilmezse `RuntimeError` ile **durur**. Onay `--allow-frozen-write` ile açıkça verilir ve hedef `writes[]`'te beyan edilip üst dizin kiralanmalıdır (T-0047…T-0049).
  > **`<PAD>` kayıp maskesi varsayılan olarak AKTİF:** hizalama dolgusu eğitim hedefi sayılmaz (`--no-pad-mask` ile kapanır). Maskeleme kayıp **ölçeğini** değiştirir (aynı rejimde ~1,58 vs ~6,85) → eski kayıp değerleriyle kıyaslama yapılmamalıdır (T-0053).
  > **Koşum sırasında makinede GPU tüketen başka iş çalıştırmayın:** aynı makinede tarayıcı/video açıkken adım süresi 0,43 → 3,45 sn/adım'a çıktı (T-0052).

---

## 🛡️ Proje Güvenlik Kısıtları ve Sözleşme Çıpası

Bu projenin en sert kısıtları canlı sözleşme belgesi olan [`.agent-bus/SPEC.md`](.agent-bus/SPEC.md) üzerinde tanımlıdır. Çift başlılık ve kural sapmalarını önlemek için standartlar burada yinelenmez; yürütücüler aşağıdaki kurallara kesin olarak uymakla yükümlüdür:

1. **Yazmadan Önce Kiralama Zorunluluğu (SPEC.md Kural 1):** Bir görevin `writes[]` listesinde olmayan veya `bus_acquire_lease` ile kiralanmamış hiçbir dosya/dizin değiştirilemez.
2. **Donmuş Yollar Salt-Okunurdur (SPEC.md Kural 2):** [`.agent-bus/frozen.json`](.agent-bus/frozen.json) içinde listelenen yollar (`data/*.pt`, `data/*.bin`, `data/vocab*.json`, `src/llm/tokenizer.py`, `src/compiler/**`, `data/lexicon/**`, `data/b1_5_splits/**`, `data/realistic_rag/**`, `data/pedagogy_canonical/**`) asla izinsiz ve kiralamasız değiştirilemez.
3. **Veri Dizini Koruma:** `data/**` altındaki tüm yollar salt-okunurdur; tek istisna değerlendirme ve ölçüm raporlarının yazıldığı `data/eval/` dizinidir.
4. **Git Bütünlüğü:** `git add -A` veya `git add .` kullanımı kesinlikle YASAKTIR (büyük ve izlenmeyen veri dosyalarının sahnelenmesini engellemek için). Yetkisiz commit veya push yapılamaz.

---

## ✒️ Kodlama Standartları

* **Tip Güvenliği:** Python type hints (`List`, `Dict`, `Any`, `str` vb.) kullanılması zorunludur.
* **Fonksiyon Tasarımı:** Sistem modülleri global durum (mutable state) barındırmayan saf fonksiyonlar (pure functions) olarak yazılmalıdır.
* **Hata Yönetimi:** Beklenmeyen durumlar ve OOV (Out-of-Vocabulary) hataları sessizce geçiştirilmemeli, loglanmalı ve uygun varsayılan değerler dönmelidir.
* **Kararlılık (Determinism):** Algoritmalar deterministik çalışmalı, olasılıksal yaklaşımlar yerine kural tabanlı yapılar tercih edilmelidir.
