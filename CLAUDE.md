# CLAUDE Yönergesi (CLAUDE.md)

Bu dosya, projede komut çalıştırma, kodlama standartları ve temel güvenlik kısıtlarını belirler.

---

## 🛠️ Temel Komutlar

### 1. Doğrulama ve Testler
* **Birim Testlerini Çalıştır:** `venv/bin/pytest`
* **Model Çıkarım Testi (Bilinen Boyut Sınırlılığı):** `venv/bin/python test_model.py`
  > [!WARNING]
  > `test_model.py` betiği modeli `data/vocab.json` (31.357) boyutuyla başlatmakta olup güncel 32.816 satırlı `kristal_model.pt` yüklendiğinde şekil uyuşmazlığı nedeniyle `RuntimeError` vermektedir. Uyumlu çıkarım için `src.llm.prompt_contract.resize_state_dict` veya `scripts/run_experiment_c.py` kullanılmalıdır (T-0027 K7/F1).

### 2. Derleme ve Eğitim
* **Wikipedia Veri Kümesini Hazırla:** `venv/bin/python scripts/prepare_wiki_dataset.py`
* **5 Fazlı Birleşik Veri Kümesini Derle:** `venv/bin/python scripts/prepare_all_phases_with_wiki.py`
* **Modeli Eğit (MPS GPU):** `venv/bin/python train.py`

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
