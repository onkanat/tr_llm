# CLAUDE Yönergesi (CLAUDE.md)

Bu dosya, projede komut çalıştırma ve kodlama standartlarını belirler.

---

## 🛠️ Temel Komutlar

### 1. Doğrulama ve Testler
* **Birim Testlerini Çalıştır:** `venv/bin/pytest`
* **Model Çıkarım Testi:** `venv/bin/python test_model.py`

### 2. Derleme ve Eğitim
* **Wikipedia Veri Kümesini Hazırla:** `venv/bin/python scripts/prepare_wiki_dataset.py`
* **5 Fazlı Birleşik Veri Kümesini Derle:** `venv/bin/python scripts/prepare_all_phases_with_wiki.py`
* **Modeli Eğit (MPS GPU):** `venv/bin/python train.py`

---

## ✒️ Kodlama Standartları

* **Tip Güvenliği:** Python type hints (`List`, `Dict`, `Any`, `str` vb.) kullanılması zorunludur.
* **Fonksiyon Tasarımı:** Sistem modülleri global durum (mutable state) barındırmayan saf fonksiyonlar (pure functions) olarak yazılmalıdır.
* **Hata Yönetimi:** Beklenmeyen durumlar ve OOV (Out-of-Vocabulary) hataları sessizce geçiştirilmemeli, loglanmalı ve uygun varsayılan değerler dönmelidir.
* **Kararlılık (Determinism):** Algoritmalar deterministik çalışmalı, olasılıksal yaklaşımlar yerine kural tabanlı yapılar tercih edilmelidir.
