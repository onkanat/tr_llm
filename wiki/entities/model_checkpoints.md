---
tags: [entity]
date: 2026-09-15
sources: [data/eval/t0027_model_state_2026-09-15.json]
status: active
---

# Model Checkpoint Envanteri ve Durum Taksonomisi (wiki/entities/model_checkpoints.md)

Bu varlık, projede üretilen PyTorch model ağırlıklarının (`data/*.pt`) yapısal özelliklerini, bit-özdeş gruplarını ve dikkat maskesi tamponlarını belgeler.

---

## 🗂️ Checkpoint Taksonomisi

Projede kaydedilen model checkpoint'leri iki ana kategoride izlenir:
1. **Üst Düzey Aktif Modeller (`data/*.pt`):** Eğitim aşamalarının son durumlarını temsil eden ve doğrudan çıkarımda kullanılan modeller.
2. **Arşiv ve Bölme Modelleri (`data/_archive/`, `data/b1_5_splits/`):** Tarihsel deneyler ve B1.5 split analizleri için saklanan modeller.

---

## 🧬 Bit-Özdeş Gruplar ve Eğitim Doğrulaması

T-0027 ampirik envanterinde incelendiği üzere:
* `kristal_b1_5_best.pt`, `kristal_b1_5_epoch3.pt` ve `kristal_model.pt` modelleri bit düzeyinde özdeştir (aynı ağırlık tensörlerini paylaşır).
* Epoch 1 ve Epoch 2 modellerinin farklı ağırlık özetlerine sahip olması, eğitimin gerçek ağırlık güncellemeleri ürettiğini kanıtlamıştır.

---

## ⚠️ Dikkat Maskesi Tamponları (Mask Buffers)

Model mimarisinde her blokta (`blocks.0..5.attn.mask`) 4096^2 boyutunda üçgen nedensel bool maske tamponları kaydedilmektedir. Bu tamponlar checkpoint dosya boyutunun önemli bir kısmını oluşturmakta olup model çıkarımı sırasında yeniden boyutlandırma kapısıyla elenebilir.

---

## 📊 Ölçüm ve Raporlar

* Tüm modellerin parametre sayıları, SHA256 özetleri ve çıkarım testleri: `data/eval/t0027_model_state_2026-09-15.json`

---

## 🔗 İlgili Sayfalar
* [[kristal_lm]]: Model mimarisi.
* [[state_dict_resizing]]: Checkpoint yükleme mekanizması.
* [[vocabulary_alignment]]: Model tensör boyutları.
