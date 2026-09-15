---
tags: [concept]
date: 2026-09-15
sources: [src/llm/prompt_contract.py, tests/test_prompt_contract.py]
status: active
---

# Kanonik Ağırlık Boyutlandırma ve Sessiz Kırpma Yasağı (wiki/concepts/state_dict_resizing.md)

Bu konsept, sözlük genişlemelerinde veya farklı kelime dağarcığına sahip modeller yüklenirken uygulanan ağırlık boyutlandırma kurallarını ve sessiz veri kırpma (silent slicing) yasağını tanımlar.

---

## 🚫 Sessiz Kırpma (Silent Slicing) Riski

Önceki yükleme mantıklarında, sözlük boyutu hedef modelin boyutundan küçük olduğunda tensörler dilimlenerek (`wte.weight[:vocab_size]`) sessizce yüklenmekteydi. Bu durum:
1. Eğitilmiş token ağırlıklarının uyarısız biçimde çöpe atılmasına,
2. Modelin temsil uzayında sessizce bozulmaya ve tahmin hatalarına neden olmaktaydı.

---

## 🛡️ Kanonik Boyutlandırma Kapısı: `resize_state_dict`

[src/llm/prompt_contract.py](file:///Users/hakankilicaslan/Git/tr_llm/src/llm/prompt_contract.py) modülünde yer alan `resize_state_dict` fonksiyonu projedeki tek kanonik ağırlık dönüştürücüdür.

Uygulanan değişmez kurallar:
1. **Yalnızca Genişletme (Zero-Padding):** Model boyutunun sözlük boyutundan küçük olduğu durumlarda tensörler sıfır dolgusuyla genişletilebilir.
2. **Daraltma ve Kırpma Yasağı:** Hedef sözlük boyutu model tensör boyutundan küçükse fonksiyon **asla sessiz kırpma yapmaz**, açık biçimde `ValueError` fırlatarak çöker.
3. **Maske Tamponu Filtreleme:** Ağırlık sözlüğünde bulunan gereksiz bool maske tamponları (`blocks.*.attn.mask`) temizlenerek bellek israfı önlenir.

---

## 🧪 Doğrulama ve Testler

* `tests/test_prompt_contract.py`: Boyutlandırma kapısının sessiz kırpmayı engellediğini ve mutant test senaryolarını doğrulayan test kümesi.
* Yüklenebilirlik ölçümleri: `data/eval/t0027_model_state_2026-09-15.json`

---

## 🔗 İlgili Sayfalar
* [[vocabulary_alignment]]: Sözlük hizalaması.
* [[model_checkpoints]]: Checkpoint yükleme envanteri.
