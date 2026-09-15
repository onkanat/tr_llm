---
tags: [entity]
date: 2026-09-15
sources: [data/pedagogy/cot_vault.jsonl, tests/test_cot_vault_and_dual_parser.py]
status: active
---

# Düşünce Zinciri Kasası (CoT Vault) ve Test İzolasyonu (wiki/entities/cot_vault.md)

Bu varlık, modelin pedagojik akıl yürütme adımlarını içeren üretim verisi kasasını (`cot_vault.jsonl`) ve test süreçlerinde bu kasanın kirlenmesini önleyen izolasyon mimarisini tanımlar.

---

## 🏛️ Kasanın Rolü ve İçeriği

`data/pedagogy/cot_vault.jsonl`, pedagojik denetçi (`pedagogical_supervisor.py`) tarafından onaylanan yüksek kaliteli morfolojik düşünce zinciri (Chain of Thought - CoT) örneklerini ve çift ayrıştırıcı (dual parser) kanıtlarını depolar.

---

## 🔒 Test İzolasyonu ve Bütünlük Koruması

T-0023 öncesinde, birim testlerin doğrudan üretim kasasına yazması nedeniyle kasaya sahte/test satırları karışmaktaydı. Bu durum şu mimari tedbirlerle giderilmiştir:
1. **Geçici Kasa İzolasyonu:** Testler üretim dosyasını değil, `tmp_path` üzerinde oluşturulan geçici kasa yollarını kullanmak zorundadır.
2. **SHA-256 Değişmezlik Çıpası:** Üretim kasasının dosya özeti (`sha256`) ve satır sayısı her test koşumu öncesinde ve sonrasında doğrulanır.
3. **Salt-Okunur Statü:** Kasa üretim ortamı dışında hiçbir betik veya test tarafından değiştirilemez.

---

## 🔗 İlgili Sayfalar
* [[kristal_lm]]: Dil modeli mimarisi.
* [[agent_bus]]: Kasa üzerindeki kiralama ve güvenlik denetimi.
