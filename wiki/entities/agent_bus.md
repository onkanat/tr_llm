---
tags: [entity]
date: 2026-09-15
sources: [.agent-bus/SPEC.md, .agent-bus/frozen.json, scripts/agent_bus_mcp.py]
status: active
---

# agent-bus Koordinasyon Protokolü ve Çoklu Etmen Mimarisi (wiki/entities/agent_bus.md)

Bu varlık, tek bir sistem kaynağı (tek GPU, tek disk, tek çalışma alanı) üzerinde iki bağımsız yapay zeka etmeninin (Danışman: Claude Code, Yürütücü: Antigravity) eşzamanlı ve güvenli çalışmasını sağlayan **agent-bus** koordinasyon protokolünü tanımlar.

---

## 🏗️ Mimari Rol Ayrımı

* **Danışman (Claude Code):** Analiz, bağımsız doğrulama, kapı tasarımı ve görev şartnamesi hazırlama.
* **Yürütücü (Antigravity):** Şartnameye dayalı kodlama, eğitim, refaktör, ölçüm ve birim test icrası.

---

## 🛡️ Üç Değişmez Kural

1. **Kiralama Olmadan Yazma Yok (Lease First):** Bir etmen değiştireceği her dosya veya dizini işlem öncesinde `bus_acquire_lease` ile kilitlemek zorundadır.
2. **Donmuş Artefakt Korunumu (Frozen Paths):** [.agent-bus/frozen.json](file:///Users/hakankilicaslan/Git/tr_llm/.agent-bus/frozen.json) içinde listelenen yollar (`data/*.pt`, `data/*.bin`, `data/vocab*.json`, `src/compiler/**`, `data/lexicon/**` vb.) izinsiz ve geniş kapsamlı kiralama olmaksızın değiştirilemez.
3. **Kira Yaşam Süresi (TTL):** Tüm kiralamalar varsayılan 120 dakika TTL ile sınırlandırılır; süresi dolan kiralama devralınabilir.

---

## 🔄 Yürütücü Çevrimi (Executor Cycle)

Protokol çekme (pull-based) esasına dayanır. Yürütücü etmen şu adımları izler:
1. **Yokla:** `bus_list_tasks(status="open")` çağrılır.
2. **Süz:** Hedef etmen (`to`) filtresi uygulanır.
3. **Seç:** Kalan adaylar arasından **en küçük açık kimlik** seçilir (`T-0007` < `T-0008`).
4. **Devral:** `bus_claim_task(id, owner)` ile görev sahiplenilir.
5. **Tek Görev İlkesi:** Aynı anda tek görev icra edilir.
6. **Kirala, Raporla, Tekrarla:** Yazmadan önce `bus_acquire_lease`, bitince `bus_report_result`, ardından `bus_release_lease`.
7. **Boş Kuyrukta Sessizce Bekle:** Açık görev yoksa soru sormaksızın ve boş rapor üretmeksizin beklenir.

---

## 🔗 İlgili Sayfalar
* [[log]]: agent-bus üzerinden yürütülen görevlerin kronolojisi.
* [[synthesis]]: Mimari borçlar ve çoklu etmen sentezi.
