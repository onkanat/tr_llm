# LLM Coder: Agent-First Software Development Pattern

Bu dosya, projede görev alacak yapay zeka etmenlerinin (AI Agents) mimariyi, tasarım standartlarını ve proje kurallarını bozmadan çalışmasını sağlayan **LLM Coder** standartlarını açıklar.

---

## 💡 "Kod Gövdedir, Wiki Zihindir" Felsefesi

Etmenler projenin kod tabanını (`src/`, `tests/`, `scripts/`) değiştirirken, bu değişikliklerin getirdiği kavramsal ve yapısal yenilikleri mutlaka `wiki/` dizini altındaki dokümanlarla senkronize etmelidir.

---

## 📂 Dizin Yapısı

* **`CLAUDE.md`**: Derleme, test ve kodlama yönergeleri.
* **`wiki/index.md`**: Proje modülleri fihristi.
* **`wiki/log.md`**: Kronolojik geliştirme günlüğü.
* **`wiki/synthesis.md`**: Teknik borçlar ve mimari sentez analizi.
* **`wiki/entities/`**: Sınıflar, veri yapıları ve API'ler.
* **`wiki/concepts/`**: Tasarım kalıpları, algoritmalar ve kurallar.

---

## 🛠️ Temel Kurallar

1. **Önce Plan, Sonra Kod:** Değişiklikler yapılmadan önce mutlaka plan hazırlanmalı ve kullanıcı onayı alınmalıdır.
2. **Wiki Senkronizasyonu:** Kodda yapılan her yapısal değişiklikten sonra `wiki/` altındaki ilgili dokümanlar güncellenmelidir.
3. **YAML Frontmatter:** Tüm `wiki/` markdown dosyaları standart frontmatter ile başlamalıdır.
4. **Sayfa Bağlantıları:** Dokümanlar arası geçişler `[[sayfa-adi]]` biçiminde olmalıdır.
