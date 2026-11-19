# GEÇERSİZ KILINMIŞ PİLOT VERİSETİ (INVALIDATED)

Bu dizindeki veriler (`train_800.jsonl`, `val_100.jsonl`, `test_100.jsonl`) ve checkpoint'ler (`kristal_rag_pilot_best.pt`), 14 Eylül 2026 tarihli ampirik gerçeklik denetimi neticesinde **GEÇERSİZ KILINMIŞTIR**.

### Geçersiz Kılınma Nedenleri:
1. **%97-98 4-Gram Sızıntısı:** Veri üretiminde kullanılan Python `f-string` şablonları (`build_rich_rag_pilot.py`), sözcükler değiştirilse dahi sentaktik iskeleti (boilerplate) %100 oranında sızdırmıştır.
2. **Syntax-Held-Out İhlali (%80.0):** Test çıktılarının %80'i eğitimdeki sözdizimsel iskeletlerle birebir eşleşmektedir.
3. **In-Domain Doc-Swap İflası:** Model sunulan belgenin özgül olgularını yalnızca %20.5 oranında üretirken, soru-şablon ezberinden eski belge olgularını %37.7 oranında sayıklamaktadır.

**Bu dizindeki hiçbir kütük yeni eğitim veya değerlendirme döngülerinde KESİNLİKLE KULLANILMAMALIDIR.**
