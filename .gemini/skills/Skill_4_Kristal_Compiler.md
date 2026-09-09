# Skill 4: Kristal Derleyici (Crystal Compiler) - Faz 4 & 5

Bu doküman, Türkçe Kristal-Vektörel Mimarisi'nin Leksikon, Morfotaktik ve Fonoloji modüllerini koordine ederek kelimeleri morfem akışlarına çözümleyen **Kristal Derleyici** ve çift yönlü derleme/çözümleme süreçlerini detaylandırır.

---

## 🏛️ Sınıf Yapısı ve `compile` Akışı

`CrystalCompiler` sınıfı (`src/compiler/__init__.py`), kelimelerin morfolojik analizini deterministik ve tam izlenebilir şekilde gerçekleştiren orkestrasyon katmanıdır.

### 1. `_turkish_lower(word: str) -> str`
Türkçe dil kurallarına uygun küçük harf dönüşümü yapar. Büyük 'İ' harfini 'i' ve büyük 'I' harfini 'ı' olacak şekilde dönüştürerek büyük-küçük harf duyarsızlığı sağlar.

### 2. `compile(word: str) -> Dict[str, Any]`
Çözümleme işleminin ana giriş noktasıdır:
1.  Girdi kelimesini `_turkish_lower` ile normalleştirir.
2.  `LexiconManager.find_stems` üzerinden kelime içindeki tüm olası kök adaylarını tespit eder.
3.  Kökün POS etiketine (`VERB`, `NOUN` vb.) göre Morfotaktik Graf üzerindeki başlangıç durumunu belirler.
4.  Derinlemesine Arama (DFS) tabanlı `_find_paths_recursive` metodunu çağırarak tüm yasal çekim ve türetim yollarını tarar.
5.  Bulunan yolları `_score_paths` ile puanlar ve en yüksek skorlu analizi seçer.
6.  En iyi iki çözümleme arasındaki fark $0.1$'den küçükse `needs_disambiguation = True` bayrağını işaretler.
7.  Standart `CrystalPack` JSON çıktısını üretir.

---

## 📐 Derinlemesine Arama ve Erken Budama (DFS & Beam Pruning)

### `_find_paths_recursive` Algoritması
Yol keşfi, özyinelemeli (recursive) bir DFS aramasıdır:
*   **Temel Durum (Base Case):** Eğer ulaşılan geçici kelime (`current_string`) girdi kelimesine tam eşitse ve bulunulan durum terminal durumsa (`graph.is_terminal`), yol geçerli bir analiz olarak kaydedilir.
*   **Arama Durumu:** Geçerli durumdan çıkış yapan tüm yasal morfotaktik geçişler (`transitions`) taranır.
*   **Erken Budama (Beam Pruning):** Geçişteki ek şablonu çözüldükten sonra oluşan yeni kelime dizisinin (`new_string`), hedef kelimenin bir öneki (prefix) olup olmadığı kontrol edilir (`target_word.startswith`). Önek uyuşmuyorsa yol derhal budanır.
    *   *Yumuşama Toleransı:* Eğer ekin özniteliklerinde `VOICING` varsa ve yeni eklenen ses sert ünsüzken hedef kelimede yumuşak karşılığı bulunuyorsa (örn. hedefte `d` varken yeni dizide `t` olması), önek kontrolünün başarılı olmasına izin verilir.
*   **Döngü Engelleme (Infinite Loop Prevention):** Ek şablonunun boş olduğu ve kelime uzunluğunu artırmayan geçişlerde, ekin daha önce yolda kullanılıp kullanılmadığı kontrol edilerek sonsuz döngüler engellenir.
*   **Yol Güncellemesi ve Senkronizasyon:** Kök sesli harf düşmesine veya ünsüz yumuşamasına uğramışsa, yol üzerindeki kök nesnesinin yüzey formu (`surface`) güncellenerek gerçekte yazılan form ile senkronize edilir.

---

## 🏷️ Kontrol Belirteçleri ve Meta Etiket Desteği

Derleyici ve tokenizer boru hattı, sistem kontrol etiketlerini (`<ARA>`, `</ARA>`, `<BELGE>`, `</BELGE>`, `<INSTRUCTION>`, `<INPUT>`, `<OUTPUT>`) leksikal aramalara sokmadan ayrıştırır. Bu sayede model hem morfolojik çözümlemeyi hem de Self-RAG otonom arama belirteçlerini kusursuz şekilde aynı akışta işler.

---

## ⚖️ Puanlama Modülü (`ScoringModule`) ve Çıktı Sözleşmesi

### Puanlama Algoritması
`_score_paths` metodu, bulguları sezgisel formülle puanlar:
$$\text{Score} = 10.0 - \text{Morfem Sayısı}$$
Bu sayede dildeki daha basit, daha az morfem içeren ve genel kabul gören analizler en üst sıraya yerleştirilir.

### `CrystalPack` JSON Çıktı Sözleşmesi
Derleyici, aşağıdaki standart JSON yapısını döner:

```json
{
  "input": "kitaplarda",
  "language": "tr",
  "analyses": [
    {
      "morphemes": [
        {"type": "ROOT", "id": "kitap", "surface": "kitap", "pos": "NOUN", "attributes": "VOICING"},
        {"type": "AFFIX", "id": "PLURAL", "surface": "lar", "attributes": "-"},
        {"type": "AFFIX", "id": "CASE_LOC", "surface": "da", "attributes": "-"}
      ],
      "surface_form": "kitaplarda",
      "score": 7.0
    }
  ],
  "best_surface": "kitaplarda",
  "token_vector": ["kitap", "PLURAL", "CASE_LOC"],
  "needs_disambiguation": false
}
```
*   **`token_vector`:** Modelin ve tokenizer'ın doğrudan anladığı temiz morfem ID listesidir.
