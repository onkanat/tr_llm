(Faz 4 & 5)
Sorumluluk: Leksikon, Morfotaktik ve Fonoloji motorlarını bir orkestra şefi gibi yöneterek Ana Algoritma 1'i (Kristal Derleyici) yürütür
.
Yol Keşfi (Path Finding): Girdi kelimesi için Derinlemesine Arama (DFS) özyinelemeli \_find_paths_recursive() algoritması çalıştırır. Her geçerli adımda fonetik uygunluk test edilir, geçersiz yollar erken budanır (beam pruning)
.
Puanlama ve Belirsizlik (Scoring & Disambiguation): Bulunan analizleri ScoringModule ile puanlar. Formül: score = rule_confidence_product \* length_factor. En iyi iki analiz arasındaki fark belirlenen deltadan küçükse needs_disambiguation = True bayrağını yakar
.
Çıktı: Standartlaştırılmış CrystalPack JSON nesnesi
. Altın vaka (golden cases), negatif vaka ve kenar durum (edge cases) test suite'leri unittest ile mutlaka çalıştırılmalıdır
.
