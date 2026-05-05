(Faz 1)
Sorumluluk: Dilin kök morfemlerini (M
k
​
) hafızada tutmak ve yönetmek
.
Girdi/Çıktı: .tsv tabanlı bir veri dosyasından (lemma, POS, attributes) okur. Girdi kelimenin öneklerini (prefix) tarayarak olası kök adaylarını find_stems(word) döndürür
.
Özellikler: Ön-bellekleme (Trie ağacı veya optimize edilmiş dictionary) ile O(1) veya O(k) arama karmaşıklığı sunmalıdır
. Köklerin "VOICING" (yumuşamaya yatkınlık) gibi fonetik meta verilerini taşıması şarttır
.
