# Skill 1: Sözlük Yönetimi (Lexicon Management) - Faz 1

Bu doküman, Türkçe Kristal-Vektörel Mimarisi'nin leksikon katmanını, kök morfemlerin ($M_k$) hafızada tutulması, ses olaylarına duyarlı aranması ve TDK GTS anlamsal ontoloji entegrasyonu mekanizmalarını detaylandırır.

---

## 🏛️ Sınıf Yapısı ve Mimari Tasarım

Sözlük yönetimi, doğrusal aramaların getirdiği maliyeti önlemek amacıyla **Trie (Önek Ağacı)** veri yapısı üzerinde çalışır. Bu sayede arama karmaşıklığı $O(1)$ veya en fazla kelime uzunluğu olan $O(k)$ seviyesine çekilmiştir.

### 1. `TrieNode` Sınıfı
Trie ağacındaki her bir düğümü temsil eder.
*   **`children`** (`Dict[str, TrieNode]`): Düğümün alt karakter dallarını tutar.
*   **`is_word`** (`bool`): Düğümün bir kelimenin son karakteri (kök bitişi) olup olmadığını belirtir.
*   **`entries`** (`List[Dict[str, Any]]`): Eşsesli (homonym) kökleri desteklemek için ilgili köke ait tüm meta veri sözlüklerini liste halinde tutar.

### 2. `LexiconManager` Sınıfı
Ağaç yapısını yöneten ve arama motorunu barındıran ana sınıftır (`src/compiler/lexicon.py`).
*   **`root`** (`TrieNode`): Ağacın kök düğümünü barındırır.
*   **`load_from_tsv(filepath: str)`**: `data/lexicon/roots.tsv` dosyasını sekme ayıracıyla okur ve her satırı Trie yapısına ekler.
*   **`_insert(word: str, data: Dict[str, Any])`**: Kelimeyi ve meta verilerini Trie düğümlerine adım adım yerleştirir.
*   **`find_stems(word: str) -> List[tuple[str, Dict[str, Any]]]`**: Verilen bir yüzey formu kelimenin tüm olası kök adaylarını ve niteliklerini döner.

---

## 🌀 Ses Yumuşaması Duyarlı Kök Bulma Algoritması (`find_stems`)

Türkçe'deki ünsüz yumuşaması ses olayı nedeniyle (`p, ç, t, k` $\rightarrow$ `b, c, d, ğ, g`), kelimenin yüzey formunda görünen kök yumuşamış olabilir (örn. *kitap* $\rightarrow$ *kitab-ı*, *git* $\rightarrow$ *gidecek*). `find_stems` metodu bu durumları Trie seviyesinde çözmek için şu adımları izler:

1.  **Doğrudan Eşleşme:** Karakter bazlı ilerleyerek Trie dalları takip edilir. Düğümde `is_word == True` ise doğrudan kök listesine eklenir.
2.  **Yumuşama Kontrolü ve Geri Dönüşüm (`reverse_voicing_map`):**
    Arama esnasında karşılaşılan yumuşak harfler, sert karşılıklarına dönüştürülerek paralel bir arama yolu açılır:
    *   `b` $\rightarrow$ `p`
    *   `c` $\rightarrow$ `ç`
    *   `d` $\rightarrow$ `t`
    *   `ğ` $\rightarrow$ `k`
    *   `g` $\rightarrow$ `k`
3.  **Öznitelik Doğrulaması:** Eğer sertleştirilmiş önek Trie üzerinde geçerli bir kök bulursa, bu kökün öznitelikleri (`attributes`) içinde `"VOICING"` değeri aranır. Sadece `"VOICING"` özniteliğine sahip kökler yumuşamış adayın kökü olarak kabul edilir.

---

## 📖 TDK GTS Anlamsal Ontoloji Entegrasyonu

Kristal-Vektörel mimarisi, istatistiksel devasa metin yığınları (raw web scrape) yerine doğrulanmış semantik kök ontolojisi kullanır:
*   **Kök Havuzu:** Yaklaşık 20.500 doğrulanmış temel kök (`roots.tsv`).
*   **TDK Güncel Türkçe Sözlük (GTS):** `scripts/generate_lexical_semantics_dataset.py` aracılığıyla her bir leksikal kökün anlam tanımları, eşanlamlıları ve anlamsal sınırları üretilip `data/pedagogy/lexical_semantics_dataset.jsonl` halinde pedagojik eğitime enjekte edilir.
*   **Zero-OOV İlkesi:** Dildeki tüm yüzey formlar sonlu atomik kök kümesi ($M_k$) ve kurallı ek morfotaktiği ($M_e$) üzerinden türetildiğinden, leksikona türemiş kelimeler (derivational noise) eklenmez.

---

## 📂 Veri ve Sözleşme Örneği

Leksikon dosyası (`data/lexicon/roots.tsv`) aşağıdaki formatta bir yapı sunar:

| lemma | pos  | attributes  |
| :---  | :--- | :---        |
| kitap | NOUN | VOICING     |
| git   | VERB | VOICING     |
| su    | NOUN | -           |
| ağız  | NOUN | VOWEL_DROP  |

### Çıktı Sözleşmesi
`find_stems("kitaba")` çağrıldığında dönen değerler:
```python
[
    ("kitab", {"lemma": "kitap", "pos": "NOUN", "attributes": "VOICING"})
]
```
Burada `"kitab"` yüzey formundaki kök kesitidir, sözlükteki asıl leksikal ID ise `"kitap"`tır.
