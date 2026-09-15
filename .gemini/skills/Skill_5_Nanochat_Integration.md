# Skill 5: Tokenizer ve Model Mimarisi (Tokenizer & Model Architecture) - Taşıyıcı Sistem

Bu doküman, Türkçe Kristal-Vektörel Mimarisi'nin metin normalizasyonunu, morfem düzeyinde tokenizasyonunu, morfem ağırlıklandırma mekanizmasını, sözlük genişletme dinamiklerini ve **RoPE + Sign Inversion** içeren Causal Transformer (KristalLM) dil modelini detaylandırır.

---

## 🏛️ Leksikal Sözlük ve Tokenizasyon (`Vocabulary` & `KristalTokenizer`)

Geleneksel istatistiksel BPE (Byte-Pair Encoding) yerine, morfem ID'lerini ve semantik kontrol belirteçlerini doğrudan sayısal ID'lere haritalayan özelleştirilmiş bir sözlük ve tokenizer kullanılır (`src/llm/tokenizer.py`).

### 1. `Vocabulary` Sınıfı
*   **Özel Kontrol Belirteçleri:**
    *   `<UNK>` (ID: 0) $\rightarrow$ Bilinmeyen/Sözlük dışı kelime.
    *   `<PAD>` (ID: 1) $\rightarrow$ Doldurma (padding).
    *   `<BOS>` (ID: 2) $\rightarrow$ Başlangıç etiketi.
    *   `<EOS>` (ID: 3) $\rightarrow$ Bitiş etiketi.
    *   `<INSTRUCTION>` / `</INSTRUCTION>` (ID: 4/5) $\rightarrow$ Talimat sınırları.
    *   `<INPUT>` / `</INPUT>` (ID: 6/7) $\rightarrow$ Görev girdisi sınırları.
    *   `<OUTPUT>` / `</OUTPUT>` (ID: 8/9) $\rightarrow$ Görev çıktısı sınırları.
    *   `<NUMBER>` (ID: 10) $\rightarrow$ Sayısal veriler.
    *   `<PROPER_NOUN>` (ID: 11) $\rightarrow$ Özel isimler.
    *   `<ARA>` / `</ARA>` $\rightarrow$ Otonom Self-RAG arama sorgusu tetikleme etiketleri.
    *   `<BELGE>` / `</BELGE>` $\rightarrow$ Geri çağrılan RAG dokümanı bağlam etiketleri.
*   **Sözlük Boyutu ve Genişleme:** Temel sözlük 31.357 token kapasitesine sahiptir (`data/vocab.json`). Güncel modeller 32.816 (`data/rebuild/vocab_b1_5_32816.json`) ve 32.852 (`data/vocab_entity.json`) token boyutundadır. Yeni morfemler `add_token(token)` metoduyla dinamik eklenir.
*   **Ağırlık Boyutlandırma (`resize_state_dict`):** Model kontrol noktaları (checkpoints) yüklenirken, sözlük boyutu genişlemişse `embedding.weight` ve `lm_head.weight` tensörleri sıfırdan eğitilmeden mevcut ağırlıklar korunarak otomatik genişletilir.

### 2. `KristalTokenizer` Sınıfı
*   **Metin Normalizasyonu:** Girdi metinleri `unicodedata.normalize('NFC', text)` ile birleşik karakter standardına dönüştürülür.
*   **JSON/JSONL Çözümleme:** Girdi bir JSON nesnesi ise otomatik olarak ayıklanıp `<INSTRUCTION> ... </INSTRUCTION> <INPUT> ... </INPUT> <OUTPUT> ... </OUTPUT>` şablonuna dönüştürülür.
*   **Düzenli İfade Bölümleme:** `r"<[^>]+>|[^\s]+"` ifadesiyle kontrol etiketleri ve normal kelimeler ayrıştırılır.
*   **Derleme ve Fallback:** Her kelime `CrystalCompiler.compile` işlemine gönderilir:
    *   Derleme başarılıysa: Kelimenin morfem ID dizisi (`token_vector`) sözlüğe eklenerek kodlanır.
    *   Derleme başarısızsa:
        *   Kelimenin ilk harfi büyükse $\rightarrow$ `<PROPER_NOUN>` olarak kodlanır.
        *   Kelimenin ilk harfi küçükse $\rightarrow$ `<UNK>` olarak kodlanır ve loglarda OOV uyarısı verilir.

---

## ⚖️ Morfem Ağırlık Modülü (`get_morpheme_weight`)

Modelin eğitiminde ve kayıp (loss) hesabında, anlam taşıyan kritik morfemlerin ağırlığını artıran katsayı yapısı işletilir:
1.  **Mantıksal Operatörler (`NEG`, `IMPOTENTIAL_NEG`, ama, fakat):** Ağırlık **`4.0`** (Olumsuzluk ve bağlaçların hatasız öğrenilmesi için en yüksek ağırlık).
2.  **Leksikal Kökler (küçük harfli kök etiketleri veya `<PROPER_NOUN>`):** Ağırlık **`3.5`** (Anlam taşıyıcı ana kelimeler).
3.  **Türetim Ekleri (`DERIV_` ile başlayanlar):** Ağırlık **`1.5`** (Kelime sınıfını ve anlamını değiştiren ekler).
4.  **Çekim Ekleri ve Kontrol Tokenleri (`TENSE_`, `PERSON_`, `PLURAL` vb.):** Ağırlık **`0.1`** (Gramer yapıları, ezberlemeyi önlemek için düşük ağırlıkta tutulur).

---

## 🏛️ Model Mimarisi (`KristalLM`)

`KristalLM`, Causal Self-Attention katmanları içeren ve bitişken dillerin yapısına göre optimize edilmiş Causal Transformer Decoder modelidir (`src/llm/model.py`).

### 1. İşaret Terslemeli Embedding (`KristalEmbedding`)
*   **Sign Inversion:** Türkçe'de olumsuzluk eki (`mA`, `(y)AmA`) alan bir kelimenin anlamsal yönü tamamen değişir. Bu durumu geometrik olarak temsil etmek için, bir kelimenin morfem dizilimi içinde `NEG` veya `IMPOTENTIAL_NEG` varsa, o kelimenin **tüm morfem embedding vektörleri** $-1.0$ ile çarpılarak vektör uzayında zıt yöne çevrilir.
*   `compute_sign_mask` metoduyla CPU üzerinde hesaplanan maske, embedding matrisine `embeddings * sign_mask` şeklinde uygulanır.

### 2. Döner Konumsal Kodlama (`RoPE`)
*   Mutlak konum embedding'i yerine `RoPE` (Rotary Positional Embeddings) kullanılır.
*   Sorgu (Query) ve anahtar (Key) vektörleri head boyutu bazında ikişerli koordinatlara ayrılır ve konumsal frekanslara göre rotasyona uğratılır:
    $$q_{rotated} = (q \times \cos) + (\text{rotate\_half}(q) \times \sin)$$
    $$k_{rotated} = (k \times \cos) + (\text{rotate\_half}(k) \times \sin)$$
*   Bağlam penceresi dinamik olarak genişletilebilir.

### 3. Donanım ve Hesaplama Optimizasyonu
*   **macOS MPS AdamW Kararlılığı:** PyTorch Metal arka yüzünün ~31k sözlük boyutundaki AdamW tensör güncellemelerinde girdiği deadlock problemini önlemek amacıyla, geriye yayılım (backward pass) ve optimizer adımları CPU üzerinde kararlı olarak işletilir. İnferans aşamasında ise MPS donanım hızlandırmasından tam verim alınır.
