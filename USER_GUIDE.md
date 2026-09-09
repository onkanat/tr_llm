# Kristal–Vektörel Mimarisi: Kullanıcı ve Geliştirici Kılavuzu (USER_GUIDE.md)

Bu kılavuz, **Kristal–Vektörel Mimarisi** tabanlı Türkçe dil modelini kurmak, eğitmek, veri kümelerini derlemek, etkileşimli olarak test etmek ve Python API'si üzerinden projelere entegre etmek için kapsamlı yönergeler sunar.

---

## 📑 İçindekiler
1. [Gereksinimler ve Kurulum](#1-gereksinimler-ve-kurulum)
2. [Etkileşimli CLI ile Sohbet ve Analiz (`chat_prompt.py`)](#2-etkileşimli-cli-ile-sohbet-ve-analiz)
3. [Model Eğitimi ve İnce Ayar (`train.py`)](#3-model-eğitimi-ve-i̇nce-ayar)
4. [Veri Seti Oluşturma ve Derleme](#4-veri-seti-oluşturma-ve-derleme)
5. [Otonom Self-RAG ve Vektörel Bellek Navigasyonu](#5-otonom-self-rag-ve-vektörel-bellek-navigasyonu)
6. [Python API ile Programatik Kullanım](#6-python-api-ile-programatik-kullanım)
7. [Sorun Giderme ve Sık Karşılaşılan Sorunlar](#7-sorun-giderme)

---

## 1. Gereksinimler ve Kurulum

### Donanım ve Yazılım Gereksinimleri
- **İşletim Sistemi:** macOS (Apple Silicon M1/M2/M3/M4 önerilir) veya Linux
- **Python:** 3.10+ (Test edilen sürüm: Python 3.14)
- **PyTorch:** 2.0+

### Kurulum Adımları
```bash
# Depoyu klonlayın
git clone https://github.com/onkanat/tr_llm.git
cd tr_llm

# Sanal ortamı hazırlayın
python3 -m venv venv
source venv/bin/activate

# Gerekli kütüphaneleri yükleyin
pip install torch numpy pytest qdrant-client
```

### Test Doğrulaması
Kurulumun eksiksiz olduğunu doğrulamak için birim testlerini çalıştırın:
```bash
./venv/bin/pytest
```
*Tüm 67 testin (kök sözlüğü, durum makinesi, ses olayları, decompiler, merak motoru, tri-modal router) eksiksiz geçtiğinden emin olun.*

---

## 2. Etkileşimli CLI ile Sohbet ve Analiz

Modeli terminalden gerçek zamanlı olarak test etmek için `chat_prompt.py` betiği kullanılır. Betik renkli morfem görselleştirmesi, anlık akıtma (streaming) ve `MorphemeDecompiler` desteği sunar.

```bash
./venv/bin/python chat_prompt.py
```

### Modlar ve Kullanım
- **SFT Modu (Varsayılan):** Modelin talimat takip yeteneklerini sınar (kök bulma, çoğul, hâl, zaman tespiti).
- **Normal Mod:** Serbest metin üretimi ve diyalog yürütme. Modu değiştirmek için terminale `mode` yazıp `NORMAL` seçin.
- **Parametre Değiştirme:** Sıcaklık (`temp`) ve `top_k` değerlerini ayarlamak için `params` komutunu kullanın.

#### Örnek Sohbet Çıktısı:
```text
[Mod: NORMAL | Temp: 0.0 | Top-K: 5]
Girdiniz: Merhaba nasılsın?

 Girdi Morfem Akışı:
<BOS> <INPUT> merhaba nasıl PS_sın </INPUT> <OUTPUT>

Model Çıktısı:
 selam PLURAL harika bir gün geçir TENSE_PROG PERSON_1SG siz CASE_DAT nasıl yardım DERIV_CI olabil TENSE_AORIST PERSON_1SG <EOS>

Decompile Edilmiş Türkçe:
selam harika bir gün geçiriyorum size nasıl yardımcı olabilirim
```

---

## 3. Model Eğitimi ve İnce Ayar

Modelin ana eğitim döngüsü `train.py` betiği üzerinden yönetilir. Causal prompt masking sayesinde model sadece `<OUTPUT>` belirteçleri arasındaki hedefleri öğrenir.

### Komut Sözdizimi
```bash
./venv/bin/python train.py [SEÇENEKLER]
```

### Desteklenen Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--data <yol>` | `data/train.bin` | Eğitilecek ikili veri dosyasının yolu (`.meta.json` ile blok boyutu otomatik algılanır). |
| `--steps <sayı>` | `100` | Çalıştırılacak optimizasyon adım sayısı. |
| `--batch-size <sayı>` | `32` | Her adımdaki mini-batch boyutu. |
| `--device <cihaz>` | `cpu` | Çalıştırılacak cihaz (`cpu`, `mps`, `cuda`). macOS'ta büyük sözlüklerde deadlock'u önlemek için varsayılan CPU'dur. |
| `--from-scratch` | `False` | Mevcut ağırlıkları yüklemeden sıfırdan eğitim başlatır. |

### Örnek Çalıştırma Senaryoları

#### A. Dengeli Sohbet ve Self-RAG İnce Ayarı (Hızlı - ~10 dk)
```bash
./venv/bin/python -u train.py --data data/train_chat_balanced.bin --steps 150 --batch-size 32
```

#### B. Tam Temel Külliyat Eğitimi (87.629 kayıt, 11.2M token)
```bash
./venv/bin/python -u train.py --data data/train_deep_sft.bin --steps 300 --batch-size 32
```

---

## 4. Veri Seti Oluşturma ve Derleme

Projedeki veri setleri sentetik, deterministik ve pedagojik olarak yapılandırılmıştır.

### 1. Zenginleştirilmiş Sohbet Verisi Üretme
```bash
./venv/bin/python scripts/generate_deep_chat_dataset.py
```
*Çıktı: `data/pedagogy/chat_conversations.jsonl` (6.000 diyalog).*

### 2. Otonom Self-RAG Verisi Üretme
```bash
./venv/bin/python scripts/generate_interactive_rag_dataset.py
```
*Çıktı: `data/pedagogy/rag_interactive_dataset.jsonl` (6.599 Self-RAG görevi).*

### 3. İkili (Binary) Paketleme Adımları
- **Dengeli İnce Ayar Verisi Derleme:**
  ```bash
  ./venv/bin/python scripts/prepare_chat_balanced_dataset.py
  # Çıktı: data/train_chat_balanced.bin (3.25 MB)
  ```
- **Tüm Külliyatı (Ana Temel Veri) Derleme:**
  ```bash
  ./venv/bin/python scripts/prepare_deep_dataset.py
  # Çıktı: data/train_deep_sft.bin (21.39 MB)
  ```

---

## 5. Otonom Self-RAG ve Vektörel Bellek Navigasyonu

Kristal–Vektörel mimarisinde model, parametrik ezber yerine aktif bir gezgin (Vector Rover) gibi davranır:

### 1. Otonom Arama Sorgusu Üretimi (`<ARA>`)
Bilgi gerektiren sorularda model hafızayı taramak için sorgu üretir:
```bash
./venv/bin/python scripts/test_rag_interactive.py
```
*Örnek Çıktı:*
```text
📥 Soru: Gürgen ağacı ne tür ahşap işlerinde kullanılır?
🧠 Model: <ARA> canlandır ağacı sözlük tanım anlam </ARA>
```

### 2. Belgeden Kanıta Dayalı Cevaplama (`<BELGE>`)
Bellekten getirilen metin verildiğinde model halüsinasyon görmeden belgeden çıkarım yapar:
```text
📥 Girdi: <BELGE> Gürgen ağacı aşırı sert ve toktur; marangoz mengenelerinde kullanılır. </BELGE> Gürgen nerede kullanılır?
🤖 Model: aşırı serttir kesme tahtalarında ve marangoz mengenelerinde tercih edilir.
```

### 3. Uçtan Uca RAG Hattı Simülasyonu
```bash
./venv/bin/python src/rag/rag_pipeline.py
```

---

## 6. Python API ile Programatik Kullanım

Kristal derleyici ve dil modelini kendi Python kodunuzda doğrudan kullanabilirsiniz:

```python
import torch
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM

# 1. Derleyici ve Sözlüğü Hazırla
vocab = Vocabulary()
vocab.load('data/vocab.json')

lexicon = LexiconManager()
lexicon.load_from_tsv('data/lexicon/roots.tsv')
compiler = CrystalCompiler(lexicon, build_default_graph())
tokenizer = KristalTokenizer(compiler, vocab)
decompiler = MorphemeDecompiler(compiler, vocab)

# 2. Kelime Analizi (Compiling)
analiz = compiler.compile("kitaplarımızdan")
print("Morfem Vektörü:", analiz["token_vector"])
# Çıktı: ['kitap', 'PLURAL', 'POSS_1PL', 'CASE_ABL']

# 3. Yüzey Biçimine Çözümleme (Decompiling)
turkce_metin = decompiler.decompile_sentence("kitap PLURAL POSS_1PL CASE_ABL")
print("Türkçe Karşılık:", turkce_metin)
# Çıktı: kitaplarımızdan
```

---

## 7. Sorun Giderme ve Sık Karşılaşılan Sorunlar

### 1. macOS Metal (MPS) Üzerinde Eğitim Kilitlenmesi (Deadlock)
- **Belirti:** `train.py` veya `chat_prompt.py` çalışırken MPS üzerinde `AdamW` adımında işlemin askıda kalması.
- **Sebep:** PyTorch 2.x Metal backend'inin büyük kelime haznelerinde (~31.000 token) seyrek gradyan güncellemelerinde kilitlenmesi.
- **Çözüm:** Eğitim her zaman CPU modunda çalıştırılmalıdır (`train.py --device cpu`). Çıkarım (inference) MPS üzerinde sorunsuz çalışır.

### 2. Qdrant Bağlantı Hatası (`ConnectionRefusedError`)
- **Belirti:** `[VectorMemory]` başlatılırken `localhost:6333` adresine bağlanılamadı uyarısı.
- **Çözüm:** Sistemde gömülü bir **Resilient Fallback** mekanizması bulunmaktadır. Qdrant sunucusu çalışmıyorsa sistem otomatik olarak RAM içi (`:memory:`) vektör veritabanına geçer. Qdrant sunucusunu yerelde başlatmak isterseniz:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

### 3. Boyut Uyuşmazlığı (`RuntimeError: size mismatch for embedding`)
- **Belirti:** Yeni kontrol belirteçleri (`<ARA>`, `<BELGE>`) eklendikten sonra eski model ağırlıkları yüklenirken hata oluşması.
- **Çözüm:** Kodlarımızdaki `resize_state_dict` fonksiyonu eski katman ağırlıklarını koruyarak yeni token slotlarını otomatik olarak genişletir. `strict=False` ile yükleme yapılır.
