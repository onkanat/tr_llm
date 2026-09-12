# Kristal–Vektörel Mimarisi: Kullanıcı ve Geliştirici Kılavuzu (USER_GUIDE.md)

Bu kılavuz, **Kristal–Vektörel Mimarisi** tabanlı Türkçe dil modelini kurmak, eğitmek, veri kümelerini derlemek, etkileşimli olarak test etmek ve Python API'si üzerinden projelere entegre etmek için kapsamlı yönergeler sunar.

---

## 📑 İçindekiler
1. [Gereksinimler ve Kurulum](#1-gereksinimler-ve-kurulum)
2. [Etkileşimli CLI ile Sohbet ve Analiz (`chat_prompt.py`)](#2-etkileşimli-cli-ile-sohbet-ve-analiz)
3. [Model Eğitimi ve İnce Ayar (`train.py`)](#3-model-eğitimi-ve-i̇nce-ayar)
4. [Veri Seti Oluşturma ve Derleme](#4-veri-seti-oluşturma-ve-derleme)
5. [RAG Belge Yükleme ve Yönetim Aracı (`rag_tool.py`)](#5-rag-belge-yükleme-ve-yönetim-aracı-rag_toolpy)
6. [Otonom Self-RAG ve Vektörel Bellek Navigasyonu](#6-otonom-self-rag-ve-vektörel-bellek-navigasyonu)
7. [Agent Gateway ve Pedagojik Arena (`run_agent_arena.py`)](#7-agent-gateway-ve-pedagojik-arena)
8. [Python API ile Programatik Kullanım](#8-python-api-ile-programatik-kullanım)
9. [Sorun Giderme ve Sık Karşılaşılan Sorunlar](#9-sorun-giderme)

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
*Tüm 87 testin (kök sözlüğü, durum makinesi, ses olayları, decompiler, merak motoru, tri-modal router, agent gateway, pedagojik supervisor, UNK merakı ve sözlük cerrahisi) eksiksiz geçtiğinden emin olun.*

---

## 2. Etkileşimli CLI ile Sohbet ve Analiz

Modeli terminalden gerçek zamanlı olarak test etmek için `chat_prompt.py` betiği kullanılır. Betik renkli morfem görselleştirmesi, anlık akıtma (streaming) ve `MorphemeDecompiler` desteği sunar.

```bash
# Temel Lise Modeli ile başlatma:
./venv/bin/python chat_prompt.py

# Dikey Ahşap Uzmanlık Modülü ile başlatma:
./venv/bin/python chat_prompt.py --model data/kristal_carpenter_model.pt
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
| `--load-path <yol>` | `--save-path` | Eğitime devam edilecek temel model ağırlık dosyası (örn: `data/kristal_model.pt`). |
| `--save-path <yol>` | `data/kristal_model.pt` | Eğitilen yeni ağırlıkların kaydedileceği dosya yolu. |
| `--steps <sayı>` | `100` | Çalıştırılacak optimizasyon adım sayısı. |
| `--batch-size <sayı>` | `32` | Her adımdaki mini-batch boyutu. |
| `--device <cihaz>` | `cpu` | Çalıştırılacak cihaz (`cpu`, `mps`, `cuda`). macOS'ta büyük sözlüklerde deadlock'u önlemek için varsayılan CPU'dur. |
| `--from-scratch` | `False` | Mevcut ağırlıkları yüklemeden sıfırdan eğitim başlatır. |

### Örnek Çalıştırma Senaryoları

#### A. Evre 1-3: Pedagoji ve Temel Lise Eğitimi (46.412 kayıt, 1.84M token)
```bash
./venv/bin/python -u train.py --data data/train_pedagogy_highschool.bin --steps 300 --batch-size 16 --device cpu
```

#### B. Evre 4: Dikey Ahşap Uzmanlık Modülü (12.775 kayıt, 793K token)
```bash
./venv/bin/python -u train.py --data data/train_carpenter_specialization.bin --load-path data/kristal_model.pt --save-path data/kristal_carpenter_model.pt --steps 150 --batch-size 16 --device cpu
```

#### C. Dengeli Sohbet ve Self-RAG İnce Ayarı
```bash
./venv/bin/python -u train.py --data data/train_chat_balanced.bin --steps 150 --batch-size 32
```

#### D. 1931 Türk Tarihi Müfredatı Sıralı Eğitimi (SFT ➔ Chat ➔ DPO)
```bash
# 1. Aşama: SFT (Supervised Fine-Tuning) - Lise Temeli & Tarih Külliyatı (MPS Hızlandırmalı)
./venv/bin/python train.py --data data/train_pedagogy_highschool.bin --steps 200 --batch-size 16 --device mps
cp data/kristal_model.pt data/kristal_model_sft.pt

# 2. Aşama: Chat (Causal Maskeli Çok Turlu Tarih & Sohbet İnce Ayarı)
./venv/bin/python scripts/train_chat_sft.py --data data/train_chat_balanced.bin --steps 200 --batch-size 16 --device mps

# 3. Aşama: DPO (Direct Preference Optimization - Bradley-Terry Sigmoid Kaybı)
./venv/bin/python train_dpo.py --data data/pedagogy/turk_tarihi_dpo_tokenized.jsonl --steps 100 --batch-size 4 --beta 0.1
```

---

## 4. Veri Seti Oluşturma ve Derleme

Projedeki veri setleri sentetik, deterministik ve pedagojik hiyerarşiye göre yapılandırılmıştır:

### 1. 1931 Türk Tarihi Külliyatı Hazırlığı ve Entegrasyonu
Hugging Face üzerindeki [`onkanat/turk-tarihi-1931-sft-dpo`](https://huggingface.co/datasets/onkanat/turk-tarihi-1931-sft-dpo) veri setini (6.477 SFT, 6.071 Chat, 406 DPO) işleyip Qdrant belleğe indekslemek için:
```bash
./venv/bin/python scripts/prepare_turk_tarihi_pipeline.py
```

### 2. Temel Lise ve Pedagoji Verisi Derleme (Evre 1-3)
```bash
./venv/bin/python scripts/prepare_pedagogy_high_school_dataset.py
```
*Çıktı: `data/train_pedagogy_highschool.bin` (4.47 MB, 52.739 örnek, 2.34M token). Bebeklik ontolojisi, ebeveynlik morfolojisi, lise müfredatı, edebiyat & şiir, TDK GTS sözlüğü ve 1931 Tarih SFT verilerini birleştirir.*

### 3. Dikey Ahşap Uzmanlık Verisi Derleme (Evre 4)
```bash
./venv/bin/python scripts/prepare_carpenter_specialization_dataset.py
```
*Çıktı: `data/train_carpenter_specialization.bin` (1.51 MB, 12.775 örnek, 793K token). Ahşap ve marangozluk teknolojisi ile unutmayı önleyici morfolojik çıpaları harmanlar.*

### 4. Zenginleştirilmiş Sohbet ve Self-RAG Verisi Derleme
```bash
./venv/bin/python scripts/prepare_chat_balanced_dataset.py
```
*Çıktı: `data/train_chat_balanced.bin` (5.34 MB, 21.875 örnek, 2.80M token).*

---

## 5. RAG Belge Yükleme ve Yönetim Aracı (`rag_tool.py`)

Test grubunun (pilot kullanıcıların) kendi dökümanlarını (TXT, MD, JSON, CSV, PDF) sisteme kolayca yükleyebilmesi ve arama testi yapabilmesi için `rag_tool.py` aracı geliştirilmiştir. Yüklenen belgeler yerel kalıcı veritabanında (`data/qdrant_db`) saklanır ve `chat_prompt.py` üzerinden hemen kullanılabilir.

### 🌟 1. Etkileşimli Menü Modu (Tavsiye Edilen)
Argümansız çalıştırıldığında test kullanıcılarına rehberlik eden renkli terminal sihirbazı açılır:
```bash
./venv/bin/python rag_tool.py
```
*Açılan menüden tek tuşla dosya/klasör yükleme, arama testi ve bellek durumu görüntülenebilir.*

### ⚡ 2. Komut Satırı (CLI) Kullanımı

#### A. Tek Bir Dosya Yükleme (.txt, .md, .json, .csv, .pdf)
```bash
./venv/bin/python rag_tool.py add --file docs/ahsap_rehberi.md --title "Ahşap Rehberi"
```

#### B. Bir Klasördeki Tüm Belgeleri Toplu Yükleme
```bash
./venv/bin/python rag_tool.py add --dir docs/
```

#### C. Doğrudan Metin / Not Ekleme
```bash
./venv/bin/python rag_tool.py add --text "Zıvana geçme mukavemeti en yüksek birleştirmedir." --title "Zıvana Kuralı"
```

#### D. Yüklenen Belgeleri Arama ve RRF Skorlama Testi
```bash
./venv/bin/python rag_tool.py search "ahşap nem oranı kaç olmalı"
```

#### E. Bellek Durumu ve Kayıtlı Belgeleri Listeleme
```bash
./venv/bin/python rag_tool.py status
./venv/bin/python rag_tool.py list
```

#### F. Belleği Sıfırlama / Temizleme
```bash
./venv/bin/python rag_tool.py reset
```

---

## 6. Otonom Self-RAG ve Vektörel Bellek Navigasyonu

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

## 7. Agent Gateway ve Pedagojik Arena (`run_agent_arena.py`)

Büyük dil modellerinin (Antigravity Agent'ları, Google Gemini API, Ollama vb.) küçük KristalLM modelini otonom pedagojik denetimden geçirmesi, eksik/hatalı bilgileri RAG belleğine (`kristal_bellek`, `simulasyon_bellek`) otomatik enjekte etmesi ve modeli sürekli öğrenme döngüsüne (Karpathy Continuous Learning Loop) sokması için **Agent Gateway** ve **Pedagogical Supervisor** mimarisi geliştirilmiştir.

### 🌟 1. Etkileşimli Arena Testi (`run_agent_arena.py`)
Küçük modeli çoklu branşta (1931 Türk Tarihi, Edebiyat & Şiir, Lise Fen/Sosyal, Marangozluk, Morfoloji) doğrudan sınava tabi tutmak ve anlık epistemik durumunu görmek için:
```bash
# 1931 Türk Tarihi sınavı:
./venv/bin/python scripts/run_agent_arena.py --domain history_1931 --rounds 3 --device cpu

# Ahşap ve marangozluk sınavı:
./venv/bin/python scripts/run_agent_arena.py --domain carpenter --rounds 3 --device cpu
```
Önemli seçenekler:
- `--domain <history_1931|carpenter|pedagogy|literary|highschool|arena_mix>` (Sınav alanı)
- `--rounds <sayı>` (Diyalog tur sayısı)
- `--retrain-threshold 5` (future_train kütüğü için otomatik eğitim tetik eşiği)
- `--auto-retrain` (future_train_vector.jsonl eşiği aşıldığında otomatik eğitimi tetikler)
- `--device <cpu|mps>` (Hesaplama donanımı)

### 🚪 2. HTTP REST Gateway Sunucusu
Dış ajanların HTTP üzerinden modeli sorgulaması ve belleğe bilgi beslemesi için bağımsız REST API sunucusu çalıştırılabilir:
```bash
./venv/bin/python -m src.gateway.agent_gateway --host 127.0.0.1 --port 8080
```
veya CLI sohbeti içerisinden ağ geçidini ayağa kaldırmak için:
```bash
./venv/bin/python chat_prompt.py --gateway --port 8080
```

#### REST API Uç Noktaları:
| Uç Nokta | Metod | Açıklama |
|---|---|---|
| `/api/query` | POST | Modele soru sorar, epistemik merak ($H(z)$) ve RAG yanıtını döndürür. |
| `/api/inject` | POST | `kristal_bellek` veya `simulasyon_bellek` koleksiyonuna doğrudan bilgi dokümanı ekler. |
| `/api/check` | POST | Modelin verilen sorgu ile enjekte edilen bilgiyi $\ge 0.85$ alaka skoruyla bulup bulamadığını denetler. |
| `/api/backlog` | GET | `future_train_vector.jsonl` içindeki kuyrukta bekleyen yeniden eğitim örneklerini listeler. |
| `/api/status` | GET | Kapının ve vektör belleklerinin genel sağlık durumunu döner. |

#### Örnek REST İstekleri (cURL):
```bash
# Model Sorgulama
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ahşapta zıvana birleştirme nedir?"}'

# Belleğe Bilgi Enjeksiyonu
curl -X POST http://localhost:8080/api/inject \
  -H "Content-Type: application/json" \
  -d '{"collection": "simulasyon_bellek", "text": "Zıvana geçme yüksek mukavemetli ahşap birleştirmedir.", "metadata": {"source": "pedagoji"}}'

# Bulunabilirlik Denetimi (Threshold: 0.85)
curl -X POST http://localhost:8080/api/check \
  -H "Content-Type: application/json" \
  -d '{"query": "ahşap zıvana birleştirme mukavemet", "expected_concept": "Zıvana geçme", "threshold": 0.85}'
```

### 🧠 3. CLI İçinden Arena Komutu
Etkileşimli `chat_prompt.py` oturumu açıkken `arena` yazarak doğrudan pedagojik denetim çalıştırılabilir:
```text
Girdiniz: arena
>>> [ARENA] Pedagojik denetim döngüsü başlatılıyor...
>>> Soru: Fotosentez nerede gerçekleşir?
...
```

### 🔄 4. Sürekli Yeniden Eğitim Hattı (Retrain Pipeline)
`future_train_vector.jsonl` dosyasında biriken (modelin merak ettiği ve RAG'ın $\ge 0.85$ alaka ile getirdiği) veriler şu komutla yeni eğitim ağırlıklarına dönüştürülür:
```bash
./venv/bin/python src/gateway/retrain_pipeline.py --steps 50 --batch-size 16
```

---

## 7.1 CoT Kasası ve Morfemik Akıl Yürütme (CoT Vault & Reasoning Isolation)

Büyük öğretmen modeller (Google Gemini 2.5 Flash, Ollama gpt-oss:20b) tarafından üretilen iç düşünce adımları (Chain-of-Thought), modelin felsefi ve pedagojik kalitesi açısından son derece değerlidir; ancak bu adımlar deklaratif arama belleğine (`kristal_bellek`) sızdığında sahte eşleşmelere yol açar.

Bu sorunu çözmek için **Çift Çıktılı Ayrıştırma ve İzolasyon Mimarisi** uygulanmıştır:

```text
Öğretmen Yanıtı (Gemini / Ollama)
           │
           ▼
extract_cot_and_card()
     ├──> <DUSUNCE>      ──> data/pedagogy/cot_vault.jsonl & Qdrant: muhakeme_bellek
     └──> <BILGI_KARTI>  ──> Qdrant: kristal_bellek (Tamamen saf deklaratif Türkçe)
```

### 1. Akıl Yürütme Verisini İkili Eğitime Derleme (Evre 5 Hazırlığı)
Kasada biriken düşünce adımlarını modelin içsel akıl yürütmesini (`<DUSUNCE> ... </DUSUNCE>`) öğrenmesi için `train_reasoning_cot.bin` ikili formatına derleyebilirsiniz:
```bash
./venv/bin/python scripts/prepare_reasoning_dataset.py \
  --vault data/pedagogy/cot_vault.jsonl \
  --output data/train_reasoning_cot.bin \
  --min-len 15
```

### 2. Vektörel Bellek Arındırma ve Denetim (Sanitization)
Bellekteki CoT sızıntılarını temizlemek ve saf bilgi kartlarını yeniden indekslemek için:
```bash
./venv/bin/python scripts/sanitize_vector_memory.py
```

---

## 8. Python API ile Programatik Kullanım

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

## 9. Sorun Giderme ve Sık Karşılaşılan Sorunlar

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
