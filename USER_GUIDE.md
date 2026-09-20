# Kristal–Vektörel Mimarisi: Kullanıcı ve Geliştirici Kılavuzu (USER_GUIDE.md)

Bu kılavuz, **Kristal–Vektörel Mimarisi** tabanlı Türkçe dil modelini kurmak, eğitmek, veri kümelerini derlemek, etkileşimli olarak test etmek ve Python API'si üzerinden projelere entegre etmek için kapsamlı yönergeler sunar.

> [!WARNING]
> **TARİHSEL BÖLÜMLER (damgalı ölçüm: 20 Eyl 2026).** Bu kılavuzun bazı bölümleri
> `data/kristal_*.pt` yollarına dayanan komutlar içerir. **Kristal checkpoint zinciri
> 18 Eyl 2026'da operatör kararıyla silinmiştir**; o dosyalar **diskte de git'te de
> yoktur** (ölçüldü). Bugün `data/` altında yalnız `data/anka_a1.pt` ve
> `data/anka_a1r.pt` bulunur. Etkilenen bölümler: **§2** (CLI model yolları),
> **§3** (`--load-path` / `--save-path`), **§4 · Adım B1.5** (checkpoint adı).
> Bu komutlar **o dönemin kaydıdır ve bugün çalışmaz**; silinmiş içerik burada
> **korunmuştur** (tarihsel kayıt), ama **güncel karşılıkları yazılmamıştır** —
> çalışan bir güncel komut bu belgede **doğrulanmadığı için iddia edilmiyor**.
> Model adı artık **Anka**'dır; mimari ve sınıf adları (`Kristal*`) değişmemiştir.

---

## 🅰️ Anka A1-r — ÖLÇÜLMÜŞ çıkarım komutu (taban model, devam üretimi)

> **Damgalı ölçüm: 20 Eyl 2026.** Bu bölümdeki her komut **bu oturumda koşuldu** ve
> çıktısı aşağıya **birebir** yapıştırıldı. Ölçülmemiş hiçbir komut yazılmadı.
> Bu bölüm, yukarıdaki TARİHSEL banner'ın *"doğrulanmadığı için iddia edilmiyor"*
> cümlesini **kısmen** günceller: **banner metni kendi damgasıyla değiştirilmeden
> korunmuştur**, ve §2 / §3 / §4 · Adım B1.5'teki tarihsel komutlar **hâlâ çalışmaz**
> ve **hâlâ değiştirilmemiştir**.

### Gereken üç dosya — ve neden bu üçü

Eğitim külliyatı **hangi sözlükle** derlendiyse, çıkarım da **o sözlükle** kurulmalıdır:

| Rol | Yol | Tam sha256 (ölçüldü) |
|---|---|---|
| Sözlük | `data/rebuild/vocab_anka_r1_33114.json` (**33.114** giriş) | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` |
| Kök sözlüğü | `data/lexicon/roots_anka_r1.tsv` (52.582 satır) | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` |
| Checkpoint | `data/anka_a1r.pt` (embedding **33114 × 768**) | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |

> **`data/vocab.json` (31.357) KULLANILMAZ.** Eşleşmezse koşum **`RuntimeError` ile
> sesli durur** — sessiz kırpma yoktur. Ölçülen hata:
> `size mismatch for embedding.embedding.weight: ... [33114, 768] ... current model is [31357, 768]`.
> `strict=False` bunu **yutmaz** (o yalnız eksik/fazla *anahtarı* yutar, *şekli* değil).

### Komut

```bash
venv/bin/python scratch/anka_r7_cikarim_sondasi.py
```

Bu sonda fail-closed'dır (`rc=0` = kapılar geçti, `rc=2` = kapı düştü) ve iki dalı da
sınar. Eşdeğer en kısa Python:

```python
from scripts.train_step_demo import KristalLM
from src.llm.tokenizer import Vocabulary, KristalTokenizer
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
import torch

v = Vocabulary(); v.load("data/rebuild/vocab_anka_r1_33114.json")          # 33.114
lex = LexiconManager(); lex.load_from_tsv("data/lexicon/roots_anka_r1.tsv")
tok = KristalTokenizer(CrystalCompiler(lex, build_default_graph()), v)

model = KristalLM(vocab_size=len(v.stoi), n_embd=768, vocab=v,
                  block_size=4096, n_layer=6, n_head=6)
sd = torch.load("data/anka_a1r.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)     # TEMİZ — anahtar atmaya GEREK YOK
model.eval()

ids = tok.encode("Yarın okula")[:-1]       # ← <EOS> KIRPILIR (aşağıdaki tuzağa bakın)
with torch.no_grad():
    logits, _ = model(torch.tensor([ids], dtype=torch.long))
    p = torch.softmax(logits[0, -1, :], dim=-1)
    top = torch.topk(p, 5)
    print([(v.decode(i.item()), round(pr.item(), 4)) for pr, i in zip(*top)])
```

### Ölçülen çıktı (birebir)

| Girdi (gövde) | Kırpılan son token | İlk 5 devam |
|---|---|---|
| `Yarın okula` | `CASE_DAT` | `<PROPER_NOUN>`=0,0883 · `ait`=0,0696 · `(`=0,0638 · `bağ`=0,0619 · `,`=0,0510 |
| `Akmayan su kımıldanmayan yer` | `yer` | `CASE_LOC`=0,1642 · `DERIV_lI`=0,1606 · `PLURAL`=0,1274 · `COPULA_AORIST`=0,0774 · `al`=0,0599 |
| `Demirkır güney tepelerinin duldalarına` | `CASE_DAT_N` | `bağ`=0,1821 · `ait`=0,1640 · `göre`=0,1411 · `doğru`=0,0265 · `sahip`=0,0246 |

### ⚠️ Tuzak 1 — `encode()` sona `<EOS>` ekler, o kırpılmazsa ölçü yanlış okunur

`KristalTokenizer.encode()` dizinin **sonuna `<EOS>` (id 3)** koyar. `<EOS>`
kırpılmadan "sırada ne var?" diye sorulursa model `<BOS>`'u **0,9971** ile verir.
Bu **dejenerasyon değil, doğru davranıştır**: belge bitmiştir, yenisi başlıyor.
Ölçülen kontrast: `Yarın okula` + `<EOS>` ⇒ `<BOS>`=0,9971 · `DERIV_CI`=0,0011 · `ver`=0,0007.

### ⚠️ Tuzak 2 — bu bir TABAN modeldir; **talimat takip etmez**

A1-r **düz-metin ön-eğitimidir** (`--pretrain`), SFT değil. Ölçüldü:
`data/anka_a1r_pretrain.bin` 100.000.000 jetonun içinde `<OUTPUT>` **yalnız 2 kez**
(oran **2,0 × 10⁻⁸**). Yani `<INSTRUCTION> … <OUTPUT>` zarfı bu modelin
**dağılımında yoktur**; zarfı verip cevap beklemek **dağılım dışı** bir istektir.
Bu checkpoint **metni sürdürür**; **soru cevaplamaz**.

### Bugün çalışmayan yollar (ölçüldü — 20 Eyl 2026 damgası)

| Betik | Neden çalışmaz |
|---|---|
| `test_model.py` | `data/vocab.json` (31.357) sabit-kodlu ⇒ şekil uyuşmazlığı; ayrıca `data/kristal_model.pt` (silinmiş) okur ve dosya yoksa **sessizce `return` eder** |
| `scripts/run_goal_pipeline.py` | Dört yol parametresi (`--base-model`, `--carpenter-model`, `--sft-model`, `--vocab`) **zorunludur**, varsayılanı yoktur; verilmezse hiçbir işe başlamadan durur. DPO aşamasının sözlük kısıtı **T-0089'da kapandı** (aşağıdaki nota bakın) |
| `scripts/run_agent_arena.py` | `--model` ve `--vocab` **zorunludur**, varsayılanı yoktur. Uçtan uca koşum ölçülmedi (qdrant bağımlılığı + ağır checkpoint yüklemesi) |
| `train_dpo.py` | `--vocab`, `--ref-model`, `--active-model` **zorunludur**, varsayılanı yoktur; üçünden biri eksikse ya da verilen yol yoksa **hiçbir işe başlamadan `rc=2` ile durur**. `--data`'nın varsayılanı (`data/pedagogy/dpo_all_tokenized.jsonl`) vardır ama dosya yoksa yine durur. Uçtan uca **eğitim** koşumu ölçülmedi (kapı ölçüldü, koşum değil) |

**Onarıldı (T-0089 · damgalı ölçüm: 20 Eyl 2026):** `train_dpo.py` artık sözlüğü ve iki model yolunu **açıkça** alır. Ölçülen önce/sonra: argümansız koşum **`rc=0`** (yani "Hata: … bulunamadı!" basıp **başarı** dönüyordu) → **`rc=2`**; bayrak yokluğu ile *yolu olmayan* girdi ayrı ayrı durur; sözlük↔checkpoint satır sayısı uyuşmazlığı (31.357 ↔ 33.114, fark **1.757**) eyleme dönük mesajla durur. Satır sayısı eşitse kapı **geçer** (pozitif kontrol). `scripts/run_goal_pipeline.py` 4. aşaması da sözlüğü artık geçirir.

**Onarıldı (T-0087, ölçüldü):** `chat_prompt.py` artık `--model` **ve** `--vocab` bayraklarının ikisini de kabul eder; ikisi de verilmezse `sys.exit(2)` ile sesli durur, yani sözlük **değiştirilebilir**. Kalan ölçülmüş kusur: sözlük yolu *verilip de dosya bulunamazsa* `return` ile çıkar (rc=0) — o dal sesli değildir.

---

## 📑 İçindekiler
0. [Anka A1-r — ölçülmüş çıkarım komutu](#️-anka-a1-r--ölçülmüş-çıkarım-komutu-taban-model-devam-üretimi)
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
*Tüm 99 testin (kök sözlüğü, durum makinesi, ses olayları, fonetik sentez ve y türemesi, decompiler, merak motoru, tri-modal router, agent gateway, pedagojik supervisor, UNK merakı, sözlük cerrahisi ve morfoloji regresyon altın paketi) eksiksiz geçtiğinden emin olun.*

---

## 2. Etkileşimli CLI ile Sohbet ve Analiz

Modeli terminalden gerçek zamanlı olarak test etmek için `chat_prompt.py` betiği kullanılır. Betik renkli morfem görselleştirmesi, anlık akıtma (streaming), `MorphemeDecompiler` ve otomatik model yönlendirme desteği sunar.

```bash
# Temel Lise, Tarih ve Genel Model ile başlatma:
./venv/bin/python chat_prompt.py

# Dikey Ahşap Uzmanlık Modeli ile başlatma:
./venv/bin/python chat_prompt.py --model data/kristal_carpenter_model.pt
```

> **Önemli (Akıllı Model Yönlendirme):** CLI arayüzü, SFT modunda seçilen personaya göre modeli dinamik olarak devreye alır. Örneğin Ahşap Uzmanı (Seçenek 6) seçildiğinde otomatik olarak `kristal_carpenter_model.pt` yüklenir; Tarih veya Genel persona seçildiğinde `kristal_model.pt` devreye girer. Ayrıca CLI içerisinde doğrudan `model` yazılarak dilediğiniz checkpoint seçilebilir.

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

Modelin ana eğitim döngüsü `train.py` betiği üzerinden yönetilir. Causal prompt masking sayesinde model yalnızca `<OUTPUT>` belirteçlerinden sonraki ve `<EOS>`'tan önceki hedefleri öğrenir; hizalama dolgusu (`<PAD>`) ise **16 Eyl 2026'dan beri varsayılan olarak kayıptan maskelenir** (`--no-pad-mask` ile kapatılabilir). Maskeleme kayıp **ölçeğini** değiştirdiği için maskeleme öncesi/sonrası kayıp değerleri kıyaslanmamalıdır. Ayrıca donmuş `data/*.pt` hedefine yazmak `--allow-frozen-write` onayı ister.

### Komut Sözdizimi
```bash
./venv/bin/python train.py [SEÇENEKLER]
```

### Desteklenen Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--data <yol>` | `data/train.bin` | Eğitilecek ikili veri dosyasının yolu (`.meta.json` ile blok boyutu otomatik algılanır). |
| `--load-path <yol>` | `--save-path` | Eğitime devam edilecek temel model ağırlık dosyası (örn: `data/anka_a1r.pt`). Verilmezse `--save-path`'e eşitlenir. |
| `--save-path <yol>` | **ZORUNLU** (varsayılan **yok**) | Eğitilen yeni ağırlıkların kaydedileceği dosya yolu. Verilmezse koşum **başlamadan sesli olarak durur** (`RuntimeError`). |
| `--steps <sayı>` | `100` | Çalıştırılacak optimizasyon adım sayısı. |
| `--batch-size <sayı>` | `32` | Her adımdaki mini-batch boyutu. |
| `--device <cihaz>` | `cpu` | Çalıştırılacak cihaz (`cpu`, `mps`, `cuda`). macOS'ta büyük sözlüklerde deadlock'u önlemek için varsayılan CPU'dur. |
| `--from-scratch` | `False` | Mevcut ağırlıkları yüklemeden sıfırdan eğitim başlatır. |

> **Not (damgalı ölçüm: 20 Eyl 2026).** Yukarıdaki tablo, `train.py`'nin **gerçek**
> varsayılanlarını anlatır ve bu bakımdan **doğrudur** — ancak `--save-path`'in
> varsayılanı hâlâ **silinmiş** `data/kristal_model.pt` adını taşımaktadır.
> Yani buradaki kusur **belgede değil, kaynak kodun varsayılanındadır**; belge kodu
> doğru anlattığı için "düzeltilmemiştir". Pratik sonuç: `--save-path`'i **açıkça
> verin**; vermezseniz koşum o ada yazmaya çalışır ve `data/*.pt` donmuş desene
> düştüğü için `check_frozen_save_path` **sesli** durur (sessiz değil).
>
> **GÜNCELLEME (T-0085 · damgalı ölçüm: 20 Eyl 2026) — yukarıdaki not artık
> TARİHSELDİR; silinmedi, kendi damgasıyla korundu.** Ölçüldü ve düzeltildi:
> `train.py`'deki silinmiş-zincir varsayılanı **kaldırıldı**; `--save-path` artık
> **zorunludur**. Varsayılanı başka bir ada taşımak **reddedildi** (uydurma ad,
> soyağacı belirsiz bir hedef yaratırdı); bunun yerine yol yokluğunda koşum
> `RuntimeError` ile **durur**. Ölçülen iki dal:
>
> | Koşum | Ölçülen sonuç |
> |---|---|
> | `--save-path` **yok** | `RuntimeError: DURDURULDU: --save-path verilmedi…` · `rc=1` · **hiçbir dosya yazılmadı** |
> | `--save-path data/zzz_kapi_testi.pt` (donmuş desen, `--allow-frozen-write` yok) | `RuntimeError: Donmuş yola yazma engellendi…` · `rc=1` · argüman kapısı **0 kez** ateşledi · dosya **yazılmadı** |
>
> İkinci satır **pozitif kontroldür**: argüman kapısı yalnız yoklukta ateşler,
> varlığında sonraki kapıya (donmuş yol koruması) ilerler.
>
> **Sayaç ölçümü — harf duyarlılığı belirtilmelidir:** `train.py`'de **küçük harf**
> `"kristal"` geçen satır sayısı **0**'dır; **harf-duyarsız** arama ise **4** satır
> bulur. O 4 satır silinen zincir değil, **korunması gereken mimari sınıf adlarıdır**
> (`KristalLM`, `KristalDataset` — [[mimari-korunur-kristallm]]). İki sayı
> çelişmiyor; **ölçütleri farklı**. Emekli notu silinen yolu **adıyla anmadığı**
> için de sayaç şişmemiştir ([[emekli-notu-sayaci-sisirir]]).

### Örnek Çalıştırma Senaryoları

> **Tavsiye Edilen Güncel Eğitim:** Mevcut v1.8 sürümünde model; şablon kopyalamalarını önleyen kanonik B1.5 boru hattı (`scripts/train_step_b1_5_rigorous.py`) ile eğitilmiştir (Bkz. [Bölüm 4.5](#5-adım-b15-cevap-düzeyi-ölçümlü-bölme-titiz-eğitim-ve-eşleştirilmiş-mcnemar-doğrulaması)). Aşağıdaki senaryolar, projenin modüler `.bin` kütükleri üzerindeki erken aşama deneysel eğitimlerini gösterir:

#### A. Evre 1-3: Pedagoji ve Temel Lise Eğitimi (Erken Aşama)
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

> **TARİHSEL Not (damgalı ölçüm: 20 Eyl 2026).** Yukarıdaki D bloğu **o dönemin kaydıdır** ve
> silinmiş Kristal zincirini anar (`data/kristal_model.pt`). Ayrıca **T-0089'dan sonra
> 3. aşamanın komutu bu hâliyle ÇALIŞMAZ**: `train_dpo.py` artık `--vocab`, `--ref-model`
> ve `--active-model` bayraklarını **zorunlu** tutar ve eksikse `rc=2` ile durur. Blok
> **silinmedi**, kendi damgasıyla korundu; güncel karşılığı **doğrulanmadığı için
> yazılmamıştır** (bkz. yukarıdaki "Bugün çalışmayan yollar" tablosu).

---

## 4. Veri Seti Oluşturma ve Derleme

Projedeki veri setleri sentetik, deterministik ve pedagojik hiyerarşiye göre yapılandırılmıştır:

> **Mimari Not (v1.8 Geçişi):** Aşağıdaki 1–4 arası veri derleyiciler (`.bin`), projenin önceki gelişim evrelerine aittir. Titiz denetimlerde tespit edilen şablon tekrarlarını ve veri sızıntılarını önlemek amacıyla, güncel ve resmi eğitim kümesi **Bölüm 4.5'te detaylandırılan Adım B1.5 Kanonik Bölme (`data/b1_5_splits/`, 16.253 kayıt)** kümesidir.

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

### 5. Adım B1.5: Cevap Düzeyi Ölçümlü Bölme, Titiz Eğitim ve Eşleştirilmiş McNemar Doğrulaması

Adım B1.5, modelin girdi istemine gerçek anlamda koşullanıp koşullanmadığını test etmek için **soru düzeyinde sıfır sızıntılı 3 yollu bölme**, **train-only sözcüksel taban (TF-IDF)** ve **diskte sabitlenmiş aday kümeleri üzerinden eşleştirilmiş McNemar testi** uygular.

#### Adım 1: Cevap Düzeyi Ölçümlü Veri Bölme ve Sabit Aday Kümeleri Üretimi
Normalize soru metni (`clean_q`) ve normalize cevap metniyle kümeleme yapılarak soru düzeyinde %0 sızıntılı (`Train ∩ Test = 0`, %0,0; cevap düzeyi sızıntı: %43,4 / 704 kayıt) Train, Val ve Test kümeleri oluşturulur:
```bash
./venv/bin/python scripts/prepare_b1_5_datasets.py
```
*Çıktılar:*
- `data/b1_5_splits/train.jsonl` (13.009 kayıt)
- `data/b1_5_splits/val.jsonl` (1.622 kayıt)
- `data/b1_5_splits/test.jsonl` (1.622 kayıt)
- `data/b1_5_splits/test_candidates_hard.jsonl` ($N=660$, en benzer morfem Jaccard zor-negatif adaylar)
- `data/b1_5_splits/test_candidates_random.jsonl` ($N=660$, rastgele çeldirici adaylar)

#### Adım 2: Sözcüksel Taban Modelinin Eğitilmesi (Null Hipotezi)
Test verisinden sıfır sızıntı garantisi için TF-IDF sözcüksel modeli **yalnızca** `train.jsonl` üzerinden eğitilir ve diskteki test adaylarını puanlar:
```bash
./venv/bin/python scripts/evaluate_lexical_baseline.py
```
*Çıktılar:* `lexical_preds_hard.json` ve `lexical_preds_random.json`.

#### Adım 3: Hızlı Tensör Önbellekleme ile MPS Model Eğitimi
Blok boyutu 128 ve önceden hesaplanan `sign_mask` tensörleri ile MPS donanımında monoton kayıp düşüşüyle 3 epoch eğitim gerçekleştirilir:
```bash
./venv/bin/python scripts/train_step_b1_5_rigorous.py
```
*Eğitim İlerlemesi:* Val Loss $2.0974 \to 1.6339 \to 1.5589 \to 1.5525$ (PPL: 4.72). En düşük doğrulama kaybını veren Epoch 3 checkpoint'i `data/kristal_b1_5_best.pt` olarak kilitlenir.

#### Adım 4: Eşleştirilmiş McNemar Testi ve Katman 4 Toplu Üretim Denetimi
Model ağırlıklarının diskteki sabit adayları çözmesi ve sonuçların sözcüksel tabanla eşleştirilerek McNemar süreklilik düzeltmeli $\chi^2$ ve $p$-değerinin hesaplanması:
```bash
./venv/bin/python scripts/evaluate_b1_5_rigorous.py
```
*Bu betik ayrıca hiç görülmemiş $N=100$ held-out test örneğinde toplu üretim yaparak Ezber Oranı ($< \%10$), Tutarsızlık Oranı ($< \%5$) ve ortalama ROUGE-L ($\ge 0.35$) metriklerini `data/b1_5_splits/final_evaluation_report.json` dosyasına kaydeder.*

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
./venv/bin/python scripts/run_agent_arena.py --domain history_1931 --rounds 3 --device cpu \
    --model data/anka_a1r.pt --vocab data/rebuild/vocab_anka_r1_33114.json

# Ahşap ve marangozluk sınavı:
./venv/bin/python scripts/run_agent_arena.py --domain carpenter --rounds 3 --device cpu \
    --model data/anka_a1r.pt --vocab data/rebuild/vocab_anka_r1_33114.json
```
Önemli seçenekler:
- `--model <checkpoint.pt>` (**ZORUNLU**; varsayılanı yoktur — eskiden silinmiş bir checkpoint'i gösteriyordu)
- `--vocab <sözlük.json>` (**ZORUNLU**; checkpoint satır sayısıyla **aynı** olmalıdır, yoksa sessiz kırpma riski doğar)
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
vocab.load('data/rebuild/vocab_base_32852.json')   # kanonik taban sözlüğü (16 Eyl 2026); eski betiklerde data/vocab.json (31.357) hâlâ geçebilir

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

### 1. macOS Metal (MPS) Üzerinde Eğitim Hızı ve Kilitlenmeler
- **Belirti:** `train.py` çalışırken MPS üzerinde `sign_mask` CPU-GPU aktarımı veya seyrek gradyanlar nedeniyle adım başına 4.7 saniyeye varan yavaşlama.
- **Çözüm:** Adım B1.5 ile gelen `scripts/train_step_b1_5_rigorous.py` mimarisi kullanılır. Bu mimaride veri kümesi (`train_fast_ds.pt`) ve `sign_mask` tensörleri önceden hesaplanıp MPS belleğine kilitlenir; adım süresi 0.58 saniyeye düşer.
- **ÖLÇÜLMÜŞ EK NEDEN (16 Eyl 2026):** Aynı makinede **GPU tüketen başka iş** (tarayıcıda video/YouTube) açıkken adım süresi **0,43 → 3,45 sn/adım**'a çıktı ve 75 saniyelik tekil takılmalar görüldü (T-0052 ölçümü; neden birinci elden teyit edildi). **Koşum sırasında makinede GPU'ya başka iş bindirmeyin** ve adım süresini koşumun **başında ve sonunda** ölçün: 2-3× sapma model/kod değil **yük** sinyalidir.

### 2. Qdrant Bağlantı Hatası (`ConnectionRefusedError`)
- **Belirti:** `[VectorMemory]` başlatılırken `localhost:6333` adresine bağlanılamadı uyarısı.
- **Çözüm:** Sistemde gömülü bir **Resilient Fallback** mekanizması bulunmaktadır. Qdrant sunucusu çalışmıyorsa sistem otomatik olarak RAM içi (`:memory:`) vektör veritabanına geçer. Qdrant sunucusunu yerelde başlatmak isterseniz:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

### 3. Boyut Uyuşmazlığı (`RuntimeError: size mismatch for embedding`)
- **Belirti:** Yeni kontrol belirteçleri (`<ARA>`, `<BELGE>`) eklendikten sonra eski model ağırlıkları yüklenirken hata oluşması.
- **Çözüm:** Kodlarımızdaki `resize_state_dict` fonksiyonu eski katman ağırlıklarını koruyarak yeni token slotlarını otomatik olarak genişletir. `strict=False` ile yükleme yapılır.

### 4. Ansiklopedik ve Tarih Alanlarında Modelin Düşük Performans Göstermesi veya `<PROPER_NOUN>` Üretmesi
- **Belirti:** Türk Tarihi veya genel kültür sorularında modelin parametrik belleğinin yetersiz kalması, ardışık özel isimler veya belirsiz ifadeler üretmesi.
- **Sebep:** 14M parametreli kompakt modelin tüm ansiklopedik olguları statik ağırlıklarında ezberlemesi beklenemez (Nitekim B1.5 zor-negatif testinde tarih doğruluğu %15.67 çıkmıştır).
- **Çözüm:** Mimari tasarımımızın gereği olarak, bu alanlarda model parametrik ezbere zorlanmamalıdır. **Vector Rover (RAG)** modülü devreye sokularak Qdrant'tan getirilen kanıt belgesi (`<BELGE>...</BELGE>`) üzerinden koşullu sentez yaptırılmalıdır.

### 5. Toplu Üretim Değerlendirmesinde Parenting Çıktılarının Yüklemsiz Görünmesi
- **Belirti:** `evaluate_b1_5_rigorous.py` çalıştırıldığında Tutarsızlık Oranının yüksek görünmesi.
- **Sebep:** Test kümesindeki parenting (ebeveynlik) morfoloji talimlerinin (örn. `olmadığına`) hedef çıktısı bir cümle değil salt kök analizidir (`ROOT: ol`). Kök analizleri `TENSE_` veya `COPULA_` çekim eki içermediğinden sentaktik olarak cümlenin yüklemi sayılmaz; bu bir çöküş değil, veri formatının doğal sonucudur.
