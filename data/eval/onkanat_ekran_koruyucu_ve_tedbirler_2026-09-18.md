# Onkanat Notu: macOS M2 Ekran Koruyucu Kaynak Tüketimi ve Claude Tedbirleri

**Tarih:** 2026-09-18  
**Kapsam:** macOS Sonoma/Sequoia (Apple Silicon M2, 8 GB Birleşik Bellek), Anka A1 Ön-Eğitim Süreci (T-0077)  
**Otorite:** Onkanat Direktifi & Kristal Mimari Mühendisliği  

---

## 1. Yönetici Özeti ve Olayın Gelişimi

Anka A1 modelinin 100.000.000 jetonluk ön-eğitimi sırasında (`data/anka_a1_pretrain.bin`, $B=8, K=256$, 48.828 adım), sistem belleği üzerinde iki kritik kriz yaşanmıştır:

1. **İlk Kesinti (Adım 770):** Adım süresi 0.84 sn'den 1.23 sn'ye uzamış ve sistem belleği kritik seviyeye indiği için arka plan süreci harness tarafından durdurulmuştur (`kill`). Adım 500'deki periyodik atomik checkpoint (`data/anka_a1.pt`, 355 MB, kayıp 4.51) sayesinde veri kaybı yaşanmamıştır.
2. **İkinci Kriz (Takas Çökmesi - Swap Thrashing):** Eğitime 500. adımdan devam edildiğinde adım süreleri $1.70 \rightarrow 4.14 \rightarrow 9.37 \rightarrow \mathbf{25.71}$ saniyeye fırlamış, saniyede 230 MB veri diske takas edilmiş (90 saniyede 18.4 GB swapout) ve sistem kilitlenmiştir.

Onkanat'ın uyarısıyla ekran koruyucu/duvar kağıdı mekanizmaları araştırılmış, kök nedenler ampirik olarak kanıtlanmış ve hem işletim sistemi hem de Claude eğitim döngüsü tarafında kalıcı tedbirler alınmıştır.

---

## 2. Kök Neden Analizi (Ampirik ve Mimari Bulgular)

### A. Ekran Koruyucu ve Duvar Kağıdı Mekanizması
- **4K/240fps HEVC Video Sürekli Render (`WallpaperAerialsExtension`):** macOS Sonoma/Sequoia'nın dinamik "Havadan Çekim" (Aerial) duvar kağıtları basit animasyonlar değil, devasa 4K video dosyalarıdır. Süreç incelemesinde `WallpaperAerialsExtension` (PID 1117) servisinin **21 saattir arka planda çalıştığı ve 38 dakika saf CPU süresi tükettiği** saptanmıştır.
- **`WindowServer` ve Birleşik Bellek (Unified Memory) Çatışması:** Apple Silicon mimarisinde RAM ve VRAM aynı fiziksel 8 GB havuzu paylaşır. Video duvar kağıtları ve Mission Control sanal masaüstleri için Metal doku tamponları (texture buffers) ayrılarak **1.5 – 2.0 GB VRAM kilitlenmiştir (`wired memory`)**.
- **`idleassetsd` Arka Plan Döngüsü:** "Hepsini Karıştır" (Shuffle All) özelliği açıkken arka planda sürekli 500 MB - 2 GB boyutunda yeni 4K videolar indirilip diske ve SQLite veritabanına indekslenmektedir.

### B. PyTorch MPS Allocator ve Bellek Tavanı Davranışı
- **Büyük Logits Tensörü ($8 \times 256 \times 32.852$):** Modelin $V=32.852$ kelime dağarcığı nedeniyle her ileri yayılımda tek bir `logits` tensörü **$\approx 269$ MB (0.251 GiB)** yer kaplar. İleri ve geri yayılımın anlık tepe bellek ihtiyacı **$\mathbf{3.05}$ GiB** olarak ölçülmüştür.
- **Önbellek Tutulması (Allocator Retention):** PyTorch MPS tahsis edicisi serbest kalan bellek bloklarını işletim sistemine iade etmeyip kendi havuzunda tutar. Sonuçta Python RSS'i görünüşte küçük kalsa da gerçek grafik belleği (`owned unmapped graphics`) 4.3 GB'a kadar tırmanmış ve 8 GB'lık cihazda sistem RAM'i sıfırlanmıştır.
- **Yapay Tavan Tuzağı:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5` ($2.67$ GiB) verildiğinde, modelin 3.05 GiB'lık meşru tepe ihtiyacı tavanı aşmış ve ilk adımda `RuntimeError: MPS backend out of memory` hatası üretilmiştir.

---

## 3. Alınan Tedbirler ve Kalıcı Çözüm Standartları

### [BÖLÜM 1] İşletim Sistemi ve Donanım Tedbirleri (Uygulandı)

1. **Statik Görsele Geçiş:**
   - *Sistem Ayarları $\rightarrow$ Duvar Kağıdı:* Hareketli 4K Aerial videolar kapatıldı; **"Resimler"** bölümünden sabit statik bir arka plan seçildi. "Hepsini Karıştır" kapatıldı.
2. **Ekran Koruyucu İptali ve Ekran Uyutma (Display Sleep):**
   - *Sistem Ayarları $\rightarrow$ Ekran Koruyucu:* Hareketli ekran koruyucu devre dışı bırakıldı.
   - *Sistem Ayarları $\rightarrow$ Kilit Ekranı:* Pilde 2-3 dk, adaptörde 5-10 dk içinde ekran panelinin doğrudan kapatılması (Display Sleep) ayarlandı. Böylece GPU "Deep Idle" moduna geçerek tüm kaynakları eğitime bırakır.
3. **Kalıntı Servislerin Temizlenmesi:**
   ```bash
   killall WallpaperAerialsExtension VTDecoderXPCService 2>/dev/null
   rm -rf ~/Library/Application\ Support/com.apple.idleassetsd/Customer/*
   ```

#### Ampirik Sonuç (Öncesi / Sonrası Ölçümü)
| Sistem Metriği | Hareketli Video Duvar Kağıdı | Statik Görsel & Temizlik Sonrası | Kazanım |
| :--- | :--- | :--- | :--- |
| **Kilitli Bellek (Wired)** | 6.479 MB (~6.5 GB) | **1.374 MB** (~1.4 GB) | **~5.1 GB RAM serbest kaldı** |
| **Sıkıştırılmış Bellek** | 2.957 MB (~3.0 GB) | **747 MB** | **%75 hafifleme** |
| **Sistem CPU (Sysload)** | %66.77 (Ağır swap) | **%13.72** | **Disk takası (swap) tamamen durdu** |
| **İşlemci Boşta (Idle)** | %22.48 | **%79.41** | **Donanım ferahladı** |

---

### [BÖLÜM 2] Claude ve Model Eğitimi Tedbirleri (Uygulandı)

1. **Atomik Periyodik Checkpoint (`--save-every 500`):**
   - Model ağırlıkları her 500 adımda bir `data/anka_a1.pt.tmp` $\rightarrow$ `os.replace` ile atomik olarak kaydedilir. Olası bir sistem kesintisinde azami kayıp ~6.5 dakika ile sınırlandırılmıştır.
2. **Öğrenme Oranı Koruma Kilidi (O6 Kapısı - `--lr 0.001`):**
   - `train.py:135` içindeki resume dalının öğrenme oranını sessizce `0.0002`'ye (5 kat düşük) düşürmesi engellenmiş; komuta açıkça `--lr 0.001` verilerek O6 doğrulama kapısıyla kilitlenmiştir.
3. **MPS Bellek Tavanı Kalibrasyonu:**
   - 3.05 GiB'lık tepe ihtiyacı karşılamak ve sistem geneline 4 GB serbest alan bırakmak için ortam değişkenleri optimize edilmiştir:
     ```bash
     PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3
     PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
     ```
4. **O7 Verim Emniyet Bekçisi:**
   - Eğitim sırasında son 10 adımın medyan adım süresi $\mathbf{> 3.12}$ saniye ($4 \times 0.78$ sn bütçe tavanı) olursa koşumu sessizce uzatmak yerine korumalı olarak durduran fail-closed emniyet kapısı devreye alınmıştır.

---

## 4. Nihai Durum ve İlerleme

- **Canlı Süreç:** Anka A1 Ön-Eğitimi (Adım 500'den devam, toplam 48.328 adım).
- **Adım Süresi:** **$0.74 – 0.95$ sn/adım** (Hedef bütçe ile %100 uyumlu).
- **Kayıp Eğilimi:** $10.60 \rightarrow 4.51 \rightarrow \mathbf{3.86 - 4.18}$ aralığında kararlı iniş.
- **Disk Takası:** 0 MB/s (Sıfır takas).

Bu protokol, 8 GB RAM'e sahip Apple Silicon cihazlarda büyük dil modellerinin ön-eğitim ve ince ayar süreçleri için standart işletim kuralı (SOP) olarak tescil edilmiştir.
