# ANKA_i1 · FAZ 1 SONUCU — talimat aşaması: kapı İLK segmentte düştü

**İlan:** `data/eval/anka_i1_faz1_ilani_2026-09-22.md` (**koşumdan önce** yazıldı)
**Koşum:** `scratch/anka_i1_kos/` · sürücü `scratch/anka_i1_surucu.py` · `rc=2` (kapı) · 204 sn

---

## 1. Hüküm: İKİ KABUL ÖLÇÜTÜ DE DÜŞTÜ — aşama bedel ödedi, karşılık vermedi

| # | kabul ölçütü (ilan §5) | eşik | ölçülen | sonuç |
|---|---|---|---|---|
| **1 (zarf)** | `tutarsızlık` | **< %10** | **%100,00** (taban da %100) | **DÜŞTÜ** |
| **2 (LM)** | Wikipedia CE | **≤ 3,8887** | **4,0003** (ppl **54,62**) | **DÜŞTÜ** |

⇒ İlan §3'ün **H-R** dalı: *"aşama ya zarfı öğretemez ya dil modelini bozar"* — **ikisi birden**.

**Sert kapı, ilan edildiği gibi çalıştı:** 1. segmentin sonunda Wikipedia CE ölçüldü
(4,0003 > 3,8887) ve koşum **204 saniyede durdu**; 6 segmentin kalanı koşulmadı.
Diğer eksenler: ezber %0 · ROUGE-L 0,0020 · kesişim %1 · B düşüş −3,49 (B iyileşti).

> **Ölçüm, "yön doğruydu ama doz fazlaydı" demiyor.** 500 adım zarfı **hiç** öğretmedi
> (tutarsızlık %100 → %100, tabanla birebir) ve bu sırada dil modelini %13,16 bozdu.
> Yani bu tarif **net zarar**: bedel ödendi, istenen davranış gelmedi.

## 2. ASIL BULGU: hasar hızı veriye BAĞLI DEĞİL

| koşum | veri | adım | CE | tabana göre | **hız (/100 adım)** |
|---|---|---|---|---|---|
| taban | — | — | 3,5352 | — | — |
| **bu koşum** | **genel TALİMAT** | 500 | **4,0003** | **+%13,16** | **+%2,63** |
| T-0096 seg_1 | **YETENEK (marangoz)** | 1000 | 4,3592 | +%23,31 | +%2,33 |

**Hasar hızı aynı** (+%2,63 ↔ +%2,33 / 100 adım). Bu iki şeyi **ELER**:

* **Yetenek verisi zararı vermiyor** — genel talimat verisiyle de aynı hızda bozuluyor.
  (r34'ün *"tam ince ayar bilgiyi ekliyor ama LM'i yok ediyor"* bulgusunun nedeni **içerik değil**.)
* **İstem maskesi kusuru zararı vermiyor** — bu koşumda maske **DOĞRU** uygulandı
  (`--no-pad-mask`; Faz 0 ölçtü: korunan hedef %24,35, PAD maskesiyle birebir aynı) ve
  hasar yine oldu. Yani `train.py:115` çakışması **asıl neden değildi**.

⇒ **Kök neden: bu tabanda `lr 2e-4` ile TAM İNCE AYAR.** Ne veri ne maske; **öğrenme oranı
ve tam-parametre güncellemesi.**

## 3. Kapı, kanıt ve temizlik

| | |
|---|---|
| duran dal | `PPL KAPISI seg_1: CE 4.0003 > tavan 3.8887` |
| `sonuc.json` | `durum: DURDU` · artımlı yazıldı (`.tmp`+`os.replace`) |
| `segments.jsonl` | 1 kayıt (artımlı append) |
| `seg_1.pt` | **356 MB** · sha256 `ecebe00ec41db902…` |
| **sentinel** | **YOK** ⇒ `scratch/anka_i1_kapanis.py` **reddetti**: *"kapanış ÜÇ SİNYALİ tutmuyor"* |
| H1 (koşum öncesi) | 0 tutucu ✓ |
| H3 (`timeout=`) | sürücüde 1 site, eksik **0** ✓ |
| **H2 (kapanış bekçisi)** | **KOŞUM CANLIYKEN sınandı**: `KOSUM CANLI (PID [24152]) ⇒ araç CALISTIRILMAZ` → `rc=2` ✓ — planın *"yazıldı ama bağlanmadı"* dediği boşluk **kapandı ve kanıtlandı** |

## 4. Külliyat: koşumdan ÖNCE bulunan iki kusur ve onarımları

**(a) "Genel talimat" külliyatında yetenek verisi vardı.** `data/train_chat_balanced.bin`
bileşiminde `carpenter_specialization` **2.500 kayıt (%10,3)** ⇒ planın *"yetenek verisiyle
DEĞİL"* öncülü ve Faz 2'nin *"yeteneği olmayan temel"* koşulu çürürdü; kabul ölçütü de
şişerdi. → Ceket **hariç** yeniden derlendi.

**(b) Derleyici `<EOS>`'u kırpıyordu.** `prepare_chat_balanced_dataset.py` uzun kaydı
`rec[:block_size]` ile kırpıyor; `encode` sona `<EOS>` eklediği için kırpma **tam da onu**
atıyordu ⇒ `mask_prompt_targets` kural 3 kaydı `<EOS>` ile kapattığından o pencerede
**komşu kaydın istemi maskesiz** kalıyordu (istem sızıntısı). **Hipotez ölçüldü ve birebir
çıktı:** `<EOS>`'suz 1.431 kaydın **tamamı** 128'i dolduran kırpılmış kayıtlardı; PAD'li
bloklarda açık **sıfırdı**. → Onarıldı: gövde 127 jeton, `<EOS>` **geri eklendi**, sayı
meta'ya yazıldı, oran %20'yi aşarsa **fail-closed DURUR**.

| | önce | sonra |
|---|---|---|
| külliyat | `data/train_chat_balanced.bin` | **`scratch/anka_i1_talimat.bin`** |
| kayıt | 24.375 | 21.875 |
| ceket | 2.500 (%10,3) | **0** |
| `<EOS>`'suz blok | **1.431 (%5,87)** | **0 (%0,000)** |
| kırpılan | sessiz, `<EOS>` kayıp | 1.448 (%6,62), **beyan edilmiş** |

## 5. Kalan soru ve tek kaldıraç

Ölçüm tek bir kaldıraca işaret ediyor: **öğrenme oranı** (ve/veya tam-parametre yerine
parametre-verimli bir aşama). Plan `lr 2e-4`'ü T-0096'dan **aynen** devralmıştı; bu koşum o
devralmanın **bedelini ölçtü**.

**Bu bir "ayar kovalama" değildir** — çünkü:
* Kapı ilan edilmişti ve **düştü**; sonuç **olumsuz** olarak yazıldı.
* Ölçüm, aday kaldıracın **hangisi olduğunu** gösterdi (veri ve maske **elendi**).
* Sıradaki adım **yeni bir ön-kayıtlı ilan** ister; eşik değiştirilmez, **tek değişken `lr`** olur.

**DOKUNULMAZ:** `data/**` (donmuş) · `CLAUDE.md` · `scratch/t0096_*` · `scratch/t0097_*` ·
kapanmış `data/eval/anka_r17…r34*` · `src/**` (bu turda **değişmedi**; `prepare_chat_balanced_dataset.py`
yalnız **derleyici** — ölçüm kodu değil).
**Commit yok · `git add -A` kullanılmadı.**

## 6. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `scratch/anka_i1_kos/seg_1.pt` | `ecebe00ec41db902356fc5ecf027ce9694ca947649c66a37cf1c4484deb03e76` |
| `scratch/anka_i1_talimat.bin` | `798c2adc55430cd57ce5dfcf182e78d312822782c507f0695779799b1f5f6c3b` |
| `data/eval/anka_i1_faz0_yeni_2026-09-22.json` | `3abd7f48cec5d199f221e249c101f4bae6439e4ef2a096a8913ca1783b08d682` |
| `data/eval/anka_i1_faz1_seg1_yetenek_2026-09-22.json` | `0fd8eba5216cf2c6914402d5f1edab37de6e36e97ad412dc1f11a1aeaf1f1ee4` |
| `scratch/anka_i1_surucu.py` | `c4a41213cfe4916207762bf211641f3e6a53bbe8b02864480bac2b2376a4119f` |
| `scratch/anka_i1_kapanis.py` | `fba6f2e927b266e899f783c8c7df750f603444af44ea7f85a0e3e545a2ab86b1` |
| `scripts/anka_i1_faz0.py` | `f966400bfc0a510d6bc5a18fd8d4a56588001d21f879ec4ad5f01fd7da115c48` |
| `scripts/prepare_chat_balanced_dataset.py` | `e318f1da809bb981ad8583e9b2432aa77a60539314f0ce87ead6dd26d04e3c88` |
| `data/anka_a1r.pt` (taban, **değişmedi**) | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
