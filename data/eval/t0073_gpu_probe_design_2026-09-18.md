# T-0073 — A1 GPU verim sondası (TASARIM + KAPI İLANI)

**Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** İlan damgası: `2026-09-18T14:44:17Z`

## 0. Neden bu sonda var — iki taraflı bir hesap hatası

| | iddia | kusur |
|---|---|---|
| T-0072 §8 (ben) | 100 M ÷ 128 = 781.250 adım ⇒ 336 saat | **batch=1 varsayımı**; adım başına jeton = batch × blok |
| Antigravity (düzeltme) | 24.414 adım × 0,43 sn ⇒ **2,91 saat** | 0,4332 sn/adım **batch=16/blok=64**'te (1.024 jeton/adım) ölçüldü; **batch=32/blok=128**'in adım sayısıyla (4.096 jeton/adım) çarpıldı ⇒ **4× iyimser** |

**Ortak kök:** `sn/adım` birimi batch ve blokla **değişir**; iki taraf da onu sabit sanıp
diğerinin adım sayısıyla çarptı. Aynı-konfigürasyon hesabı (T-0052 KOŞUM A, temiz):
`100.000.000 ÷ 1.024 = 97.656 adım × 0,4332 sn = 42.304 sn = 11,7 saat`.

⇒ **Bu sonda `sn/adım` raporlamaz.** Doğrudan **jeton/sn** ölçer:
`jeton_sn = batch × blok × adım / toplam_süre`; adım ve saat buradan türetilir.

## 1. Izgara

batch **[8, 16, 32, 64]** × blok **[128, 256]** — her hücrede 50 adım; en iyi hücrede
ayrıca **200** adımlık doğrulama koşumu (T-0052 dersi: kısa koşum tam koşumu
2,2× küçük göstermişti).

Veri: `data/anka_a1_pretrain.bin` (meta `block_size=128`, 100,000,000 jeton). Koşum `train.py`'nin **kendisiyle**
yapılır (sadakat: aynı model/optimizer/kayıp/maskeleme yolu), `--pretrain` ile (**ön-eğitim
hedefi**; bkz. §0.5), `--from-scratch` ile (hücreler arası resume imkânsız), `--save-path scratch/t0073_probe.pt`
(donmuş değil).

## 0.5 🚩 TASARIM DÜZELTMESİ (v2) — ölçümden ÖNCE

**v1** sondanın `train.py`'yi **olduğu gibi** koşmasını öngörüyordu. Sondanın kendi 3 adımlık
doğrulama koşumu, `train.py`'nin düz-metin külliyatta **hiçbir şey öğretmediğini** ortaya
çıkardı (T-0073 tanısı: maske sonrası hayatta kalan hedef **0/38.400**, kayıp **0,0000**,
MPS geçersiz hedefte hata vermediği için sessiz). Düzeltme T-0074'te yapıldı (`--pretrain`
modu; G1–G5 geçti, bkz. `data/eval/t0074_acceptance_2026-09-18.md`).

Bu yüzden ızgara **`--pretrain` ile** koşar ve S1 kapısına **canlılık** şartı eklenir.

**Neden bu bir kural ihlali değil:** kural "kapılar ölçümden ÖNCE ilan edilir"dir. Bu belgede
ızgaranın **tek bir hücresi bile ölçülmemiştir** (ne v1 ne v2). Düzeltme, ölçümden önce ve
gerekçesiyle yapılmıştır; v1 metni transkriptte duruyor. Izgara (batch/blok) ve S2–S5
**değiştirilmedi**. Eğer no-op koşumun verimi ölçülseydi sayı **iyimser** olurdu — bu yüzden
düzeltme salt biçimsel değil, ölçümün geçerliliği içindir.

## 0.6 🚩 TASARIM DÜZELTMESİ (v3) — GEÇERSİZ ÖLÇÜMDEN SONRA, GEÇERLİ ÖLÇÜMDEN ÖNCE

**18 Eyl koşumu GEÇERSİZ oldu ve durduruldu.** Kanıt: hücre 3 (`b=32 k=128`), **920 sn duvarda
yalnız 107,5 sn CPU** kullandı (%6,7 ortalama) ve 50 adımı 14,5 dakikada bitirmedi — süreç
bekliyordu, hesaplamıyordu. Hücre 2 aynı iş yükünün yarısını 72,7 sn'de bitirmişti. Aynı anda
makinede WindowServer %25, `replayd` %72, Terminal %88 CPU'daydı (loadavg 7,75) ⇒ **Metal
çekişmesi**. T-0052 kuralı: MPS zamanlaması sessiz makine ister. Hücre 1–2 de kontamine
(o sırada loadavg 10,94 ölçülmüştü), yani **tüm ızgara** geçersiz sayıldı.

İki düzeltme yapıldı; **ızgara (batch/blok), S1–S5 kapıları ve çürütme şartı DEĞİŞMEDİ**:

1. **Yük ön-kontrolü:** koşum, ilk GPU hücresinden **önce** `loadavg(1 dk) > 3.0` ise
   **hiç başlamaz** (rc=2, açık mesaj). Önceki koşum 15 dakikayı geçersiz ölçüme harcadı;
   artık harcamaz. Ayrıca her hücrenin öncesindeki yük JSON'a yazılır (kanıt).
2. **Bütçe seçimi düzeltilmiş istatistiğe bağlandı:** seçim `son/ilk < 1,5` yerine
   **adım 2 ile son adım** oranını kullanır. Gerekçe ölçülmüştür: ilan edilen istatistiğin
   payı **1. adım MPS derleme ısınmasıdır**, yük değil (b8k128: ilk 1,28 sn ↔ kararlı 0,68 sn).
   Konfounded bir istatistiğe veto ettirmek sondanın **kendini boşaltmasına** yol açıyordu:
   hücre 1–2'nin ilan edilen sapmaları 2,91 ve 1,60 ⇒ ikisi de elenirdi, tüm hücreler böyle
   olsa **doğrulama koşumu hiç yapılmaz ve bütçe üretilemezdi**. İlan edilen S2'nin **hükmü**
   yine de ayrıca raporlanır (`S2_ilan_edilen_gecen` ↔ `S2_duzeltilmis_gecen`).

**Neden bu bir kural ihlali değil:** ızgaranın geçerli hiçbir hücresi ölçülmemiştir; iki
düzeltme de geçerli ölçümden **önce** ve gerekçesiyle yapılmıştır. Kapıların *hükmü*
değiştirilmedi, yalnızca hangi istatistiğin **seçim** yapacağı düzeltildi.

## 2. 🚩 İLAN EDİLEN KAPILAR (ölçümden ÖNCE — değiştirilmez)

| kapı | koşul | ayırt edicilik |
|---|---|---|
| **S1 ÖLÇÜM GEÇERLİ** | rc=0 VE log'da `Metal GPU (MPS)` VE `Sıfırdan eğitim` VE `Toplam Süre` VE **`Hedef: ON-EGITIM`** VE **ilk kayıp > 1,0** | **sandbox içinde koşulursa MPS görünmez → CPU'ya düşer; bu kapı o hücreyi GEÇERSİZ kılar. Canlılık şartı, no-op bir koşumun verimini "ölçüp" iyimser sayı üretmeyi engeller (bkz. §0.5)** |
| **S2 YÜK YOK** | son/ilk adım süresi < 1,5 | makinede başka GPU işi sapmayı büyütür (T-0052: 2,2×) |
| **S3 BÜTÇE FORMÜLÜ** | en iyi hücreden adım + saat **formülle** yazıldı | formülsüz sayı denetlenemez |
| **S4 DOKUNULMADI** | `data/*.pt` envanteri aynı VE korpus sha256 aynı | `.pt` yazılırsa düşer |
| **S5 ARTIMLI** | her hücreden sonra yazıldı, sentinel rc=0'a bağlı | çökmede kanıt kalmaz |

## 3. 🚩 ÇÜRÜTME ŞARTI (ölçümden önce)

> Eger tum hucrelerde jeton/sn benzer cikarsa (batch/blok duyarsiz), 'blok buyutmek adim sayisini dusurur ama adim suresini de buyutur' gerekcesi CURUR ve butce yalnizca jeton/sn ile verilir.

## 4. Bu belgenin İDDİA ETMEDİĞİ şeyler

1. **"Model iyi öğrenir" DEĞİL** — yalnız **verim** ölçülür; kayıp değeri bir kapı değildir.
2. **A1 başlatılmadı** — bu belge bir planlama adımıdır.
3. **Sayılar MPS'e özgüdür**; başka donanıma taşınamaz.

İlgili: [[sure-hesabinda-boyut-kontrolu]], [[mps-yuku-adim-suresini-bozar]],
[[sandbox-hides-mps-device]], [[anka-kulliyati-duz-metin-degil]].
