# Anka · Yetenek Modülü — Pilot Ölçümü (21 Eyl 2026)

**Sorulan soru:** taban model bir kez eğilir; yetenekler ona **modül** olarak eklenir. Modül
takıldığında taban **değişmiyor** (mekanik, ölçüldü) — peki yeteneği **unutma pahasına mı**
öğreniyor? Kapı, tam ince ayarın düşürdüğü kapının aynısı: Wikipedia CE artışı **≤ +%10**.

## Hüküm: kapı GEÇİLMEDİ — ama hasarın %44'ü kesildi

| # | konfigürasyon | A artışı (unutma) | F düşüşü (uyum) | kapı |
|---|---|---|---|---|
| 1 | saf ceket · 500 adım · lr **1e-3** | **+%20,77** | −%23,93 | DÜŞTÜ |
| 2 | saf ceket · 500 adım · lr **2e-4** | **+%14,38** | −%19,65 | DÜŞTÜ |
| 3 | **mix_r4** · 1000 adım · lr 2e-4 · seed 43 | **+%13,02** | −%24,85 | **DÜŞTÜ** |

Konfigürasyon 3, T-0096'nın 1. segmentiyle **birebir aynı** veri/adım/LR/tohum ile koşuldu ⇒
tek değişken modüldür:

| | T-0096 seg_1 — TAM ince ayar | Bu modül |
|---|---|---|
| A ekseni | 3,5352 → 4,3592 = **+%23,31** | 3,5352 → 3,9953 = **+%13,02** |
| taban dosyası | değişti (356 MB yeni checkpoint) | **DEĞİŞMEDİ** |
| artefakt boyutu | 356 MB | **5,3 MB** |

**Modül, unutma hasarını +%23,31'den +%13,02'ye indiriyor** (−%44) — ve hâlâ eşiğin üstünde.
T-0096'nın 6 segmentte ulaştığı tavan: **+%75,62**; modülün tabanı dosya düzeyinde değişmediği
için sürüklenme **sıfırdan** başlıyor.

> **Ölçümün sınırı:** konfigürasyon 3, T-0096'nın **kümülatif 6000 adımıyla değil, tek
> 1000 adımlık segmentiyle** kıyaslanıyor. Modül 6000 adım koşulsaydı A ekseni nereye
> giderdi **ölçülmedi** — taban donuk olduğu için sürüklenmenin rank-16 deltasıyla
> sınırlı kalacağı **hipotezi** test edilmedi.

## Mekanik sözleşme — ölçüldü, ihlal yok

| iddia | ölçüm |
|---|---|
| takma anında çıktı tabanla özdeş | `torch.equal` **True**, delta maks **0,000e+00** |
| kontrol testi duyarsız değil (pozitif kontrol) | `lora_B[0,0]=1e-3` → fark **3,1e-5** ✔ |
| eğitilen parametre | **1.327.104** = formül; tabanın **%1,401**'i (93.424.986 sabit) |
| modül dosyası taban ağırlığı taşımıyor | sızan anahtar **0** |
| üç koşumda da taban dosyası | digest `b93cc1cd…` **başta = sonda** |
| üç koşumda da taban tensörleri | değişen **0** (BEKLENEN: 0) |
| **ölçüm kabının kendi kontrolü** | sıfır modül → A CE **birebir 3,535174** = taban ⇒ kap sağlam |
| ölçümün ayrım gücü | Wilson yarı-genişliği ±0,09 < eşik payı 0,3535 ⇒ **AYIRT EDİCİ** (3,9×) |

## Ölçüm sırasında bulunan ÜÇ cihaza-bağlı kusur (hiçbiri CPU'da görünmez)

1. **`train.py` SFT yolu CPU'da 1. adımda çöker** (`IndexError: Target -100 is out of
   bounds`), **MPS'te sessizce sürer**. T-0096/T-0097'nin SFT koşuları bu yüzden "çalıştı".
   Düzeltilmedi: düzeltmek, kapalı kayıtların kayıp sayılarını kıyaslanamaz kılar.
2. **`LoRAKatmani` yeni katmanı CPU/fp32 doğuruyordu** ⇒ model MPS'teyken `modul_ekle`
   modeli bozuyordu. Eğitim sırası gizliyordu; gerçek değerlendirme yolu zorunlu ateşler.
   Düzeltildi: cihaz/tip **inşa gereği** tabandan devralınıyor (+ CPU'da tip vekili testi).
3. **Bütünlük denetimi çapraz-cihaz karşılaştırma yapıyordu** ⇒ 500 adımlık ilk koşumun
   **en sonunda** patladı, denetim hiç çalışmadı. Düzeltildi, **MPS'te yeniden koşularak**
   doğrulandı.

## Yan bulgu: MPS maskeli kaybı SEYRELTİYOR (T-0096/T-0097 kayıtlarını açıklar)

MPS'te `-100` hedefi `ignore_index=1` ile: maskeli konum **paya 0, paydaya TAM** katılıyor
(per-konum CE açıkça hesaplandı, fark `0,000000`):

```
CE toplamı (gerçek konumlar) : 1505,1405     /gerçek=245 → 6,1434  (doğru)
                                             /tüm=512    → 2,9397  (MPS'in verdiği)
```

⇒ **raporlanan kayıp = gerçek CE × maskesiz oran** (parti başına 0,256…0,553).
T-0096 logunda CoV eğri boyunca **sabit** kaldı (0,2242 → 0,2051) ⇒ salınım parti
bileşiminden. **Kapı sayıları temiz:** unutma ekseni `maskesiz: True`, `ignore_index` hiç
geçmiyor ⇒ A/B ekseni hükümleri bu kusurdan **etkilenmedi**.

## Açık / yapılmayan

* **Kapıyı geçirmek için ayar kovalanmadı.** Üç konfigürasyon ölçüldü, hüküm ilan edildi.
  Kalan 3 puanı kapatacak bir tur **ön-kayıtlı tek deneme** olmalı (T-0067: ölçüm sonrası
  kusur tasarıma göre onarılmaz). Aday kaldıraç: `r` düşürmek (kapasite), replay oranını
  artırmak, adım sayısı.
* Yeteneğin **kendisi** (ezber/ROUGE/kesişim) ölçülmedi — `scripts/evaluate_carpenter_anka.py`
  modül yükleyemiyor. Buradaki F ekseni **uyum** ölçüsüdür, **genelleme değil**.
* `modules/*.pt` **`.gitignore:12` (`*.pt`) kapsamında** ⇒ git'te görünmez; kanıt digest
  tablosudur (aşağıda).
* **Commit yok. `git add -A` kullanılmadı.**

## Artefakt digestleri (tam sha256)

| dosya | boyut | sha256 |
|---|---|---|
| `data/anka_a1r.pt` (taban) | 373,7 MB | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `modules/marangoz.mod.pt` | 5,3 MB | `d1ae6a30589ee702525e0aeedfddcf30dfb3d7c3a9a5d98bd3450970f8caebe0` |
| `modules/marangoz_lr2e4.mod.pt` | 5,3 MB | `04818589cd2de86a04599c8147c216b7df19694d97409f4539558e07231f71fe` |
| `modules/marangoz_mixr4.mod.pt` | 5,3 MB | `2ef7d59ce586265a9aef32fc110b11c04c38007cd11bccb36c76856e18db1825` |

JSON ölçüm kayıtları: `anka_r20_modul_marangoz`, `anka_r21_modul_lr2e4`,
`anka_r22_modul_mixr4` (hepsi `2026-09-21`).

**Tam test takımı:** `1 failed, 255 passed` — düşen tek test bilinen sandbox soket yasağı
(`test_agent_gateway_http_server_endpoints`), regresyon yok.
