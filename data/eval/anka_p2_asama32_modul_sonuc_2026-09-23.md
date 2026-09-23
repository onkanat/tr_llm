# P2 · AŞAMA 3.2 — ONARICI MODÜL SONUCU (23 Eyl 2026)

**Damga:** 23 Eyl 2026 04:40 (+03) · rc=0 · İlân:
`anka_p2_asama32_modul_ilani_2026-09-23.md` (koşumdan ÖNCE yazıldı, sha256
`6acbbb48…`). Koşum: `train_module.py` · **4.000 adım · 1.149 sn (~19 dk)** ·
0,29 sn/adım sabit band. Ölçüm: `scripts/modul_ile_olcum.py` (vakum kapısı +
kanonik kap) · sandbox dışı MPS · rc=0.

## 1. Koşum kanıtı

* Başlangıç kapıları: taban digest `ca8b007b…` birebir ✓ · modül takıldı
  r=16 α=32, 36 katman · **eğitilen 1.327.104 param (%1,40), taban 93,4M DONUK**.
* **Çalışma notu (ilan §2 sapması, AÇIK):** blok boyutu meta'da üst düzeyde değil
  `kap.block_size = 128` altında olduğu için eğitici fail-closed DURDU; ilanlı
  değere birebir uyumlu `--block-size 128` açıkça verildi — sessiz varsayım OLMAZ.
* Kayıp: ilk **3,3903** → son60 **3,2990 ± 0,6866** (maskeli ölçekte —
  mps-maskeli-kayip-seyreltiyor uyarısıyla kıyaslanmaz; std geniş: partiden
  partide maskesiz pay oynak — %3,8/pencere ölçümüyle uyumlu).
* **Taban DONUK kanıtlandı:** taban dosyası `ca8b007b…` → `ca8b007b…`
  (DEĞİŞMEDİ) · taban tensörleri **0 tanesi değişti (BEKLENEN: 0)**.

## 2. Ölçüm — vakum kapısı + kanonik kap

* **Vakum kapısı GEÇTİ:** takma öncesi/sonrası logit farkı maks **1,00e+01**
  (eğitilmemiş modülde fark TAM 0 olurdu ⇒ DUR; modül eğitilmiş, kap sağlam).
* Oracle pozitif kontrol: kimlik **1,0000** (tavan 1,0 ✓) · distraktor 0,0848 ·
  sabit-tahmin 0,0780.
* **Unutma kapısı GEÇTİ (DAL-1 TETİKLENMEDİ):**

| eksen | taban (Aşama 3.1) | **taban + modül** | kanonik hüküm |
|---|---|---|---|
| A · Wikipedia CE | 3,4438 | **3,5826** (+%**4,03**) | artış ≤ %10 ⇒ **GEÇTİ** (yarı-genişlik ±0,1002 < pay 0,3444 ✓) |
| B · top-1 | %54,60 | %54,52 (−0,08 puan) | düşüş ≤ 5,0 ⇒ **GEÇTİ** |

  Pilot kıyası (ilan §3 risk beyanı): pilot anka_a1r tabanında 1.000 adımda CE
  **+%13** idi; burada 4.000 adımda **+%4,03** — taban artık SFT mix'i görmüş
  (Temel 2.0) olduğu için modülün LM bedeli küçüldü.
* **Yetenek eksenleri (100 örnek, ceket eksenine dahil olan kısmı — ceket verisi
  eğitimde OLMAZ):**

| eksen | taban | modül | kalibre eşik | hüküm |
|---|---|---|---|---|
| ezber | %0,00 | %0,00 | < 10,0 | geç (hareket YOK) |
| tutarsızlık | %87,00 | %90,00 | < 5,0 | eşik ALTINDA KALDI; +3 pp 100 örneklemde gürültü bandında — yön hükmü YOK |
| ROUGE-L | 0,0056 | 0,0068 | ≥ 0,35 | eşik ALTINDA KALDI (gürültü altı fark — işaret hükmü YOK) |
| LCS-kesişim | %0,00 (F1 0,0018) | %0,00 (F1 0,0014) | ≥ 10,92 | eşik ALTINDA KALDI (gürültü altı fark) |

* Modülün kimliği: `data/eval/anka_p2_asama32_modul_2026-09-23.json`
  (`adım=4000 r=16 lr=2e-4 veri=anka_p2_sft_mix.bin`).

## 3. Hüküm (döngü-sonlandırma tablosuyla)

**Aşama 3.2 KAPANDI.** Modül "onarıcı" rolün **LM bedelini** taşıyabildi
(CE +%4,03 < %10, taban DONUK kanıtlı) ama **yetenek eksenlerinde içerik
getirmedi** — 3 eksen de gürültü altında (delta-bilgi-ekleyemez-onarabilir
dersinin dördüncü bağımsız tekrarı: modül onarım mekanizması içerik üretmiyor).
Döngü-sonlandırma tablosunun "**Kaplar geçti, kesişim tavanın çok altında ⇒
SONUÇ yaz; ölçüt ancak yeni insan tavanı ölçümüyle değişir**" kolu: **SONUÇ
YAZILDI**, DAL-1 koşulmadı (CE eşiği aşılmadı).

**Aşama 3 (Yetenek aşaması) KAPANDI:** Temel 2.0 → ince ayar (tutarsızlık
%96→%87) → modül (LM +4% CE bedeli, yetenek +0) — kalibre eşiklere hüküm
bekleyen KALDI: kesişim, ROUGE, tutarsızlık. Anka'nın yetenek taşıyıcısı bu
yapıda (donuk taban + %1,4 LoRA) bu ölçümlerle tavanın çok altında; yeni koşum
yalnız **yeni ölçüm kanıtı** yeni ilan açarsa başlar (planın genel kuralı).

## 4. Digest tablosu

| artefakt | sha256 |
|---|---|
| `modules/anka_p2_ince_onarici.mod.pt` (Aşama 3.2 modülü) | `a9028985cf000671eec48157e543485e50219860e1e7bf292e3ef3fe73170dee` |
| `data/eval/anka_p2_asama32_modul_olcum_2026-09-23.json` (kanonik kap) | `e9124408df798b9d6809ae9813f8f2619c3662702fbd445a0ba5eb37d2f91438` |
| `data/eval/anka_p2_asama32_modul_2026-09-23.json` (modül kimliği) | `523d83f4840b564a9c3c39d5c3ede416472dced5fd3ba34bc7d8f4dcc20699e0` |
| `data/eval/anka_p2_asama32_modul_ilani_2026-09-23.md` | `6acbbb4868324cf22e2546a14e8b09815395972b4b6926065bb332e6da7cb03d` |
| `scratch/anka_p2_ince_ayr_kos/seg_2.pt` (taban, Aşama 3.1) | `ca8b007b9601dc71e2524991632f871fd3324871a399fb0ad92779a44535f249` |

*Digest'ler betikle hesaplandı; taban digest'i eğitici log'u ile modül kimlik
 dosyasından çift-yönlü çıpalı (ikisi de `ca8b007b…` yazıyor).*

## 5. Kalan (operatör kararı bekliyor)

* **Disk temizliği:** i1 eski seg'ler ~2,1G; Temel 2.0 ara yan dosyaları
  (seg_1-3 `.opt.pt`); modülün ara `--save-every` yazımları (1000/2000/3000'lik)
  sonraki koşumlara taşıyıcı değildir — son kayıt yeterli.
* Programın Aşama 0+1+2 kalıcı kazanımları: ölçüt kalibrasyonu (kesişim eşiği
  10,92 = insan tavanı × 0,84), eğitim altyapısı (maske/scheduler/clip/carry),
  Temel 2.0 (`scratch/anka_p2_temel2_kos/seg_4.pt`) ve soyağacı.