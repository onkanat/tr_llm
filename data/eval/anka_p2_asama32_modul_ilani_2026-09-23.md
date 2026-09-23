# P2 · AŞAMA 3.2 — ONARICI MODÜL İLANI (ön-kayıtlı)

**Damga:** 23 Eyl 2026 04:13 (+03) · **Yazım:** koşumdan ÖNCE · Plan:
`~/.claude/plans/enchanted-wiggling-moon.md` (Aşama 3.2) · Operatör onayı:
"**Onaylıyorum Devam et.**" (bu oturum). Aşama 3.1 kapandı
(`anka_p2_asama31_sonuc_2026-09-22.md`, 2/2 CE geçti).

## 1. Girdiler (ölçülmüş, elle sayı YOK)

| girdi | değer | kaynak |
|---|---|---|
| taban | `scratch/anka_p2_ince_ayr_kos/seg_2.pt` (`ca8b007b…`, Aşama 3.1 nihai, wiki CE **3,4438**) | Aşama 3.1 sonuc digest tablosu |
| sözlük | `data/rebuild/vocab_anka_r1_33114.json` (33.114) | kanonik |
| veri | `data/rebuild/anka_p2_sft_mix.bin` (SFT modu, istem maskeli) | ceketsiz + held-out'suz kanıtlı (sızıntı kapısı 414 kayıt düşürdü) — **ceket 5.400 + held-out 539 eğitimde OLMAZ (plan yasası)** |
| modül spec | r=16 · alpha=32 · dropout 0,05 · 36 katman (`VARSAYILAN_HEDEFLER`, lm_head dâhil) | `train_module.py` kanonik varsayılanları + ölçülmüş pilot deseni |
| eğitilen hacim | 1.327.104 param (%1,40) — taban 93,4M **DONUK** | pilot log: "Taban DONUK: 93,424,986 sabit" |

**Modülün rolü (plan §3.2, ölçülmüş sınır):** biçim/onarım — içerik EKLEMEZ,
ONARIR (modül-yetenek-mimarisi + delta-bilgi-ekleyemez-onarabilir dersleri).

## 2. Tarif (eğitici: `train_module.py` — kanonik, İMPORT düzeyinde kullanılır)

```
venv/bin/python train_module.py \
  --base scratch/anka_p2_ince_ayr_kos/seg_2.pt \
  --module modules/anka_p2_ince_onarici.mod.pt \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --data data/rebuild/anka_p2_sft_mix.bin \
  --ad anka_p2_ince_onarici --r 16 --alpha 32 \
  --lr 2e-4 --batch-size 8 --steps 4000 --save-every 1000 \
  --seed 43 --device mps
```

* **lr 2e-4, SFT maskeli hedef (PAD→−100), wd=0,0 (ağırlık çürümesi YOK — LoRA
  deltası taban ağırlığı değildir), seed 43** — pilot deseni (`pilot_modul_mixr4.log`,
  ölçülmüş: AdamW(lr=2e-4, wd=0.0), b8, 0,31 sn/adım).
* **Adım 4.000** (pilot 1.000'de son60 kayıp 5,06 ± 0,27 — maskeli ölçekte hâlâ
  yüksekti; mevcut `marangoz_mixr4_6000` deseni 6.000'e kadar çıkıyor; 4.000
  ~21 dk. Pilot'tan fark yalnız adım sayısı — İLAN EDİLDİ).
* **Kayıp ölçeği uyarısı (mps-maskeli-kayip-seyreltiyor):** maskeli kayıp ≈ gerçek
  CE × maskesiz oran; talimat penceresinde maskesiz pay ~%3,8/pencere ⇒ kayıp
  sayıları taban wiki CE (3,44) ile AYNI eksende kıyaslanmaz.
* Blok boyutu mix bin meta'sından okunur (train_module fail-closed: sessiz
  varsayım yok — 128 bekleniyor).
* Taban digest + tensör-bazlı değişim denetimi + `vocab_sha256` parmak izi:
  eğitici yerleşik kapıları (dört fail-closed).
* Koşum **sandbox DIŞINDA** MPS; log `scratch/anka_p2_asama32_modul_kos.log`.

## 3. Kabul ve ölçüm

1. **Koşum kabulü:** 4.000 adım tamamlanır, son60 kayıp sonlu ve monoton düşen
   bandda; eğitici taban-değişim kapısı ateşlenmez.
2. **Ölçüm (sırayla):**
   a. **Vakum kapısı:** `scripts/modul_ile_olcum.py` sabit girdiyle takma öncesi/
      sonra logit farkı TAM 0 değilse modül eğitilmemiş ⇒ DUR (modül takılınca
      wiki CE hareket EDER — vakum farkı yalnız başlangıç kontrolüdür).
   b. **Kanonik kap:** `modul_ile_olcum.py --module <modül> --model <taban>
      --baseline <taban> --ceket-ekseni` — baseline ile model AYNI taban (tuzağa
      karşı), sarma yalnız İLK uyan çağrıya takılır.
3. **CE kabulü (modül sonrası):** wiki CE ≤ **ESIK_A_ARTIS %10 kanonik tavanı**
   (`ECA.ESIK_A_ARTIS`; taban 3,4438 ⇒ tavan ≈ 3,788). **Ölçülmüş risk beyanı:**
   pilot (anka_a1r tabanı, 1.000 adım lr 2e-4) modül takılınca CE 3,5352→3,9953
   (**+0,46**) üretti — 4.000 adımla hareket BÜYÜYEBİLİR; bu yüzden aşağıdaki dal.

## 4. Dallar (önceden ilanlı)

* **CE tavanı AŞILIRSA (DAL-1):** modülü **lr 1e-4** + 2.000 adımla TEK tekrar
  (aynı veri/spec); hâlâ aşılırsa sonuç yazılır, modülün CE bedeli raporlanır ve
  karar operatöre döner (modül kabul edilmez — LM bozulması "onarıcı" rolle
  bağdaşmaz).
* Kanonik kap yetenek eksenlerinde eşik hükümü: plan §3 kabulü (kesişim 10,92 ·
  ROUGE 0,35 · tutarsızlık 5,0); ulaşılamazsa döngü-sonlandırma tablosu
  ("Kaplar geçti, kesişim tavanın çok altında ⇒ SONUÇ yaz") geçerlidir.

## 5. Çıktılar

* Modül: `modules/anka_p2_ince_onarici.mod.pt` (+ ara `--save-every` 1000'lik
  yazımlar eğitici meta'sında).
* Ölçüm: `data/eval/anka_p2_asama32_modul_olcum_2026-09-23.json` +
  `…_modul_2026-09-23.json` (`--modul-cikti`) · sandbox dışı MPS.
* Kapanış: `data/eval/anka_p2_asama32_modul_sonuc_2026-09-23.md` — digest'ler
  betikle, damga betikten.