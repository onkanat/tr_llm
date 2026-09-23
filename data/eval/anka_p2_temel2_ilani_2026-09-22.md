# P2 · AŞAMA 2b — TEMEL 2.0 KARIŞIMLI KOŞUM İLANI (ön-kayıtlı)

**Damga:** 22 Eyl 2026 20:30 (+03) · **Yazım:** koşumdan ÖNCE · Plan:
`~/.claude/plans/enchanted-wiggling-moon.md` (operatör onaylı). Bu ilan koşumun
TARİFİDİR; koşum çıktısı `scratch/anka_p2_temel2_kos/` altında artımlı yazılır.

---

## 1. Amaç (döngünün hangi eksenini kırar)

Temel tarifi zarfsızdı (`anka_a1r` saf Wikipedia — tutarsızlık %100). Bu koşum mevcut tabanı
**karışım** ile uzatır: **wiki %25 : talimat %75** pencere-bazlı deterministik karışım.
Talimat bileşeni `anka_p2_sft_mix.bin` (Aşama 2a; vocab **33114**, ceket + held-out HARİÇ,
sızıntı kapısı 414 kayıt AÇIKÇA düşürdü — ölçüldü). Eğitim kodu Aşama 1'de onarıldı:
PAD→−100 tek maske sözleşmesi + scheduler + clip 1,0.

## 2. Ölçülen girdiler (elle sayı YOK; hepsi betikle)

| girdi | değer | kaynak (ölçüm) |
|---|---|---|
| taban | `data/anka_a1r.pt` CE 3,5352 (ppl 34,4) | 0e zemin reproduce |
| sözlük | `data/rebuild/vocab_anka_r1_33114.json` (33.114) | A1-r külliyat |
| talimat arzı | **125.814 pencere** · 15,3M pencere jetonu · gerçek 5,16M jeton | 2a meta |
| kayıt uzunluğu | medyan 31 · ort 41 · P99 125 | 2a ölçümü ⇒ **blok 128** |
| PAD oranı | **%67,95** | 2a meta (gerçek PAD id sayımı) |
| hedef maskesiz oran | **%3,8**/pencere (talimat) — PAD+prompt+EOS-sonrası maskeli; kalan = `<OUTPUT>` içeriği | 22 Eyl probe ($TMPDIR/p2_surucu_probe.py) |
| wiki arzı | 781.250 pencere (`data/anka_a1r_pretrain.bin`, blok 128 native) | memmap ölçümü |

## 3. Tarif (`scratch/anka_p2_temel2_surucu.py` — kendi döngüsü, gerekçe §6)

* Başlangıç: `anka_a1r.pt` + `resize_state_dict`; **optimizer carry YOK** — taban moment
  taşımaz (`data/anka_a1r.pt.opt.pt` yok), AÇIK uyarıyla sıfırdan (T-0092/G4a deseni).
  Sidecar VARSA digest uyuşmazsa DUR (G5 sınıfı) — ilanda beyan: beklenen yol YOK.
* Karışım: pencere sıra sayacı `% 4 == 0 ⇒ wiki`, aksi talimat → **%25 wiki : %75 talimat**
  (b8 parti içinde 2 wiki + 6 talimat). Deterministik: seed 42.
* Hedefler: wiki = düz sonraki-jeton (`y = data[i+1:i+1+BLOK]`, kanonik `get_batch`
  konvansiyonu) + PAD→−100; talimat = `mask_prompt_targets` (prompt+EOS-sonrası) + PAD→−100
  (`maske_pad_hedefleri`, P2/Aşama 1 sözleşmesi). Tamamı −100 pencere AÇIKÇA sayılır, atılır.
* Scheduler: warmup 100 + cosine, peak **1e-4**, min_lr 1e-6, toplam 20.000 adım (B1 şablonu,
  `get_lr` train.py'den import); ilerleme 1.0'da kırpılır (tuzağın kapısı testle kilitli).
  **Clip 1,0** · weight_decay 0,01 · `ignore_index=-100` tek maske.
* 4 segment × 5.000 adım = **20.000 adım**; segment başına checkpoint + optimizer/scheduler
  carry yan dosyası (`seg_N.pt.opt.pt`, T-0092 şeması) + **Wikipedia CE kapısı**.
* Adım başına etkili hedef ≈ 2×128 + 6×~4,9 ≈ **186 jeton** (ölçülen %3,8 ile); talimat
  pencere ziyareti = 20.000 × 6 = 120.000 ≈ **1 epoch** (i1'in ölçülmüş zarf-öğretici dozu);
  wiki 40.000 pencere = arzın %5,1'i (tekrarsız).
* Süre tahmini: 0,43-0,46 sn/adım (ölçülmüş b8/blok128 MPS bandı) ⇒ ~2,5-3 saat + 4 CE
  ölçümü. Koşum **sandbox DIŞINDA** (MPS sandbox'ta gizli).

## 4. Kapılar (fail-closed; koşum sırasında)

| kapı | kural | dal |
|---|---|---|
| H1 | `tek_egitici_onkontrol(WIKI_BIN)` başlangıçta | >1 tutucu ⇒ DUR (T-0097/K5) |
| L1 | ilk adım kaybı 0,0 **veya** ≥ ln V − 1,0 (9,41) | DUR — MPS sessiz no-op / soğuk başlangıç (imza moda bağlı) |
| CE | her segment sonunda `ECA.a_ekseni` CE **≤ 3,8887** (taban 3,5352 × 1,10) | DUR + `sonuc.json` DURDU damgası |
| CARRY | sidecar varsa `model_sha256` eşleşmesi | uyuşmazsa DUR |
| YAZIM | `segments.jsonl` append · `sonuc.json` atomik · `SONUC_BITTI` sentinel yalnız rc==0 | — |

CE_TAVAN kodda kanonik sabitten bağlanmıştır: `assert |3,8887 − 3,5352×(1+ESIK_A_ARTIS/100)| < 1e-3`.
Kapının İKİ DALI sınamalı (`--kapi-sinamasi`): üstü ⇒ DUR=True, altı ⇒ DUR=False (ölçüldü ✓).

## 5. Kabul ve dallar (önceden ilanlı; planın döngü-sonlandırma tablosundan)

* **Kabul:** 4/4 segment CE ≤ 3,8887; koşum `BITTI` + sentinel. Başarısızlık anlamlı bilgi
  taşır (eşik Aşama 0'da kalibre edildi — Ç4 vakumu kapandı).
* **DAL-1 (tek ilanlı dal):** CE düşerse (kapı DÜŞTÜ) ⇒ wiki payı %25→%50 (patern %2),
  optimizer carry ile (`seg_N.opt.pt`) devam; hâlâ yüksekse program BİTER, `sonuc.json`
  DURDU damgasıyla yazılır.
* Aşama 2 sonunda zarflı istem kalitatif kontrolü: ", [?], [?], …" davranışı KAYBOLMALI
  (taban zeminde %100 — ölçüldü).

## 6. Neden train.py yerine sürücü-döngü (ilanlı gerekçe)

`train.py` tek veri kümesi + **tek hedef modu** koşar (`--pretrain` ya da SFT). Karışık
bini subprocess'la eğitmek İKİ YANDAN yanlış: pretrain modu talimat zarfını unmaskeli
öğretir; SFT modu wiki pencerelerinde sıfır-hedef ⇒ T-0073 kapısı (sessiz 0,0 sınıfı).
Sürücü pencere-bazlı hedef modu seçer; kanonik kod parçaları **import edilir** (kopya
YASAK kuralı): `get_lr`, `maske_pad_hedefleri`, `optimizer_sidecar_path`, `sha256_file`
← train.py; `a_ekseni` ← ECA; `mask_prompt_targets`, `KristalLM` ← train_step_demo.

## 7. Plandan ölçümle gerekçeli sapmalar (beyan)

| plan | uygulandı | gerekçe (ölçüm) |
|---|---|---|
| blok 256 (~30-34K adım ≈ 7 saat) | **blok 128, 20K adım ≈ 2,5-3 saat** | kayıt dağılımı P99 125 ⇒ blok 256 PAD %83,9 (ölçüldü), cevap zaten ≤127 |
| ~55-60M talimat jetonu arzı | **5,16M gerçek** (+15,3M pencere) | ham metin kaynağı ölçüldü (2a) — planın tahmini bayat bin'lere dayanıyordu |
| ilan dosyası 2026-09-24 | 2026-09-22 | damga betikten (zaman-dilimi karışıklığı dersi) |

## 8. Disk

4 ckpt ≈ 1,4G (356MB×4) + yan dosyalar ~747MB×4 ≈ 3G ⇒ **~4,4G**; `scratch/` altında.
Ara segment yan dosyaları koşum sonunda **korunur** (DAL-1 carry girişi ilanlı) — son
sonuç yazıldıktan sonra operatör kararıyla temizlenir. Kapalı i1 koşum dizinlerine
dokunulmaz (SALT OKUNUR).

## 9. Sonraki aşama

Aşama 3: kalibre eşiklerle (kesişim 10,92 · tutarsızlık 5,0 · ROUGE 0,35) yetenek aşaması —
Temel 2.0 üstünde kısa ince ayar + onarıcı modül.