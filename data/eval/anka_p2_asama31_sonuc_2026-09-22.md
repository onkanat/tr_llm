# P2 · AŞAMA 3.1 — YETENEK AŞAMASI İNCE AYAR SONUCU (22 Eyl 2026)

**Damga:** 22 Eyl 2026 · rc=0 · İlân: `anka_p2_asama3_ilani_2026-09-22.md`
(koşumdan önce yazıldı). Koşum: `scratch/anka_p2_ince_ayr_kos/` (2 × 1.000 adım,
873 sn).

## 1. Koşum kanıtı

* Başlangıç: Temel 2.0 `seg_4.pt` (`c786f118…`) + **carry yüklendi** — optimizer
  momentleri 101 parametre / adim=20000 (T-0092; digest kapısı geçti, AŞAMA 2
  koşumunun yan dosyası taşıyıcıyı ilk kez gerçekten kullandı).
* rc=0 · sentinel · `sonuc.json` BITTI · **2/2 segment CE kapısı geçti**:
  seg_1 CE 3,4473 · seg_2 CE **3,4438** (tavan 3,8887; Temel 2.0 zemin 3,4427 —
  kayıp eğrisi 3,41'e indi, wiki CE ±0,01 bandında kaldı ⇒ LM KORUNDU).
* warmup 50 + cosine (toplam 2.000) · clip 1,0 · wiki %25 patern aynen.

## 2. Kanonik kap kıyası (Aşama 3 sıfır noktası → ince ayar sonrası)

| eksen | Temel 2.0 zemin | **ince ayar sonrası** | kalibre eşik |
|---|---|---|---|
| tutarsızlık | %96,00 | **%87,00** (−9,0 pp) | < 5,0 |
| ROUGE-L | 0,0056 | 0,0056 | ≥ 0,35 |
| LCS-kesişim | %0,00 (F1 0,0018) | %0,00 (F1 0,0035 — F1 iki katına çıktı ama oran hâlâ 0) | ≥ 10,92 |
| ezber | %0,00 | %0,00 | < 10,0 |

Ölçüm: `data/eval/anka_p2_asama31_zemin_2026-09-22.json` · sandbox dışı MPS ·
`--ceket-ekseni` · rc=0. Hüküm: eşikler hâlâ uzak (beklendiği gibi — ilan §4:
yetenek hükmü modül (3.2) sonrası anlamlıdır). Tutarsızlıkta −9,0 pp hareket
(3 örneklemde %100 → %96 → %87) MONOTON ama Wilson bandı ilanda beyan edilmedi
bu aksiyon için; kıyas sayısal hüküm DEĞİL, yön göstergesidir (gürültüden küçük
farkın işareti yok dersi).

## 3. Digest tablosu

| artefakt | sha256 |
|---|---|
| `scratch/anka_p2_ince_ayr_kos/seg_2.pt` (Aşama 3.1 nihai) | `ca8b007b9601dc71e2524991632f871fd3324871a399fb0ad92779a44535f249` |
| `scratch/anka_p2_ince_ayr_kos/sonuc.json` | `882c50a481dd2c6c81e8131229baaa3ee32336b60800b1e317b470bcb6d895ff` |
| `scratch/anka_p2_temel2_surucu.py` (parametrik sürüm) | `225b428878976edcc299ead4119dea9c13e61a1c6989220c1b2da122868ff655` |
| `data/eval/anka_p2_asama3_ilani_2026-09-22.md` | `67b8572ff35cc2d5bec8a67f8bd6b033c5a1f1fb4b1641cfcc9fff5c828f7f71` |

*Digest'ler betikle hesaplandı (hash tam-digest dersi); `seg_2.pt` sha256 koşum
`sonuc.json`'undaki `dallar[-1].ckpt_sha256` alanından alındı (çift yönlü çıpa).*

## 4. Sonraki — Aşama 3.2 (onarıcı modül, AYRI İLAN)

Temel 2.0+ince ayar modeli sabit; r16 LoRA deseni (`train_module.py` eğitici, lm_head
dâhil), rol: biçim/onarım (LM'i geri getirir, içerik EKLEMEZ — ölçülmüş sınır);
ölçüm kanonik kap + `modul_ile_olcum` vakum kapısı/baseline tuzağı. Modül ilanı
operatör onayıyla yazılır.

## 5. Kapanış beyanı

Aşama 3.1 KAPANDI (koşum kabulü 2/2 CE + kanonik kıyas). Kalibre eşiklere hüküm
bekleyen eksenler: kesişim, ROUGE, tutarsızlık (modül ilanı sonrası ölçüm).