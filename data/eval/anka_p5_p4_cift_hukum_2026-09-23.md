# P5 · ÇİFT HÜKÜM — ROUGE eşik değişikliğinin (0,35 → 0,3221) koşum hükümlerine etkisi

**Damga:** 23 Eyl 2026 · **Görev:** T-0099 · **Koşum YOK** — hüküm =
f(metrik, eşik) saf fonksiyonu; metrikler mevcut sonda JSON'lardan okunur,
eşikler ilan Ekleme §9'dan (operatör kararı). Kesişim/tutarsızlık/ezber
eşikleri P5'te SABİT ⇒ yalnız ROUGE ekseninin çift hükmü.

## Hüküm tablosu

| koşum | kaynak JSON | ROUGE-L | eski eşik 0,35 | yeni eşik 0,3221 | hüküm |
|---|---|---|---|---|---|
| Temel 2.0 | (P4 sonuç raporu tablosu) | 0,0056 | ALTINDA | ALTINDA | **değişmez** |
| T-0096 | `scratch/t0096_kos/eval_seg_6.json` | 0,3035 | ALTINDA | ALTINDA | **değişmez** |
| P3 nihai | `scratch/anka_p3_kos/sonda_seg_3.json` | 0,1014 | ALTINDA | ALTINDA | **değişmez** |
| P4 nihai | `scratch/anka_p4_kos/sonda_seg_2.json` | 0,1346 | ALTINDA | ALTINDA | **değişmez** |

## Not

* Dört koşumun hükmü değişmedi — eşik değişikliği geriye dönük olarak hiçbir
  kapanmış kararı bozmuyor; yalnız ileriye dönük eşiği ölçülen tavana
  hizalıyor (0,3835 × 0,84).
* **Marj daraldı:** T-0096'nın 0,3035'i eski eşiğe 0,0465, yeni eşiğe
  **0,0186** uzaktaydı — gelecekte 0,32 bandına dokunan bir koşum artık
  kapıda GECER hükmü verebilir (eşik ölçülen tavana bağlı; bu BEYANLI
  davranıştır).
* Eski hüküm raporları (`anka_p3_sonuc_*.md`, `anka_p4_sonuc_*.md`,
  T-0096 kayıtları) **DÜZENLENMEDİ** — orijinal eşik dönemine aittir.