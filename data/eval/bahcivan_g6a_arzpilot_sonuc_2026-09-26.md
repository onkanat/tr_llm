# G6a (T-0137) RAPOR — Arz Üretim Pilotu (İLAN-2) — 2026-09-26 (claude)

ILAN ≠ RAPOR: ilan `bahcivan_g6a_ilan_2026-09-26.md` İLAN-2'de ölçüm ÖNCESİ damgalı;
hükümler bu raporun ayracı da dahil TAMAMI BETİKTEN
(`scratch/t0137/arz_pilot_olc.py`, elle sayı YOK). İlan ayracı `ROUGE-L < 0,005`
değil, İLAN-2'nin üç ayracıdır (araçtan bağımsız).

## Hüküm

**PILOT_GECTI** — `scratch/t0137/arz_pilot_hukum.json`
(`bde527f2f64023718739774d442a3182fe117e7c4638324d724bea61a964fdf9`), rc=0.

| Ayraç (İLAN-2) | Ölçüm (betikten) | Eşik | Dal |
|---|---|---|---|
| kalip_koruma | **20/20** | ≥ 18/20 (%90) | GEÇTİ |
| kalip_çekimlilik | **5** farklı aile | ≥ 5 | GEÇTİ |
| dil karışımı | **0** örnek | 0 (mutlak) | GEÇTİ |
| JSON parse hatası | 0 | — | — |

Aile dağılımı (ölçülen): nedir 5 · nasil 5 · ne_zaman 5 · hastalik 2 · islev 3.

## Koşum ve araç

- **Araç:** claude subagent × haiku (4 subagent × 5 çift = n=20) — İLAN-2 ARAÇ
  GÜNCELLEME damgası gereği (gpt-oss:120b OLLAMA'da hazır değil; Agent aracı
  model listesinde yok — ölçüldü). EŞİKLER DEĞİŞMEDİ.
- **Üretim:** her subagent'a ilanlı kalıp şablonlarından bir aile verildi
  (a1=nedir, a2=nasil, a3=ne_zaman, a4=hastalik×2+islev×3); çıktı dosyaları
  `scratch/t0137/arz_pilot_a{1..4}.jsonl`, şema `{"soru","cevap"}`.
- **Ölçüm:** betik regex kalıp eşleşmesi + cevap-sonlandırma + markdown/emoji +
  yabancı-alfabe kontrolü; hüküm betikten.

## Kanıt digest tablosu (tam sha256)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/arz_pilot_a1.jsonl` (nedir ×5) | `3f83376205999a68ad9f04001133aa68bc0dd99d237bc961824fd7353a0ff1f3` |
| `scratch/t0137/arz_pilot_a2.jsonl` (nasil ×5) | `1dba85efcb0c1b1d0e1c71a8ca7b86ef7148f794d0838c119750dc626a685df6` |
| `scratch/t0137/arz_pilot_a3.jsonl` (ne_zaman ×5) | `e0ec595009248daec36aeda395be599f12b94fac6ba4f9de948ef0abbacac316` |
| `scratch/t0137/arz_pilot_a4.jsonl` (hastalik ×2 + islev ×3) | `0c6ccff63e20887eaab16d2e528d7b2bf4873c767739d8f5faef15929f2c859d` |
| `scratch/t0137/arz_pilot_olc.py` (ölçüm betiği) | `6fe539f9fefb38acb932edf48d1adf32ab8d26028173779a649d83e834f746c9` |
| `scratch/t0137/arz_pilot_hukum.json` (hüküm) | `bde527f2f64023718739774d442a3182fe117e7c4638324d724bea61a964fdf9` |

## Dürüst kayıtlar (bulgu/limit beyanları)

1. **İLAN-2 dağılım yazımı aritmetik hatalıydı:** "(4+4+4+2+2)" toplamı 16 eder,
   20 değil. Subagent talimatlarında ve üretimde gerçek dağılım **5+5+5+2+3**
   oldu; dağılım ayraç DEĞİL (eşik aile sayısı ≥5 — o GEÇTİ). Düzeltme İLAN
   dosyasına ölçüm SONRASI damga olarak değil, bu rapora dürüst kayıt olarak
   yazıldı; İLAN-2 satırı bu rapor referansıyla işaretlenecektir.
2. **Sistem zarfı pilot üretiminde YOK:** İLAN-2 Örneklem maddesindeki
   "D3-istisna … sistem zarfı" pilot istemlerine eklenmedi — kalıp aileleri ve
   üç ayraç zarfsız ölçüldü. Zarf derleme aşamasında T-0134
   `tokenize_specialization_jsonl`'a Bahçıvan zarfı olarak uygulanır (İLAN-3).
   Pilot hükmünün zarfı ölçmediği beyan edilir.
3. **`wc -l` artefaktı:** her aN.jsonl'da son satır `\n` bitmiyor ⇒ `wc -l` 4
   gösterir; betik dosya başına **5** JSON nesnesi saydı (4×5=20, hepsi geçerli).
4. Subagent raporları model çıktısıdır; bu raporun tablosu subagent beyanı
   DEĞİL, ölçüm betiğinin kendi sayımına dayanır.

## Sonuç dalları (İLAN-2)

PILOT_GECTI ⇒ **arz üretimi bu araçla (haiku subagent) açılır**: tam arz
külliyatı üretimi aynı kalıp aileleriyle genişletilir ve
`data/pedagogy/bahcivan_arena.jsonl`'a (kiralanmış writes[]) kaydedilir —
bundan sonraki adım; kota döndüğünde karşılaştırmalı gemini pilotu İLAN'da
kalan opsiyon olarak durur. Fine-tune çıpası: LM-bedeli kapısı 0,0230, taban
çıpası 0,1392 (İLAN-3, operatör onaylı).

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · tokenizer +
`src/compiler/**` DOKUNULMAZ · kanonik eval betiği DOKUNULMAZ · bu pilot
ROUGE ölçmez (arz kalite kapısı, taban-çıpa zincirine girmek için derleme +
fine-tune ayrı fazlarda) · elle sayı YOK · `git add -A` YASAK · commit
operatör kapısıdır.