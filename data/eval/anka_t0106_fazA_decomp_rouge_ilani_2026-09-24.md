# T-0106 FAZ A · DECOMP ROUGE ÖLÇÜT GEÇİŞİ (İLÂN)

**Damga:** 24 Eyl 2026 · **Görev:** T-0106 · **Yürütücü:** claude ·
Üst: P5-A açık ölçüt kararı + T-0105 (KAPANDI). Plan:
`~/.claude/plans/enchanted-wiggling-moon.md` (operatör onayı 24 Eyl).
İlan kod yazımından ÖNCE.

## 1. Karar beyanı

Marangoz kabin ROUGE'u RAW temsilden sayılıyor (`ECA:481-486`), eşik
0,3221 de RAW'a kalibre (arena_base temiz JETON tavan 0,3835 × 0,84)
⇒ kıyas temsel olarak TUTARLI (keşif ajanı kanıtı); hüküm eşiği
**ESIK_ROUGE 0,3221 DEĞİŞMEZ** (P5 Dal-K). Bu faz, üretim akışının
(README:73) son aşaması olan **decompile'ı hükmün YANINA** ikinci
temsil olarak ekler (P5-A beyanı: "kapıyı DECOMP'a geçmek ilan +
zincir + çift-hüküm gerektirir"); sessiz fallback BEYANLI sayaca
çevrilir; ham gm dizileri kalıcı kayıt altına alınır (P3/P4'te gm
dizileri kayboldu).

## 2. Değişiklik beyanı

* `scripts/evaluate_carpenter_anka.py` K2 (481-511): `except` dalında
  `yuzey = gm; decomp_istisna += 1` (sessiz fallback YOK) · DECOMP
  ROUGE `rouge_l_score(kelimeler(yuzey), ref_kel[i])` · metrik alanları:
  `rouge_l_decomp_ort/_medyan/_std`, `decompile_istisna`,
  `yuzey_unk_sayi`, `ham_gm` (n örneklerin tam gm dizisi; ilk 5
  `ornekler` alanına dokunulmaz) · `esikler` bloğuna
  `rouge_decomp_aday`; **`hukum` sözlüğüne EKLENMEZ**.
* Yeni sabitler eşik bloğunun (68-77; NK çifti 75-77 bölünmez) SONRASINA:
  `TAVAN_ROUGE_DECOMP = 0.9509` (P5akol JSON kaynağı) ·
  `ESIK_ROUGE_DECOMP_ADAY = 0.7988` (0,9509 × 0,84). Satır numaraları
  yazım sonrası `grep -n` ile ölçülür (ezber yasak).
* Kanarya zinciri: `olcum_kabi.py` ESIK_REFERANSLARI'na yeni satır
  tuple'ları (mevcut 6 dokunulmaz) + K11'e `(satır, ESIK_ROUGE_DECOMP_ADAY, 0.9509)` —
  ham tavan eşiğe yazılırsa DÜŞMELİ · `tests/test_olcum_kabi_kapilari.py`
  (d) bloğuna sabit-değer beyanı · `anka_karsilastirma_tablosu.py`
  satir()'ına `rouge_decomp` kolonu (tanısal).

## 3. Beklenti çıpaları (kayıt JSON'larından)

| çıpa | değer | kaynak |
|---|---|---|
| r1 DECOMP tavan | 0,9509 (istisna 0, fallback 0) | `scratch/anka_p5akol_derin_analiz.json` |
| DECOMP aday eşik | 0,7988 (= tavan × 0,84; bağlayıcı DEĞİL) | hesap |
| RAW hüküm eşiği | 0,3221 sabit | `ECA:70` |
| P3/P4 RAW ROUGE | 0,1014 / 0,1346 | p3/p4 sonuc raporları |
| a2 taban ceket ROUGE | 0,0049 | T-0105 JSON `ceket_ekseni.rouge_l_ort` |

DECOMP eşik bağlama kararın beklentisi: **BAĞLANMAZ** — 0,9509 r1-vocab
gold-gm tek orneklem; k=0,84 kalibrasyonu RAW dağılımında yapıldı,
farklı temsile aynen taşınması doğrulanmadı. r2 tavan ölçümü (A3)
girilmeden aday bağlanamaz.

## 4. A3: r2 DECOMP tavan ölçümü

`scratch/anka_t0106_decomp_tavan_r2.py` — `anka_p5akol_derin_analiz`
İMPORT (kopya yasak), VOCAB/LEXICON r2 pinli (T-0104 sha'ları:
vocab `14ce9f4e…` · lexicon `c355a372…`), aynı heldout (`c397eb08…`) /
n=100 / seed 42 → `data/eval/anka_t0106_decomp_tavan_r2_2026-09-24.json`.
Beklenti: r2 tavan r1 bantında (0,9509); sapma aday eşik beyanlı revizyon.

## 5. Kabul

1. `venv/bin/pytest tests/test_olcum_kabi_kapilari.py -q` yeşil.
2. `venv/bin/python scripts/olcum_kabi.py` rc=0.
3. Duman ECA (a1r/r1, n=20): yeni alanlar dolu, `rouge_l_ort` P4
   bandında, DECOMP kolon + sayaçlar görünür.
4. `scripts/anka_karsilastirma_tablosu.py` yeni kolonu okur.