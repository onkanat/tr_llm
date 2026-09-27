# MİMARİ DOĞRULAMA PAKET-1 İLAN-2 — Onarım-Turu Yeniden-Doğrulama (T-0147)

**Damga (koşum ÖNCESİ, betikten):** 2026-09-27T14:15Z (UTC, `date -u`) — bu dosyanın sha256'sı RAPOR'a işlenir (İLAN-1 kuralı; koşum-öncesi mtime).
**Onarım görevi:** T-0147 (Tur-A: P1–P3 bulgu-onarımı) — plan `.claude/plans/enchanted-wiggling-moon.md` (operatör onaylı)
**Yürütücü:** claude
**Kira:** T-0147 — `src/compiler/`, `src/rag/`, `tests/`, `scripts/`, `data/eval/` (dir), `.agent-bus/notes/` (dir) — acquire ok:true (13:58:42Z)
**Önceki İLAN:** `mimari_dogrulama_p1_ilan_2026-09-27.md` (koşum-1; DUR rc=2 — İLAN'lı, koşum-sonrası yumuşatılmadı)

---

## 1. Bu İLAN-2 nedir

Onarım-SONRASI yeniden-doğrulama turunun **koşum-öncesi beklenti-beyanıdır** (T-0145
kalıbı: İLAN SABİT, ölçüm İLAN'a hizalanır). Betik-revizyonu **yalnız ölçüm-hizalaması**
(koşum-1'in İLAN'lı-tar-beklentisi ile onarım-sonrası davranış farkı — rapor §7'ye
kayıtlı iki kusur sınıfının operatör kararıyla onarım-kapsamına alınması).

## 2. Onarım kapsamı (T-0147 Tur-A — bu koşuma giren)

| Onarım | Dosya | Koşum-1 kusur-sınıfı |
|---|---|---|
| **A-1** affix_info çok-şablonlu + çözümle-seç | `src/compiler/decompiler.py` (DONMUŞ-desen; plan-onayı) | YOL_0 156 (GERUND_KEN 154 + GERUND_IncA 1) |
| **A-2** ikiz-satır `is_case_alias` bayrağı | `src/compiler/lexicon.py` (DONMUŞ-desen; plan-onayı) | BIT_UYUMSUZ 59 (%100 apostrof) |
| **A-3** RRF_SCORE_MAX normalizasyon + OOV fail-closed | `src/rag/vector_memory.py` | P2/P3 kapsamı (bu koşuma girmez) |

`morphotactics.py` · `phonology.py` · `roots.tsv` · `src/llm/tokenizer.py` DOKUNULMAZ.
TSV **değişmediği** için KSUR-3 (79/70) ve KSUR-1 ADJ (3.627/6.352) envanteri
**birebir KORUNUR** — onarım davranışı onarır, veriyi değil.

## 3. FAZ-B birincil kapı — İLAN-2 hedefi

**%100 (0 düşen örnek, çekimlenebilir sınıf)** — İLAN-1 §2 ile AYNI kapı, aynı betik.
Onarım-sonrası YOL_0 + BIT_UYUMSUZ sınıflarının **boşalması beklenir**:
- YOL_0'ın 154+1'i A-1 çözümle-seç ile (`ricayken` doğru yüzey — çok-şablonlu
  GERUND_KEN artık derleyicinin kabul ettiği şablonla çözülür);
- BIT_UYUMSUZ 59'u A-2 ile (kanonik küçük-harfli entry öne alınır + apostrof-dalı
  yalnız gerçek öz-adı tetikler).

**Yeni kusur sınıfı beklentisi:** onarım **yeni bir** düşen-sınıf üretirse
(ör. çözümle-seç'in yeni bir çakışması) — fail-closed DUR (İLAN-1 kuralı aynen).

## 4. FAZ-C — İLAN-2 beklentileri (birebir)

| Sınıf | İLAN-1 | **İLAN-2** | Gerekçe |
|---|---|---|---|
| KSUR-1 ADJ | 3.627 / 6.352 | **AYNI** | TSV+morphotactics değişmedi |
| KSUR-1 probe | 50/50 sessiz-boş | **44/50** | 6 yol-bulanan koşum-1 §7'de MEŞRU alternatif-kok çözülemesi olarak kayıtlı (`abuklar` VERB `abukla-`+AORIST; `ademimerkeziyetçiler` +DERIV_CI; `akışkanlar` ayrı lemma; …). A-1/A-2 bu probe'u etkilemez (yalnız-ADJ+PLURAL) → 44/50 birebir çıpa; seed-42 koşumlar-arası kararlı (koşum-1 teyidi) |
| KSUR-2 | %100 kesmeli | **%100 kesmeli (KORUNUR)** | A-2 apostrof-dalını yalnız `is_case_alias` kökte söndürür; probe büyük-harfli ANAHTAR-düğümüne gider (bayraksız-kanonik) → davranış korunur |
| KSUR-3 | 79 / 70 | **79 / 70 (KORUNUR)** | TSV değişmedi; bayrak yalnız trie-entry düzeyinde |
| KSUR-4 | 200/200 sessiz-boş | **AYNI** | compile istisna-fırlatmaz davranışı değişmedi |

FAZ-A: istisna-0 İLAN'lı davranış **AYNI**; yol-bulma oranları hükme bağlanmaz
(İLAN-1 §3 kuralı korunur).

## 5. Hüküm (koşum ÖNCESİ sabit — İLAN-1 kuralı değişmez)

FAZ-B %100 + FAZ-C birebir → `P1_GECTİ` (rc=0); aksi her dal → `DUR` (rc=2).
Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK; koşum-sonrası İLAN-2 yumuşatılmaz.

## 6. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p1_roundtrip.py \
  --lexicon data/lexicon/roots.tsv \
  --ilan data/eval/mimari_dogrulama_p1_ilan2_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p1_onarim_sonuc_2026-09-27.md \
  --hukum-json data/eval/mimari_dogrulama_p1_onarim_hukum_2026-09-27.json
```

- **Koşum-1 hüküm-JSON'u (`mimari_dogrulama_p1_hukum_2026-09-27.json`) EZİLMEZ** —
  ayrı yol (koşum-1 kanıtı commit `7c7848b`'de); "ilan==rapor yolu ön-kaydı ezer" dersi.

- Hüküm-JSON: `data/eval/mimari_dogrulama_p1_onarim_hukum_2026-09-27.json` (betikten).
- Saf CPU, sandbox İÇİNDE koşabilir; Qdrant GEREKMEZ; GERÇEK kanal
  `data/future_train_vector.jsonl` bu koşumla hiç ilişkili değil (0-bayt çıpa korunur).
- Koşum-1/3 birebir-çıpa: FAZ-A sayıları, KSUR-3, KSUR-4 koşum-1 ile birebir beklenir
  (betik-determinizmi) — sapma yeni betik-kusuru veya onarım-yan-etkisi → DUR.

## 7. DOKUNULMAZ / YAPILMAYANLAR

`morphotactics.py` · `phonology.py` · `roots.tsv` (79 anormal satır davranış-onarımıyla
çözülür — veri-değil-davranış) · `src/llm/tokenizer.py` imza-koruma (find_stems/compile
dönüş-şekli) · commit ayrı operatör onayıyla · `git add -A` YASAK · push YOK.