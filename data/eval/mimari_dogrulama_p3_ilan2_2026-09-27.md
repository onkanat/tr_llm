# MİMARİ DOĞRULAMA PAKET-3 İLAN-2 — Onarım-Turu Yeniden-Doğrulama (T-0147)

**Damga (koşum ÖNCESİ, betikten):** 2026-09-27T14:34Z (UTC, `date -u`) — bu dosyanın
sha256'sı RAPOR'a işlenir (İLAN-1 kuralı; koşum-öncesi mtime).
**Onarım görevi:** T-0147 (Tur-A) — plan `.claude/plans/enchanted-wiggling-moon.md` (operatör onaylı)
**Yürütücü:** claude
**Kira:** T-0147 — `src/rag/`, `scripts/`, `data/eval/` (dir), `.agent-bus/notes/` (dir) — acquire ok:true (13:58:42Z)
**Önceki İLAN:** `mimari_dogrulama_p3_ilan_2026-09-27.md` (koşum-1; P3_GECTİ rc=0 — hüküm-JSON `24492949…` SABİT-başvuru)

---

## 1. Bu İLAN-2 nedir

A-3 onarımının (skor-normalizasyon + OOV fail-closed) P3-yüzeyindeki
**koşum-öncesi beklenti-beyanıdır** (T-0145 kalıbı: İLAN SABİT, ölçüm İLAN'a
hizalanır; koşum-sonrası yumuşatma YOK). Betik-revizyonu **yalnız ölçüm-hizalaması**:

- **K4 (B3 OOV):** koşum-1 kapısı "skor 0,7-bandı + koşullanma-TRUE"
  (SAHTE-koşullanma beyanıydı) → onarım-sonrası **tersine**: skor ≤ 0,075 +
  koşullanma-FALSE + istisna-yok (davranış-değişimi koşum-öncesi İLAN'lı).
- **K5 (B4 Kapı-D):** koşum-1 kapısı "is_high 0/20 (0,85 > bant-maks 0,75
  ULAŞILAMAZ)" → onarım-sonrası normalizasyonla top-1 ham-tavan 0,75 → 1,0;
  Kapı-D **ERİŞİLEBİLİR** olur → kapı "is_high ≥ 1/20" olarak TERSİNE döner
  (birebir-sayı RAPOR-kaydı; koşum-1'de 0/20 birebir-beklentisi onarım-sonrası
  gereksiz-beyandır — gereççesi ortadan kalktı).

## 2. Onarım kapsamı (bu koşuma giren — T-0147 A-3)

`src/rag/vector_memory.py` `hybrid_recall`: `RRF_SCORE_MAX=0,75` sabitançlı
normalizasyon (ceza-öncesi ham-skora; kırpma YOK) + OOV fail-closed (boş-roots +
dolu-doc-tags → `has_root_match=False` + `×0,05`). **Kapı-değerleri DEĞİŞMEZ**
(0,40 / 0,85); sessiz-None DOKUNULMAZ.

## 3. İLAN-2 beklentileri (birebir / bant)

| Kapı | Koşum-1 | **İLAN-2** | Gerekçe |
|---|---|---|---|
| FAZ-A eşik-tanım/import | 1 + 2 | **1 + 2 (AYNI)** | A-3 eşik-değere DOKUNMAZ |
| K2 arz | kristal_bellek 36→36 + foreign-çıpa | **AYNI** (yalnız-okuma) | betik kanonik-koleksiyona YAZMAZ |
| K3 B1 kayıt | 20/20 istisna-0 (tetiklenme kayıt) | **AYNI kapı** (tetiklenme ≥16 — normalize-ölçek kayıt) | istisna-0 davranışı değişmez |
| K4 OOV | skor 0,6–0,8-bandı + koşullanma-TRUE | **skor ≤ 0,075 + koşullanma-FALSE** | A-3 fail-closed koşum-öncesi beklenti |
| K5 Kapı-D | is_high 0/20 (erişilemez) | **is_high ≥ 1/20** + probe-sema + kanal-0-bayt | Kapı-D onarım-sonrası ERİŞİLEBİLİR (top-1 1,0 ≥ 0,85) |
| B6 | — | probe-agent (sim=0,5) is_high-beyanı RAPOR-kaydı | hükme bağlanmaz |
| K6 dokunulmazlık | foreign 9 + kanonik 36→36 | **AYNI** | koşum kanonik-koleksiyona YAZMAZ |

**Determinizm-çıpa:** pozitif-kontrol/kayıt-skoları koşum-1 P2/P3-hüküm-skolarıyla
birebir (4-ondalık) birebir-beklenir — normalize-oran-korunan (skor_new = ham/0,75)
→ koşum-1 skoru × 1,3333… birebir-türevi; betik 4-ondalık kayıtları RAPOR'da.

## 4. Hüküm (koşum ÖNCESİ sabit)

Tüm kapılar GEÇTİ → `P3_GECTİ` (rc=0); aksi her dal → `DUR` (rc=2). Hüküm BETİK
İÇİNDEDİR; elle sayı/hüküm YOK; koşum-sonrası İLAN-2 yumuşatılmaz.

## 5. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p3_epistemik_kapilar.py \
  --host 192.168.1.5 --port 6333 \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --korpus data/realistic_rag/test_natural_150.jsonl \
  --koleksiyon kristal_bellek \
  --probe-yol data/eval/p3_future_train_probe_t0147.jsonl \
  --ilan data/eval/mimari_dogrulama_p3_ilan2_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p3_onarim_sonuc_2026-09-27.md
```

- **Koşum sandbox DIŞI** (Qdrant trafiği sandbox proxy'sinden geçmez). Model saf
  CPU'da; MPS YOK; çift-eğitici YOK; seed=42.
- **Probe-yolu YENİ** (`p3_future_train_probe_t0147.jsonl`) — koşum-1 probe-dosyası
  (`p3_future_train_probe.jsonl`, commit `7c7848b`) EZİLMEZ.
- Hüküm-JSON `_sonuc_→_hukum_` türetimiyle `mimari_dogrulama_p3_onarim_hukum_2026-09-27.json`
  — koşum-1 hüküm-JSON'u EZİLMEZ.
- GERÇEK kanal `data/future_train_vector.jsonl` 0→0 BAYT `e3b0c442…` (çıpa).

## 6. DOKUNULMAZ / YAPILMAYANLAR

`kristal_bellek` 36 yalnız-okuma · 9 foreign koleksiyon birebir ·
P2/P4/P5 hüküm-digest SABİT · `morphotactics.py` · `phonology.py` · roots.tsv ·
`src/llm/tokenizer.py` · commit ayrı operatör onayı · `git add -A` YASAK · push YOK.