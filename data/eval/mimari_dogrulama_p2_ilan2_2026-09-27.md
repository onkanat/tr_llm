# MİMARİ DOĞRULAMA PAKET-2 İLAN-2 — Onarım-Turu Yeniden-Doğrulama (T-0147)

**Damga (koşum ÖNCESİ, betikten):** 2026-09-27T14:32Z (UTC, `date -u`) — bu dosyanın
sha256'sı RAPOR'a işlenir (İLAN-1 kuralı; koşum-öncesi mtime).
**Onarım görevi:** T-0147 (Tur-A) — plan `.claude/plans/enchanted-wiggling-moon.md` (operatör onaylı)
**Yürütücü:** claude
**Kira:** T-0147 — `src/rag/`, `scripts/`, `data/eval/` (dir), `.agent-bus/notes/` (dir) — acquire ok:true (13:58:42Z)
**Önceki İLAN:** `mimari_dogrulama_p2_ilan_2026-09-27.md` (koşum-1; P2_GECTİ rc=0 — hüküm-JSON `9b217944…` SABİT-başvuru)

---

## 1. Bu İLAN-2 nedir

A-3 onarımının (skor-normalizasyon + OOV fail-closed) **koşum-öncesi beklenti-beyanıdır**
(T-0145 kalıbı: İLAN SABİT, ölçüm İLAN'a hizalanır; koşum-sonrası yumuşatma YOK).
Betik-revizyonu **yalnız ölçüm-hizalaması**:

- **K1 (FAZ-A):** koşum-1'de P2 `kristal_bellek`'i kurdu (36 nokta; kalıcı arz
  `fe7fc764…`) → sunucu-önce-durumu **10 koleksiyon / kanonik 1/2 / hibrit 1 /
  crystal_tagsli 1** beklenir (koşum-1 öncesi değerlerin 9/0/0/0 olması koşum-1
  kurulum-öncesi beyanıydı).
- **K2 (B0):** koşum-2 hedef-koleksiyonu **probe-adıdır** (`--koleksiyon
  p2_probe_bellek_t0147`) → "hedef-yok" kapısı probe-adı için geçerli kalır;
  `kristal_bellek` (36) yalnız-OKUMA (betik B0'da mevcut-hedefe DUR'da çıkar).
- **K6 (B3 OOV):** onarım-sonrası beklenti **fail-closed**: istisna-yok + skor
  ≤ 0,075 + `has_root_match=False` (koşum-1'deki skor=0,7 + koşullanma-TRUE
  sahte-sınıfı kapandı — kapı bu davranış-değişimini koşum-öncesi beyan eder).
- **K7 (dokunulmazlık):** probe-koleksiyon foreign-çıpa hesabından DIŞLANIR
  (koşum-başında yoktur; koşum-SONU betikçe silinir — İLAN'da beyan) → foreign-9
  çıpası koşum-1 birebir.

## 2. Onarım kapsamı (bu koşuma giren — T-0147 A-3)

`src/rag/vector_memory.py` `hybrid_recall`:
1. **Sabitançlı normalizasyon:** `RRF_SCORE_MAX = 0.75` (koşum-1 P2 bant-maks
   birebir-çıpa; ceza-ÖNCESİ ham-skora uygulanır — oran-korunan; kırpma YOK).
2. **OOV fail-closed:** boş `distinctive_query_roots` + dolu `doc_tags_str` →
   `has_root_match=False` + `×0,05` (ham 0,7 → 0,035 → normalize 0,0467).
3. **Kapı-değerleri DEĞİŞMEZ:** eşik 0,40 / Kapı-D 0,85 — anlamı ölçek-değişimiyle
   kazanır; sessiz-None (rag_pipeline :53-55) DOKUNULMAZ (davranış normalizasyonla azalır).

## 3. İLAN-2 beklentileri (birebir / bant)

| Kapı | Koşum-1 | **İLAN-2** | Gerekçe |
|---|---|---|---|
| K1 koleksiyon-önce | 9 | **10** | +kristal_bellek (koşum-1 arzı) |
| K1 kanonik-ad | 0/2 | **1/2** | kristal_bellek VAR; simulasyon_bellek YOK |
| K1 hibrit / crystal_tagsli | 0 / 0 | **1 / 1** | kristal_bellek hibrit-şemalı |
| K1 kristal_bellek-nokta | (yok) | **36** | arz-çıpa `fe7fc764…` (yalnız-okuma) |
| K2 hedef | kristal_bellek-yok | **probe-adı-yok** | onarım-turu probe'da koşar |
| K3 kurulum / K4 arz | hibrit-768 + 36/36 | **AYNI betik** (probe-koleksiyonda) | kurulum-yolu kanonik-import |
| K5 pozitif-kontrol | 20/20 top-1 | **20/20 AYNI** (skor-beyan: normalize-bant; top-1 ham-tavan 0,75 → 1,0) | skor-birebir-çıpa 4-ondalık koşum-1 P2-hüküm-skolarıyla birebir — determinizm (P3 B2 dersi) |
| K6 OOV | istisna-yok (skor 0,7 kayıt) | **istisna-yok + skor ≤ 0,075 + has_root_match=False** | A-3 fail-closed koşum-öncesi beklenti |
| K7 dokunulmazlık | foreign-9 birebir | **foreign-9 birebir (probe-dışlamalı)** + kanal 0-bayt | probe beyanlı-geçici |

**Etki-dışı beyan (hükme bağlanmaz):** eşik-üstü (0,40) sayısı onarım-sonrası
ARTABİLİR (koşum-1: 9/20 — normalize ile medyan 0,35 → 0,4667 eşik-üstüne çıkar);
bu kayıt raporda beyan-edilir, koşum-1 çıpasıyla birebir-beklenmez.
`kristal_bellek` 36 → 36 birebir (betik yükleme YAPMAZ — probe-adı hedef).

## 4. Hüküm (koşum ÖNCESİ sabit)

Tüm kapılar GEÇTİ → `P2_GECTİ` (rc=0); aksi her dal → `DUR` (rc=2). Hüküm BETİK
İÇİNDEDİR; elle sayı/hüküm YOK; koşum-sonrası İLAN-2 yumuşatılmaz.

## 5. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p2_rag_gezgini.py \
  --host 192.168.1.5 --port 6333 \
  --korpus data/realistic_rag/test_natural_150.jsonl \
  --vocab data/rebuild/vocab_anka_r1_33114.json \
  --koleksiyon p2_probe_bellek_t0147 \
  --ilan data/eval/mimari_dogrulama_p2_ilan2_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p2_onarim_sonuc_2026-09-27.md
```

- **Koşum sandbox DIŞI** (Qdrant trafiği sandbox proxy'sinden geçmez). Saf CPU+ağ;
  MPS YOK; çift-eğitici YOK; seed=42.
- **Hüküm-JSON yolu** `_sonuc_→_hukum_` türetimiyle: `mimari_dogrulama_p2_onarim_hukum_2026-09-27.json`
  — koşum-1 hüküm-JSON'u (`…_p2_hukum_2026-09-27.json`, commit `9ed84fb`) EZİLMEZ.
- **Koşum-sonu probe-temizliği betik-İÇİNDE** (hüküm-sonrası; rc'ye dokunmaz;
  arızası stderr+rapor-beyanı) — sunucu-sonu 10 koleksiyona döner.

## 6. DOKUNULMAZ / YAPILMAYANLAR

`kristal_bellek` 36 yalnız-okuma · 9 foreign koleksiyon birebir ·
`data/future_train_vector.jsonl` 0-bayt çıpa `e3b0c442…` · P3/P4/P5 hüküm-digest
SABİT · kanonik kod (betik-ici hüküm hariç) — onarım yalnız A-3-yüzeyi ·
commit ayrı operatör onayı · `git add -A` YASAK · push YOK.