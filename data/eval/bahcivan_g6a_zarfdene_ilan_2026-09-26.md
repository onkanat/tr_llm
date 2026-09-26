# G6a (T-0137) İLAN — ÜRETİM TANISI: ROL-ZARF YÖLLENDİRME DENEMESİ — 2026-09-26 (claude)

Operatör emri (2026-09-26): "system_message ile yönlendirme denensin — Marangozda
işe yaramıştı". Kardeş kanıt: T-0114 (rol zarfi koşullama kiliti azalttı —
S3 tutarsızlık −15 pp) · T-0115 (zarflı ROUGE korundu — 0,1342 zarflı / 0,1354
nötr). ILAN ≠ RAPOR: rapor `bahcivan_g6a_zarfdene_sonuc_2026-09-26.md`.
Hüküm BETİKTEN (elle sayı YOK). Bu TANISAL deneydir — A9/A10 kabul ayracı
DEĞİŞMEZ; kabul birleşimi nötr ölçümdendir (KABUL_YOK). Çıpa karıştırma
yasağı (İLAN-3b §5) A9 ARTİS karşılaştırması içindir; bu deney ayracaç değil,
kilit mekanizması tanısıdır ve ayrı damgayla koşulur.

## Deney beyanı (koşum ÖNCESİ)

- **Koşum (kanonik kap DOKUNULMAZ — import yok, CLI):**
  `evaluate_carpenter_anka.py --model scratch/t0137_g6a_kos/seg_3.pt --heldout
  scratch/t0137/bahcivan_heldout.jsonl --train-source
  scratch/t0137/bahcivan_egitim_adapter.jsonl --ceket-ekseni --rol-zarf
  --output scratch/t0137/bahcivan_ft_bahcivan_zarfli.json --device mps`
  (n=100, seed 42; ÖLÇÜM-2 parametreleri birebir — tek fark `--rol-zarf`;
  sandbox DIŞI; koşum öncesi EGITICI_YOK kanıtı).
- **Zarf-lı ikiz:** ROL_ZARFI (prompt_contract, marangoz metni) instruction
  başına eklenir (T-0120 salt-eklenti; bit-özdeş davranış kanıtlı).
- **Karşılaştırma ikizi (mevcut, nötr):** `bahcivan_ft_bahcivan_sonda.json`
  (`be84f93e…`) — ROUGE 0,0365 · tutarsızlık %43,0 · üretim ort 61,32 kelime.
- **Hüküm betiği** `scratch/t0137/bahcivan_zarfdene_hukum.py`: iki sonda
  JSON'dan ROUGE farkı · tutarsızlık farkı · üretim-uzunluk farkı (ham_gm)
  BETİKTEN; **dallar (tanı etiketi — kabul kararı DEĞİL):**
  - tutarsızlık zarflı < nötr → **KILIT_AZALDI**; değilse **KILIT_AZALMADI**
  - ROUGE zarflı > nötr → **ROUGE_ARTTI**; |fark| küçük → **ROUGE_KORUNDU**
    (eşik beyanı: fark ≥ +0,005 sayılabilir artış; betik sabiti); azalma →
    **ROUGE_DUSTU**
  - üretim-uzunluk zarflı < nötr → **UZUNLUK_KILANDI**; değilse **UZUNLUK_AYNI**

## DOKUNULMAZLAR

`evaluate_carpenter_anka.py` / `ROL_ZARFI` / tokenizer / compiler DOKUNULMAZ ·
çıpa 0,0236 + kapı 0,0230 değişmez (A9 nötr ölçümdendir) · ESIK_ROUGE 0,3221 /
TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · checkpoint yazımı YOK · data/**
salt-okunur · hüküm BETİKTEN · `git add -A` YASAK.