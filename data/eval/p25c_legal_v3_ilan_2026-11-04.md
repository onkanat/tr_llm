# P2.5 FAZ-2C — LEGAL CEKETİ v3: BÜYÜK VERİ + REPLAY — İLAN (kapılar koşumdan önce donduruldu)

**Tarih:** 2026-11-04 · **Repo:** ~/Git/tr_llm · **Cihaz:** MacBook Air M2 (MPS, 8 GB)
**Deliverable:** `data/pedagogy/legal_canonical_sft_v2.jsonl` (ÜRETİLDİ, kanıtlı) + `data/anka_legal_v3.pt` (YENİ —
`anka_legal.pt`/`anka_legal_replay.pt`'ye YAZILMAZ) + `data/eval/p25c_legal_v3_hukum_2026-11-04.md`.
Commit YOK. Qdrant YOK. Mevcut dosyalara yazma YOK (yalnız yeni dosyalar).

---

## 1. Ölçülmüş zemin (P2.5 FAZ-2A + FAZ-2C hüküm dosyalarından, birebir)

| Koşum | Tarif | SFT train | ΔCE (val-50) | kart4 | etiket16 | OOD uyum32 (ort) |
|---|---|---|---|---|---|---|
| TABAN pure | — | — | — | 0/20 | 0/20 | (öz 1,00) |
| tarif | 1e-5, 3e, saf SFT | 413 | −0,048 | 0/20 | 0/20 | 0,7083 |
| p26lr | 5e-5, 3e, saf SFT | 413 | −0,194 | 0/20 | 0/20 | 0,4375 |
| tani12e | 5e-5, 12e, saf SFT | 413 | −0,463 | 1/20 | 0/20 | 0,1250 |
| REPLAY (2C) | 5e-5, 10e, %19,95 pencere + KL β=0,2 | 313 | −0,306 | 0/20 | 0/20 | 0,3646 |

Taban val-50 CE çıpası: **4,782289** (`p25_legal_taban_cipasi_2026-11-04.json`; 2A/2C koşumlarında
yeniden ölçümle 0,000000 farkla çapraz-doğrulandı). Ölçüm rigi: `scripts/train_legal_ceket.py` —
masked SFT val CE (`FastSampleAlignedDataset`, BLOCK=128), **val = dosyanın son-50 kaydı**,
tokenizer `literal_entity_mode=True` (FAZ-1 sözleşmesi); kanonik-20 tohum-42 sabit örneklem;
OOD-3 = `marangoz#0`, `marangoz#1`, `bahcivan#0`, üretim maks 48, uyum = ilk-32 jeton örtüşmesi.

Ölçülen teşhis (2C hükmü): **kart edinimi VERİ ölçeğine bağlı** — 463 kayıtta üç koşum da 0/20;
marangoz 6.123 kayıtta kart öğrenmiş. Bu koşumun amacı aynı hipotezi legal alanda büyütülmüş
veriyle sınamak.

## 2. Veri: `legal_canonical_sft_v2.jsonl` (üretildi — bu İLAN'la dondu)

- **4.813 kayıt** = 4.350 YENİ + 463 eski (eski dosya satırları BAYT-BİREBİR kuyruk blok).
  Üretici: `scripts/prepare_legal_dataset_v2.py` (FAZ-1 modülü import edilir — kapılar aynen).
- Tip dağılımı (yeni): a_kavram_tanimi 154 · b_madde_sorusu 3.244 (PER_LAW_CAP kaldırıldı) ·
  b3_tum_madde 808 (YENİ: b1 dışı tüm madde satırları; 490 Geçici/Ek) · c_kavram_ayrimi 145
  (otomatik bent çiftleri) · d_madde_dizisi 157 (ardışık madde alıntısı, usul iddiası yok).
  Eski 463'nin tip dağılımı değişmez (meta kütüğünde ikisi de kayıtlı).
- **Uydurma kapıları aynen:** her alıntı korpus madde satırından (satır-satır); mülga/mezar ve
  bent-biçim filtreleri FAZ-1 birebir; şema `{instruction,input,output}`; dedupe yeni-yeni VE
  yeni-eski463 (sha256(instruction||output)); 158 duplik düşüş kayıtlı.
- **Token/UNK (mode=True, anka 33.114):** 877.721 token · UNK 16.049 (%1,83) · ort. 182/kayıt
  (FAZ-1: 80.455 token, %1,62 → 10,9× büyüme).
- **VAL TUTARLILIK SÖZLEŞMESİ (görev şartı):** v2 son-50 == FAZ-1 son-50 **BİREBİR**
  (programatik: `True`); eski 463 blok == eski dosya **BİREBİR** (`True`). Yeni kayıtlar başa
  eklendi. Böylece hem 2A-rig val-50 hem CLI val-150 (son-150 = eski 313-462) önceki koşumlarla
  özdeş kalır — K1 karşılaştırılabilirliği korunur.
- sha256 (programatik): v2 = `1c822c9f41bd0896f65ed85ff24f59d883346b46f26f8ba6c58922ad2bf6524f`;
  eski 463 = `69670bc6ddbc033cff4a38a044ef0ccd5a82c278cfd8f564755ea70634734e9c` (2C hükmündekiyle
  birebir — eski veri değişmedi).

## 3. DONMUŞ KAPILAR (ölçüm §5 rig'iyle; koşumdan önce donduruldu)

- **K1 (val CE):** v3 artefaktının 2A-rig val-50 masked SFT CE'si **≤ 4,782289 − 0,50 = 4,282289**
  (Δ = final − taban ≤ −0,50 nat; val-50 özdeş olduğundan Δ doğrudan 4,782289'ye karşı ölçülür).
- **K2 (kanonik-20):** tohum-42 örnekleminden **≥ 5/20** kayıtta ilk-16 jeton birebir
  (2A/2C metriği **etiket16**; kart4 ikincil olarak ayrıca raporlanır).
  **Nüfus beyanı (bilinçli):** örneklem v2 kayıt listesinden çekilir — nüfus 463→4.813 olduğundan
  idx listesi 2A/2C'dekinden FARKLI olur (tohum 42, n=20, rig aynı). Kapı kuralı aynı kalır.
- **K3 (OOD argmax):** 3 sabit alan-dışı sorguda taban-pure ilk-48 argmax üretimiyle ilk-32 uyum
  ortalaması **≥ 0,40** (2C İLAN'ıyla aynı bar; KL+pencere orta bantta tutma hedefi).
  OOD CE deltası bağlam için ayrıca raporlanır.

HÜKÜM: üç kapı da geçerse GEÇTİ; herhangi biri kalırsa KALDI + hangi kapı kaç nat/puan açık.

## 4. Donmuş koşum tarifi (betik DEĞİŞTİRİLMEDEN — `scripts/train_p26_sft_replay.py` CLI)

```
venv/bin/python scripts/train_p26_sft_replay.py \
  --sft-path data/pedagogy/legal_canonical_sft_v2.jsonl \
  --init-model data/anka_taban_pure.pt \
  --teacher-model data/anka_taban_pure.pt \
  --pencere-bin data/kulliyat_v2/kulliyat_v2.bin \
  --pencere-n 1166 \
  --kl-beta 0.2 \
  --peak-lr 5e-5 --min-lr 1e-5 --warmup 20 \
  --epochs 3 \
  --seed 42 \
  --save-path data/anka_legal_v3.pt --allow-frozen-write \
  --history-path data/eval/p25c_legal_v3_history.json \
  --hukum-path data/eval/p25c_legal_v3_hukum_2026-11-04.json
```

Tarif gerekçeleri (koşumdan önce beyan):

1. **--pencere-n 1166 = train/4.** Betiğin val dilimi `recs[-150:]` → v2'de train = 4.813−150 =
   4.663. Görev talimatı: karışım ~%20 → pencere-n = round(4663/4) = **1166** →
   1166/(4663+1166) = **%20,003** (2C'deki 78/(313+78)=%19,95 kuralının aynısı).
2. **--epochs 3 (görev: 2-3 epok).** SFT maruziyeti 4.663×3 = 13.989 satır-geçişi — tani12e'nin
   4.956'sının 2,8 katı, REPLAY'in 3.130'unun 4,5 katı. Nöbet + artefakt seçimi (§4b) aşırı-eğitimi
   keser; koşum ~549 opt-adım (183/epok), 2C ölçümüyle ~2,1 s/adım → ~20 dk.
3. **Epok-sonu val nöbeti POST-HOC:** betik (satır 335-338) yalnız `_e{N}.pt` ara kaydı yazar;
   epok-sonu değerlendirmeyi DESTEKLEMİYOR (2C İLAN 1.3-3 ile aynı bulgu). Nöbet, §4b artefakt
   seçim kuralıyla betik-dışı karşılanır.
4. **Artefakt-seçim kuralı (nöbet semantiği; koşumdan önce donduruldu):** koşum `_e1.pt`,
   `_e2.pt` ara kayıtlarını + finali (`anka_legal_v3.pt`) yazar. Her biri 2A-rig val-50 CE ile
   ölçülür; **artefakt `data/anka_legal_v3.pt` = en düşük val-50 CE'li epok kaydıdır** (2A/2C'nin
   en-iyi-epok semantiği). En iyi final değilse ara kayıt `anka_legal_v3.pt`'ye kopyalanır
   (yeni dosya — donmuş-yol kapısı `--allow-frozen-write` ile). K1/K2/K3 bu artefakta ölçülür.
5. **Tokenizer modu farkı (bilinçli, 2C İLAN 1.3-4 ile aynı):** betik `KristalTokenizer(comp,
   vocab)` kurar → mode=False; ölçüm rigi mode=True. 2C'de ölçülen etkisi −0,002 nat (val CE'yi
   açıklamıyor); K1 yorumunda hatırlatılacak.
6. **Val dilimi:** betiğin val-150'ü = v2 son-150 = eski kayıt 313-462 — REPLAY koşumunun
   val-150'siyle BİREBİR (CLI içi karşılaştırılabilirlik); 2A-rig ölçümü yine son-50 ile.
7. Betiğin `!= 4498` uyarısı (T-0209 sabiti) basılır — bilgi amaçlı, koşumu etkilemez.

## 5. Ölçüm rigi kimliği

- `train_legal_ceket.py` DEĞİŞTİRİLMEDİ — değerlendirme kodu scratch kopyada
  (`~/.hermes/cache/scratch/p25c_v3_olcum_rig.py`) import yoluyla kullanılır; model kaydı ve SFT
  yolu parametreleştirilir (`anka_legal_v3.pt`, `_e{N}.pt`, `legal_canonical_sft_v2.jsonl`).
- Taban-pure çıpası bu koşumda yeniden ölçülür; `p25_legal_taban_cipasi_2026-11-04.json` ile
  val CE + OOD gen_ids + kanonik özet çapraz-doğrulanır (2C'de: val CE farkı 0,000000, OOD 3/3
  birebir). v2 son-50 = eski son-50 olduğundan taban val-50 CE'si de 4,782289'yi yeniden
  üretmelidir — programatik kanıt hükümde raporlanır.
- Tüm sha256 değerleri `compute_sha256`/`hashlib` ile programatik üretilir, elle dökülmez.

## 6. Çıktılar (tümü YENİ dosya)

`data/pedagogy/legal_canonical_sft_v2.jsonl` (+meta) · `scripts/prepare_legal_dataset_v2.py` ·
`data/anka_legal_v3.pt` (+`_e1.pt`,`_e2.pt` ara kanıtlar) · `data/eval/p25c_legal_v3_history.json` ·
`data/eval/p25c_legal_v3_hukum_2026-11-04.json` · `data/eval/p25c_legal_v3_egitim_log_2026-11-04.txt` ·
`data/eval/p25c_legal_v3_olcum_rig_2026-11-04.json` · `data/eval/p25c_legal_v3_hukum_2026-11-04.md`.
BU İLAN dosyası: `data/eval/p25c_legal_v3_ilan_2026-11-04.md` (yukarıdaki hüküm MD'sine kaynak).