# P2.5 Faz-2B — TriModalRouter v3 'legal' Sınıf Raporu (İLAN→HÜKÜM→RAPOR)

**Hüküm:** **P25_ROUTER_V3_LEGAL_GECTI** (betikten; deterministik ölçüm)
**Damga:** 2026-11-04T14:36:38Z (UTC)
**İlan:** `data/eval/p25_router_v3_ilan_2026-11-04.md` (sha256 `1be2d4160bef69928fd411911922ebc010b072f4152ce5719d8e0296de0e6aa4`)
**Hüküm-JSON:** `data/eval/p25_router_v3_hukum_2026-11-04.json`
**Çıktı:** `data/anka_router_v3.pt` — `data/anka_router_v2.pt` DEĞİŞMEDİ (devir çıpası)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 SIFIR-KILIÇ (güncel kaynak-sha birebir eşleşme) | `{"data/anka_taban_pure.pt": "5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "e32677b2a80bd611f7da17b815b3891d0831a0f45d065f1e7de76acd1321630d", "data/pedagogy/bahcivan_arena.jsonl": "9fa356b44ebfc7f6d48e387d7dceb12330b36287c94d95d0e8c15b66646baa13", "data/pedagogy/legal_canonical_sft.jsonl": "69670bc6ddbc033cff4a38a044ef0ccd5a82c278cfd8f564755ea70634734e9c", "data/pedagogy/legal_router_v3_dataset.jsonl": "713719ab8eac57e70851d2a34dbe1ce8b024` | GEÇTİ |
| CİHAZ (özellik-çıkarımı: MPS) | `{"cihaz": "mps"}` | GEÇTİ |
| K5 VERİ-ENVANTER (4×500/100 + legal 50/10; kesişim-0; legal>0) | `{"train_kirilim": {"grammar_core": 500, "pedagogy": 500, "carpenter": 500, "gardener": 500, "legal": 50}, "val_kirilim": {"grammar_core": 100, "pedagogy": 100, "carpenter": 100, "gardener": 100, "legal": 10}, "val_train_kesisim": 0, "legal_ornek": 60, "legal_distinct_train": 50, "legal_oversample": 10}` | GEÇTİ |
| K2 BASELINE (fresh-init eski-4-sınıf val top-1; kayıt-değer) | `{"baseline_val_top1": 0.0, "rastgele_taban_5sinif": 0.2}` | GEÇTİ |
| K3 EĞİTİM (eski-400 top-1>=0,95 VE baseline+0,50 VE kayıp-azalıyor) | `{"val_top1_400": 1.0, "val_top1_410": 1.0, "baseline": 0.0, "epoch_kayiplar": [0.2859, 0.0478, 0.0269, 0.0217, 0.0208, 0.0112, 0.0435, 0.0124, 0.0099, 0.0112], "train_per_sinif_dogru": {"grammar_core": 500, "pedagogy": 499, "carpenter": 500, "gardener": 499, "legal": 500}, "val_per_sinif": {"grammar_core": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "top2": true}, "pedagogy": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "top2": true}, "carpenter": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "r` | GEÇTİ |
| K7 REGRESYON NÖBETİ (v3 eski-sınıf doğruluğu v2'den düşük değil) | `{"v2_val_top1_400": 1.0, "v3_val_top1_400": 1.0, "v2_val_per_sinif": {"grammar_core": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "top2": true}, "pedagogy": {"correct": 100, "total": 100, "acc": 1.0, "precision": 0.9091, "recall": 1.0, "f1": 0.9524, "top2": true}, "carpenter": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "top2": true}, "gardener": {"correct": 100, "total": 100, "acc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "top2": true}}, "v3_val_per_sinif_eski": {"grammar_core": {"correct": 100, "to` | GEÇTİ |
| K8 LEGAL PROBE (v3 isabet>=8/10 VE >v2; probe∩veri=0) | `{"v2_legal_isabet": 0, "v3_legal_isabet": 8, "n_sorgu": 10, "probe_veri_kesisim": 0}` | GEÇTİ |
| K4 DETERMİNİZM (ikinci-koşum state_dict SHA birebir) | `{"sha_1": "4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5", "sha_2": "4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5"}` | GEÇTİ |
| K6 ŞEMA + ÇIKTI (5x256 + birebir anahtarlar + v3 yazımı + v2 değişmedi) | `{"anahtar_sayi": 9, "w_g_shape": [5, 256], "cikti": "data/anka_router_v3.pt", "cikti_sha256": "5e1c32b5203d25caf6cb1c2159e51e8f3566b08441c3c2a67335186a568190fa", "v2_sha_once": "74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2", "v2_sha_sonra": "74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2"}` | GEÇTİ |

## Probe Ölçüm Tablosu (v2 vs v3 — 10 held-out hukuk sorgusu)

| # | Sorgu (kısaltılmış) | v2 top-1 | v2 ağırlık | v3 top-1 | v3 ağırlık |
|---|---|---|---|---|---|
| 1 | Türk Ceza Kanunu'nda hırsızlık suçunun cezası ka… | pedagogy | 0.9998 | legal | 0.9919 |
| 2 | İş kanununa göre kıdem tazminatı hangi hallerde … | pedagogy | 0.9911 | legal | 1.0 |
| 3 | Kira sözleşmesinde fesih için bildirim süresi ne… | carpenter | 0.5804 | legal | 1.0 |
| 4 | Miras hukukunda yasal mirasçıların payları nasıl… | pedagogy | 0.9998 | legal | 0.9989 |
| 5 | Vergi Usul Kanunu'na göre gecikme faizi nasıl he… | pedagogy | 0.9926 | legal | 1.0 |
| 6 | Boşanma davasında velayet kararını hangi mahkeme… | pedagogy | 0.9997 | legal | 0.9999 |
| 7 | Türk Ticaret Kanunu'nda limited şirket kuruluşu … | pedagogy | 0.9982 | pedagogy | 0.7645 |
| 8 | İdare hukukunda iptal davası açma süresi ne kada… | pedagogy | 1.0 | legal | 0.9998 |
| 9 | Tüketicinin ayıplı mal satın alması halindeki ha… | pedagogy | 0.999 | pedagogy | 0.8263 |
| 10 | Ceza soruşturmasında şüphelinin ifade hakkı hang… | pedagogy | 0.9945 | legal | 1.0 |

- v2 legal isabeti: **0/10** — v3 legal isabeti: **8/10**

## Regresyon Nöbeti (eski 4 sınıf val — aynı özelliklerle v2 vs v3)

| Sınıf | v2 acc | v3 acc | Nöbet |
|---|---|---|---|
| grammar_core | 1.0 | 1.0 | OK |
| pedagogy | 1.0 | 1.0 | OK |
| carpenter | 1.0 | 1.0 | OK |
| gardener | 1.0 | 1.0 | OK |
| **top-1 (400)** | **1.0** | **1.0** | OK |

- T-0210 hüküm çıpası (v2, 2026-10-15): eski-4-sınıf val top-1 = 1.0 (köprü kaydı).

## Dosyalar

| Dosya | sha256 |
|---|---|
| `data/eval/p25_router_v3_ilan_2026-11-04.md` | `1be2d4160bef69928fd411911922ebc010b072f4152ce5719d8e0296de0e6aa4` |
| `data/eval/p25_router_v3_hukum_2026-11-04.json` | `7ab95bc64a4809022cb0e228632f469cf23dc7cbbd9b9a02f6dd8ddbc33fe958` |
| `data/anka_router_v3.pt` | `5e1c32b5203d25caf6cb1c2159e51e8f3566b08441c3c2a67335186a568190fa` |
| `data/pedagogy/legal_router_v3_dataset.jsonl` | `713719ab8eac57e70851d2a34dbe1ce8b024ae61df1e182c0d8be5284c0d4e8b` |
| `scripts/build_legal_router_dataset_p25.py` | `85188233ffeddc369ef93d40f91b477265ff9053f4898ad31091718c16ceca09` |
| `data/anka_router_v2.pt` (devir, DEĞİŞMEDİ) | `74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2` |

## Notlar

- Eğitim verisi: sınıf 0-3 kaynakları T-0210 ile birebir; legal = `legal_router_v3_dataset.jsonl` (60 kayıt).
- Legal dengesi: 50 distinct × 10 oversample = 500 etkin train satırı (diğer sınıflarla eş).
- Probe sorguları ve legal val örnekleri eğitim kümesinde yok (K8 kesişim kapısı).
- Canlıya devir (agent_gateway `router_state_path=data/anka_router_v2.pt` varsayılanı) bu görevin kapsamı DIŞINDA — operatör kararı.
