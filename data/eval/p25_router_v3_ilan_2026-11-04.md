# P2.5 Faz-2B İLAN — TriModalRouter v3: 'legal' Sınıfı Örnek>0 + Yeniden Eğitim

- **Tarih:** 2026-11-04
- **Kabul ölçütü (P2.5):** router legal sınıfı örnek>0.
- **Görev:** anka_router_v2.pt'de legal sınıfı 0 örnekle eğitildi (T-0210 hükmü K5: `legal_ornek: 0`) ve 10 hukuk sorgusunda 0/10 legal der (ölçüm: scratch/p25_probe_v2.py). v3, legal sınıfına örnek kazandırıp router'ı yeniden eğitir; çıktı `data/anka_router_v3.pt` — `anka_router_v2.pt` DEĞİŞMEZ (devir çıpası).

## Veri

- Sınıf 0-3 (grammar_core/pedagogy/carpenter/gardener): T-0210 kaynakları ve 500 train + 100 val seçimi (SEED-42) BİREBİR korunur.
- Sınıf 4 (legal): `data/pedagogy/legal_router_v3_dataset.jsonl` — 60 benzersiz kayıt (50 `legal_canonical_sft.jsonl` istemi, SEED-42 + 10 el-derlemesi doğal hukuk sorgusu). Kırılım: 50 distinct train (×10 oversample → 500 etkin; sınıf-başı dengeli) + 10 val. Üretici: `scripts/build_legal_router_dataset_p25.py` (deterministik, SHA-çupalı).
- Probe: 10 held-out hukuk sorgusu eğitim/val verisinde YOK (kesişim kapısı K8'de ölçülür).

## Eğitim Kalıbı (T-0210 ile aynı; EPOCH 5→10)

Özellikler: anka_taban_pure.pt emb-ort (P) + CuriosityEngine q_merak (Q); router TriModalRouter(768/768/768, 256, num_experts=5, top_k=2); AdamW lr=1e-3, batch 32, SEED-42. Epok sayısı 5→10: 5-sınıflı sette 5. epokta kayıp hâlâ azalıyor ve eski-sınıf sınırı oturmamış (v3-e5: gardener 0,98 — K7 düşer); 10. epokta 400-val'de v2-paritesi (1,0) ölçüldü, 15'te aşırı-uyum (carpenter 0,99). Varyant teşhisi: `scratch/p25_diagnostic.py` (e5/e10/e15/ov5e10, tek özellik setiyle).

## Hüküm Kapıları

- **K1 SIFIR-KILIÇ:** Güncel kaynak SHA-256 çıpaları (roots.tsv T-0210'dan beri e32677b2…; P2.8: 3/400 encode drifti taban sonuçlarını değiştirmez; K7 v2'yi aynı güncel özelliklerle yeniden ölçer → karşılaştırma öz-tutarlı).
- **K5 VERİ-ENVANTER:** Eski 4 sınıf 500/100; legal 50 distinct train + 10 val; train/val kesişimi 0; **legal_ornek > 0**.
- **K2 BASELINE:** Fresh-init val doğruluğu (kayıt-değer).
- **K3 EĞİTİM:** Eski-4-sınıf val top-1 >= 0.95 VE >= baseline + 0.50 VE son epoch kaybı ilk epochtan düşük.
- **K7 REGRESYON NÖBETİ:** anka_router_v2.pt aynı özelliklerle yüklenip ölçülür; v3 eski-4-sınıf her sınıfta ve top-1'de v2'den DÜŞÜK DEĞİL (v2 hüküm çıpası: val_top1=1.0).
- **K8 LEGAL PROBE:** 10 held-out hukuk sorgusunda v3 legal isabeti >= 8 VE > v2 isabeti; eğitim/val ∩ probe kesişimi 0.
- **K4 DETERMİNİZM:** İkinci koşum state_dict SHA birebir.
- **K6 ŞEMA + ÇIKTI:** Kanonik anahtar kümesi + w_g 5×256 + `data/anka_router_v3.pt` yazımı (v2 YAZILMAZ).
