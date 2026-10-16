# T-0210 İLAN — TriModalRouter 5-Sınıf Saf Taban Model (anka_taban_pure.pt) Hizalaması

- **Tarih:** 2026-10-15
- **Görev:** T-0210
- **Hedef:** TriModalRouter'ı `data/anka_taban_pure.pt` model temsilleri ile hizalamak, deterministik olarak eğitmek, validation accuracy >= %95 sağlamak ve `data/anka_router_v2.pt` ağırlıklarını güncellemek.

## Sınıf Mimarisi (num_experts=5, top_k=2)

1. `0: grammar_core` (`data/pedagogy/lexical_semantics_dataset.jsonl`)
2. `1: pedagogy` (`data/pedagogy_canonical/*`)
3. `2: carpenter` (`data/pedagogy_canonical/carpenter_canonical.jsonl`)
4. `3: gardener` (`data/pedagogy/bahcivan_arena.jsonl`)
5. `4: legal` (ilerisi için ayrık sınıf)

## Kaynak Çıpaları (SHA-256)

- `data/anka_taban_pure.pt`: `5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360`
- `data/rebuild/vocab_anka_r1_33114.json`: `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`
- `data/lexicon/roots.tsv`: `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`
- `data/pedagogy/bahcivan_arena.jsonl`: `9fa356b44ebfc7f6d48e387d7dceb12330b36287c94d95d0e8c15b66646baa13`

## Hüküm Kapıları

- **K1 SIFIR-KILIÇ:** Kaynak dosyaların SHA-256 özetleri çıpalarla birebir eşleşir.
- **K2 BASELINE:** Fresh-init seed-42 router'ın doğruluk ölçümü yapılır (4 aktif sınıf için rastgele taban ~0.25).
- **K3 EĞİTİM:** Validation top-1 doğruluğu >= 0.95 VE >= baseline + 0.50 VE son epoch kaybı ilk epoch kaybından düşüktür.
- **K4 DETERMİNİZM:** İkinci eğitim koşumu (aynı seed, CPU/MPS) sonucunda elde edilen state_dict SHA-256 özeti ilk koşumla bit-özdeştir.
- **K5 VERİ-ENVANTERİ:** 4 aktif sınıfın her biri için 500 train + 100 validation örneği (toplam 2000 train, 400 val); validation ve train arasında 0 kesişim; legal sınıfından 0 örnek.
- **K6 ŞEMA VE ÇIKTI:** `TriModalRouter(num_experts=5)` kanonik state_dict anahtar kümesi ile tam eşleşir; `data/anka_router_v2.pt` diske yazılır.
