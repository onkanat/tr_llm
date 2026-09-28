# T-0155 — gateway router entegrasyonu (sonuç)

**Hüküm:** **T0155_ENTEGRASYON_GECTI** (betikten; elle sayı YOK)
**Damga:** 2026-09-28T07:16:03Z (UTC) — koşum sonu
**İlan:** `data/eval/t0155_router_entegrasyon_ilan_2026-09-28.md` (sha256 `fa41d0986ae4e8a1994974c0f34e8736785be80dacb2ee3d5d307f37d8a6c32d`)

## Kapılar

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 KAYNAK-ÇIPASI (4 dosya tam-digest birebir) | `{"data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50", "data/rebuild/vocab_anka_r1_33114.json": "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984", "data/lexicon/roots.tsv": "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598", "data/anka_router.pt": "64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720", "sha_sapma": []}` | GEÇTİ |
| K5 GATEWAY AST ZİNCİR-KAPISI (router_state_path açık-beyan) | `{"create_default_param": true, "cls_cagrisi_anahtari": true, "init_param": true, "agent_cagrisi_anahtari": true, "epistemic_param": true, "yukleyici_tanimli": true}` | GEÇTİ |
| K2 YÜKLEME-BİREBİR (state_dict sha bit-özdeş; 9-anahtar) | `{"state_dict_sha256": "9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500", "anahtar_sayi": 9}` | GEÇTİ |
| K4 FAİL-CLOSED MUTASYON (olmayan-yol RuntimeError; fresh-init != eğitilmiş) | `{"olmayan_yol_istisna": "RuntimeError: DURDURULDU: router_state_path verildi ama dosya yok (data/olmayan_router_t0155.pt); fresh-init'e sessiz dusme YOK (T-015", "fresh_init_sha256": "dbf1ceb85a967b8662b4bc05debda12020d6da731709c19d5fb00e24c80fe73b", "fresh_init_farkli": true}` | GEÇTİ |
| K3 DAVRANIŞ-ÜRETİMİ (300-val top-1 == 0,93 birebir; CPU-kopya) | `{"val_n": 300, "dogru": 279, "top1": 0.93, "beklenen": 0.93, "kirpilan_4096_ustu": 0, "kopya_state_dict_sha256": "9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500"}` | GEÇTİ |
| K6 CANLI-DOKUNULMAZLIK (ağ istemcisi kurulmadı) | `{"ag_istemcisi": "YOK", "vector_memory": ":memory: (storage_path=None)", "canli_192_168_1_9_istek": 0, "localhost_istek": 0}` | GEÇTİ |

## Onarım yüzeyi (kod-değişikliği)

- `src/rag/epistemic_agent.py`: `router_state_path` param + `load_trained_router_state` (fail-closed; strict=True)
- `src/gateway/agent_gateway.py`: `__init__` + `create_default` geçişi; dosya-yok erken-kapısı

