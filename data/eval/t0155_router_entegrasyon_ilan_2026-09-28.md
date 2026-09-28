# İLAN — T-0155: gateway TriModalRouter eğitilmiş-ağırlık entegrasyonu

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-28T07:12:22Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Hüküm BETİKTEN
(`scripts/dogrulama_t0155_router_entegrasyon.py`); elle sayı/hüküm YOK.
rc ∈ {0, 2}.

## Meşruiyet

T-0154 kapanış-notunun beyan ettiği AYRI onarım maddesi: gateway
`create_default` router'ı fresh-init kurar; `data/anka_router.pt`
state_dict'inin tüketim-yoluna yüklenmesi bu turdur. Operatör talebi
("Gateway entegrasyon turunu aç", 28 Eyl 2026). Bus T-0155; kiralamalar
claim-ile-birlikte (T-0153 ihlali dersi): `src/`, `scripts/`, `data/eval/`,
`.agent-bus/notes/`. Canlı sunucu 192.168.1.9:6333'a HİÇ erişim yok
(anka_bellek DOKUNULMAZ); localhost'a da 0 istek (betik VectorMemory
`:memory:` kullanır; `create_default` çağrılmaz).

## Onarım-yüzeyi (kod-değişikliği VAR — T-0154'ten farkı bu)

1. **`src/rag/epistemic_agent.py`** (DONMUŞ DEĞİL):
   `EpistemicCuriosityAgent.__init__` yeni opsiyonel param
   `router_state_path: Optional[str] = None`:
   - `router` VERİLMİŞ + `router_state_path` VERİLMİŞ → `ValueError`
     (ikili-belirsizlik fail-closed).
   - `router` YOK + `router_state_path` VAR → fail-closed yükleme:
     dosya-yok → `RuntimeError` (fresh-init'e sessiz düşme YOK);
     anahtar-küme uyuşmazlığı → `RuntimeError` (şema-koruma);
     `load_state_dict(strict=True)`; `[ROUTER_SOYAGACI]` görünür-satır.
   - ikisi de YOK → fresh-init (MEVCUT davranış KORUNUR — test yüzeyi
     `test_router_and_merak.py` / `test_epistemic_agent_fallback.py`
     kendi router instance'larıyla kurar; kırılmaz).
2. **`src/gateway/agent_gateway.py`** (DONMUŞ DEĞİL):
   - `AgentGateway.__init__` yeni param `router_state_path:
     Optional[str] = None` → :65 `EpistemicCuriosityAgent(...)` çağrısına
     geçirilir.
   - `create_default` yeni param `router_state_path: str =
     "data/anka_router.pt"` (AÇIK-BEYAN default — bilinçli-sabit; T-0087
     bayat-varsayılan kaldırmasından FARKLI: bu değer T-0154'te üretilen
     kanonik artefaktır, ölçülü-damgalı) + **dosya-yok erken-kapısı**
     (model_path :130 kalıbı): dosya yoksa `RuntimeError` —
     sessiz-fresh-düşme YOK. cls(...) çağrısına açık-geçirilir.
   - `create_default` çağıran-yüzeyi (run_goal_pipeline ×4,
     run_agent_arena, prepare_turk_tarihi_pipeline,
     train_literature_specialization) param geçmez → default devralır —
     çağıran-kırılması YOK.

## Sabitler (koşum-öncesi; değişmez)

- **Kaynak-çıpası (T-0154 hüküm-değerleri):**
  `data/anka_router.pt` =
  `64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720`;
  state_dict (determinizm) =
  `9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500`;
  `data/anka_base_v2.pt` =
  `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`;
  `data/rebuild/vocab_anka_r1_33114.json` =
  `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`;
  `data/lexicon/roots.tsv` =
  `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`.
- **Davranış-çıpası:** T-0154 K3 val top-1 = **0,93** (300-val;
  SEED-42; T-0154 koşum-konfigürasyonu birebir: prompt_vec mean-pool +
  q_merak fresh-init seed-42 CuriosityEngine + rag_vec=None; eğitim
  cihazı ayrımı yok — özellik MPS'te çıkarılır, MPS yoksa DUR).
- **Şema-çıpası:** 9-anahtar kanonik TriModalRouter.

## Kapılar (koşum öncesi sabit)

- **K1 KAYNAK-ÇIPASI:** yukarıdaki 5 digest koşumda yeniden ölçülür ve
  birebir olmalı.
- **K2 YÜKLEME-BİREBİR:** `EpistemicCuriosityAgent(router_state_path=
  "data/anka_router.pt")` kurulumu sonrası `agent.router.state_dict`
  özeti == `9b85f951…` (bit-özdeş; anahtar-sayısı 9).
- **K3 DAVRANIŞ-ÜRETİMİ:** aynı 300-val örnekleminde (T-0154 veri-seçimi
  SEED-42 birebir) top-1 doğruluk == **0,93** (300-örnek birebir-üretim;
  farklı-değer = entegrasyon-kırığı).
- **K4 FAİL-CLOSED MUTASYON:** (a) olmayan yol (`data/olmayan_router.pt`)
  → `RuntimeError` yükselir; (b) `router_state_path=None` → fresh-init
  state_dict özeti != `9b85f951…` (mevcut davranış KORUNUR — mutasyon-
  kanıtı iki-yönlü).
- **K5 GATEWAY AST ZİNCİR-KAPISI** (T-0143 kapı-anahtar-hizası kalıbı):
  statik-ön koşumundan AYRI; betik AST okur — `create_default` →
  `cls(...)` → `AgentGateway.__init__` → `EpistemicCuriosityAgent(...)`
  zincirinde `router_state_path` açık-beyan geçer.
- **K6 CANLI-DOKUNULMAZLIK:** betik `VectorMemory(:memory:)` dışında
  hiçbir ağ-istemcisi kurmaz; 192.168.1.9'a ve localhost'a 0 istek
  (kod-yolu beyanı + betikte ölçüm: socket bağlantısı YOK).
- **6/6 → T0155_ENTEGRASYON_GECTI rc=0**; aksi her dal → **DUR rc=2**.

## Statik-ön (koşumdan AYRI; T-0154 kalıbı)

py_compile + iki-geçişli AST tanımsız-ad + bağımsız pyflakes TEMİZ.

## pytest (ayrı koşum — hüküm-dışı; kod DEĞİŞTİ)

307 passed beklenir (testler kendi router instance'ları — yeni opsiyonel
param default None onları KIRMAZ; yeni düşen olur İLAN'a beyan edilir ve
onarılır — test-silme YOK).

## DOKUNULMAZLAR

`src/compiler/**`, `src/llm/tokenizer.py` (DONMUŞ — dokunulmaz);
`data/anka_router.pt` YALNIZ OKUNUR (donmuş `data/*.pt` deseni; kiralama
üst-dizin `data/` ALINMADI çünkü bu tur data/ köküne YAZMA yok —
yalnız `data/eval/` kiralandı); canlı sunucu; P2-P5 çıpa-betikleri.

## Beyan: damga-yöntemi

Damga betik koşulmadan `date -u` işlendi; hüküm-JSON: ilan_sha + damga +
kapılar. Hüküm BETİKTEN; elle sayı YOK.