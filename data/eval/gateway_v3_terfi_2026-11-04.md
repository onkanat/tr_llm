# Canlı Gateway Terfisi — router v3 + legal uzmanı (2026-11-04)

**Görev:** "canlı gateway uygulansın yenisi ile" (kullanıcı talimatı).
**Sert kurallar:** git commit YOK (yalnız çalışma ağacı) · Qdrant veri işlemi YOK (probeler
DIRECT modunda — `retrieval_triggered=false` tümünde) · canlı gateway KISA süre ayağa
kaldırılıp ölçüldü ve DURDURULDU.

**Hedef yığın (ölçüldü):** taban=`anka_taban_pure.pt` (değişmez) · uzmanlar=`anka_marangoz.pt`
(carpenter) + `anka_bahcivan.pt` (gardener) (değişmez) + **YENİ `anka_legal_replay.pt` (legal)**
· router=`anka_router_v3.pt` (v2'den terfi; v2 diskte DEĞİŞMEDİ).

---

## 1. Değişen satırlar (dosya:satır)

| Dosya | Satır | Değişiklik |
|---|---|---|
| `src/gateway/agent_gateway.py` | 109–116 (yorum), **117** | `create_default` `router_state_path` varsayılanı `data/anka_router_v2.pt` → **`data/anka_router_v3.pt`**; gerekçe + AÇIK UYARI yorumu |
| `scripts/baslat_canli_gateway.py` | **31** | `DEFAULT_ROUTER_PATH = "data/anka_router_v3.pt"` (yeni sabit) |
| `scripts/baslat_canli_gateway.py` | **34** | `BEKLENEN_ROUTER_SD_SHA` çıpası v2 `a2a95957…` → v3 **`4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5`** (P25 hüküm K4 determinizm değeriyle birebir; çıpa gerçek dosyayla ölçülerek doğrulandı) |
| `scripts/baslat_canli_gateway.py` | **54–81** | `build_expert_model_paths()` (yeni fonksiyon): gardener+carpenter aynen + **`legal → data/anka_legal_replay.pt` EKLENDİ**; docstring'de KALDI 0/3 uyarısı AÇIK |
| `scripts/baslat_canli_gateway.py` | **91–92** | `--router` argparse varsayılanı → `DEFAULT_ROUTER_PATH` (v3) |
| `scripts/baslat_canli_gateway.py` | **104–107** | `main()` uzman çözümü artık `build_expert_model_paths()` çağırır |
| `tests/test_gateway_v3_terfi.py` | yeni (yaklaşık 210 satır) | 6 test (bkz. §2) |

Değişmeyenler: `agent_gateway.py:100/143` roots.tsv/lexicon yüklemesi aynen (lexicon değişikliği
yok); `anka_router_v2.pt` diskte aynen (sha `74b9c808…` — hüküm K6 çıpası); taban ve 2 uzman
ceketi aynen.

## 2. TDD kanıtı (tests/test_gateway_v3_terfi.py)

**RED (önce test, izlenen başarısızlık):** 3 test tam beklenen nedenden düştü —
- `test_create_default_router_varsayilani_v3`: varsayılan `v2` idi (`'data/anka_router_v2.pt' != 'data/anka_router_v3.pt'`)
- `test_baslat_router_varsayilani_v3_ve_sd_sha_cipasi`: `DEFAULT_ROUTER_PATH`/`build_expert_model_paths` yok + çıpa v2 sha'ydı
- `test_baslat_cozumu_legal_uzmani_icerir`: çözümde `legal` anahtarı yoktu

(Karakterizasyon/artifact-guard 3 testi RED'de de geçti — mevcut davranış/paket ölçümü: router
v3 5×256 şema; legal ceketi yüklenebilir + embedding 33114 == sözlük 33114 (T-0087 kapısının
uzman tarafı); `legal` birincilken uzmanı YOKSA üretim TABANA düşer / VARSA legal ceketinden.

**GREEN:** 6/6 geçti (4,43 s).

**Regresyon (mevcut gateway testleri, değişiklik sonrası):** `test_agent_gateway.py`,
`test_moe_gateway_routing.py`, `test_moe_adversarial_stress.py`, `test_qdrant_url_param.py`,
`test_router_and_merak.py`, `test_epistemic_agent_fallback.py` → **53 passed + 23 subtests
(72,78 s)**. Mevcut testler router'ı açık argümanla v2'ye sabitliyor — v2 diskte durduğundan
hepsi aynen geçiyor; final tekrar `test_gateway_v3_terfi.py + test_agent_gateway.py` → 17 passed.

## 3. "Bugün" ölçümü — 'legal' sınıfı uzmanı olmayan sorguda ne oluyor?

Mekanizma: router 5-sınıf `[0: grammar_core, 1: pedagogy, 2: carpenter, 3: gardener, 4: legal]`
(`src/rag/epistemic_agent.py:142`); MoE dağıtımı `src/rag/epistemic_agent.py:459`:
`active_model = self.expert_models.get(primary_expert, self.model)` — **uzmanı olmayan sınıf
sessizce TABANA düşer.**

**Gerçek-ağırlık ölçümü (terfi öncesi, v3 router + uzmanlar yalnız {carpenter, gardener},
device=mps, generate_tokens ajanlı):**

```json
{"sorgu": "İş sözleşmesi feshinde ihbar süresi nedir?", "mode": "DIRECT",
 "router_experts": ["legal", "pedagogy"], "active_expert": "legal",
 "uretim_modeli": "taban(anka_taban_pure)",
 "expert_models_anahtarlari": ["carpenter", "gardener", "grammar_core", "pedagogy"]}
```

→ Router 'legal'i doğru seçiyor AMA üretim **taban modelde** çalışıyor (telemetri
`active_expert="legal"` derken ceket yok). Bu ölçülen sessiz-düşüş tuzanı terfiyle kapanıyor:
`build_expert_model_paths()` artık `legal` anahtarını veriyor → `.get()` legal ceketini buluyor.

## 4. Canlı ölçüm (3 sorgu) — router karar tablosu

Başlatma (kısa): `venv/bin/python scripts/baslat_canli_gateway.py --host 127.0.0.1 --port 8093`
(~12 s'de hazır). Konsol kanıtları: `[UZMAN_YUKLENDI] legal <- data/anka_legal_replay.pt`;
`router_sd_sha256 = 4016b210… → EŞLEŞME: True`; `anka_bellek storage: remote
(delinux01.local:6333)`; taban strict=False **tam eşleşme (0 eksik, 0 fazla)**.
Yanlış-bayrak denemesi: `--max` gibi geçersiz argüman argparse'da exit-2 ile reddedildi
(sunucu hiç açılmadı).

| # | Sorgu | Mode | router_experts (top-2) | active_expert | retrieval | Yanıt (ilk 120 krkt) | süre |
|---|---|---|---|---|---|---|---|
| 1 | İş sözleşmesi feshinde ihbar süresi nedir? | DIRECT | **legal**, pedagogy | **legal** | false | "Kanun: borçluların cezasına çarptırılır ve davaya…" | 11,2 s |
| 2 | Kırlangıç kuyruğu masif ahşapta nasıl açılır? | DIRECT | **carpenter**, gardener | **carpenter** | false | "Teknik çözüm: şerit testere kılavuz rulmanları…" | 4,0 s |
| 3 | Suyun kaynama sıcaklığı kaç derecedir? | DIRECT | **gardener**, carpenter | gardener | false | "Eğer kılcal nem testi dikkate alınmalıdır." | 0,6 s |

**Yönlendirme hükümleri:** hukuk sorgusu `legal` uzmanına gidiyor (terfinin amacı ölçüldü,
başarılı); marangozluk `carpenter`'a parite; nötr sorgu v3'nin kendi ağırlıklarıyla
`gardener`'a düştü (domain-prior terim listesi tetiklenmedi — "su"/"kaynama" listede yok;
nötr sınıf zaten pedagogy/grammar_core'e ait değil). DIRECT modunda tüm probelerde
`retrieval_triggered=false` → Qdrant okuma/yazma YAŞMADI; `/api/inject` hiç kullanılmadı.

**Temiz kapanış (PID doğrulamalı):** süreç sinyalle durduruldu; ardından ölçüldü:
PID 37281 yok · port 8093 LISTEN değil (lsof boş) · `pgrep baslat_canli_gateway` boş ·
`/api/status` cevap vermiyor. Artakalan süreç yok.

## 5. UYARILAR (açık kalır)

1. **Legal ceketi (replay) kapılardan GEÇMEDİ — KALDI 0/3** (ΔCE −0,306/−0,50; kanonik-20
   0/20; OOD 0,365/0,40 — `data/eval/p25_router_v3_hukum_2026-11-04.json`, CHANGELOG 1.17.0).
   Bu terfi **kullanıcı talimatıdır**; kod yorumunda (`baslat_canli_gateway.py:60-64`) ve bu
   raporda uyarı AÇIK kalır. Router v3'ün kendisi GEÇTİ (8/10 legal; diğer 4 sınıf v2-paritesi
   1,0; hüküm `P25_ROUTER_V3_LEGAL_GECTI`).
2. **Yanıt niteliği:** ceketin kart edinimi tamamlanmadığından (uyarı-1'in kökü) hukuk yanıtı
   hukuki bağlamdan yoksun ("Kanun: borçluların cezasına…"); nötr yanıt tutarsız. Routing
   ölçümü doğru; üretim kalitesi bilinçli olarak kapsam dışı.
3. **Çalışma ağacında yabancı değişiklik:** `src/rag/merak.py` (P2.6 `trigger_signal`
   parametresi, +43 satır) bu göreve ait DEĞİLDİR — paralel bir işin devam eden değişikliği;
   dokunulmadı, commit de yapılmadı.
4. v2 → v3 varsayılan değişimi yalnız `create_default`/`baslat` çözümünde; v2 açıkça
   `router_state_path` verilerek eskisi gibi kullanılabilir (regresyon testleri bunu sabitliyor).

## 6. Kanıt dosyaları

- Testler: `tests/test_gateway_v3_terfi.py` (RED→GREEN, 6 test)
- Probe kaydı: `~/.hermes/cache/scratch/canli_probe_3sorgu.json`, `bugun_legal_uzmansiz_probe.py`
- Hüküm kaynakları: `data/eval/p25_router_v3_hukum_2026-11-04.json` (router v3),
  `data/eval/p25_replay_legal_olcum_rig_2026-11-04.json` (ceket replay, KALDI)
- Git: değişiklikler çalışma ağacında; **commit YAPILMADI**.