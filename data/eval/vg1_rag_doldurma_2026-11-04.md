# VG-1 — RAG'ı Anlamla Doldurma: GTS 128K + 17 Kanun Maddesi → Qdrant — 2026-11-04

> **Onay:** VG-1 (kullanıcı talimatı — "RAG'ı anlamla doldur").
> **Sert kurallar:** git commit YOK · Qdrant yazma yetkisi YALNIZ yeni koleksiyonlar
> (`gts_bellek`, `kanun_madde`) · `anka_bellek`/`muhakeme_bellek`/`simulasyon_bellek` DOKUNULMAZ.
> **Sunucu:** Qdrant 1.19.1 `http://192.168.1.9:6333` (delinux01 — canlı gateway'in bellek sunucusu).

## İLAN

93M model veri açlığından semantik üretemez (ölçülen belirti: hukuk sorgusunda kanun
numarası halüsinasyonu). VG-1, bilginin GERİ-GETİRİMDEN gelmesini sağlar:

1. **Retrieval mimarisinin çıkarımı** — canlı sorgu hangi koleksiyonlara bakıyor? Çoklu destek var mı?
2. **VG-1.1** GTS 128.327 satır → `gts_bellek` (pilot öncesi ETA/RSS ölçümü, idempotent, read-back).
3. **VG-1.3** 17 kanun maddesi → `kanun_madde` (aynı kalıp).
4. **Hat:** retrieval zincirinin yeni koleksiyonları görmesi (minimal, TDD'li).
5. **VG-1.2** donmuş sorgularla grounding öncesi/sonrası fark ölçümü:
   - S1: `İş sözleşmesi feshinde ihbar süresi nedir?`
   - S2: `4857'de haftalık normal çalışma süresi hangi maddede düzenlenmiştir?`
6. Bu rapor.

## HÜKÜM

**H1 — MİMARİ (GÖREV 1):** Bugünkü canlı RAG zinciri **sabit İKİ koleksiyonlu** idi ve
env/param ile genişletilemiyordu: `AgentGateway.create_default` yalnız `anka_bellek`
(birincil, agent_gateway.py:190 → eski:197) ve `simulasyon_bellek` (yedek, :191) kurar;
`EpistemicCuriosityAgent.search_memory` (epistemic_agent.py:231-253) birincili sorgular,
kök-eşleşme yoksa YEDEĞE düşer — başka koleksiyona bakma yolu YOKTU. `RagPipeline`/
`build_from_disk` (rag_pipeline.py:227) tek-koleksiyon parametrelidir (CLI demo yolu).
Eşik kapısı `RAG_MATCH_THRESHOLD = 0.33` (rag_pipeline.py:34; koşullanma epistemic_agent.py:409).
→ **Hüküm:** çoklu koleksiyon desteği YOKTU; VG-1 yamasıyla env/arg üzerinden
`aux` zinciri eklendi (aux boşken davranış birebir — test kapılarıyla kanıtlı).

**H2 — EKİM (VG-1.1 + VG-1.3):** `gts_bellek` **128.327 nokta**; `kanun_madde`
**4.068 madde** (129 mülga madde elendi). Pilot (1.000 satır): 6.5 s, 197 nokta/s,
RSS 403 MB → tam ekim ETA ~11 dk ≪ 60 dk → **pratik; batch planına gerek kalmadan
tamamlandı** (127.327 ek / 344.9 s, tepe RSS 772 MB, 371 nokta/s). kanun_madde: 118.6 s.
**İDEMPOTENT kanıtlı:** her iki betik yeniden koşumda `EKSİK: 0`, **0 yeni yazma**,
nokta sayıları değişmedi (tam-satır/tam-kayıt sha256 nokta kimliği + chunk'lı
mevcut-kimlik sorgusu; kesilen koşum kaldığı yerden devam eder).

**H3 — GROUNDING/RETRIEVAL (VG-1.2a→b):** Donmuş sorgularda zincir ÖNCE
`simulasyon_bellek`'ten hukuk-dışı belge buldu (S1 0.29 "Acentelik sözleşmesi…",
S2 0.08 "Balköpüğü karıncaları…" — kök-eşleşmeli ama 0,33'ün ALTINDA → koşullanma YOK).
SONRA zincir aux'tan **gerçek İş Kanunu maddesini** yükseltti: S1 → `kanun_madde`
**0.70** (4857 **Madde 8**), S2 → `kanun_madde` **0.8333** (4857 **Madde 41**);
`koşullanma=False → True` — gerçek madde metni üretim prompt'una girdi.
Ayrıca: S2'de hedef **Madde 63 top-3'te** (0.5833, "haftada en çok kırkbeş saattir");
S1'de hedef Madde 17 top-3 dışı (m.8/m.15/m.59 dönüyor) — retrieval hassasiyet notu.

**H4 — GROUNDING/ÜRETİM (VG-1.2c): KALDI.** Koşullanma belgesi gerçek madde metnini
taşıyor, ancak 93M model çıktıya **birebir alıntı taşımıyor** (S1: en-uzun-ortak dizi
1 sözcük, 0 paylaşılan 4-gram; S2: 1 sözcük, 0 4-gram; çıktıda 4857/kırkbeş/saat yok,
S2'de uydurma sayılar var). Retrieval katmanı DÜZELDİ; üretim katmanının koşullanılmış
belgeden alıntı yapabilmesi için **RAG-koşullanmalı SFT** gerekiyor (VG-1 sonrası adım).

**H5 — DOKUNULMAZLIK:** `anka_bellek=46`, `muhakeme_bellek=91`, `simulasyon_bellek=4.073`
— her ekim/ölçüm koşumunda ÖNCE=SONRA (log kanıtlı; final sunucu sayımı da aynı).

**H6 — TESTLER:** Yeni `tests/test_vg1_aux_collections.py` (8 test; RED→GREEN).
Tam paket: **583 passed, 1 skipped, 1 xfailed, 0 failed** (VG-2 gözlem kancası
eşzamanlı işi yerli yerindeyken). Not: VG-2'nin kanca kablolaması çalışma ağacına
yetişmeden koşulan turda `test_yetenek_agi` sıra-bağımlı düşüşler görüldü; kanca
yerine oturduktan sonra tüm paket yeşil (düşüşler VG-1 değişikliğinden DEĞİL —
temiz ağaçta da aynısı ölçüldü, tek-başına hepsi geçiyor).

## RAPOR — kanıtlar

### 1. Mimari bulgular (ölçüm öncesi, kod-duruşu)

| Bileşen | Yer | Bulgu |
|---|---|---|
| Eşik | `src/rag/rag_pipeline.py:34` | `RAG_MATCH_THRESHOLD = 0.33` (T-0179 hükmü) |
| Sorgu vektörü | `rag_pipeline.py:37-43` | tokenizer.encode → dense(768)+sparse (deterministik) |
| Bellek arayüzü | `src/rag/vector_memory.py:73-105` | `VectorMemory` = örnek-başına TEK koleksiyon |
| Hibrit arama | `vector_memory.py:368-396` | `hybrid_recall` yalnız `self.collection_name`'i sorgular |
| Canlı zincir | `src/rag/epistemic_agent.py:231-253` (yama öncesi) | birincil(anka) → kök-eşleşme yoksa yedek(simulasyon); env/param yok |
| Gateway kurulumu | `src/gateway/agent_gateway.py:190-191` (yama öncesi) | anka_bellek + simulasyon_bellek SABİT |
| Koşullanma kapısı | `epistemic_agent.py:409` | `conditioned = skor ≥ 0.33` |
| CLI demo yolu | `rag_pipeline.py:227-238` | `build_from_disk(collection_name="simulasyon_bellek")` tek-koleksiyon (değiştirilmedi) |

### 2. Ekim kanıtları

**Pilot (gts_bellek, 1.000 satır)** — `scripts/seed_gts_bellek.py --limit 1000`:
```
[EKIM] ekim-SONRASI nokta sayısı: 1000 (öncesi: 0, yazma: 6.5s)
  [KANIT] id=7662695411205368601 satır=1 birebir=EVET madde='dulavrat otu' …
  [KANIT] id=6713691022742920908 satır=501 birebir=EVET madde='dünür' …
  [KANIT] id=1130566639434557920 satır=1000 birebir=EVET madde='ebru' …
```
ETA ölçümü: 197 nokta/s → 127.327 kalan ≈ 10,8 dk. RSS: 306→403 MB.

**Tam ekim (gts_bellek):**
```
[EKIM] mevcut: 1000, EKSİK: 127327 (kontrol 1.6s) — yalnız eksikler yazılır
  [EKIM] 127327/127327 parça 249/249 (371 nokta/s, ETA 0.0 dk, RSS 772 MB)
[EKIM] ekim-SONRASI nokta sayısı: 128327 (öncesi: 1000, yazma: 344.9s)
  [KANIT] id=… satır=1 birebir=EVET | satır=64164 birebir=EVET | satır=128327 birebir=EVET
[EKIM] dokunulmaz ÖNCE/SONRA: {anka:46, muhakeme:91, simulasyon:4073} → AYNI (DEĞİŞMEDİ)
```

**İdempotens (yeniden koşum):**
```
[EKIM] mevcut: 128327, EKSİK: 0 → TAMAM: 0 yeni nokta upsert (0.3s)
[EKIM] mevcut: 4068,   EKSİK: 0 → TAMAM: 0 yeni madde upsert (0.2s)
```

**kanun_madde (17 kanun):** ayrıştırıcı = `prepare_legal_dataset.py` kalıbı
(`MADDE_HEAD` regex'i birebir). Dağılım: 657:356, 2577:79, 2872:53, 3194:86,
4708:22, 4721:1021, 4857:118, 5237:342, 5271:348, 5378:40, 5510:244, 6098:650,
6100:460, 6331:50, 6446:72, 6502:91, 6698:36 → **4.068 nokta** (mülga elendi: 129;
manifest nominal toplamı 4.197 — fark = mülga/boş-gövde elemeleri).
Payload: `kanun_adi, kanun_no, madde_no, madde_etiketi, gecici, metin(TAM), kaynak(URL),
crystal_tags, token_ids`; text = "«Kanun Adı» (Sayı: N) Madde M - …" (2.000 karakter tavan).

**GTS dedupe notu:** `gts_definitions.txt` çıkarıcıda ZATEN tam-dedupe'lidir —
(madde,anlam) düzeyinde 128.327 benzersiz, 0 yinelenik ölçüldü (T3.3'ün 11.912
yineleniği tarball/ham-JSONL düzeyine aittir). Betikteki dedupe kapısı savunma
amaçlı kaldı ve 0 bulduğunu kanıtladı.

### 3. Grounding fark tablosu (donmuş sorgular)

| Sorgu | Evre | Zincir kaynağı | Skor | Koşullanma | Enjekte belge | Üretilen yanıt (birebir) |
|---|---|---|---|---|---|---|
| S1 İHBAR | ÖNCE | simulasyon_bellek | 0.29 | **Hayır** (<0.33) | — | "Kanun: borçluların cezasına çarptırılır ve davaya başvurmak, mahkeme kararlarına aykırı olanların cezasının uygulanmasına ilişkin hükümlere aykırı olmaktan suçlu veya ceza mahkemelerindeki davalara aykırı…" |
| S1 İHBAR | SONRA | **kanun_madde** | **0.70** | **Evet** | "İş Kanunu (Sayı: 4857) Madde 8 - İş sözleşmesi, bir tarafın (işçi) bağımlı olarak iş görmeyi, diğer tarafın (işveren) da ücret ödemeyi üstlenmesinden oluşan sözleşmedir…" | "Kanunların uygulanmasındaki kararlarının belirlendiğini ve bunun yerine, kanunlara aykırı olduğu kararını vermektedir." |
| S2 4857 | ÖNCE | simulasyon_bellek | 0.08 | **Hayır** | — | "Bu belge, [?]' ( Serin), [?]' ( Serin), [?]' ( Serin)…" (dejenere) |
| S2 4857 | SONRA | **kanun_madde** | **0.8333** | **Evet** | "İş Kanunu (Sayı: 4857) Madde 41 - Ülkenin genel yararları… haftalık kırkbeş saatten fazlası fazla çalışma sayılır…" | "Bunun yerine, bir bankasının ( örneğin) ve diğerleri ise, [?] birliklerin… 12.085 ve 397,12.08. maddesine göre, 1250 ve 34. maddelerin…" |

**Birebir alıntı taraması** (`data/eval/vg1_fark_tablosu.json`, `scripts/vg1_alinti_taramasi.py`):
yanıt ↔ enjekte belge: S1 LCS=1 sözcük ('ve'), 4-gram=0; S2 LCS=1 ('yerine'), 4-gram=0;
kavram geçişi: 'madde' yalnız S2'de (yanlış numaralarla); '4857'/'kırkbeş'/'saat' hiçbirinde yok.
→ **H4 KALDI hükmünün kanıtı.** Retriever artık doğru belgeyi prompt'a sokuyor;
model onu kullanamıyor (RAG-koşullanmalı SFT gereksinimi).

Ölçüm koşumu (öncesi=sonrası birebir aynı): `force_rag=True, mode=RAG,
temperature=0.0, max_new_tokens=100, model=anka_taban_pure.pt, router=anka_router_v3.pt
(sd-sha 4016b210… eşleşme=True), uzmanlar gardener/carpenter/legal, device=mps` —
`scripts/vg1_generation_testi.py` (baslat_canli_gateway.py yığınıyla aynı).

### 4. Kod değişiklikleri (VG-1)

| Dosya | Yer | Değişiklik |
|---|---|---|
| `src/rag/epistemic_agent.py` | :94-104 | `auxiliary_memories` param + alan (boş → davranış birebir) |
| `src/rag/epistemic_agent.py` | :254-287 | `search_memory` AUX yükseltme bloğu: zincir belgesi 0,33 altındaysa aux'lar sorgulanır; en iyi KÖK-EŞLEŞMELİ ve 0,33+ skorlu aux adayı yükseltilir; zincir belgesi kullanılabilirse aux HİÇ sorgulanmaz |
| `src/gateway/agent_gateway.py` | :40-49 | `parse_aux_collection_names` (saf fonksiyon) |
| `src/gateway/agent_gateway.py` | :67-68, :84-87 | `__init__(auxiliary_memories=…)` → epistemic_agent'a taşınır |
| `src/gateway/agent_gateway.py` | :147, :218-231, :309-310 | `create_default(aux_collections=…)`; çözüm: açık arg > `QDRANT_EXTRA_COLLECTIONS` env; aux VectorMemory'ler aynı host/port (fail-closed) |
| `tests/test_vg1_aux_collections.py` | yeni, 8 test | davranış-koruma + yükseltme + ayrıştırıcı sözleşmesi kapıları |

Canlı zincirin yeni koleksiyonları görmesi için gateway başlatıcısında
`QDRANT_EXTRA_COLLECTIONS=gts_bellek,kanun_madde` (veya `create_default(aux_collections=[...])`)
verilmelidir; **canlı gateway süreci bu env ile yeniden başlatılmalıdır** (kod değişikliği
koşan sürece kendiliğinden işlemez — operasyon notu).

### 5. Kanıt dosyaları

- `data/eval/vg1_retrieval_oncesi.json` / `vg1_retrieval_sonrasi.json` — per-koleksiyon top-3 + zincir
- `data/eval/vg1_uretim_oncesi.json` / `vg1_uretim_sonrasi.json` — tam telemetri (entropy, router, belge)
- `data/eval/vg1_fark_tablosu.json` — birebir alıntı taraması
- Yeni betikler: `scripts/seed_gts_bellek.py`, `scripts/seed_kanun_madde.py`,
  `scripts/vg1_retrieval_olcum.py`, `scripts/vg1_generation_testi.py`,
  `scripts/vg1_alinti_taramasi.py`

### 6. Son sunucu durumu (2026-11-04, UTC~21:00)

```
gts_bellek: 128327 | kanun_madde: 4068
anka_bellek: 46 | muhakeme_bellek: 91 | simulasyon_bellek: 4073   (değişmedi)
```

### 7. Notlar / tuzaklar

- **Eşzamanlı ajan (VG-2):** VG-2 bu depoda aynı gece `yetenek_agi` gözlem kancasını
  çalışma ağacına yazdı. `git checkout --` iki dosyaya koşturulduğunda kanca geçici
  olarak kayboldu (yamada kayıtlıydı; VG-2 yeniden uyguladı, VG-1 yamaları kanca'lı
  sürümün üstüne temiz yeniden uygulandı; tüm paket 583/583 yeşil). Ders: paylaşılan
  depoda `checkout --`'ten önce `git status`'un `M` satırları kontrol edilmelidir.
- `test_yetenek_agi` tam-paket koşumlarında VG-2 kanca kablolaması eksikken
  sıra-bağımlı düşüyordu (VG-1'den bağımsız — temiz ağaçta da ölçüldü); kanca
  yerleşince düşüş kalmadı.
- S1'de ideali (Madde 17 bildirim süreleri) top-3 dışı: kristal-gömme (deterministik
  hash) morfolojik kök örtüşmesine yaslanıyor; madde-hassasiyeti ileride ayrı
  değerlendirme gerektirebilir (S2'de hedef Madde 63 top-3'te, 0.5833).
- kanun metinlerinde satır-içi "(Değişik: …)" dipnotları HAM olarak korunur (FAZ-1 V2
  gereği); boş gövdeli başlıklar elenir.
