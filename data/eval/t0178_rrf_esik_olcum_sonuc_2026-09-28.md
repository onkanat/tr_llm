# T-0178 ölçüm-sonucu + eşik-kalibrasyon önerisi (operatör-onayına sunulur)

- **Damga:** koşumlar BETİKTEN damgalı (hüküm-1 20:31Z bandı, hüküm-2 20:36Z
  bandı; kesin değerler hüküm-JSON'larda) · **Yürütücü:** claude
- **Kod beyanı:** `RAG_MATCH_THRESHOLD` ve tüm `src/rag/**` DOKUNULMAZ —
  hiçbir kapı değeri değişmedi; öneri OPERATÖR-KARARIDIR.

## 1. Ölçüm (kanonik :memory: kabı; arz 36/36; 20 pozitif sorgu çıpa-hükümden birebir)

| Eşik | Pozitif-yakalama | Yüzde |
|---|---|---|
| 0,30 | 20/20 | %100 |
| 0,33 | 20/20 | %100 |
| 0,35 | 20/20 | %100 |
| 0,37 | 19/20 | %95 |
| **0,40 (mevcut)** | **10/20** | **%50** |

- Skor dağılımı: min **0,3556** / medyan **0,4444** / maks **1,3333**.
- OOV (negatif-kontrol, A-3 fail-closed): skor **0,0556**, `has_root_match=False`
  (P2 onarım b3 beyanı 0,0467 ile uyumlu bant).
- **Ayrım-aralığı:** güvenli eşik penceresi **[0,075 ; 0,3556)** — pozitif-min
  0,3556 üstünde, OOV-maks 0,075 altında.

## 2. Ölçülmüş kök-bulgu (T-0142 açık yarımını teyit eder)

`RAG_MATCH_THRESHOLD=0.40` RRF-ölçek medyanının (0,4444) ALTINDAYDI AMA
pozitif-min'in (0,3556) ÜSTÜNDE — kendi-belgelerin **%50'si eşik-altı** (P2
onarım-hüküm 10/20 ile BİREBİR teyit). Ayrıca ölçek 0,75'e normalize; 0,40
o bantta "orta" görünür ama karar-bağlamında medyan-altıdır.

## 3. Kabı-doğrulama durumu (dürüst-beyan; İLAN yumuşatma YOK)

- **koşum-1 DUR rc=2:** çıpa İLAN-1'de P2 İLK hükümden geldi — post-[2A]
  normalizasyon ÖNCESİ ölçek (bayat-çıpa; İLAN-formülasyon-kusuru sınıfı).
- **koşum-2 DUR rc=2 (onarım-hüküm çıpası, İLAN-2):** 15/20 sapma **0,0000**;
  karar-sınıflandırma (0,40 geçme/kalma) **20/20 BİREBİR**; bant-dışı 3/20 —
  tamamı yüksek-tail, `:memory:` ↔ uzak sparse-rank atlaması (`Σ 1/(k+rank)`
  kuantumu ~0,111–0,533 sapma; mutlak-bant emilemez).
- Dev-sunucu 192.168.1.5:6333 koşum anında ERİŞİLEMEZ (BETİKTEN ölçüldü) —
  uzak-birebir teyit sonraki koşuma kalır. Koşum-1/2 hükümleri DUR olarak
  kanıt-zincirinde kalır.

## 4. ÖNERİ (operatör-onayı bekler; kod bu görevde değişmedi)

**`RAG_MATCH_THRESHOLD: 0.40 → 0.33`** (rag_pipeline.py:30).

- Gerekçe: 0,33 pencere-ortası [0,075 ; 0,3556) — pozitif-yakalama 20/20,
  OOV'den ~4,4× mesafe (0,33 / 0,075), pozitif-min'e 0,0256 marj.
- Alternatif: **0,35** (marj 0,0056 — tek-sorgu kırpma riski taşır).
- 0,37 önerilmez (19/20 — mevcut kaybın yarısını bile kapatmaz).
- Değişikliğin etkisi: `is_context_usable` ve `retrieve_context`
  sessiz-None yolu — "belge yok" sahte-negatiflerin yarısı kapanır.
- Ayrı gerekçe-beyanı: 0,33–0,35 bandında pozitif/OOV ayrımı ölçülmüş;
  eşik-değişimi SONRASI aynı 20 sorguyla yeniden-ölçüm (aynı betik) ayrı
  operatör-onaylı koşumda kanıtlanır.