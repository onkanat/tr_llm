# Konsol denetimi — `tr_llm` Web Chat Gateway (192.168.1.10:8080)

**Damga:** 2026-10-11 · **Denetleyen:** aikonsol (Hermes ajanı, delinux01) · **Tür:** dış denetim raporu
(dışarıdan ölçüm; projenin kendi görev akışının parçası değildir).
**Hedef:** `http://192.168.1.10:8080/` — "Kristal-Vektörel LLM | Web Chat" (MacAir'de koşan sürüm).
**Yöntem:** dışarıdan salt-okunur HTTP + gerçek tarayıcıda uçtan uca + depo kaynağının okunması.
Hedefte kabuk/log erişimi yok (MacAir'de SSH kapalı); **deponun çalışma ağacı değiştirilmedi**.

**Ölçülen sürüm:** `origin/master@aa00fd8` — çapalama iki bağımsız kanıtla yapıldı:
`src/gateway/static/index.html` sha256 `ade12afd82e2…` canlı sayfanınkiyle birebir aynı **ve** canlı
`POST /api/query` yanıtı tam olarak `ask()`'in ürettiği **19 anahtarı** taşıyor.

---

## 1. Geçenler (çalıştığı ölçülenler)

- `GET /` 200 (12.840 B); form → `POST /api/query` → sonuç kartı tarayıcıda **1,19 sn**, konsol hatası 0.
- **XSS kuralı gerçekten uygulanmış:** sorguya `<b>…</b><script>window.__xss=1</script>` gömüldü →
  kutunun `innerHTML`'i kaçışlı, payload çalışmadı (`textContent` kuralı, `index.html:371`).
- Bozuk JSON gövdesi → temiz `400 {"error": "Invalid JSON: …"}` (`agent_gateway.py:527`).
- Belge getirme çalışıyor: "Meşe ahşabı nerelerde kullanılır?" → 507 krk meşe belgesi gelip kullanıldı.
- `GET /api/status` tutarlı; `anka_bellek` canlı sayısı ile ilan edilen sayı örtüşüyor (bkz. §3).

## 2. Bulgular — güvenlik/operasyon (önem sırasına)

### B1 — YÜKSEK · `POST /api/inject` kimlik doğrulamasız **YAZMA** ucu
`VectorMemory.add_documents_batch` çağrılıyor (`agent_gateway.py:549-554`). `0.0.0.0:8080`'da olduğu
için ağdaki herkes `anka_bellek`/`simulasyon_bellek`'e belge ekleyebilir → **RAG belleği zehirlemesi**.
README bu ucu listeliyor; konsolun ilan sayfası listelemiyor.
*Yazmadan kanıt:* bilinmeyen koleksiyon adıyla POST → `_resolve_target_memory` `ValueError` →
yanıtsız kapanma; Qdrant'ta belge sayısı ve koleksiyon sayısı **değişmedi**.

### B2 — YÜKSEK · `GET /api/backlog` kimlik doğrulamasız **OKUMA** ucu
`get_epistemic_backlog(limit=50)` kayıtları ham döndürüyor (`agent_gateway.py:481-482`): `input`
(kullanıcı sorgusu + belge), `model_failed_output`, `compiled_output`, benzerlik skoru, metadata.
Ayrıca `log_message` gövdesi `pass` (`:564-565`) → **erişim günlüğü hiç tutulmuyor**.

### B3 — YÜKSEK · Tek iş parçacıklı sunucu (`HTTPServer`) → bir ağır istek **herkesi** kilitliyor
`HTTPServer` (threaded değil, `:567`). Ölçümler (11.10):

| senaryo | uçarken atılan normal sorgunun beklemesi |
|---|---|
| 1,2 MB gövdeli sorgu | **53,6 sn** (öğlen) → **22,2 sn** (akşam) |
| `max_new_tokens=99999` (**gövde 89 bayt**) | **54,5 sn**; isteğin kendisi 47,8-121,7 sn |

Kararlı hâl 0,49-0,86 sn. Yani **DoS için büyük gövde gerekmiyor: 89 baytlık bir istek yeter**.

### B4 — YÜKSEK · Gövde ve üretim sınırı yok
`self.rfile.read(content_len)` sınırsız (512 KB kabul edildi); `max_new_tokens` doğrulanmıyor
(99999 kabul) ve `temperature=9` de kabul ediliyor (`:522-536`). **Akşam ölçümünde kötüleşti:**
`max_new_tokens=99999` → **2.737-2.969 krk / 47,8-121,7 sn** (öğlen 1.509 krk / 12,4 sn).

### B5 — YÜKSEK (veri) · Eğitim dosyasına yazma yolu
`epistemic_agent.py:461-496`: `match_score ≥ 0,85` **VE** (`entropy_post > 2,5` **VEYA** ≥2 `<UNK>`)
→ `data/future_train_vector.jsonl`'e append; kaydın **`input` alanı kullanıcının sorgusunun kendisi**.
→ Kimliği doğrulanmamış istemci, koşulu tutturan bir sorguyla dosyaya istediği metni yazdırabiliyor.
**Tetik boyuta bağlı değil** (16 KB artık tetiklemiyor; küçük bir sorgu tetikledi). Yazma koşulu
kararsız: aynı girdi 3 koşuda `False/True/True` (üretim `temperature=0`'da deterministik değil).

### B6 — ORTA · Tip doğrulaması yok; istisna yanıtsız kapanmaya dönüşüyor
`query=null` / `12345` / `["a"]`, `mode=null`, `temperature="sıcak"`, `instruction=42`, gövde `[…]`/`null`
→ **yakalanmayan istisna** → istemci `RemoteDisconnected` alır (HTTP kodu yok; sunucu ayakta kalır).
`max_new_tokens=0` / `-5` → **200 + boş yanıt**. (`ask()`'in `mode` karşılaştırması `:250`.)

### B7 — ORTA · `rag_score` ölçeği yanlış; `is_high_similarity` buna bağlı
`vector_memory.py:331` `score / RRF_SCORE_MAX` (=0,75 elle konmuş normalleştirici) + `*= 0.05` cezaları
+ `*= (0.3 + 0.7·match_ratio)`. Skor **Qdrant RRF füzyon skoru**, kosinüs benzerliği değil: **1,0'ı
aşabiliyor** (ölçüldü: 1,1111 · 1,3333 · 0,4889). Kod bunu `similarity_threshold=0,85` ile ("cosine"
yorumu altında) karşılaştırıyor. Aynı ölçek sorunu `RAG_MATCH_THRESHOLD` (0,33) kapısını da etkiliyor.

### B8 — ORTA · `needs_retrieval` yanıltıcı; gerçek tetik alanı yanıtta yok
Gerçek alan `retrieval_triggered = needs_retrieval or force_rag`, **HTTP yanıtına konmuyor**. İki yön
ölçüldü: `needs=true` + `rag_document=""` (8 vaka) ve `needs=false` + belge kullanıldı (mode=RAG).

### B9 — ORTA · `mode` etiketi fazla vaat ediyor
`use_rag = (mode.upper()=="RAG") or force_rag` (`:250`). `needs_retrieval=false` sorguda DIRECT
gerçekten belgesiz çalışıyor (ölçüldü: RAG→`conditioned=true`, DIRECT→`false`, farklı yanıt); **ama
model kendisi "belge iste" dediğinde DIRECT de RAG gibi davranıyor** → arayüzdeki "DIRECT (Yalın
Model)" her zaman yalın değil. `mode="XYZ"` doğrulanmıyor, sessizce "RAG değil" gibi davranıyor.

### B10 — ORTA · Arayüz telemetrisi
- "Router Uzman Dağılımı" daima `0.00 | 0.00 | 0.00`: istemci nesne bekliyor
  (`router_experts.anka/.carpenter/.pedagogy`, `index.html:375-377`), sunucu **dizi** gönderiyor
  (`["grammar_core","carpenter"]`) → `Number(undefined || 0).toFixed(2)`.
- Durum çubuğu sahte: servis edilen HTML'de `api/status` **0 kez** geçiyor → gerçek sağlık hiç okunmuyor.
- `epistemic_failure` yalnız B5'teki konjunksiyon sağlandığında ateşleniyor (ölçüldü: ateşlendiği vaka var).

### B11 — DÜŞÜK · HTTP katmanı
`HTTP/1.0`, `Server: BaseHTTP/0.6 Python/3.14.8` (sürüm ifşası), yanıtlarda `Content-Length` yok,
CORS yok; `HEAD`/`OPTIONS`/`PUT`/`DELETE` → 501; `GET /api/query` → 404.

## 3. Vektör bellek ve zenginleştirme (11.10 akşamı ölçümü)

| şey | öğlen | akşam |
|---|---|---|
| `anka_bellek` | 38 belge | **46** |
| bileşim | — | 36 sentetik `realistic_rag_p2` + **8 alan belgesi** (`carpenter`: gurgen, kesici_takimlar, kirlangic_kuyrugu, zivana · `gardener`: domates, patlican_biber, salatalik, toprak_gubre) + 2 damgalı eski belge |
| servis edilen `index.html` | `ade12afd82e2…` | **aynı** |
| yanıt şeması | 19 anahtar | 19 anahtar |

Eklenen 8'in kimliği metadata'da **damga olmadığı için kanıtlanamadı** (38=36+2, 46=36+8+2 aritmetiği +
belgelerin uzman alanlarıyla örtüşmesi → çıkarım). Kod/HTML değişmediğine göre değişen **çalışan
ağırlıklar + DB**; `/api/status` model alanı taşımadığı için checkpoint kimliği uçtan okunamıyor.

### Zenginleştirmenin etkisi (ölçülmüş)

- **Retrieval isabeti arttı:** alan içi 5/5 sorguda doğru belge geldi (domates→`domates`,
  kırlangıç→`kirlangic_kuyrugu`, zıvana→`zivana`, bileme→`kesici_takimlar`).
- **Üretim belgeye bağlanmıyor:** aynı 5 vakada **belge↔yanıt sözcük örtüşmesi %0** (≥5 harfli içerik
  sözcükleri). Örnek: `domates` belgesi "ilkbaharda don tehlikesi geçtikten sonra…" derken yanıt
  "Eğer kılcal nem testi dikkate alınmalıdır." → `conditioned=true` görünse de belge çıktıya yansımıyor.
- **Bir sorguda retrieval geriledi:** "Meşe ahşabı nerelerde kullanılır?" → tepe belge
  `kesici_takimlar` (alet bileme, 1,3333); doğru belge `mese_agaci` **2. sırada** (0,8889).
- **İlgisiz belge eşiği geçiyor:** "Python'da listenin toplamı" → `Fomalhaut` (astronomi, 0,4889) >
  0,33 → `conditioned=true`.
- **Retrieval kararsız (eşit skorda tie-break):** "Kırlangıç kuyruğu nedir?" 3 koşunun 2'sinde 301 krk
  belgeyi, 1'inde 108 krk'lık başka belgeyi tepe yaptı — **skor ikisinde de 1,1111112**.
- **Uzman seçimi dejenere görünüyor:** 10+ probun **hepsinde** `active_expert="grammar_core"`
  (marangozluk sorusunda bile). T-0208'in "dinamik uzman dağıtımı" iddiası bu uçtan doğrulanamadı.

## 4. Sayı iddiası çelişkisi (ölçüldü)

| kaynak | birim test sayısı |
|---|---|
| konsolun ilan sayfası | **320/320** |
| `README.md` rozeti + tablo (28 Eyl damgalı) | **307/307** |
| `tests/` içinde AST ile `def test_` | **281** |
| `data/eval/*` damgalı koşum (19 Eyl) | 213 passed / 1 failed (sandbox soket yasağı) |
| panodaki `kristal_llm_status` ilanı | 19/19 |

Depoda "320" dizesi **hiç geçmiyor**. Testler bu makinede koşturulamadı (torch/checkpoint klonda yok)
→ **ölçülemedi**.

## 5. Ölçülemeyenler

- Sunucu logları ve süreç listesi (MacAir'de SSH kapalı) → hangi `--host`/model ile açıldığı beyanla ölçülmedi.
- Çalışan modelin kimliği; `data/future_train_vector.jsonl` boyutu/içeriği.
- `tests/test_agent_gateway.py` HTTP yüzeyi için **tek happy-path** testi içeriyor: hatalı tip, büyük
  gövde, eşzamanlılık ve yetki testi yok → B1/B3/B4/B6 yolları testlerden geçmiyor.

## 6. Yeniden üretim (dışarıdan, yazma üretmez)

```bash
# sağlık ve belge sayısı
curl -s http://192.168.1.10:8080/api/status

# 89 baytlık gövdeyle sunucuyu ~50-120 sn meşgul etme (DoS yolu — dikkatli koşun)
curl -s -X POST http://192.168.1.10:8080/api/query -H 'Content-Type: application/json' \
  -d '{"query":"Test","mode":"RAG","temperature":9,"top_k":999,"max_new_tokens":99999}'

# tip doğrulaması yok: yanıtsız kapanma beklenir
curl -s -X POST http://192.168.1.10:8080/api/query -H 'Content-Type: application/json' -d '{"query":null}'

# retrieval sıralaması (yazma üretmez)
curl -s -X POST http://192.168.1.10:8080/api/check -H 'Content-Type: application/json' \
  -d '{"query":"Meşe ahşabı nerelerde kullanılır?","collection":"anka_bellek","top_k":3}'
```

## 7. Ölçümün yan etkisi (açık kayıt)

`/api/status` → `epistemic_backlog_samples`: öğlen **4 → 10** (denetim probları), akşam **10 → 13**
(bu turun katkısı **+1**; 14:xx-16:xx arasındaki 2 artış bu denetimden değil). Yazma, B5'teki koşul
sağlandığında `data/future_train_vector.jsonl`'e düşer; dosyaya erişim olmadığı için geri alınmadı.

## 8. Sınır

Bu rapor **dış denetimdir**, projenin tesliminin parçası değildir: deponun çalışma ağacı değiştirilmedi,
hiçbir kaynak dosya düzenlenmedi. Dosya `data/eval/` altına, `CLAUDE.md` **kural 3**'ün ölçüm/eval
raporları için tanıdığı istisnaya dayanılarak konuldu — `.agent-bus/SPEC.md` **kural 1** (kiralama
olmadan yazma yok) bu yazma için operatör izniyle aşılmıştır; izlenmesi istenirse rapor için bir bus
görevi açılıp `writes[]`'e bu yol eklenmelidir.
