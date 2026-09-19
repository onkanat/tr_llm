# agent-bus — Ajanlar Arası Koordinasyon Protokolü

**Sürüm:** 1.0 · **Tarih:** 2026-09-14 · **Durum:** **uygulandı ve çalışıyor** (`scripts/agent_bus_mcp.py`) · **son düzenleme:** 2026-09-19 (T-0081)

İki ajan aynı repoda çalışır: **danışman** (Claude Code — analiz, doğrulama, kapı tasarımı)
ve **yürütücü** (Antigravity — eğitim, refactor, betik, test).

## Amaç

Hedef, iki modelin aynı repoda **paralel** çalışmasıdır: duvar saati süresini kısaltmak ve
sistem kaynaklarını (tek GPU, tek disk, tek bağlam penceresi) verimli kullanmak. agent-bus
bu hedefin **aracıdır, kendisi değil**.

Protokolün işi, paralelliği mümkün kılan tek şeyi sağlamaktır: **bir ajanın diğerinin
üzerine yazmasını yapısal olarak imkânsız kılmak.** Kilit olmadan iki ajan aynı dosyaya
dokunamaz; kilit yoksa paralellik de yoktur, yalnızca sıralı çalışma vardır.

Bu ayrım bağlayıcıdır: **bus'ı kendi başına cilalamak hedefe hizmet etmez.** Bir görev bus'ı
kullanmıyorsa, o görev paralelliğe katkı yapmıyor demektir — T-0001'den T-0006'ya kadar altı
görev bus'ın kendi doğruluğuyla geçti ve hiçbiri gerçek iş üretmedi. Yedinci görev gerçek iş
olmalıdır.

## Gerekçe (gerçek olaydan)

14 Eylül 2026, 11:38:05 — `scripts/neutralize_cf_dataset.py` çalışırken
`data/realistic_rag/val.jsonl`'ı yeniden üretip ezdi. O sırada arm A (09:36) ve arm B
(10:59) checkpoint'leri çoktan kaydedilmişti; yani A/B'nin val kayıpları, artık var olmayan
bir değerlendirme kümesine dayanıyordu. Yeniden üretim denemesi: A raporladı 4.84, ölçülen
6.36; B raporladı 8.72, ölçülen 7.85. Kayıp **hesaplama eksikliğinden değil, yol kilidi
eksikliğinden** geldi. Bu protokol o sınıfı kapatır.

## Kök dizin

```
.agent-bus/
  SPEC.md                    # bu dosya (sürümlenir)
  frozen.json                # değiştirilemez yol desenleri
  tasks/T-XXXX.json          # kalıcı görev şartnameleri (sürümlenir)
  notes/T-XXXX.md            # görev yürütüm anlatı günlükleri (sürümlenir)
  state/
    tasks/T-0001.json
    leases/<slug>.json
    results/T-0001.json
    inbox/<recipient>/<ISO8601>-<sender>.json
  log/events.jsonl           # append-only denetim kaydı
```

`state/` ve `log/` **sürümlenmez** (`.gitignore`); `SPEC.md`, `frozen.json`, `tasks/` ve `notes/` **sürümlenir** — *ancak bu bir NİYETTİR, ölçülen durum değildir.* 19 Eyl 2026 ölçümü: `.agent-bus/notes/` altında diskte **44** dosya var, **27'si izleniyor**, **17'si izlenmiyor** (`git status` → `??`). İzlenen aralık **T-0031…T-0061** (aralarda boşluklar var), izlenmeyen küme **T-0062…T-0077** + **T-0081**. Yani sürümleme **T-0061'de durmuş** ve bir daha başlamamıştır. Bir dosyanın `notes/` altında *durması* onun sürümlendiği anlamına **gelmez**; commit yetkisi operatördedir ve `git add -A` yasaktır (D8, T-0081). `notes/` yüzeyi `state/` ve `log/`'dan farklıdır; yürütücünün gerekçe zincirini repo içinde kalıcı ve keşfedilebilir kılar. Elle tutulan bir indeks dosyası yoktur; dizin listesi (`ls .agent-bus/notes/`) zaten indekstir.


`tasks/` (kök altı, sürümlenir) ile `state/tasks/` (çalışma zamanı, sürümlenmez) kasıtlı
olarak ayrıdır ve **ikisi de okunur**: kökteki dosyalar elle yazılmış talimatlardır
(danışman ajanın doğrudan kaleme aldığı görev şartnameleri), `state/tasks/` ise
`bus_post_task`'ın ürettiği çalışma zamanı kuyruğudur. Aynı kimlik iki yerde bulunursa
`state/tasks/` kazanır.

## Üç değişmez kural

1. **Kiralama olmadan yazma yok.** Bir ajanın değiştirdiği her yol, o ajanın elindeki
   bir kiralamayla kapsanmalıdır (tam yol ya da dizin öneki). İhlal → yazma reddedilir.
2. **Donmuş artefakt.** `frozen.json`'daki desenlere uyan yollar yalnızca bu yolu açıkça
   bildiren **ve** dizin geneli kiralama alan bir görev tarafından değiştirilebilir.
   Donmuş kümenin **tek yetkili kaynağı `frozen.json`'dır**; 19 Eyl 2026 ölçümüyle **10
   desen**: `data/realistic_rag/**`, `data/b1_5_splits/**`, `data/pedagogy_canonical/**`,
   `data/lexicon/**`, `data/*.pt`, `data/*.bin`, `data/vocab.json`, `data/vocab_entity.json`,
   `src/llm/tokenizer.py`, `src/compiler/**`. Bu liste `bus_frozen_list` çıktısı ve
   `frozen.json` ile **iki yönlü** hizalıdır. *(Bu paragraf eskiden 5 desen sayıyordu ve tam
   da en kritik ikisini — `src/compiler/**` ile `src/llm/tokenizer.py` — atlıyordu ⇒ bir
   sonraki pencere onları "korumasız" sanabilirdi. D7, T-0081.)*

   Uygulama bu iki şartı **birlikte** dayatır: (a) yol görevin `writes` listesinde
   bildirilmiş olmalı, (b) kiralama `scope: "dir"` olmalı — donmuş bir yola **dosya
   kapsamlı kiralama verilmez**. 14 Eylül 11:38 kazasının şekli tam olarak (b)'nin
   yokluğudur: `val.jsonl`'ı bildiren ama yalnızca o dosyayı kilitleyen bir görev,
   aynı betiğin `train_cf_3000.jsonl`'ı da ezmesini engellemez.
3. **Kiraların TTL'i vardır.** Varsayılan 120 dakika. Süresi dolan kiralama devralınabilir
   ve devralma `events.jsonl`'e yazılır.

## Şema

### Görev — `state/tasks/T-XXXX.json`
```json
{
  "id": "T-0001",
  "from": "claude",
  "to": "antigravity",
  "created": "2026-09-14T11:50:00Z",
  "title": "kısa başlık",
  "spec": "yapılacak işin tam tanımı",
  "writes": [".agent-bus/", "scripts/agent_bus_mcp.py"],
  "acceptance": ["çalıştırılacak komut", "beklenen çıktı"],
  "status": "open",
  "claimed_by": null,
  "claimed_at": null,
  "ttl_minutes": 120
}
```
`status` ∈ `open | claimed | done | blocked`. Tersine dönüş yok: `done` tekrar açılmaz,
yeni görev açılır.

### Kiralama — `state/leases/<slug>.json`
```json
{
  "path": "data/realistic_rag/val.jsonl",
  "scope": "file",
  "task_id": "T-0001",
  "owner": "antigravity",
  "pid": "12345",
  "acquired": "2026-09-14T11:50:00Z",
  "expires": "2026-09-14T13:50:00Z"
}
```
`scope` ∈ `file | dir`. `slug`, yolun `/` → `__` ve `.` → `_` dönüşümüdür.
Aynı yola iki farklı `owner` kiralaması **asla** birlikte var olamaz.

### Sonuç — `state/results/T-XXXX.json`
```json
{
  "task_id": "T-0001",
  "status": "done",
  "finished": "2026-09-14T12:10:00Z",
  "summary": "ne yapıldı",
  "evidence": ["komut", "çıktı özeti", "ölçülen sayılar"],
  "changed_files": ["scripts/agent_bus_mcp.py"],
  "narrative_log": {
    "path": ".agent-bus/notes/T-0001.md",
    "sha256": "3a7b...",
    "ozet": "yürütüm gerekçe ve karar özeti"
  }
}
```
`narrative_log` alanı opsiyoneldir: `{"path": ".agent-bus/notes/T-XXXX.md", "sha256": "<64 hex>", "ozet": "<kisa>"}`. Alan uzun metni taşımaz; yalnız işaretçi + özet taşır. Böylece sonuç dosyasını okuyan danışman anlatı günlüğünü sıfır keşif maliyetiyle bulabilir.

**Eşzamanlı Yazım İlkesi:** Note yazımı RAPOR ANINDA DEĞİL, iş SÜRERKEN eşzamanlı olarak yapılır. Rapordan sonra yazılan anlatı bir rekonstrüksiyondur ve çerçeveleme iyimserliği tam oraya sızar (T-0016/T-0018'de dört kez ölçüldü: sayılar kusursuz, çerçeve iyimser). Ara düzeltmenin izi yalnız eşzamanlı yazımda yaşar.

### Mesaj — `state/inbox/<recipient>/<ISO8601>-<sender>.json`
```json
{ "from": "claude", "to": "antigravity", "sent": "...", "subject": "...", "content": "...", "read": false }
```
Dosya adı saniye çözünürlüğündedir: aynı gönderen aynı saniyede iki mesaj gönderirse
ad çakışır ve **ilki sessizce kaybolur**. Uygulama çakışmayı algılamak ve adı
`<ISO8601>-<sender>-<n>.json` olarak ayırmak zorundadır.

### `bus_send` Atıf Kuralı

Mesajlarda bir artefakta atıf yapılırken:
1. **Yol + SHA-256 Kuralı:** Bir artefakta atıf her zaman `YOL + sha256` biçiminde yapılmalıdır (örneğin: `data/eval/t0030_wiki_revival_2026-09-15.json` `sha256: 8a9f...`).
2. **Bölüm/Satır Numarasına Atıf YASAKTIR:** Bölüm numaraları (`Section 8.13`) veya satır numaraları dosya içeriği kaydığında yeşil kalır ve sahte doğrulama yanılsaması üretir.
3. **Uzun Alıntı YASAKTIR:** Rapor veya not içeriğinin uzun parçaları mesaja kopyalanmaz (ikinci bir kayma yüzeyi oluşturur; ayrıca `state/inbox/` sürümlenmez). Kritik kararlar ve olgular mesaja değil, kiralanmış sahipli yüzeylere (`SPEC.md`, `results/`, `notes/` veya `data/eval/`) yazılır.

## MCP araçları

| araç | girdi | çıktı |
|---|---|---|
| `bus_post_task` | title, spec, writes[], acceptance[], to?, ttl_minutes? | `{id}` |
| `bus_list_tasks` | status? | `[task]` |
| `bus_claim_task` | id, owner | `{ok, conflicts[]}` |
| `bus_acquire_lease` | paths[], task_id, owner, ttl_minutes? | `{ok, conflicts[]}` |
| `bus_release_lease` | paths[], task_id | `{ok}` |
| `bus_lease_status` | paths? | `[lease]` |
| `bus_report_result` | task_id, status, summary, evidence[]?, changed_files[]?, narrative_log? | `{ok}` |
| `bus_send` | to, subject, content | `{ok}` |
| `bus_inbox` | who, unread_only? | `[message]` |
| `bus_frozen_list` | — | `[glob]` |

Çakışma döndüren araçlar (`claim_task`, `acquire_lease`) **kısmi başarı uygulamaz**:
çakışma varsa hiçbir kiralama yazılmaz.

## Yürütücü çevrimi (yoklama sözleşmesi)

Bus **çekme (pull) tabanlıdır**: hiçbir şey yürütücüyü uyarmaz, bildirim göndermez.
Görevi almak için yürütücünün kendisi sormak zorundadır. Bu yüzden çevrim yürütücünün
sorumluluğundadır ve sözleşmesi şudur:

1. **Yokla.** `bus_list_tasks(status="open")` çağır.
2. **Süz.** `bus_list_tasks` yalnızca `status` ile süzer, **alıcıya göre süzmez**.
   `to` alanı kendi adın olan (veya `to` alanı boş, yani ilan edilmiş) görevleri kendin
   ayıkla. Başkasının devraldığı görevleri de bu adımda ele — onlar şu an yürütülüyor,
   hata değil.
3. **Seç.** Kalan adaylar arasından **en küçük açık kimliği** seç (`T-0007` < `T-0008`).
   Sıra kuraldır, tercih değildir: bu kural "hangisinden başlayayım?" sorusunu ortadan
   kaldırır — cevap her zaman dosyadan okunur, sorulmaz.
4. **Devral.** `bus_claim_task(id, owner)`. Kendi devraldığın bir görevi tekrar devralmak
   **serbesttir**: yeniden başlayan çevrim kendi işine kaldığı yerden devam eder.
5. **Tek görev.** Aynı anda **tek** görev yürütülür. Zaten devralınmış bir görevin varsa
   yenisini devralma; önce onu bitir.
6. **Kirala, raporla, tekrarla.** Yazmadan önce `bus_acquire_lease`, bitince
   `bus_report_result`, sonra `bus_release_lease`; ardından 1. adıma dön.
7. **Boş kuyrukta sessizce bekle.** Açık görev yoksa **soru sorma, mesaj gönderme, boş
   rapor yazma** — bekle ve sonra tekrar yokla. Cevapsız bir soru çevrimi durdurur;
   bu sözleşmede bekleyen yürütücü doğru davranan yürütücüdür.

**Ölçülmüş devralma davranışı** (14 Eyl 2026, gerçek sunucu):

| durum | sonuç |
|---|---|
| başka sahip devralınmış görevi devralır | `ok: false`, `conflicts` dolu |
| **aynı sahip kendi görevini tekrar devralır** | `ok: true` — çökme sonrası devam yolu |
| `done` görev tekrar devralınır | `ok: false`, terminal |
| var olmayan kimlik | temiz hata |

Bilinmesi gereken sınır: **devralmanın süresi dolmaz.** Görev kaydında `ttl_minutes`
alanı vardır ama devralmaya uygulanan bir zaman aşımı yoktur; farklı bir sahibin bayat
devralması sırayı süresiz kilitler. Tek yürütücülü kurulumda bu zararsızdır (aynı sahip
her zaman devam edebilir). İkinci bir yürütücü eklenirse önce devralma zaman aşımı
gerekir — bu, protokolün bilinen bir eksiğidir, sessiz bir tuzak değil.

## Bilinen açık kusurlar (yalnız ÖLÇÜLMÜŞ olanlar)

Bu bölüm **yalnız ölçülmüş** maddeleri taşır; her madde bir ölçüm tarihi ve sınanabilir bir
iddia içerir. Ölçülüp **çürütülen** iddialar buraya yazılmaz (aşağıda listelenir).

1. **`task_updated` olayı YOK.** `log/events.jsonl`'de 19 Eyl 2026 itibarıyla **1.174 olay**
   ve **6 tip** vardır: `lease_acquired` (409) · `lease_released` (372) · `message_sent` (158)
   · `result_reported` (87) · `task_claimed` (79) · `task_posted` (69). Bir görev
   şartnamesinin (`spec`, `writes`, `acceptance`) **sonradan değişmesi hiçbir olay üretmez**;
   bu yüzden kapsam genişletildiğinde denetim kaydı bunu **göstermez**. (Kaynakta
   `task_updated` dizgesi **0** kez geçer.) Bu, kapsam genişletmelerinin **yazılı** bir
   kaydını zorunlu kılar.
2. **`read` bayrağı hiçbir zaman `true` yapılmaz.** `bus_send` mesajı `read: false` ile
   yazar; kaynakta `read` yalnızca **okunur**, onu `true`'ya çeviren bir yol **yoktur** ⇒
   `bus_inbox(unread_only=True)` pratikte **her** mesajı döndürür. "Okundu" bilgisi
   güvenilir değildir; bir mesajın işlendiği, **yanıtın veya sonucun varlığıyla** anlaşılır.
3. **Kök `tasks/` yüzeyi okunur ama yazılmaz.** Görev araması **iki** dizini tarar
   (`state/tasks/` + `.agent-bus/tasks/`), fakat durum değişiklikleri **yalnız**
   `state/tasks/` altına yazılır. Sonuç: yalnız kökte bulunan elle yazılmış bir şartname
   **okunur ama durumu asla güncellenmez** (bayat `status`). Aynı kimlik iki yerde varsa
   `state/tasks/` kazandığı için bu sessiz bir yanlış okuma değil, **bayat bir kopyadır**.
4. **Devralmanın süresi dolmaz** — yukarıda "Bilinmesi gereken sınır" bölümünde belgeli.
   Tek yürütücülü kurulumda zararsız; ikinci bir yürütücüde sıra **süresiz kilitlenir**.
5. **`notes/` canlı durum tablosu taşımamalıdır.** Notlar iş **sürerken** yazılır; içlerindeki
   "şu an durum X" satırları **yazıldıkları ana** aittir. Ölçülmüş sonuç: canlı iddialar ya
   damgalanmalı (`... itibarıyla`) ya da yazımdan önce `state/`'ten **yeniden okunmalıdır**;
   aksi halde not, bayat bir hükmü kalıcılaştırır.

**Çürütülen iddialar (buraya yazılmadı, kayda geçer).** *"Okuma `state/`, yazma köke — asimetri
kusurdur"* iddiası **ölçümle çürüdü**: yön terstir (yazma `state/`'e) ve daha önemlisi bu
tasarım yukarıda "Kök dizin" bölümünde **açıkça belgelenmiştir**, kusur değildir. *"Ayna diff'i
kusurdur"* da yanlıştı: aynalı görevlerde görülen fark **beklenen** davranıştır.

## Uygulama gereksinimleri

- **Konum:** `scripts/agent_bus_mcp.py`
- **Bağımlılık: yok.** Yalnızca Python stdlib. MCP stdio taşıması satır-sınırlı
  JSON-RPC 2.0'dır; `initialize`, `tools/list`, `tools/call` yeterlidir. Repo venv'ine
  yeni paket eklenmez.
- **Taşıma:** stdio. `agy mcp add` ile stdio sunucusu olarak kaydedilir.
- **CLAUDE.md uyumu:** tip ipuçları zorunlu; saf fonksiyonlar, global değiştirilebilir
  durum yok; hatalar sessizce yutulmaz, loglanır ve uygun varsayılan döner; algoritma
  deterministik — kararlarda rastgelelik veya duvar saati yok (zaman damgası yalnızca
  üst veri).
- **Atomik yazma:** `<dosya>.tmp` yaz → `os.replace`. Yarım dosya gözlenemez.
- **Denetim:** her durum değişimi `log/events.jsonl`'e bir satır ekler (append-only).
- **Yol güvenliği:** kök dışına çıkan (`..`) yollar ve mutlak yollar reddedilir; tüm
  yollar köke göre normalize edilir. Bu kural **yalnızca `paths` ve `writes` ile sınırlı
  değildir**: ajan adları (`from`, `to`, `owner`, `who`) ve `task_id` yol değildir ama
  dosya adına girer ve aynı `..` kaçışını taşır. Adlar `^[a-z0-9][a-z0-9_-]{0,63}$`,
  görev kimliği `^T-\d{4}$` desenine uymalıdır; uymayan değer reddedilir.

## Kabul kriteri

Tek komutla çalışan bir tur testi:
1. `bus_post_task` → görev açılır
2. `bus_acquire_lease` ile `val.jsonl` alınır → `ok: true`
3. Aynı yola ikinci `bus_acquire_lease`, **farklı** `owner` ile → `ok: false`, `conflicts` dolu,
   ve diskte **hiçbir yeni kiralama dosyası oluşmamış**
4. `bus_report_result` → `results/T-XXXX.json` yazılır
5. `bus_release_lease` → kiralama silinir

## Kapsam dışı

- Otomatik çakışma çözümü yok. Çakışma raporlanır, karar ajana aittir.
- Süreçler arası kilit (flock) yok. Kiralama işbirliğe dayanır; denetim kaydı ihlali görünür kılar.
- Ağ taşıması yok. Yalnızca aynı makinedeki yerel dosya sistemi.
