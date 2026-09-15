# agent-bus — Ajanlar Arası Koordinasyon Protokolü

**Sürüm:** 1.0 · **Tarih:** 2026-09-14 · **Durum:** tasarım (uygulama bekliyor)

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
  state/
    tasks/T-0001.json
    leases/<slug>.json
    results/T-0001.json
    inbox/<recipient>/<ISO8601>-<sender>.json
  log/events.jsonl           # append-only denetim kaydı
```

`state/` ve `log/` **sürümlenmez** (`.gitignore`); `SPEC.md` ve `frozen.json` sürümlenir.

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
   Varsayılan donmuş küme: `data/realistic_rag/**`, `data/b1_5_splits/**`, `data/*.pt`,
   `data/vocab*.json`, `data/lexicon/**`.

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
  "changed_files": ["scripts/agent_bus_mcp.py"]
}
```

### Mesaj — `state/inbox/<recipient>/<ISO8601>-<sender>.json`
```json
{ "from": "claude", "to": "antigravity", "sent": "...", "subject": "...", "content": "...", "read": false }
```
Dosya adı saniye çözünürlüğündedir: aynı gönderen aynı saniyede iki mesaj gönderirse
ad çakışır ve **ilki sessizce kaybolur**. Uygulama çakışmayı algılamak ve adı
`<ISO8601>-<sender>-<n>.json` olarak ayırmak zorundadır.

## MCP araçları

| araç | girdi | çıktı |
|---|---|---|
| `bus_post_task` | title, spec, writes[], acceptance[], to?, ttl_minutes? | `{id}` |
| `bus_list_tasks` | status? | `[task]` |
| `bus_claim_task` | id, owner | `{ok, conflicts[]}` |
| `bus_acquire_lease` | paths[], task_id, owner, ttl_minutes? | `{ok, conflicts[]}` |
| `bus_release_lease` | paths[], task_id | `{ok}` |
| `bus_lease_status` | paths? | `[lease]` |
| `bus_report_result` | task_id, status, summary, evidence[], changed_files[] | `{ok}` |
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
