---
name: agent-bus
description: >-
  agent-bus görev kuyruğunu yoklar ve yürütücü çevrimi sözleşmesine göre işler: en küçük açık
  kimliği devralır, yazmadan önce kiralar, bitince ölçüme dayalı rapor yazar. "Hangi görevden
  başlayayım?" sorusunu ortadan kaldırır; kuyruk boşken sessizce bekler.
---

# agent-bus Yürütücü Çevrimi

Bu beceri çağrıldığında bus kuyruğundaki işleri aşağıdaki çevrime göre yürütürsün.
Sözleşmenin kaynağı `.agent-bus/SPEC.md` -> **"## Yürütücü çevrimi (yoklama sözleşmesi)"**.
Bu dosya ile SPEC.md çelişirse **SPEC.md kazanır**.

## Çevrim (her turda)

1. **Gelen kutusu.** `bus_inbox(who="antigravity")` çağır. Okunmamış mesaj varsa içeriğini uygula.
   Bus bir *çekme* kanalıdır: mesaj kendiliğinden gelmez, yalnızca sen sorunca görünür.

2. **Yokla.** `bus_list_tasks(status="open")`.
   - Görev listesini `.agent-bus/tasks/` **klasörünü tarayarak ALMA.** O klasör sürümlenmiş, elle
     yazılmış görev dosyalarını tutar; çalışma zamanı kuyruğu `state/tasks/` altındadır ve
     `bus_list_tasks` ikisini birleştirir. Klasör taraması, yalnızca state'te duran görevleri
     (örnek: T-0005) hiç görmez.

3. **Süz.** Araç **alıcıya göre süzmez**: `to` alanı `antigravity` olan (veya `to` alanı boş)
   görevleri kendin ayıkla. Başkasının devraldıklarını atla — onlar yürütülüyor, hata değil.

4. **Seç.** Kalan adaylar arasından **en küçük açık kimliği** seç (`T-0007` < `T-0008`).
   Sıra kuraldır, tercih değil. **"Hangisinden başlayayım?" diye SORMA** — cevap dosyadan okunur,
   sorulmaz. Bu soru bir kez çevrimi durdurdu; artık gereksizdir.

5. **Devral.** `bus_claim_task(id, owner="antigravity")`.
   - Kendi devraldığın görevi **tekrar devralmak serbesttir**: yeniden başlarsan kaldığın yerden
     devam et.
   - **Aynı anda TEK görev.** Devralınmış bir görevin varsa yenisini alma, önce onu bitir.

6. **Kirala.** Yazacağın her yol için, **yazmadan ÖNCE**
   `bus_acquire_lease(paths=[...], task_id=..., owner="antigravity")`.
   - Çakışma dönerse **yazma**. Protokol kısmi başarı uygulamaz: çakışma varsa hiçbir kiralama
     yazılmaz ve karar ajana aittir.
   - **Dosya Düzeyinde Kiralama:** Tekil dosya yazılıyorsa ASLA kapsayıcı dizini (`data/eval/`)
     kiralamaya çalışma; doğrudan üretilen dosya yolunu kovala. Dizin kiralaması diğer etmenlerin
     doğrulama raporu yazmasını saatlerce bloke eder.

7. **Uygula.** Görevin `spec` alanındaki sınırlara uy. Donmuş yolları (`bus_frozen_list`) yalnızca
   OKU. Donmuş bir dizine yazman gerekiyorsa `scope: "dir"` kiralaması şarttır.

8. **Raporla.** `bus_report_result(task_id, status="done", summary=..., evidence=[...],
   changed_files=[...], narrative_log={...})` ardından `bus_release_lease(paths, task_id)`.
   - `evidence` alanına **ölçüm** yaz, iddia değil: çalıştırdığın komut ve aldığın çıktı.
   - **Komutlarda Kısaltma Yasağı:** Rapor ve notlardaki komutlar asla `...` ile kısaltılamaz;
     bağımsız doğrulayıcının doğrudan çalıştırabileceği tam komut yazılır.
   - **Atıf Kuralı:** Artefakt atıfları yalnızca `YOL + SHA-256` ile yapılır; bölüm/satır numarası
     kullanılmaz (içerik kayınca yeşil kalır), uzun alıntı yapılmaz.
   - Raporun ilk satırı şu biçimde olsun: `bus kaydı: <görev id> | kiralanan yollar`.

9. **Not ve Anlatı Günlüğü (.agent-bus/notes/T-XXXX.md).**
   - **Repo-İçi Yüzey:** Görevin gerekçe zinciri ve ara ölçümleri repo içinde `.agent-bus/notes/T-XXXX.md`
     dosyasına yazılır (görev spec'inde `notes/` varsa kiralanır). Repo dışı `walkthrough.md` yalnızca
     yerel oturum hafızası içindir; danışmana resmi aktarım `notes/` ve sonuç JSON'u üzerinden yapılır.
   - **Eşzamanlı Yazım İlkesi:** Not rapor anında sonradan uydurularak (rekonstrüksiyon) değil;
     iş sürerken eşzamanlı tutulur.
   - **"As-of" Zaman Damgası:** Not kalıcı bir yüzey olduğu için `"Durum: Devam ediyor"` gibi sonradan
     kendini yalanlayacak canlı durum ifadeleri yerine `"Durum (14:05:00Z itibariyle): Devam ediyor"`
     zaman damgalı anlık görüntü kullanılır.
   - **Hash Tazeleme Kuralı:** Değişen dosyaların hash'leri not içerisine statik tablo olarak
     yazılmaz; rapor anında tazelenir veya nihai rapor JSON'una (`data/eval/*.json`) atıf verilir.
   - **narrative_log Bağlantısı:** `bus_report_result` çağrılırken `narrative_log: {"path": ".agent-bus/notes/T-XXXX.md", "sha256": "<hash>", "ozet": "<kısa>"}`
     şeklinde not dosyası sonuç kaydına bağlanır.
   - **Tarihi Kayıt İlkesi:** Ölçüt veya karar düzeltmelerinde eski metin silinmez; `onceki_metin`,
     düzeltme tarihi, gerekçesi ve ölçüm kaynağı (yol + sha256) açıkça korunur.

10. **Tekrarla.** 2. adıma dön. **Açık görev yoksa** sessizce bekle; soru sorma, mesaj gönderme,
    boş rapor yazma. Bu sözleşmede bekleyen yürütücü, doğru davranan yürütücüdür.

## Yasaklar

- **Kiralamasız yazma.** (SPEC kural 1 — tek istisnası yok.)
- Donmuş varlığı değiştirme; `scope: "dir"` kiralaması olmadan donmuş bir dizine yazma.
- Görevi klasör tarayarak keşfetmeye çalışma (2. adım).
- Boş kuyrukta soru sorma (10. adım).
- Gerçek bir uyumsuzluğu `strict=False` veya eşdeğeri bir susturmayla gizleme.
- Bus'ın kayıt dizinlerini (`state/tasks`, `state/leases`, `state/results`, `log/events.jsonl`)
  elle düzenleyerek durumu "düzeltme" — durum yalnızca araçlarla değişir.
- Raporlarda veya notlarda komutları `...` ile kısaltma.
- Tekil dosya yazarken geniş dizin (`data/eval/`) kiralaması alma.
- Atıflarda bölüm veya satır numarası kullanma; yalnızca `YOL + SHA-256`.

## Araçlar

`bus_post_task` · `bus_list_tasks` · `bus_claim_task` · `bus_acquire_lease` · `bus_release_lease`
· `bus_lease_status` · `bus_report_result` · `bus_send` · `bus_inbox` · `bus_frozen_list`

## Devralma davranışı (14 Eyl 2026'da ölçüldü)

| durum | sonuç |
|---|---|
| başka sahip devralınmış görevi devralır | `ok: false`, `conflicts` dolu |
| **aynı sahip kendi görevini tekrar devralır** | `ok: true` — çökme sonrası devam yolu |
| `done` görev | `ok: false`, terminal |
| var olmayan kimlik | temiz hata |

Bilinmesi gereken sınır: **devralmanın süresi dolmaz.** `ttl_minutes` görev kaydında vardır ama
devralmaya uygulanmaz; farklı bir sahibin bayat devralması sırayı süresiz kilitler. Tek yürütücülü
kurulumda zararsızdır. Bir görev beklenmedik biçimde ilerlemiyorsa önce `bus_lease_status` ve
`bus_list_tasks` ile kimin neyi tuttuğuna bak.
