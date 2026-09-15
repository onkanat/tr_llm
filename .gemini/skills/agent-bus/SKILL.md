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
   Çakışma dönerse **yazma**. Protokol kısmi başarı uygulamaz: çakışma varsa hiçbir kiralama
   yazılmaz ve karar ajana aittir.

7. **Uygula.** Görevin `spec` alanındaki sınırlara uy. Donmuş yolları (`bus_frozen_list`) yalnızca
   OKU. Donmuş bir dizine yazman gerekiyorsa `scope: "dir"` kiralaması şarttır.

8. **Raporla.** `bus_report_result(task_id, status="done", summary=..., evidence=[...],
   changed_files=[...])` ardından `bus_release_lease(paths, task_id)`.
   - `evidence` alanına **ölçüm** yaz, iddia değil: çalıştırdığın komut ve aldığın çıktı.
   - Raporun ilk satırı şu biçimde olsun: `bus kaydı: <görev id> | kiralanan yollar`.

9. **Yazılı rapor.** Bulguları `walkthrough.md`'ye işle; planı değiştiyse
   `implementation_plan.md`'yi de güncelle. Danışman ajan bu iki dosyayı **her işten sonra**
   otomatik olarak okur — bus kaydı makine içindir, bu dosyalar insan içindir; ikisi de gerekir.
   - `bus_report_result`'ın `evidence` alanına **walkthrough.md'nin tam yolunu** ekle
     (`~/.gemini/antigravity/brain/<oturum-id>/walkthrough.md`). Yol açıkça yazılmazsa danışman
     onu mtime tahminiyle aramak zorunda kalır ve yanlış oturumun raporunu okuyabilir.
   - **Plan dosyası otomatik güncellenmiyor:** 14 Eyl 2026'da `implementation_plan.md` 05:02'de
     kalmışken `walkthrough.md` 16:21'de güncellendi. İşin *ne yaptığını* yalnızca walkthrough
     anlatır; plan yalnızca niyeti gösterir. İkisini karıştırma.

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
