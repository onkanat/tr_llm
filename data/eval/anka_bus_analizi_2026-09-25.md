# T-0116 — AGENT-BUS YAPI ANALİZİ VE ÖNERİLER (ölçümlü, 2026-09-25)

**Tarih:** 2026-09-25 · **Yöntem:** salt-okunur envanter betiği + kanonik
`AgentBus` İMPORT (KOPYA YOK) ile mutasyon sınaması · kanonik bus kodu
DOKUNULMADI. Operatör hedefi: **paralel çalışma / görev dağılımı** —
öneriler yalnız ölçülmüş maddelere dayanır (SPEC "yalnız ölçülmüş" ilkesi).

## 1. Ölçümler (betik `scratch/bus_analiz/t0116_olcum.py` — elle sayı YOK)

| Ölçü | Değer (2026-09-25) | SPEC'teki damgalı kayıt |
|---|---|---|
| `events.jsonl` | **1.587 satır**, 6 tip: task_posted 104 · task_claimed 115 · lease_acquired 556 · lease_released 519 · message_sent 169 · result_reported 124 | 1.191 (20 Eyl) |
| `task_claim_takeover` olayı | kaynakta VAR (satır 411), olay sayısı **0** — hiç tetiklenmedi | SPEC'te YOK |
| `state/tasks/` | 106 görev: 105 done + 1 claimed (T-0116) · to: antigravity 51 / claude 44 / null 11 | — |
| claim→report süresi (n 91) | medyan **13,6 dk** · min 0 · max **1.194,6 dk** | — |
| kiralama diskte | 5; **2 bayat** (T-0106 `scripts/`, `tests/`; expires 2026-09-24) | — |
| inbox | 169 mesaj · **99 read=false (%59)** | kayıt 2 |
| kök `tasks/` | 24 şartname; 14'ü `state/tasks/` ile aynı kimlik | kayıt 3 |
| `notes/` diskte | 61 dosya | 44 (20 Eyl) |
| kaynak `task_updated` | grep 0 → **kayıt 1 hâlâ geçerli** (olay yok) | kayıt 1 |

## 2. Mutasyon sınaması (kanonik `AgentBus` İMPORT, sınama kiralamaları release edildi)

| Sınama | Sonuç | Okuma |
|---|---|---|
| **PC (aktif çakışma):** `scratch/` aktif T-0116 kiralaması + farklı sahip `probe` | `ok=false`, 1 conflict | **Kapı canlı** — çakışma denetimi çalışıyor |
| **Bayat kiralama:** `scripts/` (T-0106, 24 saat önce doldu) + `probe` acquire | `ok=true` | Bayat kiralama acquire'ı **engellemez** (kaynak satır 543-545: "süresi dolmuş kiralama çakışma üretmez") — ölü satır, zararsız |
| **Aynı sahip yenileme:** `state` + aynı owner/task | `ok=true` | Yenileme yolu çalışıyor |
| **Çift devralma (kaynak okuması):** `claim_task` aynı owner ikinci görev | **engel YOK** (satır 403: çakışma yalnız `claimed_by != owner` dalında) | "Tek görev" kuralı **yalnız çevrim sözleşmesinde** — uygulamada zorunlu DEĞİL |

## 3. Ana bulgu: SPEC kayıt 4 BAYAT — devralma zaman aşımı kaynakta VAR

SPEC "Bilinmesi gereken sınır" bölümü *"devralmanın süresi dolmaz; ikinci bir
yürütücü eklenirse önce devralma zaman aşımı gerekir"* yazıyor. Kaynak
(`scripts/agent_bus_mcp.py` satır 397-411, commit **9e59830**'dan beri)
`is_expired` hesaplıyor, süresi dolan devralmayı `task_claim_takeover`
olayıyla **yeniden devralınabilir** kılıyor. Olay sayısı 0 ⇒ tek yürütücü
konusunda hiç tetiklenmedi ama **yapısal engel artık yok**. Bu,
paralel-yürütücü hedefinin en büyük bilinen eksiklerini KAPATIYOR — SPEC
dokümanı kayıt 4'ü güncellemeli (doküman görevi; bu analiz yazım yapmadı).

## 4. Öneriler (ölçümlü — her biri kendi ölçümüne bağlı)

1. **P1 — SPEC güncellemesi (doküman, düşük maliyet):** kayıt 4 "devralma
   süresi dolmaz" → kaynakla çelişiyor; `task_claim_takeover` mekanizması
   SPEC §Üç değişmez kural/§Bilinen kusurlar'a yazılmalı. SPEC düzenlemesi
   oku→yaz→tekrar-oku disipliniyle ([[spec-duzenlemesi-devralma-ile-yarisir]]).
2. **P2 — İkinci yürütücü artık AÇILABİLİR (operasyonel):** devralma
   kilidi takeover'la çözüldü; çakışma kapısı canlı (mutasyon PC). Kalan
   operasyonel boşluk: yürütücü **pull-tabanlı** — antigravity çevriminin
   otomatik yoklaması (aralıklı `bus_list_tasks`) açılmalı; yoksa ikinci
   yürütücü hiç çalışır görülmez.
3. **P3 — Görev dağılımı deseni (cihaz kısıtına bağlı):** claim→report
   medyan 13,6 dk ⇒ küçük işler hızlı döner. Paralel bölme: **MPS koşumları
   tek cihaz** ([[cift-egitici-ayni-gpu-kilitler]] — koşum-sırası paralel
   koşum 0,39→18,78 sn/adım ölçüldü) ⇒ paralellik koşum-DIŞI işlerde
   açılmalı: betik yazımı, veri hazırlık, raporlama, kanonik-alet-sız
   ölçümler. Öneri: görev şartnamesine `cihaz` alanı (mps/cpu/none) —
   yürütücü seçiminde "koşum görevi tek cihaza sıraya girer" kuralı.
4. **P4 — "tek görev" kuralının zorunlu kılınması (kod dokunuşu, ayrı
   görev):** claim_task aynı owner'a ikinci görevde engel yok; paralel
   yürütücüde bir ajanın iki görevi aynı anda taşıması kiralama çakışması
   yüzeyini büyütür. Öneri: claim_task'ta aynı owner'ın diğer `claimed`
   görev sayısı ≥1 ise reddet — KAPIYA bağlanır
   ([[sayac-iki-yonde-de-yanilir]]; sessiz sayı değil, çakışma dönüşü).
5. **P5 — Bayat kiralama temizliği (öncelik düşük, ölçülü):** acquire bayat
   kiralamayı atlıyor (ölçüldü) ama diskten SİLME yok; diskte 2 bayat.
   Öneri: acquire_lease yazım anında atladığı bayat kiralamayı silsin
   (salt-eklenti); bugün zararsız — ölçüm beklentisi öncelik düşük.
6. **P6 — read-bayrak (kayıt 2) doğrulandı; öneri:** `bus_inbox` okumada
   `read=true` yazımı yok (%59 okunmamış metası yanıltıcı). Öneri:
   bus_inbox çağrısında dönen mesajları işaretleyen bir alan eklemek yerine
   **çevrim düzeltmesi**: yürütücü inbox okuduktan sonra mesaj dosyasını
   siler (ya da bus_inbox'a `--ack` eklenir). Kod dokunuşu ayrı görev.
7. **P7 — `task_updated` olayı (kayıt 1) hâlâ yok:** SPEC şartname
   değişiklikleri denetimsiz. Kod dokunuşu ayrı görev; öncelik: P4/P6
   ile birlikte tek bus-onarım görevi olarak paketlenebilir.
8. **P8 — `to` filtreli liste:** `bus_list_tasks` yalnız `status` süzer;
   antigravity 51 görev manuel süzülüyor ([[yurutucu-liste-filtreleme]]).
   Öneri: `to` parametresi (API dokunuşu, P6/P7 ile aynı görevde).
9. **P9 — Çift kimlik yüzeyi:** kökte 24 elle şartname, 14'ü state ile
   ortak — kayıt 3 desenine uygun (okunur, yazılmaz). Öneri: arşivleme
   değil **belgeleme**: SPEC kök-dizin bölümü zaten açık; dokunuş YOK
   (ölçüm yalnız envanter günceller).

## 5. Çürütülen/uyarı

* *"Bayat kiralama ikinci acquire'ı bloklar"* — **ÇÜRÜLDÜ** (ok=true).
  SPEC "süresi dolan kiralama devralınabilir" cümlesi doğru; diskte
  kalması estetik borç, işlevsel kusur değil.
* *"Devralma süresi dolmaz"* — **ÇÜRÜLDÜ** (kaynak; SPEC kayıt 4 bayat).
* events tip sayısı: SPEC 6 tip kaydı geçerli; **7. tip kaynakta tanımlı
  ama 0 olay** (takeover) — tip listesi 6'dır, "6 tip" kaydı tutarlı.

## 6. Artefakt digest tablosu (tam — önek YOK)

| sha256 (tam) | Artefakt |
|---|---|
| e970116f9a260220d2ddb6cda505b402001bd779c7f6fee791610f478301dc2d | scratch/bus_analiz/t0116_olcum.py |
| f644c6b531aca673b7e237d20650999da1e7f88e90f247f36de5d5de5b9b964a | scratch/bus_analiz/t0116_olcum.json |
| d0a3ddff8a814af1602fc0552ec7dbd3f241fce4a9ef82c1734c9b6f0ab430cc | scratch/bus_analiz/t0116_mutasyon.py |
| bbe9698461793381cd214682a4c9f55b9db007ad965c15a802abbaf1f26d4662 | scratch/bus_analiz/t0116_mutasyon.json |
| 497badb9729c085a8618359fb9dd024dcbbd257d909ccc27078035f3ec5446f6 | data/eval/anka_t0115_ilani_2026-09-25.md (bağlam) |

**Sınama kiralamaları temizlendi:** `probe/scripts` acquire→release; probe
`state` yenileme T-0116 kendi kiralaması. Kanonik bus kodu DOKUNULMADI;
commit YOK (operatör onayı ayrıca).