# P26 — Uzak `simulasyon_bellek` Tohumlama Raporu (export → Qdrant geri-yükleme)

**Damga:** 2026-11-04T18:38:21+03:00 — doğrulama sonu
**Hedef:** `http://192.168.1.9:6333` (uzak Qdrant 1.19.1, LAN)
**Kaynak:** `data/simulasyon_bellek_export.jsonl` (44.691.411 bayt, 10 Eyl 2026 13:55; 4.073 satır)
**Betik:** `scripts/seed_simulasyon_bellek.py` (241 satır; sha256 `c9304ae3485606ec6b5a1ca566e41eed84ff0bc6776327c61f9a97e12f25c2ac`)

---

## 1. İLAN

**Ne yapılacaktı:** Uzak Qdrant'taki **BOŞ** `simulasyon_bellek` koleksiyonu, 10 Eylül 2026
tarihli export yedeğinden (4.073 nokta) **idempotent** eken betik ile tohumlanacaktı.
Bu bir **geri-yükleme**dir: gömme (re-embed) YOK — export'taki HAZIR dense/sparse
vektörleri, payload'lar ve nokta kimlikleri AYNEN yazılır.

**Kapsam ve sınırlar (onaylı):**
- Kullanıcı onayı: yalnız BOŞ `simulasyon_bellek` koleksiyonunun tohumlanması.
- **SERT KURAL — yazma yetkisi YALNIZ `simulasyon_bellek`.** `anka_bellek` ve
  `muhakeme_bellek` DOKUNULMAZ (okuma serbest). Betikte koleksiyon adı SABİT
  (`--collection` arg'ı YOK) → kural betik düzeyinde yapısal garanti.
- **Git commit YOK** (tüm çıktılar yalnız diskte: 1 betik + 1 rapor).

**Yöntem (şartname):**
1. Export şeması ölçülecek (kayıt sayısı, alanlar, id'ler, vektör boyutu).
2. Eken betik yazılacak — `seed_muhakeme_bellek.py` kalıbı: upsert idempotent,
   3 denemeli bağlantı, `--timeout 30`, `--dry-run`, `--limit`; nokta kimliği
   export'taki `id`'nin AYNISI (yoksa tam-kayıt sha256, P2.1); koleksiyon şeması
   orijinal `simulasyon_bellek` şeması (kalıp: `sync_and_backup_simulasyon_bellek.py`).
3. Canlı koşum: `QDRANT_URL=http://192.168.1.9:6333`.
4. Doğrulama: read-back nokta sayısı ≥ export kayıt sayısı; 3 örnek kayıt geri oku;
   tekrar koşum idempotens (sayı aynı); `anka_bellek`(46) ve `muhakeme_bellek`(91)
   sayıları DEĞİŞMEDİ.

---

## 2. HÜKÜM

**P26_SIMULASYON_TOHUM_GECTI** — 4.073/4.073 nokta tohumlandı, read-back birebir,
idempotens kanıtlandı, dokunulmaz koleksiyonlar değişmedi.

| Kapı | Ölçülen | Hüküm |
|---|---|---|
| ŞEMA UYUŞUMU (export ↔ uzak koleksiyon) | export: dense 768 (4.073/4.073) + sparse (indices/values); uzak: `dense` 768 Cosine + `sparse` IDF, `on_disk_payload: true` — `sync_and_backup_simulasyon_bellek.py` (satır 96–106) orijinal şemasıyla birebir; anka_bellek ile aynı dense boyutu/ölçümü | GEÇTİ |
| NOKTA KİMLİĞİ | 4.073/4.073 kayıtta `id` VAR (int, 1..4073 ardisik, benzersiz) → kimlikler AYNEN korundu; `id` yoksa tam-kayıt sha256 % 2^63 (P2.1) yedek yol betikte | GEÇTİ |
| TOHUMLAMA (canlı) | ekim-öncesi 0 → **4.073 nokta** (2 koşumda: smoke 64 + tam 4.073; tam koşum 10,5 s) | GEÇTİ |
| READ-BACK SAYI (bağımsız curl) | `simulasyon_bellek`: **4073** ≥ export 4.073 | GEÇTİ |
| READ-BACK ÖRNEK (3 kayıt) | id=1, id=2037, id=4073: payload **birebir** (betik karşılaştırması 3/3 `EVET` + bağımsız curl text/crystal_tags eşleşmesi) | GEÇTİ |
| İDEMPOTENS (tekrar koşum) | 4.073 → upsert 4.073 → **4.073** (değişim YOK; 9,5 s) | GEÇTİ |
| DOKUNULMAZLAR | `anka_bellek`: 46 → 46; `muhakeme_bellek`: 91 → 91 (ekim öncesi/sonrası betik ölçümü + bağımsız curl; değişim YOK) | GEÇTİ |
| SERT KURAL / COMMIT | yazma yalnız `simulasyon_bellek` (betikte sabit; dokunulmazlar salt-okunur sayaçlarla izlendi); `git commit` YAPILMADI (working tree'de yalnız 2 yeni dosya) | GEÇTİ |

---

## 3. RAPOR

### 3.1 Kaynak şema incelemesi (ölçüldü)

`data/simulasyon_bellek_export.jsonl` — sha256
`f823f8b399aab3aa3531fdfef5af1a0828be6d79572ce0d7895a588d1626f792`:

| Ölçüm | Değer |
|---|---|
| Kayıt sayısı | 4.073 (JSON hatası 0) |
| `id` alanı | 4.073/4.073 var; hepsi int; 1..4073; ardisık; benzersiz |
| `vectors.dense` boyutu | 768 (4.073/4.073) |
| `vectors.sparse` | 4.073/4.073 düzgün `{indices, values}`; nnz 3..253 |
| `payload.text` | 4.073/4.073; uzunluk 9..3.625 (ort 76,3) |
| `payload` biçimleri | 4.000 kayıt `gts_sozluk` (`domain/kok/yazar/crystal_tags/token_ids/method/text`); 73 kayıt proje dokümanı (`source/v='v5_project_doc'/crystal_tags/token_ids/text`) |
| Uzak koleksiyon (öncesi) | VAR, `green`, **0 nokta**; şema: `dense` 768 Cosine + `sparse` IDF + `on_disk_payload: true` |

### 3.2 Eken betik — `scripts/seed_simulasyon_bellek.py`

- **Kalıp:** `seed_muhakeme_bellek.py` (argparse sözleşmesi, 3 denemeli bağlantı 5 s
  aralıkla, read-back doğrulaması, çıkış kodu kapıları).
- **Fark:** gömme YOK — export'taki hazır vektörler/payloadlar AYNEN upsert
  (`VectorMemory.add_documents_batch(upsert=True, point_ids=...)`, parça 64;
  `texts[i] = payload["text"]` → payload birebir korunur).
- **Nokta kimliği:** kayıttaki `id` AYNISI; `id` yoksa tam-kayıt kanonik-JSON
  sha256 % 2^63 (P2.1 dersleri — birebir aynı satır yeniden eklenirse AYNI nokta).
- **Koleksiyon şeması:** VectorMemory hibrit şeması = orijinal şema (dense 768
  Cosine + sparse IDF); koleksiyon sunucuda zaten VAR olduğundan yaratma/yıkım
  YAPILMADI, yalnız 768 boyut uyuşumu fail-closed kontrol edildi.
- **Argümanlar:** `--url` > `QDRANT_URL` env > `localhost:6333`; `--timeout 30`
  (varsayılan); `--dry-run` (bağlantısız sayım); `--limit N` (smoke/probe);
  `--export`.
- **Sert kural:** `KOLEKSIYON = "simulasyon_bellek"` sabit; `anka_bellek` +
  `muhakeme_bellek` yalnız salt-okuma sayaçlarında (ekim öncesi/sonrası kanıtı).
- sha256: `c9304ae3485606ec6b5a1ca566e41eed84ff0bc6776327c61f9a97e12f25c2ac`

### 3.3 Canlı koşum günlüğü (kanıt çıktıları)

Koşum ortamı: `venv/bin/python` (3.14, `qdrant_client` kurulu), cwd repo kökü.

**(a) Dry-run** (bağlantı YOK):
```
[EKIM] export kaydı: 4073
[EKIM] benzersiz nokta kimliği: 4073/4073
[EKIM] --dry-run: bağlantı KURULMADI, yazma YOK
  örnek[0] id=1 dense=768 sparse_nnz=21 domain='gts_sozluk' ...
```

**(b) Smoke — `--limit 64`** (ilk koşum): `0 → 64` nokta (0,4 s).
**(c) Smoke tekrar** (idempotens + read-back yolu): `64 → 64`; 3/3 payload birebir.

**(d) TAM KOŞUM** (4.073 kayıt):
```
[EKIM] ekim-ÖNCESİ simulasyon_bellek: 64 nokta
[EKIM] ekim-ÖNCESİ dokunulmaz: {'anka_bellek': 46, 'muhakeme_bellek': 91}
  [EKIM] upsert 4073/4073 (10.5s)
[EKIM] ekim-SONRASI nokta sayısı: 4073 (öncesi: 64, 10.5s)
  [KANIT] id=1 payload birebir=EVET domain='gts_sozluk' text='Demirkır, güney tepelerinin duldalarına çektiği atları gece ...'
  [KANIT] id=2037 payload birebir=EVET domain='gts_sozluk' text='Evin bütün pencereleri çift kanatlı, her kanat, kare biçimli ...'
  [KANIT] id=4073 payload birebir=EVET domain='v5_project_doc' text='4. Başarı Kriterleri Gated Checkpoints 1. Loss Stabilitesi : ...'
[EKIM] ekim-SONRASI dokunulmaz: {'anka_bellek': 46, 'muhakeme_bellek': 91} (değişim YOK)
[EKIM] TAMAM: 4073 kayıt upsert (10.5s), koleksiyonda 4073 nokta (≥4073 doğrulandı), 3/3 örnek payload birebir, dokunulmaz koleksiyonlar değişmedi
```

**(e) İDEMPOTENS — tam koşum tekrarı:**
```
[EKIM] ekim-ÖNCESİ simulasyon_bellek: 4073 nokta
[EKIM] ekim-SONRASI nokta sayısı: 4073 (öncesi: 4073, 9.5s)   → ÇOĞALTMA YOK
```

**(f) Bağımsız doğrulama (curl, betik dışı):**
```
simulasyon_bellek: {"result":{"count":4073}}
anka_bellek:       {"result":{"count":46}}
muhakeme_bellek:   {"result":{"count":91}}
```

### 3.4 Karşılaştırma tablosu (ekim öncesi → sonrası; bağımsız curl)

| Koleksiyon | Öncesi | Sonrası | Değişim |
|---|---|---|---|
| `simulasyon_bellek` (onaylı hedef) | 0 | **4.073** | +4.073 (onay kapsamında) |
| `anka_bellek` (dokunulmaz) | 46 | 46 | YOK |
| `muhakeme_bellek` (dokunulmaz) | 91 | 91 | YOK |

### 3.5 Notlar / düzeltmeler

- İlk smoke koşumunda read-back döngüsünde açma (unpack) hatası vardı
  (`zip` düzeyinde fazladan değişken) — upsert ZARAR GÖRMEDİ (64 nokta doğru
  yazılmıştı), satır düzeltildi (`for idx, rec in zip(...)`), koşumlar tekrarlandı.
- Betik çalışma hatası Pyright'ın venv'i görmemesinden kaynaklanan yanlış-pozitif
  `reportMissingImports` uyarıları üretir (sistem-Python çözümlemesi); koşum
  `venv/bin/python` ile temiz.
- Git durum: **commit YAPILMADI**; `scripts/seed_simulasyon_bellek.py` ve bu rapor
  working tree'de yeni (untracked) dosyalar olarak bırakıldı.
- Uzak `simulasyon_bellek` artık 10 Eylül 2026 yedeğiyle (4.073 nokta) tutarlı;
  orijinal id/payload/vektörler AYNEN korunduğundan sonraki senkronlar
  `sync_and_backup_simulasyon_bellek.py` ile fark göstermez.