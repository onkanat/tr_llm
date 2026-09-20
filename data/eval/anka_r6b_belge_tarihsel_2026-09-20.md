# T-0083 · Ölü Kristal yollarını TARİHSEL işaretleme — RAPOR

**Tarih:** 20 Eyl 2026 (UTC) · **Görev:** T-0083 · **Yürütücü:** claude (tek yürütücü)
**Not:** `.agent-bus/notes/T-0083.md` · **Karar kaynağı:** T-0082 §3.1 — operatör: *"Tarihsel. seçildi"*

> Bu rapor ölçüm raporudur. Her sayı bu turda koşulan bir komuttan alındı.

---

## 0. Hüküm

| Soru | Ölçülen cevap |
|---|---|
| Ölü bölümler tarihsel işaretlendi mi? | **EVET** — 3 damgalı ekleme (2 belge) |
| Tarihsel içerik silindi mi? | **HAYIR** — `20/0` ve `1/0`; `git diff \| grep '^-'` **boş** |
| Uydurma yol/komut yazıldı mı? | **HAYIR** — güncel komut **doğrulanmadı**, iddia edilmedi |
| Kaynak koddaki kusur (`train.py:149`) | **KAPSAM DIŞI** — yalnız not düşüldü |
| `git add -A` | **kullanılmadı** |

---

## 1. Karar ve neden "canlıya çevirme" seçilmedi

Operatör **tarihsel işaretlemeyi** seçti. Bu, ölçümle de **desteklenen** karardır:
güncel bir komutun çalıştığı **gösterilemez** — `data/anka_a1r.pt` 33.114 satırlıdır ve
`CLAUDE.md` çıkarım betiklerinin 31.357/32.852 boyutlarıyla kurulup **şekil
uyuşmazlığı** verdiğini yazar. Çalıştırılmamış bir komutu belgeye yazmak, bu projede
tekrar tekrar kayda geçen **"ölçmediğin mekanizmayı yazma"** kusuru olurdu.

---

## 2. Yapılan üç ekleme

| # | Belge | Yer | İçerik |
|---|---|---|---|
| **A** | `USER_GUIDE.md` | giriş sonrası | Damgalı TARİHSEL uyarısı: silinme kararı (18 Eyl 2026, operatör) · ölçülen envanter · **etkilenen bölümler §2 · §3 · §4·Adım B1.5** · "güncel karşılık yazılmadı" |
| **B** | `USER_GUIDE.md` | parametre tablosu sonrası | Emekli notu: tablo `train.py`'yi **doğru** anlatır ⇒ kusur **kaynakta**; `--save-path`'i **açıkça ver** |
| **C** | `README.md` | mevcut "Güncel veri hattı" kutusu | Bir madde: üstteki 3 aşamalı blok **tarihsel**; güncel karşılık **doğrulanmadığı için yazılmadı** |

### 2.1 Mekanizma iddiası koddan doğrulandı (uydurulmadı)

`--save-path` verilmezse koşumun **sesli durduğu** iddiası ölçüldü:
`train.py:170` → `check_frozen_save_path(model_save_path, allow_frozen_write=...)`,
çağrı **eğitimden önce** yapılıyor; varsayılan `data/kristal_model.pt` (satır 149)
`data/*.pt` donmuş desenine düştüğü için `--allow-frozen-write` olmadan
`RuntimeError` verir. **Yazılan cümle koddan teyit edildi.**

---

## 3. Tarihsel içerik korundu — kanıt

| Belge | Eklenen | Silinen | `git diff \| grep '^-'` |
|---|---:|---:|---|
| `USER_GUIDE.md` | **20** | **0** | boş |
| `README.md` | **1** | **0** | boş |

Saf ekleme ⇒ **hiçbir tarihsel komut, başlık veya tablo satırı değiştirilmedi/silinmedi.**
Başlık bütünlüğü: `README.md` **11 H2** (beklenen liste eksiksiz).

---

## 4. Ölçüm aracının tuzağı — ÖNGÖRÜLEN ve GERÇEKLEŞEN (kayda geçer)

Düzeltmeden **sonra** ölü atıf sayısı **arttı**: `USER_GUIDE` **7 → 8**, `README` **1 → 2**.
Neden: bir yolu **emekliye ayırırken adını yazmak zorundayım**, ve o ad bir `grep`
sayacında **yeni bir atıf gibi** görünür. Sınıflandırma:

| Belge | Tarihsel KOMUT | Emekli NOTU | Toplam |
|---|---:|---:|---:|
| `USER_GUIDE.md` | **7** (71, 74, 112, 113, 138, 150, 219) | **1** (121) | 8 |
| `README.md` | **1** (210) | **1** (225) | 2 |

⇒ Tarihsel komut sayısı **7 ve 1 olarak DEĞİŞMEDİ**; artış **tam olarak** emekli
notlarıdır (+1/+1).

**Bu, T-0081'de yazılmış bir kuralın tekrarıdır** (*"emekliye ayırırken yazımı ilan
edilen kümeden ayır"*) ve sınıf burada **yeniden üredi** ⇒ kuralı hatırlamak yetmiyor,
**denetim aracı ayıklamalı**. Bir sonraki denetim:
`ölü_atıf = grep(kristal) − emekli_notu_satırları`; aksi halde araç **kendi notunu**
bulgu sanar ([[denetim-kapsami-iddiadan-dar]], [[kabul-kosusu-olctugu-artefakti-degistirir]]).

**Bu rapor bu yüzden hem sayıyı hem sınıflandırmayı veriyor** — çıplak `7`/`8`
karşılaştırması yanıltıcıdır.

---

## 5. Kabul kriterleri

| # | Kriter | Durum | Kanıt |
|---|---|---|---|
| 1 | 8 ölü yol **tarihsel** işaretlendi | **✓** | §2 — 3 damgalı ekleme |
| 2 | Uydurma yol/komut **yazılmadı** | **✓** | §1 — çalışan güncel komut doğrulanmadı |
| 3 | Tarihsel içerik **silinmedi** | **✓** | §3 — `20/0`, `1/0`, silinen satır boş |
| 4 | Canlı iddialara dokunulmadı (kusur **kaynakta**) | **✓** | §2/B + §2.1 |
| 5 | Önce/sonra sha256 | **✓** | §6 |
| 6 | Kapanış: mesaj + kiralar boş + iki yönlü, `len==len(set)` | **✓** | §7 |

---

## 6. Önce / sonra — TAM digest

| Belge | Önce | Sonra | Satır |
|---|---|---|---:|
| `USER_GUIDE.md` | `2886261a2a8e97b575dda904cddd16e2c2d4b78cacce2b07c262359698afcd1f` | **`92fa9f2e027b65ee96237469b62401c75e17f811124e13c77b526d45007a3216`** | 452 → **472** |
| `README.md` | `9ff3266f4c3cda167fa36805406d919aa6b9167141051dc58a6e2be4885faafe` | **`818f59053665b75c500b7a5c4e7a9eb2105d63958fc1aeefaa2094a9b274d068`** | 265 → **266** |

---

## 7. Değişen dosyalar — iki yönlü

| Yol | Durum |
|---|---|
| `USER_GUIDE.md` | **değişti** (+20 / −0) |
| `README.md` | **değişti** (+1 / −0) |
| `.agent-bus/notes/T-0083.md` | yeni |
| `data/eval/anka_r6b_belge_tarihsel_2026-09-20.md` | yeni (bu rapor) |
| **Toplam** | **4** — `len(cf) == len(set(cf))` ⇒ **tekillik GEÇTİ** |

Kapsam dışı **hiçbir** dosyaya yazılmadı: `train.py`, `CHANGELOG.md`, `SPEC.md`,
`CLAUDE.md` **değişmedi**. `CHANGELOG.md` **bilinçli** kapsam dışı: changelog
**tarihsel kayıttır**, o an olanı yazar.

`git add -A` **kullanılmadı.**

---

## 8. Açık kalan işler (kapatılmadı)

1. **`train.py:149`** — kod kusuru; **kendi görevi ve kapısı** olmalı. Bu turda
   **dokunulmadı** (§2.1 yalnız teşhis).
2. **Güncel çalışan komut hâlâ belgesiz** — bu bir eksiklik değil, **ölçülmemiş**
   olmanın sonucudur. `anka_*` checkpoint'lerinin `chat_prompt.py`/`test_model.py`
   ile uyumu ölçülüp belgeye **ondan sonra** yazılmalıdır.
3. **Ölü atıf denetimi** artık **emekli notlarını ayıklamalı** (§4) — yoksa sayı
   kendi notunu bulgu sanar.
4. `notes/` **2 yeni izlenmeyen** dosya taşıyor (`T-0082`, `T-0083`);
   `6bca90b` hâlâ `ahead 1`, push edilmedi.

---

*Raporun bütün sayıları bu turda koşulan komutlardan birebir alınmıştır.
Ölçülmeyen hiçbir mekanizma yazılmadı.*
