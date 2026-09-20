# T-0086 · Ölü atıf denetimi KALICI araç oldu — emekli satırları ayıklanıyor · RAPOR

**Tarih:** 20 Eyl 2026 (UTC) · **Görev:** T-0086 · **Yürütücü:** claude (tek yürütücü)
**Not:** `.agent-bus/notes/T-0086.md` · **Kaynak:** T-0083 raporu §8, açık madde **3**
**Araç:** `scratch/anka_r9_olu_atif_denetimi.py` · **Çıktı:** `data/eval/anka_r9_olu_atif_denetimi_2026-09-20.json`

> Ölçüm raporu. Kapılar **ölçümden ÖNCE** ilan edildi, **değiştirilmedi**.
> Ölçüm sonrası çıkan kusurlar (dördü de **benim**, §7) **rapora yazıldı**.

---

## 0. Hüküm

| Soru | Ölçülen cevap |
|---|---|
| Ölü küme elle mi yazıldı? | **HAYIR** — manifestten literallen okundu, digest **birebir** |
| Araç kendini şişiriyor mu? | **HAYIR** — metninde hiç ölü yol **adı yok** |
| Kapsam sessizce daralıyor mu? | **HAYIR** — dış `grep` kullanılmaz, ağaç **kendisi** yürünür |
| Emekli ayıklama kuralı **sınandı** mı? | **EVET** — kanarya birebir: `4/1/3` ve `3/2/1` |
| Aşırı ayıklama var mı? | **HAYIR** — negatif dal `olu == ham == 11` |
| Fail-closed dalı **öldü** mü? | **HAYIR** — **8/8** dal durdu, pozitif dal durmadı |
| Gerçek ağaçta ayıklanan satır | **0** (§5 — kural canlı ama işi yok) |
| `git add -A` | **kullanılmadı** |

---

## 1. İlan edilen kapılar ve sonuçları

| Kapı | Eşik (ilan edilen) | Sonuç | Kanıt |
|---|---|---|---|
| **D1** | ölü küme manifestten; digest **birebir** | **✓ GEÇTİ** | `4e1c4d51a896936909b1750b52088e8939e03e966b42fd8ea84934e0993e6e61` |
| **D2** | kapsamlar ayrı; **hiçbir sayı kapsamsız** basılmaz | **✓ GEÇTİ** | §4 — beş kapsam etiketli + ayrıca `KANARYA ÇIKARILINCA` satırı |
| **D3** | pozitif **ve** negatif kontrol, ikisi de **vakum değil** | **✓ GEÇTİ** | §3 — kanarya birebir, negatif `olu==ham` |
| **D4** | okunamayan sinyalde **durur** (`rc=2`) | **✓ GEÇTİ** | §6 — **8/8** fail-closed dalı |
| **D5** | `olu = ham − emekli`, negatif sayı yok, tutarlılık | **✓ GEÇTİ** | §4 — `hatalar: []` |
| **D6** | `git add -A` kullanılmadı | **✓ GEÇTİ** | §9 |

**Çürütme 1 (önceden ilan edildi):** *"Emekli ayıklaması canlı bir atfı da silerse araç
KUSURLUDUR."* — **Gerçekleşmedi**: `chat_prompt.py` **11 canlı atfın tamamını** korudu
(`olu == ham == 11`).

**Çürütme 2 (önceden ilan edildi):** *"Etiket kalıbı emekli OLMAYAN bir yorum bloğuna
uyarsa ayıklama YANLIŞ olur."* — **Gerçekleşmedi**: kanarya fikstürü **yakın-kaçıran**
bir blok taşır (`# EMEKLI DEGIL (T-0086) …`); kalıp onu **reddetti** ve o atıf `CANLI`
sayıldı. Bunu kanaryanın `emekli=1` çıkması **kanıtlar** — blokta 2 eşleşme vardı,
yalnız 1'i ayıklandı.

---

## 2. Kanonik kaynak ve BİRİM

`data/eval/t0069_pre_delete_manifest.json` → `hedefler[51]`, digest kayıttaki
`manifest_sha256` ile **birebir**. 51 yol → **46 benzersiz taban ad**.

**BİRİM = TABAN AD**, tam yol DEĞİL. Gerekçe (ölçüldü): tam-yol birimi, dizin öneki
olmayan atıfları **kaçırır**. `chat_prompt.py` tam-yol birimiyle **10**, taban-ad
birimiyle **11** verdi ⇒ aradaki 1 atıf tam-yol biriminde **görünmezdi**. Taban adların
hepsi ayırt edicidir (`kristal_*`, `t0046_*`, `t0053_*`), çakışma riski yok; 5 ad iki
yol tarafından paylaşılır (51→46) ve **doğru** davranış budur: aynı ada yapılan atıf
**tek** atıftır.

> Bu bir **tanım** değişikliğidir ve **hiçbir ölçüm kabul edilmeden önce** yapıldı;
> ilan edilmiş kapı değiştirilmedi ([[iki-sayi-celisiyor-sanma-once-kume]]: eksen
> **tanım ve ADI**).

---

## 3. Kontroller (D3) — pozitif ve negatif

### 3.1 ÖLÇÜLEN SORUN: ağaçta doğal pozitif kontrol YOK

Tüm ağaç tarandı (1138 dosya): **"etiketli blok İÇİNDE ölü yol anan" HİÇBİR dosya
yoktur (0 bulgu)**. Yani ayıklama kuralının doğal kontrolü yoktu. Kural sınanmadan
bırakılsaydı "kapı geçti" demek **VAKUM GEÇİŞ** olurdu
([[vakum-kanarya-satir-numarasi]]). Bu yüzden **ilan edilmiş kanarya fikstürü** kuruldu
ve **ayrı kapsamda** (`KANARYA`) raporlanıyor; gerçek ağaç toplamlarına karışmıyor.

### 3.2 Kanarya fikstürleri — beklenen ve ölçülen

| Fikstür | İçerik | Beklenen | Ölçülen |
|---|---|---|---|
| `scratch/anka_r9_kanarya.py` | etiketli blok (ayıklanır) + **yakın-kaçıran** blok + 2 blok-dışı kod satırı | `4 / 1 / 3` | **`4 / 1 / 3`** ✓ |
| `scratch/anka_r9_kanarya.md` | `TARİHSEL` etiketli `>` bloğu (ayıklanır) + blok-dışı satır | `3 / 2 / 1` | **`3 / 2 / 1`** ✓ |

**NEGATİF DAL:** `chat_prompt.py` → `ham=11`, `olu=11` ⇒ **aşırı ayıklama yok**.
Kontrol **vakum değil** (`ham > 0`).

---

## 4. Ölçülen sayılar — KAPSAMIYLA

```
KAPSAM     dosya    ham  emekli    ÖLÜ   neyi kapsar
KOD           43    171       0    171   yürütülebilir .py (scratch hariç)
SCRATCH       85    406       0    406   gitignore'lu test yüzeyi (kanarya hariç)
KANARYA        2      7       3      4   İLAN EDİLMİŞ kontrol fikstürü — gerçek atıf DEĞİL
BELGE         98    193       0    193   kök/belge .md-.txt
KAYIT        101   1529       0   1529   data/eval/** + .agent-bus/{notes,tasks}/**
TOPLAM       329   2306       3   2303   (tüm kapsamlar)
— KANARYA ÇIKARILINCA                      2299   gerçek ağaç
```

**En çok anılan 6 ölü ad:** `kristal_model.pt` **656** · `kristal_carpenter_model.pt`
**302** · `kristal_model_f3_clean.pt` **134** · `kristal_model_sft.pt` **114** ·
`kristal_model_f4_r05.pt` **101** · `kristal_model_pre_clean.pt` **76**.

> ⚠️ **Bu sayılar bir "kusur sayısı" DEĞİLDİR.** `KAYIT` (1529) tarihsel ölçüm
> kayıtlarıdır; **değiştirilmeleri kanıtı tahrif eder**. `KOD`+`SCRATCH` (171+406=577)
> yürütülebilir yüzeydir. Tek bir sayı yazmak — hangi kapsam olduğu söylenmeden —
> [[denetim-kapsami-iddiadan-dar]] olurdu.

---

## 5. BULGU: kural canlı ama gerçek ağaçta **işi yok**

Gerçek ağaçta **2306** eşleşmenin **hiçbiri** etiketli blok içinde değil ⇒ ayıklanan
satır **0**. Yani T-0083'ün şikâyeti (*"emekli notu sayacı şişirir"*) **DEKLARE EDİLEN
işaretleme sözlüğü altında bugün yeniden üretilemiyor**:

- `EMEKL[İI]\s*\(T-\d+\)` kalıbı ağaçta **yalnız 3 yerde** geçiyor; `train.py:149`'daki
  blok ölü yolu **adıyla ANMIYOR** (T-0085'te bilinçli olarak böyle yazıldı) ⇒
  ayıklanacak satırı yok.
- `TAR[İI]HSEL` blokları (`USER_GUIDE.md:6`) ölü yol **anmıyor**.

**Beyan edilen sınırlılık:** kural **kesindir, bulanık değildir**. Emekli bir not, ölü
yolu **başka bir ifadeyle** anarsa (ör. etiketsiz bir cümleyle) araç onu **ayıklamaz**
ve sayı şişer. Kuralı "yakınında emekli kelimesi geçiyorsa" diye **gevşetmek
reddedildi**: ölçüt keyfî olur ve canlı atıfları gizleme riski doğar. Bu, aracın
**ilan edilmiş** sınırıdır; bir sonraki tur **yeni bir etiket** eklerse kurala da
eklenmelidir.

---

## 6. D4 — fail-closed dalı ÖLMEDİ (8/8)

Kapılar yalnız "geçti" demekle kalmamalı; **düştüğünde de durduğu** gösterilmelidir
([[on-kontrol-fail-closed-olmali]]). Modül **import edilerek** (kopyalanmadan) sınandı:

| Dal | Beklenen | Sonuç |
|---|---|---|
| A) manifest dosyası **yok** | dur, `rc=2` | **✓** `kanonik kaynak yok` |
| B) manifest **bozuk** (digest kayar) | dur | **✓** `manifest digest uyuşmuyor` |
| C1) `hedefler` **yok** | dur | **✓** `'hedefler' listesi yok/boş` |
| C2) `hedefler` **boş** | dur | **✓** aynı |
| C3) hedefte `yol` alanı **yok** | dur | **✓** `hedefte 'yol' alanı yok` |
| C4) `n_hedef` **uyuşmuyor** | dur | **✓** `n_hedef=1 != okunan 2` |
| C5) kayıtta `manifest_sha256` **yok** | dur | **✓** `'manifest_sha256' alanı yok` |
| D) **pozitif dal** — düzeltilmiş hâl | **DURMAMALI** | **✓** durmadı, 51 yol okundu |

**Yapı bulgusu (ölçüldü, tasarım gereği):** digest kapısı şema kapısının **ÜSTÜNDEDİR**
⇒ **tek** bir kaynağı bozmak şema kapısına **asla ulaşamaz** (B dalı C1'i gölgeler).
C1–C5'i sınamak için **iki kaynağın BİRLİKTE** bozulması gerekir. İyi bir özelliktir:
iki kaynak **uzlaşmadıkça** araç ilerlemez.

---

## 7. Bu turda ölçülen KENDİ kusurlarım (dördü de kayda geçer)

| # | Kusur | Nasıl yakalandı / düzeltildi |
|---|---|---|
| 1 | **Pozitif kontrolü, atfını BİR TUR ÖNCE kaldırdığım dosyaya koydum** (`train.py`) ⇒ kontrol **VAKUM** | Aracın kendi fail-closed kapısı `rc=2` verdi; kontrol **kanarya fikstürüne** taşındı |
| 2 | **BİRİM yanlıştı:** tam yol (51) ⇒ dizin-öneksiz atıflar kaçar (`chat_prompt.py` 10 vs 11) | Taban ada (46) çevrildi; **hiçbir ölçüm kabul edilmeden önce** |
| 3 | **Kanaryanın ilk sürümünde yakın-kaçıran satırı ETİKETLİ BLOĞUN İÇİNE koydum** ⇒ blok kuralı onu da ayıklar, kalıp sınaması **boşa giderdi** | Bloklar boş satırla **ayrıldı**; kanarya `4/1/3` ile kalıbın **reddettiğini** kanıtlar |
| 4 | **İki sondamın BEKLENTİSİ yanlıştı** (C ve C5) — araç doğru davrandı, benim kurgum eksikti | Düzeltilip yeniden koşuldu; §6'nın **yapı bulgusu** tam bu hatadan çıktı |

**Ortak ders:** dördü de *aracın* değil **tasarımın/beklentimin** kusuruydu — ve
**hiçbiri sessizce geçmedi**, çünkü kapılar fail-closed yazılmıştı. Bu, T-0075'in
"okunamayan sinyalde durmayan koruma kapı değildir" kuralının karşılığını gösterir.

---

## 8. İlan edilmiş kapsamdan sapma (beyan edilir)

Görev `writes[]` listesinde **iki kanarya fikstürü YOKTU**. Tasarım ölçümden sonra
netleşti: ağaçta doğal pozitif kontrol bulunmadığı (§3.1) için fikstür zorunlu hale
geldi. `bus_post_task`'ın **güncelleme karşılığı yoktur** ve `writes[]` değiştirilemez.

**Yapılan:** iki dosya için **ayrıca kiralama alındı** (SPEC kural 1'in kiralama
koşulu karşılandı), bu raporda ve `changed_files`'ta **açıkça beyan edildi**; kapı
sessizce gevşetilmedi.

---

## 9. Değişen dosyalar — iki yönlü

| Yol | Durum | sha256 (kısa) |
|---|---|---|
| `scratch/anka_r9_olu_atif_denetimi.py` | yeni (asıl araç, `rc=0`) | `0223938a7e6e669a…` |
| `scratch/anka_r9_kanarya.py` | yeni (ilan edilmiş fikstür) | `935b83fccb72f3f7…` |
| `scratch/anka_r9_kanarya.md` | yeni (ilan edilmiş fikstür) | `989977987be9b8e9…` |
| `data/eval/anka_r9_olu_atif_denetimi_2026-09-20.json` | yeni (makine okunur çıktı) | `e73a27b4a7162da4…` |
| `data/eval/anka_r9_olu_atif_denetimi_2026-09-20.md` | yeni (bu rapor) | — |
| `.agent-bus/notes/T-0086.md` | yeni | — |
| **Toplam** | **6** — `len(cf) == len(set(cf))` ⇒ **tekillik GEÇTİ** | |

**KOD DEĞİŞMEDİ:** `train.py`, `chat_prompt.py`, `test_model.py`, `src/**`,
`scripts/**`, `tests/**` — **hiçbiri**. Bu görev **yalnız bir ölçüm aracıdır**;
bulunan ölü atıflar **düzeltilmedi, sayıldı ve sınıflandırıldı**. `git add -A`
**kullanılmadı**; commit/push **operatör yetkisinde**.

---

## 10. Açık kalan işler (kapatılmadı)

1. **Etiket sözlüğü genişletilmeli.** Araç, emekli bir notu **yalnız** ilan edilen iki
   kalıpla tanır (§5). Yeni bir işaretleme deyimi doğarsa kurala eklenmelidir; aksi
   halde sayı sessizce şişer — aracın **ilan edilmiş** sınırı budur.
2. **`KOD` kapsamındaki 171 atıf düzeltilmedi.** LIVE/TARİHSEL ayrımı T-0085 §3'te
   yapıldı; `src/**` ve `chat_prompt.py` **KOD** değişikliği ister (ayrı görev + kapı).
3. **`data/eval/**` (`KAYIT`) tarihsel kayıttır** ve **değiştirilmemelidir**; 1529
   eşleşmenin tamamı oradadır. Bir sonraki tur bunları "kusur" sayıp düzeltmeye
   kalkışmamalıdır.
4. **Araç kendi çıktısını bir sonraki koşumda tarayacaktır** (bu rapor ölü adlar
   anar). Kapsam ayrımı (`KAYIT`) bunu görünür kılar ama **sıfırlamaz**; sayılar
   koşumdan koşuma **artabilir**. Bu, aracın **kendine gönderim** sınırıdır.

---

*Raporun bütün sayıları bu turda koşulan komutlardan alınmıştır. `hatalar: []` —
araç hiçbir kapıyı sessizce geçirmedi; ölçülen kendi kusurlarım (§7) dahil.*
