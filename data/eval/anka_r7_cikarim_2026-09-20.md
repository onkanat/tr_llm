# T-0084 · Anka A1-r için ölçülmüş çıkarım komutu — RAPOR

**Tarih:** 20 Eyl 2026 (UTC) · **Görev:** T-0084 · **Yürütücü:** claude (tek yürütücü)
**Not:** `.agent-bus/notes/T-0084.md` · **Kaynak:** T-0083 raporu §8, açık madde **2**

> Bu rapor ölçüm raporudur. **Belgeye yazılan her komut bu oturumda koşuldu.**
> Ölçülmemiş hiçbir komut yazılmadı.

---

## 0. Hüküm

| Soru | Ölçülen cevap |
|---|---|
| Belgedeki komut (`data/vocab.json`) çalışıyor mu? | **HAYIR** — `RuntimeError`, boyut uyuşmazlığı |
| Eşleşen sözlükle `anka_a1r.pt` yükleniyor mu? | **EVET** — `strict=True` **TEMİZ** |
| Anahtar (maske/rope) atmak gerekiyor mu? | **HAYIR** — 101/101 anahtar, hiçbiri maske/rope değil |
| Çıktı dejenere mi? | **HAYIR** — ama `<EOS>` kırpılmazsa **yanlış okunur** (§3) |
| Talimat takip ediyor mu? | **HAYIR** — `<OUTPUT>` oranı **2,0 × 10⁻⁸** (§4) |
| Güncel komut belgeye yazıldı mı? | **EVET** — yalnız ölçülen komut |
| `git add -A` | **kullanılmadı** |

---

## 1. İlan edilen kapılar ve sonuçları

Kapılar **ölçümden önce** yazıldı (görev şartnamesi), sonra **değiştirilmedi**.

| Kapı | Eşik | Sonuç | Kanıt |
|---|---|---|---|
| **GA** | DAL A `RuntimeError` ile **sesli durur** | **✓ GEÇTİ** | §2/A — `size mismatch … [33114,768] … [31357,768]` |
| **GB** | DAL B `strict=True` ile **temiz** yüklenir; anahtar atma **zorunlu olmamalı** | **✓ GEÇTİ** | §2/B — 101 anahtar, `strict=True` hatasız |
| **GC** | `<EOS>` kırpılmışta dağılım **tek tokenda p>0,99'a çökmemeli** | **✓ GEÇTİ** | §3 — en yüksek top-1 **0,1821** |
| **GD** | Belgeye yazılan her komut koşulur, çıktısı rapora yapıştırılır | **✓ GEÇTİ** | §2, §3, §4 |
| **GE** | `git add -A` kullanılmaz | **✓ GEÇTİ** | §7 |

**Çürütme maddesi (önceden ilan edildi):** *"Eşleşen sözlük dalı DA hata verirse
geçerli bir komut YOKTUR ve belge bunu böyle söylemek zorundadır."* — **Gerçekleşmedi**:
DAL B temiz yüklendi, dolayısıyla geçerli bir komut **vardır** ve yazıldı.

**Sonda çıkış kodu:** `rc=0` (kapılar geçti). `rc=2` ile duran yol kuruludur ve
kapılardan biri düşseydi belgeye **komut yazılmayacaktı**.

---

## 2. Ölçüm — iki dal

Sonda: **`scratch/anka_r7_cikarim_sondasi.py`** (gitignore'lu; `py_compile` temiz).
Zorunlu pozitif kontrol: tokenizer `len(encode(...)) < 3` verirse `SystemExit`
(lexicon yüklenmezse **her girdi sessizce boş döner** — T-0078'in ölüm sebebi).

### A · Belgedeki gibi — `data/vocab.json` (31.357)

```
[DAL A] belgedeki gibi: data/vocab.json
    len(stoi) = 31357
    (not: 101 anahtarın hiçbiri maske/rope değildi — atma gerekmedi)
    ✓ BEKLENDİĞİ GİBİ RuntimeError: Error(s) in loading state_dict for KristalLM:
      size mismatch for embedding.embedding.weight: copying a param with shape torch.Size([33114, 768]) from checkpoint, the shape in current model is torch.Size([31357, 768]).
      size mismatch for lm_head.weight: copying a param with shape torch.Size([33114, 768]) from checkpoint, the shape in current model is torch.Size([31357, 768]).
```

⇒ **Sesli durur, sessiz kırpma yok.** `strict=False` bu sınıfı **yutmaz**
(yalnız eksik/fazla *anahtarı* yutar; *şekli* değil).

### B · Eşleşen sözlük — `vocab_anka_r1_33114.json` + `roots_anka_r1.tsv`

```
[DAL B] eşleşen sözlük: data/rebuild/vocab_anka_r1_33114.json
    len(stoi) = 33114
    (not: 101 anahtarın hiçbiri maske/rope değildi — atma gerekmedi)
    ✓ load_state_dict(strict=True) TEMİZ
```

⇒ `test_model.py`'nin `cos_cached`/`sin_cached`/`mask` **silme** mantığı bu
checkpoint için **gereksizdir** (zararsız, ama gerekli sanılmamalıdır).

### Sözlük boyutları — kanonik kaynaktan

| Dosya | `len(Vocabulary.load().stoi)` | `wc -l` |
|---|---:|---:|
| `data/vocab.json` | **31.357** | 31.361 |
| `data/vocab_entity.json` | **32.852** | 32.856 |
| `data/rebuild/vocab_base_32852.json` | **32.852** | 32.856 |
| `data/rebuild/vocab_anka_r1_33114.json` | **33.114** | 33.118 |

> ⚠️ **`wc -l` bu dosyalar için +4 fazla gösterir.** Sayı `stoi`'dan okunmalıdır;
> aksi halde "31.361" gibi hiçbir kapıda geçmeyen bir sayı raporlanır.

---

## 3. KENDİ KUSURUM — ölçü aracım iddiadan FARKLI bir soru sordu

**İlk sondam dejenere çıktı verdi:** `Yarın okula` sonrası `<BOS>` = **0,9971**.
"Model çökmüş" diye okumak üzereydim. Nedenini ölçtüm:

```
id 2 = '<BOS>' | id 3 = '<EOS>' | id 4 = '<PROPER_NOUN>'
encode('Yarın okula') -> [2, 741, 784, 20, 3]   son_id_decode='<EOS>'
```

`encode()` dizinin **sonuna `<EOS>` ekliyor**. Yani modele verdiğim girdi
`<BOS> Yarın okula <EOS>` idi ve sorduğum soru *"belge bittikten sonra ne gelir?"*
oldu — model **doğru** cevap verdi: yeni belge `<BOS>` ile başlar. **0,9971 bir
kusur değil, doğru davranıştı.**

**Düzeltilmiş ölçü** (`[:-1]` ile `<EOS>` kırpılır):

| Girdi (gövde) | Kırpılan son token | İlk 5 devam |
|---|---|---|
| `Yarın okula` | `CASE_DAT` | `<PROPER_NOUN>`=0,0883 · `ait`=0,0696 · `(`=0,0638 · `bağ`=0,0619 · `,`=0,0510 |
| `Akmayan su kımıldanmayan yer` | `yer` | `CASE_LOC`=0,1642 · `DERIV_lI`=0,1606 · `PLURAL`=0,1274 · `COPULA_AORIST`=0,0774 · `al`=0,0599 |
| `Demirkır güney tepelerinin duldalarına` | `CASE_DAT_N` | `bağ`=0,1821 · `ait`=0,1640 · `göre`=0,1411 · `doğru`=0,0265 · `sahip`=0,0246 |

**Kontrast kanıtı (tuzağın belgesi):**
`Yarın okula` **+ `<EOS>`** ⇒ `<BOS>`=**0,9971** · `DERIV_CI`=0,0011 · `ver`=0,0007

⇒ **Sınıf:** [[denetim-kapsami-iddiadan-dar]] — sayı doğru, **sorulan soru yanlış**.
Araç, iddianın kapsamıyla hizalanmamıştı. Aynı sınıfın **beşinci** örneği.

**Bunun belgeye etkisi:** tuzak **belgeye yazıldı** — çünkü bir sonraki okuyucu
aynı hatayı yapıp modeli "bozuk" sanacaktı. `<EOS>` kırpma adımı komutun parçasıdır.

---

## 4. Bu bir TABAN modeldir — talimat takip etmez (ölçüldü)

| Külliyat | Jeton | `<PROPER_NOUN>` | oran | `<OUTPUT>` | oran |
|---|---:|---:|---:|---:|---:|
| `data/anka_a1_pretrain.bin` | 100.000.000 | 7.936.521 | %7,937 | **2** | **2,0 × 10⁻⁸** |
| `data/anka_a1r_pretrain.bin` | 100.000.000 | 7.136.316 | **%7,136** | **2** | **2,0 × 10⁻⁸** |

⇒ A1-r **düz-metin ön-eğitimidir** (`--pretrain`); `<INSTRUCTION> … <OUTPUT>` zarfı
dağılımda **yoktur**. Zarfı verip cevap beklemek **dağılım dışı** bir istektir.
Bu checkpoint **metni sürdürür**, **soru cevaplamaz**. Belgeye bu **yazıldı** —
aksi halde okuyucu `chat_prompt.py`'ı bu checkpoint'le çalıştırıp "model bozuk" sanır.

*Not:* `<PROPER_NOUN>` %7,937 → **%7,136** (D2'nin ölçülen kazancı); G3 eşiği
(%3,0) **hâlâ geçilmedi** — bu, T-0080'in "5/5 KALDI" hükmüyle **tutarlıdır**.

---

## 5. Belgeye yapılan ekleme

`USER_GUIDE.md` — **yeni H2 bölümü** (`## 🅰️ Anka A1-r — ÖLÇÜLMÜŞ çıkarım komutu`),
TARİHSEL banner'ın **hemen ardına**, İçindekiler'in **öncesine**; ayrıca İçindekiler'e
**`0.`** maddesi (mevcut 1–9 numaraları **kaymadı**).

İçerik: gereken üç dosya + **tam** sha256 · komut · **birebir ölçülmüş çıktı** ·
iki tuzak (`<EOS>` kırpma; taban model ⇒ talimat takibi yok) · bugün çalışmayan
yollar tablosu (`test_model.py`, `chat_prompt.py`).

### 5.1 TARİHSEL banner'a DOKUNULMADI — ve nedeni

Görev şartnamesinde **"mevcut TARİHSEL bloğu SİLME/DEĞİŞTİRME"** diye **ilan ettim**.
Ama banner'ın *"çalışan bir güncel komut bu belgede doğrulanmadığı için iddia
edilmiyor"* cümlesi, artık **kısmen** geçersizdir. İlan edilmiş kapıyı **sessizce
değiştirmedim**; bunun yerine yeni bölümün açılışına **açık bir atıf** koydum:
*"banner metni kendi damgasıyla değiştirilmeden korunmuştur."*

**Kalan gerilim (açık iş):** banner cümlesi bir sonraki turda **yeniden yazılmalı**
(*"§2/§3/§4 için geçerli; Anka A1-r için ölçülmüş komut aşağıdadır"*). Kapıyı
kendim gevşetmek yerine **beyan ettim**.

---

## 6. Önce / sonra — TAM digest

| Artefakt | Önce | Sonra | Not |
|---|---|---|---|
| `USER_GUIDE.md` | `92fa9f2e027b65ee96237469b62401c75e17f811124e13c77b526d45007a3216` | **`5db95f23e17ffeab4a719ab58c4bad8f5678096e74b2b5de06ce1cfab99d717b`** | 472 → **564** satır |
| `data/rebuild/vocab_anka_r1_33114.json` | — | `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984` | **değişmedi** (ölçüldü) |
| `data/lexicon/roots_anka_r1.tsv` | — | `ea874a73c0d5669a591cef00c9d4fb16916ea3e60e42df9d3c73445c4b7efd59` | **değişmedi** |
| `data/anka_a1r.pt` | — | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | **değişmedi** (T-0080 değeriyle **aynı**) |
| `data/anka_a1r_pretrain.bin` | — | `9f9875762518829dec8f5baa41bd012ee1abf72fb630e24840bafedf691d4f22` | **değişmedi** |

**Saf ekleme kanıtı:** `git diff --numstat` ⇒ **112 / 0**, silinen satır grep'i **boş**.
> ⚠️ **Ölçüm uyarısı:** bu `numstat` **`HEAD`'e** göredir ve T-0083'ün commit
> edilmemiş değişikliğini **de kapsar**: 112 = 20 (T-0083) + **92 (T-0084)**.
> T-0084'e ait net değişim **+92 / −0**'dır (472 → 564). İki sayı çelişmiyor;
> **kümeler farklı** ([[iki-sayi-celisiyor-sanma-once-kume]]).

---

## 7. Değişen dosyalar — iki yönlü

| Yol | Durum |
|---|---|
| `USER_GUIDE.md` | **değişti** (+92 / −0) |
| `scratch/anka_r7_cikarim_sondasi.py` | yeni (gitignore'lu) |
| `data/eval/anka_r7_cikarim_2026-09-20.md` | yeni (bu rapor) |
| `.agent-bus/notes/T-0084.md` | yeni |
| **Toplam** | **4** — `len(cf) == len(set(cf))` ⇒ **tekillik GEÇTİ** |

Kapsam dışı **hiçbir** dosyaya yazılmadı: `train.py`, `test_model.py`,
`chat_prompt.py`, `SPEC.md`, `CLAUDE.md`, `README.md` **değişmedi**.

`git add -A` **kullanılmadı**; commit/push **operatör yetkisinde**.

---

## 8. Açık kalan işler (kapatılmadı)

1. **TARİHSEL banner cümlesi** §5.1'deki gerilim gereği yeniden yazılmalı.
2. **`chat_prompt.py`'a `--vocab` bayrağı yok** — `--model` var ama sözlük
   değiştirilemiyor ⇒ güncel checkpoint sohbet arayüzüne **bağlanamıyor**.
   Bu bir **kod** değişikliğidir; ayrı görev + ayrı kapı ister (T-0085 ile aynı sınıf).
3. **`test_model.py` dosya yoksa `return` eder** (sessiz çıkış) — teşhis zorlaştırır.
4. A1-r **SFT görmemiştir** ⇒ "sohbet/soru-cevap" iddiası bu checkpoint için
   **kurulamaz**; ölçülen `<OUTPUT>` oranı **2,0 × 10⁻⁸**.

---

*Raporun bütün sayıları bu turda koşulan komutlardan birebir alınmıştır.
Ölçülmeyen hiçbir mekanizma yazılmadı; ölçülen kendi kusurum (§3) dahil.*
