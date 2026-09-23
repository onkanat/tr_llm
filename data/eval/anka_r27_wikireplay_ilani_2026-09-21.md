# ANKA · ÖN-KAYITLI TEK DENEME İLANI — replay'in ALANI (T-0067)

**Damga:** 21 Eyl 2026 · **Bu belge ÖLÇÜMDEN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.

---

## 1. Ölçülmüş gerekçe — replay, unutmanın ölçüldüğü ALANI hiç kapsamıyor

`scratch/t0096_kos/mix_r4.bin.meta.json` (ölçüldü, okundu):

```
replay_source.path : data/train_chat_balanced.bin          (3.120.000 jeton)
replay_every       : 4   → %25
```

`data/train_chat_balanced.bin.meta.json` → `dataset_composition`:
`chat_conversations` · `middle_school_chat` · `parenting_deep` · `lexical_semantics` ·
`carpenter_specialization` · `rag_interactive` · `classic_rag` · `turk_tarihi_1931_chat` ·
`turk_tarihi_sft` — **Wikipedia YOK.**

Unutma ekseni A ise **tam olarak Wikipedia**'da ölçülür:

```
scripts/evaluate_carpenter_anka.py:59   WIKI_BIN = "data/anka_a1r_pretrain.bin"
```

⇒ **Dört modül konfigürasyonunun HİÇBİRİNDE A ekseninin kendi alanından replay yapılmadı.**
`anka_r20` raporu replay'i *"tek karede iki ekseni birden iyileştiren tek kaldıraç"* diye
kaydetmişti; **neden A'yı düzeltemediği hiç sorulmamıştı.** Bu ilan o soruyu sorar.

### Ayırt edilecek iki hipotez

| | hipotez | iddia |
|---|---|---|
| **H1** | **kapasite baskın** | r=16 / 36 katman / 1,33 M parametre yetmiyor; modül kaybı ancak tabanı bozarak düşürebiliyor ⇒ replay'i değiştirmek A'yı **değiştirmez** |
| **H3** | **replay alanı uyuşmazlığı baskın** | unutma, korunmayan bir alanda ölçülüyor; A alanından replay verilirse A **düşer** |

H1 ile H3 **zıt yön** öngörür ⇒ tek koşum ikisini ayırt eder.

---

## 2. Tek değişken

**Replay kaynağının ALANI.** Başka hiçbir şey değişmez.

| eksen | sabit tutulan değer |
|---|---|
| jacket kaynağı | `data/train_carpenter_specialization_anka.bin` (784.797 jeton, aynı) |
| desen | `[jacket x 3, replay]` = **%25** (aynı) |
| blok / karışım tohumu | 128 / 42 (aynı) |
| eğitim tohumu · lr · r · alpha · dropout · batch | 43 · 2e-4 · 16 · 32 · 0 · 8 (aynı) |
| sözlük | `data/rebuild/vocab_anka_r1_33114.json` (aynı) |
| **adım** | **1000** — mevcut +%13,02 noktasıyla **birebir** kıyas için |
| **değişen tek şey** | `--replay` : `train_chat_balanced.bin` → **`anka_a1r_pretrain.bin` (Wikipedia)** |

## 3. SIZINTI KAPISI (bu ilanın en kritik maddesi)

A ekseni, Wikipedia bin'inden **256 pencere × 128 jeton = 32.768 jeton** okur
(`a_ekseni`, seed 7, `B_PENCERE=128`). Ölçüldü: bu 32.768 jeton 100 M jetonun **%0,0328**'i ve
dosya boyunca **dağılmıştır** (ilk pencere jeton 31.232'de, son 99.368.320'de) ⇒ "güvenli bölge"
yoktur.

**Kapı:** eğitime giren her Wikipedia bloğu, A ekseninin 256 penceresiyle **TEK JETON BİLE**
paylaşmamalıdır. Üretim betiği bunu yazdıktan **sonra dosyayı yeniden okuyup** doğrular ve
çakışma **0** değilse `rc=2` ile DURUR.

**Neden zorunlu:** çakışma olsaydı A CE'nin düşüşü **koruma** değil **ezber** olurdu ve H3
sahte biçimde doğrulanırdı.

## 4. İlan edilen hüküm sınırı — ÖLÇÜMDEN ÖNCE

Ölçülmüş referanslar: taban A CE **3,5352** · modül@1000 (chat replay) A CE **3,9953** = **+%13,0161**.

| sonuç | hüküm |
|---|---|
| A CE **≤ 3,8888** (artış ≤ +%10,0) | **H3 desteklendi** — replay alanı bağlayıcı kısıttı |
| A CE **≥ 3,9600** (artış ≥ +%12,0; düşüş < 1,0 puan) | **H1 desteklendi** — kapasite bağlayıcı; replay alanı ilgisiz |
| 3,8888 < A CE < 3,9600 | iki etki **birlikte**; hiçbiri tek başına baskın değil |

**Ayrım gücü (ilan edilmiş):** modül@1000'in A yarı-genişliği ölçüldü **±0,0901** ⇒ ilan edilen
1,0 puanlık bant gürültünün **~11 katı**. Kapı bu farkı **ayırt edebilir**; edemezse hüküm
`AYIRT EDEMEDİ` olur ([[tavan-artefakti-kapi-gecmez-kanitsizlik]]).

## 5. Bu bir "ayar kovalama" DEĞİLDİR — gerekçesi

T-0067 ölçüm sonrası kusurun tasarıma göre onarılmasını yasaklar. Burada:

* Öngörü **iki yönlüdür**; **H1 gerçekten mümkündür** ve desteklenirse kaldıraç **elener**.
* Amaç kapıyı geçirmek değil, **hangi nedenin bağlayıcı olduğunu atfetmektir**.
* Tek değişken vardır; ölçüm öncesi ilan edilmiştir; hiçbir parametre sonuç görülerek seçilmez.

## 6. Ne okunur, ne okunmaz

* **Hüküm A ekseni üzerinedir.** Tek değişken A'yı hedefliyor.
* B ekseni ve ceket ekseni (ezber · tutarsızlık · ROUGE-L · kesişim) **kaydedilir** ama bu ilanın
  hükmüne **girmez** — ayrı bir sorunun konusudur.
* **Vakum kapısı** zorunlu: modül no-op ise hüküm kurulamaz (`modul_ile_olcum.py` içinde).
* Taban dosyası digest'i başta=sonda olmalı; değişirse koşum geçersizdir.

## 7. Uygulama sırası

1. `scripts/wiki_replay_disjoint.py` → `scratch/t0099_wiki_replay.bin` + çakışma kanıtı (0 olmalı)
2. `scripts/build_replay_mix.py --replay <o> --output scratch/t0099_wiki_mix_r4.bin` (mevcut araç, **değiştirilmez**)
3. `train_module.py --data scratch/t0099_wiki_mix_r4.bin --steps 1000 …` → `modules/marangoz_wikireplay_1000.mod.pt`
4. `scripts/modul_ile_olcum.py` → `data/eval/anka_r27_wikireplay_yetenek_2026-09-21.json`

**DOKUNULMAZ:** `data/**` (donmuş `data/*.bin` dahil) · `src/**` · `CLAUDE.md` · kapanmış
`data/eval/anka_r17…r26*` · `scratch/t0096_*` · `scratch/t0097_*`. Karışım **scratch/**'e yazılır
(`data/train_f4_replay_mix.bin` donmuş desene düşüyor — ölçüldü).
