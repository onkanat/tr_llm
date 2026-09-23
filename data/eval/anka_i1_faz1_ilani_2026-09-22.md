# ANKA_i1 · FAZ 1 İLANI — talimat aşaması (ön-kayıtlı)

**Damga:** 22 Eyl 2026 · **Bu belge KOŞUMDAN ÖNCE yazıldı.** Ölçümden sonra değiştirilemez.
**Plan:** `~/.claude/plans/enchanted-wiggling-moon.md` (operatör onaylı).

---

## 1. Soru

Ölçüm şunu söyledi (r28→r34, yedi ön-kayıtlı deneme): **taban sağlam bir dil modeli** (ppl 33,4)
ama **talimat zarfını hiç görmemiş** (zarflı istemde çıktısı `, [?], [?], …`, tutarsızlık %100).
Modül bilgi ekleyemiyor; tam ince ayar ekliyor ama dil modelini 18× bozuyor.

**Soru:** Tabanı **genel talimat verisiyle** eğitmek, zarfı öğretirken dil modelini
**koruyabilir mi?**

## 2. Faz 0 — ön-ölçüm (KOŞUMDAN ÖNCE yapıldı, `data/eval/anka_i1_faz0_2026-09-22.json`)

| ölçüm | değer | hüküm |
|---|---|---|
| külliyat | `data/train_chat_balanced.bin` 3.120.000 jeton · 24.375 blok · sha256 `443f93d3…` | — |
| **PAD jetonu** | **%50,29** | ⚠ yüksek — ama aşağıya bak |
| korunan hedef — PAD maskesi **kapalı** | **%25,0803** | seçilen rejim |
| korunan hedef — PAD maskesi **açık** | **%25,0803** | **birebir aynı** |
| `<OUTPUT>`'lu blok | **%99,975** | zarf sözleşmesi uygun |
| `<OUTPUT>`'suz blokta korunan | **%0,00** | beklenen (kural 5) |
| sözlük hizası | `vocab_base_32852` ⊂ `vocab_anka_r1_33114` · ortak 32.852 · **id kayması 0** | ✓ |
| `max_id` | 32.850 < 33.114 | ✓ |
| **taban çizgisi** | tutarsızlık **%100** · ezber %0 · ROUGE 0,0000 · kesişim %0 · **CE 3,5352** (ppl 34,30) | referans |
| **CE tavanı** | **3,8887** (`ESIK_A_ARTIS=10.0`, `evaluate_carpenter_anka.py:68`) | kapı |

> **`--no-pad-mask` kararının bedeli ÖLÇÜLDÜ ve SIFIR çıktı.** Külliyatın yarısı PAD olmasına
> rağmen PAD maskesini açmak korunan hedef oranını **hiç değiştirmiyor** (%25,0803 →
> %25,0803) ⇒ PAD konumları **zaten** istem maskesiyle −100. Seçilen kodsuz yol bu külliyatta
> **bedelsizdir**.

## 3. Hipotez ve iki dal (ÖNCEDEN yazılır)

| | |
|---|---|
| **H-İ** | Genel talimat aşaması tabanı **zarf-izleyebilir** yapar **ve** dil modelini korur ⇒ mimarinin ilk aşaması **kurulabilir**; sıradaki iş yetenek modülü sınaması (Faz 2) |
| **H-R** | Aşama **ya zarfı öğretemez ya dil modelini bozar** (ya da ikisini) ⇒ "önce talimat temeli" tarifi bu veri/bütçeyle **çalışmıyor**; karar **veri/bütçe eksenine** kayar |

**Her iki dal da karar üretir.** Olumsuz sonuç başarısızlık değil, **elenmiş bir daldır**.

## 4. Tek değişken ve tarif (dondurulmuş)

| eksen | değer |
|---|---|
| **yeni olan** | **talimat aşaması** — tabana genel talimat verisiyle 500×6 adım |
| taban | `data/anka_a1r.pt` · sha256 `b93cc1cd…` (**salt okunur**) |
| veri | `data/train_chat_balanced.bin` (**donmuş**, salt okunur) |
| sözlük | `data/rebuild/vocab_anka_r1_33114.json` (**zorunlu**; 32.852 ile koşmak gömme satırlarını sessizce kırpar) |
| blok · batch · lr | **128** · 8 · 2e-4 |
| rejim | **SFT + `--no-pad-mask`** (`--pretrain` YOK) |
| segmentler | **6 × 500** adım, `--load-path` ile zincirli |
| tohum | 42+i |
| çıktı | `scratch/anka_i1_kos/seg_N.pt` (**donmuş değil**, gitignored) |
| `--save-optimizer` | **KAPALI** (disk: 28 Gi boş; optimizer'lı 6,4 GB, sizsiz 2,1 GB) |

## 5. İlan edilen kabul ölçütü (operatör kararı #3 — kanonik kap, yeni yüzey yok)

| # | ölçüt | eşik | kaynak |
|---|---|---|---|
| **1 (zarf)** | `tutarsızlık` | **< %10** (taban %100) | `evaluate_carpenter_anka.py` `ceket_ekseni` |
| **2 (LM)** | Wikipedia **CE** | **≤ 3,8887** (ppl ≤ 48,85) | aynı kap, `ESIK_A_ARTIS` sabitinden |
| kayıt | ezber · ROUGE · kesişim · A/B | hükme girmez | aynı kap |

**İkisi birden tutmalı** ⇒ H-İ. **Biri düşerse** ⇒ H-R ve hangisinin düştüğü **adıyla** yazılır
("zarf öğrenildi ama LM bozuldu" ≠ "LM korundu ama zarf öğrenilmedi").

## 6. Sert kapı — koşum SIRASINDA (fail-closed)

Her segment sonunda Wikipedia CE ölçülür; **CE > 3,8887 ise koşum rc=2 ile DURUR** ve o ana
kadar yazılan her şey diskte kalır. Gerekçe: r34 ölçtü ki hasar birikince geri dönüşü pahalı
(ppl 34 → 497). **Kapı, geçmişteki iki tam ince ayarı da erken durdururdu.**

## 7. Ölçümün SINIRI (ilan edilir)

* **Kayıp ölçeği kıyaslanamaz:** `--no-pad-mask` rejimi `seg_*` loglarından **farklı** bir kayıp
  ölçeği üretir (CLAUDE.md:26). Sayılar **yalnız kendi içinde** kıyaslanır.
* **Optimizer taşınmıyor** ⇒ segmentler arası AdamW momentleri sıfırlanır. Her segment kendi
  içinde tam bir eğitimdir; **ilan edilir** (T-0096 taşıyordu).
* **Tohum** segment başına 42+i ⇒ segmentler bağımsız değil, **zincirli**.
* **Veri ekseni sabit** — yeni külliyat üretilmiyor.

## 8. Ölçülecek ve yazılacak

* `scratch/anka_i1_kos/segments.jsonl` — her segment: adım, kayıp, **Wikipedia CE/ppl**, süre,
  checkpoint sha256, `son60` istatistiği (yapısal ayrıştırma; T-0096'nın yanlış-önek kusuru
  tekrarlanmaz).
* `scratch/anka_i1_kos/sonuc.json` — artımlı (`.tmp` + `os.replace`); `SONUC_BITTI` sentinel'i
  **yalnız rc==0** yolunda.
* `data/eval/anka_i1_faz1_sonuc_2026-09-22.md` — hüküm + digest tablosu.

## 9. Koşum işletmesi (plan §Uygulama deseni)

`timeout=` **her** `subprocess.run`'a · koşum öncesi `tek_egitici_onkontrol` · kapanış aracının
ilk satırında `kosum_bekcisi` · eşikler koddan elle yazılmaz.

**DOKUNULMAZ:** `data/**` (donmuş) · `CLAUDE.md` · `scratch/t0096_*` · `scratch/t0097_*` ·
kapanmış `data/eval/anka_r17…r34*` · `src/**`. `train.py` **DEĞİŞMEZ** (karar #5).
`git add -A` yasak; commit yok.
