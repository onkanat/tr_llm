# ANKA_i1 · ÖĞRENME ORANI DENEMESİ — SONUÇ: kapı seg_6'da düştü, lr ekseni KAPANDI

**İlan:** `data/eval/anka_i1_lr_ilani_2026-09-22.md` (**koşumdan önce** yazıldı; eşikler değiştirilmedi)
**Koşum:** `scratch/anka_i1_lr2e5_kos/` · sürücü `scratch/anka_i1_surucu.py --lr 2e-5` · `rc=2` (kapı) · 1442 sn
**Tek değişken:** `--lr` 2e-4 → 2e-5 (başka hiçbir şey değişmedi; ilan §2 birebir uygulandı)

---

## 1. Hüküm: İlan §3'ün **3. dalı** — "1 düştü ⇒ 10× lr düşüşü bile yetmiyor ⇒ kısıt
**tam-parametre güncellemesidir** (parametre-verimli aşama gerekir) ⇒ **lr ekseni KAPANIR**."

| # | ilan edilmiş okuma | eşik | ölçülen | sonuç |
|---|---|---|---|---|
| **1 (kapı)** | Wikipedia CE her segmentte | **≤ 3,8887** | **5/6 geçti**; seg_6'da **3,9275 > 3,8887** | **DÜŞTÜ** (seg_6) |
| **2 (zarf)** | `tutarsızlık` (nihai ckpt) | **< %10** | **%100,00** (taban %100) | **DÜŞTÜ** |

İki dal da kapandı: koşum 6. segmentin kapısında durdu ⇒ ilan §3'ün "1 geçti, 2 düştü" ara dalı
uygulanmaz; **3. dal** geçerli.

## 2. CE eğrisi — kapı ilan edildiği gibi çalıştı

| segment | adım | Wikipedia CE | ppl | tabana göre | kapı |
|---|---|---|---|---|---|
| taban | — | 3,5352 | 34,30 | — | — |
| seg_1 | 500 | 3,5913 | 36,28 | +%1,59 | geçti |
| seg_2 | 1000 | 3,6406 | 38,11 | +%2,98 | geçti |
| seg_3 | 1500 | 3,7053 | 40,66 | +%4,81 | geçti |
| seg_4 | 2000 | 3,7887 | 44,20 | +%7,17 | geçti |
| seg_5 | 2500 | 3,8568 | 47,31 | +%9,10 | geçti |
| **seg_6** | 3000 | **3,9275** | **50,78** | **+%11,10** | **DÜŞTÜ** |

Kapı, 40 dakikalık zincirin **son** noktasında düştü — erken segmentlere bedel ödetmedi.
Koşum DURDU (rc=2), `sonuc.json` `durum: DURDU · duran_dal: seg_6 PPL KAPISI`,
`segments.jsonl` 6 kayıt (artımlı), sentinel yok (BITTI olmadı — doğru).

## 3. Yan hipotez (ilan §4): doğrusallık **DESTEKLENMEDİ**

Hipotez: 2e-5'te +%0,263/100 adım ⇒ 3000 adımda ≈ +%7,9 ⇒ kapı geçmeli.
**Ölçülen:** +%1,59/500 adım başlangıç (≈ +%0,32/100) ama eğri **hızlanıyor**; 2500 adımda
+%9,10, 3000'de +%11,10. Eğri tavanı 2500-3000 adım bandında aştı ⇒ **koşum 6 segmenti
tamamlamadı ⇒ §4'ün "erken durdu ⇒ hasar lr'den yavaş düşüyor" dalı**.

Hız karşılaştırması (tek ölçümler, /100 adım):

| koşum | lr | hız | oran |
|---|---|---|---|
| anka_i1 (DÜŞTÜ) | 2e-4 | **+%2,63** | 1× |
| bu koşum, seg_1 | 2e-5 | **+%0,32** | 8,3× düşüş |
| bu koşum, ort. (2500 adım) | 2e-5 | +%0,44 | 6,0× düşüş |

⇒ Hasar lr ile **azalıyor ama 10× tam oransal değil**; ve azalsanın **pratikte karşılığı yok**:
kapı bütçesi (CE +%10) bu tabanda yalnız **~2500-3000 adım** açıyor (optimizer-reset bedeli
hariç; ilan §5'te beyan edilen `1,73×` ilk-güncelleme etkisi hasarı biraz **şişirir** —
optimizer taşınmış olsaydı eğri biraz daha yavaş olurdu, ama aşağıdaki zarf bulgusu bundan
bağımsızdır).

## 4. ASIL BULGU: zarf bu tarifte **hiç öğrenilmedi — iki lr'de de**

| koşum | adım | tutarsızlık | ROUGE | kesişim | ezber | B |
|---|---|---|---|---|---|---|
| taban | — | %100 | 0,0000 | %0 | %0 | — |
| lr2e-4, 500 adım (DÜŞTÜ) | 500 | %100 | 0,0020 | %1 | %0 | +3,49 |
| **lr2e-5, seg_1** | 500 | **%100** | 0,0006 | %0 | %0 | +0,32 |
| **lr2e-5, seg_6** | 3000 | **%100** | 0,0000 | %0 | %0 | **+3,65** |

⇒ **Güvenli dozda tam-parametre talimat aşaması: LM'i %11 bozar, zarfa %0 verir.**
lr ekseninin iki ucu da ölçüldü: 2e-4 **hızlı hasar, zarf yok**; 2e-5 **yavaş hasar (ama
bütçeyi yine tüketir), zarf yine yok**. B ekseninde (noktalama top-1) **iki koşum da iyileşti**
(+3,5 / +3,7) — talimat verisi B'yi öğretiyor; sorun zarf sözdiziminde.

**Kalan tek kaldıraç (ilan §3, 3. dalın söylediği): parametre-verimli aşama.** Bu, ölçülmüş
bulgularla uyumlu: modül **biçim öğreniyor ve onarıyor** (r33/r34'te iki bağımsız
checkpoint'te tekrarlandı) — talimat zarfı da bir **biçim** davranışıdır. Sıradaki
sınanmamış hücre: **talimat zarfını modülle (taban dondurulmuş) öğretmek.**

## 5. Koşum kesintisi notu (operatörün uyardığı olay)

Önceki oturum kapanırken arka plandaki koşum **ölmedi**: sürücü (PID 25043) + eğitim çocuğu
(PID 25515) oturum kesintisinden **sağ çıktı** ve zincir 6 segmenti kendi kendine koştu
(`lsof` ile iki PID de doğrulandı; `ps` sandbox'ta yasak). Oturum kesintisi koşuma **zarar
vermedi** — artımlı yazım (`segments.jsonl` append, `.tmp`+`os.replace`) bunu taşıdı.

## 6. Kapılar ve temizlik

| kapı | durum |
|---|---|
| **ppl kapısı (H1)** | **KOŞUM SIRASINDA ateşlendi**: seg_6'da `rc=2` DUR — ilan edildiği gibi |
| H1 (tek eğitici) | koşum öncesi 0 tutucu ✓ |
| H3 (`timeout=`) | sürücüde her `subprocess.run`'da ✓ (zaman aşımı yok) |
| H2 (kapanış bekçisi) | koşum DURDU (sentinel yok) ⇒ kapanış aracı koşulmadı (BITTI aracıdır) ✓ |
| G1-G6 (ölçüm kabı) | iki kanonik eval'de ✓ (kırpılmış kafa uyarısı: 33.114 kafa = sözlük ✓) |

**DOKUNULMAZ korundu:** `data/**` · `CLAUDE.md` · `scratch/t0096_*` · `scratch/t0097_*` ·
`scratch/anka_i1_kos/` · kapanmış `data/eval/anka_r17…r34*`. **Commit yok · `git add -A` yok.**

## 7. Artefakt digestleri (TAM sha256)

| dosya | sha256 |
|---|---|
| `scratch/anka_i1_lr2e5_kos/seg_1.pt` | `4ef5c12a46f90d8f847033026c0dba20c354044fdf37c0e85a964f0f6cdf378f` |
| `scratch/anka_i1_lr2e5_kos/seg_2.pt` | `f3e4fbcdd3b93a46c5fab4297fb280ff450c88566f90b11f7828c4f5ded16b8e` |
| `scratch/anka_i1_lr2e5_kos/seg_3.pt` | `99fbf4bbe4716ffc1acec6125348606e1740611dc05c98d53d3c325a5d6a66af` |
| `scratch/anka_i1_lr2e5_kos/seg_4.pt` | `a5f5bcacebf2adbb4d99b0d82e114b05ee64a850167e7828a9405c01ff0ebf50` |
| `scratch/anka_i1_lr2e5_kos/seg_5.pt` | `a9f0c0a138a531c64533b2f5204a5fd536b73f78f9dcf64ae5262eaf922aa885` |
| `scratch/anka_i1_lr2e5_kos/seg_6.pt` | `a06224cc4c5bc311706d34838a0a378875f108f52f2771a29a433ede07af5cdd` |
| `data/eval/anka_i1_lr2e5_seg1_yetenek_2026-09-22.json` | `0b140e11e8e522777cfe71ea2c4e0529e7968c0bad103a616ef8bb98542e55f8` |
| `data/eval/anka_i1_lr2e5_yetenek_2026-09-22.json` | `e4dda893fffdcc45583732235c60006ff15c74a019ef97e3c8b25f5cc85b88fb` |
| `data/anka_a1r.pt` (taban, **değişmedi**) | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` |
| `scratch/anka_i1_talimat.bin` (**değişmedi**) | `798c2adc55430cd57ce5dfcf182e78d312822782c507f0695779799b1f5f6c3b` |

Karşılaştırma tablosu: `venv/bin/python scripts/anka_karsilastirma_tablosu.py` (satırlar
`anka_i1 lr2e-5 seg_1` ve `anka_i1 lr2e-5 nihai` — ham JSON'lardan, elle sayı yok).