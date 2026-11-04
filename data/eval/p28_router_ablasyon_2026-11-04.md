# P2.8 — Router Ablasyonu: q_merak Kanalı Katkısı (İLAN → HÜKÜM → RAPOR)

- **Tarih:** 2026-11-04
- **Kaynak:** `src/llm/router.py:8-89` (`TriModalRouter`), `src/rag/merak.py` (`CuriosityEngine`)
- **Hüküm kaynağı:** scratch ölçüm betiği (bellek-içi kanal maskesi; eğitim KOŞMADI, model dosyasına YAZILMADI, commit YOK)
- **İlgili önceki kayıt:** `data/eval/t0210_router_v2_pure_hukum_2026-10-15.json` (val_top1=1.0, state_dict sha `a2a95957…`)

---

## İLAN — Ölçüm Düzeni

**Soru:** `q_merak` (merak/curiosity) kanalı sıfırlandığında router'ın val doğruluğu ne kadar düşer?

**Hattın tanımı (ölçülen kod):**
- Router: `TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256, num_experts=5, top_k=2)` — 5 uzman: `grammar_core, pedagogy, carpenter, gardener, legal`.
- Birleşme: `e_route = W_p(P) + W_m(q_merak) [+ W_r(ctx_rag)] → LayerNorm → w_g → top-2 softmax` (router.py:68-87). Değerlendirmede `ctx_rag` kullanılmaz (T-0210 `_gate_logits` deseni birebir).
- `q_merak` kaynağı: `CuriosityEngine(hidden=768, tau=2.5)` — son token gizli durumundan `LayerNorm(q_proj(z_son))`; entropi/`<UNK>` sinyalleri tetik durumunu belirler, vektörün kendisi gizli durum projeksiyonudur (merak.py:79). Router'a giren kanal = `w_m` lineer projeksiyonu (768→256).

**Val seti (T-0210 birebir yeniden üretim):**
- 4 aktif sınıf × 100 = **400 örnek** (SEED-42 seçim; sınıf başına 500 train + 100 val düzeninden val dilimi).
- Kaynaklar: grammar_core←`lexical_semantics_dataset.jsonl`; pedagogy←`pedagogy_canonical/*` (4 dosya); carpenter←`carpenter_canonical.jsonl`; gardener←`bahcivan_arena.jsonl`. Seçim yalnız bu JSONL içeriklerine bağlıdır; içerikler 15 Eki koşumundan beri değişmedi.

**Ablasyon yöntemi:**
- (a) **Taban:** `data/anka_router_v2.pt` state_dict'i olduğu gibi (sha256 `74b9c808…`, T-0210 çapası birebir; state_dict SHA `a2a95957…` hüküm kaydıyla birebir doğrulandı).
- (b) **Ablasyon:** state_dict **bellekte** klonlanır, `w_m.weight` ve `w_m.bias` sıfırlanır ⇒ `w_m(Q) ≡ 0`. Diske YAZILMADI; `anka_router_v2.pt` bayt-düzeyinde dokunulmadı.
- (c) Özellikler her iki koşulda AYNI `P, Q` tensörleri (tek özellik-çıkarımı koşulu başına; değişken yalnız router ağırlıkları). Eğitim yok, yalnız ileri-geçiş.

**Kapılar:**
- **A1 Bütünlük:** val seti seçimi + router state_dict SHA + taban top-1, T-0210 kaydıyla birebir üreyecek (beklenen taban: 1.0).
- **A2 Denklik:** sıfırlanmış `w_m` ile "q_merak verilmez" (P-only) yolu logit-bazında özdeş olacak (kanal maskesinin doğru uygulandığının kanıtı).
- **A3 Lexicon drift bağışıklığı:** roots.tsv T-0210 epoch'undan beri değişti (mevcut sha `e32677b2…` ≠ çapa `fe3005e5…`); 3/400 val metninde encode farkı ölçüldü — ölçüm hem eski (git `d75e71d`, scratch probe kopyası) hem güncel lexicon ile koşuldu; her iki koşulun taban/ablasyon sonuçları birebir çıktı (drift hiçbir tahmini çevirmedi).

---

## HÜKÜM

> **q_merak kanalı router'ın val top-1 doğruluğuna +24,75 puan katıyor: 1,0000 → 0,7525 (kanal sıfırlanınca düşüş).**
> Kanal kritik ve sınıf-seçici: katkısız en çok düşen sınıf **pedagogy (−68 puan: 1,00 → 0,32)**, ardından **carpenter (−30 puan: 1,00 → 0,70)**; grammar_core kanalsız da tam doğrulukta (0 puan), gardener −1 puan.
> Sayısal hüküm: **kanal katkısı = 24,75 puan genel top-1; top-2'de 14,25 puan (1,0000 → 0,8575). q_merak olmayan router pedagogy'yi ayırt edemez (karışıklığın %50'si pedagogy→gardener).**

---

## RAPOR

### 1. Genel sonuç (400 val örneği, MPS, top-k gate argmax)

| Ölçüm | Taban router | q_merak sıfırlı | Fark (kanal katkısı) |
|---|---|---|---|
| **Val top-1 doğruluk** | **1,0000** (400/400) | **0,7525** (301/400) | **−24,75 puan** |
| Val top-2 doğruluk | 1,0000 | 0,8575 | −14,25 puan |
| Hatalı örnek | 0 | 99 | +99 |
| Ort. top1−top2 logit marjı | 7,68 | 5,65 (doğrularda 6,41) | −2,03 |

### 2. Sınıf-bazlı kırılım (ablasyon; taban tüm sınıflarda 1,00)

| Sınıf | Ablasyon doğruluk | Doğru/Toplam | Düşüş | Baskın karışıklık |
|---|---|---|---|---|
| grammar_core | 1,00 | 100/100 | **0 puan** | — |
| gardener | 0,99 | 99/100 | −1 puan | →grammar_core (1) |
| carpenter | 0,70 | 70/100 | **−30 puan** | →gardener (24), →grammar_core (6) |
| pedagogy | 0,32 | 32/100 | **−68 puan** | →gardener (50), →grammar_core (11), →carpenter (7) |

Tahmin dağılımı (ablasyon): grammar_core 118, pedagogy 32, carpenter 77, gardener 173 (taban: her sınıf 100; legal 0). q_merak'sız model "yumuşak/sohbet" sınıfları (pedagogy, gardener) birbirine yığarken, grammar_core prompt gömmesiyle ayrışabiliyor.

### 3. Kanal büyüklüğü (LN öncesi e_route katkıları, ortalama L2)

- `‖w_p(P)‖ = 6,28` — `‖w_m(Q)‖ = 20,11` ⇒ q_merak kanalı e_route'a prompt kanalının **~6,2×** büyüklüğünde katıyor. Router'ın karar uzayı fiilen merak vektörü tarafından taşınıyor; bu, 24,75 puanlık düşüşün mekanik açıklaması.

### 4. Ölçüm bütünlük kanıtları

- **A1 GEÇTİ:** taban top-1 = 1,0000 — T-0210 hüküm kaydındaki `val_top1=1.0` birebir üredi. Router state_dict SHA: `a2a95957…` (kayıtla özdeş). Val seçimi SEED-42 ile 4×100 üredi.
- **A2 GEÇTİ:** sıfırlı `w_m` ⇔ "q_merak verilmez" yolu: maksimum logit farkı **0,0** (bit-özdeş) — maske doğru uygulanmış.
- **A3 GEÇTİ:** eski (T-0210 epoch) roots.tsv ile güncel roots.tsv koşulları tüm metriklerde birebir (3/400 encode drifti hiçbir tahmini değiştirmedi). Kanal hükmü lexicon düzenlemelerine dayanıklı.
- Kaynak SHA'lar: `anka_router_v2.pt` = `74b9c808…` ✓, `anka_taban_pure.pt` = `5479a2f3…` ✓, `vocab_anka_r1_33114.json` = `f9940a8d…` ✓, `bahcivan_arena.jsonl` = `9fa356b4…` ✓, roots.tsv (T-0210 epoch, git `d75e71d`) = `fe3005e5…` ✓; güncel roots.tsv = `e32677b2…` (bilinçli sapma, T2.1/T2.5).

### 5. Notlar

- Ablasyon ağırlıkları yalnız bellekte yaşadı; hiçbir `.pt` yazılmadı, izlenen dosya değiştirilmedi, commit yapılmadı. Ölçüm betikleri scratch'te (`~/.hermes/cache/scratch/p28_ablation.py`, `p28_val_probe.py`); ham sonuç: `p28_ablasyon_sonuc.json` (scratch).
- `w_r` (ctx_rag) değerlendirme yolunda hiç kullanılmıyor (T-0210 deseni); ablasyon kapsamı q_merak kanalıdır.
- Cihaz: MPS (T-0210 ile aynı); gate matematiği CPU float32.

**Sonuç:** q_merak kanalı, TriModalRouter'ın val doğruluğunda 24,75 puanlık belirleyici katkıdır; katkı sınıf-seçicidir (pedagogy −68, carpenter −30, grammar_core 0). Kanal, prompt gömmesinin ayrıştıramadığı pedagogy/gardener uzaklığını merak vektörü üzerinden açar.