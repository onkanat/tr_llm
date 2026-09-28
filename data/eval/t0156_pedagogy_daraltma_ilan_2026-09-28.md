# İLAN — T-0156 Pedagogy Daraltma: Router Yeniden-Eğitimi

**Damga (BETİKTEN, koşum-ÖNCESİ):** `2026-09-28T08:12:53Z`
**Meşruiyet:** T-0154 görünür-zayıflık (pedagogy train 416/500) onarımı.
**Operatör kararları:** DARALTMA (AskUserQuestion, 28 Eyl 2026 — parenting
hariç 4 dosya; per-metin `?` filtresi BU TURDA UYGULANMAZ) + **koşum-onayı
(28 Eyl 2026, ayrı mesaj: "T-0156 koşumunu onayla ve başlat")** — bu
onay `data/anka_router.pt` donmuş-desen YENİDEN-YAZIMINI da kapsar.

## Kök-neden (T-0154 keşif-kanıtlı)

`parenting_canonical.jsonl` çıplak-kelime morfoloji verisidir; grammar_core
kaynağı ile 1.097 örtük tekil-metin paylaşır; 600-örneklemde sınıflar-arası
çakışma pedagogy karışımına (416/500) akar. `data/pedagogy_canonical/**`
DONMUŞ → onarım betiğin okuma/seçim tarafındadır.

## DARALTMA (İLAN-değer)

`SINIF_KAYNAK[1]` yalnız **4 dosya**: `high_school_canonical.jsonl` +
`literature_canonical.jsonl` + `middle_school_canonical.jsonl` +
`turk_tarihi_canonical.jsonl` (**parenting HARİÇ**). grammar_core
(`lexical_semantics_dataset.jsonl`) ve carpenter DEĞİŞMEZ; legal fresh-init
kalır (hedef YOK). Keşif-ölçümü: daraltılmış pedagogy havuzu **7.048
tekil-metin** (İLAN-değer); ∩ grammar_core havuzu = **0**; 600 seçim bolca
yeter.

## T-0154'ten değişmeyen sabitler (konfigürasyon birebir)

SEED-42 · sınıf-başı 500 train + 100 val · `_metinler` (yalnız `input`,
boşsa `instruction`) · sorted(set) + rng.shuffle + ilk 600 ·
`_ozellik_cikar` (prompt_vec=mean-pool embedding; q_merak=fresh-init
seed-42 CuriosityEngine(768,768,tau=2.5) taban-hidden son-durum;
rag_vec=None; has_unk tam token_ids; 4096-kırpma görünür-sayaç) ·
`_gate_logits` = w_g(LN(w_p(p)+w_m(q))) · eğitim CPU AdamW lr=1e-3
batch=32 epoch=3 CE sınıf-ağırlığı-YOK; generator seed-42+epoch permütasyon ·
özellik-çıkarımı **MPS (yoksa DUR — sessiz-fallback YOK)** · kanonik kurulum
(LexiconManager+roots.tsv+build_default_graph+CrystalCompiler+
vocab_anka_r1_33114+KristalTokenizer+KristalLM 6/6/768 block 4096;
anahtar-silme cos_cached/sin_cached/mask; resize_state_dict; strict=False).

## Hedef-değerler (İLAN-öncesi SABİT; koşum-sonrası yumuşatma YOK)

- **Her sınıf train doğru ≥ 490/500** (500/500 BEKLENTİ-beyanı, KAPI değil;
  ≥490 kontaminasyon-kök-nedenini kapatır; artıklar karışım-matrisinde
  görünür). Pedagogy 490 altında → **DUR rc=2**; per-metin `?` filtresi
  YENİ görev olarak açılır.
- **Val top-1 ≥ 0,75 VE ≥ baseline+0,15 VE son-epoch kayıp < ilk-epoch.**
  Baseline fresh-init seed-42 router'dan **YENİ daraltılmış 300-val
  kümesinde** ölçülür; eski 0,2867 TAŞINMAZ. Val 100'ler daraltılmış
  havuzdan SEED-42 ile çekilir → eski 300-val'den FARKLI metinler (K5).
  Per-class val doğruluk RAPORLANIR; kapı yalnız bileşik val top-1.

## Çıktı (donmuş-desen yazım — operatör-onayı BU İLAN'da beyanlı)

`data/anka_router.pt` **ÜZERİNE YENİDEN YAZILIR**. **DEVİR-ÇIPASI:** mevcut
dosya sha256 ==
`64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720`
olmalı; değilse **DUR** (bilinmeyen artefakt ezilmez). Devir sd-sha
`9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500` ve
yeni `cikti_sha256` hüküm-JSON'a kaydedilir. Kurtarılabilirlik: K4
determinizm + T-0154 betiği DOKUNULMAZ (eski ağırlıklar bit-özdeş
yeniden üretilebilir).

## T-0155 çıpası BAYATLAR (bilinçli beyan)

`dogrulama_t0155_router_entegrasyon.py` K2 (`9b85f951…`) ve K3 (0,93 /
279-300) çıpaları ESKİ router'a bağlı; yeni yazım sonrası o betiğin
yeniden koşumu K2/K3'te DÜŞER — BEKLENEN; betik DOKUNULMAZ, tarihsel hüküm
DEĞİŞTİRİLMEZ. Halef kanıt bu turun **K8**'idir. Gateway `create_default`
default'u `data/anka_router.pt` olduğundan yeni ağırlıklar canlı tüketim
yoluna otomatik akar (davranış-değişimi beyanı).

## Kapılar (hüküm BETİKTEN; 8/8 şart; elle sayı/hüküm YOK; rc ∈ {0, 2})

- **K1 DEVİR-ÇIPASI + KAYNAK-DİGEST** — mevcut anka_router.pt sha ==
  `64527c72…` (T-0154 kanonik devir); base_v2 `d0f415f3…` + vocab
  `f9940a8d…` + roots.tsv `fe3005e5…` birebir.
- **K2 BASELINE** — fresh-init seed-42 router, daraltılmış 300-val'de
  top-1 (kayıt-değer; rastgele-taban 0,3333).
- **K3 EĞİTİM** — val top-1 ≥ 0,75 VE ≥ baseline+0,15 VE kayıp-azalıyor
  VE her sınıf train doğru ≥ 490/500.
- **K4 DETERMİNİZM** — ikinci eğitim koşumu state_dict SHA bit-özdeş.
- **K5 VERİ-ENVANTER (daraltma)** — SINIF_KAYNAK[1] == 4 dosya; pedagogy
  havuzu tekil == 7.048 (İLAN-değer); seçilen 600'ün parenting-havuz
  üyeliği == 0; grammar_core havuz-kesişimi == 0; 500/100×3 kırılım
  birebir; val∩train == 0; legal == 0; RAPOR: seçilen örneklerde `?` oranı.
- **K6 ŞEMA + ÇIKTI-YAZIMI** — state_dict anahtar-kümesi kanonik
  TriModalRouter birebir (9 anahtar); `data/anka_router.pt` yazılır; yeni
  cikti_sha + yeni sd_sha hüküm-JSON'a; canlı 192.168.1.9:6333 → 0 istek
  (betik VectorMemory KULLANMAZ).
- **K7 KARIŞIM-MATRİSİ** — train ve val 3×3 confusion (pred vs gerçek) +
  per-class doğruluk hüküm-JSON + rapora; off-diagonal sayaçlar GÖRÜNÜR
  (satır-kapısı K3'te; matris tamamı rapor).
- **K8 ENTEGRASYON-ÇIPASI (T-0155 halefi)** — yeni dosyadan
  `load_trained_router_state` yükleme → sd sha == K6 yeni sha; 9-anahtar;
  T-0155 betiği dokunulmaz + bayatlık beyanı hüküm-JSON'a.

8/8 → `T0156_PEDAGOGY_DARALTMA_GECTI` rc=0; aksi **DUR** rc=2 (çıktı
YAZILMAZ — K1 devir-çıpası hariç hiçbir aşamada dosya ezilmez; yazım
yalnız tüm kapılar geçtikten sonra tek noktada).

## DOKUNULMAZLAR

`data/pedagogy_canonical/**` (DONMUŞ; yalnız okunur) ·
`data/pedagogy/lexical_semantics_dataset.jsonl` ·
`scripts/dogrulama_t0154_router_egitim.py` ve
`scripts/dogrulama_t0155_router_entegrasyon.py` (tarihsel hüküm
artefaktları) · `src/llm/tokenizer.py` (DONMUŞ) · `src/compiler/**`
(DONMUŞ) · `src/gateway/agent_gateway.py` · canlı 192.168.1.9:6333
(0 istek; anka_bellek DOKUNULMAZ) · `tests/`.

## pytest (AYRI koşum — hüküm-dışı; kod DEĞİŞMEDİ)

Baseline **307/307**. Bu tur ÜRETİM-kod değiştirmiyor (yalnız yeni betik);
düşen beklenmez; düşerse İLAN'da beyan — yumuşatma YOK.

## Kiralamalar

claim-ile-birlikte (T-0153 dersi): `data/` (ÜST-DİZİN — donmuş `data/*.pt`
deseni; anka_router.pt yeniden-yazımı açık-beyan), `scripts/`, `data/eval/`,
`.agent-bus/notes/` — TTL 180 dk.

## Çıktılar

`data/eval/t0156_pedagogy_daraltma_hukum_2026-09-28.json` ·
`data/eval/t0156_pedagogy_daraltma_rapor_2026-09-28.md` ·
`data/eval/t0156_pedagogy_daraltma_kosum_2026-09-28.log` (yönlendirme) ·
notes `T-0156.md`. Hüküm BETİKTEN; İLAN ≠ RAPOR.

## Süreç-disiplini

Koşum **sandbox DIŞI** (MPS) ve **sarmal-sız** (T-0155 dersi: nohup…echo
rc'yi YUTAR); iki MPS koşumu aynı pencerde paralel YÜRÜTÜLMEZ; İLAN
koşum-sonrası YUMUŞATILMAZ; İLAN-revizyonu yalnız koşum-öncesi İLAN-2 ile.