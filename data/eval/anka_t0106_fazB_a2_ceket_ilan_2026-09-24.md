# T-0106 FAZ B İLAN — a2-tabanlı ceket koşumu (TEK DEĞİŞKEN: r1→r2 derleme)

**Tarih:** 2026-09-24 · **Görev:** T-0106 · **Plan:** `~/.claude/plans/enchanted-wiggling-moon.md` (FAZ B)
**Önkoşul:** FAZ A KAPANDI (`anka_t0106_fazA_decomp_rouge_sonuc_2026-09-24.md`).

## 1. TEK DEĞİŞKEN BEYANI

P3 paterni **BİREBİR** korunur; koşumda değişen yalnız taban (a1r→a2 — T-0105'te
hali hazırda taban-çıpa deseniyle ölçülmüştü) ve derleme temsili:

| Kavram | P3 (r1) | **Bu koşum (r2)** | Beyan |
|---|---|---|---|
| taban checkpoint | `data/anka_a1r.pt` | `data/anka_a2.pt` (sha `e2352cd3…`) | a2; sidecar YOK ⇒ carry SIFIRDAN (T-0092/G4a, açık uyarı) |
| sözlük | `vocab_anka_r1_33114.json` (f9940a8d…) | `vocab_anka_r2.json` (**14ce9f4e…**, 33.911) | ölçüldü |
| lexicon | `roots_anka_r1.tsv` (ea874a73…) | `roots_anka_r2.tsv` (**c355a372…**) | ölçüldü |
| metin kaynakları | carpenter+KATALOG_R18+parenting+highschool | **AYNI** | TEK DEĞİŞKEN |
| carve (V5) | T-0094 digest'leri | **AYNI** (kapı yeniden kanıtlayacak) | — |
| patern/blok/lr/scheduler | idx%5 · 128 · 8 · 1e-4 · 50+cosine 6000 | **AYNI** | 12.000 adım tekrarı YOK (P4 hükmü) |
| G temizlik paketi | — | **SFT/ceket'e YAPILMAZ** | G yalnız wiki külliyatındaydı (T-0104); P3 paterni bozulmaz |
| `recompile_datasets.py` | — | **KULLANILMAZ** | BAYAT (kanıt: T-0104) |

## 2. Kod değişiklikleri (koşumdan ÖNCE)

* **Ceket r2:** `scratch/anka_t0106_ceket_r2_build.py` — `anka_r18_ceket_build` İMPORT + modül-ezme
  (VOCAB/VOCAB_SHA/SOZLUK_BEKLENEN/LEXICON/LEXICON_SHA/OUT_BIN/OUT_META/BEYAN_EDILEN_DONMUS/
  **CKPT/RAPOR_MD/ILAN** — kapalı T-0097 kanıt CKPT'si `anka_r18_build_2026-09-21.json` ÜZERİNE
  YAZILMAZ). Çıktı: `data/train_carpenter_specialization_anka_r18_v2.bin` + meta.
* **SFT r2:** `scratch/anka_t0106_sft_r2_build.py` — `anka_p2_sft_build` İMPORT + ez (VOCAB/LEXICON/
  CIKTI); çıktı `data/rebuild/anka_t0106_sft_mix_r2.bin` + meta. Meta'ya üretimden hemen sonra
  `sozluk/sozluk_sha256/sozluk_giris` alanları **EKLENİR** (eklemeli, sürücü kapısı için beyanlı).
* **Sürücü:** `scratch/anka_p3_surucu.py` — CLI `--sozluk --wiki-bin --sft-bin --ceket-bin --lexicon`
  (varsayılan = P3 sabitleri; eski kayıtlar bozulmaz); **VOCAB-UYUM KAPISI** (fail-closed): üç binin
  meta'sından `sozluk_sha256` okunur, sürücü sözlüğünün tam digest'ıyla kıyaslanır; uyuşmazlık
  VEYA meta-yok/alan-yok DURUR (sessiz id-kayması yasak); sonda çağrısına `--vocab/--lexicon`
  passthrough (ECA r1-default'a düşmesin).

## 3. Beklenti çıpa tablosu (kayıt JSON'larından — elle sayı yok)

| | Kaynak | r1 ölçülen | r2 BEKLENTİ |
|---|---|---|---|
| ceket kayıt | r18 meta (22.283) | 22.283 | birebir AYNI (kayıt sayısı tokenizasyondan bağımsız) |
| ceket jeton | r18 meta | 1.568.123 | ~1,56-1,60 M (r2 ek jetonlar) |
| SFT kayıt | p2 sft r1 meta (125.814) | 125.814 | birebir AYNI |
| SFT jeton | p2 sft r1 meta | 16.104.192 | ~16,1-16,4 M |
| V2 taban-invaryans | ölçüldü (24 Eyl) | — | r2'nin ilk 32.852 id'si base ile birebir (**farklı id = 0**, ölçüldü) |
| V5 carve | T-0094 digest'leri | PASS | PASS (digest kapıları r18 betiğinde) |
| V6 sızıntı | r18/p2 kapıları | 0 | 0 (fail-closed) |
| V8 ölçek | r18 kapısı | >1,15 M | PASS |

**Koşum canlılık beklentileri (Faz B başlangıcı ÖNCE yazılır):**
* Başlangıç kaybı **3,3-3,6 bandı** (T-0105 a2 A-CE çıpası 3,3279; sıfırdan-lnV GEÇERSİZ — devam koşumu).
* CE kapısı: CE ≤ CE_TAVAN 3,8887 (3,5352 × 1,1) — a2 taban 3,3279 bant-içi.
* TEPE referansı: `--referans-sonda data/eval/anka_t0105_a2_yetenek_2026-09-24.json`
  (a2 taban ROUGE 0,0049 · ezber 0,0 · tutarsızlık %100).
* P3/P4 RAW çıpası 0,1014/0,1346 band-ÜSTÜ beklenir ama VAAT EDİLMEZ (külliyat arzı tükendi;
  P4 hükmü: bu koşum külliyat değişkenini vaat etmez — değişken taban+derlemedir).
* RAW hüküm eşiği 0,3221 SABİT; DECOMP kolon tanısal (aday 0,7988 bağlanmaz).
* pad-mask seyreltme: maskeli kayıp ≈ gerçek CE × maskesiz oran (T-0053) — L1 kapısı
  `kayip > lnV − 1,0` devam koşumunda BOZUKLUK imzası olarak çalışır.

## 4. Kabul koşulları

1. İki r2 bin + meta üretilmiş; ceket r2 V-kapıları rc=0 (TÜMÜ PASS); SFT zarf/sızıntı kapıları PASS.
2. Sürücü vocab-uyum kapısı üç meta'yı geçmiş (negatif dal beyanlı: uyuşmazlıkta DURUR — P3 sabitlerinde geri-uyum korunur).
3. a2 koşum rc=0 · 3 sonda JSON (r2 vocab/lexicon pinli, DECOMP kolon dolu) · HELD-OUT çıpası bozulmamış.
4. `scripts/anka_karsilastirma_tablosu.py` TARİFLER'e t0106 sonda eklenecek (Faz C); elle sayı YOK.

## 5. Açıkça YAPILMAYANLAR

Eski kapalı kayıtlara yazım · ESIK_ROUGE'a dokunma · G paketinin ceket/SFT'ye katılması ·
`recompile_datasets.py` · kanonik kod kopyası · 12.000 adım · `git add -A` · ilansız ölçüm ·
çift eğitici · kiralamasız donmuş yazım (`--allow-frozen-write` + `data/` üst dizin kiralama + writes[] beyanı).

## 6. Beyanlı sınırlar

* Ceket r2 meta'sının `ureten_betik` alanı r18 taban betiğini gösterecektir (betik içi literal);
  gerçek üretici `scratch/anka_t0106_ceket_r2_build.py` + override listesi ayrı beyan dosyasına yazılır.
* TEK DEĞİŞKEN beyanı **koşum girdileri** düzeyindedir: taban a2'nin kendisi r2-derleme
  külliyatla eğitilmiştir (T-0104); bu koşum yeni bir külliyat arzı VAAT ETMEZ (plan §Açık iş).