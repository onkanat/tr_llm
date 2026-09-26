# G6a (T-0137) İLAN — Bahçıvan Dikeyi — 2026-09-26 (claude devralma)

Bu dosya İLANDIR, RAPOR DEĞİLDİR (ILAN ≠ RAPOR — G4 kusur sınıfı). Ölçümler koşum
ÖNCESİ ilanlanır; hükümler BETİKTEN yazılır, elle sayı yoktur. Taban: `data/anka_base_v2.pt`
(`d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`, MUHÜRLÜ 2.0-sealed —
T-0136; mimari 6/6/768/RoPE/tying-false DOKUNULMAZ). Şartname: ceket_tasarim_2026-09-26.md §8.

## İLAN-1 — TABAN-ÇIPA (koşum başlıyor)

- **Amaç:** eğitilmemiş muhürlü tabanın kanonik kapta beklendiği gibi KÖR olduğu kanıtı
  (taban-çıpa deseni; eğitilmemiş model ROUGE 0,0000 — T-0036/T-0106 çıpası).
- **Koşum:** kanonik kap `scripts/evaluate_carpenter_anka.py` DOKUNULMAZ; `--model
  data/anka_base_v2.pt --output scratch/t0137/taban_cipa_sonda.json` (kanonik varsayılanlar:
  vocab `data/rebuild/vocab_anka_r1_33114.json` çıpa zinciri, lexicon, heldout, n=100,
  seed=42, device=mps, max-new=128). Sandbox DIŞI (MPS gizli). Tek süreç (lsof temiz).
- **AYRAÇ (koşum-öncesi ilan):** sonda JSON ROUGE-L alanı betikten okunur;
  `ROUGE-L < 0,005` → **TABAN_CIPA_GECTI** · `≥ 0,005` → **TABAN_CIPA_DUSTU**
  (koşum durmaz ama fine-tune BAŞLAMAZ — kök neden ölçülür). Eşik TAVAN/ESIK
  sabitlerine (0,3221 / 0,9509) DOKUNMAZ.
- **Kabul koşusu artefaktı notu:** çıktı `scratch/` altında; data/eval'e yazım yok.
- **Düzeltme (ölçüm ÖNCESİ, 26 Eyl 11:05Z):** ilk koşum `--ceket-ekseni` olmadan yapıldı
  ⇒ betik `metrik["olculdu"]=False`, `hukum={}`, ROUGE ÜLÇÜLMEDİ (betikten okundu:
  üretim ROUGE yalnız `--ceket-ekseni` dalında üretilir — `evaluate_carpenter_anka.py`
  K2 bölümü). İlan İLK halinde "kanonik varsayılanlar" yazmıştı — kanonik TABAN-ÇIPA
  koşumu `--ceket-ekseni` bayrağıyla tanımlıdır (T-0036/T-0106 çıpa koşumlarıyla aynı).
  İlk sonda JSON (b891afc02a1ff55e29cd8ea3bcc690676f31abd255e6f3d3fc985f5e7d547820)
  ROUGE-siz olarak kanıta saklandı; koşum `--ceket-ekseni` ile TEKRARLANIYOR. Ayraç
  DEĞİŞMEDİ: `ROUGE-L < 0,005 → TABAN_CIPA_GECTI`.

## İLAN-2 — ARZ ÜRETİMİ PİLOT KAPISI (beklemede: OLLAMA meşkul; gemini kota dışında)

Arz üretimi aracı gemini-2.5-flash'tan (kota tükendi) yerel OLLAMA'ya geçiş adayıdır
(`192.168.1.14:11434`, operatör bildirdi: meşkul). Pilot koşum AŞAĞIDAKİ eşiklerle
yapılacaktır; pilot eşikten geçmeden arz külliyatına GİRİLMEZ (kayıt YOK, "üretti" ≠ "uygun"):

- **Örneklem:** pilot n=20 (alan: bahçıvanlık/bitki bakımı; D3-istisna kalıbı:
  soru-kalıp + "Ahşap uzmanı olarak cevapla." STİLİ UYARLANMIŞ sistem zarfı + cevap).
- **AYRAÇ (betik hükmü):**
  1. `kalip_koruma >= 18/20` (%90) — her örnek tanımlı soru-kalıbına + cevap-sonlandırma
     düzenine oturur; boşluk/tekrar/dağılım-dışı son yok.
  2. `kalip_cekimlilik`: soru-kalıp sayısı ≥ 5 farklı aile (T-0125 dersi: tek kalıp homojenliği
     kilit doğurur; ağırlık dengesi §5/G4 hükmü).
  3. Türkçe jeton dökümü/başka dil karışımı: 0 örnek (mutlak).
- Hüküm dalları: **PILOT_GECTI** (arz üretimi bu araçla açılır) · **PILOT_YETERSIZ**
  (kayıt; araç değişir veya arz ertelenir — KAYIT YAPILMAZ). Pilot sonucu RAPOR'da
  (ayrı dosya), ILAN burada tek kez.
- Gemini kotası dönerse karşılaştırmalı pilot (gemini vs OLLAMA aynı istemler) yapılabilir.

## İLAN-4 — BOŞ MODEL POZİTİF KONTROLÜ (operatör onayı 2026-09-26; FAZ-0 sonrası açıldı)

- **Amaç:** ölçütün ayırt ediciliği kanıtı — TABAN_CIPA_DUSTU hükmünün kök nedeni
  varsayım hatasıydı (base_v2 devralınmış); ROUGE ölçütünün kendisi boş modeli
  yakalamalıdır (taban-çıpa deseni: ölçüt için).
- **Boş model:** `KristalLM(33114, 768, blok 4096, 6 katman, 6 kafa)` rastgele ağırlık,
  tohum **42** (determinizm kuralı), `scratch/t0137/bos_model.pt` — checkpoint'i
  DEĞİŞMEZ, data/*.pt yazım YOK.
- **Koşum:** kanonik kap, `--model scratch/t0137/bos_model.pt --ceket-ekseni`,
  aynı varsayılanlar (n=100, seed 42, mps).
- **AYRAÇ (koşum-öncesi ilan):** `ROUGE-L < 0,005` → **POZITIF_KONTROL_GECTI**
  (ölçüt ayırt ediyor) · `≥ 0,005` → **POZITIF_KONTROL_DUSTU** (ölçüt körlemesine
  geçiyor — KÖK NEDEN ölçülür). Oracle kontrolü kimlik 1,0000 ister (koşumda
  fail-closed, betik `durdur`).

## İLAN-3 — FİNE-TUNE TASARIMI (arz derlenince; koşum ÖNCESİ ayrıca damgalanacak)

- Derleme: T-0134 `dataset_compiler` (D3-istisna, `tokenize_specialization_jsonl` — system
  prompt Bahçıvan zarfına uyarlanır) + `bin_dekod_dogrula` ZORUNLU (0 sapma kabul).
- Taban DONUK; LM bedeli kapısı **0,0230** (2× ROUGE SE) — **DÜZELTME (operatör onayı
  2026-09-26): çıpa base_v2'nin KENDİ ROUGE'sidir (TABAN_CIPA ölçümü 0,1392; "0,0000"
  çıpası devralınmış zincir tabanında geçersiz — taban-çıpa raporu
  bahcivan_g6a_tabancipa_sonuc_2026-09-26.md)**; tutarsızlık çıpa %5; `--rol-zarf`
  beyanlı (0,1342 zarflı / 0,1354 nötr çıpa); oversampling tavanlı (G4 hükmü: cevap-
  sonlandırma dengesi hedefi). Checkpoint `scratch/` altında; doğrulama sonrası silinir.
- MPS disiplini: tek eğitici; koşum-öncesi `lsof` kanıtı.

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ · tokenizer + `src/compiler/**`
DOKUNULMAZ · kanonik eval betiği DOKUNULMAZ · elle sayı YOK · `git add -A` YASAK ·
commit operatör kapısıdır.