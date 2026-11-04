# P2.9 FAZ-1 ARŞİV UYGULAMASI — 20 dosya / 6,616 GiB → `data/_archive/p29-faz1-20261104/` (2026-11-04)

> **HÜKÜM ÖZETİ: FAZ-1 UYGULANDI VE DOĞRULANDI.** 20/20 dosya taşındı (C2×13 + C4×1 + C3×6), 0 atlandı, 0 bit kayıp (sha önce==sonra), canlı soyağacının sha'sı değişmedi, tam test paketi **542 passed / 1 xfailed / 0 failed** (arşiv-durum ve geri-taşınmış-durum A/B: birebir aynı). **Commit yapılmadı, silme olmadı — yalnız `mkdir`+`mv`.**

## İLAN (uygulama öncesi beyan)

- **Damga:** uygulama 2026-11-04 18:37 +03; rapor 19:42 +03.
- **Onay:** operatör (Kullanıcı) talimatı — "P2.9 faz-1 uygulanacak; C2+C3+C4, 20 dosya/~6,6 GiB".
- **Kaynak politika:** `data/eval/p29_checkpoint_envanteri_2026-11-04.md` Bölüm 7 FAZ-1 `mv` listesi (taslak, bu adımda onayla uygulandı).
- **Sert kurallar (uyuldu):** git commit YOK; silme YOK; yalnız `mkdir -p` + `mv` (geri alınabilir); canlı soyağacı (`anka_taban_pure/marangoz/bahcivan/router_v2`) DOKUNULMAZ.
- **Hedef dizin:** `data/_archive/p29-faz1-20261104` (operatör talimatındaki ad; politika taslağındaki `p29_kronolojik_eskiler` önerisi yerine talimat esas alındı).
- **Kapsam DIŞI (bu görevde yok):** C1 (`a1r/a2/base_v2`), B2 (`v3r/v3r2`), B3 (`v3.opt`), yinelenik silme önerileri (faz-3, ayrı onay).
- **Yöntem:** (1) listeden 20 ad teyidi; (2) dosya başına tam-repo bağımlılık taraması; (3) taşınma öncesi manifest + sha256 kaydı; (4) `mv` + sha birebir doğrulama; (5) sayım + canlı-5 sha + tam pytest paketi; (6) bu rapor.

## HÜKÜM (uygulama ve doğrulama kararları)

### H1. Bağımlılık taraması — ATLANAN: 0, TAŞINAN: 20/20

Her faz-1 dosyası için `grep -rn -F "<ad>" --include='*.py' .` (tam repo) koşuldu. Sonuç: **eşleşmeler yalnız `scripts/` içindeki tarihsel üretici/kanıt betiklerinde** — `train_anka_b1_5.py`, `train_t0196_arz_onarim.py`, `train_t0197_uretim_saf.py`, `train_t0198_hiza.py`, `train_t0200/0201/0203/0204`, `train_t0209_taban_pure.py` (init: t0201_b05), `train_t0207_bahcivan.py` (init: t0203_polish), `evaluate_anka_b1_5.py`, `t0199_taban_zarar_tanisi.py`, `t0200_mutasyon_kanit.py`, `t0196/0197/0198/0200/0202/0203/0204_faz3_*`, `tani_t0193/t0194/t0206`, `dogrulama_t0154…0172` (router v1 dönemi). Bunlar politika §8.2'nin önceden kabul ettiği tarihsel kanıt araçlarıdır; **canlı/varsayılan referans YOK**.

- **Canlı gateway hattı kanıtı:** `scripts/baslat_canli_gateway.py` + `src/gateway/agent_gateway.py` default'ları yalnız `anka_taban_pure.pt`, `anka_router_v2.pt` (terfi-sonrası: `DEFAULT_ROUTER_PATH=data/anka_router_v3.pt`, `agent_gateway.py:117`; + `anka_legal_replay.pt` uzmanı) ve `anka_marangoz.pt`/`anka_bahcivan.pt` — **hiçbiri faz-1 listesinde değil**.
- **tests/ kanıtı:** hiçbir test faz-1 dosyasına referans vermiyor; `test_moe_gateway_routing.py` yalnız canlı-4'ü kullanır.
- **test_model.py istisnası TETİKLENMEDİ:** `test_model.py:28` default hâlâ `data/anka_base_v2.pt` (C1 — faz-1 listesinde DEĞİL; L39-41 fallback `a1r/a2` da C1). Ön-onaylı "test_model.py:28 → `data/anka_taban_pure.pt`" düzeltmesi **uygulanmadı**; o düzeltme C1'i kapsayan faz-2a'nın önkoşuludur. Hiçbir kod değişikliği yapılmadı.
- Faz-1 dosyalarından hiç .py referansı almayanlar: `anka_b1_5_epoch1/2/3.pt`, `anka_b1_5_arz_epoch1/2.pt`, `anka_b1_5_hiza_epoch1/2.pt`, `anka_b1_5_usaf_epoch1/2.pt`, `anka_t0201_b20.pt`.

### H2. Taşıma öncesi doğrulama

- 20/20 sha256, envanter raporunun çıpalarıyla **birebir** eşleşti → dosyalar 17:08 envanterinden beri değişmemiş (`BEKLENEN_SHA256_envanter.txt` ile diff: temiz).
- Boyutlar envanter tablosuyla **bayt-bayt** aynı; toplam **7.103.354.683 bayt = 6,616 GiB** (politika Faz-1 hacim tahminiyle birebir).

### H3. Taşıma ve bütünlük

- 20 `mv` (data/ → `data/_archive/p29-faz1-20261104/`) — mv çıktısı 20/20; arşiv .pt sayısı 20.
- **Sha önce==sonra: 20/20 birebir** (`SHA256_ONCE.txt`/`SHA256_SONRA.txt` karşılaştırması — taşınmada 0 bit kayıp). A/B geri-taşıma turundan sonra yeniden arşivlemede de 20/20 aynı.
- `shasum -a 256 -c SHA256_MANIFEST.txt`: **20/20 OK** (arşiv dizininden, iki ayrı zamanda).

### H4. Canlı soyağacı DEĞİŞMEDİ

| Dosya | sha256 (öncesi == sonrası == final; envanter çıpasıyla birebir) |
|---|---|
| `anka_taban_pure.pt` | `5479a2f35a88975c…` |
| `anka_marangoz.pt` | `d1eae59457488a17…` |
| `anka_bahcivan.pt` | `2d5e5b21893bde21…` |
| `anka_router_v2.pt` | `74b9c808425cac41…` |
| `anka_taban_v3.pt` | `eb2335b8ca1ee596…` |

(Üç kez ölçüldü: taşıma öncesi, taşıma sonrası, final 19:41 — üçü de aynı.)

### H5. Tam test paketi — A/B kanıtıyla YEŞİL

`venv/bin/python -m pytest tests/ -q`, aynı stabilize test setiyle iki durumda da koşuldu:

| Durum | Sonuç |
|---|---|
| **Arşiv-durum** (20 dosya taşınmış; 19:19–19:40) | **542 passed, 1 xfailed, 0 failed — EXIT=0** (1262,64s) |
| **Geri-taşınmış durum** (64 dosya; p26 oturumunun koşusu, 18:52–19:15) | **542 passed, 1 xfailed, 0 failed — EXIT=0** (1389,23s) |

→ **Birebir aynı sonuç: taşınan 20 dosyaya duyarlı test yok; paket her iki durumda tam yeşil.** ("32−N beklenir" doğrulaması: envanter-32 bazında kökte 12 kaldı ✓.)

**Düzeltme kaydı:** ilk koşumda (18:37–18:43) görülen 8 FAILED (`test_gateway_v3_terfi`×1, `test_merak_trigger_signal`×7) taşımadan DEĞİL, eşzamanlı terfi oturumunun kod/test düzenlemeleriyle collection yarışından kaynaklanmıştı — kanıt: (a) o iki test dosyası izole koşuda 14/14 geçer; (b) `agent_gateway.py` 18:38:50, `baslat_canli_gateway.py` 18:39:26, WIP testleri 18:38–18:43'te (koşumla eşzamanlı) yazıldı; (c) stabilize setle her iki durumda 542/542 yeşil.

### H6. Eşzamanlı oturum bulguları (kapsam dışı, şeffaflık için)

Bu görev sırasında başka bir oturum repoda aktif çalıştı (P2.5/P2.6 terfi hattı):
- **32 yeni `data/*.pt`** (17:36–17:55): `anka_legal.pt`+e1..e3, `anka_legal_p26lr.pt`+e1..e3, `anka_legal_replay.pt`+e1..e9, `anka_legal_tani12e.pt`+e1..e12, `anka_router_v3.pt`; + **2 dosya daha koşum sırasında** (`anka_legal_v3.pt_e1/e2.pt`, 19:03/19:17 — aktif eğitim).
- **Gateway terfisi:** router default `router_v2` → `router_v3` + legal uzmanı (`anka_legal_replay.pt`) — terfi edilen dosyaların hiçbiri faz-1 kapsamında değil.
- Bu yüzden fiziksel kök sayısı **46** = 12 (envanter-kalan) + 34 (yeni); envanter-32 bazında beklenen **32−20=12** ✓. Bu dosyalara hiç dokunulmadı.

## RAPOR (veri)

### R1. Taşınan 20 dosya (tam liste; sha öncesi==sonrası; tam sha `SHA256_MANIFEST.txt`'te)

| # | Dosya | Sınıf | Bayt | GiB | sha256 (öneki) |
|---:|---|:--:|---:|---:|---|
| 1 | `anka_b1_5_epoch1.pt` | C2 | 373.735.672 | 0,348 | `8b6640213dead228…` |
| 2 | `anka_b1_5_epoch2.pt` | C2 | 373.735.672 | 0,348 | `03dd960c476d2d34…` |
| 3 | `anka_b1_5_epoch3.pt` | C2 | 373.735.672 | 0,348 | `e5eb114e5bd71d58…` |
| 4 | `anka_b1_5_best.pt` | C2 | 373.735.672 | 0,348 | `e5eb114e5bd71d58…` |
| 5 | `anka_b1_5_arz_epoch1.pt` | C2 | 373.736.100 | 0,348 | `4ca5e3b0f214dc2a…` |
| 6 | `anka_b1_5_arz_epoch2.pt` | C2 | 373.736.100 | 0,348 | `80b0fd9a5a60de90…` |
| 7 | `anka_b1_5_arz.pt` | C2 | 373.736.100 | 0,348 | `80b0fd9a5a60de90…` |
| 8 | `anka_b1_5_usaf_epoch1.pt` | C2 | 373.736.207 | 0,348 | `87f3c02e9cf90230…` |
| 9 | `anka_b1_5_usaf_epoch2.pt` | C2 | 373.736.207 | 0,348 | `8f846023f2fb2374…` |
| 10 | `anka_b1_5_usaf.pt` | C2 | 373.736.207 | 0,348 | `8f846023f2fb2374…` |
| 11 | `anka_b1_5_hiza_epoch1.pt` | C2 | 373.736.207 | 0,348 | `977228152dfec324…` |
| 12 | `anka_b1_5_hiza_epoch2.pt` | C2 | 373.736.207 | 0,348 | `05d8e4cf55a2e27e…` |
| 13 | `anka_b1_5_hiza.pt` | C2 | 373.736.207 | 0,348 | `05d8e4cf55a2e27e…` |
| 14 | `anka_router.pt` | C4 | 2.372.421 | 0,002 | `44a46d89f3d4260c…` |
| 15 | `anka_t0200_epoch1.pt` | C3 | 373.735.779 | 0,348 | `be5c1fb47ceb5187…` |
| 16 | `anka_t0200.pt` | C3 | 373.735.779 | 0,348 | `be5c1fb47ceb5187…` |
| 17 | `anka_t0201_b05.pt` | C3 | 373.735.565 | 0,348 | `52fa2f80c2ddcced…` |
| 18 | `anka_t0201_b20.pt` | C3 | 373.735.565 | 0,348 | `b124ab8452159164…` |
| 19 | `anka_t0203_polish.pt` | C3 | 373.735.779 | 0,348 | `4f4183caec729f61…` |
| 20 | `anka_t0204_pure.pt` | C3 | 373.735.565 | 0,348 | `f7dca8b48d6197a6…` |
| | **TOPLAM** | | **7.103.354.683** | **6,616** | |

Not: envanter §9a'daki 5 bayt-özdeş çift taşınan setin içinde (bilgi kaybı yok; silme adayı DEĞİL bu görevde — faz-3 ayrı onay).

### R2. Atlananlar

**Yok — 0/20.** Gerekçe gerektiren çıkmadı: canlı/varsayılan referansı olan faz-1 dosyası bulunmadı. `test_model.py:28` düzeltmesi (önkoşul olarak ön-onaylıydı) base_v2 C1/faz-2a kapsamında olduğundan uygulanmadı; `test_model.py`'ye bu görevde hiç dokunulmadı.

### R3. Kalan `data/` kök envanteri (faz-1 sonrası)

- **Envanter-12 (4.492.115.064 bayt = 4,184 GiB):** `anka_taban_pure.pt`, `anka_marangoz.pt`, `anka_bahcivan.pt`, `anka_router_v2.pt` (A — dokunulmaz); `anka_taban_v3.pt`, `anka_taban_v3r3.pt` (B1 — koru); `anka_taban_v3r.pt`, `anka_taban_v3r2.pt` (B2 — faz-2b adayı); `anka_taban_v3.pt.opt.pt` (B3 — faz-2c adayı); `anka_a1r.pt`, `anka_a2.pt`, `anka_base_v2.pt` (C1 — faz-2a adayı, önkoşullu).
- **Eşzamanlı oturumun yeni dosyaları (34):** 32 legal ailesi + router_v3 (yukarıda H6) + `anka_legal_v3.pt_e1/e2.pt` — bu görevin kapsamı dışında, envantere dahil değil.
- **Sayım:** arşiv 20 ✓ | kök 46 = 12+34 ✓ | envanter-32 bazında 32−20=12 ✓.

### R4. Disk

- `du -sh data/`: **23G → 23G** (değişmez — `mv` aynı disk biriminde). Arşiv dizini: 6,6G.
- **Gerçek disk kazancı: 0 bayt** (taşımadır; silme faz-3'te ayrı operatör onayı ister — politika §5/§6 notu).
- `data/` **kökü** `.pt` hacmi: 21,592 GiB (64 dosya, taşıma öncesi) → **14,976 GiB** (44 dosya, taşıma sonrası ölçümü) = **−6,616 GiB**; eşzamanlı oturumun +2 yeni dosyasıyla 19:41 itibarıyla 15,672 GiB (46 dosya).

### R5. Kayıt dosyaları

- Arşiv dizini (`data/_archive/p29-faz1-20261104/`): `MV_LISTESI.txt`, `BEKLENEN_SHA256_envanter.txt` (envanter çıpaları), `SHA256_ONCE.txt` / `SHA256_SONRA.txt`, `SHA256_MANIFEST.txt` (`shasum -c` ile 20/20 OK), `TASINMA_ONCE_boyut.txt`, `ARSIV_NOTU.md` (geri alma komutu dahil).
- Suite kanıtları: `~/.hermes/cache/scratch/p29_faz1_arsiv_durum_suite.log` (arşiv-durum, 542/0, EXIT=0) ve `~/.hermes/cache/scratch/p26_full_suite.log` (geri-taşınmış durum, 542/0, EXIT=0).
- Kaynak envanter: `data/eval/p29_checkpoint_envanteri_2026-11-04.md`.

### R6. Sonraki adımlar (bilgi; bu görevin kapsamı dışında)

- **Faz-2a (C1, önkoşullu):** `test_model.py:28` default (`base_v2`) ve L39-41 fallback (`a1r/a2`) güncellenmeden C1 taşınmaz — ayrı kod onayı gerekir.
- **Faz-2b (B2):** P2.7 pencere kararı sonrası `v3r/v3r2`; **Faz-2c (B3):** v3-devam pretrain planlanmıyorsa `v3.opt`.
- **Faz-3 (AYRI onay):** 5 bayt-özdeş çiftin silmesi (1,740 GiB) ve `v3.opt` (0,696 GiB).

---
*P2.9 faz-1 uygulaması — 2026-11-04 19:42 +03, ~/Git/tr_llm. Üretim: elle yürütülen adımlar + programatik ölçüm (shasum/stat/du/pytest); tüm sayılar araç çıktısından alındı. Commit yapılmadı.*