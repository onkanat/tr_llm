# P2.9 — Checkpoint Envanteri ve Arşiv Politikası TASLAĞI (2026-11-04)

> **DURUM: ENVANTER + POLİTİKA TASLAĞI.** Hiçbir dosya taşınmadı, silinmedi, commit yapılmadı.
> Bölüm 8'deki `mv` listeleri **TASLAKTIR, KOŞULMAZ** — uygulama yalnız operatör onayıyla ayrı adımda.
> Kanıt: tüm sha256 değerleri bu envanter için **ölçüldü** (`shasum -a 256 data/*.pt`, 2026-11-04 17:08 +03, 34,9 sn, 32 dosya).

## 1. Özet

- **`data/*.pt`: 32 dosya** (31 model checkpoint + 1 optimizer yan dosyası) — toplam **11.595.469.747 bayt = 10,799 GiB**.
- Plan P2.9 satırındaki "27 kontrol noktası, 9,7 GB" = bugünün T3.4+P2.6 koşumlarından önceki durum: **27 dosya** sayımı doğrulandı ✓; ölçülen hacim o 27 dosya için **8,711 GiB (≈9,35 GB)** — plandaki "9,7 GB" bayt-düzeyinde doğrulanamıyor (muhtemelen `du` blok sayımı ya da plan yazımı anındaki geçici dosyalar). Bugün +5 dosya (v3, v3.opt, v3r, v3r2, v3r3) = +2,088 GiB.
- **Canlı soyağacı (A): 4 dosya / 1,046 GiB** — kod düzeyinde doğrulandı: `scripts/baslat_canli_gateway.py` varsayılanları (--model taban_pure, --router router_v2, bahcivan_path, marangoz_path) + `src/gateway/agent_gateway.py` (router_state_path default `data/anka_router_v2.pt`; model yolu T-0087 fail-closed).
- **5 bayt-özdeş yinelenik çift** bulundu (Bölüm 9a): 1.868.679.965 bayt (1,740 GiB) saf çift-kayıt.
- Betik sha-çıpa doğrulaması: arz/hiza/usaf/best/base_v2/taban_pure ölçülen hash'ler, `t0199_taban_zarar_tanisi.py` / `t0200_mutasyon_kanit.py` çıpaları ve CHANGELOG damgalarıyla **birebir eşleşti** → bu dosyalar 2026-09-29/30'dan beri değişmemiş.
- Kapsam: yalnız `data/*.pt`. Alt dizinlerdeki 11 adet `.pt` (b1_5_splits 6, eval 3, _archive 2) veri kümesi/önbek dosyasıdır — model checkpoint'i değildir, kapsam dışı.

## 2. Yöntem

```bash
ls -la data/*.pt                          # 32 girdi, boyut+mtime
shasum -a 256 data/*.pt                  # 34,9 sn; tam liste bu raporun tablosunda
grep -n 'anka_' CHANGELOG.md             # 426 satırlık CHANGELOG'da isim eşleşmeleri
grep -rn --include='*.py' 'anka_...' scripts/ src/ *.py   # 214 eşleşme → üretici betikler
```

CHANGELOG kapsaması: `anka_base_v2` (T-0162), `anka_router` (T-0154..172), ceket hattı + `router_v2` (T-0205..0210), v3 ailesi (T3.4/P2.6). **a1r, a2, b1_5*, t02xx dosya adlarıyla CHANGELOG'da YOK** — attribution üretici betiklerden (`scripts/train_*.py` docstring'leri) ve `data/eval/` hüküm-kayıtlarından yapıldı; b1.5 ailesi CHANGELOG'da `kristal_b1_5_best.pt` eski adıyla geçer.

## 3. Tam envanter tablosu (kronolojik)

Sınıf gösterimi: **A**=canlı soyağacı, **B1/B2/B3**=bugünün ailesi (T3.4+P2.6), **C1**=eski tabanlar, **C2**=b1.5 ailesi, **C3**=T-02xx merdiveni, **C4**=eski router. Kayıtlar `data/eval/` görelidir.

| # | Dosya | Boyut (bayt) | mtime (+03) | sha256 | Sınıf | Üretici betik / komut | İlgili hüküm-kayıt |
|---:|---|---:|---|---|:--:|---|---|
| 1 | `anka_a1r.pt` | 373.735.137 | 2026-09-20 02:55 | `b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293` | **C1** | Pre-külliyat ön-eğitim tabanı (T-0105/T-0106 dönemi); korunan üretici betik yok — dönemin train.py koşumu | t0199_m1_a1r_referans.json; anka_t0106_duman_a1r_n20.json |
| 2 | `anka_a2.pt` | 378.634.998 | 2026-09-24 05:33 | `e2352cd3bbf3e88f0d4d2e073d3aa8cd31317073b006ef676e945d4e9c964157` | **C1** | train.py --save-path (T-0105/T-0106 a2-ceket dönemi); r2-sözlük kalıntısı (maks id 33910) — hüküm: KULLANILMAZ | anka_t0105_a2_yetenek_2026-09-24.*; anka_t0106_fazB_a2_ceket_2026-09-24.* |
| 3 | `anka_base_v2.pt` | 373.735.244 | 2026-09-26 11:53 | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` | **C1** | kristal_model.pt zincirinin T-0162 yeniden adlandırması; eski canlı-gateway tabanı (sha çıpası d0f415f3…. CHANGELOG L116) | t0199_m1_taban.json; t0199/t0200 betik sha-çıpaları; CHANGELOG T-0162 |
| 4 | `anka_router.pt` | 2.372.421 | 2026-09-28 19:19 | `44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a` | **C4** | dogrulama_t0154_router_egitim.py — T-0154 3-sınıflı router (t0156/t0172 yeniden-yazımları; son sha 44a46d89…. CHANGELOG damga 7eafe53) | t0154_router_egitim_hukum_2026-09-28.json; t0154_router_egitim_rapor_2026-09-28.md |
| 5 | `anka_b1_5_epoch1.pt` | 373.735.672 | 2026-09-29 12:21 | `8b6640213dead228dc63da4a0abfe1cc50586f4c9c189e52d12677cacc69963d` | **C2** | train_anka_b1_5.py — T-0191 koşumu epok-1 ara kaydı | t0191_training_history.json |
| 6 | `anka_b1_5_epoch2.pt` | 373.735.672 | 2026-09-29 12:59 | `03dd960c476d2d347e0aa24a3fab6d23d0ebfe07f2f2cf81c9c895dc576e652a` | **C2** | train_anka_b1_5.py — T-0191 epok-2 ara kaydı | t0191_training_history.json |
| 7 | `anka_b1_5_epoch3.pt` | 373.735.672 | 2026-09-29 13:36 | `e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7` | **C2** | train_anka_b1_5.py — T-0191 epok-3 ara kaydı (= best. bayt-özdeş) | t0191_training_history.json |
| 8 | `anka_b1_5_best.pt` | 373.735.672 | 2026-09-29 13:36 | `e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7` | **C2** | train_anka_b1_5.py (L194 best_path; reçete: train_step_b1_5_rigorous.py. 3 epok; CHANGELOG'da 'kristal_b1_5_best.pt' adıyla) | t0190_b15_regen_hukum; t0191_b15_egitim_hukum_2026-09-29.json; t0199_m1_b1_5_best.json |
| 9 | `anka_b1_5_arz_epoch1.pt` | 373.736.100 | 2026-09-29 16:04 | `4ca5e3b0f214dc2a8a692d5bb2452aeb380fc80963ba077ed19255dfe6ead204` | **C2** | train_t0196_arz_onarim.py — T-0196 epok-1 ara kaydı | t0199_m1_b1_5_arz.json |
| 10 | `anka_b1_5_arz_epoch2.pt` | 373.736.100 | 2026-09-29 16:18 | `80b0fd9a5a60de90536ad4d6d3a3f8eb1de16d1474501a0eac669d2458f106fa` | **C2** | train_t0196_arz_onarim.py — T-0196 epok-2 ara kaydı (= bitiş dosyası. bayt-özdeş) | t0199_m1_b1_5_arz.json |
| 11 | `anka_b1_5_arz.pt` | 373.736.100 | 2026-09-29 16:18 | `80b0fd9a5a60de90536ad4d6d3a3f8eb1de16d1474501a0eac669d2458f106fa` | **C2** | train_t0196_arz_onarim.py — T-0196 arz-onarımı (best → arz.pt; halef: b1_5_best. YALNIZ-OKU) | t0195_arz_hukum_2026-09-29.json; t0199_m1_b1_5_arz.json |
| 12 | `anka_b1_5_usaf_epoch1.pt` | 373.736.207 | 2026-09-29 18:53 | `87f3c02e9cf902307f3ae1545a2a95b5802b57ac5ed639aa8c7222bbf71164d7` | **C2** | train_t0197_uretim_saf.py — T-0197 epok-1 ara kaydı | t0199_m1_b1_5_usaf.json |
| 13 | `anka_b1_5_usaf_epoch2.pt` | 373.736.207 | 2026-09-29 19:06 | `8f846023f2fb2374dd411f674b199f373198c108cd961c836c44e094948b2c10` | **C2** | train_t0197_uretim_saf.py — T-0197 epok-2 ara kaydı (= bitiş. bayt-özdeş) | t0199_m1_b1_5_usaf.json |
| 14 | `anka_b1_5_usaf.pt` | 373.736.207 | 2026-09-29 19:06 | `8f846023f2fb2374dd411f674b199f373198c108cd961c836c44e094948b2c10` | **C2** | train_t0197_uretim_saf.py — T-0197 üretim-saf (best → usaf.pt) | t0187_b15_sizinti_katmani_tani_2026-09-29.md; t0199_m1_b1_5_usaf.json |
| 15 | `anka_b1_5_hiza_epoch1.pt` | 373.736.207 | 2026-09-29 22:47 | `977228152dfec3241907bd2d468b73a1a310a668997eb8a49cfa8fff877824a5` | **C2** | train_t0198_hiza.py — T-0198 epok-1 ara kaydı | t0199_m1_b1_5_hiza.json |
| 16 | `anka_b1_5_hiza_epoch2.pt` | 373.736.207 | 2026-09-29 23:01 | `05d8e4cf55a2e27e46eadab6b93f50b2cd90202b6f17e2f57958d4ac4d5727df` | **C2** | train_t0198_hiza.py — T-0198 epok-2 ara kaydı (= bitiş. bayt-özdeş) | t0199_m1_b1_5_hiza.json |
| 17 | `anka_b1_5_hiza.pt` | 373.736.207 | 2026-09-29 23:01 | `05d8e4cf55a2e27e46eadab6b93f50b2cd90202b6f17e2f57958d4ac4d5727df` | **C2** | train_t0198_hiza.py — T-0198 hıza-onarımı (best → hiza.pt) | t0193_cikti_hiza_hukum_2026-09-29.json; t0199_m1_b1_5_hiza.json |
| 18 | `anka_t0200_epoch1.pt` | 373.735.779 | 2026-09-30 12:19 | `be5c1fb47ceb5187eae51d4ef60b1d600a4b7fea721a2978ff9e99f6abd23d83` | **C3** | train_t0200_optimum_zarar.py — tek-epok koşumun ara kaydı (= bitiş. bayt-özdeş) | t0200_training_history.json |
| 19 | `anka_t0200.pt` | 373.735.779 | 2026-09-30 12:19 | `be5c1fb47ceb5187eae51d4ef60b1d600a4b7fea721a2978ff9e99f6abd23d83` | **C3** | train_t0200_optimum_zarar.py — T-0200 optimum-zarar onarımı (init: b1_5_best; öğretmen: base_v2) | t0200_egitim_hukum_2026-09-30.json; t0200_mutasyon_hukum_2026-09-30.json; t0200_training_history.json |
| 20 | `anka_t0201_b05.pt` | 373.735.565 | 2026-10-01 00:55 | `52fa2f80c2ddcced0284f10142b8bc3f4a7033dc663b6f600bc2a5929c483430` | **C3** | train_t0201_beta_gerisi.py — T-0201 β=0.5 geri-ölçekleme deneyi | t0201_b05_history.json; t0202_m1_b05.json; t0202_faz3_hukum_2026-10-01.json |
| 21 | `anka_t0201_b20.pt` | 373.735.565 | 2026-10-01 14:35 | `b124ab8452159164f350d60c0d76d8ceb109cf07144fafcd14cc7470be07f563` | **C3** | train_t0201_beta_gerisi.py — T-0201 β=2.0 geri-ölçekleme deneyi | t0201_b20_history.json; t0201_geri_egri.json; t0201_egitim_hukum_2026-09-30.json |
| 22 | `anka_t0203_polish.pt` | 373.735.779 | 2026-10-01 16:47 | `4f4183caec729f61eea4489c4faf2f14dbb7ed7b8e0843dabeeba5e97d862b2a` | **C3** | train_t0203_polish.py — T-0203 (init: t0201_b05; öğretmen: base_v2) | t0203_egitim_hukum_2026-10-01.json; t0203_m1_polish.json; t0206_net_tutarsizlik_raporu.json (üzerine tanı) |
| 23 | `anka_t0204_pure.pt` | 373.735.565 | 2026-10-01 17:31 | `f7dca8b48d6197a65beeb2f99a39307f3cf44d7fee4d9ef1dede1c3882b8744f` | **C3** | train_t0204_pure.py — T-0204 pure SFT polish (init: t0203_polish) | t0204_egitim_hukum_2026-10-01.json; t0204_m1_pure.json; t0204_faz3_hukum_2026-10-01.json |
| 24 | `anka_bahcivan.pt` | 373.735.351 | 2026-10-01 18:46 | `2d5e5b21893bde21fe5b9d1d9de913889c9918f1f9d3c4a1dba1933b628f01f1` | **A** | train_t0207_bahcivan.py — T-0205..0207 bahçıvan ceketi (init: t0203_polish; öğretmen: base_v2) | t0207_egitim_hukum_2026-10-01.json; t0207_training_history.json |
| 25 | `anka_taban_pure.pt` | 373.735.565 | 2026-10-02 00:38 | `5479a2f35a88975cae3cd8e66e4aef0bb7b09bea5274828709fe2c0e1d263360` | **A** | train_t0209_taban_pure.py — T-0209 saf taban (init: t0201_b05; öğretmen: base_v2; 4:1 SFT:pencere. KL β=0.2 beyanı — B1: KL dalı ölü koddu) | t0209_taban_pure_hukum_2026-10-02.json; t0209_taban_pure_history.json; p23_otonom_retrain_hukum_2026-11-04.json (sha değişmezlik) |
| 26 | `anka_marangoz.pt` | 373.735.351 | 2026-10-02 01:10 | `d1eae59457488a17ee07107d28b8b78050015c69cd49a3327491c3fec9ec4523` | **A** | train_t0209_marangoz.py — T-0209 marangoz ceketi (init: taban_pure; öğretmen: base_v2) | t0209_marangoz_hukum_2026-10-02.json; t0209_marangoz_history.json |
| 27 | `anka_router_v2.pt` | 2.373.490 | 2026-10-15 21:16 | `74b9c808425cac4161b01641d4ec3aec3472ded0a65b4fd69e211b45b72e2ab2` | **A** | train_t0208_router_v2.py (T-0208) → train_router_v2_pure.py (T-0210 taban_pure-hizalı nihai koşum. 2026-10-15) — 5 sınıflı | t0210_router_v2_pure_hukum_2026-10-15.json; t0208_router_v2_hukum_2026-10-01.json (ara kayıt) |
| 28 | `anka_taban_v3.pt` | 373.735.672 | 2026-11-04 14:45 | `eb2335b8ca1ee5969f42a96cd687b2ae48d5c53d7fb6682a76d5f326f33b020c` | **B1** | train.py — T3.4 devam-ön-eğitim (komut t34_egitim_ozet'te birebir; --base-model taban_pure; kulliyat_v2.bin. 1 epok. 12.9 sa) | t34_egitim_ozet_2026-11-04.md; t34_v3_degerlendirme_2026-11-04.md |
| 29 | `anka_taban_v3.pt.opt.pt` | 747.486.705 | 2026-11-04 14:45 | `9058bf59413479528a39a791a16d1f04b00f813a5377345ba5d3d0596f7f960e` | **B3** | train.py --save-optimizer (train.py:46: save_path+'.opt.pt') — T3.4'ün AdamW son-durumu; YALNIZ anka_taban_v3.pt'ye ait | t34_egitim_ozet_2026-11-04.md (--save-optimizer bayrağı) |
| 30 | `anka_taban_v3r.pt` | 373.735.779 | 2026-11-04 15:59 | `2ad5b24013f79969eedb9131cffc1917b103d6b37e1b600037e0d1eab7b3f791` | **B2** | train_p26_sft_replay.py — P2.6 tur-1 (init+KL öğretmen: v3; T-0209 kalıbı. GERÇEK KL β=0.2) | p26_replay_hukum_2026-11-04.json; p26_kapı_olcum_2026-11-04.md; p26_replay_history.json |
| 31 | `anka_taban_v3r2.pt` | 373.735.886 | 2026-11-04 16:17 | `2e0e77ccc4bdd0cc432819d841aee12a81ce10d1cdf4decd1f598b0118b0dcd7` | **B2** | train_p26_sft_replay.py — P2.6 tur-2 | p26_replay_hukum_2026-11-04.json; p26_tur2_kapi_olcum_2026-11-04.md |
| 32 | `anka_taban_v3r3.pt` | 373.735.886 | 2026-11-04 16:46 | `40c9459ac3f4693d2bbb915f80b888f0763dd9df19cd38b15a3ea1fb6dfb4a49` | **B1** | train_p26_sft_replay.py — P2.6 tur-3 (pure_base_sft_v2.jsonl veri-zenginleştirme ile) | p26_replay_hukum_2026-11-04.json; p26_tur3_kapi_olcum_2026-11-04.md |

## 4. Canlı soyağacı sınıflandırması

### 4a. ZORUNLU — DOKUNULMAZ (A, 4 dosya)

| Dosya | Rol | Kanıt |
|---|---|---|
| `anka_taban_pure.pt` | ceket tabanı (T-0209) | sha `5479a2f3…` = p23 hükümündeki değişmezlik çıpası; gateway --model default'u |
| `anka_marangoz.pt` | MoE uzmanı (carpenter) | `baslat_canli_gateway.py:76` |
| `anka_bahcivan.pt` | MoE uzmanı (gardener) | `baslat_canli_gateway.py:72` |
| `anka_router_v2.pt` | 5-sınıflı router (T-0210 nihai) | gateway --router + `agent_gateway.py:109` default'u |

### 4b. Bugünün ailesi (T3.4 + P2.6) — hükümleri AÇIK, ceket hattı `taban_pure` üzerinde KALDI

Ölçülmüş durum (hüküm dosyaları + CHANGELOG):

| Dosya | Ölçüm | Hüküm |
|---|---|---|
| `anka_taban_v3.pt` | kulliyat_v2 val CE 3,8205→**3,1936** (PPL 45,6→24,4); masked SFT CE 2,3049→**3,4789**; pedagoji paketi güveni 0,9998→0,0726 | **ceket tabanı DEĞİL** (t34_v3_degerlendirme) |
| `anka_taban_v3r.pt` | tur-1: masked SFT CE 3,4789→**2,7380**; pencere-CE 3,2118 | 0/4 kapı; ara basamak |
| `anka_taban_v3r2.pt` | tur-2: masked SFT CE →**2,5848**; pencere-CE 3,2181 | 0/4 kapı; ara basamak |
| `anka_taban_v3r3.pt` | tur-3 (+265 zenginleştirilmiş kayıt): masked SFT CE →**2,4827**; UNK %3,00→**%2,33** (ilk eşik-altı); TENSE ilk kez top-5 | **1/4 kapı → v3r3 ceket tabanı DEĞİL** (p26_tur3_kapi_olcum); merdiven kapanış hızı −0,74→−0,15→−0,10 nat/tur → protokolle durduruldu |

**Görüşüm (politika önerisi):**

1. **`anka_taban_v3.pt` KORUNMALI (B1).** P2.6 hükmü onu ceket tabanı yapmadı — doğru; ama bu onu değersiz kılmıyor: (a) kulliyat_v2 çağında **tek devam-ön-eğitim artefaktı** ve repodaki en iyi düz-metin tabanı; (b) P2.6 merdiveninin **KL öğretmeni** (`train_p26_sft_replay.py --init-model` default'u) — merdiven yeniden açılırsa şart; (c) olası gelecek hattın ("SFT-replay ile v3'ü ceket tabanı yapma") tek başlangıç noktası.
2. **`anka_taban_v3r3.pt` KORUNMALI (B1, kısa vade).** Merdivenin en iyi/son basamağı ve veri-zenginleştirme deneyinin kanıt artefaktı; P2.6 protokolü "turların artefaktı korunur" der. Yeniden üretimi `pure_base_sft_v2.jsonl` + betikle mümkün ama pahalı (her tur ~ölçümlü koşum).
3. **`anka_taban_v3r.pt` + `anka_taban_v3r2.pt` ARŞİV ADAYI (B2, koşullu).** CE zinciri (3,4789→2,7380→2,5848) hüküm dosyalarında kayıtlı; modeller ara basamak. Öneri: bir sonraki plan penceresine (P2.7 kararı) dek yerinde kalsınlar, sonra `data/_archive/`'e. Etiket önerisi: `ölçüm-kanıtlı-ara-basamak`.
4. **`anka_taban_v3.pt.opt.pt` ARŞİV ADAYI (B3, v3 ile birlikte karar).** T3.4'ün AdamW son-durumu (modelin ~2 katı, envanterin en büyük tekil dosyası, 747.486.705 bayt). LR tam-anneal ile bitti (--min-lr 2e-5, 1 tam epok) → aynı koşumdan devam senaryosu fiilen kapalı; P2.6 replay'ı v3'ü init olarak yükledi (kendi opt durumu yok). **Karar kuralı:** T3.4-devam (epok-2) planlanmıyorsa → `_archive/`; planlanıyorsa → v3 ile birlikte kalsın.

### 4c. `.opt.pt` yan dosyaları

Envanterde tek yan dosya var: `anka_taban_v3.pt.opt.pt` → sahibi `anka_taban_v3.pt` (yazan kod: `train.py:46` `save_path + ".opt.pt"`, `--save-optimizer` bayrağı; T3.4 koşumunda bayrak açık). Diğer modellerin opt yan dosyası yok (P2.6 turları ve eski betikler optimizer kaydetmez).

### 4d. Kronolojik/eski tabanlar (C, 23 dosya)

- **C1 (3):** `anka_a1r.pt` (2026-09-20, pre-külliyat ön-eğitim), `anka_a2.pt` (2026-09-24, r2-sözlük kalıntısı — hüküm gereği kullanılmaz), `anka_base_v2.pt` (2026-09-26, T-0162 çıpası, eski canlı taban). Soyağacı: base_v2 → b1_5 → t0200/t0201 → t0203 → t0204/t0207-bahcivan → t0209-taban_pure → (v3).
- **C2 (13):** b1.5 ailesi — `best` + `epoch1/2/3` (T-0191, 3-epok) ve arz/hiza/usaf onarım hatları (T-0196/0197/0198, her biri 2-epok + bitiş kopyası). Hattın tüm hüküm/kayıtları `data/eval/`'de donmuş durumda.
- **C3 (6):** T-02xx merdiveni — `t0200`(+epoch1), `t0201_b05`, `t0201_b20`, `t0203_polish`, `t0204_pure`. Bu merdiven `taban_pure`'yi üretti (t0201_b05 → t0203 → t0209 init zinciri) ama taban_pure kendisi A'da; ara modellerin işi bitti.
- **C4 (1):** `anka_router.pt` (3-sınıflı, T-0154; son yazım damgası 7eafe53/2026-09-28, sha 44a46d89… ✓). Canlıda yerini 5-sınıflı `router_v2` aldı.

## 5. Politika — koruma kategorileri

| Kategori | Dosya | Politika | Koşul |
|---|---|:--:|---|
| **A — canlı soyağacı** | 4 dosya / 1,046 GiB | **DOKUNULMAZ** | asla — gateway canlı hattı |
| **B1 — bugünün ailesi, KORU** | v3 + v3r3 (0,696 GiB) | **yerinde kal** | en az P2.7/bir-sonraki-plan kararına dek |
| **B2 — ölçüm-kanıtlı ara basamak** | v3r + v3r2 (0,696 GiB) | arşiv adayı | P2.7 penceresi sonrası; etiket: `ölçüm-kanıtlı-ara-basamak` |
| **B3 — opt yan dosyası** | v3.opt (0,696 GiB) | arşiv adayı | v3-devam planı yoksa; v3 ile birlikte karar |
| **C1 — eski tabanlar** | a1r + a2 + base_v2 (1,049 GiB) | arşiv adayı | **ÖNKOŞUL:** `test_model.py:28` default `base_v2` + `:39` fallback `a1r/a2` referansı güncellenmeli (ayrı kod onayı) |
| **C2 — b1.5 ailesi** | 13 dosya (4,525 GiB) | arşiv adayı | koşulsuz; hüküm dosyaları `data/eval/`'de |
| **C3 — T-02xx merdiveni** | 6 dosya (2,088 GiB) | arşiv adayı | koşulsuz |
| **C4 — eski router v1** | 1 dosya (0,002 GiB) | arşiv adayı | koşulsuz; router_v2 canlıda |

Notlar:
- Arşivleme **taşımaktır** (aynı disk biriminde `mv`), silme değildir: diskte yer açılmaz, yalnız `data/` kökü canlı soyağa + bekleyen kararlara temizlenir.
- Arşiv klasörü mevcut konvansiyona uyar: `data/_archive/<sebep>/` (örnek: `rag_pilot_invalidated/` + `INVALIDATED.md`). Önerilen: `data/_archive/p29_kronolojik_eskiler/` + `ARSIV_NOTU.md` + `SHA256_MANIFEST.txt`.
- `*.pt` `.gitignore`'da (satır 12) → arşivlemenın git etkisi yok; commit gerektirmez (bu görevde commit zaten yasak).
- Plan kabul ölçütünün literal hali "canlı 4 dosya hariç tümü `_archive/`" = 28 dosya; bu taslak B1 (v3+v3r3) için yerinde kalma önerisiyle ayrışır — operatör onayında hangisi seçileceği belirtilir.

## 6. Disk etkisi

| Senaryo | Taşınan | Kalan `data/*.pt` | Hacim |
|---|---:|---:|---:|
| Faz-1 (C2+C3+C4 — koşulsuz öneri) | 20 dosya | 12 | 6,616 GiB |
| Faz-2a (+C1, önkoşullu) | +3 | 9 | +1,049 GiB |
| Faz-2b (+B2, P2.7 sonrası) | +2 | 7 | +0,696 GiB |
| Faz-2c (+B3, v3-devam yoksa) | +1 | 6 | +0,696 GiB |
| **Plan-maksimum (A hariç tümü)** | **28 dosya** | **4** | **9,753 GiB** |
| Aday toplam (Faz-1+2) | 26 dosya | 6 | 9,057 GiB |

Toplam envanter: 32 dosya / 10,799 GiB. Canlı+KORU (A+B1): 6 dosya / 1,743 GiB.

**Gerçek disk kazancı** yalnız ayrı onaylı silme adımında olur; bu envanterin kanıtıyla iki aday saptandı:
- 5 bayt-özdeş epoch/bit çifti = 1.868.679.965 bayt (1,740 GiB) saf yinelenik (Bölüm 9a) — silme adayı (faz-3. AYRI onay).
- `v3.opt` = 0,696 GiB (devam-değeri düşükse yalnız arşiv değil, onaylı silme de düşünülebilir — faz-3).

## 7. Uygulama komut TASLAĞI — KOŞULMAZ!

```bash
# P2.9 uygulama taslağı. YALNIZCA operatör onayı sonrası koşulur.
# cd /Users/hakankilicaslan/Git/tr_llm

# 0) Sigorta: taşımadan önce tam manifest (bu raporun sha sütunu zaten kanıttır)
shasum -a 256 data/*.pt > data/_archive/p29_oncesi_manifest_2026-11-04.txt  # önce mkdir

ARSIV=data/_archive/p29_kronolojik_eskiler
mkdir -p "$ARSIV"

# --- FAZ-1: C2 (b1.5 ailesi, 13) + C3 (T-02xx merdiveni, 6) + C4 (eski router, 1) ---
mv data/anka_b1_5_arz.pt $ARSIV/
mv data/anka_b1_5_arz_epoch1.pt $ARSIV/
mv data/anka_b1_5_arz_epoch2.pt $ARSIV/
mv data/anka_b1_5_best.pt $ARSIV/
mv data/anka_b1_5_epoch1.pt $ARSIV/
mv data/anka_b1_5_epoch2.pt $ARSIV/
mv data/anka_b1_5_epoch3.pt $ARSIV/
mv data/anka_b1_5_hiza.pt $ARSIV/
mv data/anka_b1_5_hiza_epoch1.pt $ARSIV/
mv data/anka_b1_5_hiza_epoch2.pt $ARSIV/
mv data/anka_b1_5_usaf.pt $ARSIV/
mv data/anka_b1_5_usaf_epoch1.pt $ARSIV/
mv data/anka_b1_5_usaf_epoch2.pt $ARSIV/
mv data/anka_router.pt $ARSIV/
mv data/anka_t0200.pt $ARSIV/
mv data/anka_t0200_epoch1.pt $ARSIV/
mv data/anka_t0201_b05.pt $ARSIV/
mv data/anka_t0201_b20.pt $ARSIV/
mv data/anka_t0203_polish.pt $ARSIV/
mv data/anka_t0204_pure.pt $ARSIV/

# --- FAZ-2a: C1 (ÖNKOŞUL: test_model.py default/fallback referansları güncellenmiş olmalı) ---
# mv data/anka_a1r.pt $ARSIV/
# mv data/anka_a2.pt $ARSIV/
# mv data/anka_base_v2.pt $ARSIV/

# --- FAZ-2b: B2 (P2.7 kararı sonrası) ---
# mv data/anka_taban_v3r.pt $ARSIV/
# mv data/anka_taban_v3r2.pt $ARSIV/

# --- FAZ-2c: B3 (v3-devam pretrain planlanmıyorsa) ---
# mv data/anka_taban_v3.pt.opt.pt $ARSIV/

# 1) Doğrulama: kalan dosya sayısı ve arşiv manifest kontrolü
ls data/*.pt | wc -l   # Faz-1 sonrası beklenen: 12
shasum -a 256 "$ARSIV"/*.pt > "$ARSIV/SHA256_MANIFEST.txt"
shasum -a 256 -c "$ARSIV/SHA256_MANIFEST.txt"   # 20/20 (Faz-1) OK olmalı
```

`ARSIV_NOTU.md` şablonu (taslak):

```markdown
# p29_kronolojik_eskiler
P2.9 arşiv politikasıyla taşındı (onay: <tarih>, onaylayan: <operatör>).
Bu dosyalar superseded checkpoint'lerdir; hüküm ve ölçüm kayıtları data/eval/ içinde kalıcıdır.
Geri alma: mv data/_archive/p29_kronolojik_eskiler/<dosya> data/
Bütünlük: SHA256_MANIFEST.txt (shasum -c ile doğrulanır). Envanter: data/eval/p29_checkpoint_envanteri_2026-11-04.md
```

## 8. Uyarılar / bağımlılıklar

1. **`test_model.py`**: default `data/anka_base_v2.pt` (L28) + `kristal_model.pt` eksikse `a1r`/`a2` fallback'i (L39-41). C1 arşivlenirse bu CLI default'u kırılır → ya `--checkpoint` kullanımı kabul edilir ya da default `taban_pure`'ye taşınır (ayrı kod değişikliği + onay).
2. **Tarihsel kanıt betikleri** (`t0199_taban_zarar_tanisi.py`, `t0200_mutasyon_kanit.py`, `t0196/0197/0198_faz3_kabul.py` vb.) `data/` altındaki sabit yollarla sha doğrulaması yapar — arşiv sonrası **eski halleriyle koşamazlar**. Bunlar geçmiş hükümlerin kanıt araçlarıdır; hüküm dosyaları zaten `data/eval/`'de donmuştur, yeniden koşum gerekmez. Gerekirse dosya geri taşınarak koşulur.
3. **Örnek metinlerinde geçen adlar** (`chat_prompt.py`, `agent_gateway.py`, `retrain_pipeline.py` yardım dizgilerinde `anka_a1r.pt` örneği) yalnız yardım metni — çalışma zamanı bağımlılığı değil (gateway T-0087 fail-closed).
4. **Kapsam dışı**: `data/b1_5_splits/*.pt` (6 veri kümesi), `data/eval/*fast_ds_train.pt` (3), `data/_archive/rag_pilot_invalidated/*.pt` (2) — bu envantere dahil edilmedi.
5. **Silme yasağı**: bu görev kapsamında hiçbir dosya silinmez; faz-3 (yinelenik epoch çiftleri, opt) önerileri dahi ayrı operatör onayı ister.

## 9. Ek — kanıtlar

### 9a. Bayt-özdeş yinelenik çiftler (sha kanıtlı)

- `data/anka_b1_5_arz_epoch2.pt` == `data/anka_b1_5_arz.pt` — 373.736.100 bayt. `80b0fd9a5a60de90…` (son yazılan = bitiş/best kopyası)
- `data/anka_b1_5_epoch3.pt` == `data/anka_b1_5_best.pt` — 373.735.672 bayt. `e5eb114e5bd71d58…` (son yazılan = bitiş/best kopyası)
- `data/anka_b1_5_hiza_epoch2.pt` == `data/anka_b1_5_hiza.pt` — 373.736.207 bayt. `05d8e4cf55a2e27e…` (son yazılan = bitiş/best kopyası)
- `data/anka_b1_5_usaf_epoch2.pt` == `data/anka_b1_5_usaf.pt` — 373.736.207 bayt. `8f846023f2fb2374…` (son yazılan = bitiş/best kopyası)
- `data/anka_t0200_epoch1.pt` == `data/anka_t0200.pt` — 373.735.779 bayt. `be5c1fb47ceb5187…` (son yazılan = bitiş/best kopyası)

Yorumu: her çiftte "epoch" dosyası aynı koşumun bitiş kaydıyla özdeş — save-every/epok kalıbının doğal çift-kaydı; bilgi kaybı olmadan (faz-3, ayrı onay) silme adayıdır.

### 9b. Betik çıpası ↔ ölçülen sha doğrulaması

| Dosya | Betik/hüküm çıpası | Ölçülen (bu envanter) | Durum |
|---|---|---|---|
| `anka_b1_5_arz.pt` | `80b0fd9a5a60de90… (t0199/t0200)` | `80b0fd9a5a60de90…` | ✓ EŞ |
| `anka_b1_5_hiza.pt` | `05d8e4cf55a2e274… (t0199/t0200)` | `05d8e4cf55a2e274…` | ✓ EŞ |
| `anka_b1_5_usaf.pt` | `8f846023f2fb2374… (t0199/t0200)` | `8f846023f2fb2374…` | ✓ EŞ |
| `anka_b1_5_best.pt` | `e5eb114e5bd71d58… (evaluate_anka_b1_5.py)` | `e5eb114e5bd71d58…` | ✓ EŞ |
| `anka_base_v2.pt` | `d0f415f3d882beb4… (çoklu betik çıpası)` | `d0f415f3d882beb4…` | ✓ EŞ |
| `anka_router.pt` | `44a46d89f3d4260c… (CHANGELOG damga 7eafe53)` | `44a46d89f3d4260c…` | ✓ EŞ |
| `anka_taban_pure.pt` | `5479a2f3… (p23 hüküm: retrain sonrası değişmedi)` | `5479a2f35a88975c…` | ✓ EŞ |

---
*P2.9 envanter+politika taslağı — 2026-11-04, ~/Git/tr_llm. Üretim: otomatik envanter (ls/stat/shasum/grep); tüm sayılar programatik hesaplandı.*
