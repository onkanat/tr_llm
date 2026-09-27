# G6b (T-0138) SONUÇ-3b — BERBER FİNE-TUNE KOŞUMU — 2026-09-27 (antigravity)

**İlan:** `data/eval/berber_g6b_finetune_ilan_2026-09-27.md` (`26dbfb9fc5dc2d9cf89886dbae8ab3a4fd43ce13e6593f2ce9ebb52f3a7d80b3`)
**Hüküm:** **KABUL_YOK** (rc=2, `scratch/t0138/berber_ft_hukum.json` `20620830606c3e38d70f028d5272b97c682992f86fc3132dd21c1d1d7138a45f`)
**Gerekçe:** A9 ROUGE artışı (+0,2743) kalıp örtüşmesi ve distraktör seviyesindedir (0,3045 vs Distraktör: 0,3023; mutlak ROUGE < `ESIK_ROUGE 0,3221` dokunulmaz eşik; kanonik `rouge_gec=False`, `ceket_ekseni_gec=False`). G6a Bahçıvan zinciriyle tutarlı: biçim kazanılmış, içerik distraktör bandında kalmıştır.

---

## 1. Yürütme Özeti ve Mid-Run Mimari Dönüşüm Beyanı

- **Başlangıç:** İlk koşum P3 sürücüsü ile üç kaynaklı (%20 wiki / %40 marangoz / %40 berber) başlatıldı; kullanıcı/operatör direktifi ("bir berberden marangozluk beklemiyoruz; modüller birbirine karışmamalı, tek seferde tek yetenek eklenmeli") üzerine derhal durduruldu (`task-295` iptal edildi).
- **Modüler Ceket Mimarisi:** Doğrudan `train_module.py` ile "Değişmez Taban + Değişken Ceket" mimarisine geçildi.
- **Dil Koruma Aşaması:** Salt SFT (1.000 adım) koşumunda A-ekseni CE artışı %+25,08 (A_DUSTU) çıktı. Genel Türkçe dil dengesini korumak için `scripts/wiki_replay_disjoint.py` ile 256 A-ekseni penceresiyle **0 çakışmalı** sızıntısız Wikipedia dilimi (`scratch/t0138/wiki_replay.bin`) üretildi ve `scratch/t0138/berber_wiki_mix.bin` (%75 Berber SFT / %25 Wikipedia Replay / %0 Marangoz / %0 Bahçıvan) oluşturuldu.
- **Eğitici:** `train_module.py` (Taban %100 DONUK; 93.424.986 parametre sabit, 0 tensör değişti).
- **Hedef Katmanlar:** `attn.q_proj,attn.k_proj,attn.v_proj,attn.out_proj,mlp.0,mlp.2,lm_head` (37 katman LoRA, $r=16, \alpha=32$, $1.869.216$ eğitilen parametre / %1,962).
- **Parametreler:** lr=2e-4, batch=8, block=128, steps=1000, seed=43, device=mps.
  - Kayıp: İlk 6,0626 $\rightarrow$ Son-60 ortalama 1,1973 $\pm$ 0,6032.
- **Üretilen Modül:** `scratch/t0138/berber.mod.pt` (`c44b237ab55326dc46ede9664d82133466c92e701bdaa08e3393a51317689726`).

---

## 2. Kabul Ayraçları ve Kanonik Ölçüm Sonuçları

Kabul kuralı: **A9 VE A11 VE A12 VE A13 VE Kanonik Ceket Kapısı** (Hüküm betikten: `scratch/t0138/berber_ft_hukum_betigi.py`).

| # | Ayraç / Kapı | Ölçülen Değer | İlan / SPEC Eşiği | Hüküm Dalı | Durum |
|---|---|---|---|---|---|
| **A9** | Berber Yetenek Artışı | **+0,2743** (ROUGE: 0,3045 vs Çıpa: 0,0302) | $\ge +0,0230$ VE ROUGE $\ge 0,3221$ | `KALIP_ORTUSMESI_DISTRAKTOR_DUZEYI` | **KABUL_YOK** |
| **K-ROUGE** | Kanonik ROUGE Kapısı (`rouge_gec`) | **0,3045** | $\ge 0,3221$ (`ESIK_ROUGE` dokunulmaz) | `rouge_gec = False` | **DÜŞTÜ** |
| **K-CEKET** | Kanonik Ceket Kapısı (`ceket_ekseni_gec`)| **False** | `all(hukum.values())` | `ceket_ekseni_gec = False` | **DÜŞTÜ** |
| **A11** | Dil Tabanı (A-ekseni CE) | **%+6,40** (CE: 3,5236 vs Taban: 3,3117) | $\le +\%10,0$ | `A_GECTI` | **GEÇTİ** |
| **A12** | Model Kararlılığı (B top-1) | **3,57 puan düşüş** (%52,34 vs Taban: %55,91) | $\le 5,0$ puan | `B_GECTI` | **GEÇTİ** |
| **A13** | Tutarsızlık ve Ezber Oranı | **Tutarsızlık: %0,00 · Ezber: %0,00** | Tutarsızlık < %5 · Ezber < %10 | `GECTI` | **GEÇTİ** |

---

## 3. Derinlemesine Teşhis ve Bulgular

1. **Distraktör Seviyesi ve Kalıp Kilitlenmesi:**
   - Kanonik kap oraklı: `distraktor_rouge = 0,3023`.
   - Modül çıktısı: `ROUGE-L = 0,3045` (Distraktör seviyesinin yalnızca $+0,0022$ üstünde).
   - Ham üretim metinleri (`ham_gm`) incelendiğinde 100/100 örneğin 3 ana kalıba kilitlendiği görülmektedir:
     - `"saç ve sakal bakım problem ..."`
     - `"berber alet ve gereç ..."`
     - `"teknik / periyodik bakım aşama ..."`
   - Gerçek içerik kesişimi $LCS\text{-}F1 = 0,1169$ düzeyindedir.
   - Sonuç: Model berber dili biçimini ve rol zarfını kusursuz öğrenmiş (%0 tutarsızlık, %0 ezber), ancak içerik üretimi distraktör gürültüsü bandında kalmıştır. Bu bulgu, T-0096 ve G6a Bahçıvan raporlarındaki *"Delta bilgi eklemez, kalıp/biçim ekler"* ilkesiyle birebir örtüşmektedir.

2. **LM Bedeli Analizi (A11):**
   - Wikipedia CE artışı $\%+6,40$ ile ilan edilen $\%10$ tavanının altında kalmıştır (GEÇTİ). Ancak G6a monolitik koşumundaki $\%+0,446$'ya kıyasla modülün genel dilde yarattığı yerel sapma daha yüksektir.

---

## 4. Tam SHA-256 Digest Tablosu

| Dosya | SHA-256 |
|---|---|
| `data/anka_base_v2.pt` | `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50` |
| `data/pedagogy/berber_arena.jsonl` | `2786e9149b50ada3606ca399b44ee9ad35c2c7c282fb4a00d49f4d6bf46d40a1` |
| `scratch/t0138/berber_arena.bin` | `104f96940e0fd07eda33db97de21fddfb4252f87e40311b1372c6d5b9d3f7728` |
| `scratch/t0138/berber_heldout.jsonl` | `d063ec27f66e8cecea348706c36e405a3f9f540a1b9d9505e6eb16cfb555abb7` |
| `scratch/t0138/berber_egitim_adapter.jsonl` | `b775d0884ed7747eb00812bfffc804721d7533db2812ab7d9163e50570077963` |
| `scratch/t0138/berber_egitim.bin` | `0306b2e2e0204e2391f0617184acecdd5fca04e0347a3b91ecb5c0625850df85` |
| `scratch/t0138/wiki_replay.bin` | `49f88cb8d13f42cdc7f34e4d1fbd20600294292e516bf58d8b651660b4c4a2e7` |
| `scratch/t0138/berber_wiki_mix.bin` | `8a6b0985d8b7b5338655d1c170b635273b86275fddb1123c01a31924c519afe2` |
| `scratch/t0138/berber.mod.pt` | `c44b237ab55326dc46ede9664d82133466c92e701bdaa08e3393a51317689726` |
| `scratch/t0138/berber_taban_cipa.json` | `708bb03f37ca392dd82b4d0463f40761bb5f01944721ce2ac5139236a4c539a7` |
| `scratch/t0138/berber_ft_berber_sonda.json` | `7eda19349295b3ea2b61272ebca2027c24b4ab2993a8e410669971688fc44545` |
| `scratch/t0138/berber_ft_hukum.json` | `20620830606c3e38d70f028d5272b97c682992f86fc3132dd21c1d1d7138a45f` |
| `data/eval/berber_g6b_ilan_2026-09-27.md` | `b3d87bfe7c92b236fa7c603a1fc6c5264b910e5d03a111a9e9e6ec9d7bcfa728` |
| `data/eval/berber_g6b_tabancipa_sonuc_2026-09-27.md` | `3dd65050d97f666d0e6d29c8df0c28a05586401f2ea6cbfa1fbf8d037d89f301` |
| `data/eval/berber_g6b_arzuretim_sonuc_2026-09-27.md` | `6ef83307b22a6136be070e176378cb5cf3486a4ea8474f7678bb15c54c33f20f` |
| `data/eval/berber_g6b_derleme_sonuc_2026-09-27.md` | `c32cf1b4df5fc266ea8617ea2a373b5dfd7dfc89d20c3545fc4948a318287515` |
| `data/eval/berber_g6b_finetune_ilan_2026-09-27.md` | `26dbfb9fc5dc2d9cf89886dbae8ab3a4fd43ce13e6593f2ce9ebb52f3a7d80b3` |
| `data/eval/berber_g6b_finetune_sonuc_2026-09-27.md` | `3f1e34f2d970ceb557aa4b2697904a5a34751a8ac1faee8c34cf4c5a9037f500` |
