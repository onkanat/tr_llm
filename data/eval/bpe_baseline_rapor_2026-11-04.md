# P2.7 — BPE-Baseline Benchmark Raporu (2026-11-04)

## İLAN

- **Amaç:** Kristal tokenizer ile BPE baseline'ı aynı test korpusu üzerinde tek komutla, tekrarlanabilir biçimde karşılaştırmak (P2.7 kabul ölçütü).
- **Kaynak:** inceleme betikleri `bpe_v2.py`/`bpe_kiyas.py`/`kapsam.py` (~/.hermes/cache/scratch/trllm_olcum/) repoya taşındı → `scripts/benchmark_bpe_baseline.py`.
- **Komut:** `venv/bin/python scripts/benchmark_bpe_baseline.py --corpus data/raw_corpus.txt --samples 2000 --merges 2000,8000,32000 --seed 42` (git HEAD: `2bf8ac9`)
- **Korpus:** `data/raw_corpus.txt` — 17678 dolu satır, 42,321 benzersiz kelime, 367,530 toplam kelime.
- **BPE:** saf Python artımlı BPE (kelime-içi, `</w>` işaretli; bayat-heap geri-itmeli doğrulama), aynı korpus üzerinde eğitildi; hedefler 2000, 8000, 32000 merge.
- **Kristal:** `KristalTokenizer(literal_entity_mode=True)`, vocab `data/rebuild/vocab_anka_r1_33114.json` (33,114 ID).
- **Örneklem/tekrarlanabilirlik:** satır örneklemi 2000 (tohum 42), benzersiz kelime örneklemi 2000 (tohum 42), hız ölçümü 3 tekrarın en iyisi. BPE eğitimi kanonikleştirildi: kaynak bpe_v2.py'de etkilenen-kelime yinelemesi ham set sırasıyla yapılıyordu ve bu sıra PYTHONHASHSEED'e bağlı olduğundan beraberlik (tie) durumlarında merge listesi süreçten sürece değişiyordu (ölçüldü: 8 koşumda merge[18] bir kez takla attı). Bu portta `sorted` sırasıyla eğitim tam deterministiktir.
- **UNK ölçüm notu:** BPE'de UNK jetonu yoktur; karşılaştırılabilir ölçüm olarak eğitim alfabesinin dışında kalan **kelime-dışı karakter oranı** raporlanır. Kristalda UNK jeton oranıdır.
- **Sözleşme notu:** Kristal tok/kelime payı BOS/EOS hariç, **noktalama jetonları dahil** gerçek dizi uzunluğudur; BPE yalnız kelime jetonlarını sayar (kaynak betik konvansiyonu). Örneklemede 14,023 noktalama işareti kristal sayısına dahildir (~+0.343 tok/kelime pay).

## HÜKÜM

- **Frekans-ağırlıklı (in-domain) fertility:** korpus satır örneklemi BPE'nin EĞİTİM korpusundan alınmıştır; kristal **2.065** vs BPE-32000 **2.101** tok/kelime — BPE-32000 2% daha fazla jeton. BPE-32000 bu ölçümde ezber avantajı taşır (42,321 benzersiz kelimenin sık geçenleri ~32k merge ile bütün-kelime sembolüne dönüşmüştür).
- **Dağıtım-dışı / seyrek-kelime fertility:** 4 sabit metin ortalaması kristal **2.207** vs BPE-32000 **2.854** tok/kelime (1.29×); benzersiz kelime örneklemi kristal **1.944** vs BPE-32000 **3.309** (1.7×) — metin korpusun birebir satırları olmadığında ve seyrek kelimelerde morfolojik çözümleme subword ezberinden belirgin biçimde daha verimli.
- **UNK ekseni:** kristal UNK oranı satır örnekleminde **%1.18**, benzersiz kelime örnekleminden **%6.48** (seyrek kelimelerin bir bölümü morfolojik çözümleme bulamıyor). BPE'de UNK jetonu yoktur; ölçülen kelime-dışı karakter oranı in-domain örneklemde trivyal olarak %0.0. UNK'suzluk BPE'nin yapısal avantajıdır; dağıtım dışı metinde bu karakter oranı artar, kristalda ise UNK oranı artar.
- **Hız ekseni:** kristal **9,137.5 tok/s** vs BPE-32000 **1,148.7 tok/s** (BPE/kristal ~0.1×; kristal daha hızlı). DİKKAT: BPE hızı burada saf-Python applier ölçer; üretim BPE kurulumları (Rust tabanlı HF tokenizers vb.) kat kat hızlıdır. Bu eksen paradigma değil uygulama farkı ölçer.
- **Sözük ekonomisi:** kristal 33,114 sabit ID vs BPE-32000 ~32,106 sembol (1.03×) — benzer sözük bütçesinde karşılaştırma adil.
- **Sonuç:** BPE-baseline BPE-2000, BPE-8000, BPE-32000 varyantlarıyla formalize edildi ve tek komutla tekrarlanabilir rapora bağlandı. Eksen dengesi (BPE/kristal jeton oranı): in-domain frekans-ağırlıklı 1.02×, 4 sabit metin 1.29×, benzersiz kelime 1.7× — kristal dağıtım-dışı/seyyar eksenlerde önde, BPE yalnız in-domain ezber ölçümünde öne geçebilir; UNK'suzluk BPE'nin tek yapısal üstünlüğüdür.

## RAPOR

### Tablo 1 — Korpus satır örneklemi (2000 satır, 40,888 kelime; tok/kelime + UNK + hız)

| Kodlayıcı | Sözük | tok/kelime | UNK / kelime-dışı karakter | Hız (tok/s) |
|---|---|---|---|---|
| **Kristal** | 33,114 ID | **2.065** | UNK %1.18 (999/84,453) | 9,137.5 |
| BPE-2000 | ~2,106 sembol | 3.639 | kelime-dışı karakter %0.0 | 35,124.3 |
| BPE-8000 | ~8,106 sembol | 3.173 | kelime-dışı karakter %0.0 | 7,743.1 |
| BPE-32000 | ~32,106 sembol | 2.101 | kelime-dışı karakter %0.0 | 1,148.7 |

### Tablo 2 — 4 sabit metin (kaynak betik metinleriyle birebir aynı)

| Metin | Kelime | Kristal tok/kelime | Kristal UNK% | BPE-2000 tok/kelime | BPE-8000 tok/kelime | BPE-32000 tok/kelime |
|---|---|---|---|---|---|---|
| gunluk | 11 | 2.273 | 0.0% | 4.455 | 4.0 | 3.273 |
| teknik | 16 | 1.812 | 0.0% | 4.062 | 3.562 | 2.688 |
| edebi | 11 | 2.455 | 0.0% | 5.0 | 4.182 | 3.455 |
| tamlama | 14 | 2.286 | 3.12% | 4.286 | 3.643 | 2.0 |
| **ortalama** | — | **2.207** | — | **4.451** | **3.847** | **2.854** |

### Tablo 3 — Benzersiz kelime örneklemi (2000 kelime, tohum 42; kaynak betiğin 'korpus_2000ornek' ölçümü)

| Kodlayıcı | tok/kelime | UNK / kelime-dışı karakter% | Süre (sn) |
|---|---|---|---|
| **Kristal** | **1.944** | UNK %6.48 | 0.66 |
| BPE-2000 | 5.227 | %0.0 | 0.22 |
| BPE-8000 | 4.689 | %0.0 | 0.87 |
| BPE-32000 | 3.309 | %0.0 | 3.81 |

### Tekrar üretim

```bash
venv/bin/python scripts/benchmark_bpe_baseline.py --corpus data/raw_corpus.txt --samples 2000 --merges 2000,8000,32000 --seed 42
```

Ham ölçümler: `data/eval/bpe_baseline_olcum_2026-11-04.json`
