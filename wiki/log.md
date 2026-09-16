---
tags: [log]
date: 2026-06-14
sources: [scripts/]
status: active
---

# Geliştirme Günlüğü (wiki/log.md)

Bu dosya, Kristal-Vektörel Mimarisi projesindeki kodlama adımlarını, eğitim fazlarını ve test sonuçlarını kronolojik olarak kayıt altında tutar.

---

## 📅 Kronolojik Loglar

### 2026-09-15: agent-bus Koordinasyonu, Sözleşme Kapanışları ve Mimari Doğrulamalar (T-0001 .. T-0030)
* **T-0001..T-0011 (agent-bus Altyapısı):** Çift etmenli (Claude/Antigravity) koordinasyon protokolü, MCP stdio araçları, kiralama mekanizması ve `.agent-bus/frozen.json` değişmezleri devreye alındı.
* **T-0012 (B1.5 Bölme Sözleşmesi):** Kapalı-sınıf iskelet anahtarlarının analiziyle eğitim ve test arasındaki cevap düzeyinde veri sızıntısı tespit edildi; ayrık kök sözleşmesi tanımlandı (`data/eval/f0_identity_scope_2026-09-15.json`).
* **T-0013 & T-0015 (Kanonik İstem Zarfı ve Çift BOS Çözümü):** Değerlendirme kodlarında istem başına fazladan eklenen çift `<BOS>` belirteci kapatıldı; tüm sistem [prompt_contract.py](file:///Users/hakankilicaslan/Git/tr_llm/src/llm/prompt_contract.py) içindeki tekil `render_prompt` fonksiyonuna bağlandı ve kanarya testleri eklendi.
* **T-0016..T-0019 (Exp C ve Ezber Tanılaması):** Kanonik zarfla MPS üzerinde Exp C temel çizgisi diske kaydedildi; modelin C1'i neden ezberlemediğine dair teacher-forcing kanıtları ve soykütüğü analiz edildi (`data/eval/t0018_why_no_memorization.json`).
* **T-0020 & T-0021 (Sözlük Restorasyonu):** Noktalama bloğunun genişletilmiş modellerde tam olarak 32.137 indeksinde başladığı saptandı ve Exp C modelin gerçek kelime dağarcığıyla yeniden ölçüldü (`data/eval/vocab_restore_recheck_2026-09-15.json`).
* **T-0022 & T-0024 (F0 Envanter ve Kapsam Boşluğu):** Kapsam boşluğu tanımlandı ve süpürme seçim kuralı yazıldı.
* **T-0023 (cot_vault İzolasyonu):** Birim testlerin üretim kasasına (`data/pedagogy/cot_vault.jsonl`) sahte kayıt yazması engellendi; geçici test kasası izolasyonu sağlandı ve SHA-256 bütünlük çıpası konuldu.
* **T-0025 & T-0026 (Kanonik Boyutlandırma ve Sessiz Kırpma Yasağı):** [resize_state_dict](file:///Users/hakankilicaslan/Git/tr_llm/src/llm/prompt_contract.py) fonksiyonu tek kanonik kapı haline getirildi; sessiz kırpma yasaklanarak açık `ValueError` fırlatması sağlandı; mutant testleri eklendi.
* **T-0027 (Model Durumu ve Çıkarım Envanteri):** 23 checkpoint incelendi; bit-özdeş gruplar ve 4096^2 maske tamponları belgelendi; 4 model üzerinden deterministik çıkarım kanıtlandı (`data/eval/t0027_model_state_2026-09-15.json`).
* **T-0028 & T-0029 (Talimat Yüzeyi Hizalaması ve docs_update Genişletmesi):** KOD KAZANIR ilkesiyle tüm talimatlar (`CLAUDE.md`, `GEMINI.md`, `llmcoder.md`, `AGENT.md`, `skills/`) güncellendi; kök (48.200) ile kelime dağarcığı (31.357 / 32.816) ayrıldı; Skill_7 DPO çelişkisi giderildi; repo dışı `/docs_update` yeteneğine talimat taraması eklendi (`data/eval/t0028_instruction_alignment_2026-09-15.json`, `data/eval/t0029_v1_v6_reconciliation_2026-09-15.json`).
* **T-0030 (Wiki Canlandırma):** 22 Haziran'dan beri dondurulan `wiki/` dizini Eylül 2026 mimari çalışmalarıyla (8 yeni konsept/varlık sayfası) eşitlendi; `llmcoder.md` Kural 2 canlı hale getirildi.

### 2026-06-18: Ortaokul Sohbet SFT Hizalama Eğitimi ve Hata Çözümleri
* **Olay:** Modelin ortaokul düzeyinde samimi bir sohbet üslubu kazanması amacıyla 125 özgün sohbet çifti (`middle_school_chat.jsonl`) içeren veri kümesi derlendi. Ebeveynlik ve Bilimsel SFT veri kümeleri ile dengelenerek `data/train_chat_sft.bin` derlendi.
* **Olay (Hata Çözümü):** Yeni eklenen morfemlerin (`"bıkma"`) diske kaydedilmemesinden kaynaklanan `IndexError: index out of range` hatası, [prepare_chat_fine_tune_dataset.py](file:///Users/hakankilicaslan/Git/tr_llm/scripts/prepare_chat_fine_tune_dataset.py) dosyasına `vocab.save()` eklenerek ve antrenmanda `resize_state_dict` entegre edilerek çözüldü.
* **Olay (Optimizasyon):** CPU eğitim süresini kısaltmak için diyalog uzunlukları analiz edilerek `block_size` 512'den **128**'e düşürüldü ve CPU eğitim hızı **11 kat artırıldı** (adım başına ~16.57sn -> ~1.45sn).
* **Olay (Hata Çözümü):** Boş pencerelerdeki maskeleme nedeniyle oluşan ve eğitimi bozan `NaN` loss hatası, `active_targets > 0` olana kadar mini-batch'leri yeniden örnekleyen bir döngü ile tamamen giderildi.
* **Sonuç:** CPU üzerinde 500 adımlık sohbet ince ayar eğitimi başarıyla tamamlandı. Model, interaktif testlerde morfolojik olarak hatasız kanka diliyle sohbet çıktıları üretti ve decompiler entegrasyonuyla anlamlı Türkçe yüzey formuna geri dönüştü.

### 2026-06-18: DPO Tercih Hizalama Eğitimi Tamamlanması ve Doğrulama
* **Olay:** CPU üzerinde unconstrained sequence uzunluğundan kaynaklı yavaşlık (~130s/adım) giderilerek DPO eğitim dizisi uzunluğu **768 token** ile sınırlandırıldı. Bu sayede eğitim adım başına **~16 kat hızlanarak** ~9sn'ye indirildi.
* **Olay:** DPO eğitimi (`train_dpo.py`) 1000 adımda CPU üzerinde başarıyla tamamlandı. DPO kaybı (loss) **0.724567**'den **0.303739**'a düşürüldü ve ağırlıklar `data/kristal_model.pt` olarak kaydedildi.
* **Olay:** `test_model_decompiler.py` ile doğrulama yapıldı. Modelin SFT talimat takip yeteneğinin hatasız çalıştığı ve morfolojik çıktıları tam uyumlulukla Türkçe kelimelere geri dönüştürebildiği gözlemlendi. 61/61 pytest testi başarıyla geçti.

### 2026-06-17: Decompiler Entegrasyonu ve Model Çıktısı Doğrulama Testi
* **Olay:** Morfem dizilerini Türkçe kelimelere ve cümlelere dönüştüren `decompile_sentence` yöntemi `MorphemeDecompiler` sınıfına ([tests/test_decompiler.py](file:///Users/hakankilicaslan/Git/tr_llm/tests/test_decompiler.py)) eklendi.
* **Olay:** `simulate.py` içindeki kelime birleştirme döngüsü refaktör edilerek doğrudan `decompile_sentence` metodunu kullanacak şekilde sadeleştirildi.
* **Olay:** Model çıktısı (morfem seviyesinde) üreten ve bu morfemleri anlamlı Türkçe cümlelere geri döndüren `test_model_decompiler.py` test betiği yazıldı.
* **Olay:** `tests/test_decompiler.py` test betiği `pytest` ile otomatik çalıştırılacak şekilde standart test formatına dönüştürüldü.
* **Sonuç:** Modelin hem serbest üretim (completion) hem de SFT talimatı sonrasında ürettiği morfem dizilerinin (örn. `git TENSE_FUT PERSON_1SG`) Türkçe yüzey formuna (`gideceğim`) hatasız ve deterministik şekilde geri dönüştürülebildiği doğrulandı. Tüm pytest testleri (61/61) başarıyla geçti.

### 2026-06-16: Çoklu Bilimsel Veri Seti Eğitimi (SFT ve DPO Sosyalleşme)
* **Olay:** 13 adet bilimsel, HPC ve Linux odaklı DPO veri seti (208 MB JSONL) analiz edildi. Unicode `NFC` normalizasyonu `KristalTokenizer`'a entegre edildi.
* **Olay:** OOV oranını düşürmek için en sık geçen 1.900 adet teknik/bilimsel kök `roots.tsv` tablosuna NOUN olarak eklendi; sözlük **28.261** token'a genişletildi.
* **Olay:** 13 veri setindeki 5.792 tercih çifti derlenerek `data/train_all_chosen.bin` (**18.097.297 token**) ve `data/pedagogy/dpo_all_tokenized.jsonl` oluşturuldu.
* **Olay:** SFT ağırlıkları yeni sözlüğe göre boyutlandırılarak (`scripts/train_scientific_sft.py`) GPU üzerinde 3.000 adım eğitildi. Bitiş kaybı (loss) **2.66**'ya düşürüldü.
* **Olay:** DPO eğitimi (`train_dpo.py`) 4096 bağlam uzunluğuyla, GPU bellek limiti (MPS OOM) nedeniyle CPU üzerinde 1500 adımda arka planda başlatıldı.
* **Sonuç:** Faz 1, 2 ve 3 başarıyla tamamlandı. Modelin bilimsel kelime haznesi genişletildi ve SFT uyumu sağlandı. Faz 4 DPO tercih hizalama eğitimi CPU üzerinde stabil olarak çalışmaya devam etmektedir.

### 2026-06-15: Pedagojik Faz 2 (Parenting SFT İnce Ayar) Aşaması
* **Olay:** Wikipedia baskınlığını gidermek ve modelin talimat izleme yeteneğini mükemmelleştirmek amacıyla Stage-2 SFT İnce Ayar betiği (`scripts/train_sft_fine_tuning.py`) yazıldı ve çalıştırıldı.
* **Olay:** Bebeklik ve Ebeveynlik veri kümeleri birleştirilerek (791,017 token) RoPE mimarili model `lr=1e-4` ile 1,500 adım eğitildi.
* **Olay:** Uzak yapay zeka (Ollama) ve Qdrant veritabanı bağlantı kararsızlıkları/oturum stabilitesi sorunları analiz edilerek [Google Patent TW201018092A/en](https://patents.google.com/patent/TW201018092A/en) kalibrasyon metodolojisi doğrultusunda çözümler/fallback yapıları incelendi ve `session_stability_report.md` oluşturuldu.
* **Olay:** Komut satırından etkileşimli morfem çıkarımı ve SFT testleri yapmayı sağlayan `chat_prompt.py` betiği ve ilgili `wiki/entities/chat_prompt.md` dokümantasyonu oluşturuldu.
* **Sonuç:** Eğitim bitiş kaybı (loss) **0.198322** seviyesine düşürüldü. Model, SFT doğrulama görevlerinde (kök bulma, çoğul tespiti, hâl tespiti vb.) **%100.00 kesin doğruluk** elde etti. Pytest testleri (60/60) başarıyla tamamlandı. Bağlantı kesintileri yedekli yapılarla kontrol altına alındı.

### 2026-06-14: Wikipedia Entegrasyonu ve 10,000 Adımlık MPS Eğitimi
* **Olay:** Türkçe Wikipedia Markdown veri kümesinden 5,000 makale tokenize edilip `train_wiki.bin` (8.20M token) derlendi.
* **Olay:** Sözlük (vocab) 15,601 tokenden **25,665** tokene genişletildi.
* **Olay:** Külliyat, Bebeklik, Ebeveynlik, DPO ve Wikipedia'yı birleştiren **5 Fazlı Derleme** betiği (`scripts/prepare_all_phases_with_wiki.py`) oluşturuldu ve `data/train.bin` (10,761,665 token) derlendi.
* **Olay:** Model Apple Metal GPU (MPS) hızlandırmasıyla **10,000 adım** eğitildi. Ortalama kayıp (loss) **4.189** değerine düşürüldü.
* **Olay:** Vektörel Gezgin için uçtan uca RAG akış yöneticisi (`src/rag/rag_pipeline.py`) oluşturuldu ve Qdrant hibrit belleği ile `KristalLM` dil modelinin entegrasyonu sağlandı.
* **Olay:** Model mimarisine Döner Konumsal Kodlama (**RoPE**) eklendi ve mutlak `pos_embedding` kaldırıldı. Bağlam penceresi (`block_size`) **32'den 64'e** çıkarıldı. Model Metal GPU (MPS) üzerinde 5,000 adım eğitilerek bitiş kaybı **3.069** seviyesine çekildi.
* **Sonuç:** SFT talimatı sonrasında çıktı etiketine geçiş yapma olasılığı **%98 - %99** seviyesine ulaştı. 60/60 pytest testi başarıyla geçti. RAG çıkarım entegrasyonu başarıyla doğrulandı. RoPE entegrasyonu sonrasında SFT doğrulukları ve negatif örnek yanlılığı (PLURAL bias) **%99.82** başarı oranıyla tamamen çözüldü.

### 2026-06-11: SFT Maskeleme ve DPO Entegrasyonu (Sprint 3)
* **Olay:** Causal Prompt Masking (İstem Maskeleme) mekanizması `train.py`'a başarıyla entegre edildi. SFT görevlerinin yalnızca çıktı kısmının eğitilmesi sağlandı.
* **Olay:** DPO (Direct Preference Optimization) Chosen veri kümesi `train_dpo_chosen.bin` (1.17M token) olarak derlendi.
* **Olay:** 4 Fazlı Derleme betiği (`scripts/prepare_all_phases_with_dpo.py`) oluşturuldu ve `data/train.bin` (2.55M token) derlendi.
* **Sonuç:** Modelin tüm SFT görevlerinin çıktılarının bir `<PROPER_NOUN>` ile başlaması gerektiği öğrenildi.

### 2026-06-21: RAG Hizalama Eğitimi, Tokenizer OOV Çözümü ve Performans İyileştirmesi
* **Olay:** Tokenizer seviyesinde morfem etiketlerinin Türkçe kelime gibi derlenip `<PROPER_NOUN>` etiketine dönüştürülmesini engelleyen kontrol entegre edildi. `B:` ve `S:` önekleri `belge:` ve `sorgu:` Türkçe kelimeleriyle değiştirildi.
* **Olay:** `scripts/generate_rag_dataset.py` betiği ile Qdrant üzerindeki 4.000 adet GTS verisinden 11.963 adet sentetik RAG SFT kopyalama çifti (tekil, morfem öbekli ve tam döküman) oluşturuldu.
* **Olay:** Sohbet SFT, Parenting SFT, Bilimsel SFT ve 6.000 RAG çiftini birleştiren `data/train_chat_sft.bin` (1.232.000 padded token) derlendi.
* **Olay (Hata Çözümü):** Sözlük boyutu daraldığında oluşan dilimleme hatası `resize_state_dict` fonksiyonuna `min()` sınırlayıcıları eklenerek giderildi.
* **Olay:** Model, Apple Metal (MPS) GPU üzerinde 3 aşamalı fine-tuning işlemine tabi tutuldu (toplam 2.400 adım). SFT kaybı (Loss) son fazda **2.00**'den **0.058** seviyesine düşürüldü.
* **Olay:** Çoklu tekrarlı kelimelerin attention-sink etkisiyle kopyalama esnasında döngü oluşturmasını engellemek için kod tabanındaki tüm generator/decoder modüllerine kayan pencereli (window=12) **Repetition Penalty (1.5)** entegre edildi.
* **Sonuç:** Model RAG modunda ortaokul sohbet jargonunu tamamen bırakarak döküman kopyalama yapısına geçti. Çok uzun/tekrarlı sorgu senaryoları ve temiz test durumlarında (Clean Case) **%100 kopyalama doğruluğuna ve loop önlemeye** ulaşıldı. 61/61 pytest testi başarıyla geçti.

### 2026-09-15: RAG Pipeline Kanonik Dikişi ve Eşik Kapısı Devreye Alma (T-0032)
* **Olay:** `src/rag/rag_pipeline.py` monolitik betikten saf, import edilebilir modüler birimlere (`build_query_vectors`, `retrieve_context`, `is_context_usable`, `build_rag_prompt_tokens`, `greedy_decode`, `RagPipeline`, `build_from_disk`) dönüştürüldü.
* **Olay:** Eşik kapısı `RAG_MATCH_THRESHOLD = 0.40` adlandırılmış sabiti tanımlandı; `epistemic_agent.py` çıplak 0.40 eşiği yerine kanonik fonksiyona bağlandı ve bit-özdeşlik korundu.
* **Olay:** `tests/test_rag_pipeline.py` oluşturularak 6 zorunlu vaka (eşik 0.39, 0.41, sınır 0.40, determinizm, EOS durma / max_len, None bellek) test edildi; mutant probe (0.0 kırmızı -> 0.40 yeşil) doğrulandı.
* **Olay:** `tests/test_phase2.py` içindeki yanıltıcı test adı `test_vector_memory_and_tokenizer_integration` olarak güncellendi; `wiki/entities/rag_pipeline.md` senkronize edildi.
* **Sonuç:** RAG boru hattı simülasyonu monolitik betik olmaktan çıkarılarak test edilebilir ve güvenli biçimde devreye alındı. Eşik kapısı sayesinde eşleşme skoru < 0.40 olan belgelerde modelin halüsinasyon görmesi ve gürültüye maruz kalması engellendi.

### 2026-09-16: F2 Veri Yeniden Derleme, Donmuş-Yazım Kapıları ve F3 Tabana Yetkinlik Tamamlama (T-0032 .. T-0054)
* **Olay (F2, T-0044/T-0045):** Bayat `.bin`'ler yeni sözlükle (32.852) ve `literal_entity_mode=True` ile yeniden derlendi. `data/train_chat_balanced.bin`: 3.120.000 token, **max id 32.850**, noktalama bloğu **180.336**, kesme **39.706** (eski artefaktta noktalama **0**). `data/train_balanced_sft_v2.bin`: **yeni ad** olarak 7 jsonl kaynaktan, 1.793.433 token. Eski artefaktlar korundu; her ikisinin de kanonikliği **bağımsız yeniden koşumla bayt-özdeş** olarak doğrulandı.
* **Olay (Hat geçişi, T-0046):** `train_scientific_sft.py`, `retrain_clean_models.py`, `run_goal_pipeline.py`, `evaluate_sft_benchmarks.py` artık `_v2` okuyor; sözlük 32.852 ve model `resize_state_dict` ile hizalanıyor (`[SOZLESME_UYARI]` ile **görünür** +36 satır padding).
* **Olay (Donmuş-yazım kapıları, T-0047…T-0049):** Dört eğitim girişi (`run_goal_pipeline.py`, `train.py`, `train_dpo.py`, `retrain_clean_models.py`) donmuş desenlere yazmadan önce kapı çağırıyor ve `--allow-frozen-write` onayı istiyor; kanonik yardımcı `src/llm/frozen_guard.check_frozen_save_path`. Her tur **iki bağımsız mutantla** (sayı + sıra) non-vakum olduğu kanıtlandı.
* **Olay (F3, T-0050/T-0052):** Önce beş checkpoint'in **yedekleri** alındı (`scratch/f3_backup_2026-09-16/`, sha-özdeş + `torch.load` ile açılabilirlik kanıtı). Ardından 2×1000 adımlık koşum: `kristal_model_f3_sft.pt` (A: sft_v2) ve `kristal_model_f3.pt` (B: chat_balanced), MPS'te.
* **Olay (BULGU — `<PAD>` yozlaşması, T-0052/T-0053):** İnce ayar sonrası modelin **en sık ürettiği token `<PAD>`** oldu (tabanda ilk 5'te yok). Kontrollü **ikiz deney** (aynı taban/veri/adım, yalnız `--no-pad-mask` farkı): maskesizde PAD **39-65 (en sık 1.)**, maskelide **0** → `train.py`'ye `<PAD>` kayıp maskesi eklendi (varsayılan **aktif**). Ayrıca kayıp **ölçeğinin** maske durumuna bağlı olduğu ölçüldü (aynı rejimde 1,5836 vs 6,8515) → eski kayıp değerleriyle kıyaslama geçersiz.
* **Olay (Test):** `tests/test_pad_loss_mask.py` — maskenin iki yönlü kanıtı + regresyon; paket **195 passed** (sandbox dışında).
* **Sonuç:** Hat `_v2` ile çalışıyor ve donmuş yazımlar kapılı; `kristal_model_f3.pt` **F4 tabanı olarak kullanılmamalı** (maskesiz reçete), maskeli yeniden koşum (`f3_clean`) esas alınmalı. Ölçülmüş operasyonel ders: eğitim koşumları sırasında makinede GPU tüketen başka iş çalıştırılmamalı (adım süresi 0,43 → 3,45 sn/adım).
