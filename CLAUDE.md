# CLAUDE Yönergesi (CLAUDE.md)

Bu dosya, projede komut çalıştırma, kodlama standartları ve temel güvenlik kısıtlarını belirler.

---

## 🛠️ Temel Komutlar

### 1. Doğrulama ve Testler
* **Birim Testlerini Çalıştır:** `venv/bin/pytest` (**19 Eyl 2026 ölçümü: 213 passed, 1 failed**; düşen tek test `tests/test_agent_gateway.py::TestAgentGateway::test_gateway_http_server_endpoints` — nedeni `socketserver` `PermissionError`, yani **sandbox soket yasağı**; sandbox dışında geçer. Sayı her koşumda yeniden ölçülür, buradaki değer **damgalıdır** ve bayatlayabilir.)
* **Model Çıkarım Testi (Bilinen Boyut Sınırlılığı):** `venv/bin/python test_model.py`
  > [!WARNING]
  > `test_model.py` betiği modeli hâlâ `data/vocab.json` (**31.357**, ölçüldü) boyutuyla başlatır; güncel checkpoint'ler (ör. **33.114** satırlı `data/anka_a1r.pt`) ile şekil uyuşmazlığı `RuntimeError` verir. Uyumlu çıkarım için `src.llm.prompt_contract.resize_state_dict` kullanılmalıdır (T-0027 K7/F1).
  > **Kristal zinciri SİLİNDİ** (operatör kararı, 18 Eyl 2026): `kristal_model.pt` ve türevleri artık yoktur; model **Anka**'dır ve checkpoint'ler `data/anka_*.pt` altındadır. Bu satırdaki eski `kristal_model.pt` referansı ölüydü (D4, T-0081).
  > **Taban sözlük ile külliyat sözlüğü AYRI şeylerdir (ölçüldü, 19 Eyl 2026).** *Taban:* `data/rebuild/vocab_base_32852.json` = **32.852** giriş — ve `train.py:66`'nın **varsayılanıdır** (`33114` bu dosyada hiç geçmez, `grep -c` = 0). *Külliyat:* güncel A1-r külliyatı `data/rebuild/vocab_anka_r1_33114.json` = **33.114** giriş ister ve **`--vocab` ile açıkça verilmelidir**. Sözlük külliyat için küçük kalırsa `train.py:112-116` `RuntimeError` ile **yüksek sesle durur** (bu yol sessiz DEĞİL); kardeş `<bin>.meta.json` varsa digest/giriş cross-check'i de yapılır (bayat `.bin` sınıfı). Checkpoint `resize_state_dict` ile hizalanır — uyumsuzluk `[SOZLESME_UYARI]` satırıyla **görünür** basılır, sessiz kırpma yoktur (T-0046).

### 2. Derleme ve Eğitim
* **Wikipedia Veri Kümesini Hazırla:** `venv/bin/python scripts/prepare_wiki_dataset.py`
* **5 Fazlı Birleşik Veri Kümesini Derle:** `venv/bin/python scripts/prepare_all_phases_with_wiki.py`
* **Modeli Eğit (MPS GPU):** `venv/bin/python train.py --pretrain --vocab <sözlük.json> --device mps --data <bin> --steps N --load-path <ckpt> --save-path <yeni-ad> --allow-frozen-write --loss-report`
  > [!WARNING]
  > **`--pretrain` ZORUNLU — sessiz sıfır-kayıp tuzağı (D1, T-0081'nin en pahalı maddesi):** düz metin külliyatta SFT kayıp maskesi her pencereyi maskeler ⇒ gradyan 0 olur ve **MPS hata VERMEZ, `0,0000` basar**; 12,6 saatlik koşu tek bir uyarı bile vermeden boşa gider (T-0073/T-0074). **Canlılık kapısı:** **sıfırdan** koşumda ilk kayıp ≈ `ln V` olmalı (33.114 için ≈ 10,4076); **devam** koşumunda bu bant geçerli DEĞİLDİR — imza **koşum moduna bağlıdır** (T-0077).
  > **`--loss-report` ile kayıp referansı bir SAYI değil DAĞILIMDIR:** `Bitiş Kaybı` tek adımın örneklemidir; karşılaştırma `son60` ort ± std ile yapılır (T-0078).
  > **`--vocab` külliyatla eşleşmelidir:** varsayılan 32.852'dir ama güncel A1-r külliyatı **33.114** ister (yukarıdaki sözlük notu). Eşleşmezse koşum **başlamadan durur** — sessiz değil.
  > **Donmuş yola yazım operatör onayı ister:** `train.py`, `train_dpo.py`, `run_goal_pipeline.py` ve `retrain_clean_models.py` `data/*.pt` gibi donmuş desenlere yazmadan önce `check_frozen_save_path` çağırır; onay verilmezse `RuntimeError` ile **durur**. Onay `--allow-frozen-write` ile açıkça verilir ve hedef `writes[]`'te beyan edilip üst dizin kiralanmalıdır (T-0047…T-0049).
  > **`<PAD>` kayıp maskesi varsayılan olarak AKTİF:** hizalama dolgusu eğitim hedefi sayılmaz (`--no-pad-mask` ile kapanır). Maskeleme kayıp **ölçeğini** değiştirir (aynı rejimde ~1,58 vs ~6,85) → eski kayıp değerleriyle kıyaslama yapılmamalıdır (T-0053).
  > **Koşum sırasında makinede GPU tüketen başka iş çalıştırmayın:** aynı makinede tarayıcı/video açıkken adım süresi 0,43 → 3,45 sn/adım'a çıktı (T-0052).

---

## 🛡️ Proje Güvenlik Kısıtları ve Sözleşme Çıpası

Bu projenin en sert kısıtları canlı sözleşme belgesi olan [`.agent-bus/SPEC.md`](.agent-bus/SPEC.md) üzerinde tanımlıdır. Çift başlılık ve kural sapmalarını önlemek için standartlar burada yinelenmez; yürütücüler aşağıdaki kurallara kesin olarak uymakla yükümlüdür:

1. **Yazmadan Önce Kiralama Zorunluluğu (SPEC.md Kural 1):** Bir görevin `writes[]` listesinde olmayan veya `bus_acquire_lease` ile kiralanmamış hiçbir dosya/dizin değiştirilemez.
2. **Donmuş Yollar Salt-Okunurdur (SPEC.md Kural 2):** [`.agent-bus/frozen.json`](.agent-bus/frozen.json) içindeki **10 desen** (`data/realistic_rag/**`, `data/b1_5_splits/**`, `data/pedagogy_canonical/**`, `data/lexicon/**`, `data/*.pt`, `data/*.bin`, `data/vocab.json`, `data/vocab_entity.json`, `src/llm/tokenizer.py`, `src/compiler/**`) asla izinsiz ve kiralamasız değiştirilemez. Liste `frozen.json` ile **iki yönlü** hizalanmıştır; kaldırılan geniş `data/vocab…json` kalıbı yerine **iki açık girdi** yazılmıştır (D5, T-0081) — çünkü o kalıp, `frozen.json`'da **karşılığı olmayan** `data/vocab_<başka>.json` dosyalarını da donmuş *sayardı*. Yetki kuralı: donmuş bir **dosya** kiralanamaz ⇒ **üst dizinini** kirala ve dosyayı `writes[]`'te beyan et.
3. **Veri Dizini Koruma:** `data/**` altındaki tüm yollar salt-okunurdur; tek istisna değerlendirme ve ölçüm raporlarının yazıldığı `data/eval/` dizinidir.
4. **Git Bütünlüğü:** `git add -A` veya `git add .` kullanımı kesinlikle YASAKTIR (büyük ve izlenmeyen veri dosyalarının sahnelenmesini engellemek için). Yetkisiz commit veya push yapılamaz.

---

## ✒️ Kodlama Standartları

* **Tip Güvenliği:** Python type hints (`List`, `Dict`, `Any`, `str` vb.) kullanılması zorunludur.
* **Fonksiyon Tasarımı:** Sistem modülleri global durum (mutable state) barındırmayan saf fonksiyonlar (pure functions) olarak yazılmalıdır.
* **Hata Yönetimi:** Beklenmeyen durumlar ve OOV (Out-of-Vocabulary) hataları sessizce geçiştirilmemeli, loglanmalı ve uygun varsayılan değerler dönmelidir.
* **Kararlılık (Determinism):** Algoritmalar deterministik çalışmalı, olasılıksal yaklaşımlar yerine kural tabanlı yapılar tercih edilmelidir.
