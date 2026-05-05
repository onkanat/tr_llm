(Taşıyıcı Sistem)
Sorumluluk: Karpathy'nin 100 dolarlık Nanochat (veya eşdeğeri optimize GPT-2/LLaMA) projesine Kristal Katman'ı entegre etmek
.
Tokenizer Değişimi: Sistemin orijinal tiktoken (BPE) yapısını söküp, yerine KristalTokenizer wrapper'ını (encode/decode) yerleştirir
.
Maliyet Optimizasyonu: config.py içinde vocab_size değerini 50.000+ yerine 20500 olarak ayarlayıp embedding matrisini daraltır, böylece daha derin modellerin tek GPU'ya sığmasını sağlar
.
Hızlı Döngü: Hızlı testler için model, prepare.py üzerinden saf morfem ID dizileri (token_vector) ile beslenerek dakikalar içinde "pre-train" test koşularına (speedruns) sokulur
.
