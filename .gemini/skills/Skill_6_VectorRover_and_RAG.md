(Taşıyıcı Sistem)
Sorumluluk: Vektörel Gezgin ajanının Algoritma 2'ye göre çalışmasını sağlamak
.
Boru Hattı: q sorgusu geldiğinde, Evrensel Vektör Bellek'ten (VectorRecall) anlamsal Öklid/Kosinüs yakınlığına göre bağlamı çağırır
.
Kristalizasyon: Gelen bağlamı doğrudan LLM'e vermek yerine CrystalCompile(ctx) adımından geçirip Cpk nesnelerine ayırır
.
Akıl Yürütme (Reasoning): Ana LLM'e (Reason), anlamsız diziler değil [1001, 502, 550...] gibi token_vector değerlerini besleyerek RAG'in anlamsal sapmalarını ve halüsinasyonları engeller
.
