import os
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def run_simulation():
    print("="*50)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: CÜMLE TEST SİMÜLASYONU")
    print("="*50)

    # 1. ONTOLOJİ (KRİSTAL KATMAN) BAŞLATILIYOR
    print("\n[1] Kristal Katman (Derleyici) Başlatılıyor...")
    lexicon = LexiconManager()
    
    # Kökleri yüklüyoruz (Zaten oluşturduğumuz roots.tsv dosyasından)
    lexicon_path = "data/lexicon/roots.tsv"
    if not os.path.exists(lexicon_path):
        print(f"Hata: {lexicon_path} bulunamadı!")
        return
        
    lexicon.load_from_tsv(lexicon_path)
    print("  -> Sözlük yüklendi ve fonetik meta veriler (yumuşama vb.) işlendi.")
    
    graph = build_default_graph()
    print("  -> Morfotaktik Durum Makinesi (State Machine) oluşturuldu.")
    
    compiler = CrystalCompiler(lexicon, graph)
    
    # 2. EPİSTEMOLOJİ (TAŞIYICI SİSTEM) BAŞLATILIYOR
    print("\n[2] Tokenizer Başlatılıyor...")
    vocab = Vocabulary()
    tokenizer = KristalTokenizer(compiler, vocab)
    
    # 3. TEST CÜMLESİ
    test_sentence = "Ali'nin benim için söylediklerini duyduktan sonra koşa koşa kavga etmeye gidiyordum. yolun yarısında söylediklerinin çoğunun doğru olduğunu fark ettim."
    print("\n[3] Bağlam KristalTokenizer ile Derleniyor (LLM'e Hazırlık)...")
    
    encoded_ids = tokenizer.encode(test_sentence)
    decoded_tags = tokenizer.decode(encoded_ids)
    
    print(f"\n  GİRDİ METNİ:")
    print(f"  {test_sentence}")
    
    print(f"\n  KRİSTAL DERLEYİCİ ÇIKTISI (Anlamsal Etiketler):")
    print(f"  {decoded_tags}")
    
    print(f"\n  LLM'E BESLENECEK VEKTÖR DİZİSİ (Token IDs):")
    print(f"  {encoded_ids}")
    
    print("\n" + "="*50)
    print(" SİMÜLASYON BAŞARIYLA TAMAMLANDI")
    print("="*50)

if __name__ == "__main__":
    run_simulation()