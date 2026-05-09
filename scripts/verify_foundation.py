import sys
import os
import json

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def verify_foundation():
    lexicon_path = 'data/lexicon/roots.tsv'
    
    print("[1] Kristal Sistem Hazırlanıyor...")
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    tokenizer = KristalTokenizer(compiler, vocab)

    test_sentences = [
        "Akmayan su",
        "olduğu yerde",
        "pıhtılaşan ağırlaşan su",
        "ölümü hatırlatır bana",
        "kitabı okudum",
        "gözlüğünü temizledi",
        "gidecekti ama gelmedi"
    ]

    print("\n[2] Temel Doğrulama Testleri Başlatılıyor...")
    print("-" * 60)
    
    for sent in test_sentences:
        print(f"\nSorgu: '{sent}'")
        token_ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(token_ids)
        
        # Analiz detaylarını göster (sadece ilk kelime için örnek)
        first_word = sent.split()[0].strip(",.")
        analysis = compiler.compile(first_word.lower())
        
        print(f"  -> Kristal Pack: {decoded}")
        if not analysis['analyses']:
             print(f"  !! HATA: '{first_word}' derlenemedi!")
        else:
             best = analysis['analyses'][0]
             morphemes = " + ".join([f"{m['surface']}({m['id']})" for m in best['morphemes']])
             print(f"  -> Morfoloji: {morphemes}")

    print("\n" + "="*60)
    print(" DOĞRULAMA TAMAMLANDI")
    print("="*60)

if __name__ == "__main__":
    verify_foundation()
