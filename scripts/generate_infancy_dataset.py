import os
import sys
import json
import random
from typing import List, Dict

# Proje kök dizinini path'e ekleyelim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_memory import VectorMemory
from src.compiler.lexicon import LexiconManager
from src.llm.tokenizer import Vocabulary
from scripts.import_examples import generate_kristal_vector, generate_sparse_vector

def run_infancy_crawler(num_pairs=5000):
    print("="*50)
    print(" Vektörel Gezgin: 'Bebeklik' (Infancy) Crawler Başlıyor")
    print("="*50)

    # 1. Hazırlık
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {e}")
        return

    nouns = []
    verbs = []
    adjs = []
    advs = []
    
    with open('data/lexicon/roots.tsv', 'r', encoding='utf-8') as f:
        next(f) # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                lemma, pos = parts[0], parts[1]
                if vocab.encode(lemma) != 0: # 0 = UNK
                    if pos == "NOUN": nouns.append(lemma)
                    elif pos == "VERB": verbs.append(lemma)
                    elif pos == "ADJ": adjs.append(lemma)
                    elif pos == "ADV": advs.append(lemma)

    print(f"Sözlük Durumu -> İsim: {len(nouns)}, Fiil: {len(verbs)}, Sıfat: {len(adjs)}, Zarf: {len(advs)}")

    if not nouns or not verbs or not adjs or not advs:
        print("Hata: Yeterli kelime türü bulunamadı!")
        return

    dataset = []
    positive_count = 0
    negative_count = 0

    print(f"\nGezgin rastgele anlamsal ve sözdizimsel bağlar kuruyor ({num_pairs} deneme)...")
    
    for i in range(num_pairs):
        # Rastgele bir sentaktik kalıp seç
        pattern = random.choice(["Noun+Verb", "Adj+Noun", "Adv+Verb"])
        
        if pattern == "Noun+Verb":
            w1 = random.choice(nouns)
            w2 = random.choice(verbs)
        elif pattern == "Adj+Noun":
            w1 = random.choice(adjs)
            w2 = random.choice(nouns)
        else: # Adv+Verb
            w1 = random.choice(advs)
            w2 = random.choice(verbs)
            
        token_ids = [
            vocab.encode('<BOS>'), 
            vocab.encode(w1), 
            vocab.encode(w2), 
            vocab.encode('<EOS>')
        ]
        
        tags_str = f"<BOS> {w1} {w2} <EOS>"
        
        # Vektörleri oluştur
        dense_vec = generate_kristal_vector(token_ids, tags_str)
        
        # Hafızaya Sor (Pure Semantic Recall via Dense Vector)
        results = memory.dense_recall(dense_vec, top_k=1)
        
        score = 0.0
        match_text = "Yok"
        if results:
            score = results[0]['score']
            match_text = results[0]['text']

        # Anlam Sınırlarını (Semantic Boundaries) Belirle
        if score > 0.18:
            label = "GEÇERLİ (POZİTİF)"
            output_text = f"Pozitif anlamsal bağ ({pattern}): '{w1}' ve '{w2}'. Örnek bağlam: '{match_text}'"
            positive_count += 1
            dataset.append({
                "instruction": f"Aşağıdaki {pattern} eşleşmesinin anlamsal sınırlarını belirle: {w1} + {w2}",
                "input": f"{w1} {w2}",
                "output": output_text
            })
        elif score < 0.10:
            label = "GEÇERSİZ (NEGATİF)"
            output_text = f"Negatif çelişik bağ ({pattern}): *'{w1}' ile '{w2}' bir araya gelmez. Uzayda yankı bulunamadı."
            negative_count += 1
            dataset.append({
                "instruction": f"Aşağıdaki {pattern} eşleşmesinin anlamsal sınırlarını belirle: {w1} + {w2}",
                "input": f"{w1} {w2}",
                "output": output_text
            })

        if (i+1) % 500 == 0:
            print(f"  Deneme {i+1}: {w1} + {w2} ({pattern}) -> Skor: {score:.4f} [{label}]")

    print("\n" + "="*50)
    print(f" BEBEKLİK FAZI KEŞİF TAMAMLANDI")
    print(f" Pozitif Bağ: {positive_count}")
    print(f" Negatif Bağ: {negative_count}")
    
    out_path = 'data/pedagogy/infancy_dataset.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f" Alpaca Formatlı Veri Seti '{out_path}' konumuna kaydedildi.")
    print("="*50)

if __name__ == "__main__":
    # Parametre olarak sayı alınabilir
    num = 5000
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    run_infancy_crawler(num)
