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
from src.compiler.core import CrystalCompiler
from src.compiler.morphotactics import build_default_graph


def run_infancy_crawler(num_pairs=5000):
    print("="*50)
    print(" Vektörel Gezgin: 'Bebeklik' (Infancy) Crawler Başlıyor")
    print("="*50)

    # 1. Hazırlık
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    
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
    seen_pos_pairs = set()

    # --- POZİTİF BAĞLARIN ORGANİK ÇIKARILMASI (CRAWLER) ---
    print("\n[1] Veritabanındaki dökümanlardan organik Pozitif bağlar çıkarılıyor...")
    next_page = None
    while True:
        results, next_page = memory.client.scroll(
            collection_name=memory.collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=next_page
        )
        for r in results:
            text = r.payload.get("text", "")
            if not text:
                continue
                
            words = text.split()
            compiled_words = []
            for w in words:
                clean_w = w.strip(".,!?\"…—«»/()-;:")
                if not clean_w:
                    continue
                if clean_w.replace(".", "").replace(",", "").isdigit() or clean_w.isnumeric():
                    continue
                compile_w = clean_w.replace("'", "")
                res = compiler.compile(compile_w)
                if res.get("analyses"):
                    best = res["analyses"][0]
                    root_morpheme = best["morphemes"][0]
                    compiled_words.append((root_morpheme["id"], root_morpheme["pos"]))
                else:
                    if clean_w[0].isupper():
                        compiled_words.append((clean_w, "PROPER_NOUN"))
                    else:
                        compiled_words.append((clean_w, "UNK"))
                        
            # Ardışık kelime örüntülerini kontrol et
            for idx in range(len(compiled_words) - 1):
                w1, pos1 = compiled_words[idx]
                w2, pos2 = compiled_words[idx+1]
                
                pattern = None
                if pos1 == "NOUN" and pos2 == "VERB":
                    pattern = "Noun+Verb"
                elif pos1 == "ADJ" and pos2 == "NOUN":
                    pattern = "Adj+Noun"
                elif pos1 == "ADV" and pos2 == "VERB":
                    pattern = "Adv+Verb"
                    
                if pattern:
                    pair_key = (w1, w2, pattern)
                    if pair_key not in seen_pos_pairs:
                        seen_pos_pairs.add(pair_key)
                        output_text = f"Pozitif anlamsal bağ ({pattern}): '{w1}' ve '{w2}'. Örnek bağlam: '{text}'"
                        positive_count += 1
                        dataset.append({
                            "instruction": f"Aşağıdaki {pattern} eşleşmesinin anlamsal sınırlarını belirle: {w1} + {w2}",
                            "input": f"{w1} {w2}",
                            "output": output_text
                        })
        if not next_page:
            break

    print(f"  -> {positive_count} adet organik Pozitif bağ çıkarıldı.")

    # --- DENGELİ NEGATİF BAĞLARIN ÜRETİLMESİ (CO-OCCURRENCE / SEMANTIC PENALTY) ---
    target_negatives = positive_count
    print(f"\n[2] Dengeli Negatif bağlar üretiliyor (Hedef: {target_negatives} adet)...")
    
    attempts = 0
    max_attempts = target_negatives * 20
    
    while negative_count < target_negatives and attempts < max_attempts:
        attempts += 1
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
            
        # Zaten pozitif listesinde varsa geç
        if (w1, w2, pattern) in seen_pos_pairs:
            continue
            
        token_ids = [
            vocab.encode('<BOS>'), 
            vocab.encode(w1), 
            vocab.encode(w2), 
            vocab.encode('<EOS>')
        ]
        tags_str = f"<BOS> {w1} {w2} <EOS>"
        dense_vec = generate_kristal_vector(token_ids, tags_str)
        
        results = memory.dense_recall(dense_vec, top_k=1)
        top_score = results[0]['score'] if results else 0.0
        
        # Eğer en benzer döküman skoru çok düşükse, negatif olarak doğrula
        if top_score < 0.18:
            output_text = f"Negatif çelişik bağ ({pattern}): *'{w1}' ile '{w2}' bir araya gelmez. Uzayda yankı bulunamadı."
            negative_count += 1
            dataset.append({
                "instruction": f"Aşağıdaki {pattern} eşleşmesinin anlamsal sınırlarını belirle: {w1} + {w2}",
                "input": f"{w1} {w2}",
                "output": output_text
            })
            
            if negative_count % 200 == 0:
                print(f"  -> {negative_count} adet Negatif bağ üretildi...")

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
