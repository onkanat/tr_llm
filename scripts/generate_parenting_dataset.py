import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_memory import VectorMemory
from src.compiler.lexicon import LexiconManager
from src.compiler.core import CrystalCompiler
from src.compiler.morphotactics import build_default_graph

def run_parenting_dataset_generator():
    print("=" * 60)
    print(" VEKTÖREL GEZGİN: 'EBEVEYNLİK' (PARENTING SFT) VERİ ÜRETİCİ")
    print("=" * 60)

    # 1. Initialize Lexicon, Graph and Compiler
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)

    # 2. Connect to Qdrant/Vector Memory to pull actual corpus texts
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {e}")
        return

    # 3. Pull documents and extract words
    print("Veritabanından dökümanlar okunuyor...")
    words_to_analyze = set()
    next_page = None
    
    # Scroll through Qdrant
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
            for w in text.split():
                clean_w = w.strip(".,!?\"…—«»/()-;:")
                if clean_w and not clean_w.replace(".", "").replace(",", "").isdigit():
                    words_to_analyze.add(clean_w)
        if not next_page:
            break

    print(f"Toplam {len(words_to_analyze)} benzersiz kelime çıkarıldı. Derleniyor...")

    dataset = []
    task_counts = {
        "root": 0,
        "tense": 0,
        "case": 0,
        "plural": 0
    }

    # 4. Compile words and generate SFT tasks
    for word in words_to_analyze:
        # Ignore extremely short or capitalized words for standard tasks
        if len(word) < 3 or word[0].isupper():
            continue
            
        compile_w = word.replace("'", "").replace("’", "")
        res = compiler.compile(compile_w)
        
        if not res.get("analyses"):
            continue
            
        # Get the best morphological analysis
        best_analysis = res["analyses"][0]
        morphemes = best_analysis["morphemes"]
        
        if not morphemes:
            continue
            
        # Extract root
        root_morpheme = morphemes[0]
        root_id = root_morpheme["id"]
        
        # Build list of suffix tags
        suffix_ids = [m["id"] for m in morphemes[1:]]
        
        # A. Root Detection Task (Generate for all compiled words)
        dataset.append({
            "instruction": "Kelimedeki kök morfemini bul.",
            "input": word,
            "output": f"ROOT: {root_id}"
        })
        task_counts["root"] += 1

        # B. Tense Detection Task
        tenses = [sid for sid in suffix_ids if sid.startswith("TENSE_")]
        if tenses:
            dataset.append({
                "instruction": "Kelimedeki eylemin zamanını veya kipini tespit et.",
                "input": word,
                "output": f"TENSE: {', '.join(tenses)}"
            })
            task_counts["tense"] += 1

        # C. Case Suffix Detection Task
        cases = [sid for sid in suffix_ids if sid.startswith("CASE_")]
        if cases:
            dataset.append({
                "instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
                "input": word,
                "output": f"CASE: {', '.join(cases)}"
            })
            task_counts["case"] += 1

        # D. Plural Detection Task
        has_plural = "PLURAL" in suffix_ids
        # Let's generate a binary plural detection task
        dataset.append({
            "instruction": "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.",
            "input": word,
            "output": "PLURAL: Evet" if has_plural else "PLURAL: Hayır"
        })
        task_counts["plural"] += 1

    # Shuffle dataset to mix task types
    random.shuffle(dataset)

    # 5. Export to parenting_dataset.jsonl
    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'parenting_dataset.jsonl')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 50)
    print(" EBEVEYNLİK VERİ SETİ BAŞARIYLA OLUŞTURULDU")
    print("=" * 50)
    print(f"Toplam Örnek Sayısı: {len(dataset)}")
    print(f"  -> Kök Bulma Görevi:        {task_counts['root']}")
    print(f"  -> Zaman Tespiti Görevi:    {task_counts['tense']}")
    print(f"  -> Durum Eki Görevi:        {task_counts['case']}")
    print(f"  -> Çoğul Eki Görevi:        {task_counts['plural']}")
    print(f"Dosya Yolu: {out_path}")
    print("=" * 50)

if __name__ == '__main__':
    run_parenting_dataset_generator()
