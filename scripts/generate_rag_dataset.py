import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_memory import VectorMemory
from src.compiler.lexicon import LexiconManager
from src.compiler.core import CrystalCompiler
from src.compiler.morphotactics import build_default_graph
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def generate_rag_dataset():
    print("=" * 60)
    print(" VEKTÖREL GEZGİN: SENTETİK RAG SFT VERİ ÜRETİCİ")
    print("=" * 60)

    # 1. Initialize Lexicon, Compiler & Tokenizer
    print("[1] Kristal Derleyici ve Tokenizer yükleniyor...")
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Connect to Qdrant Vector Memory
    print("[2] Qdrant vektörel belleğe bağlanılıyor...")
    is_fallback = False
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    except Exception as e:
        print(f"Uyarı: Qdrant bağlantısı başarısız, geçici veri oluşturulacak. Hata: {e}")
        is_fallback = True

    documents = []

    if not is_fallback:
        print("Qdrant belleğinden dökümanlar okunuyor...")
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
                payload = r.payload
                text = payload.get("text", "")
                tags = payload.get("crystal_tags", "")
                if text and tags:
                    documents.append((text, tags))
            if not next_page:
                break
    else:
        # Fallback raw data from gts.json
        print("GTS sözlüğünden örnekler yükleniyor...")
        gts_path = 'data/poems/gts.json'
        if os.path.exists(gts_path):
            with open(gts_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if idx >= 500: break
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        madde = obj.get("madde", "").strip()
                        anlamlar = obj.get("anlamlarListe", [])
                        for anlam in anlamlar:
                            for ornek in anlam.get("orneklerListe", []):
                                txt = ornek.get("ornek", "").strip()
                                if txt:
                                    t_ids = tokenizer.encode(txt)
                                    tags = tokenizer.decode(t_ids)
                                    documents.append((txt, tags))
                    except:
                        continue

    print(f"Toplam {len(documents)} adet kaynak döküman sağlandı.")
    
    rag_dataset = []
    
    # 3. Process documents to generate RAG query-document pairs
    print("[3] Morfolojik RAG çiftleri üretiliyor...")
    
    for text, tags in documents:
        clean_doc_tags = tags.replace("<BOS>", "").replace("<EOS>", "").strip()
        if not clean_doc_tags:
            continue
            
        # Parse morphemes to extract valid query roots
        morphemes = clean_doc_tags.split()
        roots = []
        for tag in morphemes:
            if tag in ("<UNK>", "<PROPER_NOUN>", "<NUMBER>", "<SYMBOL>"):
                continue
            # Ignore suffixes
            if tag.startswith(("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_", "DERIV_")):
                continue
            if tag in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"):
                continue
            roots.append(tag)
            
        if not roots:
            continue
            
        # Generate 3 query variations of varying lengths per document:
        # 1. Single root query
        selected_roots = list(set(roots))[:1]
        for query_root in selected_roots:
            input_str = f"belge: {clean_doc_tags} sorgu: {query_root}"
            rag_dataset.append({
                "instruction": "Belgeye göre cevapla.",
                "input": input_str,
                "output": clean_doc_tags
            })
            
        # 2. Subphrase query (2 to 4 consecutive morphemes from the document)
        if len(morphemes) >= 3:
            subphrase_len = random.randint(2, min(4, len(morphemes)))
            start_idx = random.randint(0, len(morphemes) - subphrase_len)
            subphrase = " ".join(morphemes[start_idx:start_idx + subphrase_len])
            input_str = f"belge: {clean_doc_tags} sorgu: {subphrase}"
            rag_dataset.append({
                "instruction": "Belgeye göre cevapla.",
                "input": input_str,
                "output": clean_doc_tags
            })
            
        # 3. Full document query
        input_str = f"belge: {clean_doc_tags} sorgu: {clean_doc_tags}"
        rag_dataset.append({
            "instruction": "Belgeye göre cevapla.",
            "input": input_str,
            "output": clean_doc_tags
        })


    # Shuffle the dataset
    random.seed(42)
    random.shuffle(rag_dataset)

    # 4. Save to data/pedagogy/rag_dataset.jsonl
    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'rag_dataset.jsonl')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in rag_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("\n" + "=" * 50)
    print(" SENTETİK RAG VERİ SETİ BAŞARIYLA OLUŞTURULDU")
    print("=" * 50)
    print(f"Toplam Örnek Sayısı: {len(rag_dataset)}")
    print(f"Dosya Yolu: {out_path}")
    print("=" * 50)

if __name__ == '__main__':
    generate_rag_dataset()
