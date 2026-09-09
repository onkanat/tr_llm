import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.rag.vector_memory import VectorMemory
from scripts.import_examples import generate_kristal_vector, generate_sparse_vector

def main():
    print("=" * 60)
    print(" RAG HAFIZA GÜNCELLEME: SFT EĞİTİM VERİLERİ İNDEKSLEME SİSTEMİ")
    print("=" * 60)

    # 1. Initialize Lexicon, Compiler & Tokenizer
    print("[1] Kristal Derleyici ve Tokenizer yükleniyor...")
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    tokenizer = KristalTokenizer(compiler, vocab)
    
    # 2. Connect to VectorMemory
    print("[2] Qdrant vektörel belleğe bağlanılıyor...")
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    except Exception as e:
        print(f"Hata: Qdrant sunucusuna bağlanılamadı: {e}")
        return

    texts = []
    dense_vectors = []
    sparse_vectors = []
    metadatas = []

    # 3. Read and encode middle_school_chat.jsonl
    chat_path = 'data/pedagogy/middle_school_chat.jsonl'
    print(f"\n[3] '{chat_path}' okunuyor ve kodlanıyor...")
    if os.path.exists(chat_path):
        with open(chat_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                inst = item["instruction"].strip()
                inp = item["input"].strip()
                out = item["output"].strip()
                
                # SFT Input is the search vector trigger
                query_text = inp if inp else inst
                q_token_ids = tokenizer.encode(query_text)
                q_tags = tokenizer.decode(q_token_ids)
                
                dense_vec = generate_kristal_vector(q_token_ids, q_tags)
                sparse_vec = generate_sparse_vector(q_token_ids, q_tags)
                
                # SFT Output is the retrieved document context
                doc_text = out
                doc_token_ids = tokenizer.encode(doc_text)
                doc_tags = tokenizer.decode(doc_token_ids)
                
                texts.append(doc_text)
                dense_vectors.append(dense_vec)
                sparse_vectors.append(sparse_vec)
                metadatas.append({
                    "domain": "pedagogy_chat",
                    "system_message": inst,
                    "query_text": inp,
                    "crystal_tags": doc_tags,
                    "token_ids": doc_token_ids
                })
        print(f"  -> {len(texts)} adet sohbet çifti belleğe hazırlandı.")
    else:
        print(f"  -> Hata: '{chat_path}' bulunamadı!")

    # 4. Read and encode parenting_dataset.jsonl (first 500 items)
    parenting_path = 'data/pedagogy/parenting_dataset.jsonl'
    print(f"\n[4] '{parenting_path}' içerisinden ilk 500 örnek kodlanıyor...")
    parenting_count = 0
    if os.path.exists(parenting_path):
        with open(parenting_path, 'r', encoding='utf-8') as f:
            for line in f:
                if parenting_count >= 500: break
                if not line.strip(): continue
                item = json.loads(line)
                inst = item["instruction"].strip()
                inp = item["input"].strip()
                out = item["output"].strip()
                
                query_text = inp if inp else inst
                q_token_ids = tokenizer.encode(query_text)
                q_tags = tokenizer.decode(q_token_ids)
                
                dense_vec = generate_kristal_vector(q_token_ids, q_tags)
                sparse_vec = generate_sparse_vector(q_token_ids, q_tags)
                
                doc_text = out
                doc_token_ids = tokenizer.encode(doc_text)
                doc_tags = tokenizer.decode(doc_token_ids)
                
                texts.append(doc_text)
                dense_vectors.append(dense_vec)
                sparse_vectors.append(sparse_vec)
                metadatas.append({
                    "domain": "pedagogy_parenting",
                    "system_message": inst,
                    "query_text": inp,
                    "crystal_tags": doc_tags,
                    "token_ids": doc_token_ids
                })
                parenting_count += 1
        print(f"  -> {parenting_count} adet parenting örneği belleğe hazırlandı.")
    else:
        print(f"  -> Hata: '{parenting_path}' bulunamadı!")

    # 4.5. Read and encode DPO preference dataset
    dpo_path = '/Users/hakankilicaslan/Prompts/Outputs/datasets/dpo_preference_turkce.jsonl'
    print(f"\n[4.5] '{dpo_path}' okunuyor ve kodlanıyor...")
    dpo_count = 0
    if os.path.exists(dpo_path):
        with open(dpo_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                prompt_text = item.get("prompt", "").strip()
                chosen_text = item.get("chosen", "").strip()
                if not prompt_text or not chosen_text:
                    continue
                
                # SFT Input is the search vector trigger
                query_text = prompt_text
                q_token_ids = tokenizer.encode(query_text)
                q_tags = tokenizer.decode(q_token_ids)
                
                dense_vec = generate_kristal_vector(q_token_ids, q_tags)
                sparse_vec = generate_sparse_vector(q_token_ids, q_tags)
                
                doc_text = chosen_text
                doc_token_ids = tokenizer.encode(doc_text)
                doc_tags = tokenizer.decode(doc_token_ids)
                
                texts.append(doc_text)
                dense_vectors.append(dense_vec)
                sparse_vectors.append(sparse_vec)
                metadatas.append({
                    "domain": "pedagogy_dpo",
                    "system_message": prompt_text,
                    "query_text": prompt_text,
                    "crystal_tags": doc_tags,
                    "token_ids": doc_token_ids
                })
                dpo_count += 1
        print(f"  -> {dpo_count} adet DPO örneği belleğe hazırlandı.")
    else:
        print(f"  -> Hata: '{dpo_path}' bulunamadı!")

    # 5. Upsert to VectorMemory
    if texts:
        batch_size = 50
        total_inserted = len(texts)
        print(f"\n[5] Toplam {total_inserted} nokta Qdrant '{memory.collection_name}' koleksiyonuna ekleniyor...")
        for idx in range(0, total_inserted, batch_size):
            end_idx = min(idx + batch_size, total_inserted)
            b_texts = texts[idx:end_idx]
            b_dense = dense_vectors[idx:end_idx]
            b_sparse = sparse_vectors[idx:end_idx]
            b_meta = metadatas[idx:end_idx]
            memory.add_documents_batch(b_texts, b_dense, b_sparse, b_meta)
            print(f"  -> {end_idx}/{total_inserted} nokta eklendi...")
        print("  -> Başarıyla Qdrant veritabanına eklendi.")
    else:
        print("  -> Uyarı: İndekslenecek veri bulunamadı!")
        
    print("=" * 60)
    print(" RAG HAFIZA GÜNCELLEMESİ TAMAMLANDI")
    print("=" * 60)

if __name__ == '__main__':
    main()
