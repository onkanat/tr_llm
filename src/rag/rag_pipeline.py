import os
import sys
import json
import torch
import numpy as np

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import build_rag_input
from scripts.train_step_demo import KristalLM
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector

def main():
    print("=" * 60)
    print(" VEKTÖREL GEZGİN: UÇTAN UCA RAG (ARAMA-ÜRETİM) HATTI SİMÜLASYONU")
    print("=" * 60)

    # 1. Load compiler modules
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    
    # 2. Load vocabulary & tokenizer
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    tokenizer = KristalTokenizer(compiler, vocab)
    
    # 3. Connect to Qdrant VectorMemory
    print("\n[1] Qdrant vektörel belleğe bağlanılıyor...")
    is_fallback = False
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    except Exception as e:
        print(f"Uyarı: Qdrant sunucusuna bağlanılamadı. Geçici bellek (In-Memory) modunda çalışılıyor. Detay: {e}")
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768)
        is_fallback = True
        
    if is_fallback:
        print("  -> Geçici belleğe örnek belgeler yükleniyor...")
        sample_texts = [
            "Okul müdürüyken okulun ek inşaatında hamallarla birlikte çalışmış.",
            "Su düzeyi.",
            "Kitap okumak insanı geliştirir."
        ]
        batch_dense = []
        batch_sparse = []
        batch_meta = []
        for text in sample_texts:
            t_ids = tokenizer.encode(text)
            t_tags = tokenizer.decode(t_ids)
            batch_dense.append(generate_kristal_vector(t_ids, t_tags))
            batch_sparse.append(generate_sparse_vector(t_ids, t_tags))
            batch_meta.append({
                "domain": "gts_sozluk",
                "crystal_tags": t_tags,
                "token_ids": t_ids
            })
        memory.add_documents_batch(sample_texts, batch_dense, batch_sparse, batch_meta)
        
    # 4. Load trained model
    print("[2] Eğitilmiş dil modeli yükleniyor...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    model_path = 'data/kristal_model.pt'
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası '{model_path}' bulunamadı! Lütfen önce eğitimi tamamlayın.")
        return
        
    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    # Resize state dict if vocab changed
    new_state_dict = model.state_dict()
    for k, v in old_v_items := list(state_dict.items()):
        if k in new_state_dict and v.shape != new_state_dict[k].shape:
            if len(v.shape) == 2:
                new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])] = v[:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])]
            elif len(v.shape) == 1:
                new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0])] = v[:min(v.shape[0], new_state_dict[k].shape[0])]
            state_dict[k] = new_state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  -> Model {device} cihazına başarıyla taşındı.")

    # 5. Read prompt query
    query = "okul"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    print(f"\n[Girdi Sorgu]: '{query}'")

    # Tokenize query
    query_token_ids = tokenizer.encode(query)
    query_tags = tokenizer.decode(query_token_ids)
    print(f"  -> Sorgu Morfemleri: {query_tags}")
    
    # Generate vectors
    dense_vec = generate_kristal_vector(query_token_ids, query_tags)
    sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
    
    # Retrieve top 1 document
    print("\n[3] Vektörel bellekten en uyumlu döküman geri çağrılıyor (Retrieval)...")
    results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
    
    if not results:
        print("  -> Uyarı: Hiçbir döküman bulunamadı!")
        return
        
    doc = results[0]
    doc_text = doc["text"]
    doc_tags = doc["metadata"].get("crystal_tags", "")
    print(f"  -> Eşleşen Belge: '{doc_text}'")
    print(f"  -> Belge Morfemleri: {doc_tags}")
    print(f"  -> Eşleşme Skoru (RRF + Penalty): {doc['score']:.4f}")

    # 6. Construct highly compact prompt
    input_str = build_rag_input(doc_text, query)
    
    sft_item = {
        "instruction": "Belgeye göre cevapla.",
        "input": input_str,
        "output": ""
    }
    
    # Tokenize prompt
    prompt_json = json.dumps(sft_item, ensure_ascii=False)
    prompt_token_ids = tokenizer.encode(prompt_json)
    
    # Find <OUTPUT> position
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    if output_start_id not in prompt_token_ids:
        print("Hata: Prompt içinde <OUTPUT> belirteci bulunamadı!")
        return
        
    output_idx = prompt_token_ids.index(output_start_id)
    eval_tokens = prompt_token_ids[:output_idx + 1]
    
    print(f"\n[4] RAG Prompt Oluşturuldu (Uzunluk: {len(eval_tokens)}):")
    print(f"  -> Model Girdisi: {tokenizer.decode(eval_tokens)}")

    # 7. Autoregressive Greedy Decoding
    print("\n[5] Model Yanıt Üretiyor (Generation)...")
    eos_id = vocab.stoi.get("<EOS>", -1)
    generated_tokens = []
    max_new_tokens = 60
    
    current_tokens = list(eval_tokens)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            x_input = torch.tensor([current_tokens], dtype=torch.long, device=device)
            logits, _ = model(x_input)
            next_token_logits = logits[0, -1, :]
            
            # Apply repetition penalty only to output tokens within the sliding window
            if generated_tokens:
                window_tokens = generated_tokens[-12:]
                for token_id in set(window_tokens):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= 1.5
                    else:
                        next_token_logits[token_id] *= 1.5

                        
            next_token_id = torch.argmax(next_token_logits).item()
            
            if next_token_id == eos_id:
                break
                
            generated_tokens.append(next_token_id)
            current_tokens.append(next_token_id)
            
            if len(current_tokens) >= 128:
                print("  -> Uyarı: Maksimum sekans uzunluğuna (128) ulaşıldı!")
                break

                
    response_tags = tokenizer.decode(generated_tokens)
    print(f"\n[6] Üretilen Yanıt Morfemleri:")
    print(f"  -> {response_tags}")
    print("=" * 60)

if __name__ == '__main__':
    main()
