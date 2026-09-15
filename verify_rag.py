import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector
from src.llm.prompt_contract import resize_state_dict, build_rag_input

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Vocab, Compiler, Tokenizer
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Load Memory with resilient fallback
    memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    if memory.is_in_memory or memory.client.count(memory.collection_name).count == 0:
        seed_doc = "Bana göre ölümün en büyük vasfı durgunluk, hareketsizliktir."
        d_ids = tokenizer.encode(seed_doc)
        d_tags = tokenizer.decode(d_ids)
        d_dense = generate_kristal_vector(d_ids, d_tags)
        d_sparse = generate_sparse_vector(d_ids, d_tags)
        memory.add_document(
            text=seed_doc,
            dense_vector=d_dense,
            sparse_vector=d_sparse,
            metadata={"crystal_tags": d_tags}
        )

    # 3. Load Model
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    
    state_dict = torch.load('data/kristal_model.pt', map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    resized_state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(resized_state_dict, strict=False)
    model.to(device)
    model.eval()



    # 4. Run multiple test cases
    test_cases = [
        {
            "name": "Original Case (with UNK and multi-root query)",
            "query_text": "Durgun su ölümü hatırlatır",
            "doc_text": "Bana göre ölümün en büyük vasfı durgunluk, hareketsizliktir."
        },
        {
            "name": "Clean Case (no UNK, single-root query)",
            "query_text": "cahil",
            "doc_text": "bu maskara sosyete bana cahil diye bak TENSE_AORIST_VOWEL"
        },
        {
            "name": "Clean Case (no UNK, multi-root query)",
            "query_text": "sosyete bana bak",
            "doc_text": "bu maskara sosyete bana cahil diye bak TENSE_AORIST_VOWEL"
        }
    ]

    for tc in test_cases:
        print(f"\n==========================================")
        print(f" TEST CASE: {tc['name']}")
        print(f"==========================================")
        
        q_ids = tokenizer.encode(tc["query_text"])
        q_tags = tokenizer.decode(q_ids)
        clean_query_tags = q_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
        
        # If it's the original case, retrieve from memory, otherwise use the test case text
        if tc["query_text"] == "Durgun su ölümü hatırlatır":
            dense_vec = generate_kristal_vector(q_ids, q_tags)
            sparse_vec = generate_sparse_vector(q_ids, q_tags)
            results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=q_tags)
            if results:
                doc_tags = results[0]["metadata"].get("crystal_tags", "")
                doc_text = results[0]["text"]
            else:
                doc_tags = ""
                doc_text = ""
        else:
            doc_ids = tokenizer.encode(tc["doc_text"])
            doc_tags = tokenizer.decode(doc_ids)
            doc_text = tc["doc_text"]
            
        print(f"Retrieved Document: {doc_text}")
        print(f"Doc Tags: {doc_tags}")
        
        input_str = build_rag_input(doc_text, tc["query_text"])
        prompt_dict = {
            "instruction": "Belgeye göre cevapla.",
            "input": input_str,
            "output": ""
        }
        
        raw_prompt = json.dumps(prompt_dict)
        token_ids = tokenizer.encode(raw_prompt)
        output_start_id = vocab.stoi.get("<OUTPUT>", -1)
        output_idx = token_ids.index(output_start_id)
        eval_tokens = token_ids[:output_idx + 1]
        
        print("Prompt Morphemes:")
        print(" ".join([vocab.decode(tid) for tid in eval_tokens]))

        # 5. Generate with different temperatures
        eos_id = vocab.stoi.get("<EOS>", -1)
        
        for temp in [0.0, 0.2, 0.7]:
            print(f"\n--- Generation (Temp={temp}) ---")
            current_tokens = list(eval_tokens)
            generated_morphemes = []
            
            with torch.no_grad():
                for step in range(60):
                    x = torch.tensor([current_tokens], dtype=torch.long, device=device)
                    logits, _ = model(x)
                    logits = logits[0, -1, :]
                    
                    # Apply repetition penalty only to output tokens within the sliding window
                    output_tokens = current_tokens[len(eval_tokens):]
                    if output_tokens:
                        window_tokens = output_tokens[-12:]
                        for token_id in set(window_tokens):
                            if logits[token_id] > 0:
                                logits[token_id] /= 1.5
                            else:
                                logits[token_id] *= 1.5

                                
                    if temp > 0.0:
                        logits = logits / temp
                        probs = torch.softmax(logits, dim=-1)
                        pred_id = torch.multinomial(probs, num_samples=1).item()
                    else:
                        pred_id = torch.argmax(logits).item()
                        
                    if pred_id == eos_id or pred_id == vocab.stoi.get("</OUTPUT>", -1):
                        break
                    
                    generated_morphemes.append(vocab.decode(pred_id))
                    current_tokens.append(pred_id)

                    
            print("Generated Output:")
            print(" ".join(generated_morphemes))

if __name__ == '__main__':
    main()
