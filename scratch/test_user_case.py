import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from tests.test_decompiler import MorphemeDecompiler

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
    decompiler = MorphemeDecompiler(compiler, vocab)

    # 2. Load Model
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    
    state_dict = torch.load('data/kristal_model.pt', map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    from verify_rag import resize_state_dict
    resized_state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(resized_state_dict, strict=False)
    model.to(device)
    model.eval()



    # User test case prompt morphemes
    doc_tags = "al at aygır at doru at kula at yağız at at PLURAL POSS_2SG hangisi güzel hangisi dost"
    query_tags = "al at aygır at doru at kula at yağız at at PLURAL POSS_2SG hangisi güzel hangisi dost"
    
    input_str = f"belge: {doc_tags} sorgu: {query_tags}"
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
    
    eos_id = vocab.stoi.get("<EOS>", -1)
    
    for penalty in [1.2, 1.4, 1.6, 1.8, 2.0, 2.5]:
        print(f"\n==========================================")
        print(f" Penalty: {penalty}")
        print(f"==========================================")
        current_tokens = list(eval_tokens)
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(60):
                x = torch.tensor([current_tokens], dtype=torch.long, device=device)
                logits, _ = model(x)
                logits = logits[0, -1, :]
                
                # Apply repetition penalty only to output tokens within the sliding window
                if generated_tokens:
                    window_tokens = generated_tokens[-8:]
                    for token_id in set(window_tokens):
                        if logits[token_id] > 0:
                            logits[token_id] /= penalty
                        else:
                            logits[token_id] *= penalty
                            
                pred_id = torch.argmax(logits).item()
                    
                if pred_id == eos_id or pred_id == vocab.stoi.get("</OUTPUT>", -1):
                    break
                
                generated_tokens.append(pred_id)
                current_tokens.append(pred_id)
                
        generated_morphemes = [vocab.decode(tid) for tid in generated_tokens]
        morphemes_str = " ".join(generated_morphemes)
        print("Generated Morphemes:")
        print(morphemes_str)
        
        decompiled_text = decompiler.decompile_sentence(morphemes_str)
        print("Decompiled Sentence:")
        print(decompiled_text)

if __name__ == '__main__':
    main()
