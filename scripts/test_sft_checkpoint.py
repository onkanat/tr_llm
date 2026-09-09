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

def generate_tokens(model, vocab, prompt_tokens, max_new_tokens=15, device='cpu'):
    model.eval()
    generated = list(prompt_tokens)
    eos_id = vocab.stoi.get("<EOS>", -1)
    output_end_id = vocab.stoi.get("</OUTPUT>", -1)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :]
            pred_id = torch.argmax(logits).item()
            generated.append(pred_id)
            if pred_id == eos_id or pred_id == output_end_id:
                break
    return generated

def resize_state_dict(model, old_state_dict):
    """Resizes model embedding and linear heads to match the new vocabulary size."""
    new_state_dict = model.state_dict()
    for k, v in old_state_dict.items():
        if k in new_state_dict:
            if v.shape != new_state_dict[k].shape:
                print(f"Resizing weights for: {k} (Old: {list(v.shape)}, New: {list(new_state_dict[k].shape)})")
                if len(v.shape) == 2:
                    new_state_dict[k][:v.shape[0], :v.shape[1]] = v
                elif len(v.shape) == 1:
                    new_state_dict[k][:v.shape[0]] = v
            else:
                new_state_dict[k] = v
    return new_state_dict

def test_checkpoint(model_path):
    print(f"\n--- Testing Checkpoint: {model_path} ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    vocab_size = len(vocab.stoi)
    
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096)
    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    resized_state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(resized_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    sft_cases = [
        {"instruction": "Kelimedeki kök morfemini bul.", "input": "denemenin", "output": ""},
        {"instruction": "Kelimedeki kök morfemini bul.", "input": "dayanılmaz", "output": ""}
    ]
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    
    for case in sft_cases:
        print(f"\nTask: '{case['instruction']}' | Input: '{case['input']}'")
        raw_prompt = json.dumps(case)
        token_ids = tokenizer.encode(raw_prompt)
        
        if output_start_id in token_ids:
            output_idx = token_ids.index(output_start_id)
            prompt_tokens = token_ids[:output_idx + 1]
        else:
            prompt_tokens = token_ids
            
        generated_ids = generate_tokens(model, vocab, prompt_tokens, device=device)
        response_tokens = generated_ids[len(prompt_tokens):]
        response_morphemes = [vocab.decode(tid) for tid in response_tokens]
        print(f"Generated: {' '.join(response_morphemes)}")

if __name__ == '__main__':
    test_checkpoint('data/kristal_model_sft.pt')
    test_checkpoint('data/kristal_model.pt')
