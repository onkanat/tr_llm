import time
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_step_demo import KristalLM
from src.llm.tokenizer import Vocabulary

vocab = Vocabulary()
vocab.load('data/vocab.json')
vocab_size = len(vocab.stoi)

# Load SFT model to get realistic timing
model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
sft_model_path = 'data/kristal_model_sft.pt'
if os.path.exists(sft_model_path):
    sft_state_dict = torch.load(sft_model_path, map_location='cpu')
    keys_to_skip = [k for k in sft_state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del sft_state_dict[k]
    model.load_state_dict(sft_state_dict, strict=False)

model.eval()

for seq_len in [256, 512, 768, 1024, 2048]:
    x = torch.randint(0, vocab_size, (1, seq_len))
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
        
    # Measure
    t0 = time.time()
    n_runs = 3
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    t1 = time.time()
    avg_time = (t1 - t0) / n_runs
    print(f"Sequence Length {seq_len:4d}: average forward pass time = {avg_time:.4f} seconds")
