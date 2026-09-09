import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalDataset, KristalLM

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    vocab_size = len(vocab.stoi)
    
    dataset = KristalDataset('data/train_balanced_sft.bin', block_size=64)
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    
    print("Starting debug training...")
    for step in range(1, 10):
        print(f"\n--- Step {step} Start ---")
        t0 = time.time()
        
        print("  Getting batch...")
        x_cpu, y_cpu = dataset.get_batch(batch_size=64)
        
        print("  Applying masking...")
        sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
        targets_np = y_cpu.numpy().copy() # use copy to be safe
        batch_size_curr, seq_len = x_cpu.shape
        x_list = x_cpu.tolist()
        
        for b in range(batch_size_curr):
            seq = x_list[b]
            if output_start_id in seq:
                is_output = False
                for i in range(seq_len):
                    token_id = seq[i]
                    if token_id == output_start_id:
                        is_output = True
                    if not is_output:
                        targets_np[b, i] = -100
                    if token_id == eos_id:
                        is_output = False
            else:
                targets_np[b, :] = -100
                
        print("  Moving to device...")
        x = x_cpu.to(device)
        targets = torch.from_numpy(targets_np).to(device)
        sign_mask = sign_mask_cpu.to(device)
        
        print("  Forward pass...")
        logits, loss = model(x, targets, sign_mask)
        
        print("  Backward pass...")
        optimizer.zero_grad()
        loss.backward()
        
        print("  Optimizer step...")
        optimizer.step()
        
        # Clear MPS cache to prevent deadlocks
        if device.type == 'mps':
            torch.mps.empty_cache()
        
        print(f"  Step {step} done. Loss: {loss.item():.4f}, Time: {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
