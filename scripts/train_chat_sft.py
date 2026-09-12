import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalDataset, KristalLM

class AlignedKristalDataset(KristalDataset):
    def get_batch(self, batch_size: int):
        num_records = (len(self.data) // 128) - 1
        ix = torch.randint(0, num_records, (batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i*128 : i*128 + 128]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i*128 + 1 : i*128 + 1 + 128]).astype(np.int64)) for i in ix])
        return x, y

def resize_state_dict(model, old_state_dict):
    """Resizes model embedding and linear heads to match the new vocabulary size."""
    new_state_dict = model.state_dict()
    for k, v in old_state_dict.items():
        if k in new_state_dict:
            if v.shape != new_state_dict[k].shape:
                print(f"Resizing weights for: {k} (Old: {list(v.shape)}, New: {list(new_state_dict[k].shape)})")
                if len(v.shape) == 2:
                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])] = v[:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])]
                elif len(v.shape) == 1:
                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0])] = v[:min(v.shape[0], new_state_dict[k].shape[0])]
            else:
                new_state_dict[k] = v
    return new_state_dict


def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: ORTAOKUL SOHBET SFT EĞİTİMİ")
    print("=" * 60)

    # 1. Device Setup
    device_arg = "mps" if torch.backends.mps.is_available() else "cpu"
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--device" and arg_idx + 1 < len(sys.argv):
            device_arg = sys.argv[arg_idx + 1]

    if device_arg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Cihaz: {device}")

    # 2. Vocabulary & Dataset Setup
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}")

    train_bin_path = 'data/train_chat_balanced.bin'
    if not os.path.exists(train_bin_path):
        train_bin_path = 'data/train_chat_sft.bin'

    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--data", "--dataset") and arg_idx + 1 < len(sys.argv):
            train_bin_path = sys.argv[arg_idx + 1]

    if not os.path.exists(train_bin_path):
        print(f"Hata: {train_bin_path} bulunamadı! Lütfen önce prepare_chat_balanced_dataset.py betiğini çalıştırın.")
        return

    # Use block_size = 128 to cover prompt + output sequences in a single window while keeping training fast
    block_size = 128
    dataset = AlignedKristalDataset(train_bin_path, block_size=block_size)
    print(f"Sohbet SFT veri kümesi yüklendi: {train_bin_path}. Toplam morfem sayısı: {len(dataset.data):,}")

    # 3. Model Setup & Load Base Weights
    print("\n[1] Temel model ağırlıkları yükleniyor...")
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    
    # Load from the best SFT/DPO model weights
    base_model_path = 'data/kristal_model.pt'
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--base-model", "--load-path") and arg_idx + 1 < len(sys.argv):
            base_model_path = sys.argv[arg_idx + 1]

    if not os.path.exists(base_model_path):
        print(f"Hata: Temel model dosyası '{base_model_path}' bulunamadı!")
        return
        
    state_dict = torch.load(base_model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    resized_state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(resized_state_dict, strict=False)
    model.to(device)

    print(f"  -> Ağırlıklar '{base_model_path}' adresinden başarıyla yüklendi.")

    # Using low learning rate to prevent forgetting prior scientific knowledge
    lr = 1.5e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 4. SFT Training Loop with Causal Masking
    batch_size = 16
    max_steps = 200
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--steps" and arg_idx + 1 < len(sys.argv):
            max_steps = int(sys.argv[arg_idx + 1])
        if arg == "--batch-size" and arg_idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[arg_idx + 1])

    eval_interval = 10
    print(f"\n[2] Sohbet Fine-tuning Başlatılıyor -> Adım Sayısı: {max_steps}, LR: {lr}, Batch: {batch_size}")



    
    model.train()
    start_time = time.time()
    loss_history = []
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)

    for step in range(1, max_steps + 1):
        while True:
            x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
            
            # Compute sign mask on CPU to avoid device roundtrips
            sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
            
            # Apply Causal Prompt Masking on CPU
            targets_np = y_cpu.numpy().copy()
            batch_size_curr, seq_len = x_cpu.shape
            x_list = x_cpu.tolist()
            
            pad_id = vocab.stoi.get("<PAD>", -1)
            for b in range(batch_size_curr):
                seq = x_list[b]
                if output_start_id in seq:
                    is_output = False
                    for i in range(seq_len):
                        token_id = seq[i]
                        if token_id == output_start_id:
                            is_output = True
                        if not is_output or token_id == pad_id:
                            targets_np[b, i] = -100
                        if token_id == eos_id:
                            is_output = False
                else:
                    targets_np[b, :] = -100
            
            targets = torch.from_numpy(targets_np).to(device)
            active_targets = (targets != -100).sum().item()
            if active_targets > 0:
                break
                
        x = x_cpu.to(device)
        sign_mask = sign_mask_cpu.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets, sign_mask)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"Adım {step:4d}/{max_steps} | SFT Kayıp (Loss): {loss_val:.6f} | Geçen Süre: {elapsed:.2f}sn")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(" SFT SOHBET HİZALAMA EĞİTİMİ TAMAMLANDI!")
    print("=" * 60)
    print(f"Toplam Süre:       {total_time:.2f} saniye")
    print(f"Başlangıç Kaybı:   {loss_history[0]:.6f}")
    print(f"Bitiş Kaybı:       {loss_history[-1]:.6f}")
    
    # Save back to data/kristal_model.pt
    torch.save(model.state_dict(), base_model_path)
    print(f"Güncellenmiş model ağırlıkları '{base_model_path}' dosyasına kaydedildi.")
    print("=" * 60)

if __name__ == '__main__':
    main()
