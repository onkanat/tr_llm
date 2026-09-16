import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalDataset, KristalLM, mask_prompt_targets

def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: STAGE-2 SFT İNCE AYAR (FINE-TUNING)")
    print("=" * 60)

    # 1. Device Selection (Metal GPU support for macOS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Çıkarım ve eğitim yapılacak cihaz: {device}")

    # 2. Vocabulary & Data Loading
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}")

    # Combine infancy and parenting binaries for SFT-only dataset
    temp_sft_bin = 'data/train_sft_combined.bin'
    infancy_bin = 'data/train_infancy.bin'
    parenting_bin = 'data/train_parenting.bin'

    # Compile infancy and parenting datasets dynamically if they are missing
    from src.llm.prepare import prepare_dataset
    
    if not os.path.exists(infancy_bin):
        print("Bebeklik (Infancy) veri seti derleniyor...")
        prepare_dataset('data/pedagogy/infancy_dataset.jsonl', infancy_bin)
        
    if not os.path.exists(parenting_bin):
        print("Ebeveynlik (Parenting SFT) veri seti derleniyor...")
        prepare_dataset('data/pedagogy/parenting_dataset.jsonl', parenting_bin)

    print("\n[1] SFT verileri birleştiriliyor...")
    tokens_infancy = np.fromfile(infancy_bin, dtype=np.uint16)
    tokens_parenting = np.fromfile(parenting_bin, dtype=np.uint16)
    tokens_sft = np.concatenate([tokens_infancy, tokens_parenting])
    tokens_sft.tofile(temp_sft_bin)
    print(f"  -> Birleştirilmiş SFT token sayısı: {len(tokens_sft)}")

    block_size = 64
    dataset = KristalDataset(temp_sft_bin, block_size=block_size)
    print(f"SFT Veri kümesi yüklendi.")

    # 3. Model Setup & Load Base Weights
    print("\n[2] Temel model ağırlıkları yükleniyor...")
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=1024, n_layer=6, n_head=6)
    base_model_path = 'data/kristal_model.pt'
    if not os.path.exists(base_model_path):
        print(f"Hata: Temel model dosyası '{base_model_path}' bulunamadı!")
        return
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model.to(device)
    print(f"  -> Ağırlıklar '{base_model_path}' adresinden başarıyla yüklendi.")

    # Using lower learning rate for fine-tuning to preserve base knowledge
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 4. SFT Training Loop with Causal Masking
    batch_size = 64
    max_steps = 1500  # Focused training steps
    eval_interval = 150

    print(f"\n[3] SFT İnce Ayar Başlatılıyor -> Adım Sayısı: {max_steps}, LR: 1e-4, Batch: {batch_size}")
    
    model.train()
    start_time = time.time()
    loss_history = []
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)

    for step in range(1, max_steps + 1):
        x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
        
        # Compute sign mask on CPU to avoid device roundtrips
        sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
        
        # Apply Causal Prompt Masking on CPU
        targets_np = mask_prompt_targets(x_cpu, y_cpu, output_start_id, eos_id)
                
        x = x_cpu.to(device)
        targets = torch.from_numpy(targets_np).to(device)
        sign_mask = sign_mask_cpu.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets, sign_mask)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"Adım {step:4d}/{max_steps} | Ortalama Kayıp: {loss_val:.6f} | Geçen Süre: {elapsed:.2f}sn")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(" SFT İNCE AYAR BAŞARIYLA TAMAMLANDI!")
    print("=" * 50)
    print(f"Toplam Süre:       {total_time:.2f} saniye")
    print(f"Başlangıç Kaybı:   {loss_history[0]:.6f}")
    print(f"Bitiş Kaybı:       {loss_history[-1]:.6f}")
    
    # 5. Save SFT Model Weights
    sft_model_path = 'data/kristal_model_sft.pt'
    torch.save(model.state_dict(), sft_model_path)
    print(f"SFT ince ayarlı model ağırlıkları '{sft_model_path}' dosyasına kaydedildi.")
    print("=" * 50)

    # Clean up temporary combined binary
    if os.path.exists(temp_sft_bin):
        os.remove(temp_sft_bin)

if __name__ == '__main__':
    main()
