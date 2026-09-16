import os
import sys
import time
import json
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalDataset, KristalEmbedding, KristalLM
from src.llm.prompt_contract import resize_state_dict

def check_frozen_save_path(save_path: str, allow_frozen_write: bool = False) -> None:
    """Belirtilen kaydetme yolunun donmuş olup olmadığını denetler.
    
    Donmuş yola yazma izni (allow_frozen_write=True) açıkça verilmemişse RuntimeError fırlatır.
    """
    from src.llm.frozen_guard import is_frozen_path
    if is_frozen_path(save_path) and not allow_frozen_write:
        raise RuntimeError(f"Donmuş yola yazma engellendi: {save_path} (allow_frozen_write=False)")

def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: MODEL EĞİTİM DÖNGÜSÜ")
    print("=" * 60)

    # 1. Device Selection
    device_arg = "cpu"
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--device" and arg_idx + 1 < len(sys.argv):
            device_arg = sys.argv[arg_idx + 1]
            
    if device_arg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Metal GPU (MPS) seçildi.", flush=True)
    elif device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU seçildi.", flush=True)
    else:
        device = torch.device("cpu")
        num_cores = os.cpu_count() or 8
        torch.set_num_threads(num_cores)
        print(f"CPU modu seçildi ({num_cores} iş parçacığı aktif).", flush=True)

    # 2. Vocabulary & Data Loading
    vocab = Vocabulary()
    vocab_path = 'data/rebuild/vocab_base_32852.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}", flush=True)

    pad_ignore_index = None if "--no-pad-mask" in sys.argv else vocab.stoi.get("<PAD>", 1)
    if pad_ignore_index is not None:
        print(f"<PAD> kayıp maskesi aktif (ignore_index={pad_ignore_index}).", flush=True)
    else:
        print("<PAD> kayıp maskesi devre dışı (--no-pad-mask).", flush=True)

    bin_filepath = 'data/train.bin'
    block_size = 64
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--data", "--dataset") and arg_idx + 1 < len(sys.argv):
            bin_filepath = sys.argv[arg_idx + 1]
        if arg == "--block-size" and arg_idx + 1 < len(sys.argv):
            block_size = int(sys.argv[arg_idx + 1])

    # Auto-detect block_size from metadata if available and not explicitly provided
    meta_path = bin_filepath + '.meta.json'
    if os.path.exists(meta_path) and "--block-size" not in sys.argv:
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
                if "block_size" in meta:
                    block_size = meta["block_size"]
        except Exception:
            pass

    if not os.path.exists(bin_filepath):
        print(f"Hata: {bin_filepath} bulunamadı! Lütfen önce derleme adımını çalıştırın.", flush=True)
        return

    dataset = KristalDataset(bin_filepath, block_size=block_size)
    print(f"Veri kümesi yüklendi: {bin_filepath} (Blok boyutu: {block_size}). Toplam morfem token sayısı: {len(dataset.data):,}", flush=True)

    # 3. Model & Optimizer Setup
    n_embd = 768
    model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab)
    
    model_save_path = 'data/kristal_model.pt'
    model_load_path = None
    from_scratch = "--from-scratch" in sys.argv

    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--save-path", "--output-model") and arg_idx + 1 < len(sys.argv):
            model_save_path = sys.argv[arg_idx + 1]
        if arg in ("--load-path", "--base-model") and arg_idx + 1 < len(sys.argv):
            model_load_path = sys.argv[arg_idx + 1]

    if model_load_path is None:
        model_load_path = model_save_path

    allow_frozen_write = "--allow-frozen-write" in sys.argv
    check_frozen_save_path(model_save_path, allow_frozen_write=allow_frozen_write)

    lr = 1e-3
    if os.path.exists(model_load_path) and not from_scratch:
        print(f"Mevcut model ağırlıkları '{model_load_path}' tespit edildi, eğitim devam ettiriliyor (Resume)...", flush=True)
        state_dict = torch.load(model_load_path, map_location=device)
        keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
        for k in keys_to_skip:
            del state_dict[k]
        state_dict = resize_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        lr = 2e-4  # Lower learning rate when fine-tuning/resuming
    else:
        print("Sıfırdan eğitim (From Scratch) başlatılıyor...", flush=True)

    model.to(device)
    
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--lr" and arg_idx + 1 < len(sys.argv):
            lr = float(sys.argv[arg_idx + 1])

    # Using AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Model mimarisi kuruldu ve cihaza taşındı. (Öğrenme Oranı: {lr})", flush=True)

    # 4. Training Loop Configuration
    batch_size = 32
    max_steps = 100
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--steps" and arg_idx + 1 < len(sys.argv):
            max_steps = int(sys.argv[arg_idx + 1])
        if arg == "--batch-size" and arg_idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[arg_idx + 1])
            
    eval_interval = 10
    
    print(f"\nEğitim Başlatılıyor -> Adım Sayısı: {max_steps}, Batch Boyutu: {batch_size}, Block Boyutu: {block_size}", flush=True)
    
    model.train()
    start_time = time.time()
    
    loss_history = []
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)

    for step in range(1, max_steps + 1):
        step_t0 = time.time()
        # Fetch a batch and move tensors to device
        x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
        
        # Compute sign mask on CPU to avoid device roundtrips
        sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
        
        # Apply Causal Prompt Masking for SFT sequences on CPU
        targets_np = y_cpu.numpy().copy()
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
                        
        x = x_cpu.to(device)
        targets = torch.from_numpy(targets_np).to(device)
        sign_mask = sign_mask_cpu.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets, sign_mask, ignore_index=pad_ignore_index)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        step_dur = time.time() - step_t0
        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"Adım {step:4d}/{max_steps} | Kayıp (Loss): {loss_val:.4f} | Adım Süresi: {step_dur:.2f}s | Toplam Süre: {elapsed:.1f}s", flush=True)

    total_time = time.time() - start_time
    print("\n" + "=" * 50, flush=True)
    print(" EĞİTİM BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 50, flush=True)
    print(f"Toplam Süre:       {total_time:.2f} saniye", flush=True)
    print(f"Başlangıç Kaybı:   {loss_history[0]:.4f}", flush=True)
    print(f"Bitiş Kaybı:       {loss_history[-1]:.4f}", flush=True)
    
    # 5. Save Model Weights
    torch.save(model.state_dict(), model_save_path)
    print(f"Eğitilmiş model ağırlıkları '{model_save_path}' dosyasına kaydedildi.", flush=True)
    print("=" * 50, flush=True)

if __name__ == '__main__':
    main()
