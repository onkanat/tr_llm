import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalDataset, KristalLM

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

def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DENGELİ BİLİMSEL SFT EĞİTİMİ")
    print("=" * 60)

    # 1. Device Setup (Avoid MPS on macOS due to PyTorch AdamW deadlock & SDPA NaN bugs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Cihaz: {device}")

    # CUDA specific optimization and memory checks
    if device.type == 'cuda':
        print(f"CUDA Aygıtı: {torch.cuda.get_device_name(0)}")
        print(f"Mevcut CUDA Belleği: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
        # Enable benchmark mode for faster autotuning
        torch.backends.cudnn.benchmark = True
    else:
        print("macOS üzerinde CPU modu seçildi (MPS backend kilitlenme/deadlock ve NaN hatalarını önlemek için).")

    # 2. Vocabulary & Dataset Setup
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}")

    train_bin_path = 'data/train_balanced_sft.bin'
    if not os.path.exists(train_bin_path):
        print(f"Hata: {train_bin_path} bulunamadı! Lütfen önce prepare_balanced_sft_dataset.py betiğini çalıştırın.")
        return

    block_size_dataset = 64
    dataset = KristalDataset(train_bin_path, block_size=block_size_dataset)
    print(f"Dengeli veri kümesi yüklendi. Toplam SFT morfem sayısı: {len(dataset.data)}")

    # 3. Model Setup (Scaled up to 6 layers and 6 heads for higher capacity)
    n_layer = 6
    n_head = 6
    n_embd = 768
    block_size_model = 1024
    
    print(f"\n[1] Ölçeklendirilmiş Model Kuruluyor (n_layer={n_layer}, n_head={n_head}, n_embd={n_embd})...")
    model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab, block_size=block_size_model, n_layer=n_layer, n_head=n_head)
    
    base_model_path = 'data/kristal_model.pt'
    loaded_weights = False
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Temel model checkpoint'i bulunamadı: {base_model_path} (sessiz sıfırdan başlama engellendi)")
    try:
        old_state_dict = torch.load(base_model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Checkpoint dosyası mevcut fakat yüklenemedi ({base_model_path}): {e}") from e

    from scripts.train_step_b1_5_rigorous import compute_sha256
    sha256_val = compute_sha256(base_model_path)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(base_model_path)} sha256={sha256_val} anahtar={len(old_state_dict)}")

    keys_to_skip = [k for k in old_state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del old_state_dict[k]
    
    resized_state_dict = resize_state_dict(model, old_state_dict)
    load_res = model.load_state_dict(resized_state_dict, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(f"[SOYAGACI_UYARI] strict=False ile yüklendi: eksik={len(load_res.missing_keys)}, fazla={len(load_res.unexpected_keys)}")
        if load_res.missing_keys:
            print(f"  * Eksik anahtarlar: {load_res.missing_keys[:5]}{'...' if len(load_res.missing_keys) > 5 else ''}")
        if load_res.unexpected_keys:
            print(f"  * Fazla anahtarlar: {load_res.unexpected_keys[:5]}{'...' if len(load_res.unexpected_keys) > 5 else ''}")
    else:
        print("[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).")
    print(f"Model ağırlıkları '{base_model_path}' üzerinden yüklendi ve uyarlandı.")
    loaded_weights = True

    model.to(device)

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 5. Training Loop parameters
    batch_size = 64
    max_steps = 3000
    eval_interval = 50
    
    print(f"\n[2] SFT İnce Ayar Başlatılıyor -> Adım Sayısı: {max_steps}, LR: 1e-4, Batch: {batch_size}")
    
    start_time = time.time()
    loss_history = []
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)

    step = 1
    while step <= max_steps:
        try:
            model.train()
            x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
            
            # Compute sign mask on CPU to avoid device roundtrips
            sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
            
            # Apply Causal Prompt Masking on CPU
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
                else:
                    targets_np[b, :] = -100
                    
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
            
            step += 1

        except RuntimeError as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "alloc" in err_msg:
                print("\n" + "!" * 70)
                print(" UYARI: GPU BELLEK YETERSİZLİĞİ (OOM) TESPİT EDİLDİ!")
                print("!" * 70)
                print(f"Hata detayı: {e}")
                print("\nAlınan Tedbirler:")
                
                # CUDA specifics
                if device.type == 'cuda':
                    print("  1. CUDA önbelleği temizleniyor (torch.cuda.empty_cache())...")
                    torch.cuda.empty_cache()
                
                if batch_size > 16:
                    new_batch_size = batch_size // 2
                    print(f"  2. Batch size düşürülüyor: {batch_size} -> {new_batch_size}")
                    batch_size = new_batch_size
                    print("  3. Eğitim yeni batch size ile devam ettiriliyor...\n")
                    continue
                else:
                    print("  2. Batch size zaten minimum seviyede (<= 16).")
                    print("  3. İşlemci (CPU) moduna geçiş yapılıyor (CPU Fallback)...")
                    device = torch.device("cpu")
                    model.to(device)
                    print("  4. Eğitim CPU üzerinde devam ettiriliyor...\n")
                    continue
            else:
                # Raise other runtime errors
                raise e

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(" SFT İNCE AYAR BAŞARIYLA TAMAMLANDI!")
    print("=" * 50)
    print(f"Toplam Süre:       {total_time:.2f} saniye")
    print(f"Başlangıç Kaybı:   {loss_history[0]:.6f}")
    print(f"Bitiş Kaybı:       {loss_history[-1]:.6f}")
    
    # 6. Save Model Weights
    sft_model_path = 'data/kristal_model_sft.pt'
    torch.save(model.state_dict(), sft_model_path)
    print(f"SFT model ağırlıkları '{sft_model_path}' dosyasına başarıyla kaydedildi.")
    
    # Save as main checkpoint also so chat_prompt.py loads it
    main_model_path = 'data/kristal_model.pt'
    torch.save(model.state_dict(), main_model_path)
    print(f"Ağırlıklar ana model dosyasına kopyalandı: '{main_model_path}'")
    print("=" * 50)

if __name__ == '__main__':
    main()
