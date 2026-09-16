import os
import sys
import time
import json
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalLM

def get_logps(logits, targets, prompt_len):
    """Computes log probabilities for the output tokens only."""
    # logits shape: [1, seq_len, vocab_size] -> slice to [seq_len - 1, vocab_size]
    # targets shape: [1, seq_len] -> slice to [seq_len - 1]
    logits = logits[0, :-1, :]
    targets = targets[0, 1:]
    
    logps = torch.log_softmax(logits, dim=-1)
    target_logps = torch.gather(logps, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Slice starting from the first output token
    output_logps = target_logps[prompt_len - 1 :]
    return output_logps.sum()

def check_frozen_save_path(save_path: str, allow_frozen_write: bool = False) -> None:
    """Belirtilen kaydetme yolunun donmuş olup olmadığını denetler.
    
    Donmuş yola yazma izni (allow_frozen_write=True) açıkça verilmemişse RuntimeError fırlatır.
    """
    from src.llm.frozen_guard import is_frozen_path
    if is_frozen_path(save_path) and not allow_frozen_write:
        raise RuntimeError(f"Donmuş yola yazma engellendi: {save_path} (allow_frozen_write=False)")

def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DPO HİZALAMA EĞİTİMİ (STAGE-3)")
    print("=" * 60)

    # 1. Device Setup (Avoid MPS on macOS due to PyTorch AdamW deadlock & SDPA NaN bugs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Cihaz: {device}")

    # 2. Vocabulary & Data Loading
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}")

    dpo_jsonl_path = 'data/pedagogy/dpo_all_tokenized.jsonl'
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--data", "--dataset") and arg_idx + 1 < len(sys.argv):
            dpo_jsonl_path = sys.argv[arg_idx + 1]

    if not os.path.exists(dpo_jsonl_path):
        print(f"Hata: DPO derlenmiş veri seti '{dpo_jsonl_path}' bulunamadı! Lütfen derleme adımını çalıştırın.")
        return

    # Load all tokenized records
    print(f"DPO veri seti yükleniyor: {dpo_jsonl_path}...")
    records = []
    skipped_count = 0
    with open(dpo_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                c_len = len(rec["prompt_ids"]) + len(rec["chosen_ids"])
                r_len = len(rec["prompt_ids"]) + len(rec["rejected_ids"])
                if c_len > 4096 or r_len > 4096:
                    skipped_count += 1
                    continue
                records.append(rec)
    print(f"  -> Yüklenen Tercih Çifti Sayısı: {len(records)} (Uzunluk sınırı nedeniyle {skipped_count} adet çift atlandı)")

    # 3. Initialize Active and Reference Models
    print("\n[1] Aktif ve Referans modeller yükleniyor...")
    n_embd = 768
    
    model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    ref_model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    
    sft_model_path = 'data/kristal_model_sft.pt'
    active_model_path = 'data/kristal_model.pt'
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--ref-model", "--sft-model") and arg_idx + 1 < len(sys.argv):
            sft_model_path = sys.argv[arg_idx + 1]
        if arg in ("--active-model", "--base-model") and arg_idx + 1 < len(sys.argv):
            active_model_path = sys.argv[arg_idx + 1]

    if not os.path.exists(sft_model_path):
        print(f"Hata: SFT referans model dosyası '{sft_model_path}' bulunamadı!")
        return

    # Load weights into ref_model (filter out deterministic RoPE/Causal buffers)
    sft_state_dict = torch.load(sft_model_path, map_location=device)
    keys_to_skip = [k for k in sft_state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del sft_state_dict[k]
    ref_model.load_state_dict(sft_state_dict, strict=False)

    # Load weights into active model (from active_model_path if exists, else sft_model_path)
    if os.path.exists(active_model_path):
        print(f"  * Aktif model ağırlıkları: {active_model_path}")
        active_state_dict = torch.load(active_model_path, map_location=device)
        for k in [k for k in active_state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
            del active_state_dict[k]
        model.load_state_dict(active_state_dict, strict=False)
    else:
        print(f"  * Aktif model referans modelden başlatılıyor: {sft_model_path}")
        model.load_state_dict(sft_state_dict, strict=False)
    
    model.to(device)
    ref_model.to(device)
    
    # Freeze reference model parameters
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    print("Modeller hazırlandı. Referans model donduruldu.")

    # 4. Optimizer setup
    lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 5. DPO Training Loop
    max_steps = 30
    batch_size = 2  # Gradient accumulation batch size
    beta = 0.1      # DPO temperature parameter
    eval_interval = 5

    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--steps" and arg_idx + 1 < len(sys.argv):
            max_steps = int(sys.argv[arg_idx + 1])
        if arg == "--batch-size" and arg_idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[arg_idx + 1])
        if arg == "--beta" and arg_idx + 1 < len(sys.argv):
            beta = float(sys.argv[arg_idx + 1])

    print(f"\nDPO Hizalama Başlatılıyor -> Adım: {max_steps}, Beta: {beta}, LR: {lr}, Batch: {batch_size}", flush=True)
    
    model.train()
    start_time = time.time()
    loss_history = []

    # Simple random index sampler for training pairs
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    record_idx = 0

    for step in range(1, max_steps + 1):
        step_loss = 0.0
        optimizer.zero_grad()
        
        # Process a batch of size `batch_size` (accumulating gradients)
        accumulated = 0
        while accumulated < batch_size:
            if record_idx >= len(indices):
                indices = np.random.permutation(len(records))
                record_idx = 0
                
            rec = records[indices[record_idx]]
            record_idx += 1
            
            prompt_ids = rec["prompt_ids"]
            prompt_len = len(prompt_ids)
            if prompt_len >= 200:
                continue
                
            chosen_ids = rec["chosen_ids"]
            rejected_ids = rec["rejected_ids"]
            
            # Combine to full SFT/DPO sequences, truncated to max 256 tokens for fast CPU execution
            seq_chosen = (prompt_ids + chosen_ids)[:256]
            seq_rejected = (prompt_ids + rejected_ids)[:256]
            
            # Move to tensors
            x_chosen = torch.tensor([seq_chosen], dtype=torch.long, device=device)
            x_rejected = torch.tensor([seq_rejected], dtype=torch.long, device=device)
            
            # Active Model LogPs
            logits_chosen, _ = model(x_chosen)
            logits_rejected, _ = model(x_rejected)
            
            pi_chosen_logp = get_logps(logits_chosen, x_chosen, prompt_len)
            pi_rejected_logp = get_logps(logits_rejected, x_rejected, prompt_len)
            
            # Reference Model LogPs (Frozen)
            with torch.no_grad():
                ref_logits_chosen, _ = ref_model(x_chosen)
                ref_logits_rejected, _ = ref_model(x_rejected)
                ref_chosen_logp = get_logps(ref_logits_chosen, x_chosen, prompt_len)
                ref_rejected_logp = get_logps(ref_logits_rejected, x_rejected, prompt_len)
                
            # DPO Logits
            pi_logratio = pi_chosen_logp - pi_rejected_logp
            ref_logratio = ref_chosen_logp - ref_rejected_logp
            dpo_logits = pi_logratio - ref_logratio
            
            # Sigmoid Loss
            loss_pair = -torch.nn.functional.logsigmoid(beta * dpo_logits)
            loss_pair = loss_pair / batch_size # Normalize by batch size
            
            loss_pair.backward()
            step_loss += loss_pair.item()
            accumulated += 1

        optimizer.step()
        loss_history.append(step_loss)
        
        if step % eval_interval == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - start_time
            print(f"Adım {step:4d}/{max_steps} | DPO Kayıp (Loss): {step_loss:.6f} | Geçen Süre: {elapsed:.2f}sn", flush=True)

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(" DPO HİZALAMA EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 50)
    allow_frozen_write = "--allow-frozen-write" in sys.argv
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--save-path", "--output-model") and arg_idx + 1 < len(sys.argv):
            active_model_path = sys.argv[arg_idx + 1]

    # 6. Save final DPO-aligned weights to kristal_model.pt
    # This completes the training loop and updates the main model
    dpo_model_save_path = active_model_path
    check_frozen_save_path(dpo_model_save_path, allow_frozen_write=allow_frozen_write)
    torch.save(model.state_dict(), dpo_model_save_path)
    print(f"DPO hizalanmış nihai model ağırlıkları '{dpo_model_save_path}' dosyasına kaydedildi.")
    print("=" * 50)

if __name__ == '__main__':
    main()
