import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary, get_morpheme_weight

# ==========================================
# 1. KristalDataset (PyTorch Data Loader)
# ==========================================
class KristalDataset:
    def __init__(self, bin_filepath: str, block_size: int):
        self.data = np.memmap(bin_filepath, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def get_batch(self, batch_size: int):
        # Select random start offsets
        max_idx = len(self.data) - self.block_size
        if max_idx <= 0:
            # Fallback for very small datasets
            ix = torch.zeros((batch_size,), dtype=torch.long)
            block_sz = len(self.data) - 1
        else:
            ix = torch.randint(max_idx, (batch_size,))
            block_sz = self.block_size
            
        x = torch.stack([torch.from_numpy((self.data[i:i+block_sz]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+block_sz]).astype(np.int64)) for i in ix])
        
        return x, y # Shape: [batch_size, block_size]

# ==========================================
# 2. KristalEmbedding (Sign Inversion Layer)
# ==========================================
class KristalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, vocab: Vocabulary):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.vocab = vocab
        
        # Load negation token IDs for Sign Inversion
        self.neg_ids = set()
        for tok in ["NEG", "IMPOTENTIAL_NEG"]:
            if tok in vocab.stoi:
                self.neg_ids.add(vocab.stoi[tok])
                
        # Cache of root/control token IDs to identify word boundaries
        self.boundary_ids = set()
        for tok, token_id in vocab.stoi.items():
            # Control tokens
            if tok.startswith("<") and tok.endswith(">"):
                self.boundary_ids.add(token_id)
                continue
            # Inflection or derivation suffixes
            if tok.startswith(("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_", "DERIV_")):
                continue
            if tok in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"):
                continue
            # It's a lexical root
            self.boundary_ids.add(token_id)

    def compute_sign_mask(self, x_cpu: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x_cpu.shape
        mask_np = np.ones((batch_size, seq_len, 1), dtype=np.float32)
        
        x_list = x_cpu.tolist()
        
        for b in range(batch_size):
            seq = x_list[b]
            word_start_idx = 0
            has_negation = False
            
            for i in range(seq_len):
                token_id = seq[i]
                is_boundary = token_id in self.boundary_ids
                
                if is_boundary and i > 0:
                    if has_negation:
                        mask_np[b, word_start_idx:i] = -1.0
                    word_start_idx = i
                    has_negation = token_id in self.neg_ids
                else:
                    if token_id in self.neg_ids:
                        has_negation = True
            
            if has_negation:
                mask_np[b, word_start_idx:seq_len] = -1.0
                
        return torch.from_numpy(mask_np)

    def forward(self, x: torch.Tensor, sign_mask: torch.Tensor = None) -> torch.Tensor:
        embeddings = self.embedding(x)
        if sign_mask is None:
            # Fallback for inference or when sign_mask is not pre-computed:
            # Moves to CPU to compute and then moves back to target device.
            sign_mask = self.compute_sign_mask(x.cpu()).to(device=x.device, dtype=embeddings.dtype)
        return embeddings * sign_mask

# ==========================================
# 3. Model Simulation & Training Step (Causal Transformer Decoder with RoPE)
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Eski checkpoint'lerdeki kalıcı inv_freq anahtarını sessizce at (anında yeniden üretiliyor)
        state_dict.pop(prefix + "inv_freq", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor, seq_len: int):
        device = x.device
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=block_size)
        
        # Causal mask: True means masked out in PyTorch (persistent=False ile checkpoint'e yazılmaz)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("mask", mask, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Eski checkpoint'lerdeki kalıcı mask ve rotary inv_freq anahtarlarını sessizce at (T-0007)
        state_dict.pop(prefix + "mask", None)
        state_dict.pop(prefix + "rotary_emb.inv_freq", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # [B, n_head, T, head_dim]
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Apply causal mask
        attn_mask = self.mask[:T, :T]
        att = att.masked_fill(attn_mask, float('-inf'))
        
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        
        y = self.proj_dropout(self.out_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class KristalLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, vocab: Vocabulary, n_layer=6, n_head=6, dropout=0.1, block_size=1024):
        super().__init__()
        self.embedding = KristalEmbedding(vocab_size, n_embd, vocab)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, sign_mask: torch.Tensor = None, return_hidden_states: bool = False):
        batch_size, seq_len = x.shape
        
        # 1. Custom token embedding (with sign inversion for negated words)
        tok_emb = self.embedding(x, sign_mask) # [batch_size, seq_len, n_embd]
        
        # 2. Transformer blocks (No absolute position embedding; RoPE handles it inside attention)
        x_emb = tok_emb
        for block in self.blocks:
            x_emb = block(x_emb)
            
        x_emb = self.ln_f(x_emb)
        logits = self.lm_head(x_emb) # [batch_size, seq_len, vocab_size]
        
        loss = None
        if targets is not None:
            # Flatten tensors for cross entropy computation
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        if return_hidden_states:
            return logits, loss, x_emb
            
        return logits, loss

def run_demo():
    print("="*60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: EĞİTİM ADIMI SİMÜLASYONU (SPRINT 3)")
    print("="*60)

    # 1. Load Vocab
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}")

    # 2. Dataset and Loader Initialization
    block_size = 32  # Small block size for demo
    dataset = KristalDataset("data/train.bin", block_size=block_size)
    print(f"Veri yükleyici başlatıldı. Toplam token sayısı: {len(dataset.data)}")

    # 3. Model & Optimizer Setup
    n_embd = 768
    model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    print("Model ve Optimizer kuruldu.")

    # 4. Single Training Step (Mini-Batch)
    batch_size = 4
    x, y = dataset.get_batch(batch_size=batch_size)
    print(f"\nBatch Oluşturuldu (batch_size={batch_size}, seq_len={x.shape[1]}):")
    print(f"Girdi (x) Shape: {x.shape}")
    print(f"Hedef (y) Shape: {y.shape}")

    # Perform step
    model.train()
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    print(f"\n--- Eğitim Adımı Sonuçları ---")
    print(f"İleri Besleme Logit Shape: {logits.shape}")
    print(f"Hesaplanan Loss Değeri:    {loss.item():.6f}")
    print("Geri besleme (backward pass) ve ağırlık güncellemesi başarıyla tamamlandı.")

    # 5. SFT Tokenization Demo
    print(f"\n" + "="*50)
    print(" SFT VERİ SETİ ETİKET SARMALAMA VE TOKENİZASYON DEMOSU")
    print("="*50)
    
    # Load compiler & tokenizer
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    
    sft_example = {
        "instruction": "Cümledeki eylemin zamanını tespit et.",
        "input": "Yarın okula gideceğim.",
        "output": "TENSE_FUT"
    }
    
    print(f"Ham JSON Nesnesi:\n{json.dumps(sft_example, ensure_ascii=False, indent=2)}")
    
    # Tokenize the JSON object (KristalTokenizer automatically normalizes and wraps with control tokens)
    token_ids = tokenizer.encode(json.dumps(sft_example))
    crystal_tags = tokenizer.decode(token_ids)
    
    print(f"\nTokenleştirilmiş ve Sarmalanmış Morfem Akışı:")
    print(crystal_tags)
    print(f"\nToken ID'ler (stoi):")
    print(token_ids)

if __name__ == '__main__':
    run_demo()
