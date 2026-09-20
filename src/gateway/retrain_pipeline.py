#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: OTONOM YENİDEN EĞİTİM BORU HATTI (RETRAIN PIPELINE)
=============================================================================
Bu modül; `future_train_vector.jsonl` dosyasında biriken zorlayıcı/epistemik
açık içeren örnekleri derler, tokenize eder ve modeli otomatik olarak
güncelleyen (Karpathy Continuous Learning Loop) yeniden eğitim sürecini yönetir.
"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import render_example
import torch
import torch.nn as nn


def expand_model_vocabulary(
    model: torch.nn.Module,
    vocab: Vocabulary,
    new_tokens: List[str],
    device: str = "cpu"
) -> Tuple[torch.nn.Module, List[int]]:
    """
    Expands both the Vocabulary and the in-memory KristalLM model weights (weight surgery).
    Preserves all existing token embeddings bit-for-bit (zero-forgetting).
    Initializes new tokens with mean representation + small random variation.
    """
    new_ids = vocab.register_new_tokens(new_tokens)
    new_vocab_size = len(vocab.stoi)

    old_emb_layer = None
    if hasattr(model, "embedding") and hasattr(model.embedding, "embedding"):
        old_emb_layer = model.embedding.embedding
    elif hasattr(model, "embedding") and isinstance(model.embedding, nn.Embedding):
        old_emb_layer = model.embedding

    if old_emb_layer is None:
        return model, new_ids

    old_vocab_size, n_embd = old_emb_layer.weight.shape
    if new_vocab_size <= old_vocab_size:
        return model, new_ids

    # 1. Expand Embedding Layer
    new_embedding = nn.Embedding(new_vocab_size, n_embd).to(device)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size].copy_(old_emb_layer.weight)
        mean_emb = old_emb_layer.weight.mean(dim=0, keepdim=True)
        noise = 0.02 * torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device)
        new_embedding.weight[old_vocab_size:].copy_(mean_emb + noise)

    if hasattr(model, "embedding") and hasattr(model.embedding, "embedding"):
        model.embedding.embedding = new_embedding
    else:
        model.embedding = new_embedding

    # 2. Expand lm_head Linear Layer
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        old_head = model.lm_head
        has_bias = old_head.bias is not None
        new_head = nn.Linear(old_head.in_features, new_vocab_size, bias=has_bias).to(device)
        with torch.no_grad():
            new_head.weight[:old_vocab_size].copy_(old_head.weight)
            mean_head = old_head.weight.mean(dim=0, keepdim=True)
            noise_head = 0.02 * torch.randn(new_vocab_size - old_vocab_size, old_head.in_features, device=device)
            new_head.weight[old_vocab_size:].copy_(mean_head + noise_head)
            if has_bias and old_head.bias is not None:
                new_head.bias[:old_vocab_size].copy_(old_head.bias)
                new_head.bias[old_vocab_size:].zero_()
        model.lm_head = new_head

    if hasattr(model, "vocab_size"):
        model.vocab_size = new_vocab_size

    return model, new_ids


class RetrainPipeline:
    def __init__(
        self,
        future_train_path: str = "data/future_train_vector.jsonl",
        archive_path: str = "data/future_train_archive.jsonl",
        vocab_path: Optional[str] = None,
        lexicon_path: str = "data/lexicon/roots.tsv",
        model_path: Optional[str] = None,
        output_bin_path: str = "data/train_future_finetune.bin",
        save_path: Optional[str] = None,
        allow_frozen_write: bool = False,
        device: str = "cpu"
    ):
        """T-0087 (FAIL-CLOSED): `vocab_path`, `model_path` ve `save_path` VARSAYILANI
        KALDIRILDI. Gerekce (olculdu): eski `model_path` varsayilani SILINMIS bir
        checkpoint adiydi ve deger yalnizca SOYAGACI kaydina yaziliyordu; eski
        `vocab_path` varsayilani BAYAT bir sozluktu (data/vocab.json, 31.357) ve
        derlenen .bin'i o sozlukle uretiyordu. `save_path` ise HIC YOKTU: `run_training`
        `train.py`'yi `--save-path` OLMADAN cagiriyordu ve T-0085'in fail-closed kapisi
        bu yolu zaten oldurmustu (bkz. rapor K5). Ucu de ACIKCA verilmelidir."""
        self.future_train_path = future_train_path
        self.archive_path = archive_path
        self.vocab_path = vocab_path
        self.lexicon_path = lexicon_path
        self.model_path = model_path
        self.output_bin_path = output_bin_path
        self.save_path = save_path
        self.allow_frozen_write = allow_frozen_write
        self.device = device

    @staticmethod
    def _zorunlu(deger: Optional[str], ad: str, ornek: str) -> str:
        """Varsayilani kaldirilmis parametreyi fail-closed dogrular (saf fonksiyon)."""
        if not deger:
            raise RuntimeError(
                f"DURDURULDU: {ad} verilmedi. Varsayilan KALDIRILDI (T-0087): "
                f"eski varsayilanlar silinmis bir checkpoint adi ve BAYAT bir sozluk "
                f"tasiyordu. ACIKCA verin, or. {ad}={ornek}")
        return deger

    def get_pending_count(self) -> int:
        """Returns the number of pending samples in future_train_vector.jsonl."""
        if not os.path.exists(self.future_train_path):
            return 0
        with open(self.future_train_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def compile_backlog_to_bin(
        self,
        block_size: int = 64,
        oversample_factor: int = 20,
        replay_samples: int = 25
    ) -> Tuple[str, int]:
        """
        Reads future_train_vector.jsonl, tokenizes all records into morpheme token IDs,
        oversamples them so the small model sees enough gradient steps, and writes to uint16 .bin file.

        T-0087 (FAIL-CLOSED): `vocab_path` ZORUNLU ve kapisi EN BAŞTA çalışır — eksik
        argüman, veri dosyasının varlığından ÖNCE raporlanır (yapılandırma hatası, veri
        hatasından önce gelir; ölçüldü: aksi sırada test fikstürü FileNotFoundError alıyordu).
        """
        vocab_path = self._zorunlu(self.vocab_path, "vocab_path",
                                   "'data/rebuild/vocab_anka_r1_33114.json'")

        if not os.path.exists(self.future_train_path):
            raise FileNotFoundError(f"'{self.future_train_path}' bulunamadı.")

        vocab = Vocabulary()
        vocab.load(vocab_path)
        lexicon = LexiconManager()
        lexicon.load_from_tsv(self.lexicon_path)
        compiler = CrystalCompiler(lexicon, build_default_graph())
        tokenizer = KristalTokenizer(compiler, vocab)

        records = []
        with open(self.future_train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Normalize prompt object for SFT via canonical contract
                    inst = data.get("instruction", "Belgeye göre cevapla.")
                    inp = data.get("input", "")
                    out = data.get("output", "")
                    raw_str = render_example(inst, inp, out)
                    token_ids = tokenizer.encode(raw_str)
                    if len(token_ids) > 2:
                        records.append(token_ids)
                except Exception:
                    continue

        if not records:
            raise ValueError(f"'{self.future_train_path}' dosyasında geçerli veri bulunamadı.")

        backlog_count = len(records)
        # Add a replay buffer from foundation dataset to prevent catastrophic forgetting
        if replay_samples > 0:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            hs_path = os.path.join(base_dir, "data", "pedagogy", "high_school_foundation_dataset.jsonl")
            if os.path.exists(hs_path):
                with open(hs_path, "r", encoding="utf-8") as hf:
                    for idx, h_line in enumerate(hf):
                        if idx >= replay_samples:
                            break
                        h_line = h_line.strip()
                        if h_line:
                            try:
                                d = json.loads(h_line)
                                raw_s = json.dumps(d, ensure_ascii=False)
                                tids = tokenizer.encode(raw_s)
                                if len(tids) > 2:
                                    records.append(tids)
                            except Exception:
                                pass

        # Oversample small batches so model learns thoroughly
        all_tokens = []
        for _ in range(oversample_factor):
            for rec in records:
                all_tokens.extend(rec)

        token_arr = np.array(all_tokens, dtype=np.uint16)
        os.makedirs(os.path.dirname(os.path.abspath(self.output_bin_path)), exist_ok=True)
        token_arr.tofile(self.output_bin_path)

        # Write metadata
        meta = {
            "block_size": block_size,
            "total_tokens": len(all_tokens),
            "backlog_samples": backlog_count,
            "total_samples": len(records),
            "oversample_factor": oversample_factor,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        with open(self.output_bin_path + ".meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

        return self.output_bin_path, backlog_count

    def run_training(
        self,
        steps: int = 30,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        block_size: int = 64
    ) -> Dict[str, Any]:
        """
        Executes fine-tuning using train.py on the compiled binary dataset.
        Archives trained records upon success.

        T-0087 (FAIL-CLOSED): `model_path` (soyagaci sozlesmesi) ve `save_path`
        (T-0085'ten sonra `train.py`'nin ZORUNLU kayit hedefi) ACIKCA verilmelidir.
        Kontroller `compile_backlog_to_bin`'DEN ONCE calisir ⇒ eksik argumanda
        HICBIR YAZIM olmaz.
        """
        self._zorunlu(self.model_path, "model_path", "'data/anka_a1r.pt'")
        save_path = self._zorunlu(self.save_path, "save_path", "'data/anka_a2.pt'")

        bin_path, sample_count = self.compile_backlog_to_bin(block_size=block_size)

        python_bin = sys.executable
        train_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "train.py")

        cmd = [
            python_bin,
            train_script,
            "--data", bin_path,
            "--steps", str(steps),
            "--batch-size", str(batch_size),
            "--lr", str(learning_rate),
            "--device", self.device,
            "--save-path", save_path
        ]
        # Donmus hedefe yazim OPERATOR onayi ister: bayrak yalnizca cagiran acikca
        # istediyse gecilir; aksi halde train.py'nin kendi kapisi kosumu durdurur.
        if self.allow_frozen_write:
            cmd.append("--allow-frozen-write")

        start_time = datetime.now(timezone.utc)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stdout,
                "duration_sec": duration,
                "samples": sample_count
            }

        # Archive processed records
        self._archive_processed_records()

        return {
            "status": "success",
            "samples_trained": sample_count,
            "steps": steps,
            "duration_sec": round(duration, 2),
            "model_path": self.model_path,
            "bin_dataset": bin_path,
            "log_snippet": result.stdout[-500:] if result.stdout else ""
        }

    def _archive_processed_records(self):
        """Moves current future_train_vector.jsonl content to the archive file."""
        if not os.path.exists(self.future_train_path):
            return

        with open(self.future_train_path, "r", encoding="utf-8") as sf:
            content = sf.read()

        if content.strip():
            with open(self.archive_path, "a", encoding="utf-8") as df:
                df.write(content)

        # Clear active backlog
        with open(self.future_train_path, "w", encoding="utf-8") as sf:
            sf.write("")
