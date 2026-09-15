#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: ÖRNEK DÜZEYİNDE HİZALI VE MASKELEMELİ VERİ SETİ
========================================================================
- Örnek düzeyinde (sample-level) dizilim ve padding (<PAD>).
- İstem pozisyonlarına (INSTRUCTION + INPUT + OUTPUT etiketi) -100 hedef maskesi.
- Yalnızca OUTPUT tokenları üzerinde kayıp optimizasyonu.
- Şablon imzası (template skeleton hash) ile sızıntısız train/val ayrımı (E2).
"""

import os
import re
import json
import hashlib
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import render_prompt


def get_template_signature(record: Dict[str, str]) -> str:
    """
    Computes a structural skeleton hash for a dataset record.
    Strips specific entities and numbers to isolate the syntactic template.
    """
    inst = record.get("instruction", "").strip()
    inst_norm = " ".join(inst.split())
    inp = record.get("input", "").strip()
    inp_skeleton = re.sub(r'[\d\'".,!?:;()-]+', '', inp[:60]).strip()
    sig_raw = f"{inst_norm} || {inp_skeleton}"
    return hashlib.md5(sig_raw.encode("utf-8")).hexdigest()


def template_aware_train_val_split(
    records: List[Dict[str, str]], 
    val_ratio: float = 0.1, 
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Splits records into train and val such that all samples sharing the same
    template signature are kept strictly within either train or val (Zero Leakage).
    """
    groups: Dict[str, List[int]] = {}
    for idx, r in enumerate(records):
        sig = get_template_signature(r)
        if sig not in groups:
            groups[sig] = []
        groups[sig].append(idx)

    rng = random.Random(seed)
    signatures = list(groups.keys())
    rng.shuffle(signatures)

    total_records = len(records)
    val_target = max(1, int(total_records * val_ratio))

    train_indices: List[int] = []
    val_indices: List[int] = []

    for sig in signatures:
        group = groups[sig]
        if len(val_indices) + len(group) <= val_target or len(val_indices) == 0:
            val_indices.extend(group)
        else:
            train_indices.extend(group)

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    return train_records, val_records


class SampleAlignedDataset(Dataset):
    """
    PyTorch Dataset providing sample-aligned sequences with causal prompt masking.
    
    Format:
      prompt: <BOS> <INSTRUCTION> {inst} </INSTRUCTION> <INPUT> {inp} </INPUT> <OUTPUT>
      output: {out} <EOS>
      
    Targets:
      Indices predicting tokens inside prompt: -100
      Indices predicting tokens inside output: token_id
      Padding positions: -100
    """
    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: KristalTokenizer,
        vocab: Vocabulary,
        block_size: int = 256
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.block_size = block_size
        
        self.pad_id = vocab.stoi.get("<PAD>", 1)
        self.bos_id = vocab.stoi.get("<BOS>", 2)
        self.eos_id = vocab.stoi.get("<EOS>", 3)
        self.out_start_id = vocab.stoi.get("<OUTPUT>", 8)
        self.out_end_id = vocab.stoi.get("</OUTPUT>", 9)
        
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._encode_all(records)

    def _encode_all(self, records: List[Dict[str, str]]):
        for rec in records:
            inst = rec.get("instruction", "").strip()
            inp = rec.get("input", "").strip()
            out = rec.get("output", "").strip()

            # Encode prompt via canonical contract
            prompt_str = render_prompt(inst, inp)
            
            prompt_ids = self.tokenizer.encode(prompt_str)
            # Strip trailing EOS if tokenizer added it
            if prompt_ids and prompt_ids[-1] == self.eos_id:
                prompt_ids = prompt_ids[:-1]
                
            # Encode output
            out_ids = self.tokenizer.encode(out)
            # Strip leading BOS
            if out_ids and out_ids[0] == self.bos_id:
                out_ids = out_ids[1:]
            # Ensure trailing EOS
            if not out_ids or out_ids[-1] != self.eos_id:
                out_ids.append(self.eos_id)
                
            full_seq = prompt_ids + out_ids
            P = len(prompt_ids)
            
            if len(full_seq) < 3:
                continue
                
            # Truncate if exceeds block_size + 1
            if len(full_seq) > self.block_size + 1:
                # Keep prompt prefix (<BOS> + inst) and tail (<OUTPUT> + out)
                excess = len(full_seq) - (self.block_size + 1)
                if P - 4 > excess:
                    prompt_ids = prompt_ids[:2] + prompt_ids[2 + excess:]
                    P = len(prompt_ids)
                    full_seq = prompt_ids + out_ids
                else:
                    full_seq = full_seq[-(self.block_size + 1):]
                    P = max(1, P - excess)

            # Build inputs and targets
            x_ids = full_seq[:-1]
            y_ids = full_seq[1:]
            
            # Mask prompt targets with -100
            # For i < P - 1, y_ids[i] is a prompt token
            # At i = P - 1, x_ids[i] is <OUTPUT> and y_ids[i] is the first response token (KEPT)
            y_masked = []
            for i, target_token in enumerate(y_ids):
                if i < P - 1:
                    y_masked.append(-100)
                else:
                    y_masked.append(target_token)

            # Pad to block_size
            curr_len = len(x_ids)
            if curr_len < self.block_size:
                pad_len = self.block_size - curr_len
                x_ids = x_ids + [self.pad_id] * pad_len
                y_masked = y_masked + [-100] * pad_len
            else:
                x_ids = x_ids[:self.block_size]
                y_masked = y_masked[:self.block_size]

            x_tensor = torch.tensor(x_ids, dtype=torch.long)
            y_tensor = torch.tensor(y_masked, dtype=torch.long)
            self.samples.append((x_tensor, y_tensor))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def load_jsonl_records(paths: List[str]) -> List[Dict[str, str]]:
    """Loads all valid JSONL items from a list of file paths."""
    records = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if isinstance(d, dict):
                        records.append(d)
                except Exception:
                    continue
    return records
