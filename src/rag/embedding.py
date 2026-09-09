#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import random
from typing import List, Tuple
from qdrant_client.http import models
from src.llm.tokenizer import get_morpheme_weight

def generate_kristal_vector(token_ids: List[int], crystal_tags_str: str, size: int = 768) -> List[float]:
    """
    Kristal-Vektörel Mimarisi için yoğun (dense), konumsal (positional) 
    ve ağırlıklı (weighted) deterministik vektör üretimi.
    
    Uygulanan Mantıksal Koruma Katmanı (Sprint 3):
    Kelimeler bazında gruplama yapılarak, NEG veya IMPOTENTIAL_NEG barındıran 
    kelimelerin vektörü Sign Inversion (Kutup Değişimi) ile -1.0 ile çarpılır.
    """
    if not token_ids:
        return [0.0] * size
        
    final_vec = [0.0] * size
    tags = crystal_tags_str.split() if crystal_tags_str else []
    
    # 1. Kelime sınırlarını belirleyerek morfemleri kelime bazlı grupla
    words = []
    current_word = []
    for pos_idx, tid in enumerate(token_ids):
        tag = tags[pos_idx] if pos_idx < len(tags) else ""
        
        is_control = tag in [
            "<BOS>", "<EOS>", "<PAD>", "<UNK>",
            "<INSTRUCTION>", "</INSTRUCTION>",
            "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>",
            "<NUMBER>", "<SYMBOL>"
        ]
        is_root = not is_control and (
            tag == "<PROPER_NOUN>" or (
                not tag.startswith("DERIV_") and 
                not tag.startswith(("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_")) and 
                tag not in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG")
            )
        )
        
        if is_control or is_root:
            if current_word:
                words.append(current_word)
            current_word = [(pos_idx, tid, tag)]
        else:
            current_word.append((pos_idx, tid, tag))
    if current_word:
        words.append(current_word)
        
    # 2. Her kelime için vektör üret ve olumsuzluk durumunda kutup değişimi yap
    for word_tokens in words:
        word_vec = [0.0] * size
        has_negation = False
        
        for pos_idx, tid, tag in word_tokens:
            if tag in ("NEG", "IMPOTENTIAL_NEG"):
                has_negation = True
                
            w = get_morpheme_weight(tag)
            pos_weight = 1.0 / (1.0 + 0.05 * pos_idx)
            
            # Deterministik pseudo-random tabanlı morfem embedding projeksiyonu
            seed = int(hashlib.md5(f"morpheme_{tid}".encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            
            for i in range(size):
                base_val = rng.uniform(-1.0, 1.0)
                word_vec[i] += base_val * w * pos_weight
                
        # Kutup Değişimi (Sign Inversion)
        if has_negation:
            word_vec = [-v for v in word_vec]
            
        for i in range(size):
            final_vec[i] += word_vec[i]
            
    # L2 Normalizasyonu
    norm = sum(x * x for x in final_vec) ** 0.5
    if norm > 0:
        final_vec = [x / norm for x in final_vec]
        
    return final_vec

def generate_sparse_vector(token_ids: List[int], crystal_tags_str: str) -> models.SparseVector:
    """
    Kristal morfemlerin sıklık ve anlamsal ağırlıklarına göre Qdrant SparseVector üretimi.
    """
    if not token_ids:
        return models.SparseVector(indices=[], values=[])
        
    tags = crystal_tags_str.split() if crystal_tags_str else []
    sparse_dict = {}
    
    for idx, tid in enumerate(token_ids):
        tag = tags[idx] if idx < len(tags) else ""
        if tag in ["<BOS>", "<EOS>", "<PAD>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>"]:
            continue
            
        weight = get_morpheme_weight(tag)
        sparse_dict[tid] = sparse_dict.get(tid, 0.0) + weight
        
    indices = list(sparse_dict.keys())
    values = [float(sparse_dict[i]) for i in indices]
    
    return models.SparseVector(indices=indices, values=values)
