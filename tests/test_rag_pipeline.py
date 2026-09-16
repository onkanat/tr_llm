# -*- coding: utf-8 -*-
"""
tests.test_rag_pipeline
Kanonik RAG boru hattı birim ve entegrasyon testleri.

Qdrant sunucusu, MPS ve checkpoint gerektirmeden saf mock nesneler ile
eşik kapısı, prompt koşullaması, deterministik decoding ve negatif durumları test eder.
"""

import json
import re
from typing import Any, Dict, List, Optional
import pytest
import torch

from src.llm.prompt_contract import build_rag_input
from src.rag.rag_pipeline import (
    RagPipeline,
    build_query_vectors,
    retrieve_context,
    is_context_usable,
    build_rag_prompt_tokens,
    greedy_decode,
    RAG_MATCH_THRESHOLD,
)


class MockVocab:
    stoi = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "<INSTRUCTION>": 4,
        "</INSTRUCTION>": 5,
        "<INPUT>": 6,
        "</INPUT>": 7,
        "<OUTPUT>": 8,
        "</OUTPUT>": 9,
        "<BELGE>": 10,
        "</BELGE>": 11,
    }
    itos = {v: k for k, v in stoi.items()}

    def decode(self, idx: int) -> str:
        return self.itos.get(idx, f"tok_{idx}")


class MockTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or MockVocab()
        self.word2id = dict(self.vocab.stoi)
        self.id2word = dict(self.vocab.itos)
        self._next_id = 50

    def encode(self, text: str) -> List[int]:
        raw = text.strip()
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
                parts = []
                if data.get("instruction"):
                    parts.extend(["<INSTRUCTION>", data["instruction"], "</INSTRUCTION>"])
                if "input" in data:
                    parts.extend(["<INPUT>", data["input"], "</INPUT>"])
                if "output" in data or "output" in data.keys():
                    parts.extend(["<OUTPUT>", data.get("output", ""), "</OUTPUT>"])
                raw = " ".join(parts).strip()
            except Exception:
                pass

        tokens = re.findall(r'<[^>]+>|[^\s<"]+', raw)
        ids = []
        for t in tokens:
            if t not in self.word2id:
                self.word2id[t] = self._next_id
                self.id2word[self._next_id] = t
                self._next_id += 1
            ids.append(self.word2id[t])
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join([self.id2word.get(i, self.vocab.decode(i)) for i in ids])


class MockMemory:
    def __init__(
        self,
        return_score: float = 0.41,
        doc_text: str = "Gürgen ağacı serttir.",
        return_none: bool = False,
    ):
        self.return_score = return_score
        self.doc_text = doc_text
        self.return_none = return_none

    def hybrid_recall(
        self,
        dense: Any,
        sparse: Any,
        top_k: int = 1,
        query_tags: str = "",
    ) -> List[Dict[str, Any]]:
        if self.return_none:
            return []
        return [{
            "text": self.doc_text,
            "score": self.return_score,
            "metadata": {
                "crystal_tags": "gürgen ağaç",
                "token_ids": [10, 11]
            }
        }]


class MockDeterministicArgmaxModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 60):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        batch_size, seq_len = x.shape
        last_token = int(x[0, -1].item())
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        # Deterministik sonraki token (10..39 aralığında, EOS=2 ile çakışmaz)
        next_tok = (last_token + 7) % 30 + 10
        logits[0, -1, next_tok] = 10.0
        return logits, None


# ==============================================================================
# Zorunlu Test Vakaları (i - vi)
# ==============================================================================

def test_rag_pipeline_gate_low_similarity_skips_conditioning():
    """
    Vaka (i): match_score=0.39 -> koşullama YOK (conditioned is False, prompt'ta <BELGE> geçmez).
    """
    vocab = MockVocab()
    tokenizer = MockTokenizer(vocab)
    memory = MockMemory(return_score=0.39, doc_text="Gürgen ağacı aşırı sert ve toktur.")
    model = MockDeterministicArgmaxModel()

    pipeline = RagPipeline(
        tokenizer=tokenizer,
        vocab=vocab,
        memory=memory,
        model=model,
        device=torch.device("cpu"),
        threshold=RAG_MATCH_THRESHOLD,
    )

    result = pipeline.answer("marangozluk")

    # 1. Koşullama kapalı olmalı
    assert result["conditioned"] is False
    assert result["match_score"] == 0.39

    # 2. Prompt tokenleri içinde <BELGE> veya </BELGE> ASLA geçmemeli
    decoded_prompt = tokenizer.decode(result["prompt_tokens"])
    assert "<BELGE>" not in decoded_prompt
    assert "</BELGE>" not in decoded_prompt

    # 3. Prompt çıplak sorguyu içermeli
    assert "marangozluk" in decoded_prompt


def test_rag_pipeline_gate_high_similarity_enables_conditioning():
    """
    Vaka (ii): match_score=0.41 -> koşullama VAR ve build_rag_input çıktısıyla birebir.
    """
    vocab = MockVocab()
    tokenizer = MockTokenizer(vocab)
    doc_text = "Gürgen ağacı aşırı sert ve toktur."
    query = "marangozluk"
    memory = MockMemory(return_score=0.41, doc_text=doc_text)
    model = MockDeterministicArgmaxModel()

    pipeline = RagPipeline(
        tokenizer=tokenizer,
        vocab=vocab,
        memory=memory,
        model=model,
        device=torch.device("cpu"),
        threshold=RAG_MATCH_THRESHOLD,
    )

    result = pipeline.answer(query)

    # 1. Koşullama açık olmalı
    assert result["conditioned"] is True
    assert result["match_score"] == 0.41

    # 2. Prompt içinde kanonik build_rag_input ifadesi birebir bulunmalı
    decoded_prompt = tokenizer.decode(result["prompt_tokens"])
    expected_rag_input = build_rag_input(doc_text, query)
    assert expected_rag_input in decoded_prompt
    assert "<BELGE>" in decoded_prompt
    assert "</BELGE>" in decoded_prompt


def test_rag_pipeline_gate_boundary_threshold_40():
    """
    Vaka (iii): Sınır durumu: tam 0.40 -> davranış bugünkü ajandan miras (>= 0.40).
    is_context_usable(0.40) == True, is_context_usable(0.39999) == False.
    """
    assert is_context_usable(0.40, RAG_MATCH_THRESHOLD) is True
    assert is_context_usable(0.40001, RAG_MATCH_THRESHOLD) is True
    assert is_context_usable(0.39999, RAG_MATCH_THRESHOLD) is False

    vocab = MockVocab()
    tokenizer = MockTokenizer(vocab)
    memory = MockMemory(return_score=0.40, doc_text="Sınır belgesi.")
    model = MockDeterministicArgmaxModel()

    pipeline = RagPipeline(
        tokenizer=tokenizer,
        vocab=vocab,
        memory=memory,
        model=model,
        device=torch.device("cpu"),
        threshold=RAG_MATCH_THRESHOLD,
    )

    result = pipeline.answer("sınır sorgusu")
    assert result["conditioned"] is True
    assert result["match_score"] == 0.40
    assert "<BELGE>" in tokenizer.decode(result["prompt_tokens"])


def test_rag_pipeline_greedy_decode_determinism():
    """
    Vaka (iv): Determinizm: aynı girdiyle greedy_decode iki kez çağrıldığında
    birebir aynı token dizisini üretir (örnekleme / rastlantısallık yoktur).
    """
    model = MockDeterministicArgmaxModel()
    prompt_tokens = [1, 4, 10, 11, 8]
    device = torch.device("cpu")

    out_tokens_1 = greedy_decode(
        model=model,
        token_ids=prompt_tokens,
        eos_id=2,
        device=device,
        max_new_tokens=10,
    )

    out_tokens_2 = greedy_decode(
        model=model,
        token_ids=prompt_tokens,
        eos_id=2,
        device=device,
        max_new_tokens=10,
    )

    assert len(out_tokens_1) > 0
    assert out_tokens_1 == out_tokens_2


def test_rag_pipeline_greedy_decode_stops_at_eos_and_respects_max_len():
    """
    Vaka (v): eos_id üretilince döngü durur; max_len sınırı aşılmaz.
    """
    # 1. EOS durma kontrolü
    class MockEosModel(torch.nn.Module):
        def __init__(self, stop_at_step: int = 3, eos_id: int = 2):
            super().__init__()
            self.step = 0
            self.stop_at_step = stop_at_step
            self.eos_id = eos_id

        def forward(self, x):
            logits = torch.zeros(x.shape[0], x.shape[1], 20)
            next_id = self.eos_id if self.step >= self.stop_at_step else 15
            self.step += 1
            logits[0, -1, next_id] = 10.0
            return logits, None

    eos_model = MockEosModel(stop_at_step=3, eos_id=2)
    gen = greedy_decode(
        model=eos_model,
        token_ids=[1, 4],
        eos_id=2,
        device=torch.device("cpu"),
        max_new_tokens=50,
    )
    # 3 token ürettikten sonra 4. adımda EOS gelir ve döngü durur
    assert len(gen) == 3

    # 2. max_len sınırı kontrolü
    class MockNeverEosModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], x.shape[1], 20)
            logits[0, -1, 15] = 10.0
            return logits, None

    never_eos_model = MockNeverEosModel()
    prompt_len = 120
    max_len = 128
    prompt_tokens = [1] * prompt_len
    gen_tokens = greedy_decode(
        model=never_eos_model,
        token_ids=prompt_tokens,
        eos_id=2,
        device=torch.device("cpu"),
        max_new_tokens=50,
        max_len=max_len,
    )
    # prompt + gen toplamı max_len'i kesinlikle geçemez
    assert len(prompt_tokens) + len(gen_tokens) <= max_len
    assert len(gen_tokens) == (max_len - prompt_len)


def test_rag_pipeline_negative_retrieve_none_fallback():
    """
    Vaka (vi): Negatif durum: retrieve_context None döndüğünde answer patlamaz,
    conditioned is False döner.
    """
    vocab = MockVocab()
    tokenizer = MockTokenizer(vocab)
    memory = MockMemory(return_none=True)
    model = MockDeterministicArgmaxModel()

    pipeline = RagPipeline(
        tokenizer=tokenizer,
        vocab=vocab,
        memory=memory,
        model=model,
        device=torch.device("cpu"),
        threshold=RAG_MATCH_THRESHOLD,
    )

    result = pipeline.answer("hiçbir şey bulunamayan sorgu")

    assert result["retrieved_doc"] is None
    assert result["match_score"] == 0.0
    assert result["conditioned"] is False
    decoded_prompt = tokenizer.decode(result["prompt_tokens"])
    assert "<BELGE>" not in decoded_prompt
    assert "</BELGE>" not in decoded_prompt

    # retrieve_context None bellek kontrolü
    assert retrieve_context(None, [0.0], {}, "tags") is None
