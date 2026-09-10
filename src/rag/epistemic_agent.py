#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: EPİSTEMİK OTONOM DÖNGÜ (EPISTEMIC CURIOSITY AGENT)
=============================================================================
Bu modül; KristalLM, Merak Motoru (CuriosityEngine), Tri-Modal Dinamik Router
ve Vektörel Bellek (VectorMemory) bileşenlerini bir araya getirerek otonom
epistemik döngüyü işletir.

İşleyiş Mantığı:
1. Model girdi alır, ilk tahmin logits ve z gizli durumu üzerinden Shannon Entropisi H(z) hesaplanır.
2. H(z) > tau ise Epistemik Boşluk (Epistemic Gap) tespit edilir ve q_merak vektörü üretilir.
3. Tri-Modal Router; Girdi (P), Merak (q_merak) ve RAG bağlamını harmanlayarak dinamik yönlendirme yapar.
4. Vektörel Bellek (Qdrant) taranır. Arama skoru >= 0.85 ise kaliteli kanıt belgesi temin edilir.
5. "Model Anlamaz?!" Durumu:
   Yüksek uyumlu (>= 0.85) belge bağlama verilmesine rağmen modelin belirsizliği sürüyorsa
   (post-retrieval H(z) > tau veya tatmin edici olmayan/belirsiz yanıt), bu örnek
   modelin bir sonraki eğitim külliyatına dahil edilmesi amacıyla `future_train_vector.jsonl`
   dosyasına otomatik eklenir (append).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("EpistemicAgent")

from src.rag.merak import CuriosityEngine
from src.llm.router import TriModalRouter
from src.rag.vector_memory import VectorMemory
from src.rag.embedding import generate_kristal_vector, generate_sparse_vector


class EpistemicCuriosityAgent:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        memory: VectorMemory,
        general_memory: Optional[VectorMemory] = None,
        decompiler: Optional[Any] = None,
        curiosity_engine: Optional[CuriosityEngine] = None,
        router: Optional[TriModalRouter] = None,
        tau: float = 2.5,
        similarity_threshold: float = 0.85,
        future_train_path: str = "data/future_train_vector.jsonl",
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab if hasattr(tokenizer, "vocab") else None
        self.memory = memory
        self.general_memory = general_memory
        self.decompiler = decompiler
        self.device = torch.device(device)
        self.tau = tau
        self.similarity_threshold = similarity_threshold
        self.future_train_path = future_train_path
        
        # Initialize or assign CuriosityEngine
        hidden_dim = getattr(model, "embedding", None)
        emb_dim = getattr(hidden_dim, "embedding", None)
        n_embd = emb_dim.embedding_dim if emb_dim is not None else 768
        
        self.curiosity_engine = curiosity_engine or CuriosityEngine(
            hidden_dim=n_embd,
            curiosity_dim=n_embd,
            tau=tau
        )
        
        # Initialize or assign TriModalRouter
        self.router = router or TriModalRouter(
            prompt_dim=n_embd,
            merak_dim=n_embd,
            rag_dim=n_embd,
            router_dim=256,
            num_experts=4,
            top_k=2,
            expert_names=["grammar_core", "pedagogy", "carpenter", "legal"]
        )
        
        # Move helper modules to device
        self.curiosity_engine.to(self.device)
        self.router.to(self.device)
        self.curiosity_engine.eval()
        self.router.eval()

    def calculate_prompt_embedding(self, token_ids: List[int]) -> torch.Tensor:
        """Computes a prompt representation vector P from the token sequence."""
        if not token_ids:
            return torch.zeros(1, self.curiosity_engine.hidden_dim, device=self.device)
            
        vocab_size = getattr(self.model, "vocab_size", 31328)
        if hasattr(self.model, "embedding") and hasattr(self.model.embedding, "embedding"):
            vocab_size = self.model.embedding.embedding.weight.shape[0]
            
        valid_ids = [t for t in token_ids if 0 <= t < vocab_size]
        if not valid_ids:
            return torch.zeros(1, self.curiosity_engine.hidden_dim, device=self.device)

        with torch.no_grad():
            x = torch.tensor([valid_ids], dtype=torch.long, device=self.device)
            if hasattr(self.model, "embedding"):
                emb = self.model.embedding(x)
                return emb.mean(dim=1) # Mean pooling
            return torch.zeros(1, self.curiosity_engine.hidden_dim, device=self.device)

    def evaluate_entropy_and_merak(self, token_ids: List[int]) -> Tuple[float, bool, torch.Tensor]:
        """
        Runs KristalLM on token_ids, returns (entropy_val, needs_retrieval, q_merak).
        """
        vocab_size = getattr(self.model, "vocab_size", 31328)
        if hasattr(self.model, "embedding") and hasattr(self.model.embedding, "embedding"):
            vocab_size = self.model.embedding.embedding.weight.shape[0]
            
        valid_ids = [t for t in token_ids if 0 <= t < vocab_size]
        if not valid_ids:
            valid_ids = [0]

        unk_id = self.vocab.stoi.get("<UNK>", 1) if hasattr(self, "vocab") and self.vocab else 1
        has_unk = any(t == unk_id for t in token_ids)
        if has_unk:
            logger.info("[Merak Motoru] Girdide <UNK> (tanınmayan morfem) tespit edildi -> Epistemik merak tetiklendi!")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([valid_ids], dtype=torch.long, device=self.device)
            # Call model with hidden states if supported
            res = self.model(x, return_hidden_states=True) if "return_hidden_states" in self.model.forward.__code__.co_varnames else self.model(x)
            
            if len(res) == 3:
                logits, _, x_emb = res
                last_hidden = x_emb[:, -1, :]
            else:
                logits, _ = res
                # Fallback representation
                last_hidden = torch.zeros(logits.shape[0], self.curiosity_engine.hidden_dim, device=self.device)
                
            last_logits = logits[:, -1, :]
            entropy, needs_retrieval, q_merak = self.curiosity_engine(
                last_hidden, 
                last_logits,
                has_unk=has_unk,
                unk_token_id=unk_id
            )
            
            return entropy.item(), bool(needs_retrieval.item()), q_merak

    def search_memory(self, query_token_ids: List[int], query_tags: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Queries VectorMemory (primary + optional fallback) and returns the top match and collection name.
        """
        dense_vec = generate_kristal_vector(query_token_ids, query_tags)
        sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
        
        results = self.memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
        source = self.memory.collection_name
        
        if not (results and results[0].get("has_root_match", True)) and self.general_memory:
            gen_results = self.general_memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
            if gen_results and gen_results[0].get("has_root_match", True):
                results = gen_results
                source = self.general_memory.collection_name
                
        if results and results[0].get("has_root_match", True):
            return results[0], source
            
        return None, source

    def record_to_future_train(self, record: Dict[str, Any]) -> None:
        """Appends an epistemic gap record to the future_train_vector.jsonl file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.future_train_path)), exist_ok=True)
        with open(self.future_train_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 40,
        repetition_penalty: float = 1.4,
        repetition_window: int = 10
    ) -> Tuple[List[int], float]:
        """
        Autoregressively generates tokens from prompt_tokens.
        Also measures post-generation average entropy or final step entropy.
        """
        self.model.eval()
        vocab_size = getattr(self.model, "vocab_size", 31328)
        if hasattr(self.model, "embedding") and hasattr(self.model.embedding, "embedding"):
            vocab_size = self.model.embedding.embedding.weight.shape[0]
        generated = [t for t in prompt_tokens if 0 <= t < vocab_size]
        if not generated:
            generated = [0]
            
        eos_id = self.vocab.stoi.get("<EOS>", -1) if self.vocab else -1
        output_end_id = self.vocab.stoi.get("</OUTPUT>", -1) if self.vocab else -1
        
        last_step_entropy = 0.0
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([generated], dtype=torch.long, device=self.device)
                logits, _ = self.model(x)
                logits_last = logits[0, -1, :]
                
                # Measure current step entropy
                step_entropy = self.curiosity_engine.calculate_entropy(logits_last.unsqueeze(0)).item()
                last_step_entropy = step_entropy
                
                # Repetition penalty
                output_tokens = generated[len(prompt_tokens):]
                if repetition_penalty > 1.0 and output_tokens:
                    window_tokens = output_tokens[-repetition_window:]
                    for token_id in set(window_tokens):
                        if logits_last[token_id] > 0:
                            logits_last[token_id] /= repetition_penalty
                        else:
                            logits_last[token_id] *= repetition_penalty
                            
                pred_id = torch.argmax(logits_last).item()
                generated.append(pred_id)
                
                if pred_id in (eos_id, output_end_id):
                    break
                    
        return generated[len(prompt_tokens):], last_step_entropy

    def process_query(
        self,
        query: str,
        instruction: str = "Belgeye göre cevapla.",
        force_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the full end-to-end Epistemic Curiosity Loop:
        1. Encodes query and analyzes initial uncertainty (CuriosityEngine).
        2. Routes query with TriModalRouter.
        3. Recalls document from VectorMemory.
        4. If match score >= similarity_threshold (0.85) and model fails to understand (epistemic gap remains):
           Appends sample to future_train_vector.jsonl.
        5. Returns structured telemetry and answer.
        """
        # Encode raw query
        query_token_ids = self.tokenizer.encode(query)
        query_tags = self.tokenizer.decode(query_token_ids)
        clean_query_tags = query_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
        
        # Step 1: Initial Prompt Entropy & Epistemic Gap Detection
        prompt_dict_pre = {
            "instruction": instruction,
            "input": query,
            "output": ""
        }
        pre_tokens = self.tokenizer.encode(json.dumps(prompt_dict_pre, ensure_ascii=False))
        output_start_id = self.vocab.stoi.get("<OUTPUT>", -1) if self.vocab else -1
        if output_start_id in pre_tokens:
            eval_pre_tokens = pre_tokens[:pre_tokens.index(output_start_id) + 1]
        else:
            eval_pre_tokens = pre_tokens
            
        entropy_pre, needs_retrieval, q_merak = self.evaluate_entropy_and_merak(eval_pre_tokens)
        
        # Step 2: Vector Memory Search (triggered by curiosity or force_rag)
        retrieval_triggered = needs_retrieval or force_rag
        retrieved_doc = None
        source_coll = None
        match_score = 0.0
        
        if retrieval_triggered:
            retrieved_doc, source_coll = self.search_memory(query_token_ids, query_tags)
            if retrieved_doc:
                match_score = float(retrieved_doc.get("score", 0.0))
                
        # Step 3: Tri-Modal Routing
        prompt_vec = self.calculate_prompt_embedding(eval_pre_tokens)
        rag_vec = None
        if retrieved_doc:
            doc_token_ids = retrieved_doc.get("metadata", {}).get("token_ids", [])
            if doc_token_ids:
                rag_vec = self.calculate_prompt_embedding(doc_token_ids)
                
        router_weights, router_indices, e_route = self.router(
            prompt_vec=prompt_vec,
            merak_vec=q_merak,
            rag_vec=rag_vec
        )
        expert_names = self.router.get_selected_expert_names(router_indices)[0]
        
        # Step 4: Conditioning with Document (if retrieved) & Generating Output
        augmented_input = query
        doc_crystal_tags = ""
        doc_text = ""
        
        if retrieved_doc and match_score >= 0.40: # Valid context
            doc_text = retrieved_doc.get("text", "")
            doc_crystal_tags = retrieved_doc.get("metadata", {}).get("crystal_tags", "")
            clean_doc_tags = doc_crystal_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
            augmented_input = f"belge: {clean_doc_tags} sorgu: {clean_query_tags}"
            
        prompt_dict_aug = {
            "instruction": instruction,
            "input": augmented_input,
            "output": ""
        }
        aug_tokens = self.tokenizer.encode(json.dumps(prompt_dict_aug, ensure_ascii=False))
        if output_start_id in aug_tokens:
            eval_aug_tokens = aug_tokens[:aug_tokens.index(output_start_id) + 1]
        else:
            eval_aug_tokens = aug_tokens
            
        gen_tokens, entropy_post = self.generate_tokens(eval_aug_tokens, max_new_tokens=45)
        
        eos_id = self.vocab.stoi.get("<EOS>", -1) if self.vocab else -1
        output_end_id = self.vocab.stoi.get("</OUTPUT>", -1) if self.vocab else -1
        clean_gen_tokens = [tid for tid in gen_tokens if tid not in (eos_id, output_end_id)]
        morpheme_output = " ".join([self.vocab.decode(tid) for tid in clean_gen_tokens]) if self.vocab else ""
        
        decompiled_text = ""
        if self.decompiler and morpheme_output:
            try:
                decompiled_text = self.decompiler.decompile_sentence(morpheme_output)
            except Exception:
                decompiled_text = morpheme_output
                
        # Step 5: "Model Anlamaz?!" (Epistemic Failure) Evaluation:
        # High similarity match (>= similarity_threshold, e.g. 0.85) AND
        # model remains uncertain (entropy_post > tau or uncertain keywords)
        epistemic_failure = False
        future_train_recorded = False
        
        is_high_similarity = (match_score >= self.similarity_threshold)
        
        # Check if the model failed to understand:
        # A: Post entropy remains high (> tau)
        # B: Generated output expresses lack of information ("bilgi yok", empty, or only unk)
        has_high_post_entropy = (entropy_post > self.tau)
        expresses_uncertainty = any(phrase in decompiled_text.lower() for phrase in ["bilgi yok", "bulunamaz", "bilinmiyor"]) or (morpheme_output.count("<UNK>") >= 2)
        
        if is_high_similarity and (has_high_post_entropy or expresses_uncertainty):
            epistemic_failure = True
            # Training target MUST be the authoritative knowledge from the document, not the failed output
            target_output = clean_doc_tags if clean_doc_tags else (doc_text or morpheme_output)
            record = {
                "instruction": instruction,
                "input": augmented_input,
                "output": target_output,
                "model_failed_output": morpheme_output,
                "decompiled_output": decompiled_text,
                "rag_document": doc_text,
                "retrieval_collection": source_coll,
                "similarity_score": round(match_score, 4),
                "similarity_threshold": self.similarity_threshold,
                "entropy_pre": round(entropy_pre, 4),
                "entropy_post": round(entropy_post, 4),
                "tau": self.tau,
                "reason": "epistemic_gap_unresolved_high_similarity",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.record_to_future_train(record)
            future_train_recorded = True
            
        return {
            "query": query,
            "instruction": instruction,
            "entropy_pre": entropy_pre,
            "needs_retrieval": needs_retrieval,
            "retrieval_triggered": retrieval_triggered,
            "retrieved_document": doc_text,
            "source_collection": source_coll,
            "match_score": match_score,
            "is_high_similarity": is_high_similarity,
            "router_experts": expert_names,
            "entropy_post": entropy_post,
            "morpheme_output": morpheme_output,
            "decompiled_text": decompiled_text,
            "epistemic_failure": epistemic_failure,
            "future_train_recorded": future_train_recorded,
            "future_train_path": self.future_train_path if future_train_recorded else None
        }
