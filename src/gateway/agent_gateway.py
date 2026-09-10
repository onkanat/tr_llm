#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: AJAN KAPISI (AGENT GATEWAY)
======================================================
Bu modül; büyük agent modellerinin (Antigravity Agent, Gemini API, Ollama vb.)
küçük KristalLM modelimizle çift yönlü, yapılandırılmış diyalog kurmasını sağlar.

Temel Yetenekler:
1. `ask()`: Küçük modele soru sorma ve tüm telemetriyi (entropi, RAG skoru, morfemler) alma.
2. `inject_knowledge()`: Eksik/hatalı alanlar için kristal_bellek veya simulasyon_bellek'e otonom belge ekleme.
3. `check_memory()`: Eklenen belgenin hibrit arama ile bulunabilirliğini denetleme.
4. `get_epistemic_backlog()`: Modelin anlayamadığı (future_train_vector.jsonl) kayıtları listeleme.
5. `start_http_server()`: Dış agent sistemleri için sıfır bağımlılıklı standart HTTP/REST API sunucusu.
"""

import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.compiler.decompiler import MorphemeDecompiler
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector
from src.rag.epistemic_agent import EpistemicCuriosityAgent
from src.rag.merak import CuriosityEngine
from src.llm.router import TriModalRouter


class AgentGateway:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[KristalTokenizer] = None,
        decompiler: Optional[MorphemeDecompiler] = None,
        memory: Optional[VectorMemory] = None,
        general_memory: Optional[VectorMemory] = None,
        epistemic_agent: Optional[EpistemicCuriosityAgent] = None,
        future_train_path: str = "data/future_train_vector.jsonl",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.future_train_path = future_train_path
        
        # If components are provided directly
        self.model = model
        self.tokenizer = tokenizer
        self.decompiler = decompiler
        self.memory = memory
        self.general_memory = general_memory
        self.epistemic_agent = epistemic_agent
        
        # If epistemic_agent not provided, build from components if available
        if self.epistemic_agent is None and self.model is not None and self.tokenizer is not None and self.memory is not None:
            self.epistemic_agent = EpistemicCuriosityAgent(
                model=self.model,
                tokenizer=self.tokenizer,
                memory=self.memory,
                general_memory=self.general_memory,
                decompiler=self.decompiler,
                tau=2.5,
                similarity_threshold=0.85,
                future_train_path=self.future_train_path,
                device=device
            )

    @classmethod
    def create_default(
        cls,
        model_path: str = "data/kristal_model.pt",
        vocab_path: str = "data/vocab.json",
        lexicon_path: str = "data/lexicon/roots.tsv",
        storage_path: str = "data/qdrant_db",
        device: str = "cpu"
    ) -> "AgentGateway":
        """Factory method to load and build the complete default Gateway stack."""
        from scripts.train_step_demo import KristalLM
        from chat_prompt import resize_state_dict
        
        # Load Vocab
        vocab = Vocabulary()
        vocab.load(vocab_path)
        vocab_size = len(vocab.stoi)
        
        # Load Compiler & Decompiler
        lexicon = LexiconManager()
        lexicon.load_from_tsv(lexicon_path)
        compiler = CrystalCompiler(lexicon, build_default_graph())
        tokenizer = KristalTokenizer(compiler, vocab)
        decompiler = MorphemeDecompiler(compiler, vocab)
        
        # Load Vector Memories
        memory = VectorMemory(collection_name="kristal_bellek", vector_size=768, host="localhost", port=6333, storage_path=storage_path)
        general_memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333, storage_path=storage_path)
        
        # Load Model
        dev = torch.device(device)
        model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=dev)
            keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
            for k in keys_to_skip:
                del state_dict[k]
            state_dict = resize_state_dict(model, state_dict)
            model.load_state_dict(state_dict, strict=False)
        model.to(dev)
        model.eval()
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            decompiler=decompiler,
            memory=memory,
            general_memory=general_memory,
            future_train_path="data/future_train_vector.jsonl",
            device=device
        )

    def ask(
        self,
        query: str,
        instruction: str = "Belgeye göre cevapla.",
        mode: str = "RAG",
        force_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Asks a question to the small model through the Epistemic Curiosity Loop.
        Returns complete response and telemetry.
        """
        if self.epistemic_agent is None:
            raise RuntimeError("EpistemicCuriosityAgent is not initialized in AgentGateway.")
            
        use_rag = (mode.upper() == "RAG") or force_rag
        res = self.epistemic_agent.process_query(
            query=query,
            instruction=instruction,
            force_rag=use_rag
        )
        
        return {
            "query": query,
            "instruction": instruction,
            "mode": mode,
            "response_text": res.get("decompiled_text", ""),
            "morphemes": res.get("morpheme_output", ""),
            "entropy_pre": res.get("entropy_pre", 0.0),
            "entropy_post": res.get("entropy_post", 0.0),
            "needs_retrieval": res.get("needs_retrieval", False),
            "rag_document": res.get("retrieved_document", ""),
            "source_collection": res.get("source_collection", ""),
            "rag_score": res.get("match_score", 0.0),
            "is_high_similarity": res.get("is_high_similarity", False),
            "router_experts": res.get("router_experts", []),
            "epistemic_failure": res.get("epistemic_failure", False),
            "future_train_recorded": res.get("future_train_recorded", False),
            "future_train_path": res.get("future_train_path", None),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def inject_knowledge(
        self,
        text: str,
        target_collection: str = "kristal_bellek",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingests a new knowledge document into either kristal_bellek or simulasyon_bellek.
        Computes hybrid Kristal dense and sparse vectors and indexes into Qdrant.
        """
        target_mem = self.memory if target_collection == "kristal_bellek" else (self.general_memory or self.memory)
        
        token_ids = self.tokenizer.encode(text)
        tags = self.tokenizer.decode(token_ids)
        
        dense_vec = generate_kristal_vector(token_ids, tags)
        sparse_vec = generate_sparse_vector(token_ids, tags)
        
        meta = metadata.copy() if metadata else {}
        meta["crystal_tags"] = tags
        meta["token_ids"] = token_ids
        meta["injected_by"] = "agent_gateway"
        meta["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        target_mem.add_documents_batch(
            texts=[text],
            dense_vectors=[dense_vec],
            sparse_vectors=[sparse_vec],
            metadatas=[meta]
        )
        
        return {
            "status": "success",
            "collection": target_collection,
            "document_text": text,
            "crystal_tags": tags,
            "total_documents": target_mem.get_document_count()
        }

    def check_memory(
        self,
        query: str,
        target_collection: str = "kristal_bellek",
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Checks if a document or concept can be retrieved by the model via hybrid search.
        """
        target_mem = self.memory if target_collection == "kristal_bellek" else (self.general_memory or self.memory)
        
        token_ids = self.tokenizer.encode(query)
        tags = self.tokenizer.decode(token_ids)
        
        dense_vec = generate_kristal_vector(token_ids, tags)
        sparse_vec = generate_sparse_vector(token_ids, tags)
        
        results = target_mem.hybrid_recall(dense_vec, sparse_vec, top_k=top_k, query_tags=tags)
        return results

    def get_epistemic_backlog(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads recent unresolved epistemic gap records from future_train_vector.jsonl."""
        if not os.path.exists(self.future_train_path):
            return []
            
        records = []
        with open(self.future_train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        return records[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Returns the operational status of the gateway, memory counts, and epistemic backlog."""
        kristal_count = self.memory.get_document_count() if self.memory else 0
        simulasyon_count = self.general_memory.get_document_count() if self.general_memory else 0
        backlog_count = len(self.get_epistemic_backlog(limit=10000))
        
        return {
            "status": "online",
            "device": str(self.device),
            "kristal_bellek_docs": kristal_count,
            "simulasyon_bellek_docs": simulasyon_count,
            "epistemic_backlog_samples": backlog_count,
            "future_train_path": self.future_train_path
        }

    def close(self):
        """Closes connected vector memories cleanly."""
        if self.memory and hasattr(self.memory, "close"):
            self.memory.close()
        if self.general_memory and hasattr(self.general_memory, "close"):
            self.general_memory.close()

    def create_http_server(self, host: str = "127.0.0.1", port: int = 8080) -> HTTPServer:
        """Creates a zero-dependency lightweight HTTP REST server for external agents."""
        gateway = self
        
        class GatewayHTTPHandler(BaseHTTPRequestHandler):
            def _send_json(self, data: Any, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

            def do_GET(self):
                if self.path == "/api/status":
                    self._send_json(gateway.get_status())
                elif self.path == "/api/backlog":
                    self._send_json(gateway.get_epistemic_backlog())
                else:
                    self._send_json({"error": "Endpoint not found"}, status=404)

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except Exception as e:
                    self._send_json({"error": f"Invalid JSON: {e}"}, status=400)
                    return

                if self.path == "/api/query":
                    query_text = payload.get("query", "")
                    instruction = payload.get("instruction", "Belgeye göre cevapla.")
                    mode = payload.get("mode", "RAG")
                    res = gateway.ask(query_text, instruction=instruction, mode=mode)
                    self._send_json(res)
                elif self.path == "/api/inject":
                    text = payload.get("text", "")
                    collection = payload.get("collection", "kristal_bellek")
                    metadata = payload.get("metadata", {})
                    res = gateway.inject_knowledge(text, target_collection=collection, metadata=metadata)
                    self._send_json(res)
                elif self.path == "/api/check":
                    query_text = payload.get("query", "")
                    collection = payload.get("collection", "kristal_bellek")
                    top_k = payload.get("top_k", 3)
                    res = gateway.check_memory(query_text, target_collection=collection, top_k=top_k)
                    self._send_json({"results": res})
                else:
                    self._send_json({"error": "Endpoint not found"}, status=404)
                    
            def log_message(self, format, *args):
                pass # Silent logs to keep terminal clean

        return HTTPServer((host, port), GatewayHTTPHandler)
