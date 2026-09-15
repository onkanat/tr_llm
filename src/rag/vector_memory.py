import atexit
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional

try:
    import portalocker  # Pre-import to ensure available in sys.modules during shutdown
except ImportError:
    portalocker = None

from src.rag.embedding import generate_kristal_vector, generate_sparse_vector

class VectorMemory:
    _shared_clients: Dict[str, Any] = {}

    def __init__(self, collection_name: str = "kristal_bellek", vector_size: int = 768, host: str = None, port: int = 6333, storage_path: str = "data/qdrant_db", client: Optional[QdrantClient] = None):
        """
        Initializes Qdrant database connection for the Vector Rover prototype.
        Priority:
        1. Passed client instance (if provided)
        2. Shared/cached client if already connected in this process
        3. Remote host:port if provided and accessible
        4. Persistent local embedded storage (storage_path)
        5. Graceful fallback to in-memory mode (:memory:)
        """
        self.collection_name = collection_name
        self.is_in_memory = False
        self.storage_type = "remote"
        
        if client is not None:
            self.client = client
            self.storage_type = "shared_client"
        else:
            client_key = f"remote:{host}:{port}" if host else f"local:{storage_path}"
            if client_key in VectorMemory._shared_clients:
                cached_client, cached_type = VectorMemory._shared_clients[client_key]
                try:
                    cached_client.get_collections()
                    self.client = cached_client
                    self.storage_type = cached_type
                except Exception:
                    # Client was closed or stale; evict and reconnect
                    VectorMemory._shared_clients.pop(client_key, None)
            
            if not hasattr(self, "client") or self.client is None:
                connected = False
                if host:
                    try:
                        self.client = QdrantClient(host=host, port=port, timeout=2.0, check_compatibility=False)
                        self.client.get_collections()
                        connected = True
                        self.storage_type = f"remote ({host}:{port})"
                    except Exception:
                        pass
                        
                if not connected and storage_path:
                    try:
                        import os
                        os.makedirs(storage_path, exist_ok=True)
                        self.client = QdrantClient(path=storage_path)
                        connected = True
                        self.storage_type = f"local ({storage_path})"
                    except Exception:
                        pass
                        
                if not connected:
                    self.client = QdrantClient(":memory:")
                    self.is_in_memory = True
                    self.storage_type = "in-memory (:memory:)"
                    
                if not self.is_in_memory:
                    VectorMemory._shared_clients[client_key] = (self.client, self.storage_type)
            
        if not self.client.collection_exists(self.collection_name):
            self._create_hybrid_collection(vector_size)
            self._next_point_id = 1
        else:
            try:
                coll_info = self.client.get_collection(self.collection_name)
                existing_size = coll_info.config.params.vectors["dense"].size
                if existing_size != vector_size:
                    self.recreate_collection(vector_size)
                else:
                    count_result = self.client.count(collection_name=self.collection_name)
                    self._next_point_id = count_result.count + 1
            except Exception:
                count_result = self.client.count(collection_name=self.collection_name)
                self._next_point_id = count_result.count + 1

    def get_document_count(self) -> int:
        """Returns total number of documents in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(collection_name=self.collection_name).count

    def list_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Scrolls and returns documents stored in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return []
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return [{"id": r.id, "payload": r.payload} for r in records]


    def _create_hybrid_collection(self, vector_size: int):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF, # Automatically applies TF-IDF weighting
                )
            }
        )

    def recreate_collection(self, vector_size: int = 768):
        """Clears the collection by deleting and recreating it."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        
        self._create_hybrid_collection(vector_size)
        self._next_point_id = 1
        print(f"Collection '{self.collection_name}' has been reset for Hybrid Search.")

    def add_document(self, text: str, dense_vector: List[float], sparse_vector: models.SparseVector, metadata: Optional[Dict[str, Any]] = None):
        """Adds a single document with its dense and sparse vectors."""
        self.add_documents_batch([text], [dense_vector], [sparse_vector], [metadata or {}])

    def add_documents_batch(self, texts: List[str], dense_vectors: List[List[float]], sparse_vectors: List[models.SparseVector], metadatas: List[Dict[str, Any]]):
        """Adds multiple documents in a single batch with both dense and sparse vectors."""
        points = []
        for i in range(len(texts)):
            meta = metadatas[i] if i < len(metadatas) else {}
            meta["text"] = texts[i]
            
            points.append(models.PointStruct(
                id=self._next_point_id,
                vector={
                    "dense": dense_vectors[i],
                    "sparse": sparse_vectors[i]
                },
                payload=meta
            ))
            self._next_point_id += 1
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def dense_recall(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Recalls documents using only Dense vectors (pure semantic cosine similarity)."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=top_k
        )
        
        results = []
        for scored_point in search_result.points:
            results.append({
                "score": scored_point.score,
                "text": scored_point.payload.get("text", ""),
                "metadata": scored_point.payload
            })
        return results

    def hybrid_recall(self, dense_query: List[float], sparse_query: models.SparseVector, top_k: int = 3, query_tags: str = None) -> List[Dict[str, Any]]:
        """Recalls documents using Reciprocal Rank Fusion (RRF) over Dense and Sparse vectors."""
        # We increase the candidate pool limit from Qdrant to apply post-retrieval token constraints
        candidate_limit = max(50, top_k * 4)
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=max(20, top_k * 2)
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=max(20, top_k * 2)
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_limit
        )
        
        # 1. Extract distinctive query roots if query_tags is provided
        distinctive_query_roots = set()
        if query_tags:
            common_roots = {
                "su", "bir", "ve", "de", "da", "ki", "o", "bu", "şu", "ama", "ile", 
                "en", "daha", "her", "şey", "için", "ol", "et", "yap",
                "ne", "kim", "nasıl", "neden", "niçin", "hangi", "nere", "kaç", "mı", "mi", "mu", "mü"
            }
            inflection_prefixes = ("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_")
            special_tokens = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>", "<NUMBER>", "<SYMBOL>"]
            
            for m in query_tags.split():
                if m in special_tokens or m.startswith("DERIV_"):
                    continue
                if m.startswith(inflection_prefixes) or m in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"):
                    continue
                root = m.lower()
                if root not in common_roots:
                    distinctive_query_roots.add(root)

        results = []
        cot_filter_patterns = [
            r'\bthe user\b', r'\bwe need to\b', r'\bthey want\b', r'\blet\'s produce\b',
            r'\bhere is\b', r'\bso that should\b', r'\bmake it\b', r'\bfirst, let\b',
            r'\bin this case\b', r'\bconcise, pedagogically\b', r'<think>', r'<thought>'
        ]
        cot_regex = re.compile('|'.join(cot_filter_patterns), re.IGNORECASE)

        for scored_point in search_result.points:
            score = scored_point.score
            text = scored_point.payload.get("text", "")
            doc_tags_str = scored_point.payload.get("crystal_tags", "")
            
            # Purity Filter: reject English CoT leaks
            if cot_regex.search(text):
                continue
            
            # 2. Token-Type Constraint / Penalty Scoring
            has_root_match = True
            matching_roots = set()
            if query_tags and len(distinctive_query_roots) > 0 and doc_tags_str:
                doc_morphemes = doc_tags_str.split()
                doc_roots = set()
                inflection_prefixes = ("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_")
                special_tokens = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>", "<NUMBER>", "<SYMBOL>"]
                
                for dm in doc_morphemes:
                    if dm in special_tokens or dm.startswith("DERIV_"):
                        continue
                    if dm.startswith(inflection_prefixes) or dm in ("PLURAL", "NEG", "POTENTIAL", "IMPOTENTIAL_NEG"):
                        continue
                    doc_roots.add(dm.lower())
                
                matching_roots = distinctive_query_roots.intersection(doc_roots)
                match_ratio = len(matching_roots) / len(distinctive_query_roots)
                
                if match_ratio == 0:
                    # No distinctive query root exists in the document: severe penalty
                    score *= 0.05
                    has_root_match = False
                elif match_ratio < 0.5:
                    # Partial match
                    score *= (0.3 + 0.7 * match_ratio)
                    has_root_match = True
                else:
                    has_root_match = True
            
            results.append({
                "score": score,
                "text": text,
                "metadata": scored_point.payload,
                "has_root_match": has_root_match,
                "matching_roots": list(matching_roots)
            })
            
        # 3. Re-sort and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def close(self):
        """Cleanly closes this VectorMemory's client connection."""
        for client_key, (c, _) in list(VectorMemory._shared_clients.items()):
            if c is self.client:
                VectorMemory._shared_clients.pop(client_key, None)
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    @classmethod
    def close_all(cls):
        """Cleanly closes all cached shared QdrantClient connections."""
        for client_key, (client, _) in list(cls._shared_clients.items()):
            try:
                if hasattr(client, "close"):
                    client.close()
            except Exception:
                pass
        cls._shared_clients.clear()


@atexit.register
def _cleanup_vector_memory():
    """Ensures Qdrant clients and file locks are closed before interpreter shutdown."""
    VectorMemory.close_all()


