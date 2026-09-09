from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional

from src.rag.embedding import generate_kristal_vector, generate_sparse_vector

class VectorMemory:
    def __init__(self, collection_name: str = "evrensel_bellek", vector_size: int = 768, host: str = None, port: int = 6333):
        """
        Initializes Qdrant database connection for the Vector Rover prototype.
        Gracefully falls back to in-memory mode (:memory:) if remote host connection fails.
        """
        self.collection_name = collection_name
        self.is_in_memory = False
        
        if host:
            try:
                self.client = QdrantClient(host=host, port=port, timeout=2.0)
                # Quick probe to verify server availability
                self.client.get_collections()
            except Exception as e:
                print(f"[VectorMemory] Uyarı: '{host}:{port}' Qdrant sunucusuna bağlanılamadı ({e}). Bellek içi (:memory:) moduna geçiliyor.")
                self.client = QdrantClient(":memory:")
                self.is_in_memory = True
        else:
            self.client = QdrantClient(":memory:")
            self.is_in_memory = True
            
        if not self.client.collection_exists(self.collection_name):
            self._create_hybrid_collection(vector_size)
            self._next_point_id = 1
        else:
            count_result = self.client.count(collection_name=self.collection_name)
            self._next_point_id = count_result.count + 1

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
            common_roots = {"su", "bir", "ve", "de", "da", "ki", "o", "bu", "şu", "ama", "ile", "en", "daha", "her", "şey", "için", "ol", "et", "yap"}
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
        for scored_point in search_result.points:
            score = scored_point.score
            text = scored_point.payload.get("text", "")
            doc_tags_str = scored_point.payload.get("crystal_tags", "")
            
            # 2. Token-Type Constraint / Penalty Scoring
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
                
                # If less than 50% of distinctive query roots are present, penalize score by 50%
                if match_ratio < 0.5:
                    score *= 0.5
            
            results.append({
                "score": score,
                "text": text,
                "metadata": scored_point.payload
            })
            
        # 3. Re-sort and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

