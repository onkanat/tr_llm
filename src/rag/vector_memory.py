from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any

class VectorMemory:
    def __init__(self, collection_name: str = "evrensel_bellek", vector_size: int = 768, host: str = None, port: int = 6333):
        """
        Initializes Qdrant database connection for the Vector Rover prototype.
        """
        self.collection_name = collection_name
        
        if host:
            self.client = QdrantClient(host=host, port=port)
        else:
            self.client = QdrantClient(":memory:")
            
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

    def hybrid_recall(self, dense_query: List[float], sparse_query: models.SparseVector, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recalls documents using Reciprocal Rank Fusion (RRF) over Dense and Sparse vectors."""
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
