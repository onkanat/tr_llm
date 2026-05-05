from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any

class VectorMemory:
    def __init__(self, collection_name: str = "evrensel_bellek", vector_size: int = 768, host: str = None, port: int = 6333):
        """
        Initializes Qdrant database connection for the Vector Rover prototype.
        If host is provided, connects to that instance (e.g., "localhost").
        Otherwise, uses an in-memory database.
        """
        self.collection_name = collection_name
        
        if host:
            self.client = QdrantClient(host=host, port=port)
        else:
            self.client = QdrantClient(":memory:")
            
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        self._next_point_id = 1

    def add_document(self, text: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Adds a document and its embedding to the memory."""
        if metadata is None:
            metadata = {}
        metadata["text"] = text
        
        point = PointStruct(
            id=self._next_point_id,
            vector=vector,
            payload=metadata
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        self._next_point_id += 1

    def recall(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Recalls the most semantically similar documents."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
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
