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
from src.rag.hybrid_search import (
    RRF_SCORE_MAX,
    filter_and_rank_candidates,
    filter_thought_steps,
    extract_distinctive_roots,
    normalize_rrf_score,
    reciprocal_rank_fusion,
)

class VectorMemory:
    _shared_clients: Dict[str, Any] = {}

    def __init__(
        self,
        collection_name: str = "anka_bellek",
        vector_size: int = 768,
        host: Optional[str] = None,
        port: int = 6333,
        storage_path: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        allow_fallback: bool = False,
        location: Optional[str] = None,
    ):
        """
        Initializes Qdrant database connection for the Vector Rover prototype.
        Priority:
        1. Passed client instance (if provided)
        2. Explicit location (e.g. location=":memory:", remote url, or custom path)
        3. Shared/cached client if already connected in this process
        4. Remote host:port if provided — T-0148 4A fail-closed: bağlantı başarısızsa
           RuntimeError (allow_fallback=False ise sessiz-local-düşme KAPALI, fail-closed)
        5. storage_path verilmişse explicit-local-beyan (host'suz çağrı)
        6. Ne host ne storage_path ne location verilmişse in-memory (:memory:) — default
           data/qdrant_db KALDIRILDI (T-0148 4B; repo-içi sessiz-yazım riski)
        """
        self.collection_name = collection_name
        self.is_in_memory = False
        self.storage_type = "remote"
        
        if client is not None:
            self.client = client
            self.storage_type = "shared_client"
        elif location is not None:
            if location == ":memory:":
                self.client = QdrantClient(":memory:")
                self.is_in_memory = True
                self.storage_type = "in-memory (:memory:)"
            else:
                try:
                    self.client = QdrantClient(location=location)
                    self.client.get_collections()
                    self.storage_type = f"location ({location})"
                except Exception as e:
                    if allow_fallback:
                        self.client = QdrantClient(":memory:")
                        self.is_in_memory = True
                        self.storage_type = "in-memory (:memory:)"
                    else:
                        raise RuntimeError(
                            f"[VEKTOR_BELLEK_HATA] remote location ({location}) bağlantısı "
                            "kurulamadı; sessiz-local-fallback kapalıdır (T-0148 4A fail-closed)."
                        ) from e
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
                    # T-0148 4A fail-closed: host VERİLMİŞSE remote-bağlantı
                    # başarısızlığı sessizce local'e DÜŞMEZ — fail-closed.
                    try:
                        self.client = QdrantClient(host=host, port=port, timeout=2.0, check_compatibility=False)
                        self.client.get_collections()
                        connected = True
                        self.storage_type = f"remote ({host}:{port})"
                    except Exception as e:
                        if allow_fallback:
                            self.client = QdrantClient(":memory:")
                            self.is_in_memory = True
                            self.storage_type = "in-memory (:memory:)"
                            connected = True
                        else:
                            raise RuntimeError(
                                f"[VEKTOR_BELLEK_HATA] remote ({host}:{port}) bağlantısı "
                                "kurulamadı; sessiz-local-fallback kapalıdır (T-0148 4A fail-closed). "
                                "Local-storage isteniyorsa host=None ile açık-beyan edin."
                            ) from e

                if not connected and storage_path:
                    # host'suz + storage_path'li çağrı = AÇIK local-beyan
                    # (sessiz-fallback değil; cache-guard 4A ile korunur)
                    if storage_path == ":memory:":
                        self.client = QdrantClient(":memory:")
                        self.is_in_memory = True
                        self.storage_type = "in-memory (:memory:)"
                        connected = True
                    else:
                        try:
                            import os
                            os.makedirs(storage_path, exist_ok=True)
                            self.client = QdrantClient(path=storage_path)
                            connected = True
                            self.storage_type = f"local ({storage_path})"
                        except Exception as e:
                            if allow_fallback:
                                self.client = QdrantClient(":memory:")
                                self.is_in_memory = True
                                self.storage_type = "in-memory (:memory:)"
                                connected = True
                            else:
                                raise RuntimeError(
                                    f"[VEKTOR_BELLEK_HATA] local storage ({storage_path}) "
                                    "açılamadı; sessiz-fallback kapalıdır."
                                ) from e

                if not connected:
                    self.client = QdrantClient(":memory:")
                    self.is_in_memory = True
                    self.storage_type = "in-memory (:memory:)"

                # T-0148 4A cache-guard: yalnız REMOTE bağlantı cache'e yazılır
                # (local/:memory: fallback cache-kirliliği kapanır — P4 B6 çelişkisi)
                if not self.is_in_memory and self.storage_type.startswith("remote"):
                    VectorMemory._shared_clients[client_key] = (self.client, self.storage_type)
            
        try:
            if not self.client.collection_exists(self.collection_name):
                self._create_hybrid_collection(vector_size)
                self._next_point_id = 1
            else:
                coll_info = self.client.get_collection(self.collection_name)
                existing_size = None
                try:
                    existing_size = coll_info.config.params.vectors["dense"].size
                except Exception:
                    pass

                if existing_size is not None and existing_size != vector_size:
                    # T-0148 4E fail-closed: if non-empty, recreate_collection will raise
                    # RuntimeError; do NOT swallow it!
                    self.recreate_collection(vector_size, confirm_destroy=False)
                else:
                    count_result = self.client.count(collection_name=self.collection_name)
                    self._next_point_id = count_result.count + 1
        except Exception:
            if hasattr(self, "client") and self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
            raise

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

    def recreate_collection(self, vector_size: int = 768, confirm_destroy: bool = False):
        """Clears the collection by deleting and recreating it.
        T-0148 4E fail-closed: DOLU koleksiyonun recreate'i confirm_destroy
        onayı ister (sessiz-yıkım YOK); boş koleksiyon recreate'i izinli
        (probe-temizlik yolu)."""
        if self.client.collection_exists(self.collection_name):
            count = self.client.count(collection_name=self.collection_name).count
            if count > 0 and not confirm_destroy:
                raise RuntimeError(
                    f"[VEKTOR_BELLEK_HATA] '{self.collection_name}' DOLU ({count} nokta); "
                    "recreate için confirm_destroy=True gerekir (T-0148 4E fail-closed)."
                )
            self.client.delete_collection(self.collection_name)

        self._create_hybrid_collection(vector_size)
        self._next_point_id = 1
        if confirm_destroy:
            print(f"[UYARI] '{self.collection_name}' confirm_destroy=True ile yıkılıp "
                  f"{vector_size}-boyutlu hibrit koleksiyon olarak yeniden kuruldu "
                  f"(yıkım GÖRÜNÜR — sessiz-silme kapalı, T-0148 4E).")

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """T-0148 4D: client-düzeyi delete_collection'ın VectorMemory-wraper'ı
        (P5 temizlik-yüzeyi bulgusu). Default: kendi collection_name'i."""
        hedef = collection_name or self.collection_name
        try:
            if not self.client.collection_exists(hedef):
                return False
            self.client.delete_collection(hedef)
            return not self.client.collection_exists(hedef)
        except Exception:
            return False

    def add_document(self, text: str, dense_vector: List[float], sparse_vector: models.SparseVector, metadata: Optional[Dict[str, Any]] = None):
        """Adds a single document with its dense and sparse vectors."""
        self.add_documents_batch([text], [dense_vector], [sparse_vector], [metadata or {}])

    def add_documents_batch(self, texts: List[str], dense_vectors: List[List[float]], sparse_vectors: List[models.SparseVector], metadatas: List[Dict[str, Any]], upsert: bool = False):
        """Adds multiple documents in a single batch with both dense and sparse vectors.
        T-0148 4C: upsert=True point-id deterministiktir (payload-text sha256
        ön-ekinin 2^63-mod türevi) — aynı belge yeniden-yazımı DUBLOR YARATMAZ
        (restart-sayaç-çakışması kapalıdır). upsert=False (default) mevcut
        monoton-sayaç davranışını bit-uyumlu korur (çağıran-kırılması yok)."""
        points = []
        for i in range(len(texts)):
            meta = metadatas[i] if i < len(metadatas) else {}
            meta["text"] = texts[i]

            if upsert:
                import hashlib
                _t = hashlib.sha256(texts[i].encode("utf-8")).hexdigest()
                point_id = int(_t[:16], 16) % (2 ** 63)
            else:
                point_id = self._next_point_id
                self._next_point_id += 1

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vectors[i],
                    "sparse": sparse_vectors[i]
                },
                payload=meta
            ))

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
        
        return filter_and_rank_candidates(
            scored_points=search_result.points,
            query_tags=query_tags,
            top_k=top_k,
            rrf_score_max=RRF_SCORE_MAX,
        )

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
            self.client = None

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


