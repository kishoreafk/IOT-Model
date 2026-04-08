import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
import threading
import pickle
import os


class FaissManager:
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "FlatIP",
        persist_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.persist_path = persist_path

        self.index: Optional[faiss.Index] = None
        self.device_id_map: Dict[int, str] = {}
        self.embedding_metadata: List[Dict[str, Any]] = []
        self.cluster_labels: Optional[np.ndarray] = None
        # Store raw embeddings for retrieval since IndexIDMap doesn't support reconstruct
        self._raw_embeddings: List[np.ndarray] = []

        self._lock = threading.Lock()
        self._init_index()

        self.is_initialized = True
        self.total = 0

    def _init_index(self):
        if self.index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.index = faiss.IndexIDMap(self.index)

    def add(self, embeddings: np.ndarray, device_id: str) -> Tuple[int, int]:
        """
        Add embeddings to index and return cluster_id and total count.
        """
        if embeddings.shape[1] != self.embedding_dim:
            embeddings = embeddings.reshape(-1, self.embedding_dim)

        embeddings = self._normalize(embeddings)

        with self._lock:
            start_id = len(self.embedding_metadata)
            ids = list(range(start_id, start_id + len(embeddings)))

            self.index.add_with_ids(embeddings, np.array(ids))
            
            # Store raw embeddings for retrieval
            for i in range(len(embeddings)):
                self._raw_embeddings.append(embeddings[i].copy())

            metadata = {
                "device_id": device_id,
                "embedding_id": start_id,
            }
            self.embedding_metadata.append(metadata)

            self.device_id_map[start_id] = device_id
            self.total = self.index.ntotal

            cluster_id = 0
            if self.cluster_labels is not None and len(self.cluster_labels) > 0:
                cluster_id = int(self.cluster_labels[-1])

        return cluster_id, self.total

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        device_ids: Optional[List[str]] = None,
    ) -> List[int]:
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} != expected {self.embedding_dim}")

        embeddings = self._normalize(embeddings)

        with self._lock:
            start_id = len(self.embedding_metadata)
            ids = list(range(start_id, start_id + len(embeddings)))

            self.index.add_with_ids(embeddings, np.array(ids))
            
            # Store raw embeddings for retrieval
            for i in range(len(embeddings)):
                self._raw_embeddings.append(embeddings[i].copy())

            if metadata is None:
                metadata = [{}] * len(embeddings)
            self.embedding_metadata.extend(metadata)

            if device_ids is None:
                device_ids = ["unknown"] * len(embeddings)
            for i, dev_id in enumerate(device_ids):
                self.device_id_map[start_id + i] = dev_id

            self.total = self.index.ntotal

        return ids

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = self._normalize(query_embedding.reshape(1, -1))
        distances, ids = self.index.search(query, k)
        return distances, ids

    def get_cluster_embeddings(self, cluster_id: int) -> List[np.ndarray]:
        """Get embeddings for a specific cluster."""
        with self._lock:
            if self.total == 0 or len(self._raw_embeddings) == 0:
                return []
            
            # Return stored embeddings (IndexIDMap doesn't support reconstruct)
            return [emb for emb in self._raw_embeddings]

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of all clusters."""
        with self._lock:
            return {
                "total_clusters": 1,
                "total_embeddings": self.total,
                "clusters": {
                    "0": {
                        "size": self.total,
                        "entries": self.embedding_metadata[:10] if len(self.embedding_metadata) > 0 else [],
                    }
                },
            }

    def get_all_clusters(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            if self.total == 0:
                return {}
            return {
                0: {
                    "size": self.total,
                    "entries": self.embedding_metadata[:10],
                }
            }

    def get_total_embeddings(self) -> int:
        with self._lock:
            return self.index.ntotal if self.index else 0

    def reset(self):
        with self._lock:
            self._init_index()
            self.device_id_map = {}
            self.embedding_metadata = []
            self.cluster_labels = None
            self._raw_embeddings = []
            self.total = 0

    def save(self):
        if self.persist_path and self.index:
            with self._lock:
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                faiss.write_index(self.index, self.persist_path + ".index")

                with open(self.persist_path + ".meta", "wb") as f:
                    pickle.dump({
                        "metadata": self.embedding_metadata,
                        "device_id_map": self.device_id_map,
                        "cluster_labels": self.cluster_labels,
                    }, f)

    def load(self):
        if self.persist_path and os.path.exists(self.persist_path + ".index"):
            with self._lock:
                self.index = faiss.read_index(self.persist_path + ".index")

                if os.path.exists(self.persist_path + ".meta"):
                    with open(self.persist_path + ".meta", "rb") as f:
                        data = pickle.load(f)
                        self.embedding_metadata = data.get("metadata", [])
                        self.device_id_map = data.get("device_id_map", {})
                        self.cluster_labels = data.get("cluster_labels")
                        self.total = self.index.ntotal