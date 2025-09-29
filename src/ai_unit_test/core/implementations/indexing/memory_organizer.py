"""In-memory index organizer implementation."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from ai_unit_test.core.exceptions import IndexError
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult


class InMemoryIndexOrganizer(IndexOrganizer):
    """In-memory index organizer for testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the InMemoryIndexOrganizer.

        Args:
            config: Configuration dictionary for the index.
        """
        super().__init__(config)
        self.embeddings: np.ndarray | None = None
        self.metadata: list[dict[str, Any]] = []
        self.doc_ids: list[str] = []

    async def create_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        index_path: Path,
        model_name: str,
    ) -> IndexMetadata:
        """Create an in-memory index."""
        self.embeddings = embeddings
        self.metadata = metadata
        self.doc_ids = [str(i) for i in range(len(metadata))]
        self.index_info = IndexMetadata(
            embedding_model=model_name,
            total_documents=len(metadata),
            embedding_dimension=embeddings.shape[1],
            schema_version="1.0.0",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            backend_type="memory",
            backend_config={},
        )
        self._index_loaded = True
        return self.index_info

    async def load_index(self, index_path: Path) -> IndexMetadata:
        """Load an in-memory index (no-op)."""
        if not self._index_loaded:
            raise IndexError("Index not created")
        return self.index_info

    async def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7) -> list[SearchResult]:
        """Search the in-memory index."""
        if self.embeddings is None:
            raise IndexError("Index not created")

        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Cosine similarity
        scores = np.dot(embeddings, query_embedding.T).flatten()
        top_k_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for i in top_k_indices:
            if scores[i] >= threshold:
                results.append(
                    SearchResult(
                        metadata=self.metadata[i],
                        score=float(scores[i]),
                        document_id=self.doc_ids[i],
                    )
                )
        return results

    async def add_documents(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add documents to the in-memory index."""
        if self.embeddings is None:
            raise IndexError("Index not created")

        self.embeddings = np.vstack([self.embeddings, embeddings])
        print(f"Before extend: {len(self.metadata)}")
        self.metadata.extend(metadata)
        print(f"After extend: {len(self.metadata)}")
        new_doc_ids = [str(i) for i in range(len(self.doc_ids), len(self.metadata))]
        self.doc_ids.extend(new_doc_ids)
        self.index_info.total_documents = len(self.metadata)

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from the in-memory index."""
        if self.embeddings is None:
            raise IndexError("Index not created")

        indices_to_remove = [int(doc_id) for doc_id in document_ids]
        self.embeddings = np.delete(self.embeddings, indices_to_remove, axis=0)
        self.metadata = [m for i, m in enumerate(self.metadata) if i not in indices_to_remove]
        self.doc_ids = [d for d in self.doc_ids if d not in document_ids]
        self.index_info.total_documents = len(self.metadata)

    async def update_document(self, document_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        """Update a document in the in-memory index."""
        if self.embeddings is None:
            raise IndexError("Index not created")

        index = int(document_id)
        self.embeddings[index] = embedding
        self.metadata[index] = metadata

    async def get_index_info(self) -> IndexMetadata:
        """Get the in-memory index metadata."""
        if not self._index_loaded:
            raise IndexError("Index not loaded")
        return self.index_info

    async def validate_index(self, index_path: Path) -> bool:
        """Validate the in-memory index."""
        return self._index_loaded

    async def get_stats(self) -> IndexStats:
        """Get the in-memory index stats."""
        if not self._index_loaded:
            raise IndexError("Index not loaded")

        return IndexStats(
            total_documents=self.index_info.total_documents,
            search_latency_ms=0.1,
            memory_usage_mb=0.1,
            average_score_distribution={},
        )

    async def optimize_index(self) -> None:
        """Optimize the in-memory index (no-op)."""
        pass
