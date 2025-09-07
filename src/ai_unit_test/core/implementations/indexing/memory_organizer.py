"""In-memory index organizer implementation for testing."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ai_unit_test.core.exceptions import IndexError
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult

logger = logging.getLogger(__name__)


class MemoryIndexOrganizer(IndexOrganizer):
    """Simple in-memory index organizer for testing and development."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.embeddings = None
        self.metadata = None
        self.index_info = None
        self.use_cosine_similarity = config.get("use_cosine_similarity", True)

    async def create_index(
        self, embeddings: np.ndarray, metadata: list[dict[str, Any]], index_path: Path, model_name: str
    ) -> IndexMetadata:
        """Create in-memory index."""
        try:
            # Validate inputs
            self._validate_inputs(embeddings, metadata)

            # Store data in memory
            self.embeddings = embeddings.copy()
            self.metadata = metadata.copy()

            # Normalize embeddings for cosine similarity if requested
            if self.use_cosine_similarity:
                self.embeddings = self._normalize_embeddings(self.embeddings)

            # Prepare metadata
            self.index_info = IndexMetadata(
                embedding_model=model_name,
                schema_version="1.1.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                total_documents=len(metadata),
                embedding_dimension=embeddings.shape[1],
                backend_type="memory",
                backend_config={"use_cosine_similarity": self.use_cosine_similarity},
            )

            self._index_loaded = True
            self._index_path = index_path

            logger.info(f"Memory index created with {len(metadata)} documents")
            return self.index_info

        except Exception as e:
            logger.error(f"Failed to create memory index: {e}")
            raise IndexError(f"Index creation failed: {e}")

    async def load_index(self, index_path: Path) -> IndexMetadata:
        """Load index (no-op for memory organizer)."""
        # Memory organizer doesn't persist data
        # This method exists for interface compatibility
        if not self._index_loaded:
            raise IndexError("No index in memory to load")

        logger.info("Memory index is already loaded")
        return self.index_info

    async def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7) -> list[SearchResult]:
        """Search in-memory index using similarity calculation."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Validate dimensions
            if query_embedding.shape[1] != self.embeddings.shape[1]:
                raise IndexError(
                    f"Query embedding dimension ({query_embedding.shape[1]}) "
                    f"doesn't match index dimension ({self.embeddings.shape[1]})"
                )

            # Normalize query if using cosine similarity
            if self.use_cosine_similarity:
                query_embedding = self._normalize_embeddings(query_embedding)

            # Calculate similarities
            if self.use_cosine_similarity:
                # Cosine similarity (dot product of normalized vectors)
                similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            else:
                # Euclidean distance converted to similarity
                distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
                # Convert distance to similarity (higher is better)
                max_distance = np.max(distances) if len(distances) > 0 else 1.0
                similarities = 1.0 - (distances / max_distance)

            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]

            # Filter by threshold and create results
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= threshold:
                    results.append(SearchResult(metadata=self.metadata[idx], score=float(score), document_id=str(idx)))

            return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise IndexError(f"Search failed: {e}")

    async def add_documents(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add documents to in-memory index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            # Normalize new embeddings if using cosine similarity
            if self.use_cosine_similarity:
                embeddings = self._normalize_embeddings(embeddings)

            # Combine with existing data
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata.extend(metadata)

            # Update info
            self.index_info.total_documents = len(self.metadata)
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            logger.info(f"Added {len(metadata)} documents to memory index")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise IndexError(f"Document addition failed: {e}")

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from in-memory index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            # Convert document IDs to indices
            indices_to_remove = [int(doc_id) for doc_id in document_ids]

            # Create mask for documents to keep
            mask = np.ones(len(self.metadata), dtype=bool)
            mask[indices_to_remove] = False

            # Filter embeddings and metadata
            self.embeddings = self.embeddings[mask]
            self.metadata = [self.metadata[i] for i in range(len(self.metadata)) if mask[i]]

            # Update info
            self.index_info.total_documents = len(self.metadata)
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            logger.info(f"Removed {len(document_ids)} documents from memory index")

        except Exception as e:
            logger.error(f"Failed to remove documents: {e}")
            raise IndexError(f"Document removal failed: {e}")

    async def update_document(self, document_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        """Update document in in-memory index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            idx = int(document_id)

            # Normalize embedding if using cosine similarity
            if self.use_cosine_similarity:
                embedding = self._normalize_embeddings(embedding.reshape(1, -1))[0]

            # Update embedding and metadata
            self.embeddings[idx] = embedding
            self.metadata[idx] = metadata

            # Update info
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            logger.info(f"Updated document {document_id} in memory index")

        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise IndexError(f"Document update failed: {e}")

    async def get_index_info(self) -> IndexMetadata:
        """Get index metadata."""
        if not self.index_info:
            raise IndexError("No index loaded")
        return self.index_info

    async def validate_index(self, index_path: Path) -> bool:
        """Validate in-memory index (always returns current state)."""
        return self._index_loaded and self.embeddings is not None

    async def get_stats(self) -> IndexStats:
        """Get index statistics."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        # Perform sample search to estimate performance
        start_time = time.time()
        if len(self.embeddings) > 0:
            random_query = np.random.random((1, self.embeddings.shape[1]))
            await self.search(random_query, k=5)
        search_latency = (time.time() - start_time) * 1000

        return IndexStats(
            total_documents=len(self.embeddings),
            average_score_distribution={"high": 0.3, "medium": 0.5, "low": 0.2},
            search_latency_ms=search_latency,
            memory_usage_mb=self._estimate_memory_usage(),
        )

    async def optimize_index(self) -> None:
        """Optimize in-memory index (no-op)."""
        logger.info("Memory index doesn't need optimization")

    def clear_index(self) -> None:
        """Clear the in-memory index."""
        self.embeddings = None
        self.metadata = None
        self.index_info = None
        self._index_loaded = False
        logger.info("Memory index cleared")

    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings (useful for testing)."""
        if not self._index_loaded:
            raise IndexError("No index loaded")
        return self.embeddings.copy()

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Get all metadata (useful for testing)."""
        if not self._index_loaded:
            raise IndexError("No index loaded")
        return self.metadata.copy()

    def _validate_inputs(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Validate input data."""
        if len(embeddings) == 0:
            raise ValueError("Embeddings array is empty")

        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != metadata count ({len(metadata)})")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array")

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return embeddings / norms

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if self.embeddings is None:
            return 0.0

        # Calculate actual memory usage
        embeddings_size = self.embeddings.nbytes
        metadata_size = len(str(self.metadata).encode("utf-8"))

        return (embeddings_size + metadata_size) / (1024 * 1024)
