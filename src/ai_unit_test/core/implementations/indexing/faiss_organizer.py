"""FAISS index organizer implementation."""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ai_unit_test.core.exceptions import ConfigurationError, IndexError, IndexNotFoundError
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


class FaissIndexOrganizer(IndexOrganizer):
    """FAISS-based index organizer implementation."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        if not FAISS_AVAILABLE:
            raise ConfigurationError("FAISS is not available. Please install faiss-cpu or faiss-gpu")

        self.index = None
        self.metadata = None
        self.index_info = None
        self.index_type = config.get("index_type", "IndexFlatIP")
        self.normalize_embeddings = config.get("normalize_embeddings", True)

    async def create_index(
        self, embeddings: np.ndarray, metadata: list[dict[str, Any]], index_path: Path, model_name: str
    ) -> IndexMetadata:
        """Create and save FAISS index."""
        try:
            # Validate inputs
            self._validate_inputs(embeddings, metadata)

            # Normalize embeddings if required
            if self.normalize_embeddings:
                faiss.normalize_L2(embeddings)

            # Create index
            dimension = embeddings.shape[1]
            self.index = self._create_faiss_index(dimension)

            # Add embeddings
            self.index.add(embeddings)

            # Prepare metadata
            self.metadata = metadata
            self.index_info = IndexMetadata(
                embedding_model=model_name,
                schema_version="1.1.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                total_documents=len(metadata),
                embedding_dimension=dimension,
                backend_type="faiss",
                backend_config={"index_type": self.index_type, "normalize_embeddings": self.normalize_embeddings},
            )

            # Save to disk
            await self._save_index(index_path)

            self._index_loaded = True
            self._index_path = index_path

            logger.info(f"FAISS index created with {len(metadata)} documents")
            return self.index_info

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise IndexError(f"Index creation failed: {e}")

    async def load_index(self, index_path: Path) -> IndexMetadata:
        """Load existing FAISS index."""
        try:
            index_file = index_path / "index.faiss"
            metadata_file = index_path / "index_meta.json"
            manifest_file = index_path / "index_manifest.json"

            # Check if files exist
            if not all(f.exists() for f in [index_file, metadata_file, manifest_file]):
                raise IndexNotFoundError(f"Index files not found in {index_path}")

            # Load index
            self.index = faiss.read_index(str(index_file))

            # Load metadata
            with open(metadata_file, encoding="utf-8") as f:
                self.metadata = json.load(f)

            # Load manifest
            with open(manifest_file, encoding="utf-8") as f:
                manifest_data = json.load(f)

            self.index_info = IndexMetadata(
                embedding_model=manifest_data.get("embedding_model", "unknown"),
                schema_version=manifest_data.get("schema_version", "1.0.0"),
                created_at=manifest_data.get("created_at", "unknown"),
                updated_at=manifest_data.get("updated_at", "unknown"),
                total_documents=len(self.metadata),
                embedding_dimension=self.index.d,
                backend_type="faiss",
                backend_config=manifest_data.get("backend_config", {}),
            )

            self._index_loaded = True
            self._index_path = index_path

            logger.info(f"FAISS index loaded: {len(self.metadata)} documents")
            return self.index_info

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise IndexError(f"Index loading failed: {e}")

    async def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7) -> list[SearchResult]:
        """Search FAISS index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            # Ensure query embedding is 2D and normalized if required
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            if self.normalize_embeddings:
                faiss.normalize_L2(query_embedding)

            # Validate dimensions
            if query_embedding.shape[1] != self.index.d:
                raise IndexError(
                    f"Query embedding dimension ({query_embedding.shape[1]}) "
                    f"doesn't match index dimension ({self.index.d})"
                )

            # Search
            scores, indices = self.index.search(query_embedding, k)

            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and score >= threshold:
                    results.append(SearchResult(metadata=self.metadata[idx], score=float(score), document_id=str(idx)))

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise IndexError(f"Search failed: {e}")

    async def add_documents(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add documents to existing index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            # Validate and normalize
            if self.normalize_embeddings:
                faiss.normalize_L2(embeddings)

            # Add to index
            self.index.add(embeddings)

            # Add to metadata
            self.metadata.extend(metadata)

            # Update info
            self.index_info.total_documents = len(self.metadata)
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save changes
            if self._index_path:
                await self._save_index(self._index_path)

            logger.info(f"Added {len(metadata)} documents to FAISS index")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise IndexError(f"Document addition failed: {e}")

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from index (not supported by FAISS directly)."""
        raise NotImplementedError("FAISS doesn't support document removal. Rebuild index instead.")

    async def update_document(self, document_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        """Update document (not supported by FAISS directly)."""
        raise NotImplementedError("FAISS doesn't support document updates. Rebuild index instead.")

    async def get_index_info(self) -> IndexMetadata:
        """Get index metadata."""
        if not self.index_info:
            raise IndexError("No index loaded")
        return self.index_info

    async def validate_index(self, index_path: Path) -> bool:
        """Validate FAISS index integrity."""
        try:
            # Try to load and perform basic operations
            temp_organizer = FaissIndexOrganizer(self.config)
            await temp_organizer.load_index(index_path)

            # Test search with random query
            if temp_organizer.index.ntotal > 0:
                random_query = np.random.random((1, temp_organizer.index.d)).astype(np.float32)
                await temp_organizer.search(random_query, k=1)

            return True

        except Exception as e:
            logger.warning(f"Index validation failed: {e}")
            return False

    async def get_stats(self) -> IndexStats:
        """Get index statistics."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        # Perform sample searches to estimate performance
        start_time = time.time()
        if self.index.ntotal > 0:
            random_query = np.random.random((1, self.index.d)).astype(np.float32)
            await self.search(random_query, k=5)
        search_latency = (time.time() - start_time) * 1000

        return IndexStats(
            total_documents=self.index.ntotal,
            average_score_distribution={"high": 0.3, "medium": 0.5, "low": 0.2},
            search_latency_ms=search_latency,
            memory_usage_mb=self._estimate_memory_usage(),
        )

    async def optimize_index(self) -> None:
        """Optimize FAISS index (no-op for flat indices)."""
        if self.index_type == "IndexFlatIP":
            logger.info("Flat index doesn't need optimization")
        else:
            logger.info("Index optimization not implemented for this index type")

    def _create_faiss_index(self, dimension: int):
        """Create FAISS index based on configuration."""
        if self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(dimension)
        elif self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = self.config.get("nlist", 100)
            return faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            raise ConfigurationError(f"Unsupported index type: {self.index_type}")

    def _validate_inputs(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Validate input data."""
        if len(embeddings) == 0:
            raise ValueError("Embeddings array is empty")

        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != metadata count ({len(metadata)})")

        if embeddings.dtype != np.float32:
            raise ValueError("Embeddings must be float32 for FAISS")

    async def _save_index(self, index_path: Path) -> None:
        """Save index and metadata to disk."""
        index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "index.faiss"))

        # Save metadata
        with open(index_path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        # Save manifest
        manifest = {
            "embedding_model": self.index_info.embedding_model,
            "schema_version": self.index_info.schema_version,
            "created_at": self.index_info.created_at,
            "updated_at": self.index_info.updated_at,
            "backend_type": "faiss",
            "backend_config": self.index_info.backend_config,
        }

        with open(index_path / "index_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.index:
            return 0.0

        # Rough estimation based on index type and size
        base_size = self.index.ntotal * self.index.d * 4  # float32 = 4 bytes
        metadata_size = len(str(self.metadata).encode("utf-8"))

        return (base_size + metadata_size) / (1024 * 1024)
