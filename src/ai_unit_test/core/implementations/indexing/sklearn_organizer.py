"""Sklearn index organizer implementation."""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ai_unit_test.core.exceptions import ConfigurationError, IndexError, IndexNotFoundError
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult

if TYPE_CHECKING:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

    SKLEARN_AVAILABLE = True
else:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import joblib
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.neighbors import NearestNeighbors
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
        NearestNeighbors = None
        cosine_similarity = None
        joblib = None

logger = logging.getLogger(__name__)


class SklearnIndexOrganizer(IndexOrganizer):
    """Sklearn-based index organizer implementation."""

    index_info: IndexMetadata
    model: NearestNeighbors
    embeddings: np.ndarray
    metadata: list[dict[str, Any]]
    _index_loaded: bool
    _index_path: Path | None

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the SklearnIndexOrganizer.

        Args:
            config: Configuration dictionary for the index.
        """
        super().__init__(config)

        if not SKLEARN_AVAILABLE:
            raise ConfigurationError("Sklearn is not available. Please install scikit-learn")

        self.algorithm = config.get("algorithm", "ball_tree")
        self.metric = config.get("metric", "cosine")
        self.n_jobs = config.get("n_jobs", -1)

        if self.metric == "cosine" and self.algorithm != "brute":
            self.algorithm = "brute"
            logger.warning("Algorithm changed to 'brute' to support 'cosine' metric.")

    async def create_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        index_path: Path,
        model_name: str,
    ) -> IndexMetadata:
        """Create and save sklearn index."""
        try:
            # Validate inputs
            self._validate_inputs(embeddings, metadata)

            # Create sklearn nearest neighbors model
            self.model = NearestNeighbors(algorithm=self.algorithm, metric=self.metric, n_jobs=self.n_jobs)

            # Fit the model
            self.model.fit(embeddings)

            # Store data
            self.embeddings = embeddings.copy()
            self.metadata = metadata
            self.doc_ids = [str(i) for i in range(len(metadata))]

            # Prepare metadata
            self.index_info = IndexMetadata(
                embedding_model=model_name,
                schema_version="1.1.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                total_documents=len(metadata),
                embedding_dimension=embeddings.shape[1],
                backend_type="sklearn",
                backend_config={
                    "algorithm": self.algorithm,
                    "metric": self.metric,
                    "n_jobs": self.n_jobs,
                },
            )

            # Save to disk
            await self._save_index(index_path)

            self._index_loaded = True
            self._index_path = index_path

            logger.info(f"Sklearn index created with {len(metadata)} documents")
            return self.index_info

        except Exception as e:
            logger.error(f"Failed to create sklearn index: {e}")
            raise IndexError(f"Index creation failed: {e}")

    async def load_index(self, index_path: Path) -> IndexMetadata:
        """Load existing sklearn index."""
        try:
            model_file = index_path / "sklearn_model.joblib"
            embeddings_file = index_path / "embeddings.npy"
            metadata_file = index_path / "index_meta.json"
            manifest_file = index_path / "index_manifest.json"

            # Check if files exist
            if not all(f.exists() for f in [model_file, embeddings_file, metadata_file, manifest_file]):
                raise IndexNotFoundError(f"Index files not found in {index_path}")

            # Load model
            self.model = joblib.load(model_file)

            # Load embeddings
            self.embeddings = np.load(embeddings_file)

            # Load metadata
            with open(metadata_file, encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.doc_ids = [str(i) for i in range(len(self.metadata))]

            # Load manifest
            with open(manifest_file, encoding="utf-8") as f:
                manifest_data = json.load(f)

            self.index_info = IndexMetadata(
                embedding_model=manifest_data.get("embedding_model", "unknown"),
                schema_version=manifest_data.get("schema_version", "1.0.0"),
                created_at=manifest_data.get("created_at", "unknown"),
                updated_at=manifest_data.get("updated_at", "unknown"),
                total_documents=len(self.metadata),
                embedding_dimension=self.embeddings.shape[1],
                backend_type="sklearn",
                backend_config=manifest_data.get("backend_config", {}),
            )

            self._index_loaded = True
            self._index_path = index_path

            logger.info(f"Sklearn index loaded: {len(self.metadata)} documents")
            return self.index_info

        except Exception as e:
            logger.error(f"Failed to load sklearn index: {e}")
            raise IndexError(f"Index loading failed: {e}")

    async def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7) -> list[SearchResult]:
        """Search sklearn index."""
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

            # Find k nearest neighbors
            distances, indices = self.model.kneighbors(query_embedding, n_neighbors=k)

            # Calculate cosine similarities if needed
            if self.metric != "cosine":
                similarities = cosine_similarity(query_embedding, self.embeddings[indices[0]])
                scores = similarities[0]
            else:
                # For cosine metric, convert distances to similarities
                scores = 1.0 - distances[0]

            # Process results
            results = []
            for _, (score, idx) in enumerate(zip(scores, indices[0])):
                if score >= threshold:
                    results.append(
                        SearchResult(
                            metadata=self.metadata[idx],
                            score=float(score),
                            document_id=str(idx),
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Sklearn search failed: {e}")
            raise IndexError(f"Search failed: {e}")

    async def add_documents(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add documents to existing index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            # Combine with existing data
            combined_embeddings = np.vstack([self.embeddings, embeddings])
            combined_metadata = self.metadata + metadata

            # Retrain the model
            self.model.fit(combined_embeddings)

            # Update stored data
            self.embeddings = combined_embeddings
            self.metadata = combined_metadata

            # Update info
            self.index_info.total_documents = len(self.metadata)
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save changes
            if self._index_path:
                await self._save_index(self._index_path)

            logger.info(f"Added {len(metadata)} documents to sklearn index")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise IndexError(f"Document addition failed: {e}")

    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from index."""
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

            # Retrain the model
            self.model.fit(self.embeddings)

            # Update info
            self.index_info.total_documents = len(self.metadata)
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save changes
            if self._index_path:
                await self._save_index(self._index_path)

            logger.info(f"Removed {len(document_ids)} documents from sklearn index")

        except Exception as e:
            logger.error(f"Failed to remove documents: {e}")
            raise IndexError(f"Document removal failed: {e}")

    async def update_document(self, document_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        """Update document in index."""
        if not self._index_loaded:
            raise IndexError("No index loaded")

        try:
            idx = int(document_id)

            # Update embedding and metadata
            self.embeddings[idx] = embedding
            self.metadata[idx] = metadata

            # Retrain the model
            self.model.fit(self.embeddings)

            # Update info
            self.index_info.updated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Save changes
            if self._index_path:
                await self._save_index(self._index_path)

            logger.info(f"Updated document {document_id} in sklearn index")

        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise IndexError(f"Document update failed: {e}")

    async def get_index_info(self) -> IndexMetadata:
        """Get index metadata."""
        if not self.index_info:
            raise IndexError("No index loaded")
        return self.index_info

    async def validate_index(self, index_path: Path) -> bool:
        """Validate sklearn index integrity."""
        try:
            # Try to load and perform basic operations
            temp_organizer = SklearnIndexOrganizer(self.config)
            await temp_organizer.load_index(index_path)

            # Test search with random query
            if len(temp_organizer.embeddings) > 0:
                random_query = np.random.random((1, temp_organizer.embeddings.shape[1]))
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
        """Optimize sklearn index (retrain with current data)."""
        if not self._index_loaded or len(self.embeddings) == 0:
            logger.info("No data to optimize")
            return

        # Retrain the model for optimization
        self.model.fit(self.embeddings)
        logger.info("Sklearn index optimized by retraining")

    def _validate_inputs(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Validate input data."""
        if len(embeddings) == 0:
            raise ValueError("Embeddings array is empty")

        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != metadata count ({len(metadata)})")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array")

    async def _save_index(self, index_path: Path) -> None:
        """Save index and metadata to disk."""
        index_path.mkdir(parents=True, exist_ok=True)

        # Save sklearn model
        joblib.dump(self.model, index_path / "sklearn_model.joblib")

        # Save embeddings
        np.save(index_path / "embeddings.npy", self.embeddings)

        # Save metadata
        with open(index_path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        # Save manifest
        manifest = {
            "embedding_model": self.index_info.embedding_model,
            "schema_version": self.index_info.schema_version,
            "created_at": self.index_info.created_at,
            "updated_at": self.index_info.updated_at,
            "backend_type": "sklearn",
            "backend_config": self.index_info.backend_config,
        }

        with open(index_path / "index_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if self.embeddings is None:
            return 0.0

        # Estimate memory usage
        embeddings_size = self.embeddings.nbytes
        metadata_size = len(str(self.metadata).encode("utf-8"))
        model_size = 1024 * 1024  # Rough estimate for sklearn model

        return (embeddings_size + metadata_size + model_size) / (1024 * 1024)
