"""Abstract interface for index organizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IndexMetadata:
    """Metadata about a vector index."""

    embedding_model: str
    schema_version: str
    created_at: str
    updated_at: str
    total_documents: int
    embedding_dimension: int
    backend_type: str
    backend_config: dict[str, Any]


@dataclass
class SearchResult:
    """Result from index search operation."""

    metadata: dict[str, Any]
    score: float
    document_id: str
    embedding: np.ndarray | None = None


@dataclass
class IndexStats:
    """Statistics about index performance and content."""

    total_documents: int
    average_score_distribution: dict[str, float]
    search_latency_ms: float
    memory_usage_mb: float


class IndexOrganizer(ABC):
    """Abstract base class for all index organizers."""

    _index_loaded: bool
    _index_path: Path | None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize organizer with configuration."""
        self.config = config
        self._index_loaded = False
        self._index_path = None

    @abstractmethod
    async def create_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        index_path: Path,
        model_name: str,
    ) -> IndexMetadata:
        """Create and save a new index."""
        pass

    @abstractmethod
    async def load_index(self, index_path: Path) -> IndexMetadata:
        """Load an existing index."""
        pass

    @abstractmethod
    async def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7) -> list[SearchResult]:
        """Search the loaded index."""
        pass

    @abstractmethod
    async def add_documents(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add new documents to existing index."""
        pass

    @abstractmethod
    async def remove_documents(self, document_ids: list[str]) -> None:
        """Remove documents from index."""
        pass

    @abstractmethod
    async def update_document(self, document_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        """Update a specific document in the index."""
        pass

    @abstractmethod
    async def get_index_info(self) -> IndexMetadata:
        """Get metadata about the current index."""
        pass

    @abstractmethod
    async def validate_index(self, index_path: Path) -> bool:
        """Validate index integrity."""
        pass

    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """Get performance and content statistics."""
        pass

    @abstractmethod
    async def optimize_index(self) -> None:
        """Optimize index for better performance."""
        pass

    async def __aenter__(self) -> "IndexOrganizer":
        """Async context manager entry."""
        return self

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Async context manager exit."""
        # Override in implementations if cleanup is needed
        pass
