"""Test cases for the index factory."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Import required types at module level
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexStats, SearchResult


def test_index_factory_various_paths(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: C901
    """Test various paths of the index factory."""
    from ai_unit_test.core.factories.index_factory import ConfigurationError  # type: ignore[attr-defined]
    from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory
    from ai_unit_test.core.interfaces.index_organizer import IndexOrganizer

    # Save originals - add type hints
    orig_organizers: dict[str, type[IndexOrganizer]] = dict(getattr(IndexOrganizerFactory, "_organizers", {}))
    orig_availability_cache: dict[str, bool] = dict(getattr(IndexOrganizerFactory, "_availability_cache", {}))
    orig_check = IndexOrganizerFactory._check_availability
    orig_create = IndexOrganizerFactory.create_organizer
    orig_auto = IndexOrganizerFactory._auto_detect_backend

    class DummyOrganizer(IndexOrganizer):  # type: ignore
        """Dummy organizer for testing."""

        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)

        # Implement all required abstract methods from IndexOrganizer interface
        async def create_index(
            self,
            embeddings: np.ndarray,
            metadata: list[dict[str, Any]],
            index_path: Path,
            model_name: str,
        ) -> IndexMetadata:
            """Create index - dummy implementation."""
            return IndexMetadata(
                embedding_model=model_name,
                schema_version="1.0",
                created_at="2024-01-01",
                updated_at="2024-01-01",
                total_documents=len(metadata),
                embedding_dimension=embeddings.shape[1] if embeddings.size > 0 else 384,
                backend_type="dummy",
                backend_config={},
            )

        async def load_index(self, index_path: Path) -> IndexMetadata:
            """Load index - dummy implementation."""
            return IndexMetadata(
                embedding_model="dummy",
                schema_version="1.0",
                created_at="2024-01-01",
                updated_at="2024-01-01",
                total_documents=0,
                embedding_dimension=384,
                backend_type="dummy",
                backend_config={},
            )

        async def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7) -> list[SearchResult]:
            """Search - dummy implementation."""
            return []

        async def add_documents(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
            """Add documents - dummy implementation."""
            pass

        async def remove_documents(self, document_ids: list[str]) -> None:
            """Remove documents - dummy implementation."""
            pass

        async def update_document(self, document_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
            """Update document - dummy implementation."""
            pass

        async def get_index_info(self) -> IndexMetadata:
            """Get index info - dummy implementation."""
            return IndexMetadata(
                embedding_model="dummy",
                schema_version="1.0",
                created_at="2024-01-01",
                updated_at="2024-01-01",
                total_documents=0,
                embedding_dimension=384,
                backend_type="dummy",
                backend_config={},
            )

        async def validate_index(self, index_path: Path) -> bool:
            """Validate index - dummy implementation."""
            return True

        async def get_stats(self) -> IndexStats:
            """Get stats - dummy implementation."""
            return IndexStats(
                total_documents=0, average_score_distribution={}, search_latency_ms=0.0, memory_usage_mb=0.0
            )

        async def optimize_index(self) -> None:
            """Optimize index - dummy implementation."""
            pass

        # Legacy methods for backward compatibility (not part of interface)
        async def initialize(self) -> None:
            """Initialize the organizer."""
            pass

        async def build_index_from_directory(
            self, directory: Path, patterns: list[str], exclude_patterns: list[str] | None = None
        ) -> None:
            """Build index from directory."""
            pass

        async def search_similar_code(
            self, query_embedding: list[float], top_k: int = 5, score_threshold: float = 0.5
        ) -> list[dict[str, Any]]:
            """Search similar code."""
            return []

        def save_index(self, path: Path) -> None:
            """Save index to path."""
            pass

        def get_index_metadata(self) -> dict[str, Any]:
            """Get index metadata."""
            return {}

        def get_index_stats(self) -> dict[str, Any]:
            """Get index statistics."""
            return {}

    try:
        # 1) Basic creation with case-insensitive backend name and merging config passed through to organizer
        monkeypatch.setattr(IndexOrganizerFactory, "_organizers", {"memory": DummyOrganizer})

        def fake_check_availability(cls: object, name: str, organizer_class: type[IndexOrganizer] | None) -> bool:
            return name == "memory"

        monkeypatch.setattr(IndexOrganizerFactory, "_check_availability", classmethod(fake_check_availability))

        inst = IndexOrganizerFactory.create_organizer("MeMoRy", {"k": 1})
        assert isinstance(inst, DummyOrganizer)
        assert inst.config["k"] == 1

        # 2) Unknown backend raises ConfigurationError with available backends listed
        with pytest.raises(ConfigurationError):
            IndexOrganizerFactory.create_organizer("unknown", {})

        # 3) Backend present but not available should raise ConfigurationError
        # Add a backend that will be reported as not available by fake_check_availability
        IndexOrganizerFactory._organizers["sklearn"] = DummyOrganizer  # type: ignore[assignment,arg-type,type-abstract]
        with pytest.raises(ConfigurationError):
            IndexOrganizerFactory.create_organizer("sklearn", {})

        # 4) create_from_config: auto backend should result in create_organizer called with None
        called: dict[str, Any] = {}

        def fake_create(cls: object, backend: str | None, merged_config: dict[str, Any]) -> str:
            called["backend"] = backend
            called["merged"] = merged_config
            return "created_auto"

        monkeypatch.setattr(IndexOrganizerFactory, "create_organizer", classmethod(fake_create))

        cfg_auto = {"tool": {"ai-unit-test": {"indexing": {"backend": "auto", "a": 5}}}}
        res = IndexOrganizerFactory.create_from_config_file(cfg_auto)
        assert res == "created_auto"  # type: ignore[comparison-overlap]
        assert called["backend"] is None
        assert called["merged"]["a"] == 5

        # 5) create_from_config: explicit backend merges backend-specific config overriding keys
        called.clear()
        cfg_mem = {"tool": {"ai-unit-test": {"indexing": {"backend": "memory", "x": 1, "memory": {"x": 2, "y": 3}}}}}
        res2 = IndexOrganizerFactory.create_from_config_file(cfg_mem)
        assert res2 == "created_auto"  # type: ignore[comparison-overlap] # fake_create returns same sentinel
        assert called["backend"] == "memory"
        assert called["merged"]["x"] == 2
        assert called["merged"]["y"] == 3

        # 6) _auto_detect_backend respects priority and availability checks
        # Prepare organizers in priority order and make only memory available
        IndexOrganizerFactory._organizers = {  # type: ignore[assignment,call-overload]
            "faiss": DummyOrganizer,  # type: ignore[dict-item,arg-type,type-abstract]
            "sklearn": DummyOrganizer,  # type: ignore[dict-item,arg-type,type-abstract]
            "memory": DummyOrganizer,  # type: ignore[dict-item,arg-type,type-abstract]
        }

        def avail_only_memory(cls: object, name: str, organizer_class: type[IndexOrganizer] | None) -> bool:
            return name == "memory"

        monkeypatch.setattr(IndexOrganizerFactory, "_check_availability", classmethod(avail_only_memory))
        autodetected = IndexOrganizerFactory._auto_detect_backend()
        assert autodetected == "memory"

        # 7) _check_availability real import-based behavior: caching and ImportError handling
        # Restore original check function for import-based testing
        monkeypatch.setattr(IndexOrganizerFactory, "_check_availability", orig_check)
        # Clear cache
        IndexOrganizerFactory._availability_cache.clear()

        # Call for a backend likely not installed (faiss) -> should return False and be cached
        val1 = IndexOrganizerFactory._check_availability(
            "faiss", DummyOrganizer  # type: ignore[type-abstract]
        )  # type: ignore[arg-type,call-overload]
        assert IndexOrganizerFactory._availability_cache["faiss"] == val1
        # Calling again should hit cache (no exception) and return same
        val2 = IndexOrganizerFactory._check_availability(
            "faiss", DummyOrganizer  # type: ignore[type-abstract]
        )  # type: ignore[arg-type,call-overload]
        assert val1 == val2

        # memory should be available according to implementation
        mem_val = IndexOrganizerFactory._check_availability(
            "memory", DummyOrganizer  # type: ignore[type-abstract]
        )  # type: ignore[arg-type,call-overload]
        assert mem_val is True
        assert IndexOrganizerFactory._availability_cache["memory"] is True

    finally:
        # Restore originals
        IndexOrganizerFactory._organizers = orig_organizers
        IndexOrganizerFactory._availability_cache = orig_availability_cache
        monkeypatch.setattr(IndexOrganizerFactory, "_check_availability", orig_check)
        monkeypatch.setattr(IndexOrganizerFactory, "create_organizer", orig_create)
        monkeypatch.setattr(IndexOrganizerFactory, "_auto_detect_backend", orig_auto)
