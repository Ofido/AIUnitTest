"""Test index organizer interface compliance."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_unit_test.core.exceptions import IndexError
from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult


class TestIndexOrganizerInterface:
    """Test that all index organizers implement the interface correctly."""

    @pytest.mark.parametrize(
        ("organizer_class", "config"),
        [
            (InMemoryIndexOrganizer, {"max_documents": 1000}),
        ],
    )
    async def test_organizer_interface_compliance(
        self, organizer_class: type[IndexOrganizer], config: dict[str, Any]
    ) -> None:
        """Test that organizer implements all required interface methods."""
        organizer = organizer_class(config)
        assert isinstance(organizer, IndexOrganizer)

        required_methods = [
            "create_index",
            "load_index",
            "search",
            "add_documents",
            "remove_documents",
            "update_document",
            "get_index_info",
            "validate_index",
            "get_stats",
            "optimize_index",
        ]

        for method_name in required_methods:
            assert hasattr(organizer, method_name)
            assert callable(getattr(organizer, method_name))

    async def test_create_and_search_workflow(
        self, sample_embeddings: np.ndarray, sample_metadata: list[dict[str, Any]], temp_dir: Path
    ) -> None:
        """Test complete create and search workflow."""
        async with InMemoryIndexOrganizer({"max_documents": 1000}) as organizer:
            # Create index
            index_path = temp_dir / "test_index"
            index_info = await organizer.create_index(sample_embeddings, sample_metadata, index_path, "test-model")

            # Validate index metadata
            assert isinstance(index_info, IndexMetadata)
            assert index_info.embedding_model == "test-model"
            assert index_info.total_documents == len(sample_metadata)
            assert index_info.embedding_dimension == sample_embeddings.shape[1]

            # Test search
            query_embedding = sample_embeddings[0:1]  # Use first embedding as query
            results = await organizer.search(query_embedding, k=3, threshold=0.0)

            # Validate search results
            assert isinstance(results, list)
            assert len(results) <= 3

            for result in results:
                assert isinstance(result, SearchResult)
                assert isinstance(result.metadata, dict)
                assert isinstance(result.score, float)
                assert isinstance(result.document_id, str)
                assert 0.0 <= result.score <= 1.0

    async def test_add_documents_workflow(
        self, sample_embeddings: np.ndarray, sample_metadata: list[dict[str, Any]], temp_dir: Path
    ) -> None:
        """Test adding documents to existing index."""
        async with InMemoryIndexOrganizer({"max_documents": 1000}) as organizer:
            # Create initial index
            index_path = temp_dir / "test_index"
            await organizer.create_index(
                sample_embeddings[:5], sample_metadata[:5], index_path, "test-model"  # First 5 documents
            )

            # Add more documents
            await organizer.add_documents(sample_embeddings[5:], sample_metadata[5:])  # Remaining documents

            # Verify all documents are searchable
            query_embedding = sample_embeddings[0:1]
            results = await organizer.search(query_embedding, k=10, threshold=0.0)

            # Should be able to find documents from both batches
            assert len(results) == len(sample_metadata)

    async def test_get_stats_contract(
        self, sample_embeddings: np.ndarray, sample_metadata: list[dict[str, Any]], temp_dir: Path
    ) -> None:
        """Test get_stats method contract."""
        async with InMemoryIndexOrganizer({"max_documents": 1000}) as organizer:
            index_path = temp_dir / "test_index"
            await organizer.create_index(sample_embeddings, sample_metadata, index_path, "test-model")

            stats = await organizer.get_stats()

            assert isinstance(stats, IndexStats)
            assert isinstance(stats.total_documents, int)
            assert isinstance(stats.search_latency_ms, float)
            assert isinstance(stats.memory_usage_mb, float)
            assert isinstance(stats.average_score_distribution, dict)

            assert stats.total_documents == len(sample_metadata)
            assert stats.search_latency_ms >= 0
            assert stats.memory_usage_mb >= 0

    async def test_error_handling_contract(self) -> None:
        """Test error handling behavior."""
        organizer = InMemoryIndexOrganizer({"max_documents": 1000})

        # Test search without loading index
        with pytest.raises(IndexError):
            await organizer.search(np.random.random((1, 384)).astype(np.float32))

        # Test getting info without loading index
        with pytest.raises(IndexError):
            await organizer.get_index_info()

        # Test getting stats without loading index
        with pytest.raises(IndexError):
            await organizer.get_stats()


async def test_index_organizer_base_noop_and_context_manager() -> None:  # noqa: C901
    """Test IndexOrganizer base implementation and context manager."""
    from pathlib import Path

    import numpy as np

    from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult

    class DummyOrganizer(IndexOrganizer):  # type: ignore
        """Dummy organizer for testing base functionality."""

        async def create_index(
            self, embeddings: np.ndarray, metadata: list[dict[str, Any]], index_path: Path, model_name: str
        ) -> IndexMetadata:
            """Create index - dummy implementation."""
            return IndexMetadata(
                embedding_model=model_name,
                schema_version="1.0",
                created_at="2024-01-01",
                updated_at="2024-01-01",
                total_documents=len(metadata),
                embedding_dimension=embeddings.shape[1],
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

        async def search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.0) -> list[SearchResult]:
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

    cfg = {"test": True}
    emb = np.random.random((2, 8)).astype(np.float32)
    meta = [{"id": "a"}, {"id": "b"}]

    # Verify init attributes from base class are set
    inst = DummyOrganizer(cfg)
    assert inst.config == cfg
    assert inst._index_loaded is False
    assert inst._index_path is None

    # Use async context manager and verify it returns self
    async with DummyOrganizer(cfg) as ctx:
        assert isinstance(ctx, DummyOrganizer)
        assert ctx is not None

        # Call all methods to exercise the implementations
        res_create = await ctx.create_index(emb, meta, Path("idx"), "model-x")
        res_load = await ctx.load_index(Path("idx"))
        res_search = await ctx.search(emb[0:1], k=1, threshold=0.0)
        await ctx.add_documents(emb, meta)
        await ctx.remove_documents(["a"])
        await ctx.update_document("a", emb[0], {"id": "a"})
        res_info = await ctx.get_index_info()
        res_validate = await ctx.validate_index(Path("dummy_path"))
        res_stats = await ctx.get_stats()
        await ctx.optimize_index()

        # Verify return types
        assert isinstance(res_create, IndexMetadata)
        assert isinstance(res_load, IndexMetadata)
        assert isinstance(res_search, list)
        assert isinstance(res_info, IndexMetadata)
        assert isinstance(res_validate, bool)
        assert isinstance(res_stats, IndexStats)
