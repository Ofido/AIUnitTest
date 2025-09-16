"""Test index backend contracts."""

from pathlib import Path

import numpy as np
import pytest

try:
    from ai_unit_test.core.implementations.indexing.faiss_organizer import FaissIndexOrganizer

    faiss_installed = True
except ImportError:
    faiss_installed = False

try:
    from ai_unit_test.core.implementations.indexing.sklearn_organizer import SklearnIndexOrganizer

    sklearn_installed = True
except ImportError:
    sklearn_installed = False


@pytest.mark.skipif(not faiss_installed, reason="faiss not installed")
class TestFaissContract:
    """Test Faiss backend contract."""

    @pytest.mark.asyncio
    async def test_faiss_contract(self, temp_dir: Path) -> None:
        """Test that the Faiss backend works as expected."""
        # Arrange
        async with FaissIndexOrganizer({}) as organizer:  # pyright: ignore[reportPossiblyUnboundVariable]
            embeddings = np.random.random((10, 384)).astype(np.float32)
            metadata = [{"file_path": f"src/module_{i}.py", "function_name": f"function_{i}"} for i in range(10)]
            index_path = temp_dir / "faiss_index"

            # Act
            await organizer.create_index(embeddings, metadata, index_path, "test-model")
            await organizer.load_index(index_path)
            results = await organizer.search(embeddings[0:1], k=3)

            # Assert
            assert len(results) == 3
            assert results[0].document_id == "0"  # First document should have ID "0"


@pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
class TestSklearnContract:
    """Test Sklearn backend contract."""

    @pytest.mark.asyncio
    async def test_sklearn_contract(self, temp_dir: Path) -> None:
        """Test that the Sklearn backend works as expected."""
        # Arrange
        async with SklearnIndexOrganizer({}) as organizer:  # pyright: ignore[reportPossiblyUnboundVariable]
            embeddings = np.random.random((10, 384)).astype(np.float32)
            metadata = [{"file_path": f"src/module_{i}.py", "function_name": f"function_{i}"} for i in range(10)]
            index_path = temp_dir / "sklearn_index"

            # Act
            await organizer.create_index(embeddings, metadata, index_path, "test-model")
            await organizer.load_index(index_path)
            results = await organizer.search(embeddings[0:1], k=3)

            # Assert
            assert len(results) == 3
            assert results[0].document_id == organizer.doc_ids[0]  # type: ignore
