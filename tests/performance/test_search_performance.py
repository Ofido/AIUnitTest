"""Test search performance."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer


class TestSearchPerformance:
    """Test search performance."""

    @pytest.mark.asyncio
    async def test_search_performance(self, performance_config: dict[str, Any], temp_dir: Path) -> None:
        """Test that the search performance is within a reasonable range."""
        # Arrange
        max_response_time = performance_config["max_response_time_ms"] / 1000.0
        num_documents = 10000
        embedding_dim = 384

        embeddings = np.random.random((num_documents, embedding_dim)).astype(np.float32)
        metadata = [{"file_path": f"src/module_{i}.py", "function_name": f"function_{i}"} for i in range(num_documents)]

        async with InMemoryIndexOrganizer({"max_documents": num_documents}) as organizer:
            index_path = temp_dir / "performance_index"
            await organizer.create_index(embeddings, metadata, index_path, "test-model")

            query_embedding = np.random.random((1, embedding_dim)).astype(np.float32)

            # Act
            start_time = time.perf_counter()
            await organizer.search(query_embedding, k=10)
            end_time = time.perf_counter()

            # Assert
            search_time = end_time - start_time
            assert search_time < max_response_time
