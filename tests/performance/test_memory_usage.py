"""Test memory usage."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest

from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer


class TestMemoryUsage:
    """Test memory usage."""

    @pytest.mark.asyncio
    async def test_memory_usage(self, performance_config: dict[str, Any], temp_dir: Path) -> None:
        """Test that the memory usage is within a reasonable range."""
        # Arrange
        max_memory_usage = 1024 * 1024 * 1024  # 1GB
        num_documents = 10000
        embedding_dim = 384

        embeddings = np.random.random((num_documents, embedding_dim)).astype(np.float32)
        metadata = [{"file_path": f"src/module_{i}.py", "function_name": f"function_{i}"} for i in range(num_documents)]

        process = psutil.Process(os.getpid())

        # Act
        async with InMemoryIndexOrganizer({"max_documents": num_documents}) as organizer:
            index_path = temp_dir / "memory_index"
            await organizer.create_index(embeddings, metadata, index_path, "test-model")

            # Assert
            memory_usage = process.memory_info().rss
            assert memory_usage < max_memory_usage
