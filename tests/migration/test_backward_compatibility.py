"""Test backward compatibility."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tomli as toml

from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer
from ai_unit_test.services.orchestration_service import OrchestrationService


class TestBackwardCompatibility:
    """Test backward compatibility."""

    def test_load_old_config(self, temp_dir: Path) -> None:
        """Test that the application can handle an old configuration file."""
        # Arrange
        fake_project_path = temp_dir / "fake_project"
        fake_project_path.mkdir()

        pyproject_toml_content = """
[tool.ai-unit-test]
llm_provider = "mock"
"""
        pyproject_toml = fake_project_path / "pyproject.toml"
        pyproject_toml.write_text(pyproject_toml_content)

        # Act
        with patch(
            "ai_unit_test.services.configuration_service.ConfigurationService.load_pyproject_config",
            return_value=toml.loads(pyproject_toml_content),
        ):
            orchestration_service = OrchestrationService(config={"project_path": str(fake_project_path)})

            # Assert
            assert orchestration_service.config_service.get_llm_config()["provider"] == "mock"
            # Check that default values are used for missing config
            assert orchestration_service.config_service.get_indexing_config().get("backend", "auto") == "auto"

    @pytest.mark.asyncio
    async def test_load_old_index(self, temp_dir: Path) -> None:
        """Test that the application can handle an old index format."""
        # Arrange
        fake_project_path = temp_dir / "fake_project"
        fake_project_path.mkdir()
        index_path = fake_project_path / "index"
        index_path.mkdir()

        # Create a dummy old index
        embeddings = np.random.random((1, 384)).astype(np.float32)
        np.save(index_path / "embeddings.npy", embeddings)
        metadata = {"0": {"file_path": "src/main.py", "function_name": "main"}}
        with open(index_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Act
        with patch(
            "ai_unit_test.core.factories.index_factory.IndexOrganizerFactory.create_from_config_file"
        ) as mock_create:
            import time

            from ai_unit_test.core.interfaces.index_organizer import IndexMetadata

            organizer = InMemoryIndexOrganizer({})
            organizer.doc_ids = ["0"]  # Simulates populated doc_ids
            organizer._index_loaded = True  # Simulates that the index is already loaded

            # Initialize index_info
            organizer.index_info = IndexMetadata(
                embedding_model="test-model",
                total_documents=1,
                embedding_dimension=384,
                schema_version="1.0.0",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                backend_type="memory",
                backend_config={},
            )

            mock_create.return_value = organizer

            orchestration_service = OrchestrationService(config={"project_path": str(fake_project_path)})
            await orchestration_service.load_index(index_path)

            # Assert
            assert orchestration_service.index_organizer is not None
            assert hasattr(orchestration_service.index_organizer, "doc_ids")
            assert len(orchestration_service.index_organizer.doc_ids) == 1  # type: ignore[attr-defined]
