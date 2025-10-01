"""Test orchestration service."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_unit_test.services.orchestration_service import OrchestrationService
from ai_unit_test.services.processing_service import CoverageProcessingResult


@pytest.fixture
def mock_config_service() -> Generator[MagicMock]:
    """Mock the ConfigurationService."""
    with patch("ai_unit_test.services.configuration_service.ConfigurationService") as MockConfigService:
        mock_service = MockConfigService.return_value
        mock_service.resolve_paths_from_config.return_value = (["src"], "tests", ".coverage")
        mock_service.validate_environment.return_value = MagicMock()
        mock_service.get_llm_config.return_value = {"provider": "mock"}
        mock_service.get_indexing_config.return_value = {}
        yield mock_service


@pytest.fixture
def mock_test_processing_service() -> Generator[MagicMock]:
    """Mock the TestProcessingService."""
    with patch("ai_unit_test.services.orchestration_service.TestProcessingService") as MockTestProcessingService:
        mock_service = MockTestProcessingService.return_value
        mock_service.__aenter__.return_value.process_missing_coverage.return_value = CoverageProcessingResult(
            status="success",
            files_processed=1,
            tests_generated=1,
        )
        yield mock_service


@pytest.mark.asyncio
async def test_run_test_generation_workflow(
    mock_config_service: MagicMock, mock_test_processing_service: MagicMock
) -> None:
    """Test the test generation workflow."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    results = await orchestration_service.run_test_generation_workflow()

    assert "workflow_duration_seconds" in results
    assert "configuration" in results
    assert "environment" in results


@pytest.mark.asyncio
async def test_run_health_check_workflow(mock_config_service: MagicMock) -> None:
    """Test the health check workflow."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    with patch(
        "ai_unit_test.core.factories.llm_factory.LLMConnectorFactory.create_from_config_file"
    ) as mock_llm_factory:
        # Create async context manager mock
        mock_llm_connector = AsyncMock()

        # Configure async methods
        mock_llm_connector.health_check = AsyncMock(return_value=True)
        mock_llm_connector.get_connector_info = MagicMock(return_value={"provider": "test"})

        mock_llm_factory.return_value = mock_llm_connector

        with patch(
            "ai_unit_test.core.factories.index_factory.IndexOrganizerFactory.get_available_organizers",
            return_value=["faiss"],
        ):
            results = await orchestration_service.run_health_check_workflow()

        assert results.status == "healthy", "".join(
            [
                check.error
                for _, check in vars(results.checks).items()
                if check is not None and hasattr(check, "healthy") and not check.healthy
            ]
        )


@pytest.mark.asyncio
async def test_run_coverage_analysis_workflow(
    mock_config_service: MagicMock, mock_test_processing_service: MagicMock
) -> None:
    """Test the coverage analysis workflow."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    results = await orchestration_service.run_coverage_analysis_workflow(folders=["src"], tests_folder="tests")

    assert "status" in results
    assert results["status"] == "success"


@pytest.mark.asyncio
async def test_load_index(mock_config_service: MagicMock) -> None:
    """Test the load index workflow."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    with patch(
        "ai_unit_test.core.factories.index_factory.IndexOrganizerFactory.create_from_config_file"
    ) as mock_index_factory:
        mock_index_organizer = AsyncMock()
        mock_index_factory.return_value = mock_index_organizer

        await orchestration_service.load_index(Path("tests/temp_index"))

    mock_index_organizer.load_index.assert_called_once()


@pytest.mark.asyncio
async def test_run_index_creation_workflow_no_source_files(mock_config_service: MagicMock) -> None:
    """Test the index creation workflow when no source files are found."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    with patch("pathlib.Path.exists", return_value=False):
        with patch("pathlib.Path.rglob", return_value=[]):
            results = await orchestration_service.run_index_creation_workflow(
                source_folders=["src"], index_directory="tests/temp_index", force_rebuild=True
            )

    assert results["status"] == "success"
    assert results["message"] == "No source code found to index."


@pytest.mark.asyncio
async def test_run_index_creation_workflow_llm_exception(mock_config_service: MagicMock) -> None:
    """Test the index creation workflow when LLM connector raises an exception."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    with patch("pathlib.Path.exists", return_value=False):
        with patch(
            "ai_unit_test.services.orchestration_service.LLMConnectorFactory.create_from_config_file",
            side_effect=Exception("Test Exception"),
        ):
            results = await orchestration_service.run_index_creation_workflow(
                source_folders=["src"], index_directory="tests/temp_index", force_rebuild=True
            )

    assert results["status"] == "error"
    assert results["error"] == "Test Exception"


@pytest.mark.asyncio
async def test_run_index_creation_workflow(mock_config_service: MagicMock) -> None:
    """Test the index creation workflow."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    with patch("pathlib.Path.exists", return_value=False):
        with patch(
            "ai_unit_test.core.factories.llm_factory.LLMConnectorFactory.create_from_config_file"
        ) as mock_llm_factory:
            mock_llm_connector = AsyncMock()
            mock_llm_factory.return_value = mock_llm_connector
            with patch(
                "ai_unit_test.core.factories.index_factory.IndexOrganizerFactory.create_from_config_file"
            ) as mock_index_factory:
                mock_index_organizer = AsyncMock()
                mock_index_factory.return_value = mock_index_organizer

                results = await orchestration_service.run_index_creation_workflow(
                    source_folders=["src"], index_directory="tests/temp_index", force_rebuild=True
                )

    assert "files_processed" in results
    assert "chunks_created" in results


@pytest.mark.asyncio
async def test_run_index_creation_workflow_index_exists(mock_config_service: MagicMock) -> None:
    """Test the index creation workflow when the index already exists."""
    orchestration_service = OrchestrationService()
    orchestration_service.config_service = mock_config_service

    with patch("pathlib.Path.exists", return_value=True):
        results = await orchestration_service.run_index_creation_workflow(
            source_folders=["src"], index_directory="tests/temp_index", force_rebuild=False
        )

    assert results["status"] == "skipped"
