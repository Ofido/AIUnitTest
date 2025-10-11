"""Test CLI coverage commands."""

import time
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer import Context, Exit

from ai_unit_test.cli import create_index, generate_tests, health_check
from ai_unit_test.services.orchestration_service import (
    ConfigHealth,
    HealthStatus,
    HealthStatusChecks,
    IndexHealth,
    LlmHealth,
)


@pytest.fixture
def mock_orchestration_service() -> Generator[MagicMock]:
    """Mock the orchestration service."""
    with patch("ai_unit_test.services.orchestration_service.OrchestrationService") as MockOrchestrationService:
        mock_service = MockOrchestrationService.return_value
        # Use MagicMock instead of AsyncMock since we mock asyncio.run in each test
        mock_service.run_test_generation_workflow = MagicMock()
        mock_service.run_index_creation_workflow = MagicMock()
        mock_service.run_health_check_workflow = MagicMock()
        yield mock_service


@pytest.fixture
def mock_context(mock_orchestration_service: MagicMock) -> Context:
    """Mock the Typer context."""
    ctx = MagicMock(spec=Context)
    ctx.obj = {"orchestration_service": mock_orchestration_service}
    return ctx


def test_generate_tests_error_status(mock_context: Context, mock_orchestration_service: MagicMock) -> None:
    """Test the test generation workflow when an error occurs."""
    result = {
        "status": "error",
        "files_processed": 0,
        "tests_generated": 0,
        "errors": ["Test error"],
        "workflow_duration_seconds": 0.0,
        "file_results": {},
    }
    mock_orchestration_service.run_test_generation_workflow.return_value = result

    with patch("ai_unit_test.cli.asyncio.run", return_value=result):
        with pytest.raises(Exit) as excinfo:
            generate_tests(mock_context, auto=True)
        assert excinfo.value.exit_code == 1


def test_generate_tests_partial_success_status(mock_context: Context, mock_orchestration_service: MagicMock) -> None:
    """Test the test generation workflow when a partial success occurs."""
    result = {
        "status": "partial_success",
        "files_processed": 1,
        "tests_generated": 1,
        "file_results": {"file1.py": {"test_generated": True, "status": "success"}},
        "workflow_duration_seconds": 0.0,
    }
    mock_orchestration_service.run_test_generation_workflow.return_value = result

    with patch("ai_unit_test.cli.asyncio.run", return_value=result):
        with pytest.raises(Exit) as excinfo:
            generate_tests(mock_context, auto=True)
        assert excinfo.value.exit_code == 2


def test_generate_tests_exception_handling(mock_context: Context) -> None:
    """Test the test generation workflow exception handling."""
    with patch("ai_unit_test.cli.asyncio.run", side_effect=Exception("Async error")):
        with pytest.raises(Exit) as excinfo:
            generate_tests(mock_context, auto=True)
        assert excinfo.value.exit_code == 1


def test_create_index_error_status(mock_context: Context, mock_orchestration_service: MagicMock) -> None:
    """Test the index creation workflow when an error occurs."""
    result = {
        "status": "error",
        "error": "Index creation failed",
        "workflow_duration_seconds": 0.0,
    }
    mock_orchestration_service.run_index_creation_workflow.return_value = result

    # Also need to mock the asyncio.run call
    with patch("ai_unit_test.cli.asyncio.run", return_value=result):
        with pytest.raises(Exit) as excinfo:
            create_index(mock_context, folders=["src"])
        assert excinfo.value.exit_code == 1


def test_create_index_exception_handling(mock_context: Context) -> None:
    """Test the index creation workflow exception handling."""
    with patch("ai_unit_test.cli.asyncio.run", side_effect=Exception("Async index error")):
        with pytest.raises(Exit) as excinfo:
            create_index(mock_context, folders=["src"])
        assert excinfo.value.exit_code == 1


def test_health_check_unhealthy_status(mock_context: Context, mock_orchestration_service: MagicMock) -> None:
    """Test the health check workflow when the service is unhealthy."""
    result = HealthStatus(
        status="unhealthy",
        timestamp=time.time(),
        checks=HealthStatusChecks(
            config=ConfigHealth(healthy=True, pyproject_loaded=True, environment=MagicMock()),
            llm=LlmHealth(healthy=False, error="LLM down"),
            indexing=IndexHealth(healthy=True, available_backends=["faiss"]),
        ),
    )
    mock_orchestration_service.run_health_check_workflow = MagicMock(return_value=result)

    with patch("ai_unit_test.cli.asyncio.run", return_value=result):
        with pytest.raises(Exit) as excinfo:
            health_check(mock_context)
        assert excinfo.value.exit_code == 1


def test_health_check_error_status(mock_context: Context, mock_orchestration_service: MagicMock) -> None:
    """Test the health check workflow when an error occurs."""
    result = HealthStatus(
        status="error",
        timestamp=time.time(),
        checks=HealthStatusChecks(
            config=ConfigHealth(healthy=True, pyproject_loaded=True, environment=MagicMock()),
            llm=LlmHealth(healthy=False, error="Critical error"),
            indexing=IndexHealth(healthy=True, available_backends=["faiss"]),
        ),
    )
    # Make sure we replace the AsyncMock with MagicMock
    mock_orchestration_service.run_health_check_workflow = MagicMock(return_value=result)

    with patch("ai_unit_test.cli.asyncio.run", return_value=result):
        with pytest.raises(Exit) as excinfo:
            health_check(mock_context)
        assert excinfo.value.exit_code == 2


def test_health_check_exception_handling(mock_context: Context) -> None:
    """Test the health check workflow exception handling."""
    with patch("ai_unit_test.cli.asyncio.run", side_effect=Exception("Async health error")):
        with pytest.raises(Exit) as excinfo:
            health_check(mock_context)
        assert excinfo.value.exit_code == 1
