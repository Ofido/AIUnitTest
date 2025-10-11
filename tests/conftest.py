"""Global test configuration and fixtures."""

import asyncio
import shutil
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer
from ai_unit_test.core.implementations.llm.mock_connector import MockConnector


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_pyproject_config() -> dict[str, Any]:
    """Sample pyproject.toml configuration."""
    return {
        "tool": {
            "coverage": {"run": {"source": ["src"], "data_file": ".coverage"}},
            "pytest": {"ini_options": {"testpaths": ["tests"]}},
            "ai-unit-test": {
                "llm": {"provider": "mock", "model": "mock-model", "temperature": 0.1},
                "indexing": {"backend": "memory", "index_directory": "data/test_index"},
                "test-patterns": ["test_*.py", "*_test.py"],
            },
        }
    }


@pytest.fixture
async def mock_llm_connector() -> AsyncGenerator[MockConnector]:
    """Create and initialize a mock LLM connector."""
    connector = MockConnector({"should_fail": False})
    await connector.initialize()
    yield connector
    await connector.__aexit__(None, None, None)


@pytest.fixture
async def memory_index_organizer() -> AsyncGenerator[InMemoryIndexOrganizer]:
    """Create and initialize an in-memory index organizer."""
    organizer = InMemoryIndexOrganizer({"max_documents": 1000})
    yield organizer
    await organizer.__aexit__(None, None, None)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Sample embeddings for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.random((10, 384)).astype(np.float32)


@pytest.fixture
def sample_metadata() -> list[dict[str, Any]]:
    """Sample metadata for testing."""
    return [
        {
            "file_path": f"src/module_{i}.py",
            "function_name": f"function_{i}",
            "line_start": i * 10,
            "line_end": i * 10 + 5,
            "code_snippet": f"def function_{i}():\n    return {i}",
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_source_file(temp_dir: Path) -> Path:
    """Create a sample source file for testing."""
    source_file = temp_dir / "sample_module.py"
    source_file.write_text(
        """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

class Calculator:
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(self, base, exponent):
        return base ** exponent
"""
    )
    return source_file


@pytest.fixture
def sample_test_file(temp_dir: Path) -> Path:
    """Create a sample test file for testing."""
    test_file = temp_dir / "test_sample_module.py"
    test_file.write_text(
        """
import pytest
from sample_module import add, multiply, Calculator

def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(2, 3) == 6

class TestCalculator:
    def test_divide(self):
        calc = Calculator()
        assert calc.divide(6, 2) == 3

    def test_divide_by_zero(self):
        calc = Calculator()
        with pytest.raises(ValueError):
            calc.divide(5, 0)
"""
    )
    return test_file


@pytest.fixture
def sample_coverage_data(temp_dir: Path) -> dict[str, list[int]]:
    """Sample coverage data for testing."""
    return {str(temp_dir / "sample_module.py"): [8, 12, 15]}  # Missing lines


# Performance test fixtures
@pytest.fixture
def performance_config() -> dict[str, Any]:
    """Configure performance tests."""
    return {"max_response_time_ms": 5000, "max_memory_usage_mb": 500, "max_cli_startup_time_ms": 2000}


# Contract test fixtures
@pytest.fixture
def openai_test_config() -> dict[str, Any]:
    """Configure OpenAI contract tests."""
    return {"api_key": "test-key", "model": "gpt-5-nano", "max_tokens": 100, "temperature": 0.1}


@pytest.fixture
def huggingface_test_config() -> dict[str, Any]:
    """Configure HuggingFace contract tests."""
    return {"model": "stabilityai/stable-code-instruct-3b", "use_api": False, "device": -1}  # CPU


# Async test helpers
@pytest.fixture
async def async_test_timeout() -> float:
    """Provide timeout for async tests."""
    return 30.0  # 30 seconds


@pytest.fixture
def mock_logger_error() -> Generator[Any]:
    """Mock logger error for testing."""
    with patch("ai_unit_test.core.utils.logger.logger.error") as mock:
        yield mock


# Test data validation helpers
def validate_llm_response(response: Any) -> None:  # noqa: ANN401
    """Validate LLM response structure."""
    required_fields = ["content", "usage", "model", "finish_reason", "response_time_ms"]
    for field in required_fields:
        assert hasattr(response, field), f"Missing field: {field}"

    assert isinstance(response.content, str)
    assert isinstance(response.usage, dict)
    assert "total_tokens" in response.usage


def validate_search_results(results: Any) -> None:  # noqa: ANN401
    """Validate search results structure."""
    assert isinstance(results, list)
    for result in results:
        assert hasattr(result, "metadata")
        assert hasattr(result, "score")
        assert hasattr(result, "document_id")
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
