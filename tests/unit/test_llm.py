import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_unit_test.llm import update_test_with_llm


@pytest.fixture(autouse=True)
def mock_openai_api_key() -> Generator[None]:
    """
    Mocks the OPENAI_API_KEY environment variable.
    """
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        # Ensure OPENAI_API_URL is not set, as it can interfere with tests
        if "OPENAI_API_URL" in os.environ:
            del os.environ["OPENAI_API_URL"]
        yield


@patch("ai_unit_test.llm.AsyncOpenAI")
async def test_update_test_with_llm_success(mock_async_openai: MagicMock) -> None:
    """
    Tests that update_test_with_llm successfully calls the OpenAI API and returns the content.
    """
    # Mock the AsyncOpenAI client and its response
    mock_client_instance = mock_async_openai.return_value
    mock_create = AsyncMock()
    mock_create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="updated test content"))])
    mock_client_instance.chat.completions.create = mock_create

    source_code = "def func(): pass"
    test_code = "import pytest"
    file_name = "test_file.py"
    coverage_lines = [1, 2]
    other_tests_content = ""

    updated_content = await update_test_with_llm(
        source_code, test_code, file_name, coverage_lines, other_tests_content, "pytest_function"
    )

    assert updated_content == "updated test content"
    mock_async_openai.assert_called_once_with(api_key="test_key", base_url=None)
    mock_client_instance.chat.completions.create.assert_called_once()


@patch("ai_unit_test.llm.AsyncOpenAI")
async def test_update_test_with_llm_api_error(mock_async_openai: MagicMock) -> None:
    """
    Tests that update_test_with_llm raises an exception when the OpenAI API call fails.
    """
    mock_client_instance = mock_async_openai.return_value
    mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

    source_code = "def func(): pass"
    test_code = "import pytest"
    file_name = "test_file.py"
    coverage_lines = [1, 2]
    other_tests_content = ""

    with pytest.raises(Exception, match="API Error"):
        await update_test_with_llm(
            source_code, test_code, file_name, coverage_lines, other_tests_content, "pytest_function"
        )


async def test_update_test_with_llm_no_api_key() -> None:
    """
    Tests that update_test_with_llm raises a RuntimeError when OPENAI_API_KEY is not set.
    """
    # Temporarily remove the OPENAI_API_KEY from the environment
    with patch.dict(os.environ, {}, clear=True):
        source_code = "def func(): pass"
        test_code = "import pytest"
        file_name = "test_file.py"
        coverage_lines = [1, 2]
        other_tests_content = ""

        with pytest.raises(RuntimeError, match="Set the OPENAI_API_KEY environment variable with your OpenAI key."):
            await update_test_with_llm(
                source_code, test_code, file_name, coverage_lines, other_tests_content, "pytest_function"
            )
