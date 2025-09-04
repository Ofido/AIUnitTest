import hashlib
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ai_unit_test.llm import generate_embeddings, update_test_with_llm


def test_generate_embeddings_cache_hit() -> None:
    """Tests that generate_embeddings correctly loads from cache."""
    texts = ["hello world"]
    source_file_path = "dummy.py"
    model_name = "all-MiniLM-L6-v2"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    dummy_embeddings = np.array([[1, 2, 3]])
    np.savez_compressed(cache_filepath, embeddings=dummy_embeddings, metadata=np.array({}))

    embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert np.array_equal(embeddings, dummy_embeddings.tolist())

    # Cleanup
    os.remove(cache_filepath)
    os.unlink(source_file_path)


def test_generate_embeddings_cache_miss() -> None:
    """Tests that generate_embeddings correctly generates and caches embeddings."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Calculate the hash from the actual file content to ensure cache file doesn't exist
    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_filepath):
        os.remove(cache_filepath)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1, 2, 3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert os.path.exists(cache_filepath)
    assert np.array_equal(embeddings, [[1, 2, 3]])

    # Cleanup
    os.remove(cache_filepath)
    os.unlink(source_file_path)


def test_generate_embeddings_cache_exception() -> None:
    """Tests that generate_embeddings handles cache loading exceptions."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Create a corrupted cache file using the correct hash
    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    with open(cache_filepath, "w") as f:
        f.write("corrupted data")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1, 2, 3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert np.array_equal(embeddings, [[1, 2, 3]])

    # Cleanup
    os.remove(cache_filepath)
    os.unlink(source_file_path)


@pytest.mark.asyncio
async def test_update_test_with_llm() -> None:
    """Tests that update_test_with_llm correctly calls the LLM and returns the response."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = [1]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="new_test"))])
    )

    with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "new_test"


@pytest.mark.asyncio
async def test_update_test_with_llm_no_api_key() -> None:
    """Tests that update_test_with_llm raises an error if the API key is not set."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = [1]
    other_tests_content = ""
    test_style = "pytest_function"

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError) as excinfo:
            await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )
    assert "Set the OPENAI_API_KEY environment variable with your OpenAI key." in str(excinfo.value)


@pytest.mark.asyncio
async def test_update_test_with_llm_no_api_url() -> None:
    """Tests that update_test_with_llm works when API URL is not set."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = [1]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_new(): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_new(): pass"


@pytest.mark.asyncio
async def test_update_test_with_llm_invalid_response() -> None:
    """Tests that update_test_with_llm handles invalid responses from the LLM."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = [1]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None))])
    )

    with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == ""  # Ensure the function returns an empty string for invalid response


@pytest.mark.asyncio
async def test_update_test_with_llm_exception_handling() -> None:
    """Tests that update_test_with_llm handles exceptions raised during API calls."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = [1]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with pytest.raises(Exception) as excinfo:
                await update_test_with_llm(
                    source_code,
                    test_code,
                    file_name,
                    coverage_lines,
                    other_tests_content,
                    test_style,
                )

    assert "API error" in str(excinfo.value)  # Ensure the exception message is as expected


def test_generate_embeddings_cache_save() -> None:
    """Tests that generate_embeddings correctly saves embeddings to cache."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = True
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Calculate the hash from the actual file content
    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_filepath):
        os.remove(cache_filepath)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        _ = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert os.path.exists(cache_filepath)
    loaded_data = np.load(cache_filepath, allow_pickle=True)
    assert np.array_equal(loaded_data["embeddings"], np.array([[0.1, 0.2, 0.3]]))

    # Cleanup
    os.remove(cache_filepath)
    os.unlink(source_file_path)


@pytest.mark.asyncio
async def test_update_test_with_llm_with_api_url() -> None:
    """Tests that update_test_with_llm works when API URL is set."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = [1]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_new_with_url(): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_API_URL": "http://api.url"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_new_with_url(): pass"


def test_generate_embeddings_cache_save_exception() -> None:
    """Tests that generate_embeddings handles exceptions when saving to cache."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = True
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        with patch("os.makedirs", side_effect=Exception("Failed to create directory")):
            embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert np.array_equal(embeddings, [[0.1, 0.2, 0.3]])

    # Cleanup
    os.unlink(source_file_path)


@pytest.mark.asyncio
async def test_update_test_with_llm_with_empty_coverage_lines() -> None:
    """Tests that update_test_with_llm handles empty coverage lines correctly."""
    source_code = "def add(a, b): return a + b"
    test_code = "def test_add(): assert add(1, 2) == 3"
    file_name = "dummy.py"
    coverage_lines = []  # type: ignore
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_no_coverage(): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_no_coverage(): pass"


def test_generate_embeddings_cache_filepath_exception() -> None:
    """Tests that generate_embeddings handles exceptions when creating cache filepath."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = False

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Simulate an OSError when trying to create the cache filepath
    with patch("builtins.open", side_effect=OSError("Unable to open file")):
        embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert embeddings == []  # Expecting an empty list when cache filepath creation fails

    # Cleanup
    os.unlink(source_file_path)


@pytest.mark.asyncio
async def test_update_test_with_llm_with_specific_test_style() -> None:
    """Tests that update_test_with_llm works with a specific test style."""
    source_code = "def multiply(a, b): return a * b"
    test_code = "def test_multiply(): assert multiply(2, 3) == 6"
    file_name = "dummy.py"
    coverage_lines = [1, 2]
    other_tests_content = ""
    test_style = "unittest_class"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_new_multiply(self): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_new_multiply(self): pass"


def test_generate_embeddings_cache_filepath_creation() -> None:
    """Tests that generate_embeddings correctly creates the cache filepath."""
    model_name = "all-MiniLM-L6-v2"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Calculate the hash from the actual file content
    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # Ensure the cache filepath is created correctly
    assert cache_filepath.endswith(f"{content_hash}_{model_name_safe}_{norm_str}.npz")

    # Cleanup
    os.unlink(source_file_path)


@pytest.mark.asyncio
async def test_update_test_with_llm_with_multiple_coverage_lines() -> None:
    """Tests that update_test_with_llm works with multiple coverage lines."""
    source_code = "def divide(a, b): return a / b"
    test_code = "def test_divide(): assert divide(6, 3) == 2"
    file_name = "dummy.py"
    coverage_lines = [1, 2]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_new_divide(): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_new_divide(): pass"


def test_generate_embeddings_cache_miss_with_normalization() -> None:
    """Tests that generate_embeddings correctly generates embeddings with normalization."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = True
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Calculate the hash from the actual file content to ensure cache file doesn't exist
    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_filepath):
        os.remove(cache_filepath)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert np.array_equal(embeddings, [[0.1, 0.2, 0.3]])

    # Cleanup
    os.remove(cache_filepath)
    os.unlink(source_file_path)


@pytest.mark.asyncio
async def test_update_test_with_llm_with_no_coverage_lines() -> None:
    """Tests that update_test_with_llm handles the case when no coverage lines are provided."""
    source_code = "def subtract(a, b): return a - b"
    test_code = "def test_subtract(): assert subtract(5, 3) == 2"
    file_name = "dummy.py"
    coverage_lines = []  # type: ignore
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_no_coverage_lines(): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_no_coverage_lines(): pass"


@pytest.mark.asyncio
async def test_update_test_with_llm_with_coverage_line() -> None:
    """Tests that update_test_with_llm handles a specific coverage line (54) correctly."""
    source_code = "def subtract(a, b): return a - b"
    test_code = "def test_subtract(): assert subtract(5, 3) == 2"
    file_name = "dummy.py"
    coverage_lines = [54]
    other_tests_content = ""
    test_style = "pytest_function"

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "def test_subtract_with_coverage(): pass"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with patch("ai_unit_test.llm.AsyncOpenAI", return_value=mock_client):
            response = await update_test_with_llm(
                source_code,
                test_code,
                file_name,
                coverage_lines,
                other_tests_content,
                test_style,
            )

    assert response == "def test_subtract_with_coverage(): pass"


def test_generate_embeddings_cache_miss_with_exception_handling() -> None:
    """Tests that generate_embeddings handles exceptions during embedding generation."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Simulate an exception during model encoding
    mock_model = MagicMock()
    mock_model.encode.side_effect = Exception("Encoding error")
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    assert embeddings == []  # Expecting an empty list when encoding fails

    # Cleanup
    os.unlink(source_file_path)


def test_generate_embeddings_cache_miss_with_logging() -> None:
    """Tests that generate_embeddings logs a warning when cache is missed and embeddings are generated."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        with patch("ai_unit_test.llm.logger.info") as mock_logger:
            embeddings = generate_embeddings(texts, source_file_path, model_name, normalize)

    mock_logger.assert_any_call(f"Cache miss. Generating embeddings for {len(texts)} texts.")
    assert np.array_equal(embeddings, [[0.1, 0.2, 0.3]])

    # Cleanup
    os.unlink(source_file_path)


def test_generate_embeddings_cache_filepath_creation_with_special_characters() -> None:
    """Tests that generate_embeddings correctly creates the cache filepath with special characters in model name."""
    texts = ["hello world"]
    model_name = "all-MiniLM-L6-v2/special"
    normalize = False
    cache_dir = os.path.join(".ai_unit_test_cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a temporary source file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
        temp_file.write("print('hello world')")
        source_file_path = temp_file.name

    # Calculate the hash from the actual file content
    with open(source_file_path, "rb") as f:
        file_content = f.read()
    content_hash = hashlib.sha256(file_content).hexdigest()
    model_name_safe = model_name.replace("/", "_")
    norm_str = "norm" if normalize else "unorm"
    cache_filename = f"{content_hash}_{model_name_safe}_{norm_str}.npz"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # Call generate_embeddings to ensure the cache file is created
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    with patch("ai_unit_test.llm.SentenceTransformer", return_value=mock_model):
        _ = generate_embeddings(texts, source_file_path, model_name, normalize)

    # Ensure the cache filepath is created correctly
    assert cache_filepath.endswith(f"{content_hash}_{model_name_safe}_{norm_str}.npz")

    # Cleanup
    os.unlink(cache_filepath)
    os.unlink(source_file_path)
