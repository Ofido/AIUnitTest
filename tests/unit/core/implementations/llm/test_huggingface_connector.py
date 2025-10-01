"""Test cases for the HuggingFace LLM connector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_unit_test.core.implementations.llm import huggingface_connector as hf
from ai_unit_test.core.implementations.llm.huggingface_connector import HuggingFaceConfig, HuggingFaceConnector
from ai_unit_test.core.interfaces.llm_connector import EmbeddingRequest, LLMRequest


@pytest.mark.asyncio
async def test_huggingface_generate_embeddings_api_success() -> None:
    """Test successful embedding generation using HuggingFace API."""
    # Arrange
    config = HuggingFaceConfig(api_key="test_api_key", use_api=True)
    connector = HuggingFaceConnector(config)

    # Mock the httpx.AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [[0.1, 0.2, 0.3]]

    mock_post = AsyncMock(return_value=mock_response)
    connector.api_client = MagicMock()
    connector.api_client.post = mock_post

    request = EmbeddingRequest(texts=["test text"], model="sentence-transformers/all-MiniLM-L6-v2", normalize=False)

    # Act
    response = await connector.generate_embeddings(request)

    # Assert
    assert response is not None
    assert response.embeddings.shape == (1, 3)
    assert response.embeddings[0][0] == 0.1
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_huggingface_generate_embeddings_api_failure() -> None:
    """Test failed embedding generation using HuggingFace API."""
    # Arrange
    config = HuggingFaceConfig(api_key="test_api_key", use_api=True)
    connector = HuggingFaceConnector(config)

    # Mock the httpx.AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"

    mock_post = AsyncMock(return_value=mock_response)
    connector.api_client = MagicMock()
    connector.api_client.post = mock_post

    request = EmbeddingRequest(texts=["test text"], model="sentence-transformers/all-MiniLM-L6-v2", normalize=False)

    # Act & Assert
    with pytest.raises(
        Exception, match="HuggingFace embedding request failed: API embedding request failed: Not Found"
    ):
        await connector.generate_embeddings(request)
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_huggingface_api_and_local_branches_and_embeddings_processing() -> None:
    """Test the HuggingFace API and local branches and embeddings processing."""
    # Test _initialize_api_client requires httpx
    original_httpx_flag = hf.HTTPX_AVAILABLE
    hf.HTTPX_AVAILABLE = False
    connector_api = HuggingFaceConnector(HuggingFaceConfig(api_key="k", use_api=True))
    with pytest.raises(Exception, match="httpx is required for HuggingFace API usage"):
        await connector_api._initialize_api_client()
    hf.HTTPX_AVAILABLE = original_httpx_flag

    # Test _initialize_local_model requires transformers
    original_transformers_flag = hf.TRANSFORMERS_AVAILABLE
    hf.TRANSFORMERS_AVAILABLE = False
    connector_local = HuggingFaceConnector(HuggingFaceConfig(use_api=False))
    with pytest.raises(Exception, match="transformers is required for local HuggingFace models"):
        await connector_local._initialize_local_model()
    hf.TRANSFORMERS_AVAILABLE = original_transformers_flag

    # Test _generate_api_response without api_client
    connector_no_client = HuggingFaceConnector(HuggingFaceConfig(api_key="k", use_api=True))
    connector_no_client.api_client = None  # type: ignore
    request = LLMRequest(system_message="SYS", user_message="USER", model="test_model")
    with pytest.raises(Exception, match="API client not initialized"):
        await connector_no_client._generate_api_response(request)

    # Test _generate_api_response returns generated_text on success
    connector_ok = HuggingFaceConnector(HuggingFaceConfig(api_key="k", use_api=True, model="mymodel"))
    connector_ok._initialize_api_client = AsyncMock()  # type: ignore
    await connector_ok.initialize()
    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = [{"generated_text": "hello world"}]
    connector_ok.api_client = MagicMock()
    connector_ok.api_client.post = AsyncMock(return_value=mock_response_ok)
    result = await connector_ok._generate_api_response(request)
    assert result == "hello world"
    connector_ok.api_client.post.assert_called_once()

    # Test _generate_api_response raises on non-200
    mock_response_err = MagicMock()
    mock_response_err.status_code = 400
    mock_response_err.text = "bad request"
    connector_ok.api_client.post = AsyncMock(return_value=mock_response_err)
    with pytest.raises(Exception, match="API request failed: bad request"):
        await connector_ok._generate_api_response(request)

    # Test _generate_local_response trimming prompt and using run_in_executor
    connector_local_resp = HuggingFaceConnector(HuggingFaceConfig(use_api=False))
    # Provide dummy pipeline and tokenizer objects (pipeline not actually called, run_in_executor will return result)
    connector_local_resp.pipeline = MagicMock()
    connector_local_resp.tokenizer = MagicMock()
    prompt = f"{request.system_message}\n\nUser: {request.user_message}\nAssistant:"
    generated_full = prompt + " generated reply"
    dummy_result = [{"generated_text": generated_full}]
    with patch("asyncio.get_event_loop") as get_loop_mock:
        get_loop_mock.return_value.run_in_executor = AsyncMock(return_value=dummy_result)
        local_out = await connector_local_resp._generate_local_response(request)
        assert local_out == "generated reply"
