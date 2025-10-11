"""Test LLM provider contracts."""

import os
from typing import Any

import pytest

from ai_unit_test.core.implementations.llm.huggingface_connector import HuggingFaceConnector
from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector
from ai_unit_test.core.interfaces.llm_connector import LLMRequest


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"),
    reason="OPENAI_API_KEY not set or not a real key",
)
class TestOpenAIContract:
    """Test OpenAI API contract."""

    async def test_openai_contract(self, openai_test_config: dict[str, Any]) -> None:
        """Test that the OpenAI API returns the expected response."""
        # Arrange
        config = {
            **openai_test_config,
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ["OPENAI_API_URL"] or "https://api.openai.com/v1",
        }
        async with OpenAIConnector(config) as connector:
            request = LLMRequest(
                system_message="You are a helpful assistant.",
                user_message="Say hello!",
                model=config["model"],
            )

            # Act
            response = await connector.generate_response(request)

            # Assert
            assert response.content
            # Check if the returned model contains the requested model name
            # (APIs may return versioned model names like gpt-5-nano-2025-08-07)
            assert config["model"] in response.model or response.model.startswith(config["model"])


@pytest.mark.skipif(not os.environ.get("HF_API_KEY"), reason="HF_API_KEY not set")
class TestHuggingFaceContract:
    """Test HuggingFace API contract."""

    @pytest.mark.asyncio
    async def test_huggingface_contract(self, huggingface_test_config: dict[str, Any]) -> None:
        """Test that the HuggingFace API returns the expected response."""
        # Arrange
        config = {
            **huggingface_test_config,
            "api_key": os.environ["HF_API_KEY"],
            "use_api": False,
            "model": huggingface_test_config.get("model_name", huggingface_test_config.get("model")),
        }
        async with HuggingFaceConnector(config) as connector:
            request = LLMRequest(
                system_message="You are a helpful assistant.",
                user_message="Say hello!",
                model=config["model"],
            )

            # Act
            response = await connector.generate_response(request)

            # Assert
            assert response.content
            assert response.model == config["model"]
