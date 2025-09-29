"""Test LLM connector interface compliance."""

from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import pytest

from ai_unit_test.core.exceptions import LLMConnectionError
from ai_unit_test.core.implementations.llm.huggingface_connector import HuggingFaceConnector
from ai_unit_test.core.implementations.llm.mock_connector import MockConnector, MockConnectorConfig
from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector
from ai_unit_test.core.interfaces.llm_connector import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMConnector,
    LLMRequest,
    LLMResponse,
)


class TestLLMConnectorInterface:
    """Test that all LLM connectors implement the interface correctly."""

    @pytest.mark.parametrize(
        ("connector_class", "config"),
        [
            (MockConnector, {"should_fail": False}),
            (OpenAIConnector, {"api_key": "test-key"}),
            (HuggingFaceConnector, {"model": "gpt2", "use_api": False}),
        ],
    )
    async def test_connector_interface_compliance(
        self, connector_class: type[LLMConnector[MockConnectorConfig]], config: dict[str, Any]
    ) -> None:
        """Test that connector implements all required interface methods."""
        # Test instantiation
        connector = connector_class(config)
        assert isinstance(connector, LLMConnector)

        # Test required methods exist
        required_methods = [
            "initialize",
            "generate_response",
            "generate_stream",
            "health_check",
            "get_available_models",
            "get_connector_info",
        ]

        for method_name in required_methods:
            assert hasattr(connector, method_name)
            method = getattr(connector, method_name)
            assert callable(method)

        # Test async context manager
        assert hasattr(connector, "__aenter__")
        assert hasattr(connector, "__aexit__")

    @pytest.mark.parametrize(
        ("connector_class", "config"),
        [
            (MockConnector, {"should_fail": False}),
        ],
    )
    async def test_generate_response_contract(
        self, connector_class: type[LLMConnector[MockConnectorConfig]], config: dict[str, Any]
    ) -> None:
        """Test generate_response method contract."""
        async with connector_class(config) as connector:
            request = LLMRequest(
                system_message="You are helpful.", user_message="Say hello", model="test-model", temperature=0.1
            )

            response = await connector.generate_response(request)

            # Validate response structure
            assert isinstance(response, LLMResponse)
            assert isinstance(response.content, str)
            assert isinstance(response.usage, dict)
            assert "total_tokens" in response.usage
            assert isinstance(response.model, str)
            assert isinstance(response.finish_reason, str)
            assert isinstance(response.response_time_ms, int)
            assert response.response_time_ms >= 0

    @pytest.mark.parametrize(
        ("connector_class", "config"),
        [
            (MockConnector, {"should_fail": False}),
        ],
    )
    async def test_generate_stream_contract(
        self, connector_class: type[LLMConnector[MockConnectorConfig]], config: dict[str, Any]
    ) -> None:
        """Test generate_stream method contract."""
        async with connector_class(config) as connector:
            request = LLMRequest(
                system_message="You are helpful.", user_message="Count to 3", model="test-model", temperature=0.1
            )

            chunks = []
            async for chunk in connector.generate_stream(request):
                assert isinstance(chunk, str)
                chunks.append(chunk)

            # Should have received some chunks
            assert len(chunks) > 0

            # Reconstruct full response
            full_response = "".join(chunks)
            assert len(full_response) > 0

    async def test_error_handling_contract(self) -> None:
        """Test error handling behavior."""
        # Test initialization failure
        connector = MockConnector({"should_fail": True})

        with pytest.raises(LLMConnectionError):
            await connector.initialize()

        # Test request failure
        failing_connector = MockConnector({"should_fail": False})
        await failing_connector.initialize()
        failing_connector.set_failure_mode(True)

        request = LLMRequest(system_message="Test", user_message="Test", model="test", temperature=0.1)

        with pytest.raises((LLMConnectionError, Exception)):
            await failing_connector.generate_response(request)

    async def test_health_check_contract(self) -> None:
        """Test health check behavior."""
        # Healthy connector
        async with MockConnector({"should_fail": False}) as connector:
            health = await connector.health_check()
            assert isinstance(health, bool)
            assert health is True

        # Unhealthy connector
        unhealthy_connector = MockConnector({"should_fail": True})
        health = await unhealthy_connector.health_check()
        assert health is False

    async def test_get_available_models_contract(self) -> None:
        """Test get_available_models behavior."""
        connector = MockConnector({"should_fail": False})
        models = connector.get_available_models()

        assert isinstance(models, list)
        assert all(isinstance(model, str) for model in models)
        assert len(models) > 0

    async def test_get_connector_info_contract(self) -> None:
        """Test get_connector_info behavior."""
        connector = MockConnector({"should_fail": False})
        info = connector.get_connector_info()

        assert isinstance(info, dict)
        assert "provider" in info
        assert "version" in info
        assert "supports_streaming" in info
        assert isinstance(info["supports_streaming"], bool)


class DummyConfig:
    """Dummy config class for testing."""

    def __init__(self) -> None:
        """Initialize dummy config."""
        self.model: str | None = None
        self.api_key: str | None = None
        self.timeout: int = 30
        self.converted: bool = False


class DummyConnector(LLMConnector[DummyConfig]):  # type: ignore
    """Dummy connector class for testing."""

    def __init__(self, config: dict[str, Any] | DummyConfig) -> None:
        """Initialize the dummy connector."""
        self.init_calls = 0
        super().__init__(config)

    def _create_config_from_dict(self, config: dict[str, Any]) -> DummyConfig:
        """Create config from dictionary."""
        cfg = DummyConfig()
        cfg.converted = True
        return cfg

    async def initialize(self) -> None:
        """Initialize the connector."""
        self.init_calls += 1

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        return LLMResponse(
            content="ok", usage={"total_tokens": 1}, model="dummy", finish_reason="stop", response_time_ms=0
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate streaming response."""
        yield "chunk-1"
        yield "chunk-2"

    async def health_check(self) -> bool:
        """Check connector health."""
        return True

    def get_available_models(self) -> list[str]:
        """Get available models."""
        return ["dummy-model"]

    def get_connector_info(self) -> dict[str, Any]:
        """Get connector information."""
        return {"provider": "dummy", "version": "0.1", "supports_streaming": True}

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings."""
        return EmbeddingResponse(
            embeddings=np.array([[0.1, 0.2]]), usage={"total_tokens": 1}, model="dummy", response_time_ms=0
        )


async def test_llm_connector_config_and_context_behavior() -> None:
    """Test LLM connector configuration and context behavior."""
    # Test that dict config is converted via _create_config_from_dict
    connector_from_dict = DummyConnector({"some": "value"})
    assert isinstance(connector_from_dict.config, DummyConfig)
    assert getattr(connector_from_dict.config, "converted", False) is True

    # Test that non-dict config is assigned directly
    custom_cfg = DummyConfig()
    connector_from_obj = DummyConnector(custom_cfg)
    assert connector_from_obj.config is custom_cfg

    # Test async context manager triggers initialize once and sets _initialized
    connector = DummyConnector({"k": "v"})
    assert connector._initialized is False
    async with connector as ctx:
        assert ctx is connector
        # initialize should have been called once
        assert connector.init_calls == 1
        assert connector._initialized is True

        # generate_response returns proper LLMResponse
        req = LLMRequest(system_message="sys", user_message="u", model="m", temperature=0.0)
        resp = await connector.generate_response(req)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "ok"

        # generate_stream yields chunks
        chunks = []
        async for c in connector.generate_stream(req):
            chunks.append(c)
        assert chunks == ["chunk-1", "chunk-2"]

        # health_check works
        h = await connector.health_check()
        assert h is True

        # get_available_models and get_connector_info
        models = connector.get_available_models()
        assert isinstance(models, list)
        assert "dummy-model" in models
        info = connector.get_connector_info()
        assert info.get("provider") == "dummy"

    # Re-entering context should not call initialize again
    before_calls = connector.init_calls
    async with connector:
        pass
    assert connector.init_calls == before_calls

    # If _initialized is manually set True before entering, initialize should not be called
    c2 = DummyConnector({"x": "y"})
    c2._initialized = True
    c2.init_calls = 0
    async with c2:
        pass
    assert c2.init_calls == 0
