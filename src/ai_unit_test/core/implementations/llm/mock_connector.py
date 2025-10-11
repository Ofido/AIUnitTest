"""Mock LLM connector for testing."""

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ai_unit_test.core.exceptions import LLMConnectionError
from ai_unit_test.core.interfaces.llm_connector import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMConnector,
    LLMRequest,
    LLMResponse,
)


@dataclass
class MockConnectorConfig:
    """Configuration class for MockConnector."""

    api_key: str | None = None
    timeout: int = 30
    base_url: str | None = None
    organization: str | None = None
    model: str | None = None
    rate_limit: int = 60
    models_cache_ttl: float = 300.0
    temperature: float = 0.5
    max_retries: int = 3
    retry_delay: float = 1.0
    should_fail: bool = False
    response_delay: float = 0.1


class MockConnector(LLMConnector[MockConnectorConfig]):
    """Mock LLM connector for testing."""

    def __init__(self, config: dict[str, Any] | MockConnectorConfig) -> None:
        """
        Initialize the MockConnector.

        Args:
            config: Configuration for the mock connector.
        """
        super().__init__(config)
        self._should_fail = self.config.should_fail
        self._failure_mode = False

    def _create_config_from_dict(self, config: dict[str, Any]) -> MockConnectorConfig:
        return MockConnectorConfig(**config)

    async def initialize(self) -> None:
        """Initialize the mock connector."""
        if self._should_fail:
            raise LLMConnectionError("Failed to initialize mock connector")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a mock response."""
        if self._failure_mode:
            raise LLMConnectionError("Mock connector is in failure mode")

        start_time = time.time()
        await asyncio.sleep(self.config.response_delay)
        import ast
        import re

        response_time_ms = int((time.time() - start_time) * 1000)

        # Extrai o chunk de código fonte da mensagem do usuário
        match = re.search(r"<source_code_chunk>(.*?)</source_code_chunk>", request.user_message, re.DOTALL)
        source_code = match.group(1) if match else ""

        # Analisa o código fonte para encontrar funções
        test_functions = []
        try:
            tree = ast.parse(source_code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    # Gera uma função de teste simples para cada função detectada
                    test_func = f"def test_{func_name}():\n    assert True\n"
                    test_functions.append(test_func)
        except Exception:
            # Se não conseguir analisar, retorna um teste genérico
            test_functions.append("def test_dummy():\n    assert True\n")

        return LLMResponse(
            content="\n".join(test_functions),
            usage={"total_tokens": 10},
            model="mock-model",
            finish_reason="stop",
            response_time_ms=response_time_ms,
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate a mock stream of responses."""
        if self._failure_mode:
            raise LLMConnectionError("Mock connector is in failure mode")

        for i in range(3):
            await asyncio.sleep(0.1)
            yield f"Chunk {i} "

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate mock embeddings."""
        if self._failure_mode:
            raise LLMConnectionError("Mock connector is in failure mode")

        return EmbeddingResponse(
            embeddings=np.random.rand(len(request.texts), 384).astype(np.float32),
            usage={"total_tokens": 10},
            model="mock-model",
            response_time_ms=10,
        )

    async def health_check(self) -> bool:
        """Health check for the mock connector."""
        return not self._should_fail

    def get_available_models(self) -> list[str]:
        """Get available models for the mock connector."""
        return ["mock-model"]

    def get_connector_info(self) -> dict[str, Any]:
        """Get connector information for the mock connector."""
        return {
            "provider": "mock",
            "version": "1.0.0",
            "supports_streaming": True,
        }

    def set_failure_mode(self, should_fail: bool) -> None:
        """Set the failure mode of the mock connector."""
        self._should_fail = should_fail
        self._failure_mode = should_fail
