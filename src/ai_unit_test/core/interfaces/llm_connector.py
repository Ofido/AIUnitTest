"""Abstract interface for LLM connectors."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

try:
    from typing import Self
except ImportError:
    from typing import Self


@dataclass
class LLMRequest:
    """Request object for LLM operations."""

    system_message: str
    user_message: str
    model: str
    temperature: float = 0.1
    max_tokens: int | None = None
    stream: bool = False


@dataclass
class LLMResponse:
    """Response object from LLM operations."""

    content: str
    usage: dict[str, int]  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    model: str
    finish_reason: str
    response_time_ms: int


@dataclass
class EmbeddingRequest:
    """Request object for embedding generation."""

    texts: list[str]
    model: str
    normalize: bool = True


@dataclass
class EmbeddingResponse:
    """Response object from embedding generation."""

    embeddings: np.ndarray
    usage: dict[str, int]
    model: str
    response_time_ms: int


@dataclass
class LLMUsage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMConfigProtocol(Protocol):
    """Configuration for LLM connectors."""

    model: str | None
    api_key: str | None
    timeout: int = 30


ConfigT = TypeVar("ConfigT", bound=LLMConfigProtocol)


class LLMConnector(ABC, Generic[ConfigT]):
    """Abstract base class for all LLM connectors."""

    config: ConfigT

    def __init__(self, config: dict[str, Any] | ConfigT) -> None:
        """Initialize connector with configuration."""
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        else:
            self.config = config
        self._initialized = False

    @abstractmethod
    def _create_config_from_dict(self, config: dict[str, Any]) -> ConfigT:
        """Create config object from dictionary."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the connector (async setup)."""
        pass

    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a single response from the LLM."""
        pass

    @abstractmethod
    def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate streaming response from the LLM."""
        pass

    @abstractmethod
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the connector is healthy and can make requests."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this connector."""
        pass

    @abstractmethod
    def get_connector_info(self) -> dict[str, Any]:
        """Get information about this connector."""
        pass

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True
        return self

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Async context manager exit."""
        # Override in implementations if cleanup is needed
        pass
