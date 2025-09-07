"""Abstract interface for LLM connectors."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any


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
class LLMUsage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMConnector(ABC):
    """Abstract base class for all LLM connectors."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize connector with configuration."""
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the connector (async setup)."""
        pass

    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a single response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate streaming response from the LLM."""
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

    async def __aenter__(self) -> "LLMConnector":
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
