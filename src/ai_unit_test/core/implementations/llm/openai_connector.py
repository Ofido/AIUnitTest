"""OpenAI LLM connector implementation."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ai_unit_test.core.exceptions import ConfigurationError, LLMConnectionError, LLMProviderError
from ai_unit_test.core.interfaces.llm_connector import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMConnector,
    LLMRequest,
    LLMResponse,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion

    OPENAI_AVAILABLE = True
else:
    # Optional dependency handling
    try:
        from openai import AsyncOpenAI
        from openai.types.chat import ChatCompletion

        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        AsyncOpenAI = None
        ChatCompletion = None

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConnectorConfig:
    """Configuration class for OpenAIConnector."""

    api_key: str | None = None
    timeout: int = 30
    base_url: str | None = None
    organization: str | None = None
    model: str | None = None
    rate_limit: int = 60
    models_cache_ttl: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Permite passar configurações extras se necessário
    extra: dict[str, Any] = field(default_factory=dict)


class OpenAIConnector(LLMConnector[OpenAIConnectorConfig]):
    """OpenAI API connector implementation."""

    client: AsyncOpenAI
    _rate_limiter: "RateLimiter"
    config: OpenAIConnectorConfig

    def _create_config_from_dict(self, config: dict[str, Any]) -> OpenAIConnectorConfig:
        return OpenAIConnectorConfig(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            organization=config.get("organization"),
            timeout=config.get("timeout", 30),
            rate_limit=config.get("rate_limit", 60),
            model=config.get("model"),
            models_cache_ttl=float(config.get("models_cache_ttl", 300)),
            max_retries=config.get("max_retries", 3),
            retry_delay=float(config.get("retry_delay", 1.0)),
            extra={
                k: v
                for k, v in config.items()
                if k
                not in {
                    "api_key",
                    "base_url",
                    "organization",
                    "timeout",
                    "rate_limit",
                    "model",
                    "models_cache_ttl",
                    "max_retries",
                    "retry_delay",
                }
            },
        )

    def __init__(self, config: OpenAIConnectorConfig | dict[str, Any]) -> None:
        """
        Initialize the OpenAIConnector.

        Args:
            config: Configuration for the OpenAI connector.
        """
        super().__init__(config)  # Mantém compatibilidade, mas ignora config original

        if not getattr(self.config, "api_key", None):
            raise ConfigurationError("OpenAI API key is required for connector initialization.")

        if not OPENAI_AVAILABLE:
            raise ConfigurationError("OpenAI library is not available. Please install it with: pip install openai")

        # cache para modelos disponíveis
        self._available_models_cache: list[str] | None = None
        self._models_cache_ts: float | None = None
        self._models_cache_ttl = self.config.models_cache_ttl
        self._model_priority_chat = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-5-nano"]
        self._model_priority_embeddings = ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
        self._model_priority_streaming = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-5-nano"]

    async def initialize(self) -> None:
        """Initialize OpenAI client and rate limiter."""
        try:
            api_key = self.config.api_key
            if not api_key:
                raise ConfigurationError("OpenAI API key is required")

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
                timeout=self.config.timeout,
            )

            # Initialize rate limiter
            self._rate_limiter = RateLimiter(requests_per_minute=self.config.rate_limit)

            logger.info("OpenAI connector initialized successfully")

            # Atualiza cache de modelos após inicialização
            try:
                await self._refresh_available_models(force=True)
            except Exception:
                logger.debug("Could not refresh model list at init", exc_info=True)

        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize OpenAI client: {e}")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from OpenAI."""
        if not self.client:
            raise LLMConnectionError("Connector not initialized")

        await self._rate_limiter.acquire()

        start_time = time.time()

        try:
            response = await self._make_request_with_retry(request)

            response_time_ms = int((time.time() - start_time) * 1000)
            usage = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage=usage,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise LLMProviderError(f"OpenAI request failed: {e}")

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate streaming response from OpenAI."""
        if not self.client:
            raise LLMConnectionError("Connector not initialized")

        await self._rate_limiter.acquire()

        try:
            stream = await self.client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "user", "content": request.user_message},
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming request failed: {e}")
            raise LLMProviderError(f"OpenAI streaming failed: {e}")

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        if not self.client:
            raise LLMConnectionError("Connector not initialized")

        await self._rate_limiter.acquire()

        start_time = time.time()

        try:
            response = await self.client.embeddings.create(
                input=request.texts,
                model=request.model,
            )

            response_time_ms = int((time.time() - start_time) * 1000)
            embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)

            if request.normalize:
                from sklearn.preprocessing import normalize  # type: ignore[import-untyped]

                embeddings = normalize(embeddings, norm="l2", axis=1)

            usage = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return EmbeddingResponse(
                embeddings=embeddings,
                usage=usage,
                model=response.model,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"OpenAI embedding request failed: {e}")
            raise LLMProviderError(f"OpenAI embedding request failed: {e}")

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            if not self.client:
                return False

            # Simple test request
            test_request = LLMRequest(
                system_message="You are a test.",
                user_message="Say 'OK'",
                model=self.config.model or "gpt-5-nano",
                temperature=0.1,
                max_tokens=1,
            )

            await self.generate_response(test_request)
            return True

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Return available models (cache or fallback)."""
        if self._available_models_cache:
            return list(self._available_models_cache)

        # fallback estático
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-5-nano",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

    async def _refresh_available_models(self, force: bool = False) -> list[str]:
        """Atualiza cache de modelos via API."""
        now = time.time()
        if not force and self._available_models_cache and self._models_cache_ts:
            if now - self._models_cache_ts < self._models_cache_ttl:
                return list(self._available_models_cache)

        if not getattr(self, "client", None):
            return self.get_available_models()

        try:
            resp = await self.client.models.list()
            models = [m.id for m in getattr(resp, "data", []) if getattr(m, "id", None)]
            if models:
                self._available_models_cache = models
                self._models_cache_ts = now
                return list(models)
        except Exception as e:
            logger.debug("Failed to list OpenAI models: %s", e, exc_info=True)

        return self.get_available_models()

    async def select_model(self, capability: str = "chat") -> str:
        """
        Seleciona o melhor modelo automaticamente.

        capability: "chat" | "embeddings" | "streaming"
        """
        # preferir config explícita
        cfg_model = self.config.model
        if cfg_model:
            return cfg_model

        models = await self._refresh_available_models()

        priority: list[str]
        if capability == "embeddings":
            priority = self._model_priority_embeddings
        elif capability == "streaming":
            priority = self._model_priority_streaming
        else:
            priority = self._model_priority_chat

        for p in priority:
            if p in models:
                return p

        if models:
            return models[0]
        return "gpt-5-nano"

    def get_connector_info(self) -> dict[str, Any]:
        """Get connector information."""
        return {
            "provider": "openai",
            "version": "1.0.0",
            "supports_streaming": True,
            "rate_limit": self.config.rate_limit,
            "default_model": self.config.model or "gpt-4o-mini",
        }

    async def _make_request_with_retry(self, request: LLMRequest) -> ChatCompletion:
        """Make request with exponential backoff retry."""
        max_retries = self.config.max_retries
        base_delay = self.config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                return await self.client.chat.completions.create(
                    model=request.model,
                    messages=[
                        {"role": "system", "content": request.system_message},
                        {"role": "user", "content": request.user_message},
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            except Exception as e:
                if attempt == max_retries:
                    raise

                delay = base_delay * (2**attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        raise


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_minute: int) -> None:
        """
        Initialize the RateLimiter.

        Args:
            requests_per_minute: The maximum number of requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self.requests: list[float] = []

    async def acquire(self) -> None:
        """Acquire rate limit slot."""
        now = time.time()

        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until oldest request is more than 1 minute old
            wait_time = 60 - (now - self.requests[0]) + 0.1
            await asyncio.sleep(wait_time)

        self.requests.append(now)
