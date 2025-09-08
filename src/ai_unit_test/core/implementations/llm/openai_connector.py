"""OpenAI LLM connector implementation."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ai_unit_test.core.exceptions import ConfigurationError, LLMConnectionError, LLMProviderError
from ai_unit_test.core.interfaces.llm_connector import LLMConnector, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

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


class OpenAIConnector(LLMConnector):
    """OpenAI API connector implementation."""

    client: AsyncOpenAI
    _rate_limiter: "RateLimiter"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        if not OPENAI_AVAILABLE:
            raise ConfigurationError("OpenAI library is not available. Please install it with: pip install openai")

    async def initialize(self) -> None:
        """Initialize OpenAI client and rate limiter."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationError("OpenAI API key is required")

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.get("base_url"),
                organization=self.config.get("organization"),
                timeout=self.config.get("timeout", 30),
            )

            # Initialize rate limiter
            self._rate_limiter = RateLimiter(requests_per_minute=self.config.get("rate_limit", 60))

            logger.info("OpenAI connector initialized successfully")

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

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            if not self.client:
                return False

            # Simple test request
            test_request = LLMRequest(
                system_message="You are a test.",
                user_message="Say 'OK'",
                model=self.config.get("model", "gpt-3.5-turbo"),
                temperature=0.1,
                max_tokens=1,
            )

            await self.generate_response(test_request)
            return True

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get available OpenAI models."""
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

    def get_connector_info(self) -> dict[str, Any]:
        """Get connector information."""
        return {
            "provider": "openai",
            "version": "1.0.0",
            "supports_streaming": True,
            "rate_limit": self.config.get("rate_limit", 60),
            "default_model": self.config.get("model", "gpt-4o-mini"),
        }

    async def _make_request_with_retry(self, request: LLMRequest) -> ChatCompletion:
        """Make request with exponential backoff retry."""
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_delay", 1.0)

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
