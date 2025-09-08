"""HuggingFace LLM connector implementation."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ai_unit_test.core.exceptions import ConfigurationError, LLMConnectionError, LLMProviderError
from ai_unit_test.core.interfaces.llm_connector import LLMConnector, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# Optional dependency handling
if TYPE_CHECKING:
    import httpx

    HTTPX_AVAILABLE = True
else:
    try:
        import httpx

        HTTPX_AVAILABLE = True
    except ImportError:
        HTTPX_AVAILABLE = False
        httpx = None

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase, TextGenerationPipeline, pipeline

    TRANSFORMERS_AVAILABLE = True
else:
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerBase, TextGenerationPipeline, pipeline

        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        pipeline = None
        AutoTokenizer = None
        PreTrainedTokenizerBase = None
        TextGenerationPipeline = None


class HuggingFaceConnector(LLMConnector):
    """HuggingFace connector supporting both local and API models."""

    api_client: httpx.AsyncClient
    pipeline: TextGenerationPipeline
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model = None
        self.use_api = config.get("use_api", False)

    async def initialize(self) -> None:
        """Initialize HuggingFace model or API client."""
        try:
            if self.use_api:
                await self._initialize_api_client()
            else:
                await self._initialize_local_model()

            logger.info("HuggingFace connector initialized successfully")

        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize HuggingFace: {e}")

    async def _initialize_api_client(self) -> None:
        """Initialize API client for HuggingFace Inference API."""
        if not HTTPX_AVAILABLE:
            raise ConfigurationError("httpx is required for HuggingFace API usage")

        try:
            self.api_client = httpx.AsyncClient(
                base_url="https://api-inference.huggingface.co",
                headers={
                    "Authorization": f"Bearer {self.config.get('api_key', '')}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.get("timeout", 30),
            )

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize HuggingFace API client: {e}")

    async def _initialize_local_model(self) -> None:
        """Initialize local HuggingFace model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError("transformers is required for local HuggingFace models")

        try:
            model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")

            # Run in thread pool to avoid blocking
            self.pipeline = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline(
                    "text-generation",
                    model=model_name,
                    device=self.config.get("device", -1),  # -1 for CPU
                    max_length=self.config.get("max_length", 512),
                ),
            )

            self.tokenizer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(model_name),  # type: ignore[no-untyped-call]  # nosec
            )

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize local HuggingFace model: {e}")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using HuggingFace."""
        start_time = time.time()

        try:
            if self.use_api:
                content = await self._generate_api_response(request)
            else:
                content = await self._generate_local_response(request)

            response_time_ms = int((time.time() - start_time) * 1000)

            return LLMResponse(
                content=content,
                usage={"total_tokens": len(content.split())},  # Approximate
                model=self.config.get("model_name", "huggingface"),
                finish_reason="stop",
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"HuggingFace request failed: {e}")
            raise LLMProviderError(f"HuggingFace request failed: {e}")

    async def _generate_api_response(self, request: LLMRequest) -> str:
        """Generate response using HuggingFace API."""
        if not self.api_client:
            raise LLMConnectionError("API client not initialized")

        prompt = f"{request.system_message}\n\nUser: {request.user_message}\nAssistant:"

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens or 512,
                "return_full_text": False,
            },
        }

        model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
        response = await self.api_client.post(f"/models/{model_name}", json=payload)

        if response.status_code != 200:
            raise LLMProviderError(f"API request failed: {response.text}")

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")  # type: ignore

        return ""

    async def _generate_local_response(self, request: LLMRequest) -> str:
        """Generate response using local model."""
        if not self.pipeline or not self.tokenizer:
            raise LLMConnectionError("Local model not initialized")

        prompt = f"{request.system_message}\n\nUser: {request.user_message}\nAssistant:"

        # Run in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.pipeline(
                prompt,
                max_length=len(prompt.split()) + (request.max_tokens or 256),
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            ),
        )

        if result and len(result) > 0:
            generated_text = result[0]["generated_text"]
            # Extract only the new generated part
            return generated_text[len(prompt) :].strip()  # type: ignore[no-any-return]

        return ""

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate streaming response (simplified implementation)."""
        # For simplicity, generate full response and yield in chunks
        response = await self.generate_response(request)

        words = response.split()  # type: ignore
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay

    async def health_check(self) -> bool:
        """Check HuggingFace health."""
        try:
            test_request = LLMRequest(
                system_message="You are helpful.",
                user_message="Hi",
                model=self.config.get("model_name", "test"),
                temperature=0.1,
                max_tokens=5,
            )

            await self.generate_response(test_request)
            return True

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get available HuggingFace models."""
        return ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large", "gpt2", "gpt2-medium", "distilgpt2"]

    def get_connector_info(self) -> dict[str, Any]:
        """Get connector information."""
        return {
            "provider": "huggingface",
            "version": "1.0.0",
            "supports_streaming": True,
            "use_api": self.use_api,
            "model_name": self.config.get("model_name", "microsoft/DialoGPT-medium"),
        }

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        """Cleanup resources."""
        if hasattr(self, "api_client") and self.api_client:
            await self.api_client.aclose()
