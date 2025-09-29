"""HuggingFace LLM connector implementation."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
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


@dataclass
class HuggingFaceConfig:
    """Configuration class for HuggingFaceConnector."""

    api_key: str | None = None
    timeout: int = 30
    model: str | None = "stabilityai/stable-code-instruct-3b"
    use_api: bool = False
    device: int = -1
    max_length: int = 512
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 5


class HuggingFaceConnector(LLMConnector[HuggingFaceConfig]):
    """HuggingFace connector supporting both local and API models."""

    api_client: httpx.AsyncClient
    pipeline: TextGenerationPipeline
    tokenizer: PreTrainedTokenizerBase
    config: HuggingFaceConfig

    def _create_config_from_dict(self, config: dict[str, Any]) -> HuggingFaceConfig:
        return HuggingFaceConfig(**config)

    def __init__(self, config: dict[str, Any] | HuggingFaceConfig) -> None:
        """
        Initialize the HuggingFaceConnector.

        Args:
            config: Configuration for the HuggingFace connector.
        """
        super().__init__(config)
        self.model = None
        self.use_api = self.config.use_api

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
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize HuggingFace API client: {e}")

    async def _initialize_local_model(self) -> None:
        """Initialize local HuggingFace model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError("transformers is required for local HuggingFace models")

        try:
            model_name = self.config.model or "stabilityai/stable-code-instruct-3b"
            # Run in thread pool to avoid blocking
            self.pipeline = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=self.config.model,
                    device=self.config.device,
                    truncation=True,
                    do_sample=getattr(self.config, "do_sample", True),
                    temperature=getattr(self.config, "temperature", 1.0),
                    top_p=getattr(self.config, "top_p", 0.95),
                    repetition_penalty=getattr(self.config, "repetition_penalty", 1.0),
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
                model=self.config.model or "stabilityai/stable-code-instruct-3b",
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

        model_name = self.config.model or "stabilityai/stable-code-instruct-3b"
        response = await self.api_client.post(f"/models/{model_name}", json=payload)

        if response.status_code != 200:
            raise LLMProviderError(f"API request failed: {response.text}")

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")  # type: ignore[no-any-return]

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
                max_new_tokens=request.max_tokens or 256,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            ),
        )

        if result and len(result) > 0:
            generated_text = result[0]["generated_text"]
            # Extract only the new generated part
            return str(generated_text[len(prompt) :].strip())

        return ""

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate streaming response (simplified implementation)."""
        # For simplicity, generate full response and yield in chunks
        response = await self.generate_response(request)

        words = response.content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay

    async def _generate_embeddings_api(self, request: EmbeddingRequest) -> np.ndarray:
        if not self.api_client:
            raise LLMConnectionError("API client not initialized")
        model_name = self.config.model or "sentence-transformers/all-MiniLM-L6-v2"
        payload = {
            "inputs": request.texts,
            "options": {"wait_for_model": True},
        }
        response = await self.api_client.post(f"/pipeline/feature-extraction/{model_name}", json=payload)
        if response.status_code != 200:
            raise LLMProviderError(f"API embedding request failed: {response.text}")
        result = response.json()
        # result: List[List[List[float]]] if multiple sentences, else List[List[float]]
        # Always return shape (n_texts, embedding_dim)
        if isinstance(result, list) and isinstance(result[0], list) and isinstance(result[0][0], float):
            # Single text, shape (embedding_dim,)
            return np.array(result, dtype=np.float32)

        return np.array(result, dtype=np.float32)

    async def _generate_embeddings_local(self, request: EmbeddingRequest) -> np.ndarray:
        # Local: use appropriate model and robust pooling
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError("transformers is required for local HuggingFace embeddings")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # Ensure the model belongs to the sentence-transformers family
        if "sentence-transformers" not in model_name:
            raise LLMProviderError(
                f"The model '{model_name}' is not compatible with embedding generation. "
                "Use a model from the 'sentence-transformers' family for embeddings, "
                "for example 'sentence-transformers/all-MiniLM-L6-v2'."
            )
        # Load feature-extraction pipeline
        feature_pipeline = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline(
                "feature-extraction",
                model=model_name,
                tokenizer=model_name,
                device=self.config.device,
            ),
        )
        # Load tokenizer for truncation
        tokenizer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(model_name),  # type: ignore[no-untyped-call] # nosec
        )
        # Truncate each text to 512 tokens
        truncated_texts = []
        for text in request.texts:
            tokens = tokenizer.encode(text, max_length=512, truncation=True)
            truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        # Generate embeddings
        embeddings_list = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: feature_pipeline(truncated_texts),
        )  # embeddings_list: List[List[List[float]]] (batch, tokens, dim) or List[List[float]]
        # Robust pooling: always average tokens per sentence
        embeddings = []
        for e in embeddings_list:
            arr = np.array(e)
            # Fix shape (1, N, D) to (N, D)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(axis=0)
            if arr.ndim == 2:
                # tokens x dim
                pooled = arr.mean(axis=0)
            elif arr.ndim == 1:
                # already an embedding
                pooled = arr
            else:
                raise LLMProviderError(f"Unexpected embedding format: shape {arr.shape}")
            embeddings.append(pooled)
        return np.stack(embeddings).astype(np.float32)

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using HuggingFace."""
        start_time = time.time()

        try:
            logger.info(f"Generating embeddings for {len(request.texts)} texts. use_api={self.use_api}")
            if self.use_api:
                embeddings = await self._generate_embeddings_api(request)
            else:
                embeddings = await self._generate_embeddings_local(request)

            # Normalização opcional
            if request.normalize:
                from sklearn.preprocessing import normalize  # type: ignore[import-untyped]

                embeddings = normalize(embeddings, norm="l2", axis=1)

            response_time_ms = int((time.time() - start_time) * 1000)
            usage = {"num_texts": len(request.texts), "embedding_dim": int(embeddings.shape[1])}
            return EmbeddingResponse(
                embeddings=embeddings,
                usage=usage,
                model=self.config.model or "sentence-transformers/all-MiniLM-L6-v2",
                response_time_ms=response_time_ms,
            )
        except Exception as e:
            logger.error(f"HuggingFace embedding request failed: {e}")
            raise LLMProviderError(f"HuggingFace embedding request failed: {e}")

    async def health_check(self) -> bool:
        """Check HuggingFace health."""
        try:
            test_request = LLMRequest(
                system_message="You are helpful.",
                user_message="Hi",
                model=self.config.model or "stabilityai/stable-code-instruct-3b",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            await self.generate_response(test_request)
            return True

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get available HuggingFace models."""
        return [
            "stabilityai/stable-code-instruct-3b",
            "microsoft/DialoGPT-large",
            "gpt2",
            "gpt2-medium",
            "distilgpt2",
        ]

    def get_connector_info(self) -> dict[str, Any]:
        """Get connector information."""
        return {
            "provider": "huggingface",
            "version": "1.0.0",
            "supports_streaming": True,
            "use_api": self.use_api,
            "model_name": self.config.model or "stabilityai/stable-code-instruct-3b",
        }

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        """Cleanup resources."""
        if hasattr(self, "api_client") and self.api_client:
            await self.api_client.aclose()
