"""Mock LLM connector for testing."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ai_unit_test.core.interfaces.llm_connector import LLMConnector, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class MockConnector(LLMConnector):
    """Mock LLM connector for testing purposes."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.call_count = 0
        self.should_fail = config.get("should_fail", False)
        self.response_delay = config.get("response_delay", 0.1)

    async def initialize(self) -> None:
        """Initialize mock connector."""
        if self.should_fail:
            raise ConnectionError("Mock connector configured to fail")

        logger.info("Mock connector initialized")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response."""
        self.call_count += 1

        if self.should_fail:
            raise ConnectionError("Mock connector configured to fail")

        # Simulate API delay
        await asyncio.sleep(self.response_delay)

        # Generate deterministic response based on request
        content = self._generate_mock_content(request)

        return LLMResponse(
            content=content,
            usage={
                "prompt_tokens": len(request.user_message.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(request.user_message.split()) + len(content.split()),
            },
            model="mock-model",
            finish_reason="stop",
            response_time_ms=int(self.response_delay * 1000),
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str]:
        """Generate mock streaming response."""
        content = self._generate_mock_content(request)
        words = content.split()

        for word in words:
            yield word + " "
            await asyncio.sleep(self.response_delay / len(words))

    async def health_check(self) -> bool:
        """Mock health check."""
        return not self.should_fail

    def get_available_models(self) -> list[str]:
        """Get mock model list."""
        return ["mock-model", "mock-model-large", "mock-model-small"]

    def get_connector_info(self) -> dict[str, Any]:
        """Get mock connector info."""
        return {
            "provider": "mock",
            "version": "1.0.0",
            "supports_streaming": True,
            "call_count": self.call_count,
            "should_fail": self.should_fail,
        }

    def _generate_mock_content(self, request: LLMRequest) -> str:
        """Generate deterministic mock content."""
        # Create response based on request content for predictable testing
        if "test" in request.user_message.lower():
            return "def test_example():\n    assert True"
        elif "class" in request.user_message.lower():
            return "class TestExample:\n    def test_method(self):\n        pass"
        elif "function" in request.user_message.lower():
            return "def example_function():\n    return True"
        else:
            return f"# Generated test for: {request.user_message[:50]}...\ndef test_generated():\n    pass"

    def reset_call_count(self) -> None:
        """Reset call counter for testing."""
        self.call_count = 0

    def set_failure_mode(self, should_fail: bool) -> None:
        """Set failure mode for testing error handling."""
        self.should_fail = should_fail
