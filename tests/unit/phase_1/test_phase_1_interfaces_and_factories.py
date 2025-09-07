"""Tests for Phase 1: Interfaces and Factories Implementation."""

from typing import Any

import pytest

from ai_unit_test.core.exceptions import (
    AIUnitTestError,
    ConfigurationError,
    IndexCorruptedError,
    IndexError,
    IndexNotFoundError,
    LLMConnectionError,
    LLMProviderError,
    TestGenerationError,
    ValidationError,
)
from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory
from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory
from ai_unit_test.core.interfaces.index_organizer import IndexMetadata, IndexOrganizer, IndexStats, SearchResult
from ai_unit_test.core.interfaces.llm_connector import LLMConnector, LLMRequest, LLMResponse, LLMUsage


class TestExceptions:
    """Test custom exception hierarchy."""

    def test_exception_hierarchy(self) -> None:
        """Test that all exceptions inherit from AIUnitTestError."""
        exceptions = [
            ConfigurationError,
            LLMConnectionError,
            LLMProviderError,
            IndexError,
            IndexNotFoundError,
            IndexCorruptedError,
            ValidationError,
            TestGenerationError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, AIUnitTestError)
            assert issubclass(exc_class, Exception)

    def test_exception_instantiation(self) -> None:
        """Test that exceptions can be instantiated with messages."""
        error = ConfigurationError("Test configuration error")
        assert str(error) == "Test configuration error"
        assert isinstance(error, AIUnitTestError)

    def test_specialized_index_errors(self) -> None:
        """Test that specialized index errors inherit from IndexError."""
        assert issubclass(IndexNotFoundError, IndexError)
        assert issubclass(IndexCorruptedError, IndexError)


class TestLLMInterfaces:
    """Test LLM interface dataclasses and abstract class."""

    def test_llm_request_creation(self) -> None:
        """Test LLMRequest dataclass creation."""
        request = LLMRequest(
            system_message="You are a helpful assistant", user_message="Hello world", model="gpt-3.5-turbo"
        )

        assert request.system_message == "You are a helpful assistant"
        assert request.user_message == "Hello world"
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 0.1  # default
        assert request.max_tokens is None  # default
        assert request.stream is False  # default

    def test_llm_response_creation(self) -> None:
        """Test LLMResponse dataclass creation."""
        response = LLMResponse(
            content="Hello! How can I help you?",
            usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
            model="gpt-3.5-turbo",
            finish_reason="stop",
            response_time_ms=150,
        )

        assert response.content == "Hello! How can I help you?"
        assert response.usage["total_tokens"] == 17
        assert response.model == "gpt-3.5-turbo"
        assert response.finish_reason == "stop"
        assert response.response_time_ms == 150

    def test_llm_usage_creation(self) -> None:
        """Test LLMUsage dataclass creation."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 7
        assert usage.total_tokens == 17

    def test_llm_connector_is_abstract(self) -> None:
        """Test that LLMConnector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMConnector({})


class TestIndexInterfaces:
    """Test Index interface dataclasses and abstract class."""

    def test_index_metadata_creation(self) -> None:
        """Test IndexMetadata dataclass creation."""
        metadata = IndexMetadata(
            embedding_model="text-embedding-ada-002",
            schema_version="1.0",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            total_documents=100,
            embedding_dimension=1536,
            backend_type="faiss",
            backend_config={"index_type": "IndexFlatIP"},
        )

        assert metadata.embedding_model == "text-embedding-ada-002"
        assert metadata.total_documents == 100
        assert metadata.embedding_dimension == 1536
        assert metadata.backend_type == "faiss"
        assert metadata.backend_config["index_type"] == "IndexFlatIP"

    def test_search_result_creation(self) -> None:
        """Test SearchResult dataclass creation."""
        result = SearchResult(metadata={"content": "Test document", "id": "doc_1"}, score=0.95, document_id="doc_1")

        assert result.metadata["content"] == "Test document"
        assert result.score == 0.95
        assert result.document_id == "doc_1"
        assert result.embedding is None  # default

    def test_index_stats_creation(self) -> None:
        """Test IndexStats dataclass creation."""
        stats = IndexStats(
            total_documents=100,
            average_score_distribution={"high": 0.3, "medium": 0.5, "low": 0.2},
            search_latency_ms=25.5,
            memory_usage_mb=128.0,
        )

        assert stats.total_documents == 100
        assert stats.average_score_distribution["high"] == 0.3
        assert stats.search_latency_ms == 25.5
        assert stats.memory_usage_mb == 128.0

    def test_index_organizer_is_abstract(self) -> None:
        """Test that IndexOrganizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IndexOrganizer({})


class TestLLMFactory:
    """Test LLM connector factory functionality."""

    def test_factory_initialization(self) -> None:
        """Test that factory starts with empty connectors."""
        # Reset factory state
        LLMConnectorFactory._connectors = {}

        assert len(LLMConnectorFactory.get_available_connectors()) == 0

    def test_register_connector(self) -> None:
        """Test connector registration."""

        class MockConnector(LLMConnector):
            async def initialize(self) -> None:
                pass

            async def generate_response(self, request: Any) -> None:  # type: ignore
                pass

            async def generate_stream(self, request: Any) -> None:  # type: ignore
                pass

            async def health_check(self) -> bool:
                return True

            def get_available_models(self) -> list[str]:
                return ["mock-model"]

            def get_connector_info(self) -> dict[str, str]:
                return {"name": "mock"}

        LLMConnectorFactory.register_connector("mock", MockConnector)

        available = LLMConnectorFactory.get_available_connectors()
        assert "mock" in available

    def test_unknown_provider_error(self) -> None:
        """Test error for unknown provider."""
        LLMConnectorFactory._connectors = {}

        with pytest.raises(ConfigurationError) as exc_info:
            LLMConnectorFactory.create_connector("unknown_provider")

        assert "Unknown LLM provider: unknown_provider" in str(exc_info.value)

    def test_environment_config_merging(self) -> None:
        """Test environment variable configuration merging."""
        config = {"temperature": 0.7}
        merged = LLMConnectorFactory._merge_environment_config("openai", config)

        # Should keep original config
        assert merged["temperature"] == 0.7
        # Should not add env vars that don't exist
        assert "api_key" not in merged or merged["api_key"] is None


class TestIndexFactory:
    """Test index organizer factory functionality."""

    def test_factory_initialization(self) -> None:
        """Test that factory starts with empty organizers."""
        # Reset factory state
        IndexOrganizerFactory._organizers = {}
        IndexOrganizerFactory._availability_cache = {}

        assert len(IndexOrganizerFactory.get_available_organizers()) == 0

    def test_register_organizer(self) -> None:
        """Test organizer registration."""

        class MockOrganizer(IndexOrganizer):
            async def create_index(self, embeddings, metadata, index_path, model_name):
                pass

            async def load_index(self, index_path):
                pass

            async def search(self, query_embedding, k=5, threshold=0.7):
                pass

            async def add_documents(self, embeddings, metadata):
                pass

            async def remove_documents(self, document_ids):
                pass

            async def update_document(self, document_id, embedding, metadata):
                pass

            async def get_index_info(self) -> None:
                pass

            async def validate_index(self, index_path):
                pass

            async def get_stats(self) -> None:
                pass

            async def optimize_index(self) -> None:
                pass

        IndexOrganizerFactory.register_organizer("mock", MockOrganizer)

        # Mock availability check to return True
        IndexOrganizerFactory._availability_cache["mock"] = True

        available = IndexOrganizerFactory.get_available_organizers()
        assert "mock" in available

    def test_unknown_backend_error(self) -> None:
        """Test error for unknown backend."""
        IndexOrganizerFactory._organizers = {}

        with pytest.raises(ConfigurationError) as exc_info:
            IndexOrganizerFactory.create_organizer("unknown_backend")

        assert "Unknown index backend: unknown_backend" in str(exc_info.value)

    def test_availability_checking(self) -> None:
        """Test backend availability checking."""
        # Memory backend should always be available
        available = IndexOrganizerFactory._check_availability("memory", None)
        assert available is True

        # Should cache the result
        assert "memory" in IndexOrganizerFactory._availability_cache
        assert IndexOrganizerFactory._availability_cache["memory"] is True

    def test_default_config_merging(self) -> None:
        """Test default configuration merging."""
        config = {"custom_setting": True}
        merged = IndexOrganizerFactory._merge_default_config("faiss", config)

        # Should keep custom config
        assert merged["custom_setting"] is True
        # Should add default faiss config
        assert "index_type" in merged
        assert merged["index_type"] == "IndexFlatIP"


class TestCoreModuleImports:
    """Test that core module exports work correctly."""

    def test_core_imports(self) -> None:
        """Test that main core imports work."""
        from ai_unit_test.core import (
            AIUnitTestError,
            IndexOrganizer,
            IndexOrganizerFactory,
            LLMConnector,
            LLMConnectorFactory,
        )

        # Should be able to import all main classes
        assert AIUnitTestError
        assert LLMConnector
        assert IndexOrganizer
        assert LLMConnectorFactory
        assert IndexOrganizerFactory

    def test_interfaces_imports(self) -> None:
        """Test that interfaces submodule imports work."""
        from ai_unit_test.core.interfaces import IndexMetadata, IndexOrganizer, LLMConnector, LLMRequest

        assert LLMConnector
        assert LLMRequest
        assert IndexOrganizer
        assert IndexMetadata

    def test_factories_imports(self) -> None:
        """Test that factories submodule imports work."""
        from ai_unit_test.core.factories import IndexOrganizerFactory, LLMConnectorFactory

        assert LLMConnectorFactory
        assert IndexOrganizerFactory
