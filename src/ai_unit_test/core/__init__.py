"""Core module for AI Unit Test application.

This module provides the foundational interfaces, factories, and exceptions
for the clean architecture implementation.
"""

# Export exceptions
from .exceptions import (
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
from .factories.index_factory import IndexOrganizerFactory

# Export factories
from .factories.llm_factory import LLMConnectorFactory
from .interfaces.index_organizer import (
    IndexMetadata,
    IndexOrganizer,
    IndexStats,
    SearchResult,
)

# Export interfaces
from .interfaces.llm_connector import LLMConnector, LLMRequest, LLMResponse, LLMUsage

__all__ = [
    # Exceptions
    "AIUnitTestError",
    "ConfigurationError",
    "LLMConnectionError",
    "LLMProviderError",
    "IndexError",
    "IndexNotFoundError",
    "IndexCorruptedError",
    "ValidationError",
    "TestGenerationError",
    # LLM Interfaces
    "LLMConnector",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    # Index Interfaces
    "IndexOrganizer",
    "IndexMetadata",
    "SearchResult",
    "IndexStats",
    # Factories
    "LLMConnectorFactory",
    "IndexOrganizerFactory",
]
