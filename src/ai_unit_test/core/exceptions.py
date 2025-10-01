"""Custom exceptions for AI Unit Test application."""


class AIUnitTestError(Exception):
    """Base exception for all AI Unit Test errors."""

    pass


class ConfigurationError(AIUnitTestError):
    """Raised when configuration is invalid or missing."""

    pass


class LLMConnectionError(AIUnitTestError):
    """Raised when LLM provider connection fails."""

    pass


class LLMProviderError(AIUnitTestError):
    """Raised when LLM provider returns an error."""

    pass


class IndexError(AIUnitTestError):
    """Raised when index operations fail."""

    pass


class IndexNotFoundError(IndexError):
    """Raised when requested index doesn't exist."""

    pass


class IndexCorruptedError(IndexError):
    """Raised when index file is corrupted or invalid."""

    pass


class ValidationError(AIUnitTestError):
    """Raised when input validation fails."""

    pass


class TestGenerationError(AIUnitTestError):
    """Raised when test generation process fails."""

    __test__ = False
