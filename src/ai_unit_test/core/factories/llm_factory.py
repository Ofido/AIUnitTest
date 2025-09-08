"""Factory for creating LLM connectors."""

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_unit_test.core.interfaces.llm_connector import LLMConnector

from ai_unit_test.core.exceptions import ConfigurationError


class LLMConnectorFactory:
    """Factory for creating LLM connector instances."""

    _connectors: dict[str, type["LLMConnector"]] = {}

    @classmethod
    def register_connector(cls: type["LLMConnectorFactory"], name: str, connector_class: type["LLMConnector"]) -> None:
        """Register a new connector type."""
        cls._connectors[name.lower()] = connector_class

    @classmethod
    def get_available_connectors(cls: type["LLMConnectorFactory"]) -> list[str]:
        """Get list of available connector names."""
        return list(cls._connectors.keys())

    @classmethod
    def create_connector(
        cls: type["LLMConnectorFactory"], provider: str, config: dict[str, Any] | None = None
    ) -> "LLMConnector":
        """Create a connector instance."""
        if config is None:
            config = {}

        provider_lower = provider.lower()

        if provider_lower not in cls._connectors:
            available = ", ".join(cls.get_available_connectors())
            raise ConfigurationError(f"Unknown LLM provider: {provider}. " f"Available providers: {available}")

        # Merge with environment-based config
        merged_config = cls._merge_environment_config(provider_lower, config)

        # Validate required configuration
        cls._validate_config(provider_lower, merged_config)

        connector_class = cls._connectors[provider_lower]
        return connector_class(merged_config)

    @classmethod
    def create_from_config_file(cls: type["LLMConnectorFactory"], config: dict[str, Any]) -> "LLMConnector":
        """Create connector from pyproject.toml configuration."""
        llm_config = config.get("tool", {}).get("ai-unit-test", {}).get("llm", {})

        if not llm_config:
            # Default to OpenAI if no config specified
            return cls.create_connector("openai")

        provider = llm_config.get("provider", "openai")
        provider_config = llm_config.get(provider, {})

        # Merge general LLM config with provider-specific config
        merged_config = {**llm_config, **provider_config}

        return cls.create_connector(provider, merged_config)

    @classmethod
    def _merge_environment_config(
        cls: type["LLMConnectorFactory"], provider: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge configuration with environment variables."""
        merged = config.copy()

        # Common environment variables
        env_mappings = {
            "openai": {
                "api_key": "OPENAI_API_KEY",
                "base_url": "OPENAI_API_URL",
                "organization": "OPENAI_ORGANIZATION",
            },
            "huggingface": {"api_key": "HF_API_KEY", "api_url": "HF_API_URL"},
        }

        if provider in env_mappings:
            for config_key, env_var in env_mappings[provider].items():
                if env_var in os.environ and config_key not in merged:
                    merged[config_key] = os.environ[env_var]

        return merged

    @classmethod
    def _validate_config(cls: type["LLMConnectorFactory"], provider: str, config: dict[str, Any]) -> None:
        """Validate provider-specific configuration."""
        required_configs = {
            "openai": ["api_key"],
            "huggingface": [],  # Can work without API key for local models
            "mock": [],
        }

        if provider in required_configs:
            for required_key in required_configs[provider]:
                if required_key not in config or not config[required_key]:
                    raise ConfigurationError(f"Missing required configuration for {provider}: {required_key}")


# Auto-register connectors when they're imported
def _register_default_connectors() -> None:
    """Register default connectors."""
    try:
        from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector

        LLMConnectorFactory.register_connector("openai", OpenAIConnector)
    except ImportError:
        pass

    try:
        from ai_unit_test.core.implementations.llm.huggingface_connector import HuggingFaceConnector

        LLMConnectorFactory.register_connector("huggingface", HuggingFaceConnector)
    except ImportError:
        pass

    try:
        from ai_unit_test.core.implementations.llm.mock_connector import MockConnector

        LLMConnectorFactory.register_connector("mock", MockConnector)
    except ImportError:
        pass


# Register default connectors on module import
_register_default_connectors()
