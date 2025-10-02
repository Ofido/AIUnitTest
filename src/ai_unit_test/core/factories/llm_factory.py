"""Factory for creating LLM connectors."""

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_unit_test.core.interfaces.llm_connector import LLMConnector

from ai_unit_test.core.exceptions import ConfigurationError


class LLMConnectorFactory:
    """Factory for creating LLM connector instances."""

    _connectors: dict[str, type[Any]] = {}

    @classmethod
    def register_connector(
        cls: type["LLMConnectorFactory"],
        name: str,
        connector_class: type[Any],
    ) -> None:
        """Register a new connector type."""
        cls._connectors[name.lower()] = connector_class

    @classmethod
    def get_available_connectors(cls: type["LLMConnectorFactory"]) -> list[str]:
        """Get list of available connector names."""
        return list(cls._connectors.keys())

    @classmethod
    def _get_install_hint(cls: type["LLMConnectorFactory"], provider: str) -> str:
        """Get installation hint for missing provider."""
        hints = {
            "huggingface": "\nTip: Install with: pip install AIUnitTest[huggingface]",
            "openai": "",  # OpenAI is a core dependency, no hint needed
        }
        return hints.get(provider, "")

    @classmethod
    def create_connector(
        cls: type["LLMConnectorFactory"],
        provider: str,
        config: dict[str, Any] | None = None,
    ) -> "LLMConnector[Any]":
        """Create a connector instance."""
        if config is None:
            config = {}

        provider_lower = provider.lower()

        if provider_lower not in cls._connectors:
            available = ", ".join(cls.get_available_connectors())
            install_hint = cls._get_install_hint(provider_lower)
            raise ConfigurationError(
                f"Unknown LLM provider: {provider}. " f"Available providers: {available}" f"{install_hint}"
            )

        # Merge with environment-based config
        merged_config = cls._merge_environment_config(provider_lower, config)

        connector_class = cls._connectors[provider_lower]
        return connector_class(merged_config)  # type: ignore[no-any-return]

    @classmethod
    def create_from_config_file(cls: type["LLMConnectorFactory"], config: dict[str, Any]) -> "LLMConnector[Any]":
        """Create connector from pyproject.toml configuration."""
        llm_config = config.get("tool", {}).get("ai-unit-test", {}).get("llm", {})

        if not llm_config:
            # Default to OpenAI if no config specified
            raise ConfigurationError("No valid configuration found for LLM connector.")

        provider = llm_config.get("provider", "openai")
        provider_config = llm_config.get(provider, {})

        # Mesclar configs: se provider_config for dict, "achatar" no nível superior
        merged_config = {**llm_config}
        if isinstance(provider_config, dict):
            merged_config.update(provider_config)
        # Remover a chave do provider para evitar passar subdicionário
        if provider in merged_config:
            del merged_config[provider]

        if "provider" in merged_config:
            del merged_config["provider"]

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
