"""Test LLM connector factory."""

import os
from collections.abc import Sequence
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from ai_unit_test.core.exceptions import ConfigurationError
from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory
from ai_unit_test.core.implementations.llm.mock_connector import MockConnector


class TestLLMConnectorFactory:
    """Test LLM connector factory functionality."""

    def test_factory_registration(self) -> None:
        """Test connector registration."""
        # Test getting available connectors
        connectors = LLMConnectorFactory.get_available_connectors()
        assert isinstance(connectors, list)
        assert len(connectors) > 0
        assert "mock" in connectors

        # Test verifying registered connectors have implementations
        for connector_name in connectors:
            try:
                connector = LLMConnectorFactory.create_connector(connector_name)
                assert connector is not None
            except ConfigurationError:
                # Some connectors might need configuration
                pass

    def test_create_mock_connector(self) -> None:
        """Test creating mock connector."""
        from ai_unit_test.core.implementations.llm.mock_connector import MockConnectorConfig

        connector = LLMConnectorFactory.create_connector("mock")
        assert isinstance(connector, MockConnector)
        assert connector.config == MockConnectorConfig()

        # Test with config
        config = {"should_fail": False, "response_delay": 0.1}
        connector = LLMConnectorFactory.create_connector("mock", config)
        assert connector.config.should_fail is False
        assert connector.config.response_delay == 0.1

    def test_create_unknown_connector(self) -> None:
        """Test creating unknown connector raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            LLMConnectorFactory.create_connector("unknown")

        assert "Unknown LLM provider: unknown" in str(exc_info.value)
        assert "Available providers:" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_API_URL": "http://test"})
    def test_environment_config_merging(self) -> None:
        """Test environment variable merging."""
        # Test OpenAI environment variables
        config = {"temperature": 0.5}
        merged_config = LLMConnectorFactory._merge_environment_config("openai", config)

        assert merged_config["api_key"] == "test-key"
        assert merged_config["base_url"] == "http://test"
        assert merged_config["temperature"] == 0.5

    def test_create_from_config_file(self) -> None:
        """Test creating connector from configuration file."""
        # Test with mock provider config
        config = {
            "tool": {
                "ai-unit-test": {
                    "llm": {
                        "provider": "mock",
                        "model": "mock-model",
                        "temperature": 0.2,
                        "mock": {"should_fail": False},
                    }
                }
            }
        }

        connector = LLMConnectorFactory.create_from_config_file(config)
        assert isinstance(connector, MockConnector)
        assert connector.get_connector_info()["provider"] == "mock"
        assert connector.config.model == "mock-model"
        assert connector.config.temperature == 0.2
        assert connector.config.should_fail is False

        # Test with empty config (should default to OpenAI)
        with pytest.raises(ConfigurationError):
            LLMConnectorFactory.create_from_config_file({})


def test_llm_factory_case_insensitive_reload_and_config_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ."""
    import builtins
    import importlib
    import sys

    from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory as OriginalFactory
    from ai_unit_test.core.implementations.llm.mock_connector import MockConnector, MockConnectorConfig

    # Case-insensitive provider name and default config handling
    connector = OriginalFactory.create_connector("MOCK")
    assert isinstance(connector, MockConnector)
    assert connector.config == MockConnectorConfig()

    # Simulate ImportError for specific connector implementation imports and reload module
    orig_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] = [],  # noqa
        level: int = 0,
    ) -> ModuleType:
        if isinstance(name, str) and "ai_unit_test.core.implementations.llm" in name:
            if any(x in name for x in ("openai", "anthropic", "azure")):
                raise ImportError("Simulated missing implementation")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "ai_unit_test.core.factories.llm_factory"
    if module_name in sys.modules:
        del sys.modules[module_name]
    # Also remove connector implementation modules to force import attempts
    for mod in list(sys.modules):
        if mod.startswith("ai_unit_test.core.implementations.llm"):
            del sys.modules[mod]

    reloaded_module = importlib.import_module(module_name)

    # Restore original import
    monkeypatch.setattr(builtins, "__import__", orig_import)

    ReloadedFactory = reloaded_module.LLMConnectorFactory
    # Ensure reload succeeded and available connectors still contain 'mock'
    assert "mock" in ReloadedFactory.get_available_connectors()

    # Test create_from_config_file merging when provider is omitted (defaults to openai)
    # Use a fake connector class to capture merged configuration
    class FakeConnector:
        def __init__(self, cfg: Any) -> None:  # noqa
            self.config = cfg

    original_connectors = ReloadedFactory._connectors.copy()
    try:
        ReloadedFactory._connectors["openai"] = FakeConnector

        config = {
            "tool": {
                "ai-unit-test": {
                    "llm": {
                        "model": "my-model",
                        "temperature": 0.4,
                        "openai": {"api_key": "cfg-api", "base_url": "http://cfg"},
                    }
                }
            }
        }

        fake_conn = ReloadedFactory.create_from_config_file(config)
        assert isinstance(fake_conn, FakeConnector)
        merged = fake_conn.config
        assert merged["model"] == "my-model"
        assert merged["temperature"] == 0.4
        assert merged["api_key"] == "cfg-api"
        assert "openai" not in merged
        assert "provider" not in merged
    finally:
        ReloadedFactory._connectors = original_connectors.copy()
