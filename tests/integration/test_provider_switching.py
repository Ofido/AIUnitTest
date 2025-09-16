"""Test provider switching."""

import pytest

from ai_unit_test.core.exceptions import ConfigurationError
from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory
from ai_unit_test.core.implementations.llm.mock_connector import MockConnector


class TestProviderSwitching:
    """Test provider switching functionality."""

    def test_switch_llm_provider(self) -> None:
        """Test switching LLM provider via configuration."""
        # Arrange
        mock_config = {"tool": {"ai-unit-test": {"llm": {"provider": "mock"}}}}

        class TestConnector(MockConnector):
            pass

        LLMConnectorFactory.register_connector("test", TestConnector)

        test_config = {"tool": {"ai-unit-test": {"llm": {"provider": "test"}}}}

        # Act
        mock_connector = LLMConnectorFactory.create_from_config_file(mock_config)
        test_connector = LLMConnectorFactory.create_from_config_file(test_config)

        # Assert
        assert isinstance(mock_connector, MockConnector)
        assert isinstance(test_connector, TestConnector)

    def test_invalid_provider(self) -> None:
        """Test that an invalid provider raises an error."""
        # Arrange
        invalid_config = {"tool": {"ai-unit-test": {"llm": {"provider": "invalid-provider"}}}}

        # Act & Assert
        with pytest.raises(ConfigurationError):
            LLMConnectorFactory.create_from_config_file(invalid_config)
