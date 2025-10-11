"""Test backend switching."""

from unittest.mock import patch

import pytest

from ai_unit_test.core.exceptions import ConfigurationError
from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory
from ai_unit_test.core.implementations.indexing.memory_organizer import InMemoryIndexOrganizer


class TestBackendSwitching:
    """Test backend switching functionality."""

    def teardown_method(self) -> None:
        """Clean up registered organizers."""
        if "test" in IndexOrganizerFactory._organizers:
            del IndexOrganizerFactory._organizers["test"]

    def test_switch_index_backend(self) -> None:
        """Test switching index backend via configuration."""
        # Arrange
        memory_config = {"tool": {"ai-unit-test": {"indexing": {"backend": "memory"}}}}

        class TestOrganizer(InMemoryIndexOrganizer):
            pass

        with patch.dict(IndexOrganizerFactory._organizers, {"test": TestOrganizer}):
            test_config = {"tool": {"ai-unit-test": {"indexing": {"backend": "test"}}}}

            # Act
            memory_organizer = IndexOrganizerFactory.create_from_config_file(memory_config)
            test_organizer = IndexOrganizerFactory.create_from_config_file(test_config)

            # Assert
            assert isinstance(memory_organizer, InMemoryIndexOrganizer)
            assert isinstance(test_organizer, TestOrganizer)

    def test_invalid_backend(self) -> None:
        """Test that an invalid backend raises an error."""
        # Arrange
        invalid_config = {"tool": {"ai-unit-test": {"indexing": {"backend": "invalid-backend"}}}}

        # Act & Assert
        with pytest.raises(ConfigurationError):
            IndexOrganizerFactory.create_from_config_file(invalid_config)
