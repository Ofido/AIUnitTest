from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_logger_error() -> Iterator[MagicMock]:
    with patch("ai_unit_test.cli.logger.error") as mock:
        yield mock


@pytest.fixture
def mock_read_file_content() -> Iterator[MagicMock]:
    with patch("ai_unit_test.cli.read_file_content") as mock:
        yield mock
