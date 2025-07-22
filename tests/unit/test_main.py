import logging
from unittest.mock import patch

from ai_unit_test.main import setup_logging


def test_setup_logging_verbose_true() -> None:
    """
    Tests the setup_logging function with verbose=True.
    """
    with patch.object(logging, "basicConfig") as mock_basic_config:
        setup_logging(True)
        mock_basic_config.assert_called_once_with(level=logging.DEBUG, format="%(levelname)s: %(message)s")


def test_setup_logging_verbose_false() -> None:
    """
    Tests the setup_logging function with verbose=False.
    """
    with patch.object(logging, "basicConfig") as mock_basic_config:
        setup_logging(False)
        mock_basic_config.assert_called_once_with(level=logging.INFO, format="%(levelname)s: %(message)s")
