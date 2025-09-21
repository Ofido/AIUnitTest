"""Test cases for the CLIManager module."""

import unittest
from unittest.mock import MagicMock, patch

from ai_unit_test.core.cli_manager import CLIManager
from ai_unit_test.core.system_orchestrator import SystemOrchestrator


class TestCLIManager(unittest.TestCase):
    """Test suite for the CLIManager class."""

    def setUp(self) -> None:
        """Set up mock objects for tests."""
        self.mock_orchestrator = MagicMock(spec=SystemOrchestrator)

    @patch("ai_unit_test.core.cli_manager.typer.Typer")
    def test_init(self, mock_typer: MagicMock) -> None:
        """Test that CLIManager initializes Typer app and registers commands."""
        mock_app = mock_typer.return_value
        with patch.object(CLIManager, "_register_commands") as mock_register_commands:
            cli_manager = CLIManager(self.mock_orchestrator)
            assert cli_manager.app is mock_app
            assert cli_manager.orchestrator is self.mock_orchestrator
            mock_register_commands.assert_called_once()

    @patch("typer.Typer.command")
    def test_register_commands(self, mock_command: MagicMock) -> None:
        """Test that CLI commands are registered with the Typer app."""
        with (
            patch("ai_unit_test.cli.generate_tests"),
            patch("ai_unit_test.cli.create_index"),
            patch("ai_unit_test.cli.health_check"),
        ):
            CLIManager(self.mock_orchestrator)
            assert mock_command.call_count == 3
            # Further assertions could be made on the calls to mock_command

    def test_run(self) -> None:
        """Test that the run method calls the Typer app."""
        with patch("ai_unit_test.core.cli_manager.typer.Typer") as mock_typer:
            mock_app = mock_typer.return_value
            cli_manager = CLIManager(self.mock_orchestrator)
            cli_manager.app = mock_app
            cli_manager.run()
            mock_app.assert_called_once()


if __name__ == "__main__":
    unittest.main()
