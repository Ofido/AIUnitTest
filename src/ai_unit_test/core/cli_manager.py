"""Module for managing the CLI application."""

import typer

from ai_unit_test.core.system_orchestrator import SystemOrchestrator
from ai_unit_test.main import main as main_callback


class CLIManager:
    """Manage the Typer CLI application."""

    def __init__(self, orchestrator: SystemOrchestrator) -> None:
        """
        Initialize the CLIManager.

        Args:
            orchestrator: The system orchestrator.
        """
        self.app = typer.Typer()
        self.orchestrator = orchestrator
        self.app.callback(invoke_without_command=True)(main_callback)
        self._register_commands()

    def _register_commands(self) -> None:
        """Register CLI commands with the Typer app."""
        from ai_unit_test.cli import create_index, generate_tests, health_check

        self.app.command()(generate_tests)
        self.app.command()(create_index)
        self.app.command()(health_check)

    def run(self) -> None:
        """Run the CLI application."""
        self.app()
