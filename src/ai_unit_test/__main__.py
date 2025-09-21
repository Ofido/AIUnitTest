"""AI Unit Test - Application Entry Point."""

import sys

from ai_unit_test.core.cli_manager import CLIManager
from ai_unit_test.core.system_orchestrator import SystemOrchestrator


def run_app() -> None:
    """Entry point for console script."""
    try:
        orchestrator = SystemOrchestrator()
        cli_manager = CLIManager(orchestrator)
        cli_manager.run()
    except SystemExit:
        pass
    except Exception as e:
        print(f"❌ Application crashed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_app()
