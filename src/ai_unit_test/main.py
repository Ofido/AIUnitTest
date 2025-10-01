"""AI Unit Test - System Entry Point and Error Handler."""

import logging
import sys
from pathlib import Path
from typing import Any

import typer

from ai_unit_test.core.system_orchestrator import SystemOrchestrator
from ai_unit_test.services.orchestration_service import OrchestrationService

VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
LOG_FILE_OPTION = typer.Option(None, "--log-file", help="Path to log file")
CONFIG_FILE_OPTION = typer.Option(None, "--config", help="Path to configuration file (default: pyproject.toml)")


def main(
    ctx: typer.Context,
    verbose: bool = VERBOSE_OPTION,
    log_file: Path | None = LOG_FILE_OPTION,
    config_file: Path | None = CONFIG_FILE_OPTION,
) -> None:
    """
    AI Unit Test - Generate comprehensive unit tests using AI.

    This tool analyzes your code coverage and generates targeted unit tests
    for uncovered code using large language models.
    """
    try:
        orchestrator = SystemOrchestrator()
        orchestrator.validate_system_requirements()
        orchestrator.setup_comprehensive_logging(verbose, log_file)
        orchestrator.setup_signal_handlers()

        sys.excepthook = orchestrator.global_exception_handler

        logger = logging.getLogger(__name__)
        logger.info("=" * 50)
        logger.info("AI Unit Test system initialized successfully")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {Path.cwd()}")
        logger.info(f"Verbose mode: {verbose}")
        if config_file:
            logger.info(f"Config file: {config_file}")
        logger.info("=" * 50)

        import atexit

        atexit.register(orchestrator.cleanup_resources)

        config: dict[str, Any] = {}
        if config_file:
            config["config_file"] = str(config_file)

        ctx.obj = {
            "orchestration_service": OrchestrationService(config),
            "system_orchestrator": orchestrator,
        }

        if ctx.invoked_subcommand is None:
            typer.echo("Welcome to AI Unit Test! Please specify a command.")

    except Exception as e:
        print(f"❌ FATAL: Failed to initialize AI Unit Test system: {e}", file=sys.stderr)
        sys.exit(1)
