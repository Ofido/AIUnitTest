"""AI Unit Test - System Entry Point and Error Handler."""

import asyncio
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import NoReturn

import typer

from ai_unit_test.cli import app
from ai_unit_test.core.exceptions import AIUnitTestError

# Global state for graceful shutdown
shutdown_event = asyncio.Event()


class SystemOrchestrator:
    """System orchestrator for managing application lifecycle."""

    def __init__(self) -> None:
        self.logger = None
        self.log_handlers = []

    def setup_comprehensive_logging(self, verbose: bool, log_file: Path | None = None) -> None:
        """Configure comprehensive logging system."""

        # Determine log levels
        console_level = logging.DEBUG if verbose else logging.INFO
        file_level = logging.DEBUG

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatters
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        self.log_handlers.append(console_handler)

        # File handler for general logs
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            self.log_handlers.append(file_handler)

        # Error file handler
        error_log_path = Path("logs") / "ai_unit_test_errors.log"
        error_log_path.parent.mkdir(exist_ok=True)

        error_handler = logging.FileHandler(error_log_path)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        self.log_handlers.append(error_handler)

        # Performance log handler
        perf_log_path = Path("logs") / "ai_unit_test_performance.log"
        perf_logger = logging.getLogger("performance")
        perf_handler = logging.FileHandler(perf_log_path)
        perf_handler.setFormatter(file_formatter)
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        self.log_handlers.append(perf_handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Comprehensive logging system initialized")

        if log_file:
            self.logger.info(f"Logs will be written to: {log_file}")

        self.logger.info(f"Error logs: {error_log_path}")
        self.logger.info(f"Performance logs: {perf_log_path}")

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            if self.logger:
                self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            else:
                print(f"Received {signal_name}, shutting down...", file=sys.stderr)

            # Set shutdown event for async code
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(shutdown_event.set)
            except RuntimeError:
                # No running loop, exit immediately
                sys.exit(1)

        # Register handlers for common signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)

        if self.logger:
            self.logger.debug("Signal handlers registered")

    def global_exception_handler(self, exc_type, exc_value, exc_traceback) -> NoReturn:
        """Handle all unhandled exceptions with appropriate logging and user messages."""

        # Don't handle KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Format exception info
        exception_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        if self.logger:
            # Import here to avoid circular imports
            from ai_unit_test.core.exceptions import ConfigurationError

            if issubclass(exc_type, ConfigurationError):
                # Configuration error - provide helpful user message
                self.logger.error(f"Configuration error: {exc_value}")
                self.logger.debug(f"Configuration error traceback:\n{exception_str}")
                print(f"❌ Configuration Error: {exc_value}", file=sys.stderr)
                print("💡 Check your pyproject.toml file or command line arguments.", file=sys.stderr)
            elif issubclass(exc_type, AIUnitTestError):
                # Known application error - log as error but don't include traceback in user message
                self.logger.error(f"Application error: {exc_value}")
                self.logger.debug(f"Application error traceback:\n{exception_str}")
                print(f"❌ Error: {exc_value}", file=sys.stderr)
            else:
                # Unexpected error - full logging
                self.logger.critical(f"Unhandled exception: {exc_value}")
                self.logger.critical(f"Traceback:\n{exception_str}")
                print(f"❌ Unexpected error occurred. Check logs for details.", file=sys.stderr)
                print(f"Error: {exc_value}", file=sys.stderr)
        else:
            # Fallback when logging isn't initialized
            print(f"❌ Fatal error: {exc_value}", file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

        sys.exit(1)

    def validate_system_requirements(self) -> None:
        """Validate system requirements and environment."""

        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError(
                f"Python 3.8+ is required. Current version: {sys.version_info.major}.{sys.version_info.minor}"
            )

        # Check critical dependencies
        try:
            import numpy
            import openai
            import typer
        except ImportError as e:
            raise RuntimeError(f"Critical dependency missing: {e}")

        # Check write permissions for logs
        logs_dir = Path("logs")
        try:
            logs_dir.mkdir(exist_ok=True)
            test_file = logs_dir / "test_write_permission"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise RuntimeError(f"No write permission for logs directory: {logs_dir}")

        if self.logger:
            self.logger.info("System requirements validation passed")

    def cleanup_resources(self) -> None:
        """Cleanup resources on shutdown."""

        if self.logger:
            self.logger.info("Cleaning up resources...")

        # Close log handlers
        for handler in self.log_handlers:
            handler.close()

        if self.logger:
            self.logger.info("Resource cleanup completed")


# Global orchestrator instance
orchestrator = SystemOrchestrator()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Path | None = typer.Option(None, "--log-file", help="Path to log file"),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to configuration file (default: pyproject.toml)"
    ),
) -> None:
    """
    AI Unit Test - Generate comprehensive unit tests using AI

    This tool analyzes your code coverage and generates targeted unit tests
    for uncovered code using large language models.
    """

    try:
        # Initialize system
        orchestrator.validate_system_requirements()
        orchestrator.setup_comprehensive_logging(verbose, log_file)
        orchestrator.setup_signal_handlers()

        # Set global exception handler
        sys.excepthook = orchestrator.global_exception_handler

        # Log startup information
        logger = logging.getLogger(__name__)
        logger.info("=" * 50)
        logger.info("AI Unit Test system initialized successfully")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {Path.cwd()}")
        logger.info(f"Verbose mode: {verbose}")
        if config_file:
            logger.info(f"Config file: {config_file}")
        logger.info("=" * 50)

        # Register cleanup on exit
        import atexit

        atexit.register(orchestrator.cleanup_resources)

    except Exception as e:
        print(f"❌ FATAL: Failed to initialize AI Unit Test system: {e}", file=sys.stderr)
        sys.exit(1)


def run_app() -> None:
    """Entry point for console script."""
    try:
        app()
    except SystemExit:
        # Normal exit, don't log as error
        pass
    except Exception as e:
        # This should be caught by global exception handler, but just in case
        print(f"❌ Application crashed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_app()
