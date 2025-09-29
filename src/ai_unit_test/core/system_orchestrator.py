"""Module for the SystemOrchestrator class."""

import asyncio
import logging
import signal
import sys
import traceback
from pathlib import Path
from types import FrameType, TracebackType
from typing import NoReturn

from ai_unit_test.core.exceptions import AIUnitTestError, ConfigurationError

# Global state for graceful shutdown
shutdown_event = asyncio.Event()


class SystemOrchestrator:
    """Orchestrate the system for managing application lifecycle."""

    def __init__(self) -> None:
        """Initialize the SystemOrchestrator."""
        self.logger: logging.Logger | None = None
        self.log_handlers: list[logging.Handler] = []

    def setup_comprehensive_logging(self, verbose: bool, log_file: Path | None = None) -> None:
        """Configure comprehensive logging system."""
        console_level = logging.DEBUG if verbose else logging.INFO
        file_level = logging.DEBUG
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        self.log_handlers.append(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            self.log_handlers.append(file_handler)

        error_log_path = Path("logs") / "ai_unit_test_errors.log"
        error_log_path.parent.mkdir(exist_ok=True)
        error_handler = logging.FileHandler(error_log_path)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        self.log_handlers.append(error_handler)

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
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            signal_name = signal.Signals(signum).name
            if self.logger:
                self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            else:
                print(f"Received {signal_name}, shutting down...", file=sys.stderr)

            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(shutdown_event.set)
            except RuntimeError:
                sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)
        if self.logger:
            self.logger.debug("Signal handlers registered")

    def global_exception_handler(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> NoReturn:
        """Handle all unhandled exceptions with appropriate logging and user messages."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            sys.exit(130)

        exception_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        if self.logger:
            if issubclass(exc_type, ConfigurationError):
                self.logger.error(f"Configuration error: {exc_value}")
                self.logger.debug(f"Configuration error traceback:\n{exception_str}")
                print(f"❌ Configuration Error: {exc_value}", file=sys.stderr)
                print(
                    "💡 Check your pyproject.toml file or command line arguments.",
                    file=sys.stderr,
                )
            elif issubclass(exc_type, AIUnitTestError):
                self.logger.error(f"Application error: {exc_value}")
                self.logger.debug(f"Application error traceback:\n{exception_str}")
                print(f"❌ Error: {exc_value}", file=sys.stderr)
            else:
                self.logger.critical(f"Unhandled exception: {exc_value}")
                self.logger.critical(f"Traceback:\n{exception_str}")
                print(
                    "❌ Unexpected error occurred. Check logs for details.",
                    file=sys.stderr,
                )
                print(f"Error: {exc_value}", file=sys.stderr)
        else:
            print(f"❌ Fatal error: {exc_value}", file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

        sys.exit(1)

    def validate_system_requirements(self) -> None:
        """Validate system requirements and environment."""
        if sys.version_info < (3, 10):
            raise RuntimeError(
                f"Python 3.10+ is required. Current version: {sys.version_info.major}.{sys.version_info.minor}"
            )

        try:
            import numpy  # noqa: F401
            import openai  # noqa: F401
            import typer  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"Critical dependency missing: {e}")

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

        for handler in self.log_handlers:
            handler.close()

        if self.logger:
            self.logger.info("Resource cleanup completed")
