"""Test cases for the main module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from ai_unit_test.main import CONFIG_FILE_OPTION, LOG_FILE_OPTION, VERBOSE_OPTION, main


class TestMainFunction:
    """Test cases for the main function."""

    def test_main_successful_initialization(self, tmp_path: Path) -> None:
        """Test successful system initialization."""
        mock_ctx = Mock(spec=typer.Context)
        mock_ctx.invoked_subcommand = "some_command"
        log_file = tmp_path / "test.log"

        with (
            patch("ai_unit_test.main.SystemOrchestrator") as mock_orchestrator_class,
            patch("ai_unit_test.main.OrchestrationService") as _,
            patch("ai_unit_test.main.logging.getLogger") as mock_get_logger,
        ):

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            main(mock_ctx, verbose=True, log_file=log_file, config_file=Path("config.toml"))

            mock_orchestrator.validate_system_requirements.assert_called_once()
            mock_orchestrator.setup_comprehensive_logging.assert_called_once_with(True, log_file)
            mock_orchestrator.setup_signal_handlers.assert_called_once()

    def test_main_without_config_file(self) -> None:
        """Test main function without config file."""
        mock_ctx = Mock(spec=typer.Context)
        mock_ctx.invoked_subcommand = "test_command"

        with (
            patch("ai_unit_test.main.SystemOrchestrator") as mock_orchestrator_class,
            patch("ai_unit_test.main.OrchestrationService") as mock_service_class,
            patch("ai_unit_test.main.logging.getLogger"),
        ):

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            main(mock_ctx, verbose=False, log_file=None, config_file=None)

            mock_service_class.assert_called_once_with({})
            assert mock_ctx.obj["orchestration_service"] is not None
            assert mock_ctx.obj["system_orchestrator"] == mock_orchestrator

    def test_main_with_config_file(self) -> None:
        """Test main function with config file."""
        mock_ctx = Mock(spec=typer.Context)
        mock_ctx.invoked_subcommand = "test_command"
        config_path = Path("test_config.toml")

        with (
            patch("ai_unit_test.main.SystemOrchestrator"),
            patch("ai_unit_test.main.OrchestrationService") as mock_service_class,
            patch("ai_unit_test.main.logging.getLogger"),
        ):

            main(mock_ctx, config_file=config_path)

            mock_service_class.assert_called_once_with({"config_file": str(config_path)})

    def test_main_no_subcommand_invoked(self) -> None:
        """Test main function when no subcommand is invoked."""
        mock_ctx = Mock(spec=typer.Context)
        mock_ctx.invoked_subcommand = None

        with (
            patch("ai_unit_test.main.SystemOrchestrator"),
            patch("ai_unit_test.main.OrchestrationService"),
            patch("ai_unit_test.main.logging.getLogger"),
            patch("ai_unit_test.main.typer.echo") as mock_echo,
        ):

            main(mock_ctx)

            mock_echo.assert_called_once_with("Welcome to AI Unit Test! Please specify a command.")

    def test_main_exception_handling(self) -> None:
        """Test exception handling in main function."""
        mock_ctx = Mock(spec=typer.Context)

        with (  # noqa
            patch("ai_unit_test.main.SystemOrchestrator") as mock_orchestrator_class,
            patch("ai_unit_test.main.print") as mock_print,
            pytest.raises(SystemExit) as exc_info,
        ):

            mock_orchestrator_class.side_effect = Exception("Test error")

            main(mock_ctx)

            mock_print.assert_called_once_with(
                "❌ FATAL: Failed to initialize AI Unit Test system: Test error", file=sys.stderr
            )
            assert exc_info.value.code == 1

    def test_main_sets_global_exception_handler(self) -> None:
        """Test that main function sets global exception handler."""
        mock_ctx = Mock(spec=typer.Context)
        mock_ctx.invoked_subcommand = "test_command"

        with (
            patch("ai_unit_test.main.SystemOrchestrator") as mock_orchestrator_class,
            patch("ai_unit_test.main.OrchestrationService"),
            patch("ai_unit_test.main.logging.getLogger"),
            patch("ai_unit_test.main.sys") as mock_sys,
        ):

            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            main(mock_ctx)

            assert mock_sys.excepthook == mock_orchestrator.global_exception_handler

    def test_main_logging_messages(self) -> None:
        """Test that main function logs appropriate messages."""
        mock_ctx = Mock(spec=typer.Context)
        mock_ctx.invoked_subcommand = "test_command"
        config_file = Path("test.toml")

        with (
            patch("ai_unit_test.main.SystemOrchestrator"),
            patch("ai_unit_test.main.OrchestrationService"),
            patch("ai_unit_test.main.logging.getLogger") as mock_get_logger,
            patch("ai_unit_test.main.sys") as mock_sys,
            patch("ai_unit_test.main.Path") as mock_path,
        ):

            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_sys.version = "3.11.0"
            mock_path.cwd.return_value = Path("/test/dir")

            main(mock_ctx, verbose=True, config_file=config_file)

            # Verify logging calls
            expected_calls = [
                ("=" * 50,),
                ("AI Unit Test system initialized successfully",),
                (f"Python version: {mock_sys.version}",),
                (f"Working directory: {mock_path.cwd()}",),
                ("Verbose mode: True",),
                (f"Config file: {config_file}",),
                ("=" * 50,),
            ]

            for expected_call in expected_calls:
                mock_logger.info.assert_any_call(*expected_call)


class TestConstants:
    """Test cases for module constants."""

    def test_verbose_option_configuration(self) -> None:
        """Test VERBOSE_OPTION configuration."""
        assert VERBOSE_OPTION.default is False
        assert "--verbose" in VERBOSE_OPTION.param_decls
        assert "-v" in VERBOSE_OPTION.param_decls
        assert VERBOSE_OPTION.help == "Enable verbose logging"

    def test_log_file_option_configuration(self) -> None:
        """Test LOG_FILE_OPTION configuration."""
        assert LOG_FILE_OPTION.default is None
        assert "--log-file" in LOG_FILE_OPTION.param_decls
        assert LOG_FILE_OPTION.help == "Path to log file"

    def test_config_file_option_configuration(self) -> None:
        """Test CONFIG_FILE_OPTION configuration."""
        assert CONFIG_FILE_OPTION.default is None
        assert "--config" in CONFIG_FILE_OPTION.param_decls
        assert CONFIG_FILE_OPTION.help == "Path to configuration file (default: pyproject.toml)"
