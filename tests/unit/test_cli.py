from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ai_unit_test.cli import app


@patch("ai_unit_test.cli.logger.error")
@patch("ai_unit_test.cli.extract_function_source")
def test_func_command_function_not_found(
    mock_extract_function_source: MagicMock,
    mock_logger_error: MagicMock,
) -> None:
    """
    Tests the func command when the function is not found.
    """
    runner = CliRunner()
    mock_extract_function_source.return_value = None

    result = runner.invoke(
        app,
        [
            "func",
            "src/simple_math.py",
            "non_existent_function",
            "--tests-folder",
            "tests",
        ],
    )
    assert result.exit_code == 1
    mock_extract_function_source.assert_called_once_with("src/simple_math.py", "non_existent_function")
    mock_logger_error.assert_called_once_with("Function 'non_existent_function' not found in 'src/simple_math.py'.")


def test_command_without_arguments() -> None:
    """
    Tests the command when no arguments are provided.
    """
    runner = CliRunner()
    result = runner.invoke(app)
    assert result.exit_code == 2


@patch("ai_unit_test.cli.logger.error")
def test_main_no_folders_and_no_coverage_file_provided(mock_logger_error: MagicMock) -> None:
    """
    Tests the main command when no folders and no coverage file are provided.
    """
    runner = CliRunner()
    result = runner.invoke(app, ["main"])
    assert result.exit_code == 1
    mock_logger_error.assert_called_with("Coverage file not found: .coverage")
