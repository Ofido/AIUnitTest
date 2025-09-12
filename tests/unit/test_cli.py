from typer.testing import CliRunner

from ai_unit_test.cli import app


def test_command_without_arguments() -> None:
    """
    Tests the command when no arguments are provided.
    """
    runner = CliRunner()
    result = runner.invoke(app)
    assert result.exit_code == 2
