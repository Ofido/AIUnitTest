"""Tests for v2 CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ai_unit_test.v2.cli import v2_app
from ai_unit_test.v2.models import RunReport, TargetSpec, ValidationResult

runner = CliRunner()


def _make_report(success: bool = True) -> RunReport:
    """Create a RunReport for testing."""
    return RunReport(
        run_id="test123",
        target=TargetSpec(mode="explicit-file", files=["mod.py"]),
        attempts=1,
        success=success,
        backend_name="copilot-cli",
        final_summary="Done.",
        touched_files=["test_mod.py"],
        validation_history=[
            ValidationResult(validator_name="syntax", success=True, summary="ok"),
        ],
    )


class TestV2CliBackends:
    """Test suite for the backends command."""

    def test_list_backends(self) -> None:
        """Test listing available backends."""
        result = runner.invoke(v2_app, ["backends"])
        assert result.exit_code == 0
        assert "copilot-cli" in result.output
        assert "gemini-cli" in result.output


class TestV2CliReport:
    """Test suite for the report command."""

    @patch("ai_unit_test.v2.cli.RunStore")
    def test_report_last_run(self, mock_store_cls: MagicMock) -> None:
        """Test displaying the last run report."""
        mock_store = mock_store_cls.return_value
        mock_store.load_last_report.return_value = _make_report()

        result = runner.invoke(v2_app, ["report"])
        assert result.exit_code == 0
        assert "test123" in result.output

    @patch("ai_unit_test.v2.cli.RunStore")
    def test_report_no_runs(self, mock_store_cls: MagicMock) -> None:
        """Test report when no previous runs exist."""
        mock_store = mock_store_cls.return_value
        mock_store.load_last_report.return_value = None

        result = runner.invoke(v2_app, ["report"])
        assert result.exit_code == 1
        assert "No previous runs" in result.output

    @patch("ai_unit_test.v2.cli.RunStore")
    def test_report_json_output(self, mock_store_cls: MagicMock) -> None:
        """Test report with JSON output."""
        mock_store = mock_store_cls.return_value
        mock_store.load_last_report.return_value = _make_report()

        result = runner.invoke(v2_app, ["report", "--json"])
        assert result.exit_code == 0
        assert '"run_id"' in result.output


class TestV2CliDoctor:
    """Test suite for the doctor command."""

    @patch("ai_unit_test.v2.cli.CopilotCliBackend.is_available")
    @patch("ai_unit_test.v2.cli.GeminiCliBackend.is_available")
    def test_doctor_all_available(self, mock_gemini: MagicMock, mock_copilot: MagicMock) -> None:
        """Test doctor when all backends are available."""
        mock_copilot.return_value = True
        mock_gemini.return_value = True

        result = runner.invoke(v2_app, ["doctor"])
        assert result.exit_code == 0
        assert "✅" in result.output

    @patch("ai_unit_test.v2.cli.CopilotCliBackend.is_available")
    @patch("ai_unit_test.v2.cli.GeminiCliBackend.is_available")
    def test_doctor_some_missing(self, mock_gemini: MagicMock, mock_copilot: MagicMock) -> None:
        """Test doctor when some backends are missing."""
        mock_copilot.return_value = True
        mock_gemini.return_value = False

        result = runner.invoke(v2_app, ["doctor"])
        assert result.exit_code == 0
        assert "❌" in result.output
        assert "missing" in result.output.lower()
