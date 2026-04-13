"""Tests for v2 validation runners and feedback summarizer."""

import subprocess  # nosec B404
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_unit_test.v2.models import PatchApplication, PatchCandidate, TargetSpec, ValidationResult
from ai_unit_test.v2.validation.feedback import FeedbackSummarizer
from ai_unit_test.v2.validation.runners import PytestValidator, SyntaxValidator


def _make_application(files: list[str]) -> PatchApplication:
    """Create a PatchApplication for testing."""
    candidate = PatchCandidate(backend_name="test", plan_summary="x", patch_text="y")
    return PatchApplication(candidate=candidate, applied_files=files, success=True)


def _make_target() -> TargetSpec:
    """Create a TargetSpec for testing."""
    return TargetSpec(mode="explicit-file", files=["mod.py"])


class TestSyntaxValidator:
    """Test suite for SyntaxValidator."""

    def test_valid_syntax(self, tmp_path: Path) -> None:
        """Test that valid Python files pass."""
        valid_file = tmp_path / "test_ok.py"
        valid_file.write_text("def test_hello(): pass\n")

        validator = SyntaxValidator()
        result = validator.run(_make_application([str(valid_file)]), _make_target())

        assert result.success is True
        assert result.validator_name == "syntax"
        assert result.exit_code == 0

    def test_invalid_syntax(self, tmp_path: Path) -> None:
        """Test that invalid Python files fail."""
        bad_file = tmp_path / "test_bad.py"
        bad_file.write_text("def broken(\n")

        validator = SyntaxValidator()
        result = validator.run(_make_application([str(bad_file)]), _make_target())

        assert result.success is False
        assert result.exit_code == 1
        assert len(result.command_results) > 0

    def test_non_python_files_ignored(self, tmp_path: Path) -> None:
        """Test that non-Python files are ignored."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("hello")

        validator = SyntaxValidator()
        result = validator.run(_make_application([str(txt_file)]), _make_target())

        assert result.success is True


class TestPytestValidator:
    """Test suite for PytestValidator."""

    def test_no_test_files(self) -> None:
        """Test that no test files returns failure (prevents no-op success)."""
        validator = PytestValidator()
        result = validator.run(_make_application(["module.py"]), _make_target())

        assert result.success is False
        assert "No test files" in result.summary
        assert result.exit_code == -1

    @patch("ai_unit_test.v2.validation.runners.subprocess.run")
    def test_passing_tests(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test that passing tests return success."""
        mock_run.return_value = MagicMock(returncode=0, stdout="1 passed", stderr="")

        validator = PytestValidator(project_root=tmp_path)
        result = validator.run(_make_application(["test_mod.py"]), _make_target())

        assert result.success is True
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("ai_unit_test.v2.validation.runners.subprocess.run")
    def test_failing_tests(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test that failing tests return failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="FAILED test_x", stderr="")

        validator = PytestValidator(project_root=tmp_path)
        result = validator.run(_make_application(["test_mod.py"]), _make_target())

        assert result.success is False
        assert result.exit_code == 1

    @patch("ai_unit_test.v2.validation.runners.subprocess.run")
    def test_pytest_timeout(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test that pytest timeout is handled."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=120)

        validator = PytestValidator(project_root=tmp_path)
        result = validator.run(_make_application(["test_mod.py"]), _make_target())

        assert result.success is False
        assert "timed out" in result.summary.lower()

    @patch("ai_unit_test.v2.validation.runners.subprocess.run")
    def test_pytest_not_found(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test that missing pytest is handled."""
        mock_run.side_effect = FileNotFoundError()

        validator = PytestValidator(project_root=tmp_path)
        result = validator.run(_make_application(["test_mod.py"]), _make_target())

        assert result.success is False
        assert "not found" in result.summary.lower()

    @patch("ai_unit_test.v2.validation.runners.subprocess.run")
    def test_overrides_addopts(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test that --override-ini=addopts= is passed to avoid global config."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        validator = PytestValidator(project_root=tmp_path)
        validator.run(_make_application(["test_mod.py"]), _make_target())

        cmd = mock_run.call_args[0][0]
        assert "--override-ini=addopts=" in cmd

    @patch("ai_unit_test.v2.validation.runners.subprocess.run")
    def test_uses_sys_executable(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test that pytest is invoked via sys.executable, not hardcoded python."""
        import sys

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        validator = PytestValidator(project_root=tmp_path)
        validator.run(_make_application(["test_mod.py"]), _make_target())

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == sys.executable


class TestFeedbackSummarizer:
    """Test suite for FeedbackSummarizer."""

    def test_summarize(self) -> None:
        """Test summarizing failed results."""
        results = [
            ValidationResult(
                validator_name="syntax", success=False, summary="Syntax error.", command_results=["line 5"]
            ),
            ValidationResult(validator_name="pytest", success=True, summary="ok"),
        ]
        summarizer = FeedbackSummarizer()
        feedback = summarizer.summarize(results, attempt=1)

        assert any("Attempt 1 failed" in line for line in feedback)
        assert any("syntax" in line.lower() for line in feedback)

    def test_classify_syntax_error(self) -> None:
        """Test classifying a syntax error."""
        results = [ValidationResult(validator_name="syntax", success=False, summary="bad")]
        assert FeedbackSummarizer().classify_failure(results) == "syntax_error"

    def test_classify_test_failure(self) -> None:
        """Test classifying a test failure."""
        results = [
            ValidationResult(validator_name="syntax", success=True, summary="ok"),
            ValidationResult(validator_name="pytest", success=False, summary="fail"),
        ]
        assert FeedbackSummarizer().classify_failure(results) == "test_failure"

    def test_classify_unknown(self) -> None:
        """Test classifying when all pass."""
        results = [ValidationResult(validator_name="syntax", success=True, summary="ok")]
        assert FeedbackSummarizer().classify_failure(results) == "unknown"
