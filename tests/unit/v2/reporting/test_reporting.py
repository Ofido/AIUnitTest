"""Tests for v2 reporting store and renderers."""

import json
from pathlib import Path

from ai_unit_test.v2.models import RunReport, TargetSpec, ValidationResult
from ai_unit_test.v2.reporting.renderer import JsonRenderer, TerminalRenderer
from ai_unit_test.v2.reporting.store import RunStore


def _make_report(run_id: str = "abc123", success: bool = True) -> RunReport:
    """Create a RunReport for testing."""
    return RunReport(
        run_id=run_id,
        target=TargetSpec(mode="explicit-file", files=["mod.py"]),
        attempts=1,
        success=success,
        backend_name="test-backend",
        final_summary="Test summary.",
        touched_files=["test_mod.py"],
        validation_history=[
            ValidationResult(validator_name="syntax", success=True, summary="ok"),
        ],
    )


class TestRunStore:
    """Test suite for RunStore."""

    def test_generate_run_id(self) -> None:
        """Test that run IDs are unique."""
        store = RunStore()
        id1 = store.generate_run_id()
        id2 = store.generate_run_id()
        assert id1 != id2
        assert len(id1) == 12

    def test_save_creates_artifacts(self, tmp_path: Path) -> None:
        """Test that save creates the expected artifact files."""
        store = RunStore(root=tmp_path)
        report = _make_report()
        run_dir = store.save(report, diff_text="--- a/test\n+++ b/test\n")

        assert (run_dir / "report.json").exists()
        assert (run_dir / "summary.md").exists()
        assert (run_dir / "patch.diff").exists()

    def test_save_report_json_valid(self, tmp_path: Path) -> None:
        """Test that the saved report.json is valid JSON."""
        store = RunStore(root=tmp_path)
        report = _make_report()
        run_dir = store.save(report)

        data = json.loads((run_dir / "report.json").read_text())
        assert data["run_id"] == "abc123"
        assert data["success"] is True

    def test_save_without_diff(self, tmp_path: Path) -> None:
        """Test that patch.diff is not created when diff is empty."""
        store = RunStore(root=tmp_path)
        report = _make_report()
        run_dir = store.save(report, diff_text="")

        assert not (run_dir / "patch.diff").exists()

    def test_load_last_report(self, tmp_path: Path) -> None:
        """Test loading the most recent report."""
        store = RunStore(root=tmp_path)
        store.save(_make_report(run_id="first"))
        store.save(_make_report(run_id="second"))

        loaded = store.load_last_report()
        assert loaded is not None
        assert loaded.run_id == "second"

    def test_load_last_report_empty(self, tmp_path: Path) -> None:
        """Test loading when no runs exist."""
        store = RunStore(root=tmp_path)
        assert store.load_last_report() is None

    def test_summary_md_content(self, tmp_path: Path) -> None:
        """Test that summary.md contains expected content."""
        store = RunStore(root=tmp_path)
        report = _make_report(success=False)
        run_dir = store.save(report)

        md = (run_dir / "summary.md").read_text()
        assert "❌ Failed" in md
        assert "abc123" in md
        assert "**syntax**" in md


class TestTerminalRenderer:
    """Test suite for TerminalRenderer."""

    def test_render_success(self) -> None:
        """Test rendering a successful report."""
        output = TerminalRenderer().render(_make_report(success=True))
        assert "SUCCESS" in output
        assert "abc123" in output

    def test_render_failure(self) -> None:
        """Test rendering a failed report."""
        output = TerminalRenderer().render(_make_report(success=False))
        assert "FAILED" in output


class TestJsonRenderer:
    """Test suite for JsonRenderer."""

    def test_render_valid_json(self) -> None:
        """Test that output is valid JSON."""
        output = JsonRenderer().render(_make_report())
        data = json.loads(output)
        assert data["run_id"] == "abc123"

    def test_render_includes_all_fields(self) -> None:
        """Test that all fields are present."""
        output = JsonRenderer().render(_make_report())
        data = json.loads(output)
        assert "target" in data
        assert "validation_history" in data
        assert "touched_files" in data
