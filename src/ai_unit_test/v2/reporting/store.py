"""Artifact store for AIUnitTest v2 run reports."""

import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path

from ai_unit_test.v2.models import RunReport

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_ROOT = ".ai-unit-test/runs"


class RunStore:
    """Persist run artifacts to disk."""

    def __init__(self, root: Path | None = None) -> None:
        """Initialize with artifact storage root directory."""
        self.root = root or Path.cwd() / DEFAULT_ARTIFACTS_ROOT

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return uuid.uuid4().hex[:12]

    def save(self, report: RunReport, diff_text: str = "") -> Path:
        """Persist a run report and associated artifacts."""
        run_dir = self.root / report.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        report.artifacts_dir = str(run_dir)

        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")

        summary_path = run_dir / "summary.md"
        summary_path.write_text(self._render_summary_md(report), encoding="utf-8")

        if diff_text:
            patch_path = run_dir / "patch.diff"
            patch_path.write_text(diff_text, encoding="utf-8")

        logger.info("Run artifacts saved to %s", run_dir)
        return run_dir

    def load_last_report(self) -> RunReport | None:
        """Load the most recent run report."""
        if not self.root.exists():
            return None

        run_dirs = sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            report_path = run_dir / "report.json"
            if report_path.exists():
                return self._load_report(report_path)
        return None

    def _load_report(self, path: Path) -> RunReport | None:
        """Load a RunReport from a JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            from ai_unit_test.v2.models import TargetSpec, ValidationResult

            target = TargetSpec(**data.pop("target"))
            validation_history = [ValidationResult(**v) for v in data.pop("validation_history", [])]
            return RunReport(target=target, validation_history=validation_history, **data)
        except Exception:
            logger.exception("Failed to load report from %s", path)
            return None

    @staticmethod
    def _render_summary_md(report: RunReport) -> str:
        """Render a markdown summary for a run."""
        status = "✅ Success" if report.success else "❌ Failed"
        lines = [
            f"# AIUnitTest v2 Run: {report.run_id}",
            "",
            f"**Status:** {status}",
            f"**Backend:** {report.backend_name}",
            f"**Attempts:** {report.attempts}",
            f"**Target:** {', '.join(report.target.files)}",
            "",
            "## Summary",
            "",
            report.final_summary,
            "",
        ]

        if report.touched_files:
            lines.append("## Touched Files")
            lines.append("")
            for f in report.touched_files:
                lines.append(f"- `{f}`")
            lines.append("")

        if report.validation_history:
            lines.append("## Validation History")
            lines.append("")
            for v in report.validation_history:
                icon = "✅" if v.success else "❌"
                lines.append(f"- {icon} **{v.validator_name}**: {v.summary}")
            lines.append("")

        return "\n".join(lines)
