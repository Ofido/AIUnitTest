"""Report renderers for AIUnitTest v2."""

import json
from dataclasses import asdict

from ai_unit_test.v2.models import RunReport


class TerminalRenderer:
    """Render run reports for terminal display."""

    def render(self, report: RunReport) -> str:
        """Render a human-readable terminal summary."""
        status = "✅ SUCCESS" if report.success else "❌ FAILED"
        lines = [
            f"\n{'=' * 60}",
            f"  AIUnitTest v2 — Run {report.run_id}",
            f"{'=' * 60}",
            f"  Status:   {status}",
            f"  Backend:  {report.backend_name}",
            f"  Attempts: {report.attempts}",
            f"  Target:   {', '.join(report.target.files)}",
        ]

        if report.artifacts_dir:
            lines.append(f"  Artifacts: {report.artifacts_dir}")

        lines.append(f"{'─' * 60}")
        lines.append(f"  {report.final_summary}")
        lines.append(f"{'=' * 60}\n")

        if report.touched_files:
            lines.append("  Touched files:")
            for f in report.touched_files:
                lines.append(f"    • {f}")
            lines.append("")

        if report.validation_history:
            lines.append("  Validation:")
            for v in report.validation_history:
                icon = "✅" if v.success else "❌"
                lines.append(f"    {icon} [{v.validator_name}] {v.summary}")

        return "\n".join(lines)


class JsonRenderer:
    """Render run reports as JSON."""

    def render(self, report: RunReport) -> str:
        """Render a JSON representation of the report."""
        return json.dumps(asdict(report), indent=2, default=str)
