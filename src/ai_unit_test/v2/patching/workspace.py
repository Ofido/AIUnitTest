"""Patch workspace and applier for AIUnitTest v2."""

import difflib
import fnmatch
import logging
from pathlib import Path

from ai_unit_test.v2.models import PatchApplication, PatchCandidate, RunRequest

logger = logging.getLogger(__name__)


class PatchApplier:
    """Apply patch candidates with test-file-first guardrails and rollback."""

    def __init__(self, test_patterns: list[str] | None = None) -> None:
        """Initialize with test file name patterns."""
        self.test_patterns = test_patterns or ["test_*.py", "*_test.py"]
        self._snapshots: dict[str, str] = {}

    def apply(self, candidate: PatchCandidate, request: RunRequest) -> PatchApplication:
        """Apply a patch candidate to disk, enforcing guardrails."""
        applied_files: list[str] = []
        diff_parts: list[str] = []
        self._snapshots.clear()

        error = self._check_guardrails(candidate.touched_files, request)
        if error:
            return PatchApplication(candidate=candidate, success=False, error=error)

        if request.dry_run:
            return PatchApplication(
                candidate=candidate,
                applied_files=list(candidate.touched_files),
                diff_text=candidate.patch_text,
                success=True,
                error=None,
            )

        try:
            file_contents = self._parse_patch_text(candidate.patch_text)

            if not file_contents:
                return PatchApplication(
                    candidate=candidate,
                    applied_files=[],
                    diff_text="",
                    success=False,
                    error="Patch produced no parseable files.",
                )

            # Validate parsed paths against test-file guardrail
            error = self._check_guardrails(list(file_contents.keys()), request)
            if error:
                return PatchApplication(candidate=candidate, success=False, error=error)

            for file_path, content in file_contents.items():
                path = Path(file_path)
                self._snapshot_file(path)
                path.parent.mkdir(parents=True, exist_ok=True)

                old_content = self._snapshots.get(str(path), "")
                diff = self._compute_diff(old_content, content, file_path)
                if diff:
                    diff_parts.append(diff)

                path.write_text(content, encoding="utf-8")
                applied_files.append(file_path)

            return PatchApplication(
                candidate=candidate,
                applied_files=applied_files,
                diff_text="\n".join(diff_parts),
                success=True,
                error=None,
            )
        except Exception as exc:
            self.rollback()
            return PatchApplication(
                candidate=candidate,
                applied_files=[],
                diff_text="",
                success=False,
                error=f"Patch application failed: {exc}",
            )

    def _check_guardrails(self, file_paths: list[str], request: RunRequest) -> str | None:
        """Check that all file paths pass the test-file-first guardrail.

        Returns an error message if any file is rejected, None if all pass.
        """
        if request.allow_source_edits:
            return None
        for file_path in file_paths:
            if not self._is_test_file(Path(file_path)):
                return f"Refused to write non-test file: {file_path}. Use --allow-source-edits to override."
        return None

    def rollback(self) -> None:
        """Restore all snapshotted files to their original state."""
        for file_path, original_content in self._snapshots.items():
            path = Path(file_path)
            if original_content:
                path.write_text(original_content, encoding="utf-8")
            elif path.exists():
                path.unlink()
        self._snapshots.clear()

    def _is_test_file(self, path: Path) -> bool:
        """Check if a file matches configured test patterns."""
        return any(fnmatch.fnmatch(path.name, pattern) for pattern in self.test_patterns)

    def _snapshot_file(self, path: Path) -> None:
        """Save current file content for rollback."""
        key = str(path)
        if key not in self._snapshots:
            if path.exists():
                self._snapshots[key] = path.read_text(encoding="utf-8")
            else:
                self._snapshots[key] = ""

    def _parse_patch_text(self, patch_text: str) -> dict[str, str]:
        """Parse patch text into file_path → content mapping.

        Supports the format:
        --- file: path/to/file.py
        <content>
        """
        files: dict[str, str] = {}
        current_file: str | None = None
        current_lines: list[str] = []

        for line in patch_text.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith("--- file:"):
                if current_file is not None:
                    files[current_file] = "".join(current_lines)
                current_file = stripped[len("--- file:") :].strip()
                current_lines = []
            elif current_file is not None:
                current_lines.append(line)

        if current_file is not None:
            files[current_file] = "".join(current_lines)

        return files

    @staticmethod
    def _compute_diff(old_content: str, new_content: str, file_path: str) -> str:
        """Compute a unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}")
        return "".join(diff)
