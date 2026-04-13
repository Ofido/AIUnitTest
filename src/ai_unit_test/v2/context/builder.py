"""Context builder for AIUnitTest v2."""

import fnmatch
from pathlib import Path
from typing import Protocol

from ai_unit_test.v2.models import ContextBundle, TargetSpec


class ContextBuilder(Protocol):
    """Contract for context building strategies."""

    def build(self, target: TargetSpec, feedback: list[str] | None = None, previous_patch: str | None = None) -> ContextBundle:
        """Build a context bundle for the given target."""
        ...


class FileContextBuilder:
    """Build context from source files, nearby tests, and project config."""

    def __init__(self, project_root: Path | None = None, test_patterns: list[str] | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self.test_patterns = test_patterns or ["test_*.py", "*_test.py"]

    def build(self, target: TargetSpec, feedback: list[str] | None = None, previous_patch: str | None = None) -> ContextBundle:
        """Build a context bundle from source files and related tests."""
        source_snippets: dict[str, str] = {}
        test_snippets: dict[str, str] = {}

        for file_path in target.files:
            path = Path(file_path)
            if path.exists():
                source_snippets[file_path] = path.read_text(encoding="utf-8")

            related_tests = self._find_related_tests(path)
            for test_path in related_tests:
                test_snippets[str(test_path)] = test_path.read_text(encoding="utf-8")

        project_config = self._read_project_config()

        return ContextBundle(
            target=target,
            source_snippets=source_snippets,
            test_snippets=test_snippets,
            project_config=project_config,
            validator_feedback=feedback or [],
            previous_patch=previous_patch,
            failure_type=None,
        )

    def _find_related_tests(self, source_path: Path) -> list[Path]:
        """Find test files related to a source file."""
        results: list[Path] = []
        stem = source_path.stem

        search_dirs = [self.project_root / "tests", self.project_root / "test"]
        search_dirs = [d for d in search_dirs if d.is_dir()]

        if not search_dirs:
            search_dirs = [self.project_root]

        for search_dir in search_dirs:
            for test_file in search_dir.rglob("*.py"):
                if self._is_test_file(test_file) and stem in test_file.stem:
                    results.append(test_file)
        return results

    def _is_test_file(self, path: Path) -> bool:
        """Check if a file matches configured test patterns."""
        return any(fnmatch.fnmatch(path.name, pattern) for pattern in self.test_patterns)

    def _read_project_config(self) -> dict[str, str]:
        """Read project configuration snippets."""
        config: dict[str, str] = {}
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            config["pyproject.toml"] = pyproject_path.read_text(encoding="utf-8")
        return config
