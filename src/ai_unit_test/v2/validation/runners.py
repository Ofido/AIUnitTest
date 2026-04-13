"""Validation runners for AIUnitTest v2."""

import logging
import py_compile
import subprocess  # nosec B404 -- controlled subprocess for pytest execution
import sys
import uuid
from pathlib import Path
from typing import Protocol

from ai_unit_test.v2.models import PatchApplication, TargetSpec, ValidationResult

logger = logging.getLogger(__name__)


class Validator(Protocol):
    """Contract for patch validators."""

    def run(self, application: PatchApplication, target: TargetSpec) -> ValidationResult:
        """Validate a patch application and return a structured result."""
        ...


class SyntaxValidator:
    """Validate Python syntax of touched files using py_compile."""

    def run(self, application: PatchApplication, target: TargetSpec) -> ValidationResult:
        """Check that all applied files have valid Python syntax."""
        errors: list[str] = []

        for file_path in application.applied_files:
            path = Path(file_path)
            if not path.suffix == ".py":
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"{file_path}: {exc}")

        if errors:
            return ValidationResult(
                validator_name="syntax",
                success=False,
                summary=f"Syntax errors in {len(errors)} file(s).",
                exit_code=1,
                command_results=errors,
            )

        return ValidationResult(
            validator_name="syntax",
            success=True,
            summary="All files have valid Python syntax.",
            exit_code=0,
            command_results=[],
        )


class PytestValidator:
    """Run targeted pytest on applied test files."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize with project root for test discovery."""
        self.project_root = project_root or Path.cwd()

    def run(self, application: PatchApplication, target: TargetSpec) -> ValidationResult:
        """Run pytest on the applied test files."""
        test_files = [
            f
            for f in application.applied_files
            if Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py")
        ]

        if not test_files:
            return ValidationResult(
                validator_name="pytest",
                success=False,
                summary="No test files in applied patch. Nothing to validate.",
                exit_code=-1,
                command_results=[],
            )

        log_path = self._create_log_path()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *test_files,
            "-v",
            "--tb=short",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--override-ini=addopts=",
        ]

        try:
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            output = result.stdout + result.stderr
            Path(log_path).write_text(output, encoding="utf-8")

            if result.returncode == 0:
                return ValidationResult(
                    validator_name="pytest",
                    success=True,
                    summary="All tests passed.",
                    exit_code=0,
                    command_results=output.strip().splitlines()[-5:],
                    log_path=log_path,
                )
            else:
                return ValidationResult(
                    validator_name="pytest",
                    success=False,
                    summary=f"Pytest failed with exit code {result.returncode}.",
                    exit_code=result.returncode,
                    command_results=output.strip().splitlines()[-20:],
                    log_path=log_path,
                )

        except subprocess.TimeoutExpired:
            Path(log_path).write_text("Timeout: pytest exceeded 120s limit", encoding="utf-8")
            return ValidationResult(
                validator_name="pytest",
                success=False,
                summary="Pytest timed out after 120 seconds.",
                exit_code=-1,
                command_results=["Timeout: pytest exceeded 120s limit"],
                log_path=log_path,
            )
        except FileNotFoundError:
            return ValidationResult(
                validator_name="pytest",
                success=False,
                summary="pytest not found. Is it installed?",
                exit_code=-1,
                command_results=["FileNotFoundError: pytest executable not found"],
            )

    def _create_log_path(self) -> str:
        """Create a project-scoped log file path."""
        log_dir = self.project_root / ".ai-unit-test" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / f"v2_pytest_{uuid.uuid4().hex}.log")
