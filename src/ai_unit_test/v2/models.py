"""Core models for AIUnitTest v2."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class RunRequest:
    """Represents a user request to run the v2 workflow."""

    file_path: str
    backend_name: str
    max_attempts: int = 3
    dry_run: bool = False
    allow_source_edits: bool = False
    test_patterns: list[str] = field(default_factory=lambda: ["test_*.py", "*_test.py"])


@dataclass(slots=True)
class TargetSpec:
    """Represents the scope that needs better tests."""

    mode: str
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    uncovered_lines: dict[str, list[int]] = field(default_factory=dict)
    rationale: str | None = None


@dataclass(slots=True)
class ContextBundle:
    """Represents the context package sent to a reasoning backend."""

    target: TargetSpec
    source_snippets: dict[str, str] = field(default_factory=dict)
    test_snippets: dict[str, str] = field(default_factory=dict)
    project_config: dict[str, str] = field(default_factory=dict)
    validator_feedback: list[str] = field(default_factory=list)
    previous_patch: str | None = None
    failure_type: str | None = None


@dataclass(slots=True)
class PatchCandidate:
    """Represents a proposed patch before validation."""

    backend_name: str
    plan_summary: str
    patch_text: str
    touched_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PatchApplication:
    """Represents the result of applying a patch candidate to disk."""

    candidate: PatchCandidate
    applied_files: list[str] = field(default_factory=list)
    diff_text: str = ""
    success: bool = False
    error: str | None = None


@dataclass(slots=True)
class ValidationResult:
    """Represents the result of validating a patch candidate."""

    validator_name: str
    success: bool
    summary: str
    exit_code: int | None = None
    command_results: list[str] = field(default_factory=list)
    log_path: str | None = None
    coverage_delta: float | None = None


@dataclass(slots=True)
class RunReport:
    """Represents the final report of a v2 execution."""

    run_id: str
    target: TargetSpec
    attempts: int
    success: bool
    backend_name: str
    final_summary: str
    artifacts_dir: str | None = None
    touched_files: list[str] = field(default_factory=list)
    validation_history: list[ValidationResult] = field(default_factory=list)