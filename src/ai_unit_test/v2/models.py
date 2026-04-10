"""Core models for AIUnitTest v2."""

from dataclasses import dataclass, field


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


@dataclass(slots=True)
class PatchCandidate:
    """Represents a proposed patch before validation."""

    backend_name: str
    plan_summary: str
    patch_text: str
    touched_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationResult:
    """Represents the result of validating a patch candidate."""

    success: bool
    summary: str
    command_results: list[str] = field(default_factory=list)
    coverage_delta: float | None = None


@dataclass(slots=True)
class RunReport:
    """Represents the final report of a v2 execution."""

    target: TargetSpec
    attempts: int
    success: bool
    backend_name: str
    final_summary: str
    touched_files: list[str] = field(default_factory=list)
    validation_history: list[ValidationResult] = field(default_factory=list)