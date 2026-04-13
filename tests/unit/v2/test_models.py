"""Tests for v2 core models."""

from ai_unit_test.v2.models import (
    ContextBundle,
    PatchApplication,
    PatchCandidate,
    RunReport,
    RunRequest,
    TargetSpec,
    ValidationResult,
)


class TestRunRequest:
    """Test suite for the RunRequest dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        req = RunRequest(file_path="src/mod.py", backend_name="copilot-cli")
        assert req.max_attempts == 3
        assert req.dry_run is False
        assert req.allow_source_edits is False
        assert req.test_patterns == ["test_*.py", "*_test.py"]

    def test_custom_values(self) -> None:
        """Test custom values."""
        req = RunRequest(
            file_path="src/mod.py",
            backend_name="gemini-cli",
            max_attempts=5,
            dry_run=True,
            allow_source_edits=True,
            test_patterns=["test_*.py"],
        )
        assert req.max_attempts == 5
        assert req.dry_run is True
        assert req.allow_source_edits is True
        assert req.test_patterns == ["test_*.py"]


class TestTargetSpec:
    """Test suite for the TargetSpec dataclass."""

    def test_minimal_target(self) -> None:
        """Test creation with minimal fields."""
        ts = TargetSpec(mode="explicit-file")
        assert ts.mode == "explicit-file"
        assert ts.files == []
        assert ts.symbols == []
        assert ts.uncovered_lines == {}
        assert ts.rationale is None

    def test_full_target(self) -> None:
        """Test creation with all fields."""
        ts = TargetSpec(
            mode="explicit-file",
            files=["src/mod.py"],
            symbols=["func"],
            uncovered_lines={"src/mod.py": [10, 20]},
            rationale="coverage gap",
        )
        assert ts.files == ["src/mod.py"]
        assert ts.uncovered_lines["src/mod.py"] == [10, 20]


class TestContextBundle:
    """Test suite for the ContextBundle dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        target = TargetSpec(mode="explicit-file")
        cb = ContextBundle(target=target)
        assert cb.source_snippets == {}
        assert cb.test_snippets == {}
        assert cb.project_config == {}
        assert cb.validator_feedback == []
        assert cb.previous_patch is None
        assert cb.failure_type is None

    def test_with_feedback(self) -> None:
        """Test with validator feedback and previous patch."""
        target = TargetSpec(mode="explicit-file")
        cb = ContextBundle(
            target=target,
            validator_feedback=["test failed"],
            previous_patch="old patch",
            failure_type="test_failure",
        )
        assert cb.validator_feedback == ["test failed"]
        assert cb.previous_patch == "old patch"
        assert cb.failure_type == "test_failure"


class TestPatchCandidate:
    """Test suite for the PatchCandidate dataclass."""

    def test_creation(self) -> None:
        """Test creation with required fields."""
        pc = PatchCandidate(backend_name="copilot-cli", plan_summary="add tests", patch_text="code")
        assert pc.backend_name == "copilot-cli"
        assert pc.touched_files == []


class TestPatchApplication:
    """Test suite for the PatchApplication dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        candidate = PatchCandidate(backend_name="x", plan_summary="y", patch_text="z")
        pa = PatchApplication(candidate=candidate)
        assert pa.applied_files == []
        assert pa.diff_text == ""
        assert pa.success is False
        assert pa.error is None

    def test_success(self) -> None:
        """Test successful application."""
        candidate = PatchCandidate(backend_name="x", plan_summary="y", patch_text="z")
        pa = PatchApplication(candidate=candidate, applied_files=["test.py"], success=True)
        assert pa.success is True


class TestValidationResult:
    """Test suite for the ValidationResult dataclass."""

    def test_minimal(self) -> None:
        """Test creation with required fields."""
        vr = ValidationResult(validator_name="syntax", success=True, summary="ok")
        assert vr.exit_code is None
        assert vr.log_path is None
        assert vr.coverage_delta is None

    def test_with_all_fields(self) -> None:
        """Test creation with all fields."""
        vr = ValidationResult(
            validator_name="pytest",
            success=False,
            summary="failed",
            exit_code=1,
            command_results=["error line"],
            log_path="/tmp/log.txt",
            coverage_delta=-2.5,
        )
        assert vr.exit_code == 1
        assert vr.log_path == "/tmp/log.txt"


class TestRunReport:
    """Test suite for the RunReport dataclass."""

    def test_creation(self) -> None:
        """Test creation with required fields."""
        target = TargetSpec(mode="explicit-file", files=["mod.py"])
        rr = RunReport(
            run_id="abc123",
            target=target,
            attempts=2,
            success=True,
            backend_name="copilot-cli",
            final_summary="done",
        )
        assert rr.run_id == "abc123"
        assert rr.artifacts_dir is None
        assert rr.touched_files == []
        assert rr.validation_history == []
