"""Tests for V2Orchestrator."""

from unittest.mock import MagicMock, AsyncMock

import pytest

from ai_unit_test.v2.models import (
    ContextBundle,
    PatchApplication,
    PatchCandidate,
    RunRequest,
    TargetSpec,
    ValidationResult,
)
from ai_unit_test.v2.orchestrator import V2Orchestrator


def _make_orchestrator(
    backend_patch: PatchCandidate | None = None,
    apply_success: bool = True,
    validation_results: list[ValidationResult] | None = None,
) -> tuple[V2Orchestrator, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build an orchestrator with mocked components."""
    backend = MagicMock()
    backend.name = "mock-backend"
    patch_candidate = backend_patch or PatchCandidate(
        backend_name="mock-backend",
        plan_summary="add tests",
        patch_text="--- file: test_mod.py\ndef test(): pass\n",
        touched_files=["test_mod.py"],
    )
    backend.propose_patch = AsyncMock(return_value=patch_candidate)

    selector = MagicMock()
    selector.select.return_value = [TargetSpec(mode="explicit-file", files=["mod.py"])]

    builder = MagicMock()
    builder.build.return_value = ContextBundle(
        target=TargetSpec(mode="explicit-file", files=["mod.py"]),
        source_snippets={"mod.py": "x = 1"},
    )

    applier = MagicMock()
    applier.apply.return_value = PatchApplication(
        candidate=patch_candidate,
        applied_files=["test_mod.py"] if apply_success else [],
        diff_text="some diff",
        success=apply_success,
        error=None if apply_success else "apply failed",
    )
    applier.rollback = MagicMock()

    if validation_results is None:
        validation_results = [
            ValidationResult(validator_name="syntax", success=True, summary="ok"),
            ValidationResult(validator_name="pytest", success=True, summary="ok"),
        ]

    validator = MagicMock()
    validator.run.side_effect = validation_results

    from ai_unit_test.v2.validation.feedback import FeedbackSummarizer
    from ai_unit_test.v2.reporting.store import RunStore

    store = MagicMock(spec=RunStore)
    store.generate_run_id.return_value = "test-run-123"
    store.save.return_value = MagicMock(__str__=lambda s: "/artifacts/test-run-123")

    orchestrator = V2Orchestrator(
        backend=backend,
        target_selector=selector,
        context_builder=builder,
        patch_applier=applier,
        validators=[validator],
        feedback_summarizer=FeedbackSummarizer(),
        run_store=store,
    )

    return orchestrator, backend, selector, applier, store


class TestV2Orchestrator:
    """Test suite for V2Orchestrator."""

    @pytest.mark.asyncio
    async def test_successful_run(self) -> None:
        """Test a successful single-attempt run."""
        orch, backend, selector, applier, store = _make_orchestrator()
        request = RunRequest(file_path="mod.py", backend_name="mock-backend")

        report = await orch.run(request)

        assert report.success is True
        assert report.attempts == 1
        assert report.run_id == "test-run-123"
        store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_targets(self) -> None:
        """Test run when no targets are selected."""
        orch, _, selector, _, _ = _make_orchestrator()
        selector.select.return_value = []
        request = RunRequest(file_path="mod.py", backend_name="mock-backend")

        report = await orch.run(request)

        assert report.success is False
        assert report.attempts == 0

    @pytest.mark.asyncio
    async def test_dry_run_skips_backend(self) -> None:
        """Test that dry-run does not call the backend."""
        orch, backend, _, _, store = _make_orchestrator()
        request = RunRequest(file_path="mod.py", backend_name="mock-backend", dry_run=True)

        report = await orch.run(request)

        assert report.success is True
        assert report.attempts == 0
        assert "Dry run" in report.final_summary
        backend.propose_patch.assert_not_called()
        store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_apply_failure_triggers_retry(self) -> None:
        """Test that patch application failure triggers retry."""
        orch, backend, _, applier, _ = _make_orchestrator(apply_success=False)

        # Make second attempt succeed
        success_app = PatchApplication(
            candidate=PatchCandidate(backend_name="mock", plan_summary="x", patch_text="y", touched_files=["test_mod.py"]),
            applied_files=["test_mod.py"],
            diff_text="diff",
            success=True,
        )
        applier.apply.side_effect = [applier.apply.return_value, success_app]

        # Override validator for second attempt
        request = RunRequest(file_path="mod.py", backend_name="mock-backend", max_attempts=2)
        report = await orch.run(request)

        assert backend.propose_patch.call_count == 2
        applier.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_validation_failure_triggers_retry(self) -> None:
        """Test that validation failure triggers retry with rollback."""
        fail_results = [
            ValidationResult(validator_name="pytest", success=False, summary="test failed"),
            ValidationResult(validator_name="pytest", success=False, summary="test failed again"),
        ]
        orch, backend, _, applier, _ = _make_orchestrator(validation_results=fail_results)
        request = RunRequest(file_path="mod.py", backend_name="mock-backend", max_attempts=2)

        report = await orch.run(request)

        assert report.success is False
        assert report.attempts == 2
        assert applier.rollback.call_count == 2

    @pytest.mark.asyncio
    async def test_all_attempts_exhausted(self) -> None:
        """Test that all attempts exhausted produces a failed report."""
        fail_results = [
            ValidationResult(validator_name="syntax", success=False, summary="bad"),
        ]
        orch, _, _, _, store = _make_orchestrator(validation_results=fail_results)
        request = RunRequest(file_path="mod.py", backend_name="mock-backend", max_attempts=1)

        report = await orch.run(request)

        assert report.success is False
        assert "exhausted" in report.final_summary.lower()
        store.save.assert_called_once()
