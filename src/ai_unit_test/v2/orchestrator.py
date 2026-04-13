"""Orchestrator for AIUnitTest v2 test execution workflow."""

import logging

from ai_unit_test.v2.backends.base import ReasoningBackend
from ai_unit_test.v2.context.builder import ContextBuilder
from ai_unit_test.v2.models import (
    PatchApplication,
    RunReport,
    RunRequest,
    TargetSpec,
    ValidationResult,
)
from ai_unit_test.v2.patching.workspace import PatchApplier
from ai_unit_test.v2.reporting.store import RunStore
from ai_unit_test.v2.targeting.selectors import TargetSelector
from ai_unit_test.v2.validation.feedback import FeedbackSummarizer
from ai_unit_test.v2.validation.runners import Validator

logger = logging.getLogger(__name__)


class V2Orchestrator:
    """Coordinate the v2 workflow: target → context → backend → patch → validate → retry → report."""

    def __init__(  # noqa: D107
        self,
        backend: ReasoningBackend,
        target_selector: TargetSelector,
        context_builder: ContextBuilder,
        patch_applier: PatchApplier,
        validators: list[Validator],
        feedback_summarizer: FeedbackSummarizer,
        run_store: RunStore,
    ) -> None:
        self.backend = backend
        self.target_selector = target_selector
        self.context_builder = context_builder
        self.patch_applier = patch_applier
        self.validators = validators
        self.feedback_summarizer = feedback_summarizer
        self.run_store = run_store

    async def run(self, request: RunRequest) -> RunReport:
        """Execute the full v2 loop with bounded retries."""
        run_id = self.run_store.generate_run_id()
        targets = self.target_selector.select(request)

        if not targets:
            return self._empty_report(run_id, request, "No targets selected.")

        target = targets[0]
        validation_history: list[ValidationResult] = []
        feedback: list[str] = []
        previous_patch: str | None = None
        last_application: PatchApplication | None = None

        for attempt in range(1, request.max_attempts + 1):
            logger.info("Attempt %d/%d for run %s", attempt, request.max_attempts, run_id)

            context = self.context_builder.build(target, feedback=feedback, previous_patch=previous_patch)

            if request.dry_run:
                report = RunReport(
                    run_id=run_id,
                    target=target,
                    attempts=0,
                    success=True,
                    backend_name=self.backend.name,
                    final_summary="Dry run: context assembled, no backend call made.",
                    touched_files=[],
                    validation_history=[],
                )
                self.run_store.save(report)
                return report

            try:
                candidate = await self.backend.propose_patch(context)
            except Exception as exc:
                validation_history.append(
                    ValidationResult(
                        validator_name="backend",
                        success=False,
                        summary=f"Backend error: {exc}",
                        exit_code=-1,
                    )
                )
                feedback = self.feedback_summarizer.summarize([validation_history[-1]], attempt)
                logger.warning("Backend failed on attempt %d: %s", attempt, exc)
                continue

            application = self.patch_applier.apply(candidate, request)
            last_application = application

            if not application.success:
                validation_history.append(
                    ValidationResult(
                        validator_name="patch_apply",
                        success=False,
                        summary=application.error or "Patch application failed.",
                        exit_code=-1,
                    )
                )
                previous_patch = candidate.patch_text
                feedback = self.feedback_summarizer.summarize([validation_history[-1]], attempt)
                self.patch_applier.rollback()
                continue

            attempt_results = self._run_validators(application, target)
            validation_history.extend(attempt_results)

            all_passed = all(r.success for r in attempt_results)

            if all_passed:
                report = RunReport(
                    run_id=run_id,
                    target=target,
                    attempts=attempt,
                    success=True,
                    backend_name=self.backend.name,
                    final_summary=f"Patch validated successfully on attempt {attempt}.",
                    touched_files=application.applied_files,
                    validation_history=validation_history,
                )
                self.run_store.save(report, diff_text=application.diff_text)
                return report

            # Rollback failed attempt before retry
            self.patch_applier.rollback()
            previous_patch = candidate.patch_text
            failure_type = self.feedback_summarizer.classify_failure(attempt_results)
            feedback = self.feedback_summarizer.summarize(attempt_results, attempt)
            logger.info("Attempt %d failed (%s), retrying...", attempt, failure_type)

        # All attempts exhausted
        report = RunReport(
            run_id=run_id,
            target=target,
            attempts=request.max_attempts,
            success=False,
            backend_name=self.backend.name,
            final_summary=f"All {request.max_attempts} attempts exhausted without a valid patch.",
            touched_files=last_application.applied_files if last_application else [],
            validation_history=validation_history,
        )
        self.run_store.save(
            report,
            diff_text=last_application.diff_text if last_application else "",
        )
        return report

    def _run_validators(self, application: PatchApplication, target: TargetSpec) -> list[ValidationResult]:
        """Run all validators and return results. Treat ambiguous states as failures."""
        results: list[ValidationResult] = []
        for validator in self.validators:
            result = validator.run(application, target)
            results.append(result)
            if not result.success:
                break
        return results

    @staticmethod
    def _empty_report(run_id: str, request: RunRequest, summary: str) -> RunReport:
        """Create an empty report for edge cases."""
        return RunReport(
            run_id=run_id,
            target=TargetSpec(mode="explicit-file", files=[request.file_path]),
            attempts=0,
            success=False,
            backend_name=request.backend_name,
            final_summary=summary,
        )
