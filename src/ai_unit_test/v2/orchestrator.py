"""Incubation orchestrator for AIUnitTest v2."""

from ai_unit_test.v2.backends.base import ReasoningBackend
from ai_unit_test.v2.models import ContextBundle, RunReport, TargetSpec, ValidationResult


class V2Orchestrator:
    """Coordinates the future v2 workflow without replacing the current v1 CLI yet."""

    def __init__(self, backend: ReasoningBackend, max_attempts: int = 2) -> None:
        self.backend = backend
        self.max_attempts = max_attempts

    async def run(self, context: ContextBundle) -> RunReport:
        """Execute the minimal v2 loop contract.

        This is intentionally a scaffold. The real implementation should add:
        target selection, patch application, validator execution, and retry logic.
        """
        candidate = await self.backend.propose_patch(context)
        validation = ValidationResult(
            success=False,
            summary="Validation loop not implemented yet.",
            command_results=[],
            coverage_delta=None,
        )
        return RunReport(
            target=context.target,
            attempts=1,
            success=False,
            backend_name=candidate.backend_name,
            final_summary="Scaffold only: patch proposal available, execution loop pending.",
            touched_files=candidate.touched_files,
            validation_history=[validation],
        )

    @staticmethod
    def make_explicit_target(file_path: str, rationale: str | None = None) -> TargetSpec:
        """Create a minimal explicit-file target for early v2 experiments."""
        return TargetSpec(mode="explicit-file", files=[file_path], rationale=rationale)