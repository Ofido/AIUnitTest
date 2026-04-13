"""Target selectors for AIUnitTest v2."""

from pathlib import Path
from typing import Protocol

from ai_unit_test.v2.models import RunRequest, TargetSpec


class TargetSelector(Protocol):
    """Contract for target selection strategies."""

    def select(self, request: RunRequest) -> list[TargetSpec]:
        """Select targets from a run request."""
        ...


class ExplicitFileSelector:
    """Select a target from an explicit file path."""

    def select(self, request: RunRequest) -> list[TargetSpec]:
        """Create a TargetSpec from the explicit file path in the request."""
        path = Path(request.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Target file not found: {request.file_path}")
        return [
            TargetSpec(
                mode="explicit-file",
                files=[str(path)],
                rationale=f"Explicit file target: {path.name}",
            )
        ]
