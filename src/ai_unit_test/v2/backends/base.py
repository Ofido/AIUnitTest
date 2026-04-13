"""Base contracts for AIUnitTest v2 reasoning backends."""

from typing import Protocol

from ai_unit_test.v2.models import ContextBundle, PatchCandidate


class ReasoningBackend(Protocol):
    """Contract for external reasoning backends used by AIUnitTest v2."""

    name: str

    async def propose_patch(self, context: ContextBundle) -> PatchCandidate:
        """Return a plan and patch candidate for the provided context."""
        ...


class BackendRegistry:
    """In-memory registry for v2 reasoning backends."""

    def __init__(self) -> None:
        """Initialize an empty backend registry."""
        self._backends: dict[str, ReasoningBackend] = {}

    def register(self, backend: ReasoningBackend) -> None:
        """Register a backend instance by its name."""
        self._backends[backend.name] = backend

    def get(self, name: str) -> ReasoningBackend:
        """Retrieve a registered backend by name."""
        return self._backends[name]

    def list_names(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._backends.keys())
