"""AIUnitTest v2 — tool-first test execution layer for coding agents."""

from ai_unit_test.v2.context.builder import ContextBuilder, FileContextBuilder
from ai_unit_test.v2.models import (
    ContextBundle,
    PatchApplication,
    PatchCandidate,
    RunReport,
    RunRequest,
    TargetSpec,
    ValidationResult,
)
from ai_unit_test.v2.orchestrator import V2Orchestrator
from ai_unit_test.v2.patching.workspace import PatchApplier
from ai_unit_test.v2.reporting.renderer import JsonRenderer, TerminalRenderer
from ai_unit_test.v2.reporting.store import RunStore
from ai_unit_test.v2.targeting.selectors import ExplicitFileSelector, TargetSelector
from ai_unit_test.v2.validation.feedback import FeedbackSummarizer
from ai_unit_test.v2.validation.runners import PytestValidator, SyntaxValidator, Validator

__all__ = [
    "ContextBuilder",
    "ContextBundle",
    "ExplicitFileSelector",
    "FeedbackSummarizer",
    "FileContextBuilder",
    "JsonRenderer",
    "PatchApplication",
    "PatchApplier",
    "PatchCandidate",
    "PytestValidator",
    "RunReport",
    "RunRequest",
    "RunStore",
    "SyntaxValidator",
    "TargetSelector",
    "TargetSpec",
    "TerminalRenderer",
    "V2Orchestrator",
    "ValidationResult",
    "Validator",
]